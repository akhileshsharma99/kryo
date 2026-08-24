<div align="center">

# ❄️ Kryo

**Sub-second cold starts for GPU inference**

*pronounced 'cry-oh'*

[![Rust](https://img.shields.io/badge/rust-1.85+-orange.svg)](https://www.rust-lang.org/) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

---

## Table of Contents

- [❄️ Kryo](#️-kryo)
  - [Table of Contents](#table-of-contents)
  - [The Problem](#the-problem)
  - [The Solution](#the-solution)
  - [How It Works](#how-it-works)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Limitations](#limitations)
  - [Usage](#usage)
    - [Create a snapshot](#create-a-snapshot)
    - [Restore and run](#restore-and-run)
    - [Manage snapshots](#manage-snapshots)
    - [Other](#other)
    - [Examples](#examples)
  - [Benchmarks](#benchmarks)
  - [Contributing](#contributing)

---

## The Problem

GPU inference has slow cold starts. A model that runs in milliseconds takes **5-10+ seconds to start**.

| Phase                                | Time      |
| ------------------------------------ | --------- |
| Python imports (torch, transformers) | 3-5s      |
| CUDA initialization                  | ~0.9s     |
| Model loading                        | 0.1-5s    |
| **Total cold start**                 | **5-11s** |

This makes serverless difficult and requires keeping instances warm to avoid startup overhead, defeating scale-to-zero.

## The Solution

Initialize once. Restore when the application starts.

```python
# app.py
import kryo

model = load_model()
model.to("cuda")
warm_up(model)

kryo.checkpoint()  # Snapshot creation stops here

serve(model)       # Production starts here after restore
```

1. **Create a snapshot during deployment.** This runs setup through
   `kryo.checkpoint()` and then saves the process.

```bash
kryo snapshot create --name llm -- python app.py
```

2. **Restore when starting the production process.** Execution resumes after
   `kryo.checkpoint()` with the initialized model in memory.

```bash
kryo run --snapshot llm
```

3. **Recreate the snapshot** when the code, model, dependencies, or runtime
   environment changes.

## How It Works

Kryo combines [CRIU](https://criu.org/) (process checkpointing) with NVIDIA's [cuda-checkpoint](https://github.com/NVIDIA/cuda-checkpoint) (GPU state management).

```mermaid
sequenceDiagram
    participant K as Kryo CLI
    participant P as Your Process
    participant G as GPU State

    rect rgb(40, 40, 40)
    note over K,G: kryo snapshot create
    K->>P: Spawn process
    P->>P: Imports, model load, warmup
    P->>K: SIGUSR1 (ready)
    K->>G: Suspend CUDA state
    K->>P: CRIU dump (freeze)
    note over P: 💾 Snapshot saved
    end

    rect rgb(40, 40, 40)
    note over K,G: kryo run --snapshot
    K->>P: CRIU restore (unfreeze)
    K->>G: Resume CUDA state
    K->>P: SIGUSR2 (wake up)
    P->>P: Continue from checkpoint
    note over P: ⚡ Sub-second cold start
    end
```

## Requirements

- Linux
- NVIDIA driver 550+
- [CRIU](https://criu.org/)
- [cuda-checkpoint](https://github.com/NVIDIA/cuda-checkpoint)

## Installation

Always install the CLI, CRIU, and cuda-checkpoint. The Python package is optional: only if you want `kryo.checkpoint()` in Python. Other languages use signals or `--wait` (see [Usage](#usage)).

`kryo snapshot create` and `kryo run` must be run as root (`sudo`). CRIU has to freeze and restore another process, which Linux does not allow for a normal user.

**Kryo CLI** (Linux x86_64 or arm64):

```bash
curl -fsSL https://raw.githubusercontent.com/akhileshsharma99/kryo/main/install.sh | sh
```

That installs to `/usr/local/bin`. To use another directory:

```bash
INSTALL_DIR="$HOME/.local/bin" curl -fsSL https://raw.githubusercontent.com/akhileshsharma99/kryo/main/install.sh | sh
```

Or build from source (`~/.cargo/bin`):

```bash
cargo install --git https://github.com/akhileshsharma99/kryo
``` 

**CRIU** (Ubuntu):

```bash
sudo add-apt-repository -y ppa:criu/ppa
sudo apt-get update
sudo apt-get install -y criu
```

Other distros: [CRIU install docs](https://criu.org/Installation).

**cuda-checkpoint:** build [NVIDIA/cuda-checkpoint](https://github.com/NVIDIA/cuda-checkpoint) and put `cuda-checkpoint` on your `PATH`.

**Python package** (optional):

```bash
pip install kryo
```

## Limitations

Alpha. Linux + NVIDIA only. Recreate the snapshot when the code, model, CUDA toolkit, or NVIDIA driver changes. Restored processes are not portable across machines or driver versions.

## Usage

After setup, tell Kryo the process is ready to snapshot:

- **Python:** `import kryo` then `kryo.checkpoint()` (needs the optional package above)
- **Any other language:** block `SIGUSR2`, send `SIGUSR1` to the PID in `KRYO_CLI_PID`, then wait for `SIGUSR2`
- **Can't change the program:** skip signaling and use `--wait` so Kryo snapshots after N seconds

### Create a snapshot

```bash
# Signal-based (default) - your code calls kryo.checkpoint()
sudo kryo snapshot create --name <name> -- <command>

# Timer-based - checkpoint after N seconds (for code you can't modify)
sudo kryo snapshot create --name <name> --wait 30 -- <command>
```

### Restore and run

```bash
sudo kryo run --snapshot <name>
```

### Manage snapshots

```bash
kryo snapshot list                  # List all snapshots
kryo snapshot inspect <name>        # Show details
kryo snapshot delete <name>         # Delete a snapshot
```

### Other

```bash
kryo --version                      # Check version
kryo --help                         # Show help
```

### Examples

```bash
# Navigate to example
cd examples/python/qwen
uv sync

# Create snapshot (runs setup, freezes at kryo.checkpoint())
sudo kryo snapshot create --name qwen -- uv run python qwen.py

# Restore and run (sub-second cold start)
sudo kryo run --snapshot qwen
```

See [examples/](examples/) for more.

## Benchmarks

Baseline measurements on NVIDIA H100:

![Cold Start Benchmark Results](benchmarks/graphs/table.png)

![Cold Start Phase Breakdown](benchmarks/graphs/phase_breakdown.png)

**Key findings:**
- **Import time dominates:** PyTorch 1.8s, transformers adds 3-4s
- **CUDA init is fixed:** ~0.9s unavoidable tax
- **Model loading varies:** 0.13s (YOLO) to 4.7s (Jina)

See [benchmarks/](benchmarks/) for methodology.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Report vulnerabilities via [GitHub security advisories](https://github.com/akhileshsharma99/kryo/security/advisories/new), not public issues.
