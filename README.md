<div align="center">

# ❄️ Kryo

**Sub-second cold starts for GPU inference**

*pronounced 'cry-oh'*

[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org/) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

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
  - [Usage](#usage)
    - [Create a snapshot](#create-a-snapshot)
    - [Restore and run](#restore-and-run)
    - [Manage snapshots](#manage-snapshots)
    - [Other](#other)
    - [Examples](#examples)
  - [Benchmarks](#benchmarks)

---

## The Problem

GPU inference has slow cold starts. A model that runs in milliseconds takes **5-10+ seconds to start**.

| Phase                                | Time      |
| ------------------------------------ | --------- |
| Python imports (torch, transformers) | 3-5s      |
| CUDA initialization                  | ~0.9s     |
| Model loading                        | 0.1-5s    |
| **Total cold start**                 | **5-11s** |

This breaks serverless. You keep instances warm to avoid startup overhead, defeating scale-to-zero.

## The Solution

Checkpoint your process after setup. Restore in milliseconds.

```bash
# Snapshot after setup
kryo snapshot create --name llm -- python setup.py

# Restore and run
kryo run --snapshot llm -- python inference.py
```

> **~100ms** restore instead of **5-11s** cold start

## How It Works

Kryo combines [CRIU](https://criu.org/) (process checkpointing) with NVIDIA's [cuda-checkpoint](https://github.com/NVIDIA/cuda-checkpoint) (GPU state management).

```
┌─────────────────────────────────────────────────────────┐
│  kryo snapshot create                                   │
│  1. Run setup command (imports, model load, warmup)     │
│  2. cuda-checkpoint --toggle (suspend GPU state)        │
│  3. criu dump (checkpoint process)                      │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  kryo run --snapshot                                    │
│  1. criu restore (restore process)                      │
│  2. cuda-checkpoint --toggle (resume GPU state)         │
│  3. exec command (run inference)                        │
└─────────────────────────────────────────────────────────┘
```

## Requirements

- **Linux** — CRIU is Linux-only
- **NVIDIA driver 550+** — for cuda-checkpoint
- **[CRIU](https://criu.org/Installation)** — installed and available in PATH

## Installation

```bash
cargo install kryo
```

## Usage

Kryo works with any command that runs on Linux with CUDA.

### Create a snapshot

```bash
kryo snapshot create --name <name> -- <command>
```

### Restore and run

```bash
kryo run --snapshot <name> -- <command>
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

# Create snapshot (once)
kryo snapshot create --name qwen -- uv run python setup.py

# Run from snapshot (sub-second cold start)
kryo run --snapshot qwen -- uv run python inference.py
```

See [examples/](examples/) for more.

## Benchmarks

Baseline measurements on NVIDIA H100:

![Cold Start Benchmark Results](benchmarks/graphs/table.png)

![Cold Start Phase Breakdown](benchmarks/graphs/phase_breakdown.png)

**Key findings:**
- **Import time dominates** — PyTorch 1.8s, transformers adds 3-4s
- **CUDA init is fixed** — ~0.9s unavoidable tax
- **Model loading varies** — 0.13s (YOLO) to 4.7s (Jina)

See [benchmarks/](benchmarks/) for methodology.
