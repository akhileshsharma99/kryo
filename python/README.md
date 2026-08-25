# Kryo Python SDK

Python helper for [Kryo](https://github.com/akhileshsharma99/kryo) checkpoint signaling. It is optional: other languages send `SIGUSR1` / wait for `SIGUSR2`, or use `kryo snapshot create --wait`.

`kryo.checkpoint()` does not snapshot by itself. The Kryo CLI freezes the process after it receives that signal.

## Installation

```bash
pip install kryo
```

You still need the [Kryo CLI](https://github.com/akhileshsharma99/kryo#installation), CRIU, and `cuda-checkpoint` on Linux. `kryo snapshot create` and `kryo run` must be run as root (`sudo`).

## Usage

```python
import kryo

# Setup (runs once, gets checkpointed)
model = load_model()
model.to("cuda")

kryo.checkpoint()  # Freeze here

# Inference (runs after restore)
result = model(input)
```

Then:

```bash
sudo kryo snapshot create --name mymodel -- python app.py
sudo kryo run --snapshot mymodel
```

`checkpoint()` is idempotent. Recreate the snapshot when the code, model, CUDA toolkit, or NVIDIA driver changes.

## Requirements

- Linux (CRIU is Linux-only)
- Kryo CLI on `PATH`
- NVIDIA driver 550+ for GPU checkpointing
- CRIU and `cuda-checkpoint`
