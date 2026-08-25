# Examples

Worked examples for snapshotting a process with Kryo.

| Path | What it does |
|------|----------------|
| [python/qwen](python/qwen/) | Load Qwen 2.5-0.5B, warm CUDA, freeze at `kryo.checkpoint()`, generate after restore |

Each example needs Linux, an NVIDIA GPU (driver 550+), CRIU, `cuda-checkpoint`, and the Kryo CLI as root. See the [root README](../README.md) for install.
