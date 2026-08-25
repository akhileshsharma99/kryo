# Benchmarks

Kryo vs a normal Python start, measured as **time to first inference** on a
real NVIDIA GPU.

Cold start loads the model from disk. Kryo restores a snapshot, then runs the
same first inference. The Linux page cache is dropped before every timed run,
so this is a new pod with files on disk, not a warm machine with weights
already in RAM.

Numbers on the [main README](../README.md) are from NVIDIA A10. Larger models
below need an 80GB GPU.

| Workload | What starts |
|----------|-------------|
| PyTorch CUDA | import, CUDA init, one GPU matmul |
| YOLOv8n | load weights, one predict |
| Qwen 2.5-0.5B | load weights, first generate |
| Whisper-tiny | load weights, one transcription |
| Qwen 2.5-7B | load weights, first generate |
| torch.compile on 7B | compile + CUDA graphs, first generate |
| vLLM on 7B | engine start + CUDA-graph capture, first generate |
| Qwen 2.5-32B | load weights, first generate (80GB GPU) |

Creating the snapshot is a deploy step. It is not included in the restore
time.

These run in CI on [Lambda Cloud](https://lambdalabs.com/) VMs, not on a
laptop. A Linux NVIDIA box with CRIU, `cuda-checkpoint`, and root can
reproduce them with `benchmarks/runner.py`.
