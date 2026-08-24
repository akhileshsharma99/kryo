# Kryo cold-start benchmarks

Times **cold start** (fresh Python process) vs **Kryo restore** on the same NVIDIA GPU.

Do not run this on Modal or other serverless hosts: they snapshot their own containers, and CRIU needs a real VM with root.

## Scenarios

| Scenario | Workload |
|----------|----------|
| `torch_cuda` | PyTorch import, CUDA init, GPU matmul |
| `yolo` | YOLOv8n load + one predict |
| `qwen` | Qwen 2.5-0.5B load + warmup generate |
| `whisper` | Whisper-tiny load + dummy transcription |

Each scenario warms CUDA, then calls `kryo.checkpoint()` when launched under the Kryo CLI.

## Local (GPU VM)

Needs Linux, NVIDIA driver 550+, CRIU, `cuda-checkpoint`, and the Kryo CLI (as root).

```bash
cd benchmarks
uv sync
# CUDA PyTorch (Lambda / any NVIDIA box):
uv pip install torch --index-url https://download.pytorch.org/whl/cu124

uv run python runner.py --all --runs 10
```

Results: `results/latest.json`.

## Cloud

GitHub Actions (`workflow_dispatch`) rents a Lambda GPU, bootstraps the box, runs this runner, then destroys the instance. See [ci/benchmark/README.md](../ci/benchmark/README.md).
