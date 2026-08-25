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

Each scenario warms CUDA, then calls `kryo.checkpoint()` when launched under the Kryo CLI. After the checkpoint the process `_exit`s so timings measure time-to-ready, not interpreter shutdown.

## Local (GPU VM)

Needs Linux, NVIDIA driver 550+, CRIU, `cuda-checkpoint`, and the Kryo CLI (as root).

```bash
cd benchmarks
uv sync
# Match the wheel to the driver CUDA version (Lambda A10 is often cu128):
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

.venv/bin/python download_models.py
.venv/bin/python runner.py --all --runs 10 --timeout 90
```

Use `.venv/bin/python`, not `uv run`, after installing the CUDA wheel. `uv run` re-syncs the lockfile and can replace torch with a CPU build.

`--timeout 90` kills a hung dump/restore. HuggingFace is forced offline during timed runs, so download weights first.

Results: `results/latest.json` (gitignored). The README chart is `results/charts/cold-vs-kryo.svg`.

```bash
python3 ../ci/benchmark/format_results.py \
  --json results/latest.json \
  --svg results/charts/cold-vs-kryo.svg \
  --patch-readme ../README.md
```

## Cloud

GitHub Actions rents a Lambda GPU, bootstraps the box, runs this runner, then destroys the instance.

- **Each GitHub Release** — the Release workflow dispatches this job with 10 runs (not `release: published`; that event never fires when release-please uses `GITHUB_TOKEN`)
- **Actions → GPU Benchmark** — `workflow_dispatch`; optional `tag` publishes JSON/SVG to that release and opens a `chore:` PR

See [ci/benchmark/README.md](../ci/benchmark/README.md).
