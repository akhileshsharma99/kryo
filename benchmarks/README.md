# Kryo cold-start benchmarks

Times **cold start** (fresh Python process, load weights, first inference) vs
**Kryo restore** (restore + first inference) on the same NVIDIA GPU.

Do not run this on Modal or other serverless hosts: they snapshot their own
containers, and CRIU needs a real VM with root.

## Scenarios

| Scenario | Workload |
|----------|----------|
| `torch_cuda` | PyTorch import, CUDA init, GPU matmul |
| `yolo` | YOLOv8n load + one predict |
| `qwen` | Qwen 2.5-0.5B load + first generate |
| `whisper` | Whisper-tiny load + dummy transcription |
| `qwen7` | Qwen 2.5-7B load + first generate (optional) |
| `qwen32` | Qwen 2.5-32B load + first generate (optional, 80GB GPU) |
| `vllm_engine` | vLLM engine + CUDA-graph capture on Qwen 2.5-7B (optional) |
| `torch_compile` | `torch.compile` (Triton kernels / CUDA graphs) on Qwen 2.5-7B (optional) |

Release CI runs `ci/benchmark/jobs/release.yaml` (the first four) on one A10.
`jobs/prod-fair.yaml` is the H100 7B probe.

Each timed run **drops the Linux page cache** first so weights and snapshots
are read from disk, like a new pod on a node that has the files in the image
but not in RAM. `--no-drop-caches` restores the old in-RAM behavior. Snapshots
stay on disk (`~/.kryo/snapshots`); `--tmpfs-snapshots` is only for I/O
experiments.

Each scenario runs first inference, then `checkpoint_or_exit(resume=infer)`.
Under Kryo, restore continues into `resume()` so timings include
time-to-first-inference, then `_exit` so interpreter shutdown is not counted.

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

Use `.venv/bin/python`, not `uv run`, after installing the CUDA wheel. `uv run`
re-syncs the lockfile and can replace torch with a CPU build.

`--timeout 90` kills a hung dump/restore. HuggingFace is forced offline during
timed runs, so download weights first.

`--once cold|create|restore` runs a single sample (no warmup) so a scheduler
can reuse snapshots across a machine pool.

Results: `results/latest.json` (gitignored). The README chart is
`results/charts/cold-vs-kryo.svg`.

```bash
python3 ../ci/benchmark/format_results.py \
  --json results/latest.json \
  --svg results/charts/cold-vs-kryo.svg \
  --patch-readme ../README.md
```

## Cloud

GitHub Actions runs the YAML scheduler in `ci/benchmark/`. It rents a capped
pool of Lambda GPUs, bootstraps each box once (`setup.sh`), caches CRIU
snapshots between samples, then destroys idle VMs.

- **Each GitHub Release** — `jobs/release.yaml` (not `release: published`; that event never fires when release-please uses `GITHUB_TOKEN`)
- **Actions → GPU Benchmark** — `workflow_dispatch`; pick a job file; optional `tag` publishes JSON/SVG to that release and opens a `chore:` PR

See [ci/benchmark/README.md](../ci/benchmark/README.md).
