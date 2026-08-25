# Lambda GPU benchmarks

Rents a Lambda Cloud GPU VM, installs CRIU / cuda-checkpoint / Kryo, runs cold start vs restore timings, then destroys the VM.

Do not use Docker here. CRIU and `cuda-checkpoint` need host PIDs and the NVIDIA driver.

## Triggers

- **GitHub Release** — `release.yml` calls `gh workflow run benchmark.yml` after release-please publishes. This is required: GitHub drops `release: published` when the release is created with `GITHUB_TOKEN` (release-please's default), so that event will never run this job.
- **Actions → GPU Benchmark** — `workflow_dispatch` with `runs`, `gpu`, and optional `tag`

Not on pull requests. Public repo, paid GPU, secrets.

## Secret

- `LAMBDA_API_KEY` — Lambda Cloud API key (Basic auth username, empty password)

The workflow creates an ephemeral SSH key, registers it with Lambda, launches a 1x GPU (A10 if in stock, otherwise the next available 1x type, or a type you pass in), copies this checkout, runs the bench, and always terminates the instance. The next run also kills leftover `kryo-gha-*` instances if a previous job was cancelled.

On the VM, `setup.sh` installs the prebuilt `cuda-checkpoint` binary (NVIDIA's repo has no Makefile) and a CUDA torch wheel that matches `nvidia-smi`. Timed runs use `benchmarks/.venv/bin/python`, not `uv run`.

## Publishing

When `tag` is set (Release always passes the new tag):

1. Upload `kryo-benchmarks.json`, `.svg`, and `.md` as **release assets**
2. Append a chart + table to the **release notes**
3. Open a `chore:` PR that updates `benchmarks/results/<tag>.json`, the README chart, and the README table

Dispatch without `tag` only uploads Actions artifacts. Bench failure does not un-publish crates or PyPI; it fails the GPU workflow only.

## Local orchestrator

Needs `ssh`, `ssh-keygen`, `rsync`, and the Doppler CLI (`kryo` / `dev_personal`). GitHub Actions still uses the `LAMBDA_API_KEY` repo secret.

```bash
doppler run -- python3 -u ci/benchmark/run.py --runs 10 --gpu auto
doppler run -- python3 -u ci/benchmark/run.py \
  --gpu gpu_1x_h100_pcie --scenarios qwen7,vllm_engine,torch_compile --runs 3 --timeout 900
```

Keep one VM up so later runs skip boot and `setup.sh` (~10+ minutes):

```bash
doppler run -- python3 -u ci/benchmark/run.py --keep --setup-only \
  --gpu gpu_1x_h100_pcie --scenarios qwen7,qwen32
doppler run -- python3 -u ci/benchmark/run.py --reuse --skip-setup \
  --scenarios qwen7 --runs 3 --timeout 600 --output /tmp/kryo-large.json
doppler run -- python3 -u ci/benchmark/run.py --destroy
```

`--keep` names the instance `kryo-dev` (the janitor only kills `kryo-gha-*`). SSH state lives in gitignored `ci/benchmark/.session/`. `--reuse` rsyncs this checkout, then runs the bench. `--skip-setup` skips CRIU/cuda-checkpoint/Kryo install.

`--gpu gpu_1x_a10` / `--gpu gpu_1x_h100` forces a type. `auto` picks the first in-stock 1x GPU from a cheap-first list.
