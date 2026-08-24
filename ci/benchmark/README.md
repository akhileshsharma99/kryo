# Lambda GPU benchmarks

Rents a Lambda Cloud GPU VM, installs CRIU / cuda-checkpoint / Kryo, runs cold start vs restore timings, then destroys the VM.

Do not use Docker here. CRIU and `cuda-checkpoint` need host PIDs and the NVIDIA driver; a container would need `--privileged` and still fights process restore.

## GitHub Actions

Manual only (`workflow_dispatch`). Add one repository secret:

- `LAMBDA_API_KEY` — Lambda Cloud API key (Basic auth username, empty password)

The workflow creates an ephemeral SSH key, registers it with Lambda, launches a 1x GPU (A10 if in stock, otherwise the next available 1x type, or a type you pass in), copies this checkout, runs the bench, uploads `benchmarks/results/latest.json`, and always terminates the instance.

Next run also kills any leftover `kryo-gha-*` instances if a previous job was cancelled.

## Local orchestrator

Needs `LAMBDA_API_KEY`, `ssh`, `ssh-keygen`, and `rsync`:

```bash
export LAMBDA_API_KEY=...
python3 ci/benchmark/run.py --runs 10 --gpu auto
```

`--gpu gpu_1x_a10` / `--gpu gpu_1x_h100` forces a type. `auto` picks the first in-stock 1x GPU from a cheap-first list.
