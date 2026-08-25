# GPU benchmark scheduler

Reads a YAML job file, rents a pool of GPU VMs through a provider plugin, and
measures **time to first inference** for cold start vs Kryo restore.

The scheduler does not know about Lambda URLs. Lambda Cloud is one provider
under `providers/`. Downloads, golden-image setup, and snapshot create are
**not timed**. Timed runs drop the Linux page cache so weights and snapshots
are read from disk.

Do not restore inside Docker. CRIU and `cuda-checkpoint` need host PIDs and
the NVIDIA driver.

## Job files

| File | GPU | What |
|------|-----|------|
| `jobs/release.yaml` | 1× A10 | Tiny models (`torch_cuda`, `yolo`, `qwen` 0.5B, `whisper`). GitHub Release uses this. |
| `jobs/prod-fair.yaml` | 1× H100 PCIe | 7B probes (`qwen7`, `torch_compile`, `vllm_engine`). Manual only. |

Caps in the YAML bound how many VMs of each SKU can exist at once. Idle VMs
are terminated after `idle_timeout` so a hung controller cannot leave boxes
up overnight.

## Triggers

- **GitHub Release** — `release.yml` dispatches `benchmark.yml` with `jobs=release.yaml`
- **Actions → GPU Benchmark** — `workflow_dispatch`; pick the job file; optional `tag` publishes

Not on pull requests. Public repo, paid GPU, secrets.

## Secret

- `LAMBDA_API_KEY` — Lambda Cloud API key (Basic auth username, empty password)

## Local

Needs `ssh`, `ssh-keygen`, `rsync`, and the Doppler CLI (`kryo` / `dev_personal`).
GitHub Actions uses the `LAMBDA_API_KEY` repo secret.

```bash
doppler run -- uv run --directory ci/benchmark python -u run.py --jobs jobs/release.yaml
doppler run -- uv run --directory ci/benchmark python -u run.py --jobs jobs/prod-fair.yaml
doppler run -- uv run --directory ci/benchmark python -u run.py --destroy
```

`--destroy` kills leftover `kryo-gha-*` instances and any saved `kryo-dev`
session. `--keep` leaves the pool running when the process exits (you must
`--destroy` later).

On the VM, `setup.sh` writes `/var/lib/kryo-bench/golden.stamp`. Later samples
on that box skip bootstrap and only rebuild Kryo after rsync. Snapshot tarballs
are cached on the controller under gitignored `ci/benchmark/.snapshots/` and
reused when the scenario script, weights id, Kryo version, GPU SKU, and driver
match.

Timed runs use `benchmarks/.venv/bin/python`, not `uv run`.

## Publishing

When `tag` is set (Release always passes the new tag):

1. Upload `kryo-benchmarks.json`, `.svg`, and `.md` as **release assets**
2. Append a chart + table to the **release notes**
3. Open a `chore:` PR that updates `benchmarks/results/<tag>.json`, the README chart, and the README table

Dispatch without `tag` only uploads Actions artifacts. Bench failure does not
un-publish crates or PyPI; it fails the GPU workflow only.
