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
| `jobs/a10.yaml` | 1× A10 | Tiny models (`torch_cuda`, `yolo`, `qwen` 0.5B, `whisper`). |
| `jobs/h100.yaml` | 1× H100 SXM5 | 7B + 32B. |

Caps in the YAML bound how many VMs of each SKU can exist at once. Idle VMs
are terminated after `idle_timeout`. The whole process also has a hard stop
(`BENCH_MAX_SECONDS`, default 3 hours) and destroys the pool on SIGINT/SIGTERM
or interpreter exit. If the controller is killed with SIGKILL, run
`python -u run.py --destroy` immediately. A scheduled **Lambda janitor**
workflow also reaps `kryo-gha-*` VMs older than 4 hours.

## Triggers

- **GitHub Release** — `release.yml` dispatches `benchmark.yml` with `jobs=all`
  (A10 and H100 in parallel). The publish job merges JSON, updates the README
  chart and table, uploads release assets, and opens a `chore:` PR.
- **Actions → GPU Benchmark** — `workflow_dispatch`; `all`, `a10.yaml`, or
  `h100.yaml`; optional `tag` publishes
- **Lambda janitor** — every 30 minutes (`--reap-stale`)
- **Keepalive** — monthly; re-enables scheduled workflows so GitHub does not
  disable them after 60 days of inactivity on a public repo

Not on pull requests. Public repo, paid GPU, secrets.

## Secret

- `LAMBDA_API_KEY` — Lambda Cloud API key (Basic auth username, empty password)

## Local

Needs `ssh`, `ssh-keygen`, `rsync`, and the Doppler CLI (`kryo` / `dev_personal`).
GitHub Actions uses the `LAMBDA_API_KEY` repo secret.

```bash
doppler run -- uv run --directory ci/benchmark python -u run.py --jobs jobs/a10.yaml
doppler run -- uv run --directory ci/benchmark python -u run.py --jobs jobs/h100.yaml
doppler run -- uv run --directory ci/benchmark python -u run.py --destroy
doppler run -- uv run --directory ci/benchmark python -u run.py --reap-stale
```

`--destroy` kills leftover `kryo-gha-*` instances and any saved `kryo-dev`
session. `--keep` leaves the pool running when the process exits (you must
`--destroy` later).

On a **new** VM the scheduler copies a golden directory (CRIU, cuda-checkpoint,
Rust/uv, CUDA torch venv) instead of running `setup.sh` again. Lambda cannot
snapshot the root disk as a custom AMI, so the tree lives on a persistent
filesystem attached at launch (`/lambda/nfs/kryo-golden`). It is a directory,
not a gzip tarball: compressing Hugging Face weights was slower than the
benches, so weights stay on the VM (`extra_weights`). CRIU GPU snapshots stay
out of that image. Each VM dumps once, then reuses that dump for every timed
restore on that box.

The first run in a region still pays `setup.sh`, then copies the tree. Later
CI jobs and idle-reaped replacements apply it and only rebuild Kryo. Filesystem
storage is billed by Lambda; `golden.mode: setup` disables packing if you do
not want that.

Timed runs use `benchmarks/.venv/bin/python`, not `uv run`.

## Publishing

When `tag` is set (Release always passes the new tag):

1. Upload `kryo-benchmarks.json`, `.svg`, and `.md` as **release assets**
2. Append a chart + table to the **release notes**
3. Open a `chore:` PR that updates `benchmarks/results/<tag>.json`, the README chart, and the README table

Dispatch without `tag` only uploads Actions artifacts. Bench failure does not
un-publish crates or PyPI; it fails the GPU workflow only.
