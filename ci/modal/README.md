# Modal Deployment

Runs cold start benchmarks on [Modal](https://modal.com/) H100 GPUs. Each benchmark spawns a fresh container, measuring true cold start times.

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- [Modal](https://modal.com/) account

## Setup

```bash
cd ci/modal
uv sync
uv run modal setup  # Authenticate with Modal
```

## Run

```bash
# Default: 50 runs on H100
uv run modal run app.py

# Different GPU
uv run modal run app.py --gpu A100

# Custom runs
uv run modal run app.py --runs 10
uv run modal run app.py --runs 100 --gpu A10G
```

**Options:**
- `--runs` - Number of runs per scenario (default: 50)
- `--gpu` - GPU type: `H100`, `A100`, `A10G`, `T4` (default: H100)
- `--output` - Output path (default: results/latest.json)

Results are saved to `benchmarks/results/latest.json`.
