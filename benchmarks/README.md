# Cold Start Benchmarks

Measures cold start times across ML scenarios (PyTorch, YOLO, Whisper, etc.) with consistent methodology, JSON results, and graph generation.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) - Python package manager
- Python 3.11+

## Scenarios

| Scenario | What it measures |
|----------|------------------|
| `baseline` | Python stdlib imports |
| `numpy_only` | NumPy import + matrix op |
| `torch_cpu` | PyTorch CPU import + inference |
| `torch_cuda` | PyTorch + CUDA init + inference |
| `yolo` | Ultralytics YOLOv8 full pipeline |
| `qwen3` | Qwen 2.5 LLM load + generate |
| `whisper` | Whisper model load + transcribe |
| `jina_embeddings` | Jina v3 embeddings |

## Setup

```bash
cd benchmarks
uv sync
```

## Run Benchmarks

```bash
# Single scenario
uv run python runner.py --scenario baseline --runs 10

# All scenarios
uv run python runner.py --all --runs 50

# Results saved to results/latest.json
```

## Analyze Results

```bash
uv run python analyze.py

# Generates:
#   - graphs/phase_breakdown.png
#   - graphs/total_comparison.png
#   - graphs/variance.png
```

## Development

```bash
# Lint
uv run ruff check .

# Lint + auto-fix
uv run ruff check --fix .

# Format
uv run ruff format .

# Type check
uv run mypy .
```

## Output Format

Results are saved as JSON with statistics per phase:

```json
{
  "scenarios": {
    "torch_cuda": {
      "phases": {
        "import": {"mean": 1.2, "std": 0.05, "p50": 1.19, "p95": 1.28, "p99": 1.32},
        "cuda_init": {"mean": 0.45, "std": 0.02, "p50": 0.44, "p95": 0.48, "p99": 0.51}
      },
      "total": {"mean": 2.10, "std": 0.08, "p50": 2.05, "p95": 2.20, "p99": 2.28}
    }
  }
}
```
