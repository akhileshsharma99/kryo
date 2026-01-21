"""Modal deployment for cold start benchmarks.

Run from repo root:
    cd ci/modal && uv run modal run app.py --runs 50
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import modal
import numpy as np

REPO_ROOT = Path(__file__).parent.parent.parent

# Build image from pyproject.toml
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")  # OpenCV deps for ultralytics
    # Install PyTorch with CUDA first
    .pip_install(
        "torch", "torchvision", "torchaudio",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    # Install remaining deps from pyproject.toml
    .pip_install_from_pyproject(str(REPO_ROOT / "benchmarks" / "pyproject.toml"))
    # Pre-download model weights (copy=True allows run_commands after)
    .add_local_file(
        REPO_ROOT / "benchmarks" / "download_models.py",
        remote_path="/app/download_models.py",
        copy=True,
    )
    .run_commands("cd /app && python download_models.py")
    # Copy benchmark code (last step, no copy=True needed)
    .add_local_dir(
        REPO_ROOT / "benchmarks",
        remote_path="/app",
        ignore=[
            ".venv",
            "__pycache__",
            "*.pyc",
            ".git",
            ".ruff_cache",
            ".mypy_cache",
            ".pytest_cache",
            "results/*.json",  # Don't copy local results
            "graphs/*.png",    # Don't copy local graphs
        ],
    )
)

app = modal.App("kryo-benchmarks")

SCENARIOS = [
    "baseline",
    "numpy_only",
    "torch_cpu",
    "torch_cuda",
    "yolo",
    "qwen3",
    "whisper",
    "jina_embeddings",
]


def _run_benchmark_impl(scenario: str, run_id: int) -> dict:
    """Run a single benchmark scenario. Each container IS a cold start."""
    import subprocess

    proc = subprocess.run(
        ["python", f"/app/scenarios/{scenario}.py"],
        capture_output=True,
        text=True,
        cwd="/app/scenarios",
        check=False,
    )

    if proc.returncode != 0:
        return {
            "scenario": scenario,
            "run_id": run_id,
            "error": proc.stderr,
        }

    try:
        result = json.loads(proc.stdout)
        return {
            "scenario": scenario,
            "run_id": run_id,
            **result,
        }
    except json.JSONDecodeError as e:
        return {
            "scenario": scenario,
            "run_id": run_id,
            "error": f"Invalid JSON: {e}\nstdout: {proc.stdout}",
        }


# Define benchmark functions for each GPU type
@app.function(image=image, gpu="H100", timeout=600)
def run_benchmark_h100(scenario: str, run_id: int) -> dict:
    """Run benchmark on H100."""
    return _run_benchmark_impl(scenario, run_id)


@app.function(image=image, gpu="A100", timeout=600)
def run_benchmark_a100(scenario: str, run_id: int) -> dict:
    """Run benchmark on A100."""
    return _run_benchmark_impl(scenario, run_id)


@app.function(image=image, gpu="A10G", timeout=600)
def run_benchmark_a10g(scenario: str, run_id: int) -> dict:
    """Run benchmark on A10G."""
    return _run_benchmark_impl(scenario, run_id)


@app.function(image=image, gpu="T4", timeout=600)
def run_benchmark_t4(scenario: str, run_id: int) -> dict:
    """Run benchmark on T4."""
    return _run_benchmark_impl(scenario, run_id)


GPU_FUNCTIONS = {
    "H100": run_benchmark_h100,
    "A100": run_benchmark_a100,
    "A10G": run_benchmark_a10g,
    "T4": run_benchmark_t4,
}


def compute_stats(values: list[float]) -> dict[str, float]:
    """Compute statistics for a list of values."""
    if not values:
        return {}

    arr = np.array(values)
    return {
        "mean": float(mean(values)),
        "std": float(stdev(values)) if len(values) > 1 else 0.0,
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def aggregate_results(raw_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate raw results from multiple runs into statistics."""
    # Group by scenario
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in raw_results:
        if "error" not in result:
            by_scenario[result["scenario"]].append(result)

    # Get metadata from first successful result
    metadata: dict[str, Any] = {"timestamp": datetime.now().isoformat()}
    for results in by_scenario.values():
        if results and "metadata" in results[0]:
            metadata.update(results[0]["metadata"])
            break

    # Aggregate statistics
    scenarios: dict[str, Any] = {}
    for scenario, results in by_scenario.items():
        if not results:
            scenarios[scenario] = {"error": "All runs failed"}
            continue

        # Collect all phase timings and totals
        all_phases: dict[str, list[float]] = defaultdict(list)
        all_totals: list[float] = []

        for result in results:
            all_totals.append(result["total"])
            for phase, duration in result.get("phases", {}).items():
                all_phases[phase].append(duration)

        scenarios[scenario] = {
            "phases": {
                phase: compute_stats(values) for phase, values in all_phases.items()
            },
            "total": compute_stats(all_totals),
            "runs": len(results),
        }

    return {"metadata": metadata, "scenarios": scenarios}


@app.local_entrypoint()
def main(runs: int = 50, gpu: str = "H100", output: str = "results/latest.json"):
    """Run all benchmarks on Modal and aggregate results."""
    if gpu not in GPU_FUNCTIONS:
        raise ValueError(f"Unknown GPU: {gpu}. Available: {list(GPU_FUNCTIONS.keys())}")

    run_fn = GPU_FUNCTIONS[gpu]

    print(f"GPU: {gpu}")
    print(
        f"Starting {len(SCENARIOS)} scenarios x {runs} runs = {len(SCENARIOS) * runs} containers"
    )

    # Create all (scenario, run_id) combinations
    inputs = [(s, i) for s in SCENARIOS for i in range(runs)]

    # Spawn all containers in parallel - each container IS a cold start
    print("Spawning containers...")
    raw_results = list(run_fn.starmap(inputs))

    # Count successes and failures
    successes = sum(1 for r in raw_results if "error" not in r)
    failures = len(raw_results) - successes
    print(f"Completed: {successes} succeeded, {failures} failed")

    # Aggregate results
    results = aggregate_results(raw_results)

    # Save results to benchmarks/results/
    output_path = REPO_ROOT / "benchmarks" / output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")

    # Print summary
    print("\nSummary:")
    for scenario, data in results["scenarios"].items():
        if "error" in data:
            print(f"  {scenario}: ERROR")
        else:
            total = data["total"]["mean"]
            print(f"  {scenario}: {total:.2f}s (mean)")
