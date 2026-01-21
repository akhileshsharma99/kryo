"""Local CLI runner for cold start benchmarks."""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
RESULTS_DIR = Path(__file__).parent / "results"

ALL_SCENARIOS = [
    "baseline",
    "numpy_only",
    "torch_cpu",
    "torch_cuda",
    "yolo",
    "qwen3",
    "whisper",
    "jina_embeddings",
]


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


def run_scenario(scenario: str, runs: int) -> dict[str, Any]:
    """Run a scenario multiple times and collect results."""
    # Validate scenario is in allowed list (security: prevent arbitrary script execution)
    if scenario not in ALL_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")

    script_path = SCENARIOS_DIR / f"{scenario}.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Scenario script not found: {script_path}")

    # Resolve to absolute path for subprocess
    script_abs = script_path.resolve()

    results = []
    for i in range(runs):
        print(f"  Run {i + 1}/{runs}...", end=" ", flush=True)

        proc = subprocess.run(
            [sys.executable, str(script_abs)],
            capture_output=True,
            text=True,
            cwd=str(SCENARIOS_DIR),
            check=False,
        )

        if proc.returncode != 0:
            print("FAILED")
            print(f"    stderr: {proc.stderr}")
            continue

        try:
            result = json.loads(proc.stdout)
            results.append(result)
            print(f"OK ({result['total']:.3f}s)")
        except json.JSONDecodeError as e:
            print("FAILED (invalid JSON)")
            print(f"    stdout: {proc.stdout}")
            print(f"    error: {e}")
            continue

    if not results:
        return {"error": "All runs failed"}

    # Aggregate phases across all runs
    all_phases: dict[str, list[float]] = {}
    all_totals: list[float] = []

    for result in results:
        all_totals.append(result["total"])
        for phase, duration in result["phases"].items():
            if phase not in all_phases:
                all_phases[phase] = []
            all_phases[phase].append(duration)

    return {
        "phases": {
            phase: compute_stats(values) for phase, values in all_phases.items()
        },
        "total": compute_stats(all_totals),
        "runs": len(results),
        "metadata": results[0].get("metadata", {}),
    }


def run_all(scenarios: list[str], runs: int) -> dict[str, Any]:
    """Run all scenarios and aggregate results."""
    results: dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now(UTC).isoformat(),
        },
        "scenarios": {},
    }

    for scenario in scenarios:
        print(f"\nRunning scenario: {scenario}")
        try:
            scenario_results = run_scenario(scenario, runs)
            results["scenarios"][scenario] = scenario_results

            # Update global metadata from first successful scenario
            if (
                "python_version" not in results["metadata"]
                and "metadata" in scenario_results
            ):
                results["metadata"].update(scenario_results["metadata"])
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            results["scenarios"][scenario] = {"error": str(e)}
        except (ValueError, OSError, json.JSONDecodeError) as e:
            print(f"  Error: {e}")
            results["scenarios"][scenario] = {"error": str(e)}

    return results


def main() -> None:
    """CLI entrypoint for running cold start benchmarks."""
    parser = argparse.ArgumentParser(description="Run cold start benchmarks")
    parser.add_argument(
        "--scenario",
        type=str,
        choices=ALL_SCENARIOS,
        help="Scenario to run (default: run all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all scenarios",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of runs per scenario (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: results/latest.json)",
    )

    args = parser.parse_args()

    if args.scenario:
        scenarios = [args.scenario]
    elif args.all:
        scenarios = ALL_SCENARIOS
    else:
        parser.print_help()
        print("\nError: Must specify --scenario or --all")
        sys.exit(1)

    results = run_all(scenarios, args.runs)

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save results
    output_path = Path(args.output) if args.output else RESULTS_DIR / "latest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
