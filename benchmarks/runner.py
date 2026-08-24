"""Run cold vs Kryo-restore timings for GPU scenarios.

Must run on a Linux NVIDIA box with CRIU, cuda-checkpoint, and the Kryo CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
RESULTS_DIR = Path(__file__).parent / "results"

ALL_SCENARIOS = [
    "torch_cuda",
    "yolo",
    "qwen",
    "whisper",
]

RUN_TIMEOUT_SECONDS = 600


def compute_stats(values: list[float]) -> dict[str, float]:
    """Compute summary statistics for a list of timings."""
    if not values:
        return {}

    ranked = sorted(values)
    n = len(ranked)

    def percentile(p: float) -> float:
        if n == 1:
            return ranked[0]
        index = min(n - 1, max(0, round((p / 100) * (n - 1))))
        return ranked[index]

    return {
        "mean": float(mean(values)),
        "std": float(stdev(values)) if n > 1 else 0.0,
        "p50": float(percentile(50)),
        "p95": float(percentile(95)),
        "p99": float(percentile(99)),
        "min": float(ranked[0]),
        "max": float(ranked[-1]),
    }


def gpu_metadata() -> dict[str, str]:
    """Collect GPU name and driver from nvidia-smi when available."""
    metadata: dict[str, str] = {}
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return metadata
    query = subprocess.run(
        [
            nvidia_smi,
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if query.returncode != 0 or not query.stdout.strip():
        return metadata
    name, _, driver = query.stdout.strip().splitlines()[0].partition(",")
    metadata["gpu"] = name.strip()
    metadata["driver"] = driver.strip()
    return metadata


def kryo_command() -> list[str]:
    """Prefix that runs the Kryo CLI with root, keeping PATH."""
    kryo = shutil.which("kryo")
    if kryo is None:
        raise FileNotFoundError("kryo CLI not found on PATH")
    if os.geteuid() == 0:
        return [kryo]
    sudo = shutil.which("sudo")
    if sudo is None:
        raise PermissionError("kryo snapshot/run need root; sudo not found")
    path = os.environ.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin")
    return [sudo, "-n", "-E", "env", f"PATH={path}", kryo]


def python_command(scenario: str) -> list[str]:
    """Absolute interpreter + scenario script so sudo/uv cannot change cwd meaning."""
    if scenario not in ALL_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    script = (SCENARIOS_DIR / f"{scenario}.py").resolve()
    if not script.exists():
        raise FileNotFoundError(f"Scenario script not found: {script}")
    return [sys.executable, str(script)]


def run_timed(command: list[str], cwd: Path) -> tuple[float, str]:
    """Run a command and return wall-clock seconds plus combined output."""
    started = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_SECONDS,
        check=False,
    )
    elapsed = time.perf_counter() - started
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {output[-4000:]}")
    return elapsed, output


def measure_runs(command: list[str], cwd: Path, runs: int) -> dict[str, Any]:
    """Time a command `runs` times, dropping the first sample as warmup."""
    samples: list[float] = []
    errors: list[str] = []
    total_attempts = runs + 1
    for attempt in range(total_attempts):
        label = "warmup" if attempt == 0 else f"{attempt}/{runs}"
        print(f"    {label}...", end=" ", flush=True)
        try:
            elapsed, _ = run_timed(command, cwd)
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            print("FAILED")
            errors.append(str(error))
            continue
        print(f"{elapsed:.3f}s")
        if attempt > 0:
            samples.append(elapsed)

    if not samples:
        return {"error": "All runs failed", "errors": errors}

    result: dict[str, Any] = {
        "runs": len(samples),
        "total": compute_stats(samples),
    }
    if errors:
        result["errors"] = errors
    return result


def run_scenario(scenario: str, runs: int) -> dict[str, Any]:
    """Create one snapshot, then time cold start vs Kryo restore."""
    python = python_command(scenario)
    kryo = kryo_command()
    cwd = SCENARIOS_DIR
    snapshot = f"bench-{scenario}"

    print(f"  cold start ({runs} runs, 1 warmup)")
    cold = measure_runs(python, cwd, runs)

    print("  creating snapshot")
    subprocess.run([*kryo, "snapshot", "delete", snapshot], check=False, capture_output=True)
    try:
        run_timed([*kryo, "snapshot", "create", "--name", snapshot, "--", *python], cwd)
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        return {
            "cold": cold,
            "kryo": {"error": f"snapshot create failed: {error}"},
        }

    print(f"  kryo restore ({runs} runs, 1 warmup)")
    restored = measure_runs([*kryo, "run", "--snapshot", snapshot], cwd, runs)
    subprocess.run([*kryo, "snapshot", "delete", snapshot], check=False, capture_output=True)

    result: dict[str, Any] = {"cold": cold, "kryo": restored}
    cold_mean = cold.get("total", {}).get("mean")
    kryo_mean = restored.get("total", {}).get("mean")
    if isinstance(cold_mean, float) and isinstance(kryo_mean, float) and kryo_mean > 0:
        result["speedup"] = cold_mean / kryo_mean
    return result


def run_all(scenarios: list[str], runs: int) -> dict[str, Any]:
    """Run every requested scenario and attach host metadata."""
    results: dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now(UTC).isoformat(),
            "runs_per_mode": runs,
            **gpu_metadata(),
        },
        "scenarios": {},
    }
    instance_type = os.environ.get("BENCH_INSTANCE_TYPE")
    if instance_type:
        results["metadata"]["instance_type"] = instance_type

    for scenario in scenarios:
        print(f"\nScenario: {scenario}")
        try:
            results["scenarios"][scenario] = run_scenario(scenario, runs)
        except (ValueError, OSError, RuntimeError) as error:
            print(f"  Error: {error}")
            results["scenarios"][scenario] = {"error": str(error)}
    return results


def main() -> None:
    """CLI entrypoint for cold vs Kryo restore benchmarks."""
    parser = argparse.ArgumentParser(description="Run Kryo with/without snapshot benchmarks")
    parser.add_argument("--scenario", choices=ALL_SCENARIOS, help="Single scenario")
    parser.add_argument("--all", action="store_true", help="Run all scenarios")
    parser.add_argument("--runs", type=int, default=10, help="Timed runs per mode (default: 10)")
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be >= 1")

    if args.scenario:
        scenarios = [args.scenario]
    elif args.all:
        scenarios = ALL_SCENARIOS
    else:
        parser.print_help()
        print("\nError: Must specify --scenario or --all")
        sys.exit(1)

    results = run_all(scenarios, args.runs)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else RESULTS_DIR / "latest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"\nResults saved to: {output_path}")
    for name, data in results["scenarios"].items():
        if "error" in data:
            print(f"  {name}: ERROR {data['error']}")
            continue
        cold = data.get("cold", {}).get("total", {}).get("mean")
        kryo = data.get("kryo", {}).get("total", {}).get("mean")
        speedup = data.get("speedup")
        if isinstance(cold, float) and isinstance(kryo, float):
            extra = f"  ({speedup:.1f}x)" if isinstance(speedup, float) else ""
            print(f"  {name}: cold {cold:.3f}s  kryo {kryo:.3f}s{extra}")
        else:
            print(f"  {name}: incomplete ({data})")


if __name__ == "__main__":
    main()
