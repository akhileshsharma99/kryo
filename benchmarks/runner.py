"""Run cold vs Kryo-restore timings for GPU scenarios.

Must run on a Linux NVIDIA box with CRIU, cuda-checkpoint, and the Kryo CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
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

# Optional probes; not part of --all / release CI.
OPTIONAL_SCENARIOS = [
    "qwen7",
    "qwen32",
    "vllm_engine",
    "torch_compile",
]

KNOWN_SCENARIOS = ALL_SCENARIOS + OPTIONAL_SCENARIOS

DEFAULT_TIMEOUT_SECONDS = 90


@dataclass(frozen=True)
class BenchFlags:
    """How timed runs treat disk cache, warmup, and leftover snapshots."""

    drop_caches: bool = True
    warmup: bool = True
    keep_snapshot: bool = False
    tmpfs_snapshots: bool = False


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


def kryo_version() -> str | None:
    """Best-effort Kryo CLI version string."""
    kryo = shutil.which("kryo")
    if kryo is None:
        return None
    query = subprocess.run([kryo, "--version"], capture_output=True, text=True, check=False)
    if query.returncode != 0:
        return None
    text = (query.stdout or query.stderr).strip()
    return text or None


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
    """Prefix that runs the Kryo CLI with root, keeping PATH and snapshot env."""
    kryo = shutil.which("kryo")
    if kryo is None:
        raise FileNotFoundError("kryo CLI not found on PATH")
    extra: list[str] = []
    snapshots = os.environ.get("KRYO_SNAPSHOTS_DIR", "").strip()
    if snapshots:
        extra.append(f"KRYO_SNAPSHOTS_DIR={snapshots}")
    lazy = os.environ.get("KRYO_LAZY_PAGES", "").strip()
    if lazy:
        extra.append(f"KRYO_LAZY_PAGES={lazy}")
    if os.geteuid() == 0:
        if extra:
            return ["env", *extra, kryo]
        return [kryo]
    sudo = shutil.which("sudo")
    if sudo is None:
        raise PermissionError("kryo snapshot/run need root; sudo not found")
    path = os.environ.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin")
    return [sudo, "-n", "-E", "env", f"PATH={path}", *extra, kryo]


def python_command(scenario: str) -> list[str]:
    """Absolute interpreter + scenario script so sudo/uv cannot change cwd meaning."""
    if scenario not in KNOWN_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    script = (SCENARIOS_DIR / f"{scenario}.py").resolve()
    if not script.exists():
        raise FileNotFoundError(f"Scenario script not found: {script}")
    return [sys.executable, str(script)]


def scenario_env() -> dict[str, str]:
    """Keep HuggingFace from leaving TCP sockets that CRIU cannot dump."""
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("YOLO_OFFLINE", "True")
    return env


def kill_process_group(pid: int) -> None:
    """Kill a timed-out command and anything in its session."""
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        with suppress(ProcessLookupError, PermissionError, OSError):
            os.kill(pid, signal.SIGKILL)
        sudo = shutil.which("sudo")
        if sudo is not None:
            subprocess.run([sudo, "-n", "kill", "-9", f"-{pid}"], check=False, capture_output=True)
            subprocess.run([sudo, "-n", "kill", "-9", str(pid)], check=False, capture_output=True)


def kill_stray_scenarios() -> None:
    """CRIU restore reparents the workload to PID 1, so killing kryo is not enough."""
    pkill = shutil.which("pkill")
    if pkill is None:
        return
    prefix: list[str] = []
    sudo = shutil.which("sudo")
    if os.geteuid() != 0 and sudo is not None:
        prefix = [sudo, "-n"]
    for scenario in KNOWN_SCENARIOS:
        script = str((SCENARIOS_DIR / f"{scenario}.py").resolve())
        subprocess.run(
            [*prefix, pkill, "-9", "-f", script], check=False, capture_output=True
        )
    subprocess.run(
        [*prefix, pkill, "-9", "-f", "criu lazy-pages"], check=False, capture_output=True
    )
    subprocess.run([*prefix, pkill, "-9", "-f", "VLLM::"], check=False, capture_output=True)
    subprocess.run(
        [*prefix, pkill, "-9", "-f", "/usr/local/bin/kryo run"],
        check=False,
        capture_output=True,
    )


def sudo_prefix() -> list[str]:
    """Root prefix for privileged bench helpers."""
    if os.geteuid() == 0:
        return []
    sudo = shutil.which("sudo")
    if sudo is None:
        raise PermissionError("sudo not found")
    return [sudo, "-n"]


def drop_page_cache() -> None:
    """Evict file pages so timed runs read weights/snapshots from disk."""
    drop = Path("/proc/sys/vm/drop_caches")
    if not drop.is_file():
        raise RuntimeError("drop_caches is not available (need Linux)")
    subprocess.run([*sudo_prefix(), "sync"], check=True)
    result = subprocess.run(
        [*sudo_prefix(), "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"drop_caches failed: {detail}")


def configure_tmpfs_snapshots() -> str:
    """Put CRIU images on tmpfs. Not used for prod-fair benches."""
    configured = os.environ.get("KRYO_SNAPSHOTS_DIR", "").strip()
    if configured:
        Path(configured).mkdir(parents=True, exist_ok=True)
        return configured

    shm = Path("/dev/shm")
    ram = memtotal_bytes()
    if ram >= 64 * 1024**3 and shutil.which("sudo") is not None:
        size_kb = max(1, int(ram * 0.7 / 1024))
        subprocess.run(
            ["sudo", "-n", "mount", "-o", f"remount,size={size_kb}k", "/dev/shm"],
            check=False,
            capture_output=True,
        )

    target = Path("/dev/shm/kryo-snapshots")
    if shm.is_dir() and available_bytes(shm) >= 24 * 1024**3:
        target.mkdir(parents=True, exist_ok=True)
        os.environ["KRYO_SNAPSHOTS_DIR"] = str(target)
        return str(target)

    fallback = Path("/tmp/kryo-snapshots")
    fallback.mkdir(parents=True, exist_ok=True)
    os.environ["KRYO_SNAPSHOTS_DIR"] = str(fallback)
    return str(fallback)


def available_bytes(path: Path) -> int:
    """Free bytes on the filesystem that contains path."""
    try:
        stats = os.statvfs(path)
    except OSError:
        return 0
    return int(stats.f_bavail * stats.f_frsize)


def memtotal_bytes() -> int:
    """Host RAM from /proc/meminfo, or 0 if unavailable."""
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return 0
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if not line.startswith("MemTotal:"):
            continue
        fields = line.split()
        try:
            return int(fields[1]) * 1024
        except (IndexError, ValueError):
            return 0
    return 0


def run_timed(
    command: list[str], cwd: Path, timeout: int, *, drop_caches: bool = False
) -> tuple[float, str]:
    """Run a command and return wall-clock seconds plus combined output."""
    if drop_caches:
        drop_page_cache()
    started = time.perf_counter()
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=scenario_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_process_group(proc.pid)
        kill_stray_scenarios()
        leftover = ""
        try:
            leftover, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            leftover = ""
        raise RuntimeError(f"timed out after {timeout}s: {(leftover or '')[-2000:]}") from None
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {(output or '')[-4000:]}")
    return elapsed, output or ""


def measure_runs(
    command: list[str],
    cwd: Path,
    runs: int,
    timeout: int,
    flags: BenchFlags,
) -> dict[str, Any]:
    """Time a command `runs` times. The first attempt is warmup unless disabled."""
    samples: list[float] = []
    errors: list[str] = []
    total_attempts = runs + (1 if flags.warmup else 0)
    for attempt in range(total_attempts):
        is_warmup = flags.warmup and attempt == 0
        if is_warmup:
            label = "warmup"
        else:
            numbered = attempt + 1 if not flags.warmup else attempt
            label = f"{numbered}/{runs}"
        print(f"    {label}...", end=" ", flush=True)
        try:
            elapsed, _ = run_timed(command, cwd, timeout, drop_caches=flags.drop_caches)
        except RuntimeError as error:
            print("FAILED")
            message = str(error).strip().splitlines()[-1] if str(error).strip() else str(error)
            print(f"      {message}")
            errors.append(str(error))
            kill_stray_scenarios()
            continue
        print(f"{elapsed:.3f}s")
        if not is_warmup:
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


def snapshot_bytes(name: str) -> int | None:
    """CRIU image size on disk after dump, via `kryo snapshot inspect`."""
    inspect = subprocess.run(
        [*kryo_command(), "snapshot", "inspect", name],
        capture_output=True,
        text=True,
        check=False,
    )
    path_line = next(
        (line for line in inspect.stdout.splitlines() if line.startswith("Path:")),
        None,
    )
    if path_line is None:
        return None
    images = Path(path_line.split(":", 1)[1].strip()) / "images"
    prefix = [] if os.geteuid() == 0 else ["sudo", "-n"]
    du = subprocess.run(
        [*prefix, "du", "-sb", str(images)],
        capture_output=True,
        text=True,
        check=False,
    )
    if du.returncode != 0 or not du.stdout.strip():
        return None
    try:
        return int(du.stdout.split()[0])
    except ValueError:
        return None


def kryo_run_command(snapshot: str) -> list[str]:
    """Restore in a fresh PID namespace so dump PIDs are not already taken."""
    prefix = kryo_command()
    args = ["run", "--snapshot", snapshot]
    unshare = shutil.which("unshare")
    if unshare is None:
        return [*prefix, *args]
    return [
        *prefix[:-1],
        unshare,
        "--fork",
        "--pid",
        "--mount",
        "--mount-proc",
        "--",
        prefix[-1],
        *args,
    ]


def snapshot_dir(name: str) -> Path:
    """On-disk CRIU snapshot directory for this process."""
    configured = os.environ.get("KRYO_SNAPSHOTS_DIR", "").strip()
    base = Path(configured) if configured else Path("/root/.kryo/snapshots")
    return base / name


def snapshot_name(scenario: str) -> str:
    """Stable CRIU snapshot name for a scenario."""
    return f"bench-{scenario}"


def prepare_for_restore(name: str) -> None:
    """Drop leftover restore/cold processes so CRIU can recreate dump PIDs."""
    kill_stray_scenarios()
    meta = snapshot_dir(name) / "metadata.json"
    inspect = subprocess.run(
        [*sudo_prefix(), "cat", str(meta)],
        capture_output=True,
        text=True,
        check=False,
    )
    if inspect.returncode != 0 or not inspect.stdout.strip():
        return
    try:
        data = json.loads(inspect.stdout)
    except json.JSONDecodeError:
        return
    pid = data.get("workload_pid")
    if isinstance(pid, int) and pid > 1:
        subprocess.run([*sudo_prefix(), "kill", "-9", str(pid)], check=False, capture_output=True)


def run_once(
    scenario: str,
    mode: str,
    timeout: int,
    flags: BenchFlags,
) -> dict[str, Any]:
    """One untimed create, or one timed cold/restore sample. No warmup."""
    python = python_command(scenario)
    kryo = kryo_command()
    cwd = SCENARIOS_DIR
    snapshot = snapshot_name(scenario)

    if mode == "cold":
        print("  cold start (1 sample)")
        elapsed, _ = run_timed(python, cwd, timeout, drop_caches=flags.drop_caches)
        print(f"    1/1 {elapsed:.3f}s")
        return {
            "scenario": scenario,
            "mode": "cold",
            "seconds": elapsed,
            "cold": {"runs": 1, "samples": [elapsed], "total": compute_stats([elapsed])},
        }

    if mode == "create":
        print("  creating snapshot")
        subprocess.run([*kryo, "snapshot", "delete", snapshot], check=False, capture_output=True)
        run_timed([*kryo, "snapshot", "create", "--name", snapshot, "--", *python], cwd, timeout)
        image_bytes = snapshot_bytes(snapshot)
        if image_bytes is not None:
            print(f"  snapshot images {image_bytes / (1024**3):.2f} GiB")
        kill_stray_scenarios()
        result: dict[str, Any] = {"scenario": scenario, "mode": "create"}
        if image_bytes is not None:
            result["snapshot_bytes"] = image_bytes
        return result

    if mode == "restore":
        print("  kryo restore (1 sample)")
        kill_stray_scenarios()
        elapsed, _ = run_timed(
            [*kryo, "run", "--snapshot", snapshot], cwd, timeout, drop_caches=flags.drop_caches
        )
        print(f"    1/1 {elapsed:.3f}s")
        if not flags.keep_snapshot:
            subprocess.run(
                [*kryo, "snapshot", "delete", snapshot],
                check=False,
                capture_output=True,
            )
        kill_stray_scenarios()
        return {
            "scenario": scenario,
            "mode": "restore",
            "seconds": elapsed,
            "kryo": {"runs": 1, "samples": [elapsed], "total": compute_stats([elapsed])},
        }

    raise ValueError(f"unknown --once mode: {mode}")


def run_scenario(
    scenario: str,
    runs: int,
    timeout: int,
    flags: BenchFlags,
) -> dict[str, Any]:
    """Create one snapshot, then time cold start vs Kryo restore."""
    python = python_command(scenario)
    kryo = kryo_command()
    cwd = SCENARIOS_DIR
    snapshot = snapshot_name(scenario)
    extra = ", 1 warmup" if flags.warmup else ""

    print(f"  cold start ({runs} runs{extra})")
    cold = measure_runs(python, cwd, runs, timeout, flags)

    print("  creating snapshot")
    subprocess.run([*kryo, "snapshot", "delete", snapshot], check=False, capture_output=True)
    try:
        run_timed([*kryo, "snapshot", "create", "--name", snapshot, "--", *python], cwd, timeout)
    except RuntimeError as error:
        kill_stray_scenarios()
        return {
            "cold": cold,
            "kryo": {"error": f"snapshot create failed: {error}"},
        }

    image_bytes = snapshot_bytes(snapshot)
    if image_bytes is not None:
        print(f"  snapshot images {image_bytes / (1024**3):.2f} GiB")

    print(f"  kryo restore ({runs} runs{extra})")
    prepare_for_restore(snapshot)
    restored = measure_runs(
        [*kryo, "run", "--snapshot", snapshot],
        cwd,
        runs,
        timeout,
        flags,
    )
    if not flags.keep_snapshot:
        subprocess.run([*kryo, "snapshot", "delete", snapshot], check=False, capture_output=True)
    kill_stray_scenarios()

    result: dict[str, Any] = {"cold": cold, "kryo": restored}
    if image_bytes is not None:
        result["snapshot_bytes"] = image_bytes
    cold_mean = cold.get("total", {}).get("mean")
    kryo_mean = restored.get("total", {}).get("mean")
    if isinstance(cold_mean, float) and isinstance(kryo_mean, float) and kryo_mean > 0:
        result["speedup"] = cold_mean / kryo_mean
    return result


def prepare_snapshot_env(*, tmpfs_snapshots: bool = False) -> str:
    """Set snapshot directory and lazy-pages defaults used by timed runs."""
    if tmpfs_snapshots:
        snapshots_dir = configure_tmpfs_snapshots()
    else:
        configured = os.environ.get("KRYO_SNAPSHOTS_DIR", "").strip()
        snapshots_dir = configured or "default-disk"
        if configured:
            subprocess.run([*sudo_prefix(), "mkdir", "-p", configured], check=False)
    if not os.environ.get("KRYO_LAZY_PAGES", "").strip():
        os.environ["KRYO_LAZY_PAGES"] = "0"
    print(f"  snapshots dir {snapshots_dir}")
    print(f"  lazy pages {os.environ.get('KRYO_LAZY_PAGES')}")
    return snapshots_dir


def run_all(
    scenarios: list[str],
    runs: int,
    timeout: int,
    flags: BenchFlags,
) -> dict[str, Any]:
    """Run every requested scenario and attach host metadata."""
    snapshots_dir = prepare_snapshot_env(tmpfs_snapshots=flags.tmpfs_snapshots)
    print(f"  drop page cache {flags.drop_caches}")
    version = kryo_version()
    results: dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now(UTC).isoformat(),
            "runs_per_mode": runs,
            "timeout_seconds": timeout,
            "snapshots_dir": snapshots_dir,
            "lazy_pages": os.environ.get("KRYO_LAZY_PAGES", ""),
            "drop_caches": flags.drop_caches,
            **gpu_metadata(),
        },
        "scenarios": {},
    }
    if version:
        results["metadata"]["kryo"] = version
    for key, env_name in (
        ("instance_type", "BENCH_INSTANCE_TYPE"),
        ("git_sha", "BENCH_GIT_SHA"),
        ("release_tag", "BENCH_RELEASE_TAG"),
    ):
        value = os.environ.get(env_name, "").strip()
        if value:
            results["metadata"][key] = value

    for scenario in scenarios:
        print(f"\nScenario: {scenario}")
        try:
            results["scenarios"][scenario] = run_scenario(scenario, runs, timeout, flags)
        except (ValueError, OSError, RuntimeError) as error:
            print(f"  Error: {error}")
            results["scenarios"][scenario] = {"error": str(error)}
            kill_stray_scenarios()
    return results


def select_scenarios(parser: argparse.ArgumentParser, args: argparse.Namespace) -> list[str]:
    """Resolve --scenario / --scenarios / --all into a list of names."""
    if args.scenario:
        return [args.scenario]
    if args.scenarios:
        scenarios = [item.strip() for item in args.scenarios.split(",") if item.strip()]
        unknown = [item for item in scenarios if item not in KNOWN_SCENARIOS]
        if unknown:
            parser.error(f"unknown scenarios: {', '.join(unknown)}")
        return scenarios
    if args.all:
        return ALL_SCENARIOS
    parser.print_help()
    print("\nError: Must specify --scenario, --scenarios, or --all")
    sys.exit(1)


def print_summary(results: dict[str, Any], *, once: bool, scenarios: list[str]) -> None:
    """Print a short cold vs restore summary."""
    if once:
        seconds = results.get("seconds")
        mode = results.get("mode")
        if isinstance(seconds, float):
            print(f"  {scenarios[0]} {mode}: {seconds:.3f}s")
        else:
            print(f"  {scenarios[0]} {mode}: {results}")
        return
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


def main() -> None:
    """CLI entrypoint for cold vs Kryo restore benchmarks."""
    parser = argparse.ArgumentParser(description="Run Kryo with/without snapshot benchmarks")
    parser.add_argument("--scenario", choices=KNOWN_SCENARIOS, help="Single scenario")
    parser.add_argument(
        "--scenarios",
        help="Comma-separated scenarios (qwen7, vllm_engine, torch_compile, ...)",
    )
    parser.add_argument("--all", action="store_true", help="Run release scenarios only")
    parser.add_argument("--runs", type=int, default=10, help="Timed runs per mode (default: 10)")
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Seconds before a hung run is killed (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    parser.add_argument(
        "--drop-caches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop the Linux page cache before each timed run (default: on)",
    )
    parser.add_argument(
        "--tmpfs-snapshots",
        action="store_true",
        help="Store CRIU images on tmpfs (not prod-fair; default is disk)",
    )
    parser.add_argument(
        "--once",
        choices=["cold", "create", "restore"],
        help="One sample: timed cold, untimed snapshot create, or timed restore",
    )
    parser.add_argument(
        "--keep-snapshot",
        action="store_true",
        help="Leave the CRIU snapshot on disk after restore (for reuse)",
    )
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop the first attempt as warmup (default: on; --once never warms up)",
    )
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if args.timeout < 1:
        parser.error("--timeout must be >= 1")

    scenarios = select_scenarios(parser, args)
    flags = BenchFlags(
        drop_caches=args.drop_caches,
        warmup=False if args.once else args.warmup,
        keep_snapshot=True if args.once else args.keep_snapshot,
        tmpfs_snapshots=args.tmpfs_snapshots,
    )

    if args.once:
        if len(scenarios) != 1:
            parser.error("--once requires exactly one --scenario")
        prepare_snapshot_env(tmpfs_snapshots=flags.tmpfs_snapshots)
        print(f"  drop page cache {flags.drop_caches}")
        results = run_once(scenarios[0], args.once, args.timeout, flags)
    else:
        results = run_all(scenarios, args.runs, args.timeout, flags)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else RESULTS_DIR / "latest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"\nResults saved to: {output_path}")
    print_summary(results, once=bool(args.once), scenarios=scenarios)


if __name__ == "__main__":
    main()
