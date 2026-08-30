"""Load YAML job files for the GPU benchmark scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

KNOWN_SCENARIOS = (
    "torch_cuda",
    "yolo",
    "qwen",
    "whisper",
    "qwen7",
    "qwen32",
    "torch_compile",
)

SCENARIO_WEIGHTS: dict[str, str] = {
    "torch_cuda": "",
    "yolo": "yolov8n.pt",
    "qwen": "Qwen/Qwen2.5-0.5B",
    "whisper": "openai/whisper-tiny",
    "qwen7": "Qwen/Qwen2.5-7B",
    "qwen32": "Qwen/Qwen2.5-32B",
    "torch_compile": "Qwen/Qwen2.5-7B",
}

LLM_SCENARIOS = frozenset(SCENARIO_WEIGHTS) - {"torch_cuda", "yolo", "whisper"}

DEFAULT_IDLE_TIMEOUT = 600
DEFAULT_TIMEOUT = 90
DEFAULT_SAMPLES = 10
DEFAULT_RETRIES = 2


@dataclass(frozen=True)
class GoldenConfig:
    """How to bring a fresh VM to a runnable image.

    `setup` always runs setup.sh.
    `tarball` restores a packed image (local cache and/or a Lambda filesystem).
    CRIU GPU snapshots are not part of the golden image.
    """

    mode: str = "tarball"
    store: str = "filesystem"
    filesystem: str = "kryo-golden"


@dataclass(frozen=True)
class SnapshotConfig:
    """CRIU dumps stay on the VM that created them (`local`)."""

    store: str = "local"


@dataclass(frozen=True)
class Job:
    """One scenario queued onto a GPU SKU pool."""

    scenario: str
    gpu: str
    samples: int
    timeout: int
    retries: int = DEFAULT_RETRIES


@dataclass(frozen=True)
class BenchPlan:
    """Fully parsed job file."""

    path: Path
    provider: str
    idle_timeout: int
    caps: dict[str, int]
    golden: GoldenConfig
    snapshots: SnapshotConfig
    jobs: tuple[Job, ...]


def parse_duration(value: object) -> int:
    """Parse seconds from an int or a string like `10m`, `1h`, `30s`."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"invalid duration: {value!r}")
    if isinstance(value, (int, float)):
        seconds = int(value)
        if seconds < 1:
            raise ValueError("duration must be >= 1s")
        return seconds
    text = value.strip().lower()
    if not text:
        raise ValueError("empty duration")
    multipliers = {"s": 1, "m": 60, "h": 3600}
    suffix = text[-1]
    if suffix in multipliers:
        number = text[:-1]
        unit = multipliers[suffix]
    else:
        number = text
        unit = 1
    try:
        amount = float(number)
    except ValueError as error:
        raise ValueError(f"invalid duration: {value!r}") from error
    seconds = int(amount * unit)
    if seconds < 1:
        raise ValueError("duration must be >= 1s")
    return seconds


def _require_mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _parse_caps(raw: object) -> dict[str, int]:
    if raw is None:
        return {}
    mapping = _require_mapping(raw, "caps")
    caps: dict[str, int] = {}
    for sku, limit in mapping.items():
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError(f"caps.{sku} must be an integer >= 1")
        caps[sku] = limit
    return caps


def _parse_job(raw: object, index: int) -> Job:
    if not isinstance(raw, dict):
        raise ValueError(f"jobs[{index}] must be a mapping")
    scenario = raw.get("scenario")
    if not isinstance(scenario, str) or scenario not in KNOWN_SCENARIOS:
        raise ValueError(f"jobs[{index}].scenario is unknown: {scenario!r}")
    gpu = raw.get("gpu")
    if not isinstance(gpu, str) or not gpu.strip():
        raise ValueError(f"jobs[{index}].gpu is required")
    samples = raw.get("samples", DEFAULT_SAMPLES)
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 1:
        raise ValueError(f"jobs[{index}].samples must be an integer >= 1")
    timeout = raw.get("timeout", DEFAULT_TIMEOUT)
    timeout_seconds = parse_duration(timeout)
    retries = raw.get("retries", DEFAULT_RETRIES)
    if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
        raise ValueError(f"jobs[{index}].retries must be an integer >= 0")
    return Job(
        scenario=scenario,
        gpu=gpu.strip(),
        samples=samples,
        timeout=timeout_seconds,
        retries=retries,
    )


def load_plan(path: Path) -> BenchPlan:
    """Parse a YAML job file into a BenchPlan."""
    text = path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(text)
    data = _require_mapping(loaded, path.name)
    provider = data.get("provider", "lambda")
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("provider must be a non-empty string")
    idle = parse_duration(data.get("idle_timeout", DEFAULT_IDLE_TIMEOUT))
    golden_raw = data.get("golden") or {}
    golden_map = _require_mapping(golden_raw, "golden") if golden_raw else {}
    golden_mode = str(golden_map.get("mode", "tarball"))
    if golden_mode not in {"setup", "tarball"}:
        raise ValueError(f"unsupported golden.mode: {golden_mode!r} (setup or tarball)")
    default_store = "filesystem" if golden_mode == "tarball" else "local"
    golden_store = str(golden_map.get("store", default_store))
    if golden_store not in {"local", "filesystem"}:
        raise ValueError(f"unsupported golden.store: {golden_store!r} (local or filesystem)")
    if golden_mode == "setup":
        golden_store = "local"
    filesystem_raw = golden_map.get("filesystem", "kryo-golden")
    if not isinstance(filesystem_raw, str) or not filesystem_raw.strip():
        raise ValueError("golden.filesystem must be a non-empty string")
    filesystem = filesystem_raw.strip()
    snap_raw = data.get("snapshots") or {}
    snap_map = _require_mapping(snap_raw, "snapshots") if snap_raw else {}
    snap_store = str(snap_map.get("store", "local"))
    if snap_store not in {"local", "filesystem"}:
        raise ValueError(f"unsupported snapshots.store: {snap_store!r} (local or filesystem)")
    jobs_raw = data.get("jobs")
    if not isinstance(jobs_raw, list) or not jobs_raw:
        raise ValueError("jobs must be a non-empty list")
    jobs = tuple(_parse_job(item, index) for index, item in enumerate(jobs_raw))
    caps = _parse_caps(data.get("caps"))
    for job in jobs:
        if job.gpu not in caps:
            caps = {**caps, job.gpu: 1}
    return BenchPlan(
        path=path,
        provider=provider.strip(),
        idle_timeout=idle,
        caps=caps,
        golden=GoldenConfig(
            mode=golden_mode,
            store=golden_store,
            filesystem=filesystem,
        ),
        snapshots=SnapshotConfig(store=snap_store),
        jobs=jobs,
    )
