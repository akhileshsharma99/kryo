"""Base utilities for cold start benchmarking."""

import json
import platform
import time
from typing import Any

_timings: dict[str, float] = {}
_start_time = time.perf_counter()


class Timing:
    """Context manager for timing code blocks.

    Usage:
        with Timing('import'):
            import torch
    """

    def __init__(self, name: str):
        self.name = name
        self.t0: float = 0.0

    def __enter__(self) -> "Timing":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        _timings[self.name] = time.perf_counter() - self.t0


def get_gpu_info() -> str | None:
    """Get GPU name if available."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return None


def output_results(extra_metadata: dict[str, Any] | None = None) -> None:
    """Output benchmark results as JSON to stdout."""
    # Capture total time BEFORE any extra work (like GPU detection)
    total_time = time.perf_counter() - _start_time

    metadata = {
        "python_version": platform.python_version(),
        "platform": platform.system(),
    }

    gpu = get_gpu_info()
    if gpu:
        metadata["gpu"] = gpu

    if extra_metadata:
        metadata.update(extra_metadata)

    result = {
        "metadata": metadata,
        "phases": _timings,
        "total": total_time,
    }

    print(json.dumps(result))
