"""Shared helpers for Kryo with/without cold-start scenarios."""

from __future__ import annotations

import os
from collections.abc import Callable

from kryo import checkpoint


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except ImportError:
        pass


def checkpoint_or_exit(*, resume: Callable[[], None] | None = None) -> None:
    """Dump under Kryo, then either resume (restore) or exit (cold start).

    Cold start: the caller already did first inference, then we _exit.
    Snapshot create: CRIU kills the process at dump; resume does not run.
    Restore: dump returns, resume() is first inference with weights already
    on the GPU, then we _exit so shutdown is not timed.
    """
    if os.environ.get("KRYO_CLI_PID"):
        _empty_cuda_cache()
        checkpoint()
        if resume is not None:
            resume()
    os._exit(0)


def maybe_checkpoint() -> None:
    """Dump under Kryo without a post-restore inference, then exit."""
    checkpoint_or_exit()
