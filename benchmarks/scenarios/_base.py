"""Shared helpers for Kryo with/without cold-start scenarios."""

import os

from kryo import checkpoint


def maybe_checkpoint() -> None:
    """Freeze under Kryo, then exit.

    Benchmarks measure time-to-ready, not interpreter shutdown. Ultralytics
    and other CUDA apps can deadlock joining threads after CRIU restore.
    """
    if os.environ.get("KRYO_CLI_PID"):
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except ImportError:
            pass
        checkpoint()
    os._exit(0)
