"""Shared helpers for Kryo with/without cold-start scenarios."""

import os

from kryo import checkpoint


def maybe_checkpoint() -> None:
    """Freeze when launched by `kryo snapshot create`; no-op for cold runs."""
    if os.environ.get("KRYO_CLI_PID"):
        checkpoint()
