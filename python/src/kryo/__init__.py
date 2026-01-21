"""Kryo Python SDK for checkpoint signaling.

Usage:
    import kryo

    model = load_model()
    model.to("cuda")
    kryo.checkpoint()  # Freeze here

    result = model(input)  # Runs after restore
"""

import os
import signal
from importlib.metadata import version

__version__ = version("kryo")
__all__ = ["checkpoint"]

_checkpointed = False


def checkpoint() -> None:
    """Mark the checkpoint location and wait for restore.

    Call this after setup is complete. The process will:
    1. Send SIGUSR1 to parent (kryo CLI) to signal "ready"
    2. Block until SIGUSR2 is received (after restore)
    3. Continue execution

    This function is idempotent - subsequent calls are no-ops.
    """
    global _checkpointed
    if _checkpointed:
        return
    _checkpointed = True

    # Set up handler for wake signal (does nothing, just breaks pause)
    signal.signal(signal.SIGUSR2, lambda *_: None)

    # Tell parent we're ready to be checkpointed
    os.kill(os.getppid(), signal.SIGUSR1)

    # Wait until parent sends SIGUSR2 (after restore)
    signal.pause()
