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
import time
from importlib.metadata import version
from typing import Any

__version__ = version("kryo")
__all__ = ["checkpoint"]

_checkpointed = False
_CLI_PID_ENV = "KRYO_CLI_PID"


def _wake_signals() -> list[int]:
    """Signals that mean 'restore finished, continue inference'."""
    signals: list[int] = [int(signal.SIGUSR2)]
    rt_min = getattr(signal, "SIGRTMIN", None)
    if isinstance(rt_min, int):
        signals.append(int(rt_min) + 1)
    return signals


def checkpoint() -> None:
    """Mark the checkpoint location and wait for restore.

    Call this after setup is complete. The process will:
    1. Send SIGUSR1 to the kryo CLI to signal "ready"
    2. Block until SIGUSR2 is received (after restore)
    3. Continue execution

    This function is idempotent - subsequent calls are no-ops.
    """
    global _checkpointed
    if _checkpointed:
        return

    # Older Kryo CLI versions do not provide KRYO_CLI_PID. Falling back to the
    # parent preserves direct-launch compatibility; wrapper support requires
    # the new CLI protocol.
    cli_pid_value = os.environ.get(_CLI_PID_ENV, str(os.getppid()))

    try:
        cli_pid = int(cli_pid_value)
    except ValueError as error:
        raise RuntimeError(f"invalid {_CLI_PID_ENV}: {cli_pid_value!r}") from error

    if cli_pid <= 0:
        raise RuntimeError(f"invalid {_CLI_PID_ENV}: {cli_pid_value!r}")

    restored = False

    def wake(*_: object) -> None:
        nonlocal restored
        restored = True

    # CUDA restore often leaves SIGUSR2 ignored. SIGRTMIN+1 is the wake the
    # CLI sends; keep SIGUSR2 for older CLIs when the kernel still delivers it.
    previous: list[tuple[int, Any]] = []
    for sig in _wake_signals():
        previous.append((sig, signal.signal(sig, wake)))
    try:
        os.kill(cli_pid, signal.SIGUSR1)
        _checkpointed = True
        # Poll so the main thread regularly runs pending Python signal
        # handlers even if the wake was delivered to another thread.
        while not restored:
            time.sleep(0.1)
    finally:
        for sig, handler in previous:
            signal.signal(sig, handler)
