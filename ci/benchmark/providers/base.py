"""Cloud GPU provider protocol. Scheduler talks only to this surface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Machine:
    """One rented GPU VM. Connection details stay inside the provider."""

    id: str
    sku: str
    name: str
    region: str = ""
    filesystem: str = ""
    requested: str = ""


class Provider(Protocol):
    """Rent, talk to, and destroy GPU VMs. Implementations are cloud-specific."""

    def janitor(self) -> None:
        """Terminate leaked CI instances from crashed controller runs."""

    def launch(self, sku: str, filesystem: str | None = None) -> Machine:
        """Create a VM of this SKU (or `auto`) and wait until SSH works.

        If filesystem is set, attach (and create if needed) a persistent disk
        in the launch region so golden tarballs survive VM teardown.
        """

    def rsync(self, machine: Machine) -> None:
        """Copy the Kryo checkout onto the VM."""

    def run(self, machine: Machine, command: str, timeout: int | None = None) -> None:
        """Run a remote shell command, streaming output."""

    def run_output(self, machine: Machine, command: str, timeout: int | None = None) -> str:
        """Run a remote shell command and return stdout."""

    def put(self, machine: Machine, local: Path, remote: str) -> None:
        """Copy a local file onto the VM."""

    def get(self, machine: Machine, remote: str, local: Path) -> None:
        """Copy a remote file onto the controller."""

    def terminate(self, machine: Machine) -> None:
        """Destroy the VM and its SSH key."""
