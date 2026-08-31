"""Content-addressed golden-image directories (tools + venv, not CRIU)."""

from __future__ import annotations

import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / ".golden"
SETUP_SCRIPTS = (
    HERE / "setup.sh",
    HERE / "pack_golden.sh",
    HERE / "apply_golden.sh",
)
DIGEST_FILE = "/var/lib/kryo-bench/golden.digest"


def digest(sku: str, driver: str, cuda: str) -> str:
    """Hash that must change when a restored golden would be the wrong image."""
    hasher = hashlib.sha256()
    hasher.update(sku.encode())
    hasher.update(b"\0")
    hasher.update(driver.encode())
    hasher.update(b"\0")
    hasher.update(cuda.encode())
    hasher.update(b"\0")
    for path in SETUP_SCRIPTS:
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()[:20]


def local_path(sku: str, golden_digest: str) -> Path:
    """Controller cache path for one golden directory."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{sku}-{golden_digest}"


def nfs_dir(filesystem: str) -> str:
    """Lambda mount point for an attached filesystem."""
    return f"/lambda/nfs/{filesystem}"


def nfs_path(filesystem: str, sku: str, golden_digest: str) -> str:
    """On-box path of a golden directory on the attached filesystem."""
    return f"{nfs_dir(filesystem)}/golden/{sku}-{golden_digest}"


def read_digest_command() -> str:
    """Print the on-box golden digest, or empty if missing."""
    return f"sudo cat {DIGEST_FILE} 2>/dev/null || true"


def write_digest_command(golden_digest: str) -> str:
    """Record that this VM was brought up from this golden."""
    return (
        "sudo mkdir -p /var/lib/kryo-bench && "
        f"echo {golden_digest} | sudo tee {DIGEST_FILE} >/dev/null"
    )


def pack_command(destination: str) -> str:
    """Copy the golden tree to destination (NFS directory)."""
    return f"bash kryo/ci/benchmark/pack_golden.sh {destination}"


def apply_command(source: str) -> str:
    """Copy a golden directory onto the root disk."""
    return f"bash kryo/ci/benchmark/apply_golden.sh {source}"
