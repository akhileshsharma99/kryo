"""Content-addressed CRIU snapshot tarballs on the controller."""

from __future__ import annotations

import hashlib
import shlex
from pathlib import Path

from config import SCENARIO_WEIGHTS
from golden import nfs_dir

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = Path(__file__).resolve().parent / ".snapshots"
SCENARIOS_DIR = REPO_ROOT / "benchmarks" / "scenarios"
# Must match KRYO_SNAPSHOTS_DIR in scheduler.remote_env. sudo -E keeps HOME=/home/ubuntu,
# so Kryo's default ~/.kryo/snapshots is not /root/.kryo/snapshots.
REMOTE_SNAP_ROOT = "/var/lib/kryo-bench/criu"
REMOTE_HASH_ROOT = "/var/lib/kryo-bench/snapshots"


def snapshot_id(name: str) -> str:
    """CRIU snapshot directory name used by runner.py."""
    return f"bench-{name}"


def digest(scenario: str, sku: str, kryo_version: str, driver: str) -> str:
    """Hash that must change when restore would be invalid."""
    script = SCENARIOS_DIR / f"{scenario}.py"
    hasher = hashlib.sha256()
    hasher.update(scenario.encode())
    hasher.update(b"\0")
    hasher.update(sku.encode())
    hasher.update(b"\0")
    hasher.update(kryo_version.encode())
    hasher.update(b"\0")
    hasher.update(driver.encode())
    hasher.update(b"\0")
    hasher.update(SCENARIO_WEIGHTS.get(scenario, "").encode())
    hasher.update(b"\0")
    hasher.update(script.read_bytes())
    return hasher.hexdigest()[:20]


def tarball_path(scenario: str, sku: str, snap_digest: str) -> Path:
    """Local cache path for one snapshot tarball."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{scenario}-{sku}-{snap_digest}.tgz"


def remote_hash_path(scenario: str) -> str:
    """On-box file storing the digest currently unpacked for this scenario."""
    return f"{REMOTE_HASH_ROOT}/{scenario}.hash"


def nfs_tarball(filesystem: str, scenario: str, sku: str, snap_digest: str) -> str:
    """On-box path of a CRIU snapshot tarball on the attached filesystem."""
    return f"{nfs_dir(filesystem)}/snapshots/{scenario}-{sku}-{snap_digest}.tgz"


def pack_command(scenario: str, destination: str = "/tmp/kryo-snap.tgz") -> str:
    """Tar the CRIU directory on the VM so later restores can skip dump."""
    name = snapshot_id(scenario)
    quoted = shlex.quote(destination)
    parent = shlex.quote(destination.rsplit("/", 1)[0])
    # Sticky /tmp: do not write an archive that ubuntu already created.
    pack_tmp = (
        "sudo rm -f /tmp/kryo-snap.tgz && "
        f"sudo mkdir -p {REMOTE_SNAP_ROOT} && "
        f"sudo tar -C {REMOTE_SNAP_ROOT} -czf /tmp/kryo-snap.tgz {name} && "
        "sudo chmod 644 /tmp/kryo-snap.tgz"
    )
    if destination == "/tmp/kryo-snap.tgz":
        return pack_tmp
    return f"{pack_tmp} && mkdir -p {parent} && cp /tmp/kryo-snap.tgz {quoted}"


def unpack_command(scenario: str, snap_digest: str, source: str = "/tmp/kryo-snap.tgz") -> str:
    """Install a cached tarball as the live CRIU snapshot and record its digest."""
    name = snapshot_id(scenario)
    hash_path = remote_hash_path(scenario)
    quoted = shlex.quote(source)
    cleanup = f" && sudo rm -f {quoted}" if source == "/tmp/kryo-snap.tgz" else ""
    return (
        f"sudo mkdir -p {REMOTE_SNAP_ROOT} {REMOTE_HASH_ROOT} && "
        f"sudo rm -rf {REMOTE_SNAP_ROOT}/{name} && "
        f"sudo tar -C {REMOTE_SNAP_ROOT} -xzf {quoted}{cleanup} && "
        f"echo {snap_digest} | sudo tee {hash_path} >/dev/null"
    )


def write_hash_command(scenario: str, snap_digest: str) -> str:
    """Record that the on-box snapshot matches this digest."""
    hash_path = remote_hash_path(scenario)
    return (
        f"sudo mkdir -p {REMOTE_HASH_ROOT} && echo {snap_digest} | sudo tee {hash_path} >/dev/null"
    )


def read_hash_command(scenario: str) -> str:
    """Print the on-box digest, or empty if missing."""
    hash_path = remote_hash_path(scenario)
    return f"sudo cat {hash_path} 2>/dev/null || true"
