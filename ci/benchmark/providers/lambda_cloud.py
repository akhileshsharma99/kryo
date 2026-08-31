"""Lambda Cloud GPU provider. The only file that talks to Lambda's API."""

from __future__ import annotations

import base64
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from providers.base import Machine

print = partial(print, flush=True)

API_BASE = "https://cloud.lambdalabs.com/api/v1"
INSTANCE_NAME_PREFIX = os.environ.get("KRYO_VM_PREFIX", "kryo-gha").strip() or "kryo-gha"
SSH_KEY_PREFIX = INSTANCE_NAME_PREFIX
SSH_USER = "ubuntu"
REPO_REMOTE = "kryo"
DEV_INSTANCE_NAME = "kryo-dev"

PREFERRED_INSTANCE_TYPES = [
    "gpu_1x_a10",
    "gpu_1x_l4",
    "gpu_1x_a6000",
    "gpu_1x_a100",
    "gpu_1x_a100_sxm4",
    "gpu_1x_l40s",
    "gpu_1x_h100",
    "gpu_1x_h100_pcie",
    "gpu_1x_h100_sxm5",
]

# Same generation / memory class. Used only when the requested SKU is sold out.
SKU_FALLBACKS: dict[str, tuple[str, ...]] = {
    "gpu_1x_h100_pcie": ("gpu_1x_h100_sxm5", "gpu_1x_h100"),
    "gpu_1x_h100": ("gpu_1x_h100_pcie", "gpu_1x_h100_sxm5"),
    "gpu_1x_h100_sxm5": ("gpu_1x_h100_pcie", "gpu_1x_h100"),
}

_CAPACITY_LOCK = threading.Lock()
_CAPACITY_CACHE: tuple[float, dict[str, list[str]]] | None = None
CAPACITY_TTL_SECONDS = 20.0
LAUNCH_WAIT_SECONDS = 1800

INSTANCE_TYPE_RE = re.compile(r"^gpu_1x_[a-z0-9_]+$")
IPV4_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
GHA_NAME_RE = re.compile(rf"^{re.escape(INSTANCE_NAME_PREFIX)}-t(\d+)-([^-]+)-")

# Cron/janitor backstop. The bench process cap is BENCH_MAX_SECONDS (3h);
# this is slightly higher so a healthy run is not reaped mid-job.
DEFAULT_MAX_AGE_SECONDS = 4 * 60 * 60

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SESSION_DIR = REPO_ROOT / "ci" / "benchmark" / ".session"
SESSION_KEY = SESSION_DIR / "id_ed25519"
SESSION_FILE = SESSION_DIR / "session.json"


class LambdaError(RuntimeError):
    """Lambda Cloud API error."""


def load_dotenv() -> None:
    """Load KEY=value pairs from the repo-root .env without overriding the process env."""
    path = REPO_ROOT / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        value = value.strip().strip("'").strip('"')
        if name and name not in os.environ:
            os.environ[name] = value


def api_key() -> str:
    """Read LAMBDA_API_KEY from the environment or repo-root .env."""
    load_dotenv()
    key = os.environ.get("LAMBDA_API_KEY", "").strip()
    if not key:
        raise LambdaError("LAMBDA_API_KEY is not set (export it or put it in .env)")
    return key


def _retry_after_seconds(detail: str, attempt: int) -> float:
    """Cloudflare retry-after if present, otherwise exponential backoff."""
    try:
        payload = json.loads(detail)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        raw = payload.get("retry_after")
        if isinstance(raw, (int, float)) and raw > 0:
            return float(raw)
    return float(min(30 * (2**attempt), 240))


def request(method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    """Call the Lambda Cloud API and return parsed JSON."""
    payload = None if body is None else json.dumps(body).encode()
    headers = {
        "Accept": "application/json",
        "User-Agent": "kryo-benchmark/0.3",
    }
    if payload is not None:
        headers["Content-Type"] = "application/json"
    token = base64.b64encode(f"{api_key()}:".encode()).decode()
    headers["Authorization"] = f"Basic {token}"
    last_error: LambdaError | None = None
    for attempt in range(7):
        req = urllib.request.Request(
            f"{API_BASE}{path}",
            data=payload,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                raw = response.read().decode()
        except urllib.error.HTTPError as error:
            detail = error.read().decode() if error.fp else ""
            last_error = LambdaError(f"{method} {path} failed ({error.code}): {detail}")
            if error.code == 429 and attempt < 6:
                wait = _retry_after_seconds(detail, attempt)
                print(f"Lambda API rate limited; retrying in {int(wait)}s")
                time.sleep(wait)
                continue
            raise last_error from error
        return json.loads(raw) if raw else {}
    if last_error is None:
        raise LambdaError(f"{method} {path} failed after retries")
    raise last_error


def list_capacity() -> dict[str, list[str]]:
    """Map instance type name to region names that currently have capacity."""
    global _CAPACITY_CACHE
    now = time.monotonic()
    with _CAPACITY_LOCK:
        cached = _CAPACITY_CACHE
        if cached is not None and now - cached[0] < CAPACITY_TTL_SECONDS:
            return cached[1]
    data = request("GET", "/instance-types").get("data", {})
    available: dict[str, list[str]] = {}
    if not isinstance(data, dict):
        return available
    for name, info in data.items():
        if not isinstance(name, str) or not isinstance(info, dict):
            continue
        regions = info.get("regions_with_capacity_available") or []
        region_names = [
            region["name"]
            for region in regions
            if isinstance(region, dict) and isinstance(region.get("name"), str)
        ]
        if region_names:
            available[name] = region_names
    with _CAPACITY_LOCK:
        _CAPACITY_CACHE = (time.monotonic(), available)
    return available


def choose_instance_type(requested: str) -> tuple[str, str]:
    """Pick an in-stock 1x GPU type and a region."""
    available = list_capacity()
    if requested != "auto":
        if not INSTANCE_TYPE_RE.fullmatch(requested):
            raise LambdaError(f"invalid instance type: {requested}")
        candidates = (requested, *SKU_FALLBACKS.get(requested, ()))
        for name in candidates:
            regions = available.get(name)
            if regions:
                if name != requested:
                    print(f"{requested} has no capacity; using {name}")
                return name, regions[0]
        stock = ", ".join(sorted(available)) or "none"
        raise LambdaError(f"{requested} has no capacity (available: {stock})")

    for name in PREFERRED_INSTANCE_TYPES:
        regions = available.get(name)
        if regions:
            return name, regions[0]

    ones = sorted(name for name in available if INSTANCE_TYPE_RE.fullmatch(name))
    if not ones:
        raise LambdaError("no 1x GPU instance types currently have capacity")
    name = ones[0]
    return name, available[name][0]


def find_ssh_key_id(name: str) -> str | None:
    """Return the Lambda SSH key id for a name, if it exists."""
    data = request("GET", "/ssh-keys").get("data", [])
    if not isinstance(data, list):
        return None
    for item in data:
        if isinstance(item, dict) and item.get("name") == name:
            key_id = item.get("id")
            if isinstance(key_id, str):
                return key_id
            return name
    return None


def add_ssh_key(name: str, public_key: str) -> str:
    """Upload an SSH public key; return the Lambda key id."""
    existing = find_ssh_key_id(name)
    if existing is not None:
        delete_ssh_key(existing)
    data = request("POST", "/ssh-keys", {"name": name, "public_key": public_key}).get("data", {})
    key_id = data.get("id")
    if not isinstance(key_id, str):
        return name
    return key_id


def delete_ssh_key(key_id: str) -> None:
    """Best-effort delete of a Lambda SSH key."""
    try:
        request("DELETE", f"/ssh-keys/{key_id}")
    except LambdaError as error:
        print(f"warning: could not delete SSH key {key_id}: {error}", file=sys.stderr)


def launch_instance(
    instance_type: str,
    region: str,
    ssh_key_name: str,
    name: str,
    file_systems: list[str] | None = None,
) -> str:
    """Launch one instance and return its id."""
    body: dict[str, Any] = {
        "region_name": region,
        "instance_type_name": instance_type,
        "ssh_key_names": [ssh_key_name],
        "quantity": 1,
        "name": name,
    }
    if file_systems:
        body["file_system_names"] = file_systems
    data = request("POST", "/instance-operations/launch", body).get("data", {})
    ids = data.get("instance_ids") or []
    if not ids or not isinstance(ids[0], str):
        raise LambdaError(f"launch did not return an instance id: {data}")
    return ids[0]


def _filesystem_region(item: dict[str, Any]) -> str:
    """Read the region name from a file-systems API record."""
    raw = item.get("region")
    if isinstance(raw, dict):
        nested = raw.get("name")
        if isinstance(nested, str):
            return nested
    if isinstance(raw, str):
        return raw
    name = item.get("region_name")
    if isinstance(name, str):
        return name
    return ""


def list_filesystems() -> list[dict[str, Any]]:
    """List persistent Lambda filesystems."""
    for path in ("/file-systems", "/filesystems"):
        try:
            data = request("GET", path).get("data", [])
        except LambdaError:
            continue
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    return []


def filesystem_in_region(name: str, region: str) -> bool:
    """True if a filesystem with this name already exists in the region."""
    return any(
        item.get("name") == name and _filesystem_region(item) == region
        for item in list_filesystems()
    )


def ensure_filesystem(name: str, region: str) -> str:
    """Create a regional filesystem if needed. Returns the name to attach, or empty."""
    if filesystem_in_region(name, region):
        return name
    regional = f"{name}-{region}"
    if filesystem_in_region(regional, region):
        return regional
    for candidate, body in (
        (name, {"name": name, "region": region}),
        (name, {"name": name, "region_name": region}),
        (regional, {"name": regional, "region": region}),
        (regional, {"name": regional, "region_name": region}),
    ):
        try:
            request("POST", "/file-systems", body)
            print(f"created filesystem {candidate} in {region}")
            return candidate
        except LambdaError:
            try:
                request("POST", "/filesystems", body)
                print(f"created filesystem {candidate} in {region}")
                return candidate
            except LambdaError:
                if filesystem_in_region(candidate, region):
                    return candidate
                continue
    print(f"warning: could not create filesystem {name} in {region}; golden stays ephemeral")
    return ""


def get_instance(instance_id: str) -> dict[str, Any]:
    """Fetch one instance record."""
    data = request("GET", f"/instances/{instance_id}").get("data", {})
    if not isinstance(data, dict):
        raise LambdaError(f"unexpected instance payload: {data}")
    return data


def terminate_instance(instance_id: str) -> None:
    """Terminate a Lambda instance."""
    request("POST", "/instance-operations/terminate", {"instance_ids": [instance_id]})


def wait_for_ip(instance_id: str, timeout: int = 1200) -> str:
    """Poll until the instance is active and has an IPv4 address."""
    print("waiting for Lambda boot (often several minutes)...")
    deadline = time.monotonic() + timeout
    last_line = ""
    while time.monotonic() < deadline:
        instance = get_instance(instance_id)
        status = instance.get("status")
        ip = instance.get("ip")
        line = f"instance {instance_id} status={status} ip={ip}"
        if line != last_line:
            print(line)
            last_line = line
        if status in {"unhealthy", "terminated"}:
            raise LambdaError(f"instance entered {status}")
        if isinstance(ip, str) and IPV4_RE.fullmatch(ip):
            return ip
        time.sleep(10)
    raise LambdaError(f"timed out waiting for instance {instance_id}")


def require_bin(name: str) -> str:
    """Return the absolute path to a required executable."""
    path = shutil.which(name)
    if path is None:
        raise LambdaError(f"{name} is required on PATH")
    return path


def ssh_base(identity: Path, ip: str) -> list[str]:
    """Common ssh identity options plus user@host."""
    if not IPV4_RE.fullmatch(ip):
        raise LambdaError(f"refusing non-ipv4 host: {ip}")
    return [
        "-i",
        str(identity),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=30",
        "-o",
        "ConnectTimeout=10",
        f"{SSH_USER}@{ip}",
    ]


def wait_for_ssh(identity: Path, ip: str, timeout: int = 1200) -> None:
    """Retry SSH until the VM accepts the key."""
    print(f"waiting for ssh on {ip}...")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = subprocess.run(
            [require_bin("ssh"), *ssh_base(identity, ip), "true"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return
        print("waiting for ssh...")
        time.sleep(10)
    raise LambdaError(f"timed out waiting for ssh on {ip}")


def generate_ssh_key(directory: Path) -> tuple[Path, str]:
    """Create an ed25519 key pair, replacing any leftover files."""
    private = directory / "id_ed25519"
    private.unlink(missing_ok=True)
    (directory / "id_ed25519.pub").unlink(missing_ok=True)
    subprocess.run(
        [require_bin("ssh-keygen"), "-t", "ed25519", "-f", str(private), "-N", "", "-q"],
        check=True,
    )
    public = (directory / "id_ed25519.pub").read_text(encoding="utf-8").strip()
    private.chmod(0o600)
    return private, public


def instance_type_name(instance: dict[str, Any]) -> str:
    """Read the instance type from a Lambda instance record."""
    raw = instance.get("instance_type")
    if isinstance(raw, dict):
        nested = raw.get("name")
        if isinstance(nested, str):
            return nested
    if isinstance(raw, str):
        return raw
    name = instance.get("instance_type_name")
    if isinstance(name, str):
        return name
    return "unknown"


def parse_gha_name(name: str) -> tuple[int | None, str | None]:
    """Return (launch epoch, GitHub run id) from a kryo-gha-* VM name."""
    if not name.startswith(f"{INSTANCE_NAME_PREFIX}-"):
        return None, None
    match = GHA_NAME_RE.match(name)
    if not match:
        return None, None
    return int(match.group(1)), match.group(2)


def max_age_seconds() -> int:
    """How long a kryo-gha-* VM may live before the janitor/cron kills it."""
    raw = os.environ.get("LAMBDA_MAX_AGE_SECONDS", str(DEFAULT_MAX_AGE_SECONDS)).strip()
    try:
        return max(60, int(raw))
    except ValueError:
        return DEFAULT_MAX_AGE_SECONDS


def should_reap_instance(
    name: str,
    *,
    keep_run_id: str | None,
    max_age: int | None,
    now: float,
) -> bool:
    """True when this kryo-gha-* VM should be terminated."""
    if not name.startswith(f"{INSTANCE_NAME_PREFIX}-"):
        return False
    epoch, run_id = parse_gha_name(name)
    if keep_run_id and run_id == keep_run_id:
        if max_age is None or epoch is None:
            return epoch is None
        return (now - epoch) >= max_age
    if keep_run_id:
        return True
    if max_age is None:
        return True
    if epoch is None:
        return True
    return (now - epoch) >= max_age


def find_instance(instance_id: str) -> dict[str, Any] | None:
    """Return one instance record, or None if it is gone."""
    data = request("GET", "/instances").get("data", [])
    if not isinstance(data, list):
        return None
    for item in data:
        if isinstance(item, dict) and item.get("id") == instance_id:
            return item
    return None


def find_instance_named(name: str) -> dict[str, Any] | None:
    """Return a live instance with this name, if any."""
    data = request("GET", "/instances").get("data", [])
    if not isinstance(data, list):
        return None
    for item in data:
        if (
            isinstance(item, dict)
            and item.get("name") == name
            and item.get("status") not in {"terminated", "terminating"}
        ):
            return item
    return None


def terminate_leaked(
    *,
    max_age_seconds: int | None = None,
    keep_run_id: str | None = None,
) -> int:
    """Destroy kryo-gha-* instances that match the reap policy. Return count."""
    data = request("GET", "/instances").get("data", [])
    if not isinstance(data, list):
        return 0
    now = time.time()
    killed = 0
    for item in data:
        if not isinstance(item, dict):
            continue
        instance_id = item.get("id")
        name = item.get("name")
        status = item.get("status")
        if not isinstance(instance_id, str) or not isinstance(name, str):
            continue
        if status in {"terminated", "terminating"}:
            continue
        if not should_reap_instance(
            name, keep_run_id=keep_run_id, max_age=max_age_seconds, now=now
        ):
            continue
        print(f"terminating leaked instance {instance_id} ({name})")
        try:
            terminate_instance(instance_id)
            killed += 1
        except LambdaError as error:
            print(f"warning: could not terminate {instance_id}: {error}")
    return killed


def save_session(
    identity: Path,
    instance_id: str,
    ip: str,
    instance_type: str,
    key_id: str,
    ssh_key_name: str,
) -> None:
    """Persist SSH key and instance metadata for a leftover kryo-dev VM."""
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    SESSION_DIR.chmod(0o700)
    if identity.resolve() != SESSION_KEY.resolve():
        shutil.copy2(identity, SESSION_KEY)
    SESSION_KEY.chmod(0o600)
    SESSION_FILE.write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "ip": ip,
                "instance_type": instance_type,
                "key_id": key_id,
                "ssh_key_name": ssh_key_name,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def load_session() -> dict[str, str]:
    """Read a previously saved kryo-dev session."""
    if not SESSION_FILE.is_file() or not SESSION_KEY.is_file():
        raise LambdaError("no saved session; nothing to destroy")
    data = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise LambdaError("invalid session file")
    required = ("instance_id", "ip", "instance_type", "key_id", "ssh_key_name")
    session = {key: data[key] for key in required if isinstance(data.get(key), str)}
    if len(session) != len(required):
        raise LambdaError("invalid session file")
    return session


def recover_session() -> dict[str, str]:
    """Load a saved session, or rebuild one from a live kryo-dev VM."""
    if SESSION_FILE.is_file() and SESSION_KEY.is_file():
        return load_session()
    if not SESSION_KEY.is_file():
        raise LambdaError("no saved session")
    live = find_instance_named(DEV_INSTANCE_NAME)
    if live is None or not isinstance(live.get("id"), str):
        raise LambdaError("no saved session")
    ip = live.get("ip")
    if not isinstance(ip, str) or not IPV4_RE.fullmatch(ip):
        raise LambdaError(f"{DEV_INSTANCE_NAME} has no ipv4 yet")
    key_id = find_ssh_key_id(DEV_INSTANCE_NAME) or DEV_INSTANCE_NAME
    save_session(
        SESSION_KEY,
        live["id"],
        ip,
        instance_type_name(live),
        key_id,
        DEV_INSTANCE_NAME,
    )
    print(f"recovered session for {live['id']}")
    return load_session()


def destroy_dev_session() -> None:
    """Terminate the kept kryo-dev instance and drop its SSH key."""
    try:
        session = recover_session()
    except LambdaError:
        try:
            session = load_session()
        except LambdaError as error:
            print(f"no kryo-dev session: {error}")
            terminate_leaked()
            return
    print(f"terminating {session['instance_id']}")
    try:
        terminate_instance(session["instance_id"])
    except LambdaError as error:
        print(f"warning: terminate failed: {error}", file=sys.stderr)
    delete_ssh_key(session["key_id"])
    SESSION_KEY.unlink(missing_ok=True)
    SESSION_FILE.unlink(missing_ok=True)
    terminate_leaked()
    print("dev instance destroyed")


@dataclass
class _Conn:
    identity: Path
    ip: str
    key_id: str
    ssh_key_name: str
    persist_dir: Path


class LambdaProvider:
    """Rent Lambda 1x GPU VMs and talk to them over SSH."""

    def __init__(self) -> None:
        self._conns: dict[str, _Conn] = {}
        self._seq = 0
        self._lock = threading.Lock()

    def janitor(self) -> None:
        """Drop VMs from other runs (and this run if older than LAMBDA_MAX_AGE_SECONDS)."""
        keep = os.environ.get("GITHUB_RUN_ID", "").strip() or None
        terminate_leaked(keep_run_id=keep, max_age_seconds=max_age_seconds())

    def launch(self, sku: str, filesystem: str | None = None) -> Machine:
        """Create a VM, wait for SSH, and remember the connection."""
        with self._lock:
            self._seq += 1
            seq = self._seq
        run_id = os.environ.get("GITHUB_RUN_ID", str(os.getpid()))
        job = os.environ.get("GITHUB_JOB") or os.environ.get("BENCH_SHARD") or "local"
        job = re.sub(r"[^a-z0-9]+", "", job.lower()) or "local"
        epoch = int(time.time())
        instance_name = f"{INSTANCE_NAME_PREFIX}-t{epoch}-{run_id}-{job}-{seq}"
        ssh_key_name = f"{SSH_KEY_PREFIX}-t{epoch}-{run_id}-{job}-{seq}"
        deadline = time.monotonic() + LAUNCH_WAIT_SECONDS
        while True:
            try:
                instance_type, region = choose_instance_type(sku)
                break
            except LambdaError as error:
                if time.monotonic() >= deadline or "no capacity" not in str(error).lower():
                    raise
                print(f"{error}; waiting 30s for capacity")
                time.sleep(30)
        attached = ""
        if filesystem:
            attached = ensure_filesystem(filesystem, region)
        print(f"using {instance_type} in {region}" + (f" fs={attached}" if attached else ""))
        persist_dir = Path(tempfile.mkdtemp(prefix="kryo-bench-"))
        identity, public_key = generate_ssh_key(persist_dir)
        key_id = add_ssh_key(ssh_key_name, public_key)
        try:
            instance_id = launch_instance(
                instance_type,
                region,
                ssh_key_name,
                instance_name,
                file_systems=[attached] if attached else None,
            )
            ip = wait_for_ip(instance_id)
            wait_for_ssh(identity, ip)
        except Exception:
            delete_ssh_key(key_id)
            shutil.rmtree(persist_dir, ignore_errors=True)
            raise
        machine = Machine(
            id=instance_id,
            sku=instance_type,
            name=instance_name,
            region=region,
            filesystem=attached,
            requested=sku,
        )
        with self._lock:
            self._conns[machine.id] = _Conn(
                identity=identity,
                ip=ip,
                key_id=key_id,
                ssh_key_name=ssh_key_name,
                persist_dir=persist_dir,
            )
        print(f"ready {machine.id} {ip} ({instance_type})")
        return machine

    def rsync(self, machine: Machine) -> None:
        """Copy this checkout onto the instance."""
        conn = self._conn(machine)
        ssh = [
            require_bin("ssh"),
            "-i",
            str(conn.identity),
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "UserKnownHostsFile=/dev/null",
        ]
        subprocess.run(
            [require_bin("ssh"), *ssh_base(conn.identity, conn.ip), f"mkdir -p {REPO_REMOTE}"],
            check=True,
        )
        subprocess.run(
            [
                require_bin("rsync"),
                "-az",
                "--delete",
                "-e",
                " ".join(ssh),
                "--exclude",
                ".git",
                "--exclude",
                ".venv",
                "--exclude",
                "target",
                "--exclude",
                "__pycache__",
                "--exclude",
                "ci/benchmark/.session",
                "--exclude",
                "ci/benchmark/.snapshots",
                "--exclude",
                "ci/benchmark/.golden",
                f"{REPO_ROOT}/",
                f"{SSH_USER}@{conn.ip}:{REPO_REMOTE}/",
            ],
            check=True,
        )

    def run(self, machine: Machine, command: str, timeout: int | None = None) -> None:
        """Run a remote shell command, streaming output.

        Commands that may run for 10+ minutes (setup, docker pull, server
        benches) are detached with nohup so a dropped laptop SSH does not
        kill the VM-side process.
        """
        if timeout is not None and timeout >= 600:
            self._run_nohup(machine, command, timeout)
            return
        conn = self._conn(machine)
        subprocess.run(
            [require_bin("ssh"), *ssh_base(conn.identity, conn.ip), command],
            check=True,
            timeout=timeout,
        )

    def _run_nohup(self, machine: Machine, command: str, timeout: int) -> None:
        """Start `command` under nohup, then poll the log until it exits."""
        conn = self._conn(machine)
        stamp = f"{os.getpid()}-{machine.id[:8]}"
        log_path = f"/tmp/kryo-nohup-{stamp}.log"
        exit_path = f"/tmp/kryo-nohup-{stamp}.exit"
        inner = f"({command})\nprintf '%s\\n' $? > {exit_path}"
        start = (
            f"rm -f {shlex.quote(exit_path)}; "
            f"nohup bash -lc {shlex.quote(inner)} "
            f"> {shlex.quote(log_path)} 2>&1 < /dev/null & echo $!"
        )
        print(f"detached {machine.id} -> {log_path}")
        subprocess.run(
            [require_bin("ssh"), *ssh_base(conn.identity, conn.ip), start],
            check=True,
            timeout=60,
        )
        deadline = time.monotonic() + timeout
        poll = (
            f"tail -n 50 {shlex.quote(log_path)} 2>/dev/null || true; "
            f"if [ -f {shlex.quote(exit_path)} ]; then "
            f"echo __KRYO_EXIT__:$(cat {shlex.quote(exit_path)}); fi"
        )
        while time.monotonic() < deadline:
            try:
                result = subprocess.run(
                    [require_bin("ssh"), *ssh_base(conn.identity, conn.ip), poll],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=90,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                print(f"ssh poll lost ({error}); retrying in 15s")
                time.sleep(15)
                continue
            out = (result.stdout or "") + (result.stderr or "")
            if out.strip():
                print(out.rstrip())
            if "__KRYO_EXIT__:" in out:
                code_raw = out.rsplit("__KRYO_EXIT__:", 1)[-1].strip().splitlines()[0]
                try:
                    code = int(code_raw)
                except ValueError:
                    code = 1
                if code != 0:
                    raise subprocess.CalledProcessError(code, command)
                return
            time.sleep(20)
        raise TimeoutError(f"remote command exceeded {timeout}s ({log_path})")

    def run_output(self, machine: Machine, command: str, timeout: int | None = None) -> str:
        """Run a remote shell command and return stdout."""
        conn = self._conn(machine)
        result = subprocess.run(
            [require_bin("ssh"), *ssh_base(conn.identity, conn.ip), command],
            check=True,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        return result.stdout

    def put(self, machine: Machine, local: Path, remote: str) -> None:
        """Copy a local file onto the VM."""
        conn = self._conn(machine)
        subprocess.run(
            [
                require_bin("scp"),
                "-i",
                str(conn.identity),
                "-o",
                "IdentitiesOnly=yes",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                "UserKnownHostsFile=/dev/null",
                str(local),
                f"{SSH_USER}@{conn.ip}:{remote}",
            ],
            check=True,
        )

    def get(self, machine: Machine, remote: str, local: Path) -> None:
        """Copy a remote file onto the controller."""
        conn = self._conn(machine)
        local.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                require_bin("scp"),
                "-i",
                str(conn.identity),
                "-o",
                "IdentitiesOnly=yes",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                "UserKnownHostsFile=/dev/null",
                f"{SSH_USER}@{conn.ip}:{remote}",
                str(local),
            ],
            check=True,
        )

    def terminate(self, machine: Machine) -> None:
        """Destroy the VM, SSH key, and local keydir."""
        with self._lock:
            conn = self._conns.pop(machine.id, None)
        print(f"terminating {machine.id}")
        try:
            terminate_instance(machine.id)
        except LambdaError as error:
            print(f"warning: terminate failed: {error}", file=sys.stderr)
        if conn is not None:
            delete_ssh_key(conn.key_id)
            shutil.rmtree(conn.persist_dir, ignore_errors=True)

    def _conn(self, machine: Machine) -> _Conn:
        with self._lock:
            conn = self._conns.get(machine.id)
        if conn is None:
            raise LambdaError(f"no connection for {machine.id}")
        return conn
