"""Provision a Lambda GPU VM, run Kryo benchmarks, then destroy the VM."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from functools import partial
from pathlib import Path
from typing import Any

print = partial(print, flush=True)

API_BASE = "https://cloud.lambdalabs.com/api/v1"
INSTANCE_NAME_PREFIX = "kryo-gha"
SSH_KEY_PREFIX = "kryo-gha"
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

INSTANCE_TYPE_RE = re.compile(r"^gpu_1x_[a-z0-9_]+$")
IPV4_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
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


def request(method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    """Call the Lambda Cloud API and return parsed JSON."""
    payload = None if body is None else json.dumps(body).encode()
    headers = {
        "Accept": "application/json",
        "User-Agent": "kryo-benchmark/0.2",
    }
    if payload is not None:
        headers["Content-Type"] = "application/json"
    token = base64.b64encode(f"{api_key()}:".encode()).decode()
    headers["Authorization"] = f"Basic {token}"
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
        raise LambdaError(f"{method} {path} failed ({error.code}): {detail}") from error
    return json.loads(raw) if raw else {}


def list_capacity() -> dict[str, list[str]]:
    """Map instance type name to region names that currently have capacity."""
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
    return available


def choose_instance_type(requested: str) -> tuple[str, str]:
    """Pick an in-stock 1x GPU type and a region."""
    available = list_capacity()
    if requested != "auto":
        if not INSTANCE_TYPE_RE.fullmatch(requested):
            raise LambdaError(f"invalid instance type: {requested}")
        regions = available.get(requested)
        if not regions:
            stock = ", ".join(sorted(available)) or "none"
            raise LambdaError(f"{requested} has no capacity (available: {stock})")
        return requested, regions[0]

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


def launch_instance(instance_type: str, region: str, ssh_key_name: str, name: str) -> str:
    """Launch one instance and return its id."""
    data = request(
        "POST",
        "/instance-operations/launch",
        {
            "region_name": region,
            "instance_type_name": instance_type,
            "ssh_key_names": [ssh_key_name],
            "quantity": 1,
            "name": name,
        },
    ).get("data", {})
    ids = data.get("instance_ids") or []
    if not ids or not isinstance(ids[0], str):
        raise LambdaError(f"launch did not return an instance id: {data}")
    return ids[0]


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
    while time.monotonic() < deadline:
        instance = get_instance(instance_id)
        status = instance.get("status")
        ip = instance.get("ip")
        print(f"instance {instance_id} status={status} ip={ip}")
        if status == "active" and isinstance(ip, str) and IPV4_RE.fullmatch(ip):
            return ip
        if status in {"unhealthy", "terminated"}:
            raise LambdaError(f"instance entered {status}")
        time.sleep(10)
    raise LambdaError(f"timed out waiting for instance {instance_id}")


def require_bin(name: str) -> str:
    """Return the absolute path to a required executable."""
    path = shutil.which(name)
    if path is None:
        raise LambdaError(f"{name} is required on PATH")
    return path


def ssh_base(identity: Path, ip: str) -> list[str]:
    """Common ssh/scp/rsync identity options."""
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


def rsync_repo(identity: Path, ip: str) -> None:
    """Copy this checkout onto the instance."""
    ssh = [
        require_bin("ssh"),
        "-i",
        str(identity),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "UserKnownHostsFile=/dev/null",
    ]
    subprocess.run(
        [require_bin("ssh"), *ssh_base(identity, ip), f"mkdir -p {REPO_REMOTE}"],
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
            f"{REPO_ROOT}/",
            f"{SSH_USER}@{ip}:{REPO_REMOTE}/",
        ],
        check=True,
    )


def ssh_run(identity: Path, ip: str, remote: str, timeout: int | None = None) -> None:
    """Run a remote shell command, streaming output."""
    subprocess.run(
        [require_bin("ssh"), *ssh_base(identity, ip), remote],
        check=True,
        timeout=timeout,
    )


def scp_from(identity: Path, ip: str, remote_path: str, local_path: Path) -> None:
    """Copy a file from the instance."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            require_bin("scp"),
            "-i",
            str(identity),
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "UserKnownHostsFile=/dev/null",
            f"{SSH_USER}@{ip}:{remote_path}",
            str(local_path),
        ],
        check=True,
    )


def terminate_leaked() -> None:
    """Destroy leftover CI instances from previous crashed runs."""
    data = request("GET", "/instances").get("data", [])
    if not isinstance(data, list):
        return
    leaked = [
        item["id"]
        for item in data
        if isinstance(item, dict)
        and isinstance(item.get("id"), str)
        and isinstance(item.get("name"), str)
        and item["name"].startswith(f"{INSTANCE_NAME_PREFIX}-")
    ]
    for instance_id in leaked:
        print(f"terminating leaked instance {instance_id}")
        try:
            terminate_instance(instance_id)
        except LambdaError as error:
            print(f"warning: could not terminate {instance_id}: {error}")


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


def save_session(
    identity: Path,
    instance_id: str,
    ip: str,
    instance_type: str,
    key_id: str,
    ssh_key_name: str,
) -> None:
    """Persist SSH key and instance metadata for --reuse."""
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


def recover_session() -> dict[str, str]:
    """Load a saved session, or rebuild one from a live kryo-dev VM."""
    if SESSION_FILE.is_file() and SESSION_KEY.is_file():
        return load_session()
    if not SESSION_KEY.is_file():
        raise LambdaError("no saved session; launch with --keep first")
    live = find_instance_named(DEV_INSTANCE_NAME)
    if live is None or not isinstance(live.get("id"), str):
        raise LambdaError("no saved session; launch with --keep first")
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


def load_session() -> dict[str, str]:
    """Read a previously saved --keep session."""
    if not SESSION_FILE.is_file() or not SESSION_KEY.is_file():
        raise LambdaError("no saved session; launch with --keep first")
    data = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise LambdaError("invalid session file")
    required = ("instance_id", "ip", "instance_type", "key_id", "ssh_key_name")
    session = {key: data[key] for key in required if isinstance(data.get(key), str)}
    if len(session) != len(required):
        raise LambdaError("invalid session file")
    return session


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


def destroy_dev_session() -> None:
    """Terminate the kept dev instance and drop its SSH key."""
    try:
        session = recover_session()
    except LambdaError:
        session = load_session()
    print(f"terminating {session['instance_id']}")
    try:
        terminate_instance(session["instance_id"])
    except LambdaError as error:
        print(f"warning: terminate failed: {error}", file=sys.stderr)
    delete_ssh_key(session["key_id"])
    SESSION_KEY.unlink(missing_ok=True)
    SESSION_FILE.unlink(missing_ok=True)
    print("dev instance destroyed")


def git_sha() -> str:
    """Commit being benched: explicit env, else this checkout, else GITHUB_SHA."""
    override = os.environ.get("BENCH_GIT_SHA", "").strip()
    if override:
        return override
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        sha = ""
    return sha or os.environ.get("GITHUB_SHA", "").strip()


def run_benchmark(
    runs: int,
    gpu: str,
    output: Path,
    scenarios: list[str] | None = None,
    timeout: int = 90,
    keep: bool = False,
    reuse: bool = False,
    skip_setup: bool = False,
    setup_only: bool = False,
) -> None:
    """Launch or reuse a GPU VM, run benches, optionally leave the VM up."""
    run_id = os.environ.get("GITHUB_RUN_ID", str(os.getpid()))
    instance_name = DEV_INSTANCE_NAME if keep or reuse else f"{INSTANCE_NAME_PREFIX}-{run_id}"
    ssh_key_name = DEV_INSTANCE_NAME if keep or reuse else f"{SSH_KEY_PREFIX}-{run_id}"

    if not reuse:
        terminate_leaked()
        if keep:
            live = find_instance_named(DEV_INSTANCE_NAME)
            if live is not None:
                raise LambdaError(
                    f"{DEV_INSTANCE_NAME} is already running ({live.get('id')}); "
                    "use --reuse or --destroy"
                )
            if SESSION_FILE.is_file() and SESSION_KEY.is_file():
                session = load_session()
                leftover = find_instance(session["instance_id"])
                if leftover is not None:
                    raise LambdaError(
                        f"saved instance {session['instance_id']} is still up; "
                        "use --reuse or --destroy"
                    )

    instance_id: str | None = None
    key_id: str | None = None
    ip = ""
    instance_type = gpu
    persist_dir = SESSION_DIR if keep or reuse else Path(tempfile.mkdtemp(prefix="kryo-bench-"))

    try:
        if reuse:
            session = recover_session()
            instance_id = session["instance_id"]
            key_id = session["key_id"]
            instance_type = session["instance_type"]
            identity = SESSION_KEY
            live = find_instance(instance_id)
            if live is None:
                raise LambdaError(f"saved instance {instance_id} is gone")
            status = live.get("status")
            if status not in {"active", "booting"}:
                raise LambdaError(f"instance {instance_id} is {status}")
            print(f"reusing {instance_id} ({instance_type})")
            ip = wait_for_ip(instance_id)
            wait_for_ssh(identity, ip)
        else:
            instance_type, region = choose_instance_type(gpu)
            print(f"using {instance_type} in {region}")
            persist_dir.mkdir(parents=True, exist_ok=True)
            identity, public_key = generate_ssh_key(persist_dir)
            key_id = add_ssh_key(ssh_key_name, public_key)
            instance_id = launch_instance(instance_type, region, ssh_key_name, instance_name)
            ip = wait_for_ip(instance_id)
            wait_for_ssh(identity, ip)
            if keep and key_id is not None:
                save_session(identity, instance_id, ip, instance_type, key_id, ssh_key_name)

        rsync_repo(identity, ip)
        if not skip_setup:
            ssh_run(
                identity,
                ip,
                f"bash {REPO_REMOTE}/ci/benchmark/setup.sh {REPO_REMOTE}",
                timeout=3600,
            )
        else:
            ssh_run(
                identity,
                ip,
                f"source $HOME/.cargo/env && cd {REPO_REMOTE} && "
                "cargo build --release --locked && "
                "sudo install -m 755 target/release/kryo /usr/local/bin/kryo",
                timeout=600,
            )
        extra_llms = [
            name for name in (scenarios or []) if name in {"qwen7", "qwen32", "vllm_engine", "torch_compile"}
        ]
        if extra_llms:
            flags = " ".join(f"--scenario {shlex.quote(name)}" for name in extra_llms)
            ssh_run(
                identity,
                ip,
                f"cd {REPO_REMOTE}/benchmarks && .venv/bin/python download_models.py {flags}",
                timeout=7200,
            )
        if scenarios and "vllm_engine" in scenarios:
            ssh_run(
                identity,
                ip,
                f"bash {REPO_REMOTE}/ci/benchmark/install_vllm.sh {REPO_REMOTE}",
                timeout=1800,
            )
        if setup_only:
            print("setup only; skipping runner")
        else:
            sha = git_sha()
            tag = os.environ.get("BENCH_RELEASE_TAG", "").strip()
            remote_env = (
                "export PATH=/usr/local/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH; "
                f"export BENCH_INSTANCE_TYPE={shlex.quote(instance_type)}; "
            )
            if sha:
                remote_env += f"export BENCH_GIT_SHA={shlex.quote(sha)}; "
            if tag:
                remote_env += f"export BENCH_RELEASE_TAG={shlex.quote(tag)}; "
            if scenarios:
                select = f"--scenarios {shlex.quote(','.join(scenarios))}"
            else:
                select = "--all"
            ssh_run(
                identity,
                ip,
                remote_env + f"cd {REPO_REMOTE}/benchmarks && .venv/bin/python runner.py "
                f"{select} --runs {int(runs)} --timeout {int(timeout)} "
                "--output results/latest.json",
                timeout=10800,
            )
            scp_from(identity, ip, f"{REPO_REMOTE}/benchmarks/results/latest.json", output)
            print(f"results copied to {output}")
        if keep and instance_id is not None:
            print(f"leaving {instance_id} ({ip}) running as {instance_name}")
            print("reuse: doppler run -- python3 -u ci/benchmark/run.py --reuse --skip-setup ...")
    finally:
        if keep or reuse:
            pass
        else:
            if instance_id is not None:
                print(f"terminating {instance_id}")
                try:
                    terminate_instance(instance_id)
                except LambdaError as error:
                    print(f"warning: terminate failed: {error}", file=sys.stderr)
            if key_id is not None:
                delete_ssh_key(key_id)
            if persist_dir != SESSION_DIR:
                shutil.rmtree(persist_dir, ignore_errors=True)


def main() -> None:
    """CLI for the GitHub Actions GPU benchmark job."""
    parser = argparse.ArgumentParser(description="Run Kryo benchmarks on Lambda Cloud")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument(
        "--gpu",
        default="auto",
        help="Lambda instance type (gpu_1x_a10, gpu_1x_h100, ...) or auto",
    )
    parser.add_argument(
        "--scenarios",
        default="",
        help="Comma-separated scenarios (default: release --all set)",
    )
    parser.add_argument("--timeout", type=int, default=90, help="Seconds per timed command")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "benchmarks" / "results" / "latest.json"),
    )
    parser.add_argument("--keep", action="store_true", help="Leave the VM running after the job")
    parser.add_argument("--reuse", action="store_true", help="Attach to a VM saved by --keep")
    parser.add_argument("--skip-setup", action="store_true", help="Skip bootstrap on --reuse")
    parser.add_argument("--setup-only", action="store_true", help="Provision/sync but do not bench")
    parser.add_argument("--destroy", action="store_true", help="Terminate the kept dev VM")
    args = parser.parse_args()
    if args.destroy:
        destroy_dev_session()
        return
    if args.reuse and args.keep:
        parser.error("use --reuse or --keep, not both")
    if args.skip_setup and not args.reuse:
        parser.error("--skip-setup only applies with --reuse")
    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if args.timeout < 1:
        parser.error("--timeout must be >= 1")
    scenarios = [item.strip() for item in args.scenarios.split(",") if item.strip()] or None
    run_benchmark(
        args.runs,
        args.gpu,
        Path(args.output),
        scenarios,
        args.timeout,
        keep=args.keep,
        reuse=args.reuse,
        skip_setup=args.skip_setup,
        setup_only=args.setup_only,
    )


if __name__ == "__main__":
    main()
