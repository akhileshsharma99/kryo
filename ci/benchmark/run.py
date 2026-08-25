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


class LambdaError(RuntimeError):
    """Lambda Cloud API error."""


def api_key() -> str:
    """Read LAMBDA_API_KEY from the environment."""
    key = os.environ.get("LAMBDA_API_KEY", "").strip()
    if not key:
        raise LambdaError("LAMBDA_API_KEY is not set")
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


def add_ssh_key(name: str, public_key: str) -> str:
    """Upload an SSH public key; return the Lambda key id."""
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
    """Create an ephemeral ed25519 key pair."""
    private = directory / "id_ed25519"
    subprocess.run(
        [require_bin("ssh-keygen"), "-t", "ed25519", "-f", str(private), "-N", "", "-q"],
        check=True,
    )
    public = (directory / "id_ed25519.pub").read_text(encoding="utf-8").strip()
    private.chmod(0o600)
    return private, public


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


def run_benchmark(runs: int, gpu: str, output: Path) -> None:
    """Launch, bench, copy results, always terminate."""
    run_id = os.environ.get("GITHUB_RUN_ID", str(os.getpid()))
    instance_name = f"{INSTANCE_NAME_PREFIX}-{run_id}"
    ssh_key_name = f"{SSH_KEY_PREFIX}-{run_id}"

    terminate_leaked()
    instance_type, region = choose_instance_type(gpu)
    print(f"using {instance_type} in {region}")

    instance_id: str | None = None
    key_id: str | None = None
    with tempfile.TemporaryDirectory() as tmp:
        identity, public_key = generate_ssh_key(Path(tmp))
        try:
            key_id = add_ssh_key(ssh_key_name, public_key)
            instance_id = launch_instance(instance_type, region, ssh_key_name, instance_name)
            ip = wait_for_ip(instance_id)
            wait_for_ssh(identity, ip)
            rsync_repo(identity, ip)
            ssh_run(
                identity,
                ip,
                f"bash {REPO_REMOTE}/ci/benchmark/setup.sh {REPO_REMOTE}",
                timeout=3600,
            )
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
            ssh_run(
                identity,
                ip,
                remote_env + f"cd {REPO_REMOTE}/benchmarks && .venv/bin/python runner.py "
                f"--all --runs {int(runs)} --timeout 90 --output results/latest.json",
                timeout=3600,
            )
            scp_from(identity, ip, f"{REPO_REMOTE}/benchmarks/results/latest.json", output)
            print(f"results copied to {output}")
        finally:
            if instance_id is not None:
                print(f"terminating {instance_id}")
                try:
                    terminate_instance(instance_id)
                except LambdaError as error:
                    print(f"warning: terminate failed: {error}", file=sys.stderr)
            if key_id is not None:
                delete_ssh_key(key_id)


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
        "--output",
        default=str(REPO_ROOT / "benchmarks" / "results" / "latest.json"),
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be >= 1")
    run_benchmark(args.runs, args.gpu, Path(args.output))


if __name__ == "__main__":
    main()
