"""Time-to-first-token for vLLM and Triton servers: cold start vs Kryo restore.

Runs on the GPU VM. Official docker images are unpacked to a stable rootfs so
host CRIU can dump them (restore inside Docker is not supported).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = Path.home() / "kryo"
ROOTFS_BASE = Path("/opt/kryo-images")
SNAP_ROOT = Path("/var/lib/kryo-bench/criu")
HF_CACHE = Path.home() / ".cache" / "huggingface"
RESULTS = Path("/tmp/kryo-server-bench.json")

VLLM_IMAGES = [
    "vllm/vllm-openai:v0.10.2",
    "vllm/vllm-openai:v0.9.2",
    "vllm/vllm-openai:latest",
]
TRITON_IMAGES = [
    "nvcr.io/nvidia/tritonserver:25.12-vllm-python-py3",
    "nvcr.io/nvidia/tritonserver:25.05-vllm-python-py3",
    "nvcr.io/nvidia/tritonserver:24.12-vllm-python-py3",
]

MODELS = {
    "7b": "Qwen/Qwen2.5-7B",
    "32b": "Qwen/Qwen2.5-32B",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def sudo() -> list[str]:
    if os.geteuid() == 0:
        return []
    return ["sudo", "-n"]


def run(
    cmd: list[str],
    *,
    timeout: int | None = None,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    log("+ " + " ".join(cmd[:12]) + (" ..." if len(cmd) > 12 else ""))
    return subprocess.run(
        cmd,
        check=check,
        timeout=timeout,
        text=True,
        env=env,
    )


def drop_page_cache() -> None:
    subprocess.run([*sudo(), "sync"], check=True)
    result = subprocess.run(
        [*sudo(), "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"drop_caches failed: {(result.stderr or result.stdout or '').strip()}")


def kryo_cmd() -> list[str]:
    kryo = shutil.which("kryo")
    if kryo is None:
        raise FileNotFoundError("kryo not on PATH")
    extra = [f"KRYO_SNAPSHOTS_DIR={SNAP_ROOT}"]
    lazy = os.environ.get("KRYO_LAZY_PAGES", "").strip()
    if lazy:
        extra.append(f"KRYO_LAZY_PAGES={lazy}")
    if os.geteuid() == 0:
        if extra:
            return ["env", *extra, kryo]
        return [kryo]
    path = os.environ.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin")
    return ["sudo", "-n", "-E", "env", f"PATH={path}", *extra, kryo]


def kill_tree(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    subprocess.run([*sudo(), "kill", "-9", str(pid)], check=False, capture_output=True)


def kill_port(port: int) -> None:
    subprocess.run(
        [*sudo(), "sh", "-c", f"fuser -k {port}/tcp >/dev/null 2>&1 || true"],
        check=False,
    )


def wait_http(url: str, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= response.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
            last = str(error)
        time.sleep(0.25)
    raise TimeoutError(f"timed out waiting for {url}: {last}")


def post_json(url: str, payload: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read().decode()
    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        parsed = {"raw": raw}
    if not isinstance(parsed, dict):
        return {"raw": parsed}
    return parsed


def which_in_rootfs(rootfs: Path, names: list[str]) -> str | None:
    rels: list[str] = []
    for name in names:
        rels.extend(
            [
                name.lstrip("/"),
                f"usr/bin/{name}",
                f"usr/local/bin/{name}",
                f"opt/venv/bin/{name}",
                f"opt/tritonserver/bin/{name}",
            ]
        )
    seen: set[str] = set()
    for rel in rels:
        if rel in seen:
            continue
        seen.add(rel)
        path = rootfs / rel
        probe = subprocess.run(
            [*sudo(), "test", "-x", str(path)],
            check=False,
            capture_output=True,
        )
        if probe.returncode == 0:
            return "/" + rel
    log(f"no executable python/binary in {rootfs} among {sorted(seen)[:12]}")
    return None


def mount_rootfs(rootfs: Path, extra_binds: list[tuple[str, str]]) -> None:
    binds = [
        ("/proc", "proc"),
        ("/sys", "sys"),
        ("/dev", "dev"),
        ("/dev/shm", "dev/shm"),
        ("/run", "run"),
        (str(HF_CACHE), "root/.cache/huggingface"),
        ("/tmp", "tmp"),
    ]
    binds.extend(extra_binds)
    for src, rel in binds:
        dest = rootfs / rel
        dest.mkdir(parents=True, exist_ok=True)
        if src and Path(src).exists():
            subprocess.run(
                [*sudo(), "mount", "--bind", src, str(dest)],
                check=False,
                capture_output=True,
            )


def unmount_rootfs(rootfs: Path) -> None:
    for rel in (
        "root/.cache/huggingface",
        "tmp",
        "run",
        "dev/shm",
        "dev",
        "sys",
        "proc",
        "scripts",
        "models",
    ):
        subprocess.run(
            [*sudo(), "umount", "-l", str(rootfs / rel)],
            check=False,
            capture_output=True,
        )


def chroot_env(model: str, port: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PATH": "/opt/venv/bin:/opt/tritonserver/bin:/usr/local/bin:/usr/bin:/bin",
            "HOME": "/root",
            "HF_HOME": "/root/.cache/huggingface",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HUGGINGFACE_HUB_CACHE": "/root/.cache/huggingface/hub",
            "BENCH_MODEL": model,
            "BENCH_PORT": str(port),
            "NVIDIA_VISIBLE_DEVICES": "all",
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )
    return env


def pull_first(images: list[str], dest: Path) -> str:
    setup = HERE / "setup_rootfs.sh"
    subprocess.run(["chmod", "+x", str(HERE / "setup_rootfs.sh")], check=False)
    last = ""
    for image in images:
        try:
            run(["bash", str(setup), image, str(dest)], timeout=3600)
            return image
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            last = str(error)
            log(f"image failed {image}: {error}")
    raise RuntimeError(f"could not pull any image: {last}")


def write_triton_repo(repo: Path, model: str) -> None:
    model_dir = repo / "qwen" / "1"
    model_dir.mkdir(parents=True, exist_ok=True)
    (repo / "qwen" / "config.pbtxt").write_text(
        """name: "qwen"
backend: "vllm"
max_batch_size: 0

model_transaction_policy {
  decoupled: True
}

input [
  {
    name: "text_input"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }
]

output [
  {
    name: "text_output"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_MODEL
  }
]
""",
        encoding="utf-8",
    )
    (model_dir / "model.json").write_text(
        json.dumps(
            {
                "model": model,
                "disable_log_stats": True,
                "gpu_memory_utilization": 0.90,
                "dtype": "bfloat16",
                "max_model_len": 512,
                "max_num_seqs": 4,
                "enforce_eager": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def env_prefix(model: str, port: int, gpu: str) -> list[str]:
    gpu_util = "0.85" if "32B" in model else "0.90"
    max_len = "256" if "32B" in model else "512"
    return [
        "env",
        f"BENCH_MODEL={model}",
        f"BENCH_PORT={str(port)}",
        f"BENCH_GPU_UTIL={gpu_util}",
        f"BENCH_MAX_MODEL_LEN={max_len}",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "VLLM_ENABLE_V1_MULTIPROCESSING=0",
        "VLLM_NO_USAGE_STATS=1",
        "HOME=/root",
        "HF_HOME=/root/.cache/huggingface",
        "NVIDIA_VISIBLE_DEVICES=all",
        f"CUDA_VISIBLE_DEVICES={gpu}",
        "PATH=/opt/venv/bin:/opt/tritonserver/bin:/usr/local/bin:/usr/bin:/bin",
    ]


def vllm_command(rootfs: Path, model: str, port: int, gpu: str) -> list[str]:
    python = which_in_rootfs(
        rootfs,
        [
            "python3.12",
            "python3.10",
            "python3",
            "opt/venv/bin/python3",
            "usr/bin/python3.12",
            "usr/bin/python3.10",
            "usr/bin/python3",
        ],
    ) or "/usr/bin/python3.12"
    gpu_util = "0.85" if "32B" in model else "0.90"
    max_len = "256" if "32B" in model else "512"
    return [
        "chroot",
        str(rootfs),
        *env_prefix(model, port, gpu),
        python,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--gpu-memory-utilization",
        gpu_util,
        "--max-model-len",
        max_len,
        "--dtype",
        "bfloat16",
        "--disable-log-stats",
    ]


def triton_command(rootfs: Path, model: str, port: int, gpu: str, size: str) -> list[str]:
    repo = Path(f"/tmp/triton-repos/{size}")
    write_triton_repo(repo, model)
    models = rootfs / "models"
    subprocess.run([*sudo(), "mkdir", "-p", str(models)], check=True)
    subprocess.run([*sudo(), "mount", "--bind", "/tmp/triton-repos", str(models)], check=False)
    binary = which_in_rootfs(rootfs, ["tritonserver"]) or "/opt/tritonserver/bin/tritonserver"
    return [
        "chroot",
        str(rootfs),
        *env_prefix(model, port, gpu),
        binary,
        f"--model-repository=/models/{size}",
        f"--http-port={port}",
        "--grpc-port=0",
        "--metrics-port=0",
        "--log-verbose=0",
        "--model-control-mode=none",
    ]


def snapshot_name(server: str, size: str) -> str:
    return f"server-{server}-{size}"


def start_process(cmd: list[str], env: dict[str, str] | None = None) -> subprocess.Popen[str]:
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def drain(proc: subprocess.Popen[str], limit: int = 4000) -> str:
    if proc.stdout is None:
        return ""
    try:
        return proc.stdout.read()[-limit:]
    except OSError:
        return ""


def first_token(
    *,
    cmd: list[str],
    health: str,
    generate: str,
    payloads: list[dict[str, Any]],
    timeout: int,
    drop_caches: bool,
    env: dict[str, str] | None = None,
    port: int = 8000,
) -> tuple[float, dict[str, Any]]:
    kill_port(port)
    if drop_caches:
        drop_page_cache()
    started = time.perf_counter()
    proc = start_process([*sudo(), *cmd] if os.geteuid() != 0 else cmd, env=env)
    try:
        wait_http(health, timeout)
        last_error: Exception | None = None
        body: dict[str, Any] = {}
        for payload in payloads:
            try:
                body = post_json(generate, payload, timeout=min(180, timeout))
                last_error = None
                break
            except Exception as error:  # noqa: BLE001
                last_error = error
        if last_error is not None:
            raise last_error
        elapsed = time.perf_counter() - started
        return elapsed, body
    finally:
        kill_tree(proc.pid)
        kill_port(port)
        leftover = ""
        try:
            leftover, _ = proc.communicate(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            leftover = drain(proc)
        if leftover:
            log(leftover[-2000:])


def create_snapshot(
    name: str,
    cmd: list[str],
    *,
    wait: int | None,
    timeout: int,
    env: dict[str, str] | None = None,
    port: int | None = None,
) -> None:
    subprocess.run(
        [*kryo_cmd(), "snapshot", "delete", name],
        check=False,
        capture_output=True,
    )
    SNAP_ROOT.mkdir(parents=True, exist_ok=True)
    subprocess.run([*sudo(), "mkdir", "-p", str(SNAP_ROOT)], check=True)
    args = [*kryo_cmd(), "snapshot", "create", "--name", name]
    if wait is not None:
        args.extend(["--wait", str(wait)])
    args.extend(["--", *cmd])
    run(args, timeout=timeout, env=env)
    if port is not None:
        kill_port(port)


def kryo_run_cmd(name: str) -> list[str]:
    prefix = kryo_cmd()
    unshare = shutil.which("unshare")
    args = ["run", "--snapshot", name]
    if unshare is None:
        return [*prefix, *args]
    return [
        *prefix[:-1],
        unshare,
        "--fork",
        "--pid",
        "--mount",
        "--mount-proc",
        "--",
        prefix[-1],
        *args,
    ]


def snapshot_bytes(name: str) -> int | None:
    images = SNAP_ROOT / name / "images"
    du = subprocess.run(
        [*sudo(), "du", "-sb", str(images)],
        capture_output=True,
        text=True,
        check=False,
    )
    if du.returncode != 0 or not du.stdout.strip():
        return None
    try:
        return int(du.stdout.split()[0])
    except ValueError:
        return None


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def bench_server(
    *,
    server: str,
    size: str,
    model: str,
    rootfs: Path,
    samples: int,
    timeout: int,
    port: int,
    gpu: str,
) -> dict[str, Any]:
    kill_port(port)
    extra_binds: list[tuple[str, str]] = []
    if server == "vllm":
        cmd = vllm_command(rootfs, model, port, gpu)
        health = f"http://127.0.0.1:{port}/v1/models"
        generate = f"http://127.0.0.1:{port}/v1/completions"
        payloads: list[dict[str, Any]] = [
            {
                "model": model,
                "prompt": "Hello, world!",
                "max_tokens": 1,
                "temperature": 0.0,
            }
        ]
    else:
        extra_binds.append(("/tmp/triton-repos", "models"))
        write_triton_repo(Path(f"/tmp/triton-repos/{size}"), model)
        cmd = triton_command(rootfs, model, port, gpu, size)
        health = f"http://127.0.0.1:{port}/v2/health/ready"
        generate = f"http://127.0.0.1:{port}/v2/models/qwen/generate"
        payloads = [
            {"text_input": "Hello, world!", "max_tokens": 1, "stream": False},
            {
                "text_input": "Hello, world!",
                "parameters": {"max_tokens": 1, "stream": False},
            },
        ]

    mount_rootfs(rootfs, extra_binds)
    name = snapshot_name(server, size)
    result: dict[str, Any] = {"server": server, "model": model, "size": size}

    try:
        log(f"\n=== {server} {size} cold ===")
        colds: list[float] = []
        for i in range(samples):
            log(f"  cold {i + 1}/{samples}")
            elapsed, body = first_token(
                cmd=cmd,
                health=health,
                generate=generate,
                payloads=payloads,
                timeout=timeout,
                drop_caches=True,
                port=port,
            )
            log(f"    {elapsed:.3f}s  {body}")
            colds.append(elapsed)
        result["cold"] = {
            "runs": len(colds),
            "samples": colds,
            "total": {"mean": mean(colds)},
        }
        wait_seconds = max(int(mean(colds)) + 45, 90)

        log(f"=== {server} {size} snapshot create (wait={wait_seconds}) ===")
        create_snapshot(name, cmd, wait=wait_seconds, timeout=timeout * 2, port=port)
        image_bytes = snapshot_bytes(name)
        if image_bytes is not None:
            result["snapshot_bytes"] = image_bytes
            log(f"  snapshot {image_bytes / 1024**3:.2f} GiB")

        log(f"=== {server} {size} kryo restore ===")
        restores: list[float] = []
        restore_cmd = kryo_run_cmd(name)
        for i in range(samples):
            log(f"  restore {i + 1}/{samples}")
            kill_port(port)
            elapsed, body = first_token(
                cmd=restore_cmd,
                health=health,
                generate=generate,
                payloads=payloads,
                timeout=timeout,
                drop_caches=True,
                port=port,
            )
            log(f"    {elapsed:.3f}s  {body}")
            restores.append(elapsed)
        result["kryo"] = {
            "runs": len(restores),
            "samples": restores,
            "total": {"mean": mean(restores)},
        }
        cold_mean = result["cold"]["total"]["mean"]
        kryo_mean = result["kryo"]["total"]["mean"]
        if kryo_mean > 0:
            result["speedup"] = cold_mean / kryo_mean
    except Exception as error:
        result["error"] = str(error)
        log(f"  FAILED {error}")
    finally:
        kill_port(port)
        subprocess.run(
            [*kryo_cmd(), "snapshot", "delete", name],
            check=False,
            capture_output=True,
        )
        unmount_rootfs(rootfs)
    return result


def ensure_docker_images(servers: list[str]) -> dict[str, str]:
    chosen: dict[str, str] = {}
    errors: dict[str, str] = {}

    def pull_one(server: str) -> None:
        try:
            if server == "vllm":
                chosen["vllm"] = pull_first(VLLM_IMAGES, ROOTFS_BASE / "vllm")
            else:
                chosen["triton"] = pull_first(TRITON_IMAGES, ROOTFS_BASE / "triton")
            log(f"{server} image {chosen[server]}")
        except Exception as error:  # noqa: BLE001
            errors[server] = str(error)
            log(f"{server} image failed: {error}")

    threads = [
        threading.Thread(target=pull_one, args=(server,), name=f"pull-{server}")
        for server in servers
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        chosen["_errors"] = json.dumps(errors)
    return chosen


def probe_vllm_python(rootfs: Path) -> None:
    python = which_in_rootfs(
        rootfs,
        [
            "python3.12",
            "python3.10",
            "python3",
            "opt/venv/bin/python3",
            "usr/bin/python3.12",
            "usr/bin/python3.10",
            "usr/bin/python3",
        ],
    )
    if python is None:
        raise RuntimeError(f"no python in {rootfs}")
    mount_rootfs(rootfs, [])
    try:
        run(
            [*sudo(), "chroot", str(rootfs), python, "-c", "import torch, vllm; print(vllm.__version__, torch.cuda.is_available())"],
            timeout=120,
        )
    finally:
        unmount_rootfs(rootfs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", choices=["vllm", "triton"], required=True)
    parser.add_argument("--size", choices=["7b", "32b"], required=True)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output", default=str(RESULTS))
    args = parser.parse_args()

    os.environ.setdefault("KRYO_LAZY_PAGES", "0")
    subprocess.run([*sudo(), "mkdir", "-p", str(SNAP_ROOT)], check=True)

    images = ensure_docker_images([args.server])
    if args.server == "vllm" and "vllm" in images:
        probe_vllm_python(ROOTFS_BASE / "vllm")

    gpu_name = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
    ).strip()
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    image_meta = {k: v for k, v in images.items() if k != "_errors"}
    image_errors = json.loads(images["_errors"]) if "_errors" in images else {}
    model = MODELS[args.size]
    results: dict[str, Any] = {
        "server": args.server,
        "size": args.size,
        "model": model,
        "images": image_meta,
        "image_errors": image_errors,
        "samples": args.samples,
        "gpu": gpu_name,
        "driver": driver,
    }
    if args.server not in images:
        results["error"] = image_errors.get(args.server, "image pull failed")
    else:
        rootfs = ROOTFS_BASE / ("vllm" if args.server == "vllm" else "triton")
        results.update(
            bench_server(
                server=args.server,
                size=args.size,
                model=model,
                rootfs=rootfs,
                samples=args.samples,
                timeout=args.timeout,
                port=args.port,
                gpu=args.gpu,
            )
        )

    Path(args.output).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {args.output}")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
