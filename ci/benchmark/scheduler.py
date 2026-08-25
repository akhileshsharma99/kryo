"""Queue samples onto a capped GPU pool. Provider-agnostic."""

from __future__ import annotations

import json
import os
import queue
import shlex
import subprocess
import tempfile
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from config import LLM_SCENARIOS, VLLM_SCENARIOS, BenchPlan, Job
from golden import apply_command, nfs_path, read_digest_command, write_digest_command
from golden import digest as golden_digest
from golden import local_path as golden_local_path
from golden import pack_command as pack_golden_command
from providers import get_provider
from providers.base import Machine, Provider
from snapshots import digest as snapshot_digest
from snapshots import pack_command as pack_snapshot_command
from snapshots import read_hash_command, tarball_path, unpack_command, write_hash_command

print = partial(print, flush=True)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
REPO_REMOTE = "kryo"
GOLDEN_STAMP = "/var/lib/kryo-bench/golden.stamp"
VLLM_STAMP = "/var/lib/kryo-bench/vllm.stamp"
REMOTE_SAMPLE = "/tmp/kryo-sample.json"


@dataclass
class Sample:
    """One cold+restore pair for a job."""

    job: Job
    index: int
    attempt: int = 0


@dataclass
class Pooled:
    """Pool bookkeeping around a provider Machine."""

    machine: Machine
    busy: bool = False
    idle_since: float = field(default_factory=time.monotonic)
    golden: bool = False
    snap_digests: set[str] = field(default_factory=set)


class MachinePool:
    """Reuse VMs up to per-SKU caps; reap them after idle_timeout."""

    def __init__(
        self,
        provider: Provider,
        caps: dict[str, int],
        idle_timeout: int,
        filesystem: str | None = None,
    ) -> None:
        self._provider = provider
        self._caps = caps
        self._idle_timeout = idle_timeout
        self._filesystem = filesystem
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._machines: list[Pooled] = []
        self._stop = False
        self._reaper = threading.Thread(target=self._reap_loop, name="idle-reaper", daemon=True)
        self._reaper.start()

    def acquire(self, sku: str) -> Pooled:
        """Block until an idle VM of this SKU is available, launching if under cap."""
        while True:
            launch = False
            with self._cv:
                if self._stop:
                    raise RuntimeError("machine pool is shutting down")
                idle = [
                    item for item in self._machines if item.machine.sku == sku and not item.busy
                ]
                if idle:
                    item = idle[0]
                    item.busy = True
                    return item
                count = sum(1 for item in self._machines if item.machine.sku == sku)
                cap = self._caps.get(sku, 1)
                if count < cap:
                    launch = True
                else:
                    self._cv.wait(timeout=1)
                    continue
            if launch:
                machine = self._provider.launch(sku, filesystem=self._filesystem)
                pooled = Pooled(machine=machine, busy=True)
                with self._cv:
                    self._machines.append(pooled)
                return pooled

    def release(self, pooled: Pooled) -> None:
        """Return a VM to the idle set."""
        with self._cv:
            pooled.busy = False
            pooled.idle_since = time.monotonic()
            self._cv.notify_all()

    def shutdown(self, *, terminate: bool) -> None:
        """Stop the reaper and optionally destroy every VM."""
        with self._cv:
            self._stop = True
            self._cv.notify_all()
            machines = list(self._machines)
            self._machines.clear()
        if terminate:
            for pooled in machines:
                try:
                    self._provider.terminate(pooled.machine)
                except Exception as error:
                    print(f"warning: terminate {pooled.machine.id}: {error}")

    def _reap_loop(self) -> None:
        while True:
            time.sleep(5)
            victims: list[Pooled] = []
            with self._cv:
                if self._stop:
                    return
                now = time.monotonic()
                keep: list[Pooled] = []
                for pooled in self._machines:
                    idle = (not pooled.busy) and (now - pooled.idle_since >= self._idle_timeout)
                    if idle:
                        victims.append(pooled)
                    else:
                        keep.append(pooled)
                self._machines = keep
                if victims:
                    self._cv.notify_all()
            for pooled in victims:
                print(
                    f"idle timeout: terminating {pooled.machine.id} "
                    f"({pooled.machine.sku}) after {self._idle_timeout}s"
                )
                try:
                    self._provider.terminate(pooled.machine)
                except Exception as error:
                    print(f"warning: idle terminate {pooled.machine.id}: {error}")


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


def compute_stats(values: list[float]) -> dict[str, float]:
    """Summary statistics for a list of timings."""
    if not values:
        return {}
    ranked = sorted(values)
    n = len(ranked)

    def percentile(p: float) -> float:
        if n == 1:
            return ranked[0]
        index = min(n - 1, max(0, round((p / 100) * (n - 1))))
        return ranked[index]

    return {
        "mean": float(mean(values)),
        "std": float(stdev(values)) if n > 1 else 0.0,
        "p50": float(percentile(50)),
        "p95": float(percentile(95)),
        "p99": float(percentile(99)),
        "min": float(ranked[0]),
        "max": float(ranked[-1]),
    }


def remote_env(sku: str) -> str:
    """Exports copied onto every timed runner invocation."""
    sha = git_sha()
    tag = os.environ.get("BENCH_RELEASE_TAG", "").strip()
    parts = [
        "export PATH=/usr/local/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH",
        f"export BENCH_INSTANCE_TYPE={shlex.quote(sku)}",
    ]
    if sha:
        parts.append(f"export BENCH_GIT_SHA={shlex.quote(sha)}")
    if tag:
        parts.append(f"export BENCH_RELEASE_TAG={shlex.quote(tag)}")
    return "; ".join(parts) + "; "


def rebuild_kryo() -> str:
    """Rebuild the CLI after rsync so the VM matches this checkout."""
    return (
        f"source $HOME/.cargo/env && cd {REPO_REMOTE} && "
        "cargo build --release --locked && "
        "sudo install -m 755 target/release/kryo /usr/local/bin/kryo"
    )


def probe_driver(provider: Provider, machine: Machine) -> tuple[str, str]:
    """NVIDIA driver and CUDA version reported by nvidia-smi."""
    driver = provider.run_output(
        machine,
        "nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1",
    ).strip()
    cuda = provider.run_output(
        machine,
        r"nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n1",
    ).strip()
    return driver, cuda


def extra_weights(provider: Provider, machine: Machine, job: Job) -> bool:
    """Download scenario weights and install vLLM if needed. Returns True if vLLM was installed."""
    if job.scenario in LLM_SCENARIOS:
        flag = shlex.quote(job.scenario)
        provider.run(
            machine,
            f"cd {REPO_REMOTE}/benchmarks && .venv/bin/python download_models.py --scenario {flag}",
            timeout=7200,
        )
    if job.scenario not in VLLM_SCENARIOS:
        return False
    have = provider.run_output(machine, f"sudo test -f {VLLM_STAMP} && echo yes || echo no").strip()
    if have == "yes":
        return False
    provider.run(
        machine,
        f"bash {REPO_REMOTE}/ci/benchmark/install_vllm.sh {REPO_REMOTE} && "
        f"sudo mkdir -p /var/lib/kryo-bench && sudo touch {VLLM_STAMP}",
        timeout=1800,
    )
    return True


def save_golden(
    provider: Provider, machine: Machine, sku: str, wanted: str, plan: BenchPlan
) -> None:
    """Pack the golden image onto NFS and/or the controller. Untimed."""
    dest = "/tmp/kryo-golden.tgz"
    if plan.golden.store == "filesystem" and machine.filesystem:
        dest = nfs_path(machine.filesystem, sku, wanted)
        parent = dest.rsplit("/", 1)[0]
        provider.run(machine, f"mkdir -p {shlex.quote(parent)}")
    print(f"packing golden to {dest}")
    provider.run(machine, pack_golden_command(dest), timeout=3600)
    if dest == "/tmp/kryo-golden.tgz":
        cache = golden_local_path(sku, wanted)
        provider.get(machine, dest, cache)
        print(f"golden cached locally {cache.name}")


def ensure_golden(provider: Provider, pooled: Pooled, job: Job, plan: BenchPlan) -> None:
    """Untimed image bring-up: restore a golden tarball, or run setup.sh once."""
    machine = pooled.machine
    provider.rsync(machine)
    if pooled.golden:
        extra_weights(provider, machine, job)
        return

    if plan.golden.mode == "setup":
        stamped = provider.run_output(
            machine, f"sudo test -f {GOLDEN_STAMP} && echo yes || echo no"
        ).strip()
        if stamped != "yes":
            print(f"golden setup on {machine.id}")
            provider.run(
                machine,
                f"bash {REPO_REMOTE}/ci/benchmark/setup.sh {REPO_REMOTE}",
                timeout=3600,
            )
        else:
            print(f"golden hit on {machine.id}; rebuilding kryo")
            provider.run(machine, rebuild_kryo(), timeout=600)
        pooled.golden = True
        extra_weights(provider, machine, job)
        return

    driver, cuda = probe_driver(provider, machine)
    wanted = golden_digest(machine.sku, driver, cuda)
    on_box = provider.run_output(machine, read_digest_command()).strip()
    if on_box == wanted:
        print(f"golden digest hit on {machine.id} {wanted}")
        provider.run(machine, rebuild_kryo(), timeout=600)
        pooled.golden = True
        extra_weights(provider, machine, job)
        return

    applied = False
    if machine.filesystem:
        nfs = nfs_path(machine.filesystem, machine.sku, wanted)
        have = provider.run_output(
            machine, f"test -f {shlex.quote(nfs)} && echo yes || echo no"
        ).strip()
        if have == "yes":
            print(f"golden hit filesystem {nfs}")
            provider.run(machine, apply_command(nfs), timeout=1800)
            applied = True

    cache = golden_local_path(machine.sku, wanted)
    if not applied and cache.is_file():
        print(f"golden hit local cache {cache.name}")
        provider.put(machine, cache, "/tmp/kryo-golden.tgz")
        provider.run(machine, apply_command("/tmp/kryo-golden.tgz"), timeout=1800)
        applied = True

    if not applied:
        print(f"golden miss {wanted}; running setup.sh")
        provider.run(
            machine,
            f"bash {REPO_REMOTE}/ci/benchmark/setup.sh {REPO_REMOTE}",
            timeout=3600,
        )
    else:
        provider.run(machine, rebuild_kryo(), timeout=600)

    provider.run(machine, write_digest_command(wanted))
    installed_vllm = extra_weights(provider, machine, job)
    if plan.golden.mode == "tarball" and (not applied or installed_vllm):
        try:
            save_golden(provider, machine, machine.sku, wanted, plan)
        except Exception as error:
            print(f"warning: could not save golden tarball: {error}")
    pooled.golden = True


def ensure_snapshot(provider: Provider, pooled: Pooled, job: Job) -> int | None:
    """Make sure `bench-{scenario}` on the VM matches the current digest.

    Downloads and snapshot create are untimed. Timed restore later reads the
    files from disk after drop_caches.
    """
    machine = pooled.machine
    kryo_ver = provider.run_output(machine, "kryo --version").strip()
    driver = provider.run_output(
        machine,
        "nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1",
    ).strip()
    wanted = snapshot_digest(job.scenario, machine.sku, kryo_ver, driver)
    on_box = provider.run_output(machine, read_hash_command(job.scenario)).strip()
    if on_box == wanted:
        print(f"snapshot hit on-box {job.scenario} {wanted}")
        pooled.snap_digests.add(wanted)
        return None

    cache = tarball_path(job.scenario, machine.sku, wanted)
    if cache.is_file():
        print(f"snapshot hit local cache {cache.name}")
        provider.put(machine, cache, "/tmp/kryo-snap.tgz")
        provider.run(machine, unpack_command(job.scenario, wanted), timeout=600)
        pooled.snap_digests.add(wanted)
        return None

    print(f"snapshot miss {job.scenario} {wanted}; creating")
    env = remote_env(machine.sku)
    scenario = shlex.quote(job.scenario)
    provider.run(
        machine,
        env + f"cd {REPO_REMOTE}/benchmarks && .venv/bin/python runner.py "
        f"--scenario {scenario} --once create --timeout {int(job.timeout)} "
        f"--output {REMOTE_SAMPLE}",
        timeout=job.timeout + 120,
    )
    provider.run(machine, write_hash_command(job.scenario, wanted))
    provider.run(machine, pack_snapshot_command(job.scenario), timeout=600)
    partial = cache.with_suffix(".partial")
    provider.get(machine, "/tmp/kryo-snap.tgz", partial)
    partial.replace(cache)
    provider.run(machine, "rm -f /tmp/kryo-snap.tgz")
    pooled.snap_digests.add(wanted)
    local_json = Path(tempfile.gettempdir()) / f"kryo-create-{job.scenario}.json"
    try:
        provider.get(machine, REMOTE_SAMPLE, local_json)
        data = json.loads(local_json.read_text(encoding="utf-8"))
        bytes_used = data.get("snapshot_bytes")
        return int(bytes_used) if isinstance(bytes_used, int) else None
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def run_timed_once(
    provider: Provider,
    machine: Machine,
    job: Job,
    mode: str,
    local_json: Path,
) -> dict[str, Any]:
    """One timed cold or restore sample. Weights and snapshots are already on disk."""
    env = remote_env(machine.sku)
    scenario = shlex.quote(job.scenario)
    keep = " --keep-snapshot" if mode == "restore" else ""
    provider.run(
        machine,
        env + f"cd {REPO_REMOTE}/benchmarks && .venv/bin/python runner.py "
        f"--scenario {scenario} --once {mode} --timeout {int(job.timeout)} "
        f"--output {REMOTE_SAMPLE}{keep}",
        timeout=job.timeout + 120,
    )
    provider.get(machine, REMOTE_SAMPLE, local_json)
    data = json.loads(local_json.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"invalid {mode} payload from {machine.id}")
    return data


def run_sample(
    provider: Provider, pooled: Pooled, sample: Sample, plan: BenchPlan
) -> dict[str, Any]:
    """Untimed prepare, then timed cold and timed restore."""
    job = sample.job
    print(f"sample {job.scenario}[{sample.index + 1}/{job.samples}] on {pooled.machine.id}")
    ensure_golden(provider, pooled, job, plan)
    image_bytes = ensure_snapshot(provider, pooled, job)
    work = Path("/tmp")
    cold_json = work / f"kryo-cold-{job.scenario}-{sample.index}.json"
    restore_json = work / f"kryo-restore-{job.scenario}-{sample.index}.json"
    cold = run_timed_once(provider, pooled.machine, job, "cold", cold_json)
    restore = run_timed_once(provider, pooled.machine, job, "restore", restore_json)
    result: dict[str, Any] = {
        "scenario": job.scenario,
        "gpu": pooled.machine.sku,
        "index": sample.index,
        "cold_seconds": cold.get("seconds"),
        "kryo_seconds": restore.get("seconds"),
        "machine": pooled.machine.id,
    }
    if image_bytes is not None:
        result["snapshot_bytes"] = image_bytes
    return result


def aggregate(
    plan: BenchPlan, rows: list[dict[str, Any]], errors: list[dict[str, str]]
) -> dict[str, Any]:
    """Fold per-sample rows into the JSON format_results.py already understands."""
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        name = row.get("scenario")
        if isinstance(name, str):
            by_scenario[name].append(row)

    scenarios: dict[str, Any] = {}
    for job in plan.jobs:
        samples = by_scenario.get(job.scenario, [])
        colds = [
            float(row["cold_seconds"])
            for row in samples
            if isinstance(row.get("cold_seconds"), (int, float))
        ]
        kryos = [
            float(row["kryo_seconds"])
            for row in samples
            if isinstance(row.get("kryo_seconds"), (int, float))
        ]
        failed = [item for item in errors if item.get("scenario") == job.scenario]
        block: dict[str, Any] = {
            "gpu": job.gpu,
            "cold": {"runs": len(colds), "samples": colds, "total": compute_stats(colds)},
            "kryo": {"runs": len(kryos), "samples": kryos, "total": compute_stats(kryos)},
        }
        sizes = [
            int(row["snapshot_bytes"])
            for row in samples
            if isinstance(row.get("snapshot_bytes"), int)
        ]
        if sizes:
            block["snapshot_bytes"] = sizes[-1]
        if colds and kryos:
            cold_mean = float(mean(colds))
            kryo_mean = float(mean(kryos))
            if kryo_mean > 0:
                block["speedup"] = cold_mean / kryo_mean
        if failed:
            block["errors"] = [item["error"] for item in failed]
        if not colds and not kryos:
            block["error"] = failed[0]["error"] if failed else "no successful samples"
        scenarios[job.scenario] = block

    metadata: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "jobs_file": str(plan.path),
        "provider": plan.provider,
        "idle_timeout": plan.idle_timeout,
        "caps": plan.caps,
        "golden_mode": plan.golden.mode,
        "golden_store": plan.golden.store,
        "drop_caches": True,
        "warmup": False,
    }
    sha = git_sha()
    if sha:
        metadata["git_sha"] = sha
    tag = os.environ.get("BENCH_RELEASE_TAG", "").strip()
    if tag:
        metadata["release_tag"] = tag
    return {"metadata": metadata, "scenarios": scenarios, "samples": rows}


def run_plan(plan: BenchPlan, output: Path, *, keep: bool = False) -> dict[str, Any]:
    """Execute every sample in the plan and write aggregated JSON."""
    provider = get_provider(plan.provider)
    provider.janitor()
    fs = (
        plan.golden.filesystem
        if plan.golden.mode == "tarball" and plan.golden.store == "filesystem"
        else None
    )
    pool = MachinePool(provider, plan.caps, plan.idle_timeout, filesystem=fs)
    pending: queue.Queue[Sample | None] = queue.Queue()
    for job in plan.jobs:
        for index in range(job.samples):
            pending.put(Sample(job=job, index=index))

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    lock = threading.Lock()

    def worker() -> None:
        while True:
            sample = pending.get()
            if sample is None:
                pending.task_done()
                return
            try:
                pooled = pool.acquire(sample.job.gpu)
                try:
                    row = run_sample(provider, pooled, sample, plan)
                finally:
                    pool.release(pooled)
                with lock:
                    rows.append(row)
                    cold = row.get("cold_seconds")
                    kryo = row.get("kryo_seconds")
                    print(f"  {sample.job.scenario}[{sample.index + 1}] cold={cold} kryo={kryo}")
            except Exception as error:
                message = str(error)
                print(f"sample failed {sample.job.scenario}[{sample.index + 1}]: {message}")
                if sample.attempt < sample.job.retries:
                    pending.put(
                        Sample(job=sample.job, index=sample.index, attempt=sample.attempt + 1)
                    )
                else:
                    with lock:
                        errors.append(
                            {
                                "scenario": sample.job.scenario,
                                "error": message,
                            }
                        )
            finally:
                pending.task_done()

    workers = max(sum(plan.caps.values()), 1)
    threads = [
        threading.Thread(target=worker, name=f"sample-{i}", daemon=True) for i in range(workers)
    ]
    for thread in threads:
        thread.start()
    try:
        pending.join()
    finally:
        for _ in threads:
            pending.put(None)
        for thread in threads:
            thread.join(timeout=5)
        pool.shutdown(terminate=not keep)

    results = aggregate(plan, rows, errors)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"results copied to {output}")
    if keep:
        print("leaving VMs running (--keep); they will not idle-reap after this process exits")
    return results
