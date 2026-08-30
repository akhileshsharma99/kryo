"""Queue samples onto a capped GPU pool. Provider-agnostic."""

from __future__ import annotations

import atexit
import json
import os
import queue
import shlex
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from config import LLM_SCENARIOS, SERVER_SCENARIOS, BenchPlan, Job
from golden import apply_command, nfs_path, read_digest_command, write_digest_command
from golden import digest as golden_digest
from golden import pack_command as pack_golden_command
from providers import get_provider
from providers.base import Machine, Provider
from snapshots import REMOTE_SNAP_ROOT, read_hash_command, write_hash_command
from snapshots import digest as snapshot_digest

print = partial(print, flush=True)

LAUNCH_WAIT_SECONDS = 1800


def pool_sku(machine: Machine) -> str:
    """SKU the scheduler queued this VM under (may differ from the launched type)."""
    return machine.requested or machine.sku


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
REPO_REMOTE = "kryo"
GOLDEN_STAMP = "/var/lib/kryo-bench/golden.stamp"
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
        self._launching: dict[str, int] = {}
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._machines: list[Pooled] = []
        self._stop = False
        self._closed = False
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
                    item
                    for item in self._machines
                    if pool_sku(item.machine) == sku and not item.busy
                ]
                if idle:
                    item = idle[0]
                    item.busy = True
                    return item
                live = sum(1 for item in self._machines if pool_sku(item.machine) == sku)
                pending_launches = self._launching.get(sku, 0)
                cap = self._caps.get(sku, 1)
                if live + pending_launches < cap:
                    self._launching[sku] = pending_launches + 1
                    launch = True
                else:
                    self._cv.wait(timeout=1)
                    continue
            if launch:
                started = time.monotonic()
                while True:
                    try:
                        machine = self._provider.launch(sku, filesystem=self._filesystem)
                        pooled = Pooled(machine=machine, busy=True)
                        with self._cv:
                            self._machines.append(pooled)
                            self._launching[sku] = max(0, self._launching.get(sku, 1) - 1)
                            self._cv.notify_all()
                        return pooled
                    except Exception as error:
                        retry = _launch_retryable(error) and (
                            time.monotonic() - started < LAUNCH_WAIT_SECONDS
                        )
                        with self._cv:
                            stopping = self._stop
                        if stopping or not retry:
                            with self._cv:
                                self._launching[sku] = max(0, self._launching.get(sku, 1) - 1)
                                self._cv.notify_all()
                            raise
                        print(f"waiting 45s to launch {sku}: {error}")
                        time.sleep(45)
                    except BaseException:
                        with self._cv:
                            self._launching[sku] = max(0, self._launching.get(sku, 1) - 1)
                            self._cv.notify_all()
                        raise

    def release(self, pooled: Pooled) -> None:
        """Return a VM to the idle set."""
        with self._cv:
            pooled.busy = False
            pooled.idle_since = time.monotonic()
            self._cv.notify_all()

    def discard(self, pooled: Pooled) -> None:
        """Terminate a VM that is no longer reachable and drop it from the pool."""
        with self._cv:
            self._machines = [item for item in self._machines if item is not pooled]
            self._cv.notify_all()
        print(f"discarding {pooled.machine.id} ({pooled.machine.sku})")
        try:
            self._provider.terminate(pooled.machine)
        except Exception as error:
            print(f"warning: discard terminate {pooled.machine.id}: {error}")

    def shutdown(self, *, terminate: bool) -> None:
        """Stop the reaper and optionally destroy every VM. Safe to call twice."""
        with self._cv:
            if self._closed:
                return
            self._closed = True
            self._stop = True
            self._cv.notify_all()
            machines = list(self._machines)
            self._machines.clear()
        if terminate:
            for pooled in machines:
                try:
                    print(f"terminating {pooled.machine.id} ({pooled.machine.sku})")
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


def _launch_retryable(error: BaseException) -> bool:
    """Capacity and rate-limit errors should wait, not burn sample retries."""
    text = str(error).lower()
    return any(needle in text for needle in ("no capacity", "429", "rate limited", "rate_limited"))


def machine_lost(error: BaseException) -> bool:
    """SSH died; the VM should not be reused."""
    text = str(error).lower()
    return any(
        needle in text
        for needle in (
            "broken pipe",
            "operation timed out",
            "connection refused",
            "connection reset",
            "no route to host",
            "exit status 255",
        )
    )


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
        f"export KRYO_SNAPSHOTS_DIR={shlex.quote(REMOTE_SNAP_ROOT)}",
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


def extra_weights(provider: Provider, machine: Machine, job: Job) -> None:
    """Download scenario weights. Untimed."""
    if job.scenario not in LLM_SCENARIOS:
        return
    flag = shlex.quote(job.scenario)
    provider.run(
        machine,
        f"cd {REPO_REMOTE}/benchmarks && .venv/bin/python download_models.py --scenario {flag}",
        timeout=7200,
    )


def save_golden(
    provider: Provider, machine: Machine, sku: str, wanted: str, plan: BenchPlan
) -> None:
    """Copy the golden tree onto NFS. Untimed. No gzip."""
    if not (plan.golden.store == "filesystem" and machine.filesystem):
        print("golden.store is not filesystem; skip packing (no local gzip copy)")
        return
    dest = nfs_path(machine.filesystem, sku, wanted)
    parent = dest.rsplit("/", 1)[0]
    provider.run(machine, f"mkdir -p {shlex.quote(parent)}")
    print(f"packing golden to {dest}")
    provider.run(machine, pack_golden_command(dest), timeout=3600)


def golden_on_filesystem(provider: Provider, machine: Machine, wanted: str) -> bool:
    """True if this SKU's golden directory is already on the attached filesystem."""
    if not machine.filesystem:
        return False
    nfs = nfs_path(machine.filesystem, machine.sku, wanted)
    have = provider.run_output(
        machine, f"test -f {shlex.quote(nfs)}/.golden-ok && echo yes || echo no"
    ).strip()
    return have == "yes"


def maybe_save_golden(
    provider: Provider,
    machine: Machine,
    wanted: str,
    plan: BenchPlan,
    *,
    force: bool = False,
) -> None:
    """Pack golden once. Safe to call again; no-ops when the tarball exists."""
    if plan.golden.mode != "tarball":
        return
    if (
        not force
        and plan.golden.store == "filesystem"
        and golden_on_filesystem(provider, machine, wanted)
    ):
        return
    try:
        save_golden(provider, machine, machine.sku, wanted, plan)
    except Exception as error:
        print(f"warning: could not save golden directory: {error}")


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
        maybe_save_golden(provider, machine, wanted, plan)
        return

    applied = False
    if machine.filesystem:
        nfs = nfs_path(machine.filesystem, machine.sku, wanted)
        have = provider.run_output(
            machine, f"test -f {shlex.quote(nfs)}/.golden-ok && echo yes || echo no"
        ).strip()
        if have == "yes":
            print(f"golden hit filesystem {nfs}")
            provider.run(machine, apply_command(nfs), timeout=1800)
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
    extra_weights(provider, machine, job)
    maybe_save_golden(provider, machine, wanted, plan, force=not applied)
    pooled.golden = True


def ensure_snapshot(provider: Provider, pooled: Pooled, job: Job, plan: BenchPlan) -> int | None:
    """Create the CRIU snapshot once on this VM, then reuse it here.

    Dump is untimed. Do not restore a dump from another machine: CRIU needs the
    original PIDs, and a busy host PID table will collide. Golden images still
    move across VMs; GPU snapshots do not.
    """
    del plan
    machine = pooled.machine
    provider.run(machine, f"sudo mkdir -p {REMOTE_SNAP_ROOT}")
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

    print(f"snapshot miss {job.scenario} {wanted}; creating once on this VM")
    env = remote_env(machine.sku)
    scenario = shlex.quote(job.scenario)
    create_timeout = max(job.timeout * 2, job.timeout + 180)
    provider.run(
        machine,
        env + f"cd {REPO_REMOTE}/benchmarks && .venv/bin/python runner.py "
        f"--scenario {scenario} --once create --timeout {int(job.timeout)} "
        f"--output {REMOTE_SAMPLE}",
        timeout=create_timeout,
    )
    provider.run(machine, write_hash_command(job.scenario, wanted))
    pooled.snap_digests.add(wanted)
    local_json = Path(tempfile.gettempdir()) / f"kryo-create-{job.scenario}.json"
    try:
        provider.get(machine, REMOTE_SAMPLE, local_json)
        data = json.loads(local_json.read_text(encoding="utf-8"))
        bytes_used = data.get("snapshot_bytes")
        return int(bytes_used) if isinstance(bytes_used, int) else None
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def server_remote(job: Job) -> str:
    """One vLLM or Triton model on this VM. The other three jobs run on other VMs."""
    if job.scenario.startswith("vllm"):
        server = "vllm"
    else:
        server = "triton"
    size = "32b" if job.scenario.endswith("32") else "7b"
    return (
        f"cd {REPO_REMOTE}/benchmarks && python3 -u servers/remote_bench.py "
        f"--server {server} --size {size} --samples {int(job.samples)} "
        f"--timeout {int(job.timeout)} --output {REMOTE_SAMPLE}"
    )


def run_job(provider: Provider, pooled: Pooled, job: Job, plan: BenchPlan) -> dict[str, Any]:
    """Golden once, then on this VM: all colds, one dump, all restores."""
    print(f"job {job.scenario} ({job.samples} runs) on {pooled.machine.id}")
    ensure_golden(provider, pooled, job, plan)
    env = remote_env(pooled.machine.sku)
    if job.scenario in SERVER_SCENARIOS:
        command = env + server_remote(job)
        batch_timeout = int(job.timeout) * (int(job.samples) + 3) * 2 + 1800
    else:
        scenario = shlex.quote(job.scenario)
        command = (
            env + f"cd {REPO_REMOTE}/benchmarks && .venv/bin/python runner.py "
            f"--scenario {scenario} --runs {int(job.samples)} --timeout {int(job.timeout)} "
            f"--output {REMOTE_SAMPLE}"
        )
        batch_timeout = int(job.timeout) * (int(job.samples) + 3) * 2 + 300
    provider.run(pooled.machine, command, timeout=batch_timeout)
    local_json = Path(tempfile.gettempdir()) / f"kryo-job-{job.scenario}.json"
    provider.get(pooled.machine, REMOTE_SAMPLE, local_json)
    data = json.loads(local_json.read_text(encoding="utf-8"))
    if job.scenario in SERVER_SCENARIOS:
        block = data if isinstance(data, dict) else None
    else:
        scenarios = data.get("scenarios") if isinstance(data, dict) else None
        block = scenarios.get(job.scenario) if isinstance(scenarios, dict) else None
    if not isinstance(block, dict):
        raise RuntimeError(f"missing scenario payload for {job.scenario}")
    block["gpu"] = pooled.machine.sku
    block["machine"] = pooled.machine.id
    return block


def results_metadata(plan: BenchPlan) -> dict[str, Any]:
    """Controller metadata attached to the aggregated JSON."""
    metadata: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "jobs_file": str(plan.path),
        "provider": plan.provider,
        "idle_timeout": plan.idle_timeout,
        "caps": plan.caps,
        "golden_mode": plan.golden.mode,
        "golden_store": plan.golden.store,
        "drop_caches": True,
        "warmup": True,
    }
    sha = git_sha()
    if sha:
        metadata["git_sha"] = sha
    tag = os.environ.get("BENCH_RELEASE_TAG", "").strip()
    if tag:
        metadata["release_tag"] = tag
    return metadata


DEFAULT_MAX_SECONDS = 3 * 60 * 60


def max_run_seconds() -> int:
    """Hard cap on a bench process so a dead session cannot bill GPUs for days."""
    raw = os.environ.get("BENCH_MAX_SECONDS", str(DEFAULT_MAX_SECONDS)).strip()
    try:
        return max(60, int(raw))
    except ValueError:
        return DEFAULT_MAX_SECONDS


def run_plan(plan: BenchPlan, output: Path, *, keep: bool = False) -> dict[str, Any]:
    """Run each job on a pooled VM: colds, one dump, restores."""
    provider = get_provider(plan.provider)
    provider.janitor()
    fs = (
        plan.golden.filesystem
        if plan.golden.mode == "tarball" and plan.golden.store == "filesystem"
        else None
    )
    pool = MachinePool(provider, plan.caps, plan.idle_timeout, filesystem=fs)
    limit = max_run_seconds()
    print(f"GPU pool hard stop after {limit}s (BENCH_MAX_SECONDS); Ctrl-C or SIGTERM destroys VMs")

    def _kill_pool() -> None:
        pool.shutdown(terminate=not keep)

    atexit.register(_kill_pool)

    def _on_signal(signum: int, _frame: Any) -> None:
        print(f"received signal {signum}; destroying GPU pool")
        _kill_pool()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    by_sku: dict[str, queue.Queue[Sample | None]] = {}
    for job in plan.jobs:
        by_sku.setdefault(job.gpu, queue.Queue()).put(Sample(job=job, index=0))

    scenarios: dict[str, Any] = {}
    lock = threading.Lock()

    def worker(sku: str, pending: queue.Queue[Sample | None]) -> None:
        while True:
            sample = pending.get()
            if sample is None:
                pending.task_done()
                return
            job = sample.job
            pooled = None
            try:
                pooled = pool.acquire(sku)
                block = run_job(provider, pooled, job, plan)
                with lock:
                    scenarios[job.scenario] = block
                    cold = block.get("cold", {}).get("total", {}).get("mean")
                    kryo = block.get("kryo", {}).get("total", {}).get("mean")
                    speedup = block.get("speedup")
                    print(f"  {job.scenario} cold={cold} kryo={kryo} speedup={speedup}")
                pool.release(pooled)
                pooled = None
            except Exception as error:
                message = str(error)
                print(f"job failed {job.scenario}: {message}")
                if pooled is not None and machine_lost(error):
                    pool.discard(pooled)
                    pooled = None
                elif pooled is not None:
                    pool.release(pooled)
                    pooled = None
                if sample.attempt < job.retries:
                    delay = min(30 * (2**sample.attempt), 120)
                    print(f"retrying {job.scenario} in {delay}s")
                    time.sleep(delay)
                    pending.put(Sample(job=job, index=0, attempt=sample.attempt + 1))
                else:
                    with lock:
                        scenarios[job.scenario] = {"gpu": job.gpu, "error": message}
            finally:
                pending.task_done()

    threads: list[threading.Thread] = []
    for sku, pending in by_sku.items():
        n_workers = min(plan.caps.get(sku, 1), max(pending.qsize(), 1))
        for index in range(n_workers):
            thread = threading.Thread(
                target=worker, args=(sku, pending), name=f"{sku}-{index}", daemon=True
            )
            threads.append(thread)
            thread.start()
    finished = threading.Event()

    def _wait_queues() -> None:
        for pending in by_sku.values():
            pending.join()
        finished.set()

    waiter = threading.Thread(target=_wait_queues, name="queue-join", daemon=True)
    waiter.start()
    timed_out = False
    try:
        if not finished.wait(timeout=limit):
            timed_out = True
            print(f"hit BENCH_MAX_SECONDS={limit}; destroying GPUs")
            pool.shutdown(terminate=True)
    finally:
        for sku, pending in by_sku.items():
            n_workers = min(plan.caps.get(sku, 1), 8)
            for _ in range(n_workers):
                pending.put(None)
        for thread in threads:
            thread.join(timeout=5)
        pool.shutdown(terminate=not keep)
    if timed_out:
        raise RuntimeError(
            f"benchmark exceeded {limit}s; GPUs terminated. "
            "raise BENCH_MAX_SECONDS if this was a real run."
        )

    results = {"metadata": results_metadata(plan), "scenarios": scenarios}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"results copied to {output}")
    if keep:
        print("leaving VMs running (--keep); they will not idle-reap after this process exits")
    return results
