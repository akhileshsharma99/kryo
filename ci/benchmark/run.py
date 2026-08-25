"""CLI for the GPU benchmark scheduler. Provider details live under providers/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from config import load_plan
from providers.lambda_cloud import destroy_dev_session
from scheduler import run_plan

REPO_ROOT = HERE.parent.parent
DEFAULT_JOBS = HERE / "jobs" / "release.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "results" / "latest.json"


def resolve_jobs(raw: str) -> Path:
    """Accept a path, a path relative to this package, or a file name in jobs/."""
    if not raw:
        return DEFAULT_JOBS
    path = Path(raw)
    for candidate in (path, HERE / path, HERE / "jobs" / path.name):
        if candidate.is_file():
            return candidate
    return path


def main() -> None:
    """Load a YAML job file and run it, or destroy leftover VMs."""
    parser = argparse.ArgumentParser(description="Run Kryo GPU benchmarks from a YAML job file")
    parser.add_argument(
        "--jobs",
        default="",
        help=f"YAML job file (default: {DEFAULT_JOBS})",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Aggregated JSON path")
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Leave VMs running when the job finishes (no idle reap after exit)",
    )
    parser.add_argument(
        "--destroy",
        action="store_true",
        help="Terminate leftover kryo-gha-* VMs and any saved kryo-dev session",
    )
    args = parser.parse_args()
    if args.destroy:
        destroy_dev_session()
        return
    jobs_path = resolve_jobs(args.jobs)
    if not jobs_path.is_file():
        parser.error(f"job file not found: {jobs_path}")
    plan = load_plan(jobs_path)
    run_plan(plan, Path(args.output), keep=args.keep)


if __name__ == "__main__":
    main()
