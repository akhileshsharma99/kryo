"""Tests for YAML job loading. Run from ci/benchmark: python3 -m unittest."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from config import load_plan, parse_duration
from format_results import merge_result_files, readme_image
from golden import digest as golden_digest
from providers.lambda_cloud import parse_gha_name, should_reap_instance
from scheduler import max_run_seconds


class ParseDurationTests(unittest.TestCase):
    def test_seconds_int(self) -> None:
        self.assertEqual(parse_duration(90), 90)

    def test_suffixes(self) -> None:
        self.assertEqual(parse_duration("10m"), 600)
        self.assertEqual(parse_duration("1h"), 3600)
        self.assertEqual(parse_duration("30s"), 30)

    def test_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            parse_duration("0")


class MaxRunSecondsTests(unittest.TestCase):
    def test_default_is_three_hours(self) -> None:
        env = {key: value for key, value in os.environ.items() if key != "BENCH_MAX_SECONDS"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(max_run_seconds(), 3 * 60 * 60)

    def test_env_override(self) -> None:
        with patch.dict(os.environ, {"BENCH_MAX_SECONDS": "120"}):
            self.assertEqual(max_run_seconds(), 120)


class LoadPlanTests(unittest.TestCase):
    def test_a10_yaml(self) -> None:
        plan = load_plan(HERE / "jobs" / "a10.yaml")
        self.assertEqual(plan.provider, "lambda")
        self.assertEqual(plan.idle_timeout, 600)
        self.assertEqual(plan.caps["gpu_1x_a10"], 1)
        self.assertEqual(len(plan.jobs), 4)
        self.assertEqual(plan.jobs[0].scenario, "torch_cuda")
        self.assertEqual(plan.jobs[0].samples, 10)
        self.assertEqual(plan.golden.mode, "tarball")
        self.assertEqual(plan.golden.store, "filesystem")
        self.assertEqual(plan.golden.filesystem, "kryo-golden")
        self.assertEqual(plan.snapshots.store, "local")

    def test_h100_yaml(self) -> None:
        plan = load_plan(HERE / "jobs" / "h100.yaml")
        self.assertEqual(plan.caps["gpu_1x_h100_sxm5"], 1)
        self.assertTrue(all(job.gpu == "gpu_1x_h100_sxm5" for job in plan.jobs))
        names = [job.scenario for job in plan.jobs]
        self.assertEqual(names, ["qwen7", "torch_compile", "qwen32"])

    def test_digest_changes_with_sku(self) -> None:
        left = golden_digest("gpu_1x_a10", "550.00", "12.8")
        right = golden_digest("gpu_1x_h100_pcie", "550.00", "12.8")
        self.assertNotEqual(left, right)

    def test_snapshot_pack_uses_canonical_dir(self) -> None:
        from snapshots import REMOTE_SNAP_ROOT, pack_command

        command = pack_command("torch_cuda", "/tmp/kryo-snap.tgz")
        self.assertIn(REMOTE_SNAP_ROOT, command)
        self.assertIn("bench-torch_cuda", command)
        self.assertEqual(REMOTE_SNAP_ROOT, "/var/lib/kryo-bench/criu")

    def test_unknown_scenario(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text(
                "provider: lambda\njobs:\n  - scenario: not_a_real_scenario\n    gpu: gpu_1x_a10\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_plan(path)


class GhaNameTests(unittest.TestCase):
    def test_parse_timestamped_name(self) -> None:
        epoch, run_id = parse_gha_name("kryo-gha-t1700000000-99-a10-1")
        self.assertEqual(epoch, 1700000000)
        self.assertEqual(run_id, "99")

    def test_legacy_name_has_no_epoch(self) -> None:
        epoch, run_id = parse_gha_name("kryo-gha-92480-2")
        self.assertIsNone(epoch)
        self.assertIsNone(run_id)

    def test_keep_sibling_shard(self) -> None:
        now = 1700000000 + 60
        self.assertFalse(
            should_reap_instance(
                "kryo-gha-t1700000000-99-h100-1",
                keep_run_id="99",
                max_age=14400,
                now=now,
            )
        )

    def test_reap_other_run(self) -> None:
        self.assertTrue(
            should_reap_instance(
                "kryo-gha-t1700000000-88-a10-1",
                keep_run_id="99",
                max_age=14400,
                now=1700000000 + 60,
            )
        )

    def test_reap_stale_and_legacy_on_cron(self) -> None:
        now = 1700000000 + 5 * 3600
        self.assertTrue(
            should_reap_instance(
                "kryo-gha-t1700000000-99-a10-1",
                keep_run_id=None,
                max_age=14400,
                now=now,
            )
        )
        self.assertTrue(
            should_reap_instance(
                "kryo-gha-92480-2",
                keep_run_id=None,
                max_age=14400,
                now=now,
            )
        )
        self.assertFalse(
            should_reap_instance(
                "kryo-gha-t1700000000-99-a10-1",
                keep_run_id=None,
                max_age=14400,
                now=1700000000 + 60,
            )
        )


class MergeResultsTests(unittest.TestCase):
    def test_merge_combines_scenarios(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "a10.json"
            right = Path(tmp) / "h100.json"
            left.write_text(
                '{"metadata": {"a": 1}, "scenarios": {"yolo": {"gpu": "gpu_1x_a10"}}}\n',
                encoding="utf-8",
            )
            right.write_text(
                '{"metadata": {"b": 2}, "scenarios": {"qwen7": {"gpu": "gpu_1x_h100_pcie"}}}\n',
                encoding="utf-8",
            )
            merged = merge_result_files([left, right])
            self.assertEqual(merged["metadata"]["a"], 1)
            self.assertEqual(merged["metadata"]["b"], 2)
            self.assertIn("yolo", merged["scenarios"])
            self.assertIn("qwen7", merged["scenarios"])


class ReadmeImageTests(unittest.TestCase):
    def test_cache_key_changes_with_results(self) -> None:
        first = {
            "metadata": {"release_tag": "v0.4.0"},
            "scenarios": {"yolo": {"cold": {"total": {"mean": 3.4}}}},
        }
        second = {
            "metadata": {"release_tag": "v0.4.0"},
            "scenarios": {"yolo": {"cold": {"total": {"mean": 3.5}}}},
        }

        first_image = readme_image(first, "benchmarks/results/chart.svg")
        second_image = readme_image(second, "benchmarks/results/chart.svg")

        self.assertRegex(first_image, r"\?v=v0\.4\.0-[0-9a-f]{12}$")
        self.assertNotEqual(first_image, second_image)


if __name__ == "__main__":
    unittest.main()
