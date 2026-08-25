"""Tests for YAML job loading. Run from ci/benchmark: python3 -m unittest."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from config import load_plan, parse_duration


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


class LoadPlanTests(unittest.TestCase):
    def test_release_yaml(self) -> None:
        plan = load_plan(HERE / "jobs" / "release.yaml")
        self.assertEqual(plan.provider, "lambda")
        self.assertEqual(plan.idle_timeout, 600)
        self.assertEqual(plan.caps["gpu_1x_a10"], 1)
        self.assertEqual(len(plan.jobs), 4)
        self.assertEqual(plan.jobs[0].scenario, "torch_cuda")
        self.assertEqual(plan.jobs[0].samples, 10)

    def test_unknown_scenario(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text(
                "provider: lambda\njobs:\n  - scenario: not_a_real_scenario\n    gpu: gpu_1x_a10\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_plan(path)


if __name__ == "__main__":
    unittest.main()
