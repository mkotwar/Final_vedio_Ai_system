from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import patch

from tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts import build_global_vehicle_objects


class _FakeReport:
    def to_dict(self):
        return {
            "run_code": "RUN_20260724_151402",
            "mode": "dry_run",
            "rule_version": "global_match_v1",
            "tracks_loaded": 2,
            "candidate_count": 1,
            "decisions": {"confirmed": 1},
            "global_objects": [],
            "matches": [],
            "errors": [],
        }


class _FakeService:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def build_for_run(self, *_args, **_kwargs):
        return _FakeReport()


class BuildGlobalVehicleObjectsTests(unittest.TestCase):
    def test_cli_runs_in_default_dry_run_mode(self) -> None:
        output = io.StringIO()
        with (
            patch.object(build_global_vehicle_objects, "load_global_match_config", return_value=object()),
            patch.object(build_global_vehicle_objects, "AnalyticsDatabaseClient", return_value=object()),
            patch.object(build_global_vehicle_objects, "GlobalMatchService", _FakeService),
            contextlib.redirect_stdout(output),
        ):
            exit_code = build_global_vehicle_objects.run(["--run-code", "RUN_20260724_151402"])
        self.assertEqual(exit_code, build_global_vehicle_objects.EXIT_SUCCESS)
        self.assertIn('"candidate_count": 1', output.getvalue())

    def test_cli_rejects_persist_and_dry_run_together(self) -> None:
        with self.assertRaises(SystemExit):
            build_global_vehicle_objects.run(["--run-code", "RUN_20260724_151402", "--dry-run", "--persist"])


if __name__ == "__main__":
    unittest.main()
