from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import patch

from tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts import verify_global_vehicle_objects


class _FakeResponse:
    def __init__(self, data):
        self.data = data


class _FakeQuery:
    def __init__(self, table_name: str, dataset: dict[str, list[dict]], filters=None):
        self.table_name = table_name
        self.dataset = dataset
        self.filters = filters or []

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        return _FakeQuery(self.table_name, self.dataset, self.filters + [("eq", key, value)])

    def in_(self, key, values):
        return _FakeQuery(self.table_name, self.dataset, self.filters + [("in", key, list(values))])

    def limit(self, _value):
        return self

    def execute(self):
        rows = [dict(row) for row in self.dataset.get(self.table_name, [])]
        for filter_type, key, value in self.filters:
            if filter_type == "eq":
                rows = [row for row in rows if row.get(key) == value]
            else:
                rows = [row for row in rows if row.get(key) in value]
        return _FakeResponse(rows)


class _FakeAnalyticsClient:
    def __init__(self, dataset):
        self.dataset = dataset

    def table(self, table_name: str):
        return _FakeQuery(table_name, self.dataset)


def _dataset():
    return {
        "processing_run": [{"id": "run-1", "run_code": "RUN_20260724_151402", "status": "COMPLETED"}],
        "camera": [{"id": "cam-1", "camera_code": "CAM_001"}, {"id": "cam-2", "camera_code": "CAM_002"}],
        "vehicle_track": [
            {"id": "track-1", "track_uuid": "RUN_20260724_151402:CAM_001:TRACK_4", "camera_id": "cam-1", "vehicle_class": "CAR", "processing_run_id": "run-1"},
            {"id": "track-2", "track_uuid": "RUN_20260724_151402:CAM_002:TRACK_4", "camera_id": "cam-2", "vehicle_class": "CAR", "processing_run_id": "run-1"},
        ],
        "global_vehicle": [
            {
                "id": "gvo-1",
                "global_vehicle_code": "GVO:RUN_20260724_151402:ABC",
                "processing_run_id": "run-1",
                "status": "CONFIRMED",
                "canonical_plate": "DL8CBF6268",
                "canonical_color": "WHITE",
                "canonical_vehicle_class": "CAR",
            }
        ],
        "global_vehicle_track": [
            {"id": "member-1", "global_vehicle_id": "gvo-1", "vehicle_track_id": "track-1", "association_status": "CONFIRMED", "association_score": 0.95, "is_current": True},
            {"id": "member-2", "global_vehicle_id": "gvo-1", "vehicle_track_id": "track-2", "association_status": "CONFIRMED", "association_score": 0.95, "is_current": True},
        ],
        "cross_camera_match": [{"id": "match-1", "processing_run_id": "run-1", "decision": "CONFIRMED"}],
    }


class VerifyGlobalVehicleObjectsTests(unittest.TestCase):
    def test_generate_report_detects_multi_camera_object(self) -> None:
        report = verify_global_vehicle_objects.generate_report(_FakeAnalyticsClient(_dataset()), "RUN_20260724_151402")
        self.assertEqual(report["global_object_count"], 1)
        self.assertEqual(report["multi_camera_global_objects"], 1)
        self.assertEqual(report["verification_status"], "PASS")

    def test_cli_strict_failure_is_reported(self) -> None:
        broken = _dataset()
        broken["global_vehicle_track"] = broken["global_vehicle_track"][:1]
        output = io.StringIO()
        with patch.object(verify_global_vehicle_objects, "AnalyticsDatabaseClient", return_value=_FakeAnalyticsClient(broken)), contextlib.redirect_stdout(output):
            exit_code = verify_global_vehicle_objects.run(["--run-code", "RUN_20260724_151402", "--strict"])
        self.assertEqual(exit_code, verify_global_vehicle_objects.EXIT_STRICT_FAILED)


if __name__ == "__main__":
    unittest.main()
