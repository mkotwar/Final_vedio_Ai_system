from __future__ import annotations

import unittest
from datetime import datetime, timezone

from tests.td_case2.multicamera_vehicle_tracking_pipeline.cross_camera.global_match_models import GlobalObjectMembership, GlobalVehicleObjectProposal
from tests.td_case2.multicamera_vehicle_tracking_pipeline.persistence.global_vehicle_object_repository import GlobalVehicleObjectRepository


class _FakeResponse:
    def __init__(self, data):
        self.data = data


class _FakeQuery:
    def __init__(self, table_name: str, dataset: dict[str, list[dict]], operations=None):
        self.table_name = table_name
        self.dataset = dataset
        self.operations = operations or []

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        return _FakeQuery(self.table_name, self.dataset, self.operations + [("eq", key, value)])

    def in_(self, key, values):
        return _FakeQuery(self.table_name, self.dataset, self.operations + [("in", key, list(values))])

    def limit(self, _value):
        return self

    def insert(self, payload):
        row = dict(payload)
        row.setdefault("id", f"{self.table_name}-{len(self.dataset.setdefault(self.table_name, [])) + 1}")
        self.dataset.setdefault(self.table_name, []).append(row)
        return _FakeQuery(self.table_name, self.dataset, self.operations + [("insert", row)])

    def update(self, payload):
        self.dataset["_pending_update"] = dict(payload)
        return self

    def execute(self):
        rows = [dict(row) for row in self.dataset.get(self.table_name, [])]
        for operation in self.operations:
            if operation[0] == "eq":
                _, key, value = operation
                rows = [row for row in rows if row.get(key) == value]
            elif operation[0] == "in":
                _, key, values = operation
                rows = [row for row in rows if row.get(key) in values]
            elif operation[0] == "insert":
                _, row = operation
                rows = [row]
        if "_pending_update" in self.dataset and rows:
            rows[0].update(self.dataset.pop("_pending_update"))
        return _FakeResponse(rows)


class _FakeAnalyticsClient:
    def __init__(self, dataset: dict[str, list[dict]]):
        self.dataset = dataset

    def table(self, table_name: str):
        return _FakeQuery(table_name, self.dataset)


class GlobalVehicleObjectRepositoryTests(unittest.TestCase):
    def test_create_or_get_global_object_is_idempotent(self) -> None:
        repo = GlobalVehicleObjectRepository(_FakeAnalyticsClient({}))
        now = datetime(2026, 7, 24, 10, 0, 0, tzinfo=timezone.utc)
        proposal = GlobalVehicleObjectProposal(
            processing_run_id="run-1",
            global_object_code="GVO:RUN:001",
            status="CONFIRMED",
            confidence=0.95,
            canonical_plate="DL8CBF6268",
            canonical_colour="WHITE",
            canonical_vehicle_class="CAR",
            first_seen_at=now,
            last_seen_at=now,
            creation_method="VERIFIED_PLATE",
            camera_count=2,
            track_count=2,
            members=(GlobalObjectMembership("track-1", "RUN:CAM_001:TRACK_1", "CONFIRMED", 0.95, "VERIFIED_PLATE"),),
        )
        first = repo.create_or_get_global_object(proposal)
        second = repo.create_or_get_global_object(proposal)
        self.assertEqual(first["global_vehicle_code"], second["global_vehicle_code"])

    def test_rejects_second_active_object_for_same_track(self) -> None:
        dataset = {
            "global_vehicle_track": [{"id": "member-1", "global_vehicle_id": "gvo-1", "vehicle_track_id": "track-1", "is_current": True}],
        }
        repo = GlobalVehicleObjectRepository(_FakeAnalyticsClient(dataset))
        with self.assertRaises(RuntimeError):
            repo.add_or_update_member("gvo-2", GlobalObjectMembership("track-1", "RUN:CAM_001:TRACK_1", "CONFIRMED", 0.9, "RULE_BASED"))


if __name__ == "__main__":
    unittest.main()
