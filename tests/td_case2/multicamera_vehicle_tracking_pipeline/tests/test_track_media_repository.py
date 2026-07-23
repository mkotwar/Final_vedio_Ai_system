from __future__ import annotations

import unittest

from tests.td_case2.multicamera_vehicle_tracking_pipeline.persistence.analytics_repository_base import AnalyticsRepositoryError
from tests.td_case2.multicamera_vehicle_tracking_pipeline.persistence.persistence_models import TrackMediaRecord
from tests.td_case2.multicamera_vehicle_tracking_pipeline.persistence.track_media_repository import TrackMediaRepository


class _FakeResponse:
    def __init__(self, data):
        self.data = data


class _FakeQuery:
    def __init__(self, table_name: str, response_data=None, should_fail: bool = False):
        self.table_name = table_name
        self.response_data = response_data if response_data is not None else []
        self.should_fail = should_fail
        self.calls = []

    def select(self, payload):
        self.calls.append(("select", payload))
        return self

    def eq(self, field, value):
        self.calls.append(("eq", (field, value)))
        return self

    def in_(self, field, values):
        self.calls.append(("in", (field, tuple(values))))
        return self

    def insert(self, payload):
        self.calls.append(("insert", payload))
        return self

    def execute(self):
        self.calls.append(("execute", None))
        if self.should_fail:
            raise RuntimeError(f"boom:{self.table_name}")
        return _FakeResponse(self.response_data)


class _FakeAnalyticsClient:
    def __init__(self, query_factory):
        self.query_factory = query_factory

    def table(self, table_name: str):
        return self.query_factory(table_name)


def _record(path: str = "RUN_1/CAM_001/track_000001/best_overall.jpg") -> TrackMediaRecord:
    return TrackMediaRecord(vehicle_track_id="track-1", media_type="BEST_VEHICLE_CROP", storage_uri=path, is_primary=True)


class TrackMediaRepositoryTests(unittest.TestCase):
    def test_upsert_inserts(self) -> None:
        queries = [
            _FakeQuery("track_media", []),
            _FakeQuery("track_media", [{"id": "media-id", "storage_uri": "RUN_1/CAM_001/track_000001/best_overall.jpg"}]),
        ]
        repo = TrackMediaRepository(_FakeAnalyticsClient(lambda _: queries.pop(0)))
        row = repo.upsert(_record())
        self.assertEqual(row["id"], "media-id")

    def test_retry_is_idempotent(self) -> None:
        existing_row = {"id": "existing-id", "vehicle_track_id": "track-1", "media_type": "BEST_VEHICLE_CROP", "storage_uri": "RUN_1/CAM_001/track_000001/best_overall.jpg"}
        repo = TrackMediaRepository(_FakeAnalyticsClient(lambda _: _FakeQuery("track_media", [existing_row])))
        row = repo.upsert(_record())
        self.assertEqual(row["id"], "existing-id")

    def test_bulk_result_counts_existing_and_inserted(self) -> None:
        prefetch = _FakeQuery("track_media", [{"vehicle_track_id": "track-1", "media_type": "BEST_VEHICLE_CROP", "storage_uri": "RUN_1/CAM_001/track_000001/best_overall.jpg"}])
        insert = _FakeQuery("track_media", [{"id": "new-id"}])
        queries = [prefetch, insert]
        repo = TrackMediaRepository(_FakeAnalyticsClient(lambda _: queries.pop(0)))
        result = repo.bulk_upsert([_record(), _record("RUN_1/CAM_001/track_000001/first.jpg")])
        self.assertEqual(result.attempted, 2)
        self.assertEqual(result.inserted, 1)
        self.assertEqual(result.already_existing, 1)

    def test_wrapped_failures(self) -> None:
        repo = TrackMediaRepository(_FakeAnalyticsClient(lambda _: _FakeQuery("track_media", should_fail=True)))
        with self.assertRaises(AnalyticsRepositoryError):
            repo.upsert(_record())


if __name__ == "__main__":
    unittest.main()
