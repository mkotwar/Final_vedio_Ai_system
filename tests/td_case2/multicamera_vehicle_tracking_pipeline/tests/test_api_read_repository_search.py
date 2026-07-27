from __future__ import annotations

import unittest
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any

from ..persistence.api_read_repository import AnalyticsReadRepository


@dataclass
class FakeResponse:
    data: list[dict[str, Any]]
    count: int | None = None


class FakeQuery:
    def __init__(self, client: "FakeClient", table_name: str) -> None:
        self.client = client
        self.table_name = table_name
        self.filters: list[tuple[str, str, Any]] = []
        self.count_requested = False
        self.order_by: tuple[str, bool] | None = None
        self.range_values: tuple[int, int] | None = None

    def select(self, _: str = "*", count: str | None = None) -> "FakeQuery":
        self.count_requested = count == "exact"
        return self

    def eq(self, field: str, value: Any) -> "FakeQuery":
        self.filters.append(("eq", field, value))
        return self

    def gte(self, field: str, value: Any) -> "FakeQuery":
        self.filters.append(("gte", field, value))
        return self

    def lte(self, field: str, value: Any) -> "FakeQuery":
        self.filters.append(("lte", field, value))
        return self

    def ilike(self, field: str, value: str) -> "FakeQuery":
        self.filters.append(("ilike", field, value))
        return self

    def in_(self, field: str, values: list[Any]) -> "FakeQuery":
        self.filters.append(("in", field, list(values)))
        return self

    def order(self, field: str, desc: bool = False) -> "FakeQuery":
        self.order_by = (field, desc)
        return self

    def or_(self, _: str) -> "FakeQuery":
        return self

    def limit(self, value: int) -> "FakeQuery":
        self.range_values = (0, max(value - 1, 0))
        return self

    def range(self, start: int, end: int) -> "FakeQuery":
        self.range_values = (start, end)
        return self

    def execute(self) -> FakeResponse:
        rows = [dict(row) for row in self.client.tables[self.table_name]]
        for op, field, value in self.filters:
            if op == "eq":
                rows = [row for row in rows if row.get(field) == value]
            elif op == "gte":
                rows = [row for row in rows if row.get(field) is not None and row.get(field) >= value]
            elif op == "lte":
                rows = [row for row in rows if row.get(field) is not None and row.get(field) <= value]
            elif op == "in":
                rows = [row for row in rows if row.get(field) in value]
            elif op == "ilike":
                pattern = value.replace("%", "*").upper()
                rows = [row for row in rows if fnmatch(str(row.get(field) or "").upper(), pattern)]
        total = len(rows)
        if self.order_by:
            field, desc = self.order_by
            rows.sort(key=lambda row: row.get(field) or "", reverse=desc)
        if self.range_values:
            start, end = self.range_values
            rows = rows[start : end + 1]
        return FakeResponse(data=rows, count=total if self.count_requested else None)


class FakeClient:
    def __init__(self) -> None:
        self.schema_name = "analytics"
        self.tables = {
            "processing_run": [{"id": "run-1", "run_code": "RUN_20260725_131944"}],
            "camera": [
                {"id": "cam-1", "camera_code": "CAM_001", "camera_name": "North", "location_name": "North"},
                {"id": "cam-2", "camera_code": "CAM_002", "camera_name": "South", "location_name": "South"},
            ],
            "vehicle_track": [
                {
                    "id": "track-1",
                    "processing_run_id": "run-1",
                    "camera_id": "cam-1",
                    "track_uuid": "RUN_20260725_131944:CAM_001:TRACK_4",
                    "local_track_id": 4,
                    "vehicle_class": "CAR",
                    "lifecycle_state": "COMPLETED",
                    "first_seen_at": "2026-07-25T13:19:44+05:30",
                    "last_seen_at": "2026-07-25T13:19:54+05:30",
                    "first_video_time_seconds": 12.0,
                    "last_video_time_seconds": 22.0,
                    "observation_count": 10,
                    "best_detection_confidence": 0.93,
                    "average_detection_confidence": 0.91,
                },
                {
                    "id": "track-2",
                    "processing_run_id": "run-1",
                    "camera_id": "cam-2",
                    "track_uuid": "RUN_20260725_131944:CAM_002:TRACK_4",
                    "local_track_id": 4,
                    "vehicle_class": "CAR",
                    "lifecycle_state": "COMPLETED",
                    "first_seen_at": "2026-07-25T13:19:50+05:30",
                    "last_seen_at": "2026-07-25T13:20:04+05:30",
                    "first_video_time_seconds": 18.0,
                    "last_video_time_seconds": 32.0,
                    "observation_count": 12,
                    "best_detection_confidence": 0.95,
                    "average_detection_confidence": 0.92,
                },
            ],
            "vehicle_attribute": [
                {"vehicle_track_id": "track-1", "primary_color": "GREY", "color_confidence": 0.91, "attribute_status": "CURRENT"},
                {"vehicle_track_id": "track-2", "primary_color": "GREY", "color_confidence": 0.9, "attribute_status": "CURRENT"},
            ],
            "plate_summary": [
                {
                    "vehicle_track_id": "track-1",
                    "selected_plate_reading_id": "reading-1",
                    "canonical_plate": None,
                    "plate_pattern": "STANDARD",
                    "status": "VERIFIED",
                    "confidence": 0.97,
                    "reading_count": 5,
                },
                {
                    "vehicle_track_id": "track-2",
                    "selected_plate_reading_id": "reading-2",
                    "canonical_plate": "DL8CBF6268",
                    "plate_pattern": "STANDARD",
                    "status": "UNKNOWN",
                    "confidence": 0.96,
                    "reading_count": 3,
                },
            ],
            "plate_detection": [
                {"id": "detect-1", "track_media_id": "media-plate-1", "confidence": 0.8, "frame_number": 12},
                {"id": "detect-2", "track_media_id": "media-plate-2", "confidence": 0.7, "frame_number": 16},
            ],
            "plate_reading": [
                {
                    "id": "reading-1",
                    "plate_detection_id": "detect-1",
                    "status": "VERIFIED",
                    "raw_text": "DL8CBF6268",
                    "normalized_text": "DL8CBF6268",
                    "plate_pattern": "STANDARD",
                    "confidence": 0.97,
                    "is_selected": True,
                    "metadata": {"verification_status": "VERIFIED"},
                },
                {
                    "id": "reading-2",
                    "plate_detection_id": "detect-2",
                    "status": "PARTIAL",
                    "raw_text": "MH12AB1715",
                    "normalized_text": "MH12AB1715",
                    "plate_pattern": "STANDARD",
                    "confidence": 0.72,
                    "is_selected": True,
                    "metadata": {"verification_status": "PARTIAL"},
                },
            ],
            "track_media": [
                {"id": "media-1", "vehicle_track_id": "track-1", "media_type": "BEST_VEHICLE_CROP", "storage_provider": "LOCAL", "storage_uri": "safe/ref.jpg", "selection_rank": 1, "is_primary": True},
                {"id": "media-plate-1", "vehicle_track_id": "track-1", "media_type": "PLATE_CROP", "storage_provider": "LOCAL", "storage_uri": "safe/plate1.jpg", "selection_rank": 1, "is_primary": True},
                {"id": "media-2", "vehicle_track_id": "track-2", "media_type": "BEST_VEHICLE_CROP", "storage_provider": "LOCAL", "storage_uri": "safe/ref2.jpg", "selection_rank": 1, "is_primary": True},
                {"id": "media-plate-2", "vehicle_track_id": "track-2", "media_type": "PLATE_CROP", "storage_provider": "LOCAL", "storage_uri": "safe/plate2.jpg", "selection_rank": 1, "is_primary": True},
            ],
            "global_vehicle": [
                {
                    "id": "global-1",
                    "processing_run_id": "run-1",
                    "global_vehicle_code": "GVO:RUN_20260725_131944:FA3FCF9E3ABC",
                    "canonical_plate": "DL8CBF6268",
                    "canonical_color": "GREY",
                    "canonical_vehicle_class": "CAR",
                    "first_seen_at": "2026-07-25T13:19:44+05:30",
                    "last_seen_at": "2026-07-25T13:20:04+05:30",
                    "identity_confidence": 0.95,
                    "camera_count": 2,
                    "track_count": 2,
                }
            ],
            "global_vehicle_track": [
                {"global_vehicle_id": "global-1", "vehicle_track_id": "track-1", "is_current": True, "association_status": "CONFIRMED", "attached_at": "2026-07-25T13:20:10+05:30"},
                {"global_vehicle_id": "global-1", "vehicle_track_id": "track-2", "is_current": True, "association_status": "CONFIRMED", "attached_at": "2026-07-25T13:20:10+05:30"},
            ],
            "track_observation": [],
            "processing_error": [],
            "cross_camera_match": [],
        }

    def table(self, table_name: str) -> FakeQuery:
        return FakeQuery(self, table_name)


class ApiReadRepositorySearchTests(unittest.TestCase):
    def test_repository_search_local_tracks_filters_by_colour_and_plate(self) -> None:
        repository = AnalyticsReadRepository(FakeClient())

        page = repository.search_local_tracks(
            run_code="RUN_20260725_131944",
            vehicle_class="CAR",
            colour="GREY",
            plate="6268",
            plate_match_type="ENDS_WITH",
            camera_codes=["CAM_001"],
            window_start=None,
            window_end=None,
            minimum_confidence=0.9,
            verified_plate_only=True,
            sort_by="RELEVANCE",
            sort_order="DESC",
            fetch_limit=25,
        )

        self.assertEqual(page.total, 1)
        self.assertEqual(page.items[0]["track_uuid"], "RUN_20260725_131944:CAM_001:TRACK_4")
        self.assertEqual(page.items[0]["plate_result"]["normalized_text"], "DL8CBF6268")
        self.assertEqual(page.items[0]["plate_result"]["source_media_id"], "media-plate-1")

    def test_repository_search_global_vehicles_requires_multi_camera_membership(self) -> None:
        repository = AnalyticsReadRepository(FakeClient())

        page = repository.search_global_vehicles(
            run_code="RUN_20260725_131944",
            vehicle_class="CAR",
            colour="GREY",
            plate="DL8CBF6268",
            plate_match_type="EXACT",
            camera_codes=["CAM_001", "CAM_002"],
            window_start=None,
            window_end=None,
            minimum_confidence=0.9,
            multi_camera_only=True,
            verified_plate_only=True,
            sort_by="RELEVANCE",
            sort_order="DESC",
            fetch_limit=25,
        )

        self.assertEqual(page.total, 1)
        self.assertEqual(page.items[0]["global_vehicle_code"], "GVO:RUN_20260725_131944:FA3FCF9E3ABC")
        self.assertEqual(page.items[0]["camera_codes"], ["CAM_001", "CAM_002"])
        self.assertEqual(page.items[0]["plate_result"]["status"], "VERIFIED")
        self.assertEqual(page.items[0]["plate_result"]["normalized_text"], "DL8CBF6268")

    def test_decorated_track_uses_selected_reading_when_summary_canonical_plate_is_null(self) -> None:
        repository = AnalyticsReadRepository(FakeClient())

        track = repository.get_track_by_uuid("RUN_20260725_131944:CAM_001:TRACK_4")

        assert track is not None
        self.assertEqual(track["track"]["canonical_plate"], "DL8CBF6268")
        self.assertEqual(track["track"]["plate_result"]["display_text"], "DL8CBF6268")
        self.assertEqual(track["plate"]["plate_result"]["status"], "VERIFIED")

    def test_best_plate_result_prefers_partial_over_unreadable(self) -> None:
        repository = AnalyticsReadRepository(FakeClient())

        best = repository._best_plate_result(
            [
                {"status": "UNREADABLE", "display_text": "Unreadable", "ocr_confidence": 0.91},
                {"status": "PARTIAL", "display_text": "...1715", "ocr_confidence": 0.72},
            ]
        )

        assert best is not None
        self.assertEqual(best["status"], "PARTIAL")

    def test_primary_vehicle_media_never_returns_plate_media(self) -> None:
        client = FakeClient()
        client.tables["track_media"] = [
            {
                "id": "media-plate-only",
                "vehicle_track_id": "track-1",
                "media_type": "PLATE_CROP",
                "storage_provider": "LOCAL",
                "storage_uri": "safe/plate-only.jpg",
                "selection_rank": 1,
                "is_primary": True,
            },
            {
                "id": "media-overall",
                "vehicle_track_id": "track-2",
                "media_type": "BEST_OVERALL",
                "storage_provider": "LOCAL",
                "storage_uri": "safe/overall.jpg",
                "selection_rank": 1,
                "is_primary": True,
            },
            {
                "id": "media-plate-2",
                "vehicle_track_id": "track-2",
                "media_type": "PLATE_CROP",
                "storage_provider": "LOCAL",
                "storage_uri": "safe/plate-2.jpg",
                "selection_rank": 1,
                "is_primary": True,
            },
        ]
        repository = AnalyticsReadRepository(client)

        bundle = repository._primary_media_bundle_by_track_id(["track-1", "track-2"])

        self.assertIsNone(bundle["track-1"]["primary_vehicle_media"])
        self.assertEqual(bundle["track-1"]["primary_plate_media"]["media_type"], "PLATE_CROP")
        self.assertEqual(bundle["track-2"]["primary_vehicle_media"]["media_type"], "BEST_OVERALL")
        self.assertEqual(bundle["track-2"]["primary_plate_media"]["media_type"], "PLATE_CROP")


if __name__ == "__main__":
    unittest.main()
