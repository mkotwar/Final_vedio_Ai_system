from __future__ import annotations

import unittest

from ..api.search_models import VehicleSearchQuery
from ..api.services.vehicle_search_service import VehicleSearchService
from ..persistence.api_read_repository import Page


class StubSearchRepository:
    def search_local_tracks(self, **_: object) -> Page:
        return Page(
            items=[
                {
                    "result_type": "LOCAL_TRACK",
                    "track_uuid": "RUN_20260725_131944:CAM_001:TRACK_4",
                    "global_vehicle_code": None,
                    "class_name": "CAR",
                    "colour": "GREY",
                    "plate": "DL8CBF6268",
                    "plate_status": "VERIFIED",
                    "camera_codes": ["CAM_001"],
                    "first_seen_at": "2026-07-25T13:19:44+05:30",
                    "last_seen_at": "2026-07-25T13:19:54+05:30",
                    "confidence": 0.91,
                    "member_track_count": 1,
                    "primary_media": {"media_id": "media-local"},
                }
            ],
            page=1,
            page_size=100,
            total=1,
        )

    def search_global_vehicles(self, **_: object) -> Page:
        return Page(
            items=[
                {
                    "result_type": "GLOBAL_VEHICLE",
                    "track_uuid": None,
                    "global_vehicle_code": "GVO:RUN_20260725_131944:FA3FCF9E3ABC",
                    "class_name": "CAR",
                    "colour": "GREY",
                    "plate": "DL8CBF6268",
                    "plate_status": "VERIFIED",
                    "camera_codes": ["CAM_001", "CAM_002"],
                    "first_seen_at": "2026-07-25T13:19:44+05:30",
                    "last_seen_at": "2026-07-25T13:20:04+05:30",
                    "confidence": 0.95,
                    "member_track_count": 2,
                    "primary_media": {"media_id": "media-global"},
                }
            ],
            page=1,
            page_size=100,
            total=1,
        )


class StubMediaService:
    def decorate_media_reference(self, row: dict | None) -> dict | None:
        if row is None:
            return None
        return {"media_id": row.get("media_id"), "availability": "REFERENCE_ONLY", "content_url": None}


class VehicleSearchServiceTests(unittest.TestCase):
    def test_vehicle_search_service_ranks_exact_verified_global_result_first(self) -> None:
        service = VehicleSearchService(StubSearchRepository(), media_service=StubMediaService())

        response = service.search(
            VehicleSearchQuery(
                run_code="RUN_20260725_131944",
                vehicle_class="CAR",
                colour="GREY",
                plate="DL8CBF6268",
                camera_codes="CAM_001,CAM_002",
                result_scope="ALL",
            )
        )

        self.assertEqual(response["pagination"]["total"], 2)
        self.assertEqual(response["results"][0]["result_type"], "GLOBAL_VEHICLE")
        self.assertIn("exact verified plate", response["results"][0]["match_reasons"])

    def test_vehicle_search_service_applies_offset_and_limit(self) -> None:
        service = VehicleSearchService(StubSearchRepository(), media_service=StubMediaService())

        response = service.search(
            VehicleSearchQuery(
                run_code="RUN_20260725_131944",
                result_scope="ALL",
                limit=1,
                offset=1,
            )
        )

        self.assertEqual(response["pagination"]["returned"], 1)
        self.assertEqual(response["results"][0]["result_type"], "LOCAL_TRACK")


if __name__ == "__main__":
    unittest.main()
