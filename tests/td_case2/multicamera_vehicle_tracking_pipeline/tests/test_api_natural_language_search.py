from __future__ import annotations

import unittest

from ..api.main import create_app
from ..api.services.natural_language_search_service import NaturalLanguageSearchService
from ..api.settings import ApiSettings
from ..persistence.api_read_repository import Page
from .test_api_helpers import build_test_client


class _NaturalLanguageRepository:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def find_run_by_code(self, run_code: str):
        self.calls.append(("find_run_by_code", {"run_code": run_code}))
        return {
            "id": "run-1",
            "run_code": run_code,
            "started_at": "2026-07-25T13:19:44+05:30",
        }

    def list_run_cameras(self, *, run_code: str, page: int, page_size: int):
        self.calls.append(("list_run_cameras", {"run_code": run_code, "page": page, "page_size": page_size}))
        return {"id": "run-1"}, Page(
            items=[{"camera_code": "CAM_001"}, {"camera_code": "CAM_002"}],
            page=page,
            page_size=page_size,
            total=2,
        )

    def search_local_tracks(self, **kwargs):
        self.calls.append(("search_local_tracks", kwargs))
        return Page(items=[], page=1, page_size=kwargs["fetch_limit"], total=0)

    def search_global_vehicles(self, **kwargs):
        self.calls.append(("search_global_vehicles", kwargs))
        return Page(
            items=[
                {
                    "result_type": "GLOBAL_VEHICLE",
                    "global_vehicle_code": "GVO:RUN_20260725_131944:FA3FCF9E3ABC",
                    "track_uuid": None,
                    "class_name": "CAR",
                    "colour": "GREY",
                    "plate": "DL8CBF6268",
                    "plate_status": "VERIFIED",
                    "camera_codes": ["CAM_001", "CAM_002"],
                    "first_seen_at": "2026-07-25T13:19:44+05:30",
                    "last_seen_at": "2026-07-25T13:20:04+05:30",
                    "confidence": 0.95,
                    "member_track_count": 2,
                    "primary_media": None,
                }
            ],
            page=1,
            page_size=kwargs["fetch_limit"],
            total=1,
        )


class ApiNaturalLanguageSearchTests(unittest.TestCase):
    def test_search_endpoint_returns_expected_global_vehicle(self) -> None:
        repository = _NaturalLanguageRepository()
        client = build_test_client(repository)

        response = client.post(
            "/api/v1/search/natural-language",
            json={
                "query": "Find the grey car with plate ending in 6268.",
                "run_code": "RUN_20260725_131944",
                "result_scope": "GLOBAL_VEHICLES",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["results"][0]["global_vehicle_code"], "GVO:RUN_20260725_131944:FA3FCF9E3ABC")
        self.assertTrue(body["parser"]["fallback_used"])
        self.assertIn(("search_global_vehicles",), [(name,) for name, _ in repository.calls])

    def test_parse_endpoint_does_not_query_database(self) -> None:
        repository = _NaturalLanguageRepository()
        client = build_test_client(repository)

        response = client.post(
            "/api/v1/search/natural-language/parse",
            json={
                "query": "Find vehicle DL8CBF6268.",
                "run_code": "RUN_20260725_131944",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["parsed_intent"]["plate"], "DL8CBF6268")
        self.assertEqual(repository.calls, [])

    def test_feature_disabled_returns_503(self) -> None:
        repository = _NaturalLanguageRepository()
        client = build_test_client(repository, NATURAL_LANGUAGE_SEARCH_ENABLED=False)

        response = client.post(
            "/api/v1/search/natural-language",
            json={
                "query": "Find vehicle DL8CBF6268.",
                "run_code": "RUN_20260725_131944",
            },
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "NATURAL_LANGUAGE_SEARCH_DISABLED")

    def test_unknown_camera_returns_safe_validation_error(self) -> None:
        repository = _NaturalLanguageRepository()
        client = build_test_client(repository)

        response = client.post(
            "/api/v1/search/natural-language",
            json={
                "query": "Find vehicles in CAM_999.",
                "run_code": "RUN_20260725_131944",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "UNKNOWN_CAMERA_CODE")
        self.assertFalse(any(name.startswith("search_") for name, _ in repository.calls))


if __name__ == "__main__":
    unittest.main()
