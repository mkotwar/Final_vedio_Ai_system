from __future__ import annotations

import unittest

from ..api.errors import ApiError
from ..api.search_models import NaturalLanguageSearchRequest
from ..api.services.natural_language_search_service import NaturalLanguageSearchService, ProviderUnavailableError
from ..api.settings import ApiSettings
from ..persistence.api_read_repository import Page


class _Repository:
    def __init__(self) -> None:
        self.camera_calls = 0
        self.run_calls = 0

    def list_run_cameras(self, *, run_code: str, page: int, page_size: int):
        self.camera_calls += 1
        return {"id": "run-1"}, Page(
            items=[
                {"camera_code": "CAM_001"},
                {"camera_code": "CAM_002"},
            ],
            page=page,
            page_size=page_size,
            total=2,
        )

    def find_run_by_code(self, run_code: str):
        self.run_calls += 1
        return {
            "id": "run-1",
            "run_code": run_code,
            "started_at": "2026-07-25T13:19:44+05:30",
        }


class _StructuredSearchService:
    def __init__(self) -> None:
        self.calls = []

    def search(self, request):
        self.calls.append(request)
        return {
            "filters": request.applied_filters(),
            "pagination": {
                "limit": request.limit,
                "offset": request.offset,
                "returned": 1,
                "total": 1,
                "has_more": False,
            },
            "results": [
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
                    "match_reasons": ["exact verified plate"],
                    "relevance_score": 127.5,
                }
            ],
        }


class _Provider:
    provider_name = "gemini"
    model_name = "gemini-2.5-flash"

    def __init__(self, payload=None, *, error: Exception | None = None) -> None:
        self.payload = payload or {}
        self.error = error

    def parse_vehicle_search(self, query, context):
        if self.error is not None:
            raise self.error
        return self.payload


class NaturalLanguageSearchServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = _Repository()
        self.structured = _StructuredSearchService()
        self.settings = ApiSettings(SUPABASE_URL="https://example.supabase.co", SUPABASE_SERVICE_ROLE_KEY="secret")

    def test_explicit_run_and_scope_override_provider_output(self) -> None:
        provider = _Provider(
            {
                "result_scope": "LOCAL_TRACKS",
                "vehicle_class": "CAR",
                "colour": "GREY",
                "plate": "6268",
                "plate_match_type": "ENDS_WITH",
            }
        )
        service = NaturalLanguageSearchService(self.repository, self.structured, settings=self.settings, parser_provider=provider)

        response = service.search(
            NaturalLanguageSearchRequest(
                query="Find the grey car with plate ending in 6268.",
                run_code="RUN_20260725_131944",
                result_scope="GLOBAL_VEHICLES",
            )
        )

        self.assertEqual(response.interpreted_filters["run_code"], "RUN_20260725_131944")
        self.assertEqual(response.interpreted_filters["result_scope"], "GLOBAL_VEHICLES")
        self.assertEqual(len(self.structured.calls), 1)

    def test_invalid_provider_output_uses_fallback(self) -> None:
        provider = _Provider({"sql": "select * from vehicle_track"})
        service = NaturalLanguageSearchService(self.repository, self.structured, settings=self.settings, parser_provider=provider)

        response = service.search(
            NaturalLanguageSearchRequest(
                query="Find vehicle DL8CBF6268.",
                run_code="RUN_20260725_131944",
            )
        )

        self.assertTrue(response.parser.fallback_used)
        self.assertEqual(response.results[0].global_vehicle_code, "GVO:RUN_20260725_131944:FA3FCF9E3ABC")

    def test_provider_timeout_uses_fallback(self) -> None:
        provider = _Provider(error=TimeoutError("timed out"))
        service = NaturalLanguageSearchService(self.repository, self.structured, settings=self.settings, parser_provider=provider)

        response = service.search(
            NaturalLanguageSearchRequest(
                query="Find the grey car with plate ending in 6268.",
                run_code="RUN_20260725_131944",
            )
        )

        self.assertTrue(response.parser.fallback_used)
        self.assertEqual(len(self.structured.calls), 1)

    def test_clarification_prevents_database_execution(self) -> None:
        service = NaturalLanguageSearchService(self.repository, self.structured, settings=self.settings, parser_provider=None)

        response = service.search(
            NaturalLanguageSearchRequest(
                query="Find cars around 2 PM.",
            )
        )

        self.assertTrue(response.clarification_required)
        self.assertEqual(len(self.structured.calls), 0)

    def test_unknown_camera_code_is_rejected(self) -> None:
        service = NaturalLanguageSearchService(self.repository, self.structured, settings=self.settings, parser_provider=None)

        with self.assertRaises(ApiError) as exc:
            service.search(
                NaturalLanguageSearchRequest(
                    query="Find vehicles in CAM_999.",
                    run_code="RUN_20260725_131944",
                )
            )
        self.assertEqual(exc.exception.code, "UNKNOWN_CAMERA_CODE")
        self.assertEqual(len(self.structured.calls), 0)

    def test_time_query_uses_selected_run_date_without_inventing_today(self) -> None:
        service = NaturalLanguageSearchService(self.repository, self.structured, settings=self.settings, parser_provider=None)

        response = service.search(
            NaturalLanguageSearchRequest(
                query="Find cars around 2 PM.",
                run_code="RUN_20260725_131944",
            )
        )

        self.assertEqual(response.interpreted_filters["date"], "2026-07-25")
        self.assertEqual(response.interpreted_filters["start_time"], "13:45:00")
        self.assertEqual(response.interpreted_filters["end_time"], "14:15:00")
        self.assertEqual(len(self.structured.calls), 1)

    def test_pagination_is_preserved(self) -> None:
        service = NaturalLanguageSearchService(self.repository, self.structured, settings=self.settings, parser_provider=None)

        response = service.search(
            NaturalLanguageSearchRequest(
                query="Find vehicle DL8CBF6268.",
                run_code="RUN_20260725_131944",
                limit=10,
                offset=5,
            )
        )

        self.assertEqual(response.pagination.limit, 10)
        self.assertEqual(response.pagination.offset, 5)
        self.assertEqual(self.structured.calls[0].limit, 10)
        self.assertEqual(self.structured.calls[0].offset, 5)


if __name__ == "__main__":
    unittest.main()
