from __future__ import annotations

import unittest

from ..api.services.natural_language_search_service import (
    NaturalLanguageSearchContext,
    NaturalLanguageSearchService,
)
from ..api.settings import ApiSettings


class _Repo:
    pass


class _SearchService:
    def search(self, request):  # pragma: no cover - not used here
        raise AssertionError("search should not be called")


class NaturalLanguageFallbackParserTests(unittest.TestCase):
    def setUp(self) -> None:
        settings = ApiSettings(SUPABASE_URL="https://example.supabase.co", SUPABASE_SERVICE_ROLE_KEY="secret")
        self.service = NaturalLanguageSearchService(_Repo(), _SearchService(), settings=settings, parser_provider=None)
        self.context = NaturalLanguageSearchContext(
            run_code="RUN_20260725_131944",
            result_scope=None,
            default_time_tolerance_minutes=15,
        )

    def test_exact_plate_is_parsed(self) -> None:
        intent = self.service._fallback_parse("Find vehicle DL8CBF6268.", self.context)
        self.assertEqual(intent.plate, "DL8CBF6268")
        self.assertEqual(intent.plate_match_type, "EXACT")

    def test_plate_ending_is_parsed(self) -> None:
        intent = self.service._fallback_parse("Find the grey car with plate ending in 6268.", self.context)
        self.assertEqual(intent.plate, "6268")
        self.assertEqual(intent.plate_match_type, "ENDS_WITH")
        self.assertEqual(intent.vehicle_class, "CAR")
        self.assertEqual(intent.colour, "GREY")

    def test_plate_contains_is_parsed(self) -> None:
        intent = self.service._fallback_parse("Find plate contains CBF.", self.context)
        self.assertEqual(intent.plate, "CBF")
        self.assertEqual(intent.plate_match_type, "CONTAINS")

    def test_camera_codes_and_multi_camera_phrase_are_parsed(self) -> None:
        intent = self.service._fallback_parse("Show the same vehicle seen in CAM_001 and CAM_002.", self.context)
        self.assertEqual(intent.camera_codes, ["CAM_001", "CAM_002"])
        self.assertTrue(intent.multi_camera_only)
        self.assertEqual(intent.result_scope, "GLOBAL_VEHICLES")

    def test_verified_plate_phrase_is_parsed(self) -> None:
        intent = self.service._fallback_parse("Show vehicles with verified plates.", self.context)
        self.assertTrue(intent.verified_plate_only)

    def test_around_time_sets_target_time_and_tolerance(self) -> None:
        intent = self.service._fallback_parse("Find cars around 2 PM.", self.context)
        self.assertEqual(intent.target_time.isoformat(), "14:00:00")
        self.assertEqual(intent.time_tolerance_minutes, 15)

    def test_between_after_and_before_times_are_parsed(self) -> None:
        between = self.service._fallback_parse("Find buses between 2 PM and 3 PM.", self.context)
        self.assertEqual(between.start_time.isoformat(), "14:00:00")
        self.assertEqual(between.end_time.isoformat(), "15:00:00")

        after = self.service._fallback_parse("Show all trucks after 1:30 PM.", self.context)
        self.assertEqual(after.start_time.isoformat(), "13:30:00")

        before = self.service._fallback_parse("Show cars before 3 PM.", self.context)
        self.assertEqual(before.end_time.isoformat(), "15:00:00")


if __name__ == "__main__":
    unittest.main()
