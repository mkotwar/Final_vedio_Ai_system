from __future__ import annotations

import unittest

from tests.td_case2.streaming_tracking_pipeline.search_query_parser import parse_vehicle_search_query
from tests.td_case2.streaming_tracking_pipeline.search_result_ranker import assign_ranks, rank_vehicle_record


class SearchResultRankerTests(unittest.TestCase):
    def test_verified_exact_plate_ranks_above_weak_exact_plate(self) -> None:
        query = parse_vehicle_search_query("UP81CH4158")
        verified = rank_vehicle_record(_record("v", "verified"), query)
        weak = rank_vehicle_record(_record("w", "weak"), query)
        ranked = assign_ranks([weak, verified])
        self.assertEqual(ranked[0].record_id, "v")
        self.assertGreater(ranked[0].score, ranked[1].score)

    def test_score_components_are_preserved(self) -> None:
        query = parse_vehicle_search_query("white car")
        result = rank_vehicle_record(_record("v", "verified"), query)
        self.assertIn("class", result.score_components)
        self.assertIn("colour", result.score_components)
        self.assertIn("evidence_completeness", result.score_components)


def _record(record_id: str, status: str) -> dict[str, object]:
    return {
        "record_id": record_id,
        "source_id": "s",
        "track_id": 1,
        "track_generation": 0,
        "object_class": "car",
        "normalized_colour": "white",
        "plate_status": status,
        "plate_text": "UP81CH4158",
        "first_seen_sec": 1.0,
        "last_seen_sec": 2.0,
        "representative_vehicle_crop_path": "vehicle.jpg",
        "representative_plate_crop_path": "plate.jpg",
        "searchable_tokens": ["vehicle", "car", "white", "up81ch4158"],
        "search_text": "vehicle car white up81ch4158",
        "warnings": [],
    }


if __name__ == "__main__":
    unittest.main()
