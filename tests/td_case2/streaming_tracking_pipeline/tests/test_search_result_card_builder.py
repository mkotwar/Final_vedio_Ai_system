from __future__ import annotations

import unittest

from tests.td_case2.streaming_tracking_pipeline.search_result_card_builder import (
    build_vehicle_result_card,
    build_vehicle_result_card_package,
)


class SearchResultCardBuilderTests(unittest.TestCase):
    def test_verified_card_uses_plate_normally(self) -> None:
        card = build_vehicle_result_card(_result("verified", "UP81CH4158"), _record("verified", "UP81CH4158"))
        self.assertEqual(card.title, "White Car - UP81CH4158")
        self.assertEqual(card.status_badge, "Verified")
        self.assertEqual(card.plate_label, "UP81CH4158")
        self.assertEqual(card.confidence_label, "0.900")
        self.assertEqual(card.subtitle, "Track 21 · 22.5s-23.0s")

    def test_weak_card_gets_weak_badge(self) -> None:
        card = build_vehicle_result_card(_result("weak", "ABC123"), _record("weak", "ABC123"))
        self.assertEqual(card.status_badge, "Weak OCR")
        self.assertEqual(card.plate_label, "ABC123 (Weak OCR)")

    def test_invalid_and_no_plate_do_not_show_valid_plate(self) -> None:
        invalid = build_vehicle_result_card(_result("invalid", "BAD123"), _record("invalid", None))
        no_plate = build_vehicle_result_card(_result("no_plate_detected", None), _record("no_plate_detected", None))
        self.assertIsNone(invalid.plate_text)
        self.assertEqual(invalid.plate_label, "Invalid plate candidate")
        self.assertEqual(no_plate.plate_label, "No plate detected")

    def test_missing_images_are_warnings_not_drops(self) -> None:
        result = _result("verified", "UP81CH4158")
        result["representative_vehicle_crop_path"] = None
        result["representative_plate_crop_path"] = None
        card = build_vehicle_result_card(result, {})
        self.assertIn("missing_vehicle_image", card.warnings)
        self.assertIn("missing_plate_image", card.warnings)

    def test_package_preserves_top_k_order(self) -> None:
        response = {
            "query": {"raw_query": "white car"},
            "total_matches": 2,
            "runtime_sec": 0.1,
            "warnings": [],
            "results": [_result("verified", "A1", rank=1), _result("weak", "A2", rank=2)],
        }
        package = build_vehicle_result_card_package(response, {"r1": _record("verified", "A1"), "r2": _record("weak", "A2")}, top_k=1)
        self.assertEqual(package.returned_cards, 1)
        self.assertEqual(package.cards[0].rank, 1)


def _result(status: str, plate: str | None, *, rank: int = 1) -> dict[str, object]:
    return {
        "rank": rank,
        "score": 42.0,
        "record_id": f"r{rank}",
        "source_id": "s",
        "track_id": 21,
        "track_generation": 0,
        "object_class": "car",
        "colour": "white",
        "plate_text": plate,
        "plate_status": status,
        "first_seen_sec": 22.5,
        "last_seen_sec": 23.0,
        "representative_vehicle_crop_path": "vehicle.jpg",
        "representative_plate_crop_path": "plate.jpg" if plate else None,
        "matched_filters": ["colour"],
        "matched_tokens": [],
        "warnings": [],
        "score_components": {"colour": 18.0},
    }


def _record(status: str, plate: str | None) -> dict[str, object]:
    return {
        "record_id": "r1",
        "duration_sec": 0.5,
        "plate_status": status,
        "plate_text": plate,
        "plate_confidence": 0.9,
        "representative_vehicle_crop_path": "vehicle.jpg",
        "representative_plate_crop_path": "plate.jpg" if plate else None,
        "warnings": [],
    }


if __name__ == "__main__":
    unittest.main()
