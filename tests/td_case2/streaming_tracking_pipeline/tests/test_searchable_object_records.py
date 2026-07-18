from __future__ import annotations

import unittest

from tests.td_case2.streaming_tracking_pipeline.searchable_object_records import (
    build_record_id,
    build_searchable_object_record,
    build_searchable_person_record,
    build_searchable_vehicle_record,
    run_validation_queries,
)


class SearchableObjectRecordTests(unittest.TestCase):
    def test_record_id_is_generation_aware(self) -> None:
        self.assertNotEqual(build_record_id("s", 7, 0), build_record_id("s", 7, 1))

    def test_verified_plate_is_searchable(self) -> None:
        record = build_searchable_vehicle_record(
            _lifecycle(),
            video_path="video.mp4",
            selected_crop_set=_selected(),
            final_anpr=_final("verified", "UP81CH4158", "white"),
        )
        self.assertEqual(record.plate_text, "UP81CH4158")
        self.assertIn("verified_plate", record.searchable_tokens)
        self.assertIn("up81ch4158", record.searchable_tokens)
        self.assertIn("state_up", record.searchable_tokens)

    def test_weak_plate_is_tagged(self) -> None:
        record = build_searchable_vehicle_record(
            _lifecycle(),
            video_path=None,
            final_anpr=_final("weak", "G39RFZH71", "yellow"),
            include_weak_plate_text=True,
        )
        self.assertEqual(record.plate_text, "G39RFZH71")
        self.assertIn("weak_plate", record.searchable_tokens)

    def test_invalid_plate_is_not_searchable_plate(self) -> None:
        record = build_searchable_vehicle_record(_lifecycle(), video_path=None, final_anpr=_final("invalid", "K01", "red"))
        self.assertIsNone(record.plate_text)
        self.assertNotIn("k01", record.searchable_tokens)

    def test_no_plate_vehicle_record_is_retained(self) -> None:
        record = build_searchable_vehicle_record(_lifecycle(), video_path=None, final_anpr=_final("no_plate_detected", None, "green"))
        self.assertIsNone(record.plate_text)
        self.assertIn("no_plate", record.searchable_tokens)

    def test_person_record_is_written_without_plate_or_colour(self) -> None:
        record = build_searchable_person_record(
            _lifecycle(last_class_name="person"),
            video_path="video.mp4",
            selected_crop_set=_selected(),
        )

        self.assertEqual(record.object_group, "person")
        self.assertEqual(record.object_type, "person")
        self.assertIsNone(record.plate_text)
        self.assertEqual(record.plate_status, "not_applicable")
        self.assertIsNone(record.normalized_colour)
        self.assertIn("person", record.searchable_tokens)
        self.assertTrue(record.metadata["anpr_bypassed"])
        self.assertTrue(record.metadata["colour_bypassed"])

    def test_person_record_includes_clothing_colour_tokens(self) -> None:
        record = build_searchable_person_record(
            _lifecycle(last_class_name="person"),
            video_path="video.mp4",
            selected_crop_set=_selected(),
            person_clothing={
                "upper_clothing_color": "white",
                "lower_clothing_color": "blue",
                "dominant_clothing_color": "white",
                "clothing_color_confidence": 0.8,
                "clothing_color_status": "detected",
            },
        )

        self.assertEqual(record.upper_clothing_color, "white")
        self.assertEqual(record.lower_clothing_color, "blue")
        self.assertEqual(record.dominant_clothing_color, "white")
        self.assertIn("white_shirt", record.searchable_tokens)
        self.assertIn("blue_lower", record.searchable_tokens)

    def test_full_frame_matches_representative_crop_path(self) -> None:
        record = build_searchable_vehicle_record(
            _lifecycle(),
            video_path="video.mp4",
            selected_crop_set={
                "primary_crops": [
                    {"vehicle_crop_path": "other.jpg", "full_frame_path": "frames/other.jpg", "frame_index": 9, "timestamp_sec": 0.9},
                    {"vehicle_crop_path": "vehicle.jpg", "full_frame_path": "frames/vehicle.jpg", "frame_index": 10, "timestamp_sec": 1.0},
                ],
                "selection_status": "primary_selected",
            },
            final_anpr=_final("verified", "UP81CH4158", "white"),
        )

        self.assertEqual(record.full_frame_path, "frames/vehicle.jpg")
        self.assertEqual(record.selected_frame_index, 10)

    def test_generic_builder_routes_person_away_from_vehicle_anpr_record(self) -> None:
        record = build_searchable_object_record(_lifecycle(last_class_name="person"), video_path=None, final_anpr=None)

        self.assertEqual(record.object_group, "person")
        self.assertEqual(record.object_class, "person")

    def test_validation_queries(self) -> None:
        records = [
            build_searchable_vehicle_record(_lifecycle(), video_path=None, final_anpr=_final("verified", "UP81CH4158", "white")),
            build_searchable_vehicle_record(_lifecycle(track_id=2), video_path=None, final_anpr=_final("weak", "ABC123", "red")),
        ]
        result = run_validation_queries(records)
        self.assertEqual(result["white car"]["count"], 1)
        self.assertEqual(result["verified plates"]["count"], 1)
        self.assertEqual(result["weak OCR"]["count"], 1)


def _lifecycle(track_id: int = 1, last_class_name: str = "car") -> dict[str, object]:
    return {
        "source_id": "s",
        "track_id": track_id,
        "track_generation": 0,
        "last_class_name": last_class_name,
        "first_seen_frame": 10,
        "last_seen_frame": 20,
        "first_seen_sec": 1.0,
        "last_seen_sec": 2.0,
        "status": "completed",
        "completion_reason": "lost_buffer_expired",
    }


def _selected() -> dict[str, object]:
    return {
        "primary_crops": [{"vehicle_crop_path": "primary.jpg"}],
        "fallback_crop": {"vehicle_crop_path": "fallback.jpg"},
        "selection_status": "primary_selected",
    }


def _final(status: str, plate: str | None, colour: str | None) -> dict[str, object]:
    return {
        "source_id": "s",
        "track_id": 1,
        "track_generation": 0,
        "final_plate_text": plate,
        "plate_status": status,
        "confidence": 0.9,
        "support_count": 1,
        "normalized_colour": colour,
        "raw_colour": colour,
        "representative_frame_index": 10,
        "representative_timestamp_sec": 1.0,
        "representative_vehicle_crop_path": "vehicle.jpg",
        "representative_plate_crop_path": "plate.jpg" if plate else None,
    }


if __name__ == "__main__":
    unittest.main()
