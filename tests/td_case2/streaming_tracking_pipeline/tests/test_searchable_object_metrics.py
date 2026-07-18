from __future__ import annotations

import unittest

from tests.td_case2.streaming_tracking_pipeline.searchable_object_metrics import build_searchable_object_metrics
from tests.td_case2.streaming_tracking_pipeline.searchable_object_schemas import SearchableVehicleRecord


class SearchableObjectMetricsTests(unittest.TestCase):
    def test_metrics_count_status_colour_and_warnings(self) -> None:
        records = [
            _record("r1", "verified", "white"),
            _record("r2", "weak", None, ["missing_colour"]),
            _record("r2", "no_plate_detected", "red"),
        ]
        metrics = build_searchable_object_metrics(records, join_failures=["x"], missing_input_artifacts=["y"])
        self.assertEqual(metrics["vehicle_records_created"], 3)
        self.assertEqual(metrics["verified_plate_records"], 1)
        self.assertEqual(metrics["weak_plate_records"], 1)
        self.assertEqual(metrics["no_plate_records"], 1)
        self.assertEqual(metrics["records_with_colour"], 2)
        self.assertEqual(metrics["records_without_colour"], 1)
        self.assertEqual(metrics["duplicate_record_ids"], ["r2"])
        self.assertEqual(metrics["join_failures"], ["x"])
        self.assertEqual(metrics["missing_input_artifacts"], ["y"])


def _record(record_id: str, status: str, colour: str | None, warnings: list[str] | None = None) -> SearchableVehicleRecord:
    return SearchableVehicleRecord(
        record_id=record_id,
        source_id="s",
        video_path=None,
        track_id=1,
        track_generation=0,
        object_class="car",
        first_frame_index=1,
        last_frame_index=2,
        first_seen_sec=0.1,
        last_seen_sec=0.2,
        duration_sec=0.1,
        plate_text="ABC123" if status in {"verified", "weak"} else None,
        plate_status=status,
        plate_confidence=0.5,
        plate_support_count=1,
        normalized_colour=colour,
        raw_colour=colour,
        representative_frame_index=1,
        representative_timestamp_sec=0.1,
        representative_vehicle_crop_path="vehicle.jpg",
        representative_plate_crop_path="plate.jpg" if status in {"verified", "weak"} else None,
        warnings=warnings or [],
    )


if __name__ == "__main__":
    unittest.main()
