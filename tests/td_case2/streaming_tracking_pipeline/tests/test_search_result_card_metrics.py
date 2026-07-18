from __future__ import annotations

import unittest

from tests.td_case2.streaming_tracking_pipeline.search_result_card_metrics import build_result_card_metrics
from tests.td_case2.streaming_tracking_pipeline.search_result_card_schemas import VehicleResultCard, VehicleResultCardPackage


class SearchResultCardMetricsTests(unittest.TestCase):
    def test_counts_cards_images_status_and_duplicates(self) -> None:
        package = VehicleResultCardPackage(
            raw_query="q",
            parsed_query={},
            total_matches=2,
            returned_cards=2,
            cards=[_card("r1", "verified", "car", "white", True, True), _card("r1", "no_plate_detected", "truck", None, False, False)],
            runtime_sec=0.1,
        )
        metrics = build_result_card_metrics([package], packaging_runtime=0.25)
        self.assertEqual(metrics["queries_packaged"], 1)
        self.assertEqual(metrics["cards_created"], 2)
        self.assertEqual(metrics["cards_with_vehicle_image"], 1)
        self.assertEqual(metrics["cards_missing_plate_image"], 1)
        self.assertEqual(metrics["verified_plate_cards"], 1)
        self.assertEqual(metrics["no_plate_cards"], 1)
        self.assertEqual(metrics["duplicate_card_ids"], ["q:r1"])


def _card(record_id: str, status: str, object_class: str, colour: str | None, vehicle: bool, plate: bool) -> VehicleResultCard:
    return VehicleResultCard(
        rank=1,
        record_id=record_id,
        source_id="s",
        track_id=1,
        track_generation=0,
        title="Title",
        subtitle="Sub",
        time_label="1.0s-2.0s",
        plate_label="Plate",
        colour_label=colour or "Unknown",
        status_badge=status,
        confidence_label="0.900",
        object_class=object_class,
        colour=colour,
        plate_text="ABC" if status in {"verified", "weak"} else None,
        plate_status=status,
        plate_confidence=0.9,
        first_seen_sec=1.0,
        last_seen_sec=2.0,
        duration_sec=1.0,
        thumbnail_path="vehicle.jpg" if vehicle else None,
        secondary_image_path="plate.jpg" if plate else None,
        search_score=1.0,
    )


if __name__ == "__main__":
    unittest.main()
