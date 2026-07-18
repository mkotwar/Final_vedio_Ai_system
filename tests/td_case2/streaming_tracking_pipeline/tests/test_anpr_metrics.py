from __future__ import annotations

import unittest

from tests.td_case2.streaming_tracking_pipeline.anpr_metrics import build_step7_metrics, confidence_bucket
from tests.td_case2.streaming_tracking_pipeline.anpr_schemas import PlateDetectionCandidate, TrackAnprColourResult


class AnprMetricTests(unittest.TestCase):
    def test_confidence_bucket(self) -> None:
        self.assertEqual(confidence_bucket(0.81), "0.80-1.00")
        self.assertEqual(confidence_bucket(0.19), "0.00-0.19")

    def test_builds_breakdowns(self) -> None:
        candidate = PlateDetectionCandidate("cam", 1, 0, "primary", 1, 1, "crop.jpg", 1, 0.91, (1, 1, 10, 8), (0, 0, 11, 9), "plate.jpg")
        result = TrackAnprColourResult(
            "cam",
            1,
            0,
            None,
            "car",
            "done",
            "success",
            plate_candidates=[candidate],
            raw_plate_texts=["MH12AB1234"],
            normalized_plate_texts=["MH12AB1234"],
            normalized_colour="white",
        )
        metrics = build_step7_metrics([result])
        self.assertEqual(metrics["tracks_processed"], 1)
        self.assertEqual(metrics["by_crop_role"]["primary"]["success"], 1)
        self.assertEqual(metrics["tracks_with_raw_plate_text"], 1)


if __name__ == "__main__":
    unittest.main()
