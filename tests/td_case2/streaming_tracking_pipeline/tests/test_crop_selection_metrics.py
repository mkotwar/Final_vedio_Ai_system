from __future__ import annotations

import unittest

from tests.td_case2.streaming_tracking_pipeline.config import BestCropSelectionConfig
from tests.td_case2.streaming_tracking_pipeline.crop_selection import FinalBestCropSelector
from tests.td_case2.streaming_tracking_pipeline.crop_selection_metrics import BestCropSelectionMetricsAccumulator, build_selection_summary
from tests.td_case2.streaming_tracking_pipeline.tests.test_crop_selection import bundle, candidate


class CropSelectionMetricsTest(unittest.TestCase):
    def test_counts_statuses_scores_and_rejections(self) -> None:
        selector = FinalBestCropSelector(BestCropSelectionConfig(primary_crop_count=2))
        results = [
            selector.select(bundle([candidate(0), candidate(4), candidate(8)])),
            selector.select(bundle([candidate(0)], observation_count=1)),
            selector.select(bundle([candidate(0, path=None)], observation_count=1)),
        ]

        summary = build_selection_summary(results, primary_target=2)

        self.assertEqual(summary["completed_track_bundles_processed"], 3)
        self.assertEqual(summary["tracks_with_primary_crops"], 1)
        self.assertEqual(summary["tracks_with_fallback_only"], 1)
        self.assertEqual(summary["tracks_with_no_valid_crop"], 1)
        self.assertGreater(summary["total_primary_crops_selected"], 0)
        self.assertGreater(summary["total_fallback_crops_selected"], 0)
        self.assertIn("missing_crop_path", summary["rejection_reason_counts"])
        self.assertIn("primary_selected", summary["selection_status_counts"])
        self.assertGreaterEqual(summary["average_selected_primary_score"], 0.0)

    def test_empty_input(self) -> None:
        summary = build_selection_summary([], primary_target=3)

        self.assertEqual(summary["completed_track_bundles_processed"], 0)
        self.assertEqual(summary["average_primary_crops_per_track"], 0.0)
        self.assertEqual(summary["median_primary_crops_per_track"], 0.0)

    def test_missing_metrics_and_component_averages(self) -> None:
        selector = FinalBestCropSelector()
        result = selector.select(bundle([candidate(0, brightness=None, sharpness=None, contrast=None)]))
        accumulator = BestCropSelectionMetricsAccumulator()
        accumulator.update(result, primary_target=3)
        summary = accumulator.to_dict()

        self.assertIn("brightness", summary["missing_metric_counts"])
        self.assertIn("confidence_component", summary["score_component_averages"])

    def test_grouping_fields(self) -> None:
        selector = FinalBestCropSelector()
        result = selector.select(bundle([candidate(0)], observation_count=1))
        summary = build_selection_summary([result], primary_target=3)

        self.assertIn("video_ended", summary["by_completion_reason"])
        self.assertIn("car", summary["by_dominant_class"])
        self.assertIn("1", summary["by_observation_count_bucket"])


if __name__ == "__main__":
    unittest.main()
