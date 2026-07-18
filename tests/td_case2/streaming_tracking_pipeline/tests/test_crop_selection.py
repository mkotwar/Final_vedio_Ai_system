from __future__ import annotations

import copy
import unittest

from tests.td_case2.streaming_tracking_pipeline.config import BestCropScoreConfig, BestCropSelectionConfig
from tests.td_case2.streaming_tracking_pipeline.crop_artifacts import CompletedTrackCropBundle
from tests.td_case2.streaming_tracking_pipeline.crop_selection import FinalBestCropSelector
from tests.td_case2.streaming_tracking_pipeline.schemas import BoundingBox, CropCandidate, CropQualityMetrics
from tests.td_case2.streaming_tracking_pipeline.serialization import dataclass_to_dict


SOURCE = "selection_test_source"


def candidate(
    frame_index: int,
    *,
    track_id: int = 1,
    generation: int = 0,
    confidence: float = 0.9,
    brightness: float | None = 0.5,
    sharpness: float | None = 3000.0,
    contrast: float | None = 40.0,
    complete: bool = True,
    edge: bool = False,
    path: str | None = "crop.jpg",
    bbox: BoundingBox | None = None,
) -> CropCandidate:
    bbox = bbox or BoundingBox(10 + frame_index, 10, 50 + frame_index, 42)
    return CropCandidate(
        source_id=SOURCE,
        track_id=track_id,
        track_generation=generation,
        source_track_id=f"raw_{track_id}",
        frame_index=frame_index,
        timestamp_sec=frame_index / 4.0,
        bbox=bbox,
        crop_bbox=bbox,
        full_frame_path=None,
        vehicle_crop_path=path,
        quality=CropQualityMetrics(
            detection_confidence=confidence,
            bbox_area_ratio=0.02,
            sharpness=sharpness,
            brightness=brightness,
            contrast=contrast,
            crop_width=40,
            crop_height=32,
            crop_completeness=1.0 if complete else 0.8,
            edge_touching=edge,
            padding_clipped=not complete,
            preliminary_score=confidence,
        ),
        class_name="car",
        detection_confidence=confidence,
        preliminary_rank_score=confidence,
        retention_reason="test",
    )


def bundle(candidates: list[CropCandidate], *, observation_count: int = 6, track_id: int = 1, generation: int = 0) -> CompletedTrackCropBundle:
    return CompletedTrackCropBundle(
        source_id=SOURCE,
        track_id=track_id,
        track_generation=generation,
        source_track_id=f"raw_{track_id}",
        completion_reason="video_ended",
        lifecycle_status="completed",
        observation_count=observation_count,
        retained_candidate_count=len(candidates),
        candidates=candidates,
        metadata={"dominant_class": "car"},
    )


class FinalBestCropSelectorTest(unittest.TestCase):
    def test_final_score_is_bounded_and_reports_missing_metrics(self) -> None:
        selector = FinalBestCropSelector()
        score = selector.score_candidate(candidate(1, brightness=None, sharpness=None, contrast=None))

        self.assertGreaterEqual(score.final_score, 0.0)
        self.assertLessEqual(score.final_score, 1.0)
        self.assertIn("brightness", score.missing_metric_names)
        self.assertIn("sharpness", score.missing_metric_names)

    def test_primary_selection_uses_thresholds_and_count_limit(self) -> None:
        selector = FinalBestCropSelector(BestCropSelectionConfig(primary_crop_count=2, minimum_frame_separation=2))
        result = selector.select(bundle([candidate(0), candidate(4, confidence=0.85), candidate(8, confidence=0.8)]))

        self.assertEqual(result.selection_status, "primary_selected")
        self.assertEqual(len(result.primary_crops), 2)
        self.assertEqual([crop.rank for crop in result.primary_crops], [1, 2])

    def test_short_track_gets_fallback_only(self) -> None:
        selector = FinalBestCropSelector()
        result = selector.select(bundle([candidate(0)], observation_count=1))

        self.assertEqual(result.selection_status, "fallback_only")
        self.assertIsNotNone(result.fallback_crop)
        self.assertIn("track_observation_count_below_primary_min", result.rejection_reason_counts)

    def test_no_valid_crop_when_path_required_and_missing(self) -> None:
        selector = FinalBestCropSelector()
        result = selector.select(bundle([candidate(0, path=None)], observation_count=1))

        self.assertEqual(result.selection_status, "no_valid_crop")
        self.assertIn("missing_crop_path", result.rejection_reason_counts)
        self.assertIn("fallback_missing_crop_path", result.rejection_reason_counts)

    def test_edge_and_incomplete_policy_reject_primary_but_allows_fallback(self) -> None:
        selector = FinalBestCropSelector()
        edge_result = selector.select(bundle([candidate(0, edge=True)], observation_count=4))
        incomplete_result = selector.select(bundle([candidate(0, complete=False)], observation_count=4))

        self.assertEqual(edge_result.selection_status, "fallback_only")
        self.assertEqual(incomplete_result.selection_status, "fallback_only")
        self.assertIn("edge_touching", edge_result.rejection_reason_counts)
        self.assertIn("crop_incomplete", incomplete_result.rejection_reason_counts)

    def test_brightness_sharpness_and_contrast_primary_thresholds(self) -> None:
        config = BestCropSelectionConfig(
            minimum_sharpness_for_primary=100.0,
            minimum_brightness_for_primary=0.2,
            maximum_brightness_for_primary=0.8,
            minimum_contrast_for_primary=10.0,
        )
        selector = FinalBestCropSelector(config)

        self.assertIn("brightness_out_of_range", selector.select(bundle([candidate(0, brightness=0.05)])).rejection_reason_counts)
        self.assertIn("brightness_out_of_range", selector.select(bundle([candidate(0, brightness=0.95)])).rejection_reason_counts)
        self.assertIn("sharpness_below_threshold", selector.select(bundle([candidate(0, sharpness=1.0)])).rejection_reason_counts)
        self.assertIn("contrast_below_threshold", selector.select(bundle([candidate(0, contrast=1.0)])).rejection_reason_counts)

    def test_temporal_and_frame_separation(self) -> None:
        selector = FinalBestCropSelector(BestCropSelectionConfig(primary_crop_count=3, minimum_frame_separation=3, minimum_temporal_separation_sec=1.0))
        result = selector.select(bundle([candidate(0), candidate(1), candidate(2), candidate(8)]))

        self.assertLess(len(result.primary_crops), 3)
        self.assertTrue(
            "insufficient_frame_separation" in result.rejection_reason_counts
            or "insufficient_time_separation" in result.rejection_reason_counts
        )

    def test_visual_duplicate_rejection(self) -> None:
        bbox = BoundingBox(10, 10, 50, 42)
        selector = FinalBestCropSelector(BestCropSelectionConfig(primary_crop_count=2, primary_selection_policy="quality_with_visual_diversity", maximum_bbox_overlap_similarity=0.5))
        result = selector.select(bundle([candidate(0, bbox=bbox), candidate(8, bbox=bbox)]))

        self.assertEqual(len(result.primary_crops), 1)
        self.assertIn("duplicate_visual_candidate", result.rejection_reason_counts)

    def test_fallback_is_distinct_from_primary_when_primary_short(self) -> None:
        selector = FinalBestCropSelector(BestCropSelectionConfig(primary_crop_count=3, minimum_frame_separation=6))
        result = selector.select(bundle([candidate(0), candidate(4), candidate(12)]))

        selected_frames = {crop.frame_index for crop in result.primary_crops}
        if result.fallback_crop is not None:
            self.assertNotIn(result.fallback_crop.frame_index, selected_frames)

    def test_disabled_fallback_still_emits_result(self) -> None:
        selector = FinalBestCropSelector(BestCropSelectionConfig(keep_fallback_crop=False))
        result = selector.select(bundle([candidate(0)], observation_count=1))

        self.assertEqual(result.selection_status, "no_valid_crop")
        self.assertIsNone(result.fallback_crop)

    def test_generation_identity_validation(self) -> None:
        selector = FinalBestCropSelector()
        with self.assertRaises(ValueError):
            selector.select(bundle([candidate(0, generation=1)], generation=0))

    def test_duplicate_candidate_removed_and_empty_bundle(self) -> None:
        selector = FinalBestCropSelector()
        dup = candidate(0)
        duplicate_result = selector.select(bundle([dup, dup]))
        empty_result = selector.select(bundle([]))

        self.assertEqual(duplicate_result.metadata["duplicate_candidates_removed"], 1)
        self.assertEqual(empty_result.selection_status, "no_valid_crop")

    def test_deterministic_and_no_input_mutation(self) -> None:
        selector = FinalBestCropSelector()
        source_bundle = bundle([candidate(0), candidate(4), candidate(8)])
        before = copy.deepcopy(dataclass_to_dict(source_bundle))

        left = selector.select(source_bundle).to_dict()
        right = selector.select(source_bundle).to_dict()

        self.assertEqual(left, right)
        self.assertEqual(before, dataclass_to_dict(source_bundle))

    def test_selected_crop_jobs_include_step7_fields(self) -> None:
        selector = FinalBestCropSelector()
        result = selector.select(bundle([candidate(0), candidate(4)]))
        jobs = result.to_crop_jobs()

        self.assertGreater(len(jobs), 0)
        self.assertEqual(jobs[0].crop_role, "primary")
        self.assertEqual(jobs[0].lifecycle_completion_reason, "video_ended")
        self.assertEqual(jobs[0].object_class, "car")


if __name__ == "__main__":
    unittest.main()
