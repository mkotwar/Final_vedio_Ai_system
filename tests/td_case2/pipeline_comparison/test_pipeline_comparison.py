from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from comparison_metrics import (
        build_class_count_comparison,
        build_config_differences,
        build_crop_quality_comparison,
        build_detector_usage_comparison,
        build_manual_review_summary,
        build_normalized_metrics,
        build_runtime_comparison,
        build_tracking_count_comparison,
        normalize_per_minute,
        safe_divide,
    )
    from comparison_visual_review import generate_visual_review_timestamps
else:
    from .comparison_metrics import (
        build_class_count_comparison,
        build_config_differences,
        build_crop_quality_comparison,
        build_detector_usage_comparison,
        build_manual_review_summary,
        build_normalized_metrics,
        build_runtime_comparison,
        build_tracking_count_comparison,
        normalize_per_minute,
        safe_divide,
    )
    from .comparison_visual_review import generate_visual_review_timestamps


class PipelineComparisonTests(unittest.TestCase):
    def _td_case2_metrics(self) -> dict:
        return {
            "timings": {
                "tracking_runtime_seconds": 10.0,
                "step05_runtime_seconds": 2.0,
                "step06_runtime_seconds": 3.0,
                "step07_runtime_seconds": 1.0,
            },
            "known_stage_runtime_seconds": 16.0,
            "processed_frames": 100,
            "source_duration_seconds": 50.0,
            "source_frame_count": 1500,
            "yolo_calls": 100,
            "raw_tracks": 20,
            "confirmed_tracks": 15,
            "reconciled_objects": None,
            "persons": 5,
            "vehicles": 10,
            "short_tracks_under_0_5_seconds": 4,
            "short_tracks_under_1_0_second": 8,
            "track_duration_stats": {"avg": 2.0, "median": 1.5, "max": 9.0},
            "track_quality_counts": {"good": 7, "fragmented": 5},
            "tracks_with_primary_crop": 9,
            "tracks_with_three_representative_crops": 4,
            "objects_with_full_scene_frame": 9,
            "fallback_crop_count": 2,
            "invalid_crop_candidates": 0,
            "crop_failures": 1,
            "yolo_selected_crops": 9,
            "kcf_selected_crops": 0,
            "plate_candidates": 6,
            "class_counts": {"person": 5, "car": 7},
            "warnings": [],
            "failures": [],
            "best_frames": {"tracks": []},
            "tracker_name": "td_case2_step04b_tracker",
        }

    def _hybrid_metrics(self) -> dict:
        return {
            "tracking_report": {
                "video_metadata": {"duration_seconds": 50.0, "frame_count": 1500},
                "processing_speed": {"total_runtime_seconds": 12.0},
                "yolo_call_count": 40,
                "raw_track_id_count": 25,
                "confirmed_raw_track_count": 18,
                "track_durations": {"mean": 1.8, "median": 1.2, "max": 8.5},
                "warnings": [],
            },
            "frame_metrics": {"frames": [{"timestamp_seconds": 1.0, "tracks": []}] * 100},
            "quality_report": {
                "quality_breakdown": {"high": 2, "medium": 5, "low": 10},
                "tracks": [
                    {"duration_seconds": 0.4, "frozen_kcf_detected": True, "boundary_stuck_detected": False},
                    {"duration_seconds": 1.4, "frozen_kcf_detected": False, "boundary_stuck_detected": True},
                ],
            },
            "reconciled_tracks": {"tracks": [{"local_object_id": 1}, {"local_object_id": 2}]},
            "merge_events": {"events": [{"local_object_id": 1}]},
            "candidates": {"candidates": [{"from_track_id": 1, "to_track_id": 2}, {"from_track_id": 3, "to_track_id": 4}]},
            "representative_report": {
                "valid_primary_crops": 12,
                "objects_with_alternatives": 7,
                "saved_full_frame_count": 12,
                "fallback_crops": 1,
                "valid_yolo_primary_crops": 10,
                "valid_kcf_primary_crops": 2,
                "objects_with_plate_candidate": 8,
                "saved_crop_count": 20,
                "runtime_seconds": 4.0,
            },
            "invalid_crop_candidates": {"candidates": [1, 2]},
            "crop_failures": {"failures": [1]},
            "package_report": {"person_packages": 6, "vehicle_packages": 14},
            "packages": {"packages": [{"final_class": "car"}, {"final_class": "person"}]},
            "manual_review_summary": {"reviewed_local_objects": 22, "duplicate_tracks": 3, "false_detections": 2, "good_primary_crops": 10, "alternative_crops_preferred": 5, "all_crops_bad": 1},
            "manual_review_progress": {"reviewed_objects": 22, "total_objects": 83},
            "failures": {"failures": []},
        }

    def test_runtime_normalization_and_no_division_by_zero(self) -> None:
        self.assertEqual(safe_divide(1, 0), 0.0)
        self.assertEqual(normalize_per_minute(5, 0), 0.0)

    def test_runtime_comparison(self) -> None:
        payload = build_runtime_comparison(self._td_case2_metrics(), self._hybrid_metrics())
        self.assertIn("td_case2", payload)
        self.assertIn("hybrid", payload)

    def test_detector_reduction_calculations(self) -> None:
        payload = build_detector_usage_comparison(self._td_case2_metrics(), self._hybrid_metrics())
        self.assertEqual(payload["hybrid"]["yolo_calls"], 40)

    def test_tracking_counts(self) -> None:
        payload = build_tracking_count_comparison(self._td_case2_metrics(), self._hybrid_metrics())
        self.assertEqual(payload["hybrid"]["accepted_merges"], 1)

    def test_crop_and_class_counts(self) -> None:
        crop = build_crop_quality_comparison(self._td_case2_metrics(), self._hybrid_metrics())
        classes = build_class_count_comparison(self._td_case2_metrics(), self._hybrid_metrics())
        self.assertEqual(crop["hybrid"]["plate_candidate_count"], 8)
        self.assertEqual(classes["hybrid"]["car"], 1)

    def test_visual_review_timestamp_generation(self) -> None:
        timestamps = generate_visual_review_timestamps(100.0, count=20)
        self.assertEqual(len(timestamps), 20)
        self.assertTrue(all(item >= 0.0 for item in timestamps))

    def test_incomplete_manual_review_handling(self) -> None:
        payload = build_manual_review_summary(self._hybrid_metrics())
        self.assertEqual(payload["reviewed_object_count"], 22)
        self.assertEqual(payload["unreviewed_object_count"], 61)

    def test_config_differences(self) -> None:
        td_artifacts = {"config_snapshot": {"env_overrides": {"TD_CASE2_YOLO_MODEL_PATH": "a"}}, "metrics": self._td_case2_metrics()}
        hy_artifacts = {"config_snapshot": {"env_overrides": {"TD_CASE2_YOLO_MODEL_PATH": "b"}}, "metrics": self._hybrid_metrics()}
        payload = build_config_differences(td_artifacts, hy_artifacts, "test_cam_01", "single_camera_comparison", "Asia/Kolkata")
        self.assertTrue(payload["differences"])

    def test_normalized_metrics(self) -> None:
        runtime = build_runtime_comparison(self._td_case2_metrics(), self._hybrid_metrics())
        payload = build_normalized_metrics(self._td_case2_metrics(), self._hybrid_metrics(), runtime)
        self.assertIn("td_case2", payload)


if __name__ == "__main__":
    unittest.main()
