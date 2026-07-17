from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hybrid_tracking_test.local_identity_package import build_local_identity_packages
from hybrid_tracking_test.track_fragment_reconciliation import reconcile_track_fragments
from hybrid_tracking_test.track_merge_scoring import MergeScoringConfig, compute_merge_candidate
from hybrid_tracking_test.track_quality import build_track_quality_report, evaluate_track_quality, object_family_for_class


def _vehicle_track(track_id: int, start: float, end: float, start_box: list[float], end_box: list[float], *, class_name: str = "car", confirmed: bool = True, detection_hits: int = 4) -> dict:
    return {
        "track_id": track_id,
        "class_name": class_name,
        "duration_seconds": round(end - start, 6),
        "start_timestamp_seconds": start,
        "end_timestamp_seconds": end,
        "first_source_frame_index": int(round(start * 10)),
        "last_source_frame_index": int(round(end * 10)),
        "detection_hits": detection_hits,
        "propagation_hits": 2,
        "kcf_failures": 0,
        "missed_detection_refreshes": 0,
        "maximum_seconds_without_detection": 0.2,
        "termination_reason": "video_end",
        "is_confirmed": confirmed,
        "class_votes": {class_name: float(detection_hits)},
        "reactivation_count": 0,
        "trajectory": [
            {
                "source_frame_index": int(round(start * 10)),
                "processed_frame_index": int(round(start * 10)),
                "timestamp_seconds": start,
                "bbox_xyxy": start_box,
                "bbox_source": "yolo",
            },
            {
                "source_frame_index": int(round(end * 10)),
                "processed_frame_index": int(round(end * 10)),
                "timestamp_seconds": end,
                "bbox_xyxy": end_box,
                "bbox_source": "yolo",
            },
        ],
    }


class ReconciliationTests(unittest.TestCase):
    def test_object_family_compatibility(self) -> None:
        self.assertEqual(object_family_for_class("person"), "person")
        self.assertEqual(object_family_for_class("car"), "vehicle")

    def test_short_track_quality_flags(self) -> None:
        track = _vehicle_track(1, 0.0, 0.2, [0, 0, 20, 20], [2, 2, 22, 22], detection_hits=1, confirmed=False)
        evaluation = evaluate_track_quality(track, frame_width=100, frame_height=100)
        self.assertIn("short_track", evaluation["quality_flags"])
        self.assertIn("single_detection", evaluation["quality_flags"])

    def test_merge_candidate_accepts_temporally_continuous_vehicle(self) -> None:
        left = {
            "track_id": 1,
            "final_class": "car",
            "quality_score": 0.8,
            "entry_boundary": None,
            "exit_boundary": None,
            "trajectory": [
                {"timestamp_seconds": 0.0, "bbox_xyxy": [10, 10, 30, 30]},
                {"timestamp_seconds": 2.0, "bbox_xyxy": [20, 40, 40, 60]},
            ],
            "start_timestamp_seconds": 0.0,
            "end_timestamp_seconds": 2.0,
        }
        right = {
            "track_id": 2,
            "final_class": "truck",
            "quality_score": 0.75,
            "entry_boundary": None,
            "exit_boundary": None,
            "trajectory": [
                {"timestamp_seconds": 2.2, "bbox_xyxy": [22, 44, 42, 64]},
                {"timestamp_seconds": 4.0, "bbox_xyxy": [32, 74, 52, 94]},
            ],
            "start_timestamp_seconds": 2.2,
            "end_timestamp_seconds": 4.0,
        }
        candidate = compute_merge_candidate(left, right, config=MergeScoringConfig())
        self.assertTrue(candidate["compatible"])

    def test_merge_candidate_rejects_temporal_overlap(self) -> None:
        left = {"track_id": 1, "final_class": "car", "quality_score": 0.8, "trajectory": [{"timestamp_seconds": 0.0, "bbox_xyxy": [0, 0, 10, 10]}, {"timestamp_seconds": 2.0, "bbox_xyxy": [10, 0, 20, 10]}], "start_timestamp_seconds": 0.0, "end_timestamp_seconds": 2.0}
        right = {"track_id": 2, "final_class": "car", "quality_score": 0.8, "trajectory": [{"timestamp_seconds": 1.5, "bbox_xyxy": [100, 100, 110, 110]}, {"timestamp_seconds": 2.5, "bbox_xyxy": [110, 100, 120, 110]}], "start_timestamp_seconds": 1.5, "end_timestamp_seconds": 2.5}
        candidate = compute_merge_candidate(left, right, config=MergeScoringConfig())
        self.assertFalse(candidate["compatible"])
        self.assertIn("temporal_overlap_conflict", candidate["reasons"])

    def test_reconcile_valid_fragment_merge(self) -> None:
        raw_tracks = [
            _vehicle_track(1, 0.0, 2.0, [10, 10, 30, 30], [20, 40, 40, 60]),
            _vehicle_track(2, 2.2, 4.0, [22, 44, 42, 64], [35, 80, 55, 100], class_name="truck"),
        ]
        quality_report = build_track_quality_report(raw_tracks, frame_width=200, frame_height=200)
        local_objects, merges, rejected, report = reconcile_track_fragments(
            raw_tracks,
            quality_report,
            camera_id="test_cam_01",
            camera_group="single_camera_test",
            camera_timezone="Asia/Kolkata",
            scoring_config=MergeScoringConfig(),
        )
        self.assertEqual(report["reconciled_local_object_count"], 1)
        self.assertEqual(len(merges), 1)
        self.assertEqual(len(rejected), 0)
        self.assertEqual(local_objects[0]["source_raw_track_ids"], [1, 2])

    def test_reconcile_rejects_person_vehicle_merge(self) -> None:
        raw_tracks = [
            _vehicle_track(1, 0.0, 1.0, [10, 10, 30, 30], [20, 20, 40, 40], class_name="person"),
            _vehicle_track(2, 1.1, 2.0, [22, 22, 42, 42], [32, 32, 52, 52], class_name="car"),
        ]
        quality_report = build_track_quality_report(raw_tracks, frame_width=200, frame_height=200)
        local_objects, merges, _rejected, report = reconcile_track_fragments(
            raw_tracks,
            quality_report,
            camera_id="test_cam_01",
            camera_group="single_camera_test",
            camera_timezone="Asia/Kolkata",
            scoring_config=MergeScoringConfig(),
        )
        self.assertEqual(report["reconciled_local_object_count"], 2)
        self.assertEqual(len(merges), 0)
        self.assertEqual(len(local_objects), 2)

    def test_local_identity_package_generation(self) -> None:
        local_objects = [
            {
                "camera_id": "test_cam_01",
                "camera_group": "single_camera_test",
                "camera_timezone": "Asia/Kolkata",
                "local_object_id": 1,
                "local_object_key": "test_cam_01:1",
                "object_family": "vehicle",
                "final_class": "car",
                "source_raw_track_ids": [1],
                "start_timestamp_seconds": 0.0,
                "end_timestamp_seconds": 1.0,
                "duration_seconds": 1.0,
                "first_source_frame_index": 0,
                "last_source_frame_index": 10,
                "combined_trajectory": [{"timestamp_seconds": 0.0, "bbox_xyxy": [0, 0, 10, 10]}, {"timestamp_seconds": 1.0, "bbox_xyxy": [0, 10, 10, 20]}],
                "class_votes": {"car": 3.0},
                "quality_level": "high",
                "quality_score": 0.8,
                "entry_boundary": "top",
                "exit_boundary": "bottom",
                "confirmed": True,
                "warnings": [],
                "quality_flags": [],
            }
        ]
        representative_frames = [
            {
                "local_object_key": "test_cam_01:1",
                "downstream_status": "ready",
                "representative_frames": {
                    "primary": {"crop_path": "crop.jpg", "full_frame_path": "frame.jpg"},
                    "plate_candidate": {"crop_path": "plate.jpg"},
                },
            }
        ]
        packages, flat_rows, report = build_local_identity_packages(local_objects=local_objects, representative_frames=representative_frames)
        self.assertEqual(len(packages), 1)
        self.assertEqual(flat_rows[0]["primary_crop_path"], "crop.jpg")
        self.assertEqual(report["ready_packages"], 1)


if __name__ == "__main__":
    unittest.main()
