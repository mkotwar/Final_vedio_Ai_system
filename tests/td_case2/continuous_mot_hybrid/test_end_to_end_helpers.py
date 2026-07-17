from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.td_case2.continuous_mot_hybrid.local_identity_package import build_local_identity_packages
from tests.td_case2.continuous_mot_hybrid.metrics import build_runtime_report


class EndToEndHelperTests(unittest.TestCase):
    def test_ready_fallback_manual_review_rejected_packaging(self) -> None:
        local_objects = [
            {
                "camera_id": "test_cam_01",
                "camera_group": "single_camera_comparison",
                "camera_timezone": "Asia/Kolkata",
                "local_object_id": 1,
                "local_object_key": "test_cam_01:1",
                "object_family": "vehicle",
                "final_class": "car",
                "class_votes": {"car": 3.0},
                "source_raw_track_ids": ["1"],
                "start_timestamp_seconds": 0.0,
                "end_timestamp_seconds": 1.0,
                "duration_seconds": 1.0,
                "first_source_frame_index": 0,
                "last_source_frame_index": 10,
                "sanitized_valid_timeline": [{"bbox_source": "yolo"}] * 3,
                "confirmed": True,
                "quality_level": "high",
                "track_integrity_status": "usable",
                "warnings": [],
                "quality_flags": [],
            }
        ]
        representative_frames = [
            {
                "local_object_key": "test_cam_01:1",
                "representative_frames": {
                    "primary": {"crop_path": "a.jpg", "full_frame_path": "a_full.jpg", "bbox_source": "yolo"},
                },
                "warnings": [],
            }
        ]
        packages, _, report = build_local_identity_packages(local_objects=local_objects, representative_frames=representative_frames)
        self.assertEqual(report["ready_packages"], 1)
        self.assertEqual(packages[0]["downstream_status"], "ready")

    def test_runtime_metric_calculation(self) -> None:
        report = build_runtime_report(
            video_duration_seconds=10.0,
            processed_frames=100,
            tracking_runtime_seconds=3.0,
            cleanup_runtime_seconds=1.0,
            crop_runtime_seconds=1.0,
        )
        self.assertEqual(report["total_runtime_seconds"], 5.0)
        self.assertEqual(report["realtime_factor"], 0.5)

    def test_three_pipeline_comparison_and_missing_optional_file_handling(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            previous = temp_root / "previous"
            current = temp_root / "current"
            for path in [
                previous / "comparison",
                current / "01_video",
                current / "02_frames",
                current / "05_integrity",
                current / "07_representative_frames",
                current / "08_identity_packages",
                current / "09_reports",
            ]:
                path.mkdir(parents=True, exist_ok=True)
            (previous / "comparison" / "10_final_comparison_report.md").write_text("# old\n", encoding="utf-8")
            json_payloads = {
                current / "09_reports" / "runtime_report.json": {"total_runtime_seconds": 10.0},
                current / "09_reports" / "tracking_report.json": {"raw_track_ids": 5, "confirmed_tracks": 4, "tracks_under_0_5_seconds": 1, "tracks_under_1_second": 2},
                current / "09_reports" / "reconciliation_report.json": {"reconciled_objects": 4, "accepted_merges": 1},
                current / "09_reports" / "crop_report.json": {"primary_crops": 4, "plate_candidates": 2},
                current / "09_reports" / "detector_report.json": {"total_yolo_calls": 8},
                current / "08_identity_packages" / "local_identity_package_report.json": {"ready_packages": 2, "manual_review_packages": 1, "rejected_packages": 1},
                current / "02_frames" / "frame_stream_metrics.json": {"processed_frame_count": 50},
                current / "05_integrity" / "track_integrity_report.json": {"frozen_tracks": 0, "boundary_stuck_tracks": 1},
                current / "07_representative_frames" / "representative_frames.json": {"status": "success", "objects": [{"representative_frames": {"alternatives": [{}, {}]}}]},
            }
            for path, payload in json_payloads.items():
                path.write_text(json.dumps(payload), encoding="utf-8")
            command = [
                sys.executable,
                str(Path(__file__).resolve().parent / "compare_with_previous_pipelines.py"),
                "--continuous-run-dir",
                str(current),
                "--previous-comparison-dir",
                str(previous),
            ]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            output_path = current / "09_reports" / "three_pipeline_comparison.json"
            self.assertTrue(output_path.exists())
            self.assertIn("continuous_run_dir=", result.stdout)


if __name__ == "__main__":
    unittest.main()
