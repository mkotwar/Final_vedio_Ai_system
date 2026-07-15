from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from event_preview_video import build_event_preview_clip, get_existing_event_preview
from stage_checks import read_json, write_json


class EventPreviewVideoTests(unittest.TestCase):
    def _write_frame(self, path: Path, color_bgr: tuple[int, int, int]) -> None:
        image = np.full((120, 180, 3), color_bgr, dtype=np.uint8)
        ok = cv2.imwrite(str(path), image)
        self.assertTrue(ok)

    def test_builds_track_event_preview_clip_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            frame_dir = run_dir / "02_sampled_frames"
            frame_dir.mkdir()
            frame_a = frame_dir / "frame_000000.jpg"
            frame_b = frame_dir / "frame_000030.jpg"
            self._write_frame(frame_a, (0, 0, 255))
            self._write_frame(frame_b, (0, 255, 0))

            write_json(
                run_dir / "02A_adaptive_frames.json",
                {
                    "status": "success",
                    "selected_frames": [
                        {"frame_id": "frame_000000", "image_path": "02_sampled_frames/frame_000000.jpg"},
                        {"frame_id": "frame_000030", "image_path": "02_sampled_frames/frame_000030.jpg"},
                    ],
                },
            )
            write_json(
                run_dir / "04B_tracks.json",
                {
                    "status": "success",
                    "tracks": [
                        {
                            "track_id": "vehicle_track_0001",
                            "detections": [
                                {
                                    "frame_id": "frame_000000",
                                    "timestamp_seconds": 0.0,
                                    "bbox_xyxy": [10, 10, 60, 60],
                                },
                                {
                                    "frame_id": "frame_000030",
                                    "timestamp_seconds": 1.0,
                                    "bbox_xyxy": [15, 15, 80, 80],
                                },
                            ],
                        }
                    ],
                },
            )

            record = {
                "object_record_id": "obj_track_vehicle_track_0001",
                "track_id": "vehicle_track_0001",
                "class_name": "car",
                "source_type": "track",
                "timestamp_text": "00:01",
                "full_frame_path": "02_sampled_frames/frame_000030.jpg",
                "first_seen_seconds": 0.0,
                "last_seen_seconds": 1.0,
                "duration_seconds": 1.0,
            }

            clip_payload = build_event_preview_clip(run_dir, record, clip_fps=2.0, best_frame_hold_seconds=1.0)
            clip_path = run_dir / clip_payload["event_clip_path"]
            self.assertTrue(clip_path.exists())
            self.assertGreaterEqual(clip_payload["clip_frame_count"], 4)

            manifest = read_json(run_dir / "10C_search_event_clips_manifest.json")
            self.assertIn("obj_track_vehicle_track_0001", manifest["clips"])
            existing = get_existing_event_preview(run_dir, record)
            self.assertIsNotNone(existing)
            self.assertEqual(existing["event_clip_path"], clip_payload["event_clip_path"])

    def test_builds_single_detection_preview_when_track_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            frame_dir = run_dir / "05_selected_track_full_frames"
            frame_dir.mkdir()
            frame_path = frame_dir / "single.jpg"
            self._write_frame(frame_path, (255, 0, 0))

            write_json(run_dir / "04B_tracks.json", {"status": "success", "tracks": []})
            record = {
                "object_record_id": "obj_det_frame_000001_combined_001",
                "class_name": "motorcycle",
                "source_type": "detection",
                "timestamp_seconds": 3.0,
                "timestamp_text": "00:03",
                "full_frame_path": "05_selected_track_full_frames/single.jpg",
                "bbox_xyxy": [5, 5, 40, 40],
            }

            clip_payload = build_event_preview_clip(run_dir, record, clip_fps=2.0, best_frame_hold_seconds=1.0)
            self.assertTrue((run_dir / clip_payload["event_clip_path"]).exists())
            self.assertEqual(clip_payload["source_type"], "detection")


if __name__ == "__main__":
    unittest.main()
