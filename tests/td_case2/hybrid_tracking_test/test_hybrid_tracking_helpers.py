from __future__ import annotations

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hybrid_tracking_test.box_validation import bbox_iou, clip_bbox_to_frame, validate_propagated_bbox, xywh_to_xyxy, xyxy_to_xywh
from hybrid_tracking_test.config import resolve_config
from hybrid_tracking_test.data_models import EventRecord
from hybrid_tracking_test.hybrid_track_manager import greedy_assignment


class HybridTrackingHelperTests(unittest.TestCase):
    def test_xyxy_xywh_round_trip(self) -> None:
        box = [10.0, 20.0, 40.0, 60.0]
        self.assertEqual(xywh_to_xyxy(xyxy_to_xywh(box)), box)

    def test_bbox_iou(self) -> None:
        self.assertAlmostEqual(bbox_iou([0, 0, 10, 10], [5, 5, 15, 15]), 25.0 / 175.0, places=6)

    def test_clip_bbox(self) -> None:
        self.assertEqual(clip_bbox_to_frame([-5, 2, 110, 50], 100, 40), [0.0, 2.0, 100.0, 40.0])

    def test_validation_rejects_large_jump(self) -> None:
        result = validate_propagated_bbox(
            current_bbox_xyxy=[100, 100, 150, 150],
            previous_bbox_xyxy=[0, 0, 50, 50],
            frame_width=200,
            frame_height=200,
            minimum_area_ratio_change=0.5,
            maximum_area_ratio_change=2.0,
            minimum_aspect_ratio_change=0.5,
            maximum_aspect_ratio_change=2.0,
            maximum_center_jump_diagonals=0.5,
            minimum_visible_area_ratio=0.0001,
        )
        self.assertFalse(result.valid)
        self.assertIn("center_jump_too_large", result.reasons)

    def test_greedy_assignment(self) -> None:
        assignments = greedy_assignment([[0.1, 0.9], [0.2, 0.3]], max_cost=0.5)
        self.assertEqual(assignments, [(0, 0), (1, 1)])

    def test_resolve_config_accepts_typed_cli_values(self) -> None:
        run_dir = Path(tempfile.mkdtemp(prefix="hybrid_cfg_run_"))
        video_path = run_dir / "video.mp4"
        video_path.write_bytes(b"")
        config = resolve_config(
            Namespace(
                video_path=str(video_path),
                run_dir=str(run_dir),
                processing_fps=10.0,
                yolo_interval_frames=3,
                max_yolo_gap_seconds=0.5,
                minimum_detection_confidence=0.25,
                minimum_iou_match=0.30,
                maximum_missed_yolo_refreshes=3,
                device=None,
                save_annotated_video=False,
                no_save_annotated_video=False,
                enable_motion_trigger=False,
                disable_motion_trigger=False,
                enable_entry_zone_trigger=False,
                disable_entry_zone_trigger=False,
                enable_overlap_trigger=False,
                disable_overlap_trigger=False,
                entry_zones_file=None,
            )
        )
        self.assertEqual(config.processing_fps, 10.0)
        self.assertEqual(config.yolo_interval_frames, 3)
        self.assertEqual(config.max_yolo_gap_seconds, 0.5)

    def test_event_record_to_dict_serializes_numpy_values(self) -> None:
        event = EventRecord(
            timestamp_seconds=1.25,
            source_frame_index=7,
            event_type="uncovered_motion_trigger",
            details={
                "mask": np.zeros((2, 2), dtype=np.uint8),
                "score": np.float32(0.5),
                "regions": [{"bbox_xyxy": np.asarray([1, 2, 3, 4], dtype=np.int32)}],
            },
        )
        payload = event.to_dict()
        self.assertEqual(payload["details"]["mask"], [[0, 0], [0, 0]])
        self.assertEqual(payload["details"]["score"], 0.5)
        self.assertEqual(payload["details"]["regions"][0]["bbox_xyxy"], [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
