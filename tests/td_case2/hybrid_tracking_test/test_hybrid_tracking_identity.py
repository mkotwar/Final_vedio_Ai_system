from __future__ import annotations

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hybrid_tracking_test.config import resolve_config
from hybrid_tracking_test.data_models import DetectionObservation
from hybrid_tracking_test.hybrid_track_manager import HybridTrackManager
from hybrid_tracking_test.track_fragment_reconciliation import reconcile_tracks


class _DummyTracker:
    def is_initialized(self) -> bool:
        return True

    def reset(self, frame, bbox_xyxy) -> None:
        return None

    def update(self, frame):
        return True, (0.0, 0.0, 1.0, 1.0)


def _build_config():
    run_dir = Path(tempfile.mkdtemp(prefix="hybrid_identity_run_"))
    video_path = run_dir / "video.mp4"
    video_path.write_bytes(b"")
    return resolve_config(
        Namespace(
            video_path=str(video_path),
            run_dir=str(run_dir),
            processing_fps=10.0,
            yolo_interval_frames=3,
            max_yolo_gap_seconds=0.5,
            minimum_detection_confidence=0.35,
            minimum_iou_match=0.20,
            maximum_missed_yolo_refreshes=8,
            maximum_track_idle_seconds=2.0,
            minimum_track_hits=3,
            motion_min_area_ratio=0.006,
            motion_persistence_frames=3,
            motion_track_region_expansion=0.30,
            lost_track_recovery_seconds=2.0,
            lost_track_max_center_distance_ratio=1.5,
            lost_track_min_area_ratio=0.40,
            lost_track_max_area_ratio=2.50,
            empty_scene_yolo_interval_seconds=0.5,
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


class HybridTrackingIdentityTests(unittest.TestCase):
    def test_lost_track_reactivation_reuses_original_id(self) -> None:
        config = _build_config()
        manager = HybridTrackManager(config)
        manager._initialize_kcf = lambda track, frame, bbox_xyxy: setattr(track, "kcf_instance", _DummyTracker()) or setattr(track, "kcf_initialized", True)  # type: ignore[method-assign]
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        first = DetectionObservation(
            class_id=2,
            class_name="car",
            confidence=0.8,
            bbox_xyxy=[10.0, 20.0, 40.0, 50.0],
            model_source="combined",
            source_frame_index=0,
            processed_frame_index=0,
            timestamp_seconds=0.0,
        )
        track = manager._create_track(first, frame)
        manager.active_tracks[track.track_id] = track
        track.append_trajectory(
            source_frame_index=1,
            processed_frame_index=1,
            timestamp_seconds=0.1,
            bbox_xyxy=[12.0, 20.0, 42.0, 50.0],
            bbox_source="kcf",
            limit=config.trajectory_history_limit,
        )
        manager._move_to_lost(track, timestamp_seconds=0.2, source_frame_index=2, reason="synthetic_test")
        recovered_detection = DetectionObservation(
            class_id=2,
            class_name="car",
            confidence=0.9,
            bbox_xyxy=[14.0, 20.0, 44.0, 50.0],
            model_source="combined",
            source_frame_index=4,
            processed_frame_index=4,
            timestamp_seconds=0.4,
        )
        _tracks, _detections, associations = manager.refresh_with_detections(
            detections=[recovered_detection],
            frame=frame,
            source_frame_index=4,
            processed_frame_index=4,
            timestamp_seconds=0.4,
        )
        self.assertEqual(manager.counters.get("tracks_created", 0), 1)
        self.assertEqual(manager.counters.get("tracks_reactivated", 0), 1)
        self.assertIn(track.track_id, manager.active_tracks)
        self.assertEqual(manager.active_tracks[track.track_id].reactivation_count, 1)
        self.assertEqual(associations[0]["track_id"], track.track_id)
        self.assertEqual(associations[0]["result"], "reactivated_track")

    def test_class_vote_accumulation_keeps_dominant_vehicle_label(self) -> None:
        config = _build_config()
        manager = HybridTrackManager(config)
        manager._initialize_kcf = lambda track, frame, bbox_xyxy: setattr(track, "kcf_instance", _DummyTracker()) or setattr(track, "kcf_initialized", True)  # type: ignore[method-assign]
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        first = DetectionObservation(2, "car", 0.8, [10.0, 20.0, 40.0, 50.0], "combined", 0, 0, 0, 0.0)
        second = DetectionObservation(7, "truck", 0.3, [11.0, 20.0, 41.0, 50.0], "combined", 0, 1, 1, 0.1)
        third = DetectionObservation(2, "car", 0.7, [12.0, 20.0, 42.0, 50.0], "combined", 0, 2, 2, 0.2)
        track = manager._create_track(first, frame)
        manager.active_tracks[track.track_id] = track
        manager._apply_detection_update(track=track, detection=second, frame=frame, source_frame_index=1, processed_frame_index=1, timestamp_seconds=0.1, matching_stage="same_class_iou")
        manager._apply_detection_update(track=track, detection=third, frame=frame, source_frame_index=2, processed_frame_index=2, timestamp_seconds=0.2, matching_stage="same_class_iou")
        self.assertEqual(track.class_name, "car")
        self.assertGreater(track.class_votes["car"], track.class_votes["truck"])

    def test_reconciliation_does_not_merge_simultaneous_tracks(self) -> None:
        raw_tracks = [
            {
                "track_id": 1,
                "class_name": "car",
                "is_confirmed": True,
                "start_timestamp_seconds": 1.0,
                "end_timestamp_seconds": 2.0,
                "duration_seconds": 1.0,
                "detection_hits": 3,
                "propagation_hits": 2,
                "trajectory": [
                    {"timestamp_seconds": 1.0, "bbox_xyxy": [0.0, 0.0, 20.0, 20.0]},
                    {"timestamp_seconds": 2.0, "bbox_xyxy": [10.0, 0.0, 30.0, 20.0]},
                ],
            },
            {
                "track_id": 2,
                "class_name": "car",
                "is_confirmed": True,
                "start_timestamp_seconds": 1.5,
                "end_timestamp_seconds": 2.5,
                "duration_seconds": 1.0,
                "detection_hits": 3,
                "propagation_hits": 2,
                "trajectory": [
                    {"timestamp_seconds": 1.5, "bbox_xyxy": [100.0, 0.0, 120.0, 20.0]},
                    {"timestamp_seconds": 2.5, "bbox_xyxy": [110.0, 0.0, 130.0, 20.0]},
                ],
            },
        ]
        reconciled_tracks, merge_events, report = reconcile_tracks(raw_tracks)
        self.assertEqual(len(reconciled_tracks), 2)
        self.assertEqual(len(merge_events), 0)
        self.assertEqual(report["reconciled_track_count"], 2)


if __name__ == "__main__":
    unittest.main()
