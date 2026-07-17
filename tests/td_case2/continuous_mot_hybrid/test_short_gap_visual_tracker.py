from __future__ import annotations

import unittest

import numpy as np

from tests.td_case2.continuous_mot_hybrid.short_gap_visual_tracker import ShortGapVisualTrackerManager


class FakeTracker:
    def __init__(self, boxes):
        self.boxes = list(boxes)

    def init(self, frame, bbox):
        return True

    def update(self, frame):
        if not self.boxes:
            return False, (0, 0, 0, 0)
        return True, self.boxes.pop(0)


class VisualTrackerTests(unittest.TestCase):
    def test_visual_bridge_maximum_duration_and_frozen_rejection(self) -> None:
        manager = ShortGapVisualTrackerManager(tracker_name="csrt", maximum_bridge_seconds=0.3, frame_width=128, frame_height=128)
        session = manager.sessions
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        manager.start_or_refresh(track_id="vehicle_track_0001", frame=frame, bbox_xyxy=[10, 10, 30, 30], timestamp_seconds=0.0)
        manager.sessions["vehicle_track_0001"].tracker = FakeTracker([(10, 10, 20, 20)])
        first = manager.update(track_id="vehicle_track_0001", frame=frame, timestamp_seconds=0.1)
        self.assertEqual(first["bbox_source"], "visual_bridge_invalid")

        manager.start_or_refresh(track_id="vehicle_track_0002", frame=frame, bbox_xyxy=[10, 10, 30, 30], timestamp_seconds=0.0)
        manager.sessions["vehicle_track_0002"].tracker = FakeTracker([(12, 12, 20, 20)])
        second = manager.update(track_id="vehicle_track_0002", frame=frame, timestamp_seconds=0.4)
        self.assertEqual(second["reason"], "bridge_duration_exceeded")

    def test_detector_visual_bridge_disagreement(self) -> None:
        manager = ShortGapVisualTrackerManager(tracker_name="csrt", maximum_bridge_seconds=0.3, frame_width=128, frame_height=128)
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        manager.start_or_refresh(track_id="vehicle_track_0003", frame=frame, bbox_xyxy=[10, 10, 30, 30], timestamp_seconds=0.0)
        event = manager.reconcile_with_detector(
            track_id="vehicle_track_0003",
            detector_bbox_xyxy=[80, 80, 110, 110],
            timestamp_seconds=0.2,
        )
        self.assertEqual(event["reason"], "detector_disagreement")


if __name__ == "__main__":
    unittest.main()
