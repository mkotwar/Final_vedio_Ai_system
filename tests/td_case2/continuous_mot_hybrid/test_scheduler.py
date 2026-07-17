from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tests.td_case2.continuous_mot_hybrid.adaptive_detector_scheduler import (
    AdaptiveDetectorScheduler,
    SchedulerObservation,
    STATE_DENSE,
    STATE_EMERGENCY,
    STATE_IDLE,
    STATE_SPARSE,
)
from tests.td_case2.continuous_mot_hybrid.video_frame_stream import stream_processed_frames


class SchedulerTests(unittest.TestCase):
    def test_timestamp_based_10fps_frame_scheduling(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "sample.mp4"
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 64))
            for index in range(25):
                frame = np.full((64, 64, 3), index, dtype=np.uint8)
                writer.write(frame)
            writer.release()
            _, _, _, iterator = stream_processed_frames(video_path=video_path, processing_fps=10.0)
            frames = [record.to_dict() for record, _ in iterator]
        self.assertEqual(len(frames), 10)
        self.assertAlmostEqual(frames[1]["timestamp_seconds"], 0.12, places=2)
        self.assertGreater(frames[-1]["timestamp_seconds"], 0.8)

    def test_adaptive_detector_state_changes(self) -> None:
        scheduler = AdaptiveDetectorScheduler(0.2, 0.3, 0.5, 0.5)
        idle = scheduler.decide(
            SchedulerObservation(0.0, 0, 0, 0, 0, 0.0, 0.0, 0, False, False, None)
        )
        dense = scheduler.decide(
            SchedulerObservation(0.2, 2, 2, 0, 2, 0.4, 0.2, 1, True, False, 0.0)
        )
        sparse = scheduler.decide(
            SchedulerObservation(0.6, 2, 0, 0, 0, 0.1, 0.02, 0, False, False, 0.4)
        )
        self.assertEqual(idle.state, STATE_IDLE)
        self.assertEqual(dense.state, STATE_DENSE)
        self.assertEqual(sparse.state, STATE_SPARSE)

    def test_maximum_detector_gap_enforcement_and_emergency_trigger(self) -> None:
        scheduler = AdaptiveDetectorScheduler(0.2, 0.3, 0.5, 0.5)
        decision = scheduler.decide(
            SchedulerObservation(1.2, 1, 0, 0, 0, 0.1, 0.0, 0, False, False, 0.5)
        )
        self.assertEqual(decision.state, STATE_EMERGENCY)
        self.assertTrue(decision.should_run_detector)
        self.assertIn("maximum_gap_enforced", decision.reasons)


if __name__ == "__main__":
    unittest.main()

