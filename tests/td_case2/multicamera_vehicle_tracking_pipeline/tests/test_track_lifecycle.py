from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path

from tests.td_case2.multicamera_vehicle_tracking_pipeline.detection.detection_models import DetectionPacket
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.track_lifecycle import LocalTrackLifecycle
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.tracking_config import TrackingConfig
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.tracking_models import TrackObservation


def _packet(frame_number: int, camera_code: str = "CAM_001") -> DetectionPacket:
    return DetectionPacket(
        camera_code=camera_code,
        camera_name="North Gate",
        source_path=Path("camera.mp4"),
        frame_number=frame_number,
        video_time_seconds=float(frame_number),
        camera_timestamp=datetime(2026, 7, 22, 10, 0, min(frame_number, 59)),
        frame_width=128,
        frame_height=72,
        detections=[],
        inference_time_ms=0.0,
        detector_model="fake.pt",
        detector_device="cpu",
    )


def _observation(frame_number: int, local_track_id: int = 1, camera_code: str = "CAM_001") -> TrackObservation:
    return TrackObservation(
        camera_code=camera_code,
        local_track_id=local_track_id,
        frame_number=frame_number,
        video_time_seconds=float(frame_number),
        camera_timestamp=datetime(2026, 7, 22, 10, 0, min(frame_number, 59)),
        class_name="car",
        confidence=0.9,
        bbox_xyxy=(1.0, 2.0, 10.0, 12.0),
    )


class LocalTrackLifecycleTests(unittest.TestCase):
    def test_tentative_to_active_transition(self) -> None:
        lifecycle = LocalTrackLifecycle(TrackingConfig(min_confirmed_observations=2))
        first = lifecycle.update(_packet(0), [_observation(0)])
        second = lifecycle.update(_packet(1), [_observation(1)])
        self.assertEqual(first.observations[0].state, "tentative")
        self.assertEqual(second.observations[0].state, "active")

    def test_active_to_temporarily_lost_transition(self) -> None:
        lifecycle = LocalTrackLifecycle(TrackingConfig(min_confirmed_observations=1, max_lost_frames=5))
        lifecycle.update(_packet(0), [_observation(0)])
        lifecycle.update(_packet(1), [])
        self.assertEqual(lifecycle.get_active_tracks("CAM_001")[0].state, "temporarily_lost")

    def test_temporarily_lost_to_active_recovery(self) -> None:
        lifecycle = LocalTrackLifecycle(TrackingConfig(min_confirmed_observations=1, max_lost_frames=5))
        lifecycle.update(_packet(0), [_observation(0)])
        lifecycle.update(_packet(1), [])
        recovered = lifecycle.update(_packet(2), [_observation(2)])
        self.assertEqual(recovered.observations[0].state, "active")

    def test_lost_timeout_completes_active_track(self) -> None:
        lifecycle = LocalTrackLifecycle(TrackingConfig(min_confirmed_observations=1, max_lost_frames=0))
        lifecycle.update(_packet(0), [_observation(0)])
        result = lifecycle.update(_packet(1), [])
        self.assertEqual(result.completed_tracks[0].state, "completed")

    def test_short_tentative_track_is_discarded(self) -> None:
        lifecycle = LocalTrackLifecycle(TrackingConfig(min_confirmed_observations=2, max_lost_frames=0))
        lifecycle.update(_packet(0), [_observation(0)])
        result = lifecycle.update(_packet(1), [])
        self.assertEqual(result.completed_tracks[0].state, "discarded")

    def test_flush_completes_valid_tracks(self) -> None:
        lifecycle = LocalTrackLifecycle(TrackingConfig(min_confirmed_observations=1))
        lifecycle.update(_packet(0), [_observation(0)])
        result = lifecycle.flush_camera("CAM_001")
        self.assertEqual(result.completed_tracks[0].state, "completed")


if __name__ == "__main__":
    unittest.main()
