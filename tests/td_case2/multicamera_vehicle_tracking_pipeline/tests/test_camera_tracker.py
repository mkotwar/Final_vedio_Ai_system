from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path

from tests.td_case2.multicamera_vehicle_tracking_pipeline.detection.detection_models import DetectionPacket, VehicleDetection
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.camera_tracker import CameraTracker
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.track_lifecycle import LocalTrackLifecycle
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.tracker_factory import TrackerFactory
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.tracking_config import TrackingConfig


class _FakeTracker:
    def __init__(self, rows_per_call: list[list[list[float]]]) -> None:
        self.rows_per_call = rows_per_call
        self.calls = 0
        self.last_len = None
        self.last_img = None

    def update(self, results, img=None):
        self.calls += 1
        self.last_len = len(results)
        self.last_img = img
        index = self.calls - 1
        if index >= len(self.rows_per_call):
            return []
        return self.rows_per_call[index]


def _packet(camera_code: str = "CAM_001", frame_number: int = 0, detections: list[VehicleDetection] | None = None) -> DetectionPacket:
    return DetectionPacket(
        camera_code=camera_code,
        camera_name="North Gate",
        source_path=Path("camera.mp4"),
        frame_number=frame_number,
        video_time_seconds=float(frame_number),
        camera_timestamp=datetime(2026, 7, 22, 10, 0, frame_number),
        frame_width=128,
        frame_height=72,
        detections=detections if detections is not None else [VehicleDetection(0, "car", 0.9, (1.0, 2.0, 10.0, 12.0))],
        inference_time_ms=1.0,
        detector_model="fake.pt",
        detector_device="cpu",
        frame="frame_object",
    )


class CameraTrackerTests(unittest.TestCase):
    def test_correct_camera_packet_is_accepted(self) -> None:
        fake_tracker = _FakeTracker([[[1, 2, 10, 12, 7, 0.9, 0, 0]]])
        factory = TrackerFactory(TrackingConfig(min_confirmed_observations=1), tracker_creator=lambda config: fake_tracker)
        tracker = CameraTracker("CAM_001", TrackingConfig(min_confirmed_observations=1), factory, LocalTrackLifecycle(TrackingConfig(min_confirmed_observations=1)))
        result = tracker.update(_packet())
        self.assertEqual(len(result.observations), 1)
        self.assertEqual(result.observations[0].local_track_id, 7)

    def test_wrong_camera_packet_is_rejected(self) -> None:
        fake_tracker = _FakeTracker([[]])
        factory = TrackerFactory(TrackingConfig(), tracker_creator=lambda config: fake_tracker)
        tracker = CameraTracker("CAM_001", TrackingConfig(), factory, LocalTrackLifecycle(TrackingConfig()))
        with self.assertRaisesRegex(ValueError, "received packet for CAM_002"):
            tracker.update(_packet(camera_code="CAM_002"))

    def test_empty_detections_return_zero_observations(self) -> None:
        fake_tracker = _FakeTracker([[]])
        factory = TrackerFactory(TrackingConfig(), tracker_creator=lambda config: fake_tracker)
        tracker = CameraTracker("CAM_001", TrackingConfig(), factory, LocalTrackLifecycle(TrackingConfig()))
        result = tracker.update(_packet(detections=[]))
        self.assertEqual(result.observations, [])
        self.assertEqual(fake_tracker.last_len, 0)

    def test_track_ids_and_metadata_are_preserved(self) -> None:
        fake_tracker = _FakeTracker(
            [
                [[1, 2, 10, 12, 7, 0.9, 0, 0]],
                [[2, 3, 11, 13, 7, 0.92, 0, 0]],
            ]
        )
        config = TrackingConfig(min_confirmed_observations=1)
        factory = TrackerFactory(config, tracker_creator=lambda tracking_config: fake_tracker)
        tracker = CameraTracker("CAM_001", config, factory, LocalTrackLifecycle(config))
        first = tracker.update(_packet(frame_number=0))
        second = tracker.update(_packet(frame_number=1))
        self.assertEqual(first.observations[0].track_uuid, "CAM_001:TRACK_7")
        self.assertEqual(second.observations[0].track_uuid, "CAM_001:TRACK_7")
        self.assertEqual(second.observations[0].frame_number, 1)
        self.assertEqual(second.observations[0].camera_timestamp, datetime(2026, 7, 22, 10, 0, 1))
        self.assertEqual(fake_tracker.last_img, "frame_object")


if __name__ == "__main__":
    unittest.main()
