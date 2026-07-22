from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path

from tests.td_case2.multicamera_vehicle_tracking_pipeline.detection.detection_models import DetectionPacket, VehicleDetection
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.camera_detection_router import CameraDetectionRouter
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.tracker_factory import TrackerFactory
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.tracking_config import TrackingConfig


class _SharedIdTracker:
    def update(self, results, img=None):
        if len(results) == 0:
            return []
        return [[1, 2, 10, 12, 1, 0.9, 0, 0]]


def _packet(camera_code: str, frame_number: int) -> DetectionPacket:
    return DetectionPacket(
        camera_code=camera_code,
        camera_name=f"Camera {camera_code}",
        source_path=Path(f"{camera_code}.mp4"),
        frame_number=frame_number,
        video_time_seconds=float(frame_number),
        camera_timestamp=datetime(2026, 7, 22, 10, 0, min(frame_number, 59)),
        frame_width=128,
        frame_height=72,
        detections=[VehicleDetection(0, "car", 0.9, (1.0, 2.0, 10.0, 12.0))],
        inference_time_ms=1.0,
        detector_model="fake.pt",
        detector_device="cpu",
        frame=None,
    )


class CameraDetectionRouterTests(unittest.TestCase):
    def test_packets_route_to_correct_camera_trackers(self) -> None:
        factory = TrackerFactory(TrackingConfig(min_confirmed_observations=1), tracker_creator=lambda config: _SharedIdTracker())
        router = CameraDetectionRouter(TrackingConfig(min_confirmed_observations=1), tracker_factory=factory)
        first = router.route(_packet("CAM_001", 0))
        second = router.route(_packet("CAM_002", 0))
        self.assertEqual(first.observations[0].camera_code, "CAM_001")
        self.assertEqual(second.observations[0].camera_code, "CAM_002")
        self.assertEqual(set(factory.configured_camera_codes()), {"CAM_001", "CAM_002"})

    def test_same_numeric_track_id_stays_separate_per_camera(self) -> None:
        factory = TrackerFactory(TrackingConfig(min_confirmed_observations=1), tracker_creator=lambda config: _SharedIdTracker())
        router = CameraDetectionRouter(TrackingConfig(min_confirmed_observations=1), tracker_factory=factory)
        first = router.route(_packet("CAM_001", 0))
        second = router.route(_packet("CAM_002", 0))
        self.assertEqual(first.observations[0].local_track_id, second.observations[0].local_track_id)
        self.assertNotEqual(first.observations[0].track_uuid, second.observations[0].track_uuid)


if __name__ == "__main__":
    unittest.main()
