from __future__ import annotations

import unittest

from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.tracker_factory import TrackerFactory
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.tracking_config import TrackingConfig


class TrackerFactoryTests(unittest.TestCase):
    def test_same_camera_returns_same_tracker(self) -> None:
        factory = TrackerFactory(TrackingConfig(), tracker_creator=lambda config: object())
        first = factory.get_or_create("CAM_001")
        second = factory.get_or_create("CAM_001")
        self.assertIs(first, second)

    def test_different_cameras_return_different_trackers(self) -> None:
        factory = TrackerFactory(TrackingConfig(), tracker_creator=lambda config: object())
        first = factory.get_or_create("CAM_001")
        second = factory.get_or_create("CAM_002")
        self.assertIsNot(first, second)

    def test_reset_clears_cached_trackers(self) -> None:
        factory = TrackerFactory(TrackingConfig(), tracker_creator=lambda config: object())
        first = factory.get_or_create("CAM_001")
        factory.reset()
        second = factory.get_or_create("CAM_001")
        self.assertIsNot(first, second)

    def test_camera_frame_rate_overrides_config_when_creating_tracker(self) -> None:
        seen = {}

        def _creator(config, **kwargs):
            seen["frame_rate"] = config.frame_rate
            seen["camera_code"] = kwargs.get("camera_code")
            return object()

        factory = TrackerFactory(TrackingConfig(frame_rate=30.0), tracker_creator=_creator)
        factory.get_or_create("CAM_009", frame_rate=19.951)
        self.assertAlmostEqual(seen["frame_rate"], 19.951, places=3)
        self.assertEqual(seen["camera_code"], "CAM_009")


if __name__ == "__main__":
    unittest.main()
