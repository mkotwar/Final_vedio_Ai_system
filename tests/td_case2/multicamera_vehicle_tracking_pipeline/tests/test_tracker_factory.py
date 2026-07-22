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


if __name__ == "__main__":
    unittest.main()
