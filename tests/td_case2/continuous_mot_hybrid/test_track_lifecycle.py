from __future__ import annotations

import unittest

from tests.td_case2.continuous_mot_hybrid.track_lifecycle import (
    LIFECYCLE_ACTIVELY_TRACKED,
    LIFECYCLE_RECOVERABLE,
    LIFECYCLE_TERMINATED,
    LIFECYCLE_WEAK_SINGLE_DETECTION,
    LifecycleConfig,
    infer_lifecycle_state,
)


class TrackLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = LifecycleConfig(min_person_confirm_hits=3, min_vehicle_confirm_hits=2, lost_recovery_seconds=1.0)

    def test_track_confirmation_rules(self) -> None:
        self.assertEqual(
            infer_lifecycle_state(
                object_family="vehicle",
                detector_hits=2,
                duration_seconds=0.6,
                bbox_source="yolo",
                backend_state="tracked",
                time_since_update_seconds=0.0,
                config=self.config,
            ),
            LIFECYCLE_ACTIVELY_TRACKED,
        )

    def test_weak_single_detection_and_lost_track_recovery(self) -> None:
        weak = infer_lifecycle_state(
            object_family="person",
            detector_hits=1,
            duration_seconds=0.2,
            bbox_source="yolo",
            backend_state="tracked",
            time_since_update_seconds=0.0,
            config=self.config,
        )
        recoverable = infer_lifecycle_state(
            object_family="vehicle",
            detector_hits=2,
            duration_seconds=1.0,
            bbox_source="yolo",
            backend_state="lost",
            time_since_update_seconds=0.8,
            config=self.config,
        )
        terminated = infer_lifecycle_state(
            object_family="vehicle",
            detector_hits=2,
            duration_seconds=1.0,
            bbox_source="yolo",
            backend_state="lost",
            time_since_update_seconds=1.5,
            config=self.config,
        )
        self.assertEqual(weak, LIFECYCLE_WEAK_SINGLE_DETECTION)
        self.assertEqual(recoverable, LIFECYCLE_RECOVERABLE)
        self.assertEqual(terminated, LIFECYCLE_TERMINATED)


if __name__ == "__main__":
    unittest.main()

