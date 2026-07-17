from __future__ import annotations

import unittest

from tests.td_case2.continuous_mot_hybrid.bytetrack_backend import ByteTrackBackend
from tests.td_case2.continuous_mot_hybrid.fixed_5fps_validation_core import (
    TrackLifecycleRecord,
    build_validation_checks,
    classify_zone,
    compute_new_id_reason,
    detector_should_run,
    update_confirmation,
)


class FixedFiveFpsValidationTests(unittest.TestCase):
    def test_skipped_detector_frame_does_not_age_tracks(self) -> None:
        backend = ByteTrackBackend(track_high_thresh=0.3, track_low_thresh=0.1, match_thresh=0.8, track_buffer_frames=5)
        self.assertEqual(backend.handle_detector_skipped(), [])

    def test_detector_schedule_and_zone_classification(self) -> None:
        self.assertTrue(detector_should_run(processed_frame_index=0, processing_fps=10.0, detector_fps=5.0))
        self.assertFalse(detector_should_run(processed_frame_index=1, processing_fps=10.0, detector_fps=5.0))
        self.assertEqual(classify_zone([0, 10, 20, 30], frame_width=100, frame_height=100), "left")
        self.assertEqual(classify_zone([40, 40, 60, 60], frame_width=100, frame_height=100), "interior")

    def test_confirmation_and_new_id_reasons(self) -> None:
        record = TrackLifecycleRecord(track_id="vehicle_track_0001", family="vehicle", class_name="car", created_timestamp_seconds=0.0, created_zone="left")
        record.detector_hit_timestamps.extend([0.0, 0.4])
        update_confirmation(record, timestamp_seconds=0.4)
        self.assertTrue(record.confirmed)
        self.assertEqual(
            compute_new_id_reason(
                bbox_xyxy=[40, 40, 60, 60],
                class_name="bus",
                family="vehicle",
                recoverable_tracks=[],
                frame_width=100,
                frame_height=100,
            ),
            "unmatched_detection",
        )

    def test_validation_checks(self) -> None:
        checks = build_validation_checks(
            per_frame_events=[{"detector_ran": False, "lost_track_ids": [], "terminated_track_ids": [], "new_track_ids": []}],
            new_id_events=[{"detector_ran": True}],
            reactivation_events=[{"detector_ran": True}],
            records={"a": TrackLifecycleRecord(track_id="a", family="vehicle", class_name="car", created_timestamp_seconds=0.0, created_zone="left")},
        )
        self.assertTrue(checks["passed"])


if __name__ == "__main__":
    unittest.main()
