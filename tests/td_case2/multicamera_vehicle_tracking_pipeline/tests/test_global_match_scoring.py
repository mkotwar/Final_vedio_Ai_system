from __future__ import annotations

import unittest
from datetime import datetime, timezone

from tests.td_case2.multicamera_vehicle_tracking_pipeline.cross_camera.global_match_config import GlobalMatchConfig, TimeMatchingConfig
from tests.td_case2.multicamera_vehicle_tracking_pipeline.cross_camera.global_match_models import TrackIdentityFeatures
from tests.td_case2.multicamera_vehicle_tracking_pipeline.cross_camera.global_match_scoring import evaluate_track_pair


def _track(
    track_id: str,
    camera_id: str,
    camera_code: str,
    *,
    plate: str | None = None,
    plate_status: str | None = None,
    colour: str | None = "WHITE",
    vehicle_class: str | None = "CAR",
) -> TrackIdentityFeatures:
    now = datetime(2026, 7, 24, 10, 0, 0, tzinfo=timezone.utc)
    return TrackIdentityFeatures(
        vehicle_track_id=track_id,
        track_uuid=f"RUN:{camera_code}:TRACK_{track_id}",
        processing_run_id="run-1",
        camera_id=camera_id,
        camera_code=camera_code,
        canonical_class=vehicle_class,
        canonical_colour=colour,
        colour_confidence=0.9,
        normalized_plate=plate,
        plate_status=plate_status,
        plate_confidence=0.95 if plate else None,
        first_seen_at=now,
        last_seen_at=now,
        first_video_time_seconds=10.0,
        last_video_time_seconds=11.0,
    )


class GlobalMatchScoringTests(unittest.TestCase):
    def test_same_verified_plate_across_different_cameras_confirms(self) -> None:
        config = GlobalMatchConfig()
        result = evaluate_track_pair(
            _track("1", "cam-1", "CAM_001", plate="DL8CBF6268", plate_status="VERIFIED"),
            _track("2", "cam-2", "CAM_002", plate="DL8CBF6268", plate_status="VERIFIED"),
            config,
        ).result
        self.assertEqual(result.decision, "CONFIRMED")

    def test_different_verified_plates_reject(self) -> None:
        config = GlobalMatchConfig()
        result = evaluate_track_pair(
            _track("1", "cam-1", "CAM_001", plate="DL8CBF6268", plate_status="VERIFIED"),
            _track("2", "cam-2", "CAM_002", plate="DL8CBF6269", plate_status="VERIFIED"),
            config,
        ).result
        self.assertEqual(result.decision, "REJECTED")

    def test_same_camera_pair_rejects(self) -> None:
        config = GlobalMatchConfig()
        result = evaluate_track_pair(
            _track("1", "cam-1", "CAM_001", plate="DL8CBF6268", plate_status="VERIFIED"),
            _track("2", "cam-1", "CAM_001", plate="DL8CBF6268", plate_status="VERIFIED"),
            config,
        ).result
        self.assertEqual(result.decision, "REJECTED")

    def test_colour_and_class_alone_do_not_confirm(self) -> None:
        config = GlobalMatchConfig()
        result = evaluate_track_pair(
            _track("1", "cam-1", "CAM_001", plate=None, plate_status=None),
            _track("2", "cam-2", "CAM_002", plate=None, plate_status=None),
            config,
        ).result
        self.assertNotEqual(result.decision, "CONFIRMED")

    def test_disabled_time_mode_is_neutral(self) -> None:
        config = GlobalMatchConfig(time_matching=TimeMatchingConfig(mode="disabled"))
        result = evaluate_track_pair(
            _track("1", "cam-1", "CAM_001", plate="DL8CBF6268", plate_status="VERIFIED"),
            _track("2", "cam-2", "CAM_002", plate="DL8CBF6268", plate_status="VERIFIED"),
            config,
        ).result
        self.assertEqual(result.time_score, 0.5)


if __name__ == "__main__":
    unittest.main()
