from __future__ import annotations

import unittest
from datetime import datetime, timezone

from tests.td_case2.multicamera_vehicle_tracking_pipeline.cross_camera.global_match_models import (
    CrossCameraMatchResult,
    GlobalMatchModelError,
    GlobalObjectMembership,
    GlobalVehicleObjectProposal,
    TrackIdentityFeatures,
)


class GlobalMatchModelTests(unittest.TestCase):
    def test_track_identity_requires_timezone_aware_datetimes(self) -> None:
        with self.assertRaises(GlobalMatchModelError):
            TrackIdentityFeatures(
                vehicle_track_id="track-1",
                track_uuid="RUN:CAM:TRACK_1",
                processing_run_id="run-1",
                camera_id="camera-1",
                camera_code="CAM_001",
                canonical_class="CAR",
                canonical_colour="WHITE",
                colour_confidence=0.9,
                normalized_plate="DL8CBF6268",
                plate_status="VERIFIED",
                plate_confidence=0.95,
                first_seen_at=datetime(2026, 7, 24, 10, 0, 0),
                last_seen_at=datetime(2026, 7, 24, 10, 1, 0),
                first_video_time_seconds=1.0,
                last_video_time_seconds=2.0,
            )

    def test_global_object_requires_members(self) -> None:
        now = datetime(2026, 7, 24, 10, 0, 0, tzinfo=timezone.utc)
        with self.assertRaises(GlobalMatchModelError):
            GlobalVehicleObjectProposal(
                processing_run_id="run-1",
                global_object_code="GVO:1",
                status="CONFIRMED",
                confidence=0.9,
                canonical_plate="DL8CBF6268",
                canonical_colour="WHITE",
                canonical_vehicle_class="CAR",
                first_seen_at=now,
                last_seen_at=now,
                creation_method="VERIFIED_PLATE",
                camera_count=2,
                track_count=2,
                members=(),
            )

    def test_match_result_accepts_valid_scores(self) -> None:
        result = CrossCameraMatchResult(
            left_track_uuid="A",
            right_track_uuid="B",
            left_vehicle_track_id="1",
            right_vehicle_track_id="2",
            decision="CONFIRMED",
            score=0.95,
            plate_score=1.0,
            time_score=1.0,
            camera_route_score=1.0,
            class_score=1.0,
            colour_score=1.0,
            visual_score=0.0,
            reasons=("verified-plate-match",),
            rule_version="global_match_v1",
        )
        membership = GlobalObjectMembership("1", "A", "CONFIRMED", 0.95, "VERIFIED_PLATE")
        self.assertEqual(result.decision, "CONFIRMED")
        self.assertEqual(membership.membership_status, "CONFIRMED")


if __name__ == "__main__":
    unittest.main()
