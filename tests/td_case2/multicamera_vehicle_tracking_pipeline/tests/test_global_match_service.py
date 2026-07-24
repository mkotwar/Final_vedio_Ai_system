from __future__ import annotations

import unittest
from datetime import datetime, timezone

from tests.td_case2.multicamera_vehicle_tracking_pipeline.cross_camera.global_match_config import GlobalMatchConfig
from tests.td_case2.multicamera_vehicle_tracking_pipeline.cross_camera.global_match_models import TrackIdentityFeatures
from tests.td_case2.multicamera_vehicle_tracking_pipeline.cross_camera.global_match_service import GlobalMatchService


def _track(track_id: str, camera_code: str, *, plate: str | None = None, plate_status: str | None = None) -> TrackIdentityFeatures:
    now = datetime(2026, 7, 24, 10, 0, 0, tzinfo=timezone.utc)
    return TrackIdentityFeatures(
        vehicle_track_id=track_id,
        track_uuid=f"RUN_20260724_151402:{camera_code}:TRACK_{track_id}",
        processing_run_id="run-1",
        camera_id=camera_code.lower(),
        camera_code=camera_code,
        canonical_class="CAR",
        canonical_colour="WHITE",
        colour_confidence=0.9,
        normalized_plate=plate,
        plate_status=plate_status,
        plate_confidence=0.95 if plate else None,
        first_seen_at=now,
        last_seen_at=now,
        first_video_time_seconds=10.0,
        last_video_time_seconds=11.0,
    )


class _FakeMatchRepository:
    def __init__(self, tracks: list[TrackIdentityFeatures]) -> None:
        self.tracks = tracks
        self.upserts: list[tuple[str, object, str | None]] = []

    def find_run_by_code(self, run_code: str):
        return {"id": "run-1", "run_code": run_code}

    def find_tracks_for_run(self, processing_run_id: str):
        return list(self.tracks)

    def upsert_match(self, processing_run_id: str, result, *, global_vehicle_id: str | None = None):
        self.upserts.append((processing_run_id, result, global_vehicle_id))
        return {"id": "match-1"}


class _FakeGlobalObjectRepository:
    def __init__(self) -> None:
        self.objects = []
        self.members = []

    def create_or_get_global_object(self, proposal):
        row = {"id": proposal.global_object_code, "global_vehicle_code": proposal.global_object_code}
        self.objects.append(proposal)
        return row

    def add_or_update_member(self, global_vehicle_id: str, membership):
        self.members.append((global_vehicle_id, membership))
        return {"id": f"member-{membership.vehicle_track_id}"}


class GlobalMatchServiceTests(unittest.TestCase):
    def test_confirmed_pair_creates_one_multi_camera_object(self) -> None:
        tracks = [
            _track("1", "CAM_001", plate="DL8CBF6268", plate_status="VERIFIED"),
            _track("2", "CAM_002", plate="DL8CBF6268", plate_status="VERIFIED"),
        ]
        service = GlobalMatchService(GlobalMatchConfig(), _FakeMatchRepository(tracks), _FakeGlobalObjectRepository())
        report = service.build_for_run("RUN_20260724_151402", persist=False)
        self.assertEqual(report.decisions["confirmed"], 1)
        self.assertEqual(report.multi_camera_objects, 1)
        self.assertEqual(report.single_track_objects, 0)

    def test_unmatched_track_creates_single_track_object(self) -> None:
        tracks = [_track("1", "CAM_001", plate=None, plate_status=None)]
        service = GlobalMatchService(GlobalMatchConfig(), _FakeMatchRepository(tracks), _FakeGlobalObjectRepository())
        report = service.build_for_run("RUN_20260724_151402", persist=False)
        self.assertEqual(report.single_track_objects, 1)
        self.assertEqual(report.multi_camera_objects, 0)


if __name__ == "__main__":
    unittest.main()
