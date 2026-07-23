from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path

from tests.td_case2.multicamera_vehicle_tracking_pipeline.database.repository import RepositoryConstraintError, SimpleVehicleRepository
from tests.td_case2.multicamera_vehicle_tracking_pipeline.ingestion.camera_config import CameraConfig
from tests.td_case2.multicamera_vehicle_tracking_pipeline.persistence.persistence_config import PersistenceConfig
from tests.td_case2.multicamera_vehicle_tracking_pipeline.persistence.tracking_persistence_service import TrackingPersistenceService
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.tracking_models import LocalVehicleTrack, TrackObservation


def _camera_config(code: str = "CAM_001") -> CameraConfig:
    return CameraConfig(camera_code=code, camera_name=f"Camera {code}", source_path=Path(f"{code}.mp4"), enabled=True, start_time=datetime(2026, 7, 22, 10, 0, 0))


def _observation(frame_number: int, *, camera_code: str = "CAM_001", local_track_id: int = 1) -> TrackObservation:
    return TrackObservation(
        camera_code=camera_code,
        local_track_id=local_track_id,
        frame_number=frame_number,
        video_time_seconds=float(frame_number),
        camera_timestamp=datetime(2026, 7, 22, 10, 0, min(frame_number, 59)),
        class_name="motorcycle",
        confidence=0.8 + (frame_number * 0.01),
        bbox_xyxy=(1.0 + frame_number, 2.0, 10.0 + frame_number, 12.0),
        track_uuid=f"{camera_code}:TRACK_{local_track_id}",
        state="active",
    )


def _track(
    *,
    camera_code: str = "CAM_001",
    local_track_id: int = 1,
    state: str = "completed",
    observation_count: int = 3,
    observations: list[TrackObservation] | None = None,
    class_name: str = "motorcycle",
) -> LocalVehicleTrack:
    selected_observations = observations if observations is not None else [_observation(index, camera_code=camera_code, local_track_id=local_track_id) for index in range(observation_count)]
    return LocalVehicleTrack(
        track_uuid=f"{camera_code}:TRACK_{local_track_id}",
        camera_code=camera_code,
        local_track_id=local_track_id,
        class_name=class_name,
        first_frame_number=selected_observations[0].frame_number,
        last_frame_number=selected_observations[-1].frame_number,
        first_seen_at=selected_observations[0].camera_timestamp,
        last_seen_at=selected_observations[-1].camera_timestamp,
        first_video_time_seconds=selected_observations[0].video_time_seconds,
        last_video_time_seconds=selected_observations[-1].video_time_seconds,
        observation_count=len(selected_observations),
        best_confidence=max(item.confidence for item in selected_observations),
        state=state,
        observations=selected_observations,
        camera_name="North Gate",
        source_path=Path("camera.mp4"),
    )


class _CountingRepository(SimpleVehicleRepository):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def add_vehicle_observations(self, observations):
        self.batch_sizes.append(len(observations))
        return super().add_vehicle_observations(observations)


class _FailingObservationRepository(_CountingRepository):
    def add_vehicle_observations(self, observations):
        raise RepositoryConstraintError("boom")


class TrackingPersistenceServiceTests(unittest.TestCase):
    def test_missing_camera_inserted_and_existing_camera_reused(self) -> None:
        repo = SimpleVehicleRepository()
        service = TrackingPersistenceService(repo, PersistenceConfig())
        service.sync_cameras([_camera_config("CAM_001"), _camera_config("CAM_002")])
        service.sync_cameras([_camera_config("CAM_001")])
        self.assertEqual(len(repo.list_cameras()), 2)
        self.assertIn("CAM_001", service.camera_id_by_code)

    def test_completed_track_inserted(self) -> None:
        repo = SimpleVehicleRepository()
        service = TrackingPersistenceService(repo, PersistenceConfig())
        service.sync_cameras([_camera_config()])
        result = service.save_completed_track(_track())
        self.assertEqual(result.status, "inserted")
        stored = repo.get_track_by_uuid("CAM_001:TRACK_1")
        self.assertIsNotNone(stored)
        self.assertEqual(len(repo.get_track_observations(stored.id)), 3)

    def test_discarded_track_skipped_by_default(self) -> None:
        repo = SimpleVehicleRepository()
        service = TrackingPersistenceService(repo, PersistenceConfig())
        service.sync_cameras([_camera_config()])
        result = service.save_completed_track(_track(state="discarded"))
        self.assertEqual(result.status, "skipped_discarded")

    def test_discarded_track_can_be_inserted_when_enabled(self) -> None:
        repo = SimpleVehicleRepository()
        service = TrackingPersistenceService(repo, PersistenceConfig(include_discarded_tracks=True, write_completed_tracks_only=False))
        service.sync_cameras([_camera_config()])
        result = service.save_completed_track(_track(state="discarded"))
        self.assertEqual(result.status, "inserted")

    def test_active_track_skipped(self) -> None:
        repo = SimpleVehicleRepository()
        service = TrackingPersistenceService(repo, PersistenceConfig())
        service.sync_cameras([_camera_config()])
        result = service.save_completed_track(_track(state="active"))
        self.assertEqual(result.status, "skipped_invalid_state")

    def test_duplicate_track_returns_already_exists(self) -> None:
        repo = SimpleVehicleRepository()
        service = TrackingPersistenceService(repo, PersistenceConfig())
        service.sync_cameras([_camera_config()])
        first = service.save_completed_track(_track())
        second = service.save_completed_track(_track())
        self.assertEqual(first.status, "inserted")
        self.assertEqual(second.status, "already_exists")

    def test_dry_run_performs_no_writes(self) -> None:
        repo = SimpleVehicleRepository()
        service = TrackingPersistenceService(repo, PersistenceConfig(enabled=True, dry_run=True))
        service.sync_cameras([_camera_config()])
        result = service.save_completed_track(_track())
        self.assertEqual(result.status, "dry_run")
        self.assertEqual(repo.list_cameras(), [])
        self.assertIsNone(repo.get_track_by_uuid("CAM_001:TRACK_1"))

    def test_unknown_camera_rejected(self) -> None:
        repo = SimpleVehicleRepository()
        service = TrackingPersistenceService(repo, PersistenceConfig())
        with self.assertRaises(ValueError):
            service.save_completed_track(_track())

    def test_invalid_class_rejected(self) -> None:
        repo = SimpleVehicleRepository()
        service = TrackingPersistenceService(repo, PersistenceConfig())
        service.sync_cameras([_camera_config()])
        with self.assertRaises(ValueError):
            service.save_completed_track(_track(class_name="plane"))

    def test_3wheeler_class_is_accepted(self) -> None:
        repo = SimpleVehicleRepository()
        service = TrackingPersistenceService(repo, PersistenceConfig())
        service.sync_cameras([_camera_config()])
        result = service.save_completed_track(_track(class_name="3Wheeler"))
        self.assertEqual(result.status, "inserted")
        stored = repo.get_track_by_uuid("CAM_001:TRACK_1")
        self.assertIsNotNone(stored)
        self.assertEqual(stored.vehicle_class, "3wheeler")

    def test_invalid_timestamps_rejected(self) -> None:
        repo = SimpleVehicleRepository()
        service = TrackingPersistenceService(repo, PersistenceConfig())
        service.sync_cameras([_camera_config()])
        broken = _track()
        broken.last_seen_at = datetime(2026, 7, 22, 9, 59, 59)
        with self.assertRaises(ValueError):
            service.save_completed_track(broken)

    def test_all_observations_stored(self) -> None:
        repo = SimpleVehicleRepository()
        service = TrackingPersistenceService(repo, PersistenceConfig(observation_mode="all"))
        service.sync_cameras([_camera_config()])
        result = service.save_completed_track(_track(observation_count=4))
        self.assertEqual(result.observations_written, 4)

    def test_sampled_observations_store_first_and_last(self) -> None:
        repo = SimpleVehicleRepository()
        service = TrackingPersistenceService(repo, PersistenceConfig(observation_mode="sampled", observation_sample_every_n=2))
        service.sync_cameras([_camera_config()])
        track = _track(observation_count=5)
        result = service.save_completed_track(track)
        stored = repo.get_track_by_uuid(track.track_uuid)
        frames = [item.frame_number for item in repo.get_track_observations(stored.id)]
        self.assertEqual(result.observations_written, 3)
        self.assertIn(0, frames)
        self.assertIn(4, frames)

    def test_none_mode_stores_no_observations(self) -> None:
        repo = SimpleVehicleRepository()
        service = TrackingPersistenceService(repo, PersistenceConfig(observation_mode="none"))
        service.sync_cameras([_camera_config()])
        result = service.save_completed_track(_track(observation_count=4))
        self.assertEqual(result.observations_written, 0)

    def test_batch_sizes_respected(self) -> None:
        repo = _CountingRepository()
        service = TrackingPersistenceService(repo, PersistenceConfig(observation_batch_size=2))
        service.sync_cameras([_camera_config()])
        service.save_completed_track(_track(observation_count=5))
        self.assertEqual(repo.batch_sizes, [2, 2, 1])

    def test_invalid_bbox_rejected(self) -> None:
        repo = SimpleVehicleRepository()
        service = TrackingPersistenceService(repo, PersistenceConfig())
        service.sync_cameras([_camera_config()])
        bad_observation = TrackObservation(
            camera_code="CAM_001",
            local_track_id=1,
            frame_number=0,
            video_time_seconds=0.0,
            camera_timestamp=datetime(2026, 7, 22, 10, 0, 0),
            class_name="motorcycle",
            confidence=0.9,
            bbox_xyxy=(5.0, 5.0, 5.0, 7.0),
            track_uuid="CAM_001:TRACK_1",
            state="active",
        )
        with self.assertRaises(ValueError):
            service.save_completed_track(_track(observations=[bad_observation]))

    def test_missing_timestamp_rejected(self) -> None:
        repo = SimpleVehicleRepository()
        service = TrackingPersistenceService(repo, PersistenceConfig())
        service.sync_cameras([_camera_config()])
        bad_observation = TrackObservation(
            camera_code="CAM_001",
            local_track_id=1,
            frame_number=0,
            video_time_seconds=0.0,
            camera_timestamp=None,
            class_name="motorcycle",
            confidence=0.9,
            bbox_xyxy=(1.0, 2.0, 5.0, 7.0),
            track_uuid="CAM_001:TRACK_1",
            state="active",
        )
        with self.assertRaises(ValueError):
            service.save_completed_track(_track(observations=[bad_observation]))

    def test_wrong_camera_rejected(self) -> None:
        repo = SimpleVehicleRepository()
        service = TrackingPersistenceService(repo, PersistenceConfig())
        service.sync_cameras([_camera_config()])
        with self.assertRaises(ValueError):
            service.save_completed_track(_track(observations=[_observation(0, camera_code="CAM_002")]))

    def test_wrong_local_track_id_rejected(self) -> None:
        repo = SimpleVehicleRepository()
        service = TrackingPersistenceService(repo, PersistenceConfig())
        service.sync_cameras([_camera_config()])
        with self.assertRaises(ValueError):
            service.save_completed_track(_track(observations=[_observation(0, local_track_id=99)]))

    def test_database_error_behavior_follows_configuration(self) -> None:
        repo = _FailingObservationRepository()
        service = TrackingPersistenceService(repo, PersistenceConfig(fail_on_database_error=False))
        service.sync_cameras([_camera_config()])
        result = service.save_completed_track(_track())
        self.assertEqual(result.status, "failed")


if __name__ == "__main__":
    unittest.main()
