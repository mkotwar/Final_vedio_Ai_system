from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from uuid import UUID, uuid4

from tests.td_case2.multicamera_vehicle_tracking_pipeline.detection.detection_config import DetectionConfig
from tests.td_case2.multicamera_vehicle_tracking_pipeline.evidence.evidence_models import EvidenceCandidate, TrackEvidencePackage
from tests.td_case2.multicamera_vehicle_tracking_pipeline.ingestion.camera_config import CameraConfig
from tests.td_case2.multicamera_vehicle_tracking_pipeline.persistence.analytics_persistence_service import AnalyticsPersistenceService
from tests.td_case2.multicamera_vehicle_tracking_pipeline.persistence.persistence_config import PersistenceConfig
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.tracking_config import TrackingConfig
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.tracking_models import LocalVehicleTrack, TrackObservation


def _utc(second: int = 0) -> datetime:
    return datetime(2026, 7, 23, 12, 0, second, tzinfo=timezone.utc)


def _camera_config(code: str = "CAM_001") -> CameraConfig:
    return CameraConfig(
        camera_code=code,
        camera_name=f"Camera {code}",
        source_path=Path(f"{code}.mp4"),
        enabled=True,
        start_time=_utc(),
    )


def _observation(frame_number: int, *, camera_code: str = "CAM_001", local_track_id: int = 1) -> TrackObservation:
    return TrackObservation(
        camera_code=camera_code,
        local_track_id=local_track_id,
        frame_number=frame_number,
        video_time_seconds=float(frame_number),
        camera_timestamp=_utc(frame_number),
        class_name="car",
        confidence=0.8,
        bbox_xyxy=(1.0, 2.0, 5.0, 6.0),
        track_uuid=f"{camera_code}:TRACK_{local_track_id}",
        state="active",
    )


def _track(observation_count: int = 3) -> LocalVehicleTrack:
    observations = [_observation(index) for index in range(observation_count)]
    track = LocalVehicleTrack(
        track_uuid="CAM_001:TRACK_1",
        camera_code="CAM_001",
        local_track_id=1,
        class_name="car",
        first_frame_number=0,
        last_frame_number=observation_count - 1,
        first_seen_at=observations[0].camera_timestamp,
        last_seen_at=observations[-1].camera_timestamp,
        first_video_time_seconds=observations[0].video_time_seconds,
        last_video_time_seconds=observations[-1].video_time_seconds,
        observation_count=len(observations),
        best_confidence=0.8,
        state="completed",
        observations=observations,
        camera_name="North Gate",
        source_path=Path("camera.mp4"),
    )
    track.stable_class_name = "car"
    track.provisional_class_name = "car"
    track.class_is_locked = True
    track.class_confidence = 0.88
    track.class_winner_margin = 1.25
    track.class_observation_count = observation_count
    track.class_scores = {"car": 2.4}
    track.class_observation_counts = {"car": observation_count}
    track.class_max_confidences = {"car": 0.8}
    return track


class _NoopClient:
    def table(self, table_name: str):
        raise AssertionError(f"Unexpected table access: {table_name}")


class _FakeUpsertRepository:
    def __init__(self, row_id: UUID | None = None) -> None:
        self.row_id = row_id or uuid4()
        self.records = []

    def upsert_camera(self, record):
        self.records.append(record)
        return {"id": str(self.row_id), "camera_code": record.camera_code}

    def upsert_processing_run(self, record):
        self.records.append(record)
        return {"id": str(self.row_id), "run_code": record.run_code}

    def upsert_camera_run(self, record):
        self.records.append(record)
        return {"id": str(self.row_id)}

    def update_camera_run_by_id(self, row_id, record):
        self.records.append((row_id, record))
        return {"id": str(row_id)}

    def upsert_ai_model(self, record):
        self.records.append(record)
        return {"id": str(uuid4())}

    def upsert_run_model(self, record):
        self.records.append(record)
        return {"id": str(uuid4())}

    def upsert_vehicle_track(self, record):
        self.records.append(record)
        return {"id": str(self.row_id), "track_uuid": record.track_uuid}


class _FakeInsertRepository:
    def __init__(self, row_id: UUID | None = None) -> None:
        self.row_id = row_id or uuid4()
        self.records = []

    def create_video_source(self, record):
        self.records.append(record)
        return {"id": str(self.row_id)}

    def create_processing_job(self, record):
        self.records.append(record)
        return {"id": str(self.row_id)}

    def update_processing_job_by_id(self, row_id, record):
        self.records.append((row_id, record))
        return {"id": str(row_id)}

    def create_processing_error(self, record):
        self.records.append(record)
        return {"id": str(uuid4())}


class _FakeVehicleTrackRepository(_FakeUpsertRepository):
    def __init__(self, row_id: UUID | None = None) -> None:
        super().__init__(row_id)
        self.existing_by_uuid = {}

    def get_vehicle_track_by_uuid(self, track_uuid: str):
        return self.existing_by_uuid.get(track_uuid)


class _FakeObservationRepository:
    def __init__(self) -> None:
        self.batches = []

    def upsert_observations_batch(self, records):
        self.batches.append(records)
        return [{"id": index + 1} for index, _ in enumerate(records)]


class _FakeTrackMediaRepository:
    def __init__(self) -> None:
        self.records = []

    def bulk_upsert(self, records):
        self.records.append(records)
        from tests.td_case2.multicamera_vehicle_tracking_pipeline.persistence.track_media_repository import TrackMediaBatchResult

        return TrackMediaBatchResult(attempted=len(records), inserted=len(records), already_existing=0, failed=0)


class AnalyticsPersistenceServiceTests(unittest.TestCase):
    def _build_service(
        self,
        *,
        config: PersistenceConfig | None = None,
        enable_database_writes: bool = True,
    ) -> AnalyticsPersistenceService:
        self.tempdir = tempfile.TemporaryDirectory()
        artifact_root = Path(self.tempdir.name)
        service = AnalyticsPersistenceService(
            _NoopClient() if enable_database_writes else None,
            config or PersistenceConfig(backend="analytics_supabase", observation_batch_size=2, persist_track_media=True, track_media_roles=("BEST_OVERALL",)),
            run_code="RUN_TEST",
            detection_config=DetectionConfig(model_path="yolov8n.pt"),
            tracking_config=TrackingConfig(min_confirmed_observations=1),
            execution_mode="THREADED",
            runtime_device="cpu",
            artifact_root=artifact_root,
            enable_database_writes=enable_database_writes,
        )
        if enable_database_writes:
            service._camera_repository = _FakeUpsertRepository(uuid4())
            service._video_source_repository = _FakeInsertRepository(uuid4())
            service._processing_run_repository = _FakeUpsertRepository(uuid4())
            service._camera_run_repository = _FakeUpsertRepository(uuid4())
            service._processing_job_repository = _FakeInsertRepository(uuid4())
            service._vehicle_track_repository = _FakeVehicleTrackRepository(uuid4())
            service._track_observation_repository = _FakeObservationRepository()
            service._ai_model_repository = _FakeUpsertRepository(uuid4())
            service._run_model_repository = _FakeUpsertRepository(uuid4())
            service._processing_error_repository = _FakeInsertRepository(uuid4())
            service._track_media_repository = _FakeTrackMediaRepository()
        return service

    def tearDown(self) -> None:
        if hasattr(self, "tempdir"):
            self.tempdir.cleanup()

    def _attach_best_overall_evidence(self, track: LocalVehicleTrack) -> None:
        target = Path(self.tempdir.name) / "RUN_TEST" / "CAM_001" / "track_000001" / "RUN_TEST_CAM_001_TRACK_1" / "best_overall.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"jpeg")
        track.evidence_package = TrackEvidencePackage(
            run_id="RUN_TEST",
            camera_code="CAM_001",
            local_track_id=1,
            track_uuid=track.track_uuid,
            class_name=track.class_name,
            candidates={
                "best_overall": EvidenceCandidate(
                    candidate_type="best_overall",
                    frame_number=1,
                    video_time_seconds=1.0,
                    confidence=0.8,
                    original_bbox_xyxy=(1.0, 2.0, 5.0, 6.0),
                    expanded_bbox_xyxy=(0.0, 1.0, 6.0, 7.0),
                    bbox_xyxy=(1.0, 2.0, 5.0, 6.0),
                    crop_width=100,
                    crop_height=50,
                    area=5000,
                    sharpness_score=100.0,
                    visibility_score=0.9,
                    centeredness_score=0.8,
                    visible_bbox_ratio=0.95,
                    edge_penalty=0.1,
                    overall_score=0.9,
                    crop_clipped=False,
                    touches_left_edge=False,
                    touches_right_edge=False,
                    touches_top_edge=False,
                    touches_bottom_edge=False,
                    encoded_jpeg=b"jpeg",
                    file_path=str(target),
                )
            },
            output_directory=str(target.parent),
        )

    def test_sync_cameras_initializes_processing_run_models_and_camera_context(self) -> None:
        service = self._build_service()
        result = service.sync_cameras([_camera_config("CAM_001"), _camera_config("CAM_002")])
        self.assertEqual(set(result), {"CAM_001", "CAM_002"})
        self.assertEqual(service.get_metrics().cameras_synced, 2)
        self.assertEqual(len(service._ai_model_repository.records), 2)
        self.assertEqual(len(service._run_model_repository.records), 2)

    def test_run_started_at_override_is_used_for_processing_run(self) -> None:
        fixed_started_at = _utc(15)
        self.tempdir = tempfile.TemporaryDirectory()
        service = AnalyticsPersistenceService(
            None,
            PersistenceConfig(backend="dry_run"),
            run_code="RUN_TEST",
            detection_config=DetectionConfig(model_path="yolov8n.pt"),
            tracking_config=TrackingConfig(min_confirmed_observations=1),
            execution_mode="THREADED",
            runtime_device="cpu",
            artifact_root=Path(self.tempdir.name),
            enable_database_writes=False,
            run_started_at=fixed_started_at,
        )
        service.sync_cameras(
            [
                CameraConfig(
                    camera_code="CAM_001",
                    camera_name="Camera CAM_001",
                    source_path=Path("CAM_001.mp4"),
                    enabled=True,
                    start_time=fixed_started_at,
                )
            ]
        )
        self.assertEqual(service._started_at, fixed_started_at)

    def test_save_completed_track_batches_observations(self) -> None:
        service = self._build_service()
        service.sync_cameras([_camera_config()])
        track = _track(observation_count=5)
        self._attach_best_overall_evidence(track)
        result = service.save_completed_track(track)
        self.assertEqual(result.status, "inserted")
        self.assertEqual(result.observations_written, 5)
        self.assertEqual([len(batch) for batch in service._track_observation_repository.batches], [2, 2, 1])
        stored_track = service._vehicle_track_repository.records[0]
        self.assertEqual(stored_track.vehicle_class, "CAR")
        self.assertEqual(stored_track.metadata["class_diagnostics"]["stable_class_name"], "car")
        self.assertEqual(result.media_persistence["attempted"], 1)
        self.assertEqual(len(service._track_media_repository.records), 1)

    def test_dry_run_validates_media_without_writes(self) -> None:
        service = self._build_service(
            config=PersistenceConfig(backend="dry_run", observation_batch_size=2, persist_track_media=True, track_media_roles=("BEST_OVERALL",)),
            enable_database_writes=False,
        )
        service.sync_cameras([_camera_config()])
        track = _track(observation_count=2)
        self._attach_best_overall_evidence(track)
        result = service.save_completed_track(track)
        self.assertEqual(result.status, "dry_run")
        self.assertEqual(result.database_track_id, "DRYRUN:TRACK:CAM_001:TRACK_1")
        self.assertEqual(result.media_persistence["attempted"], 1)
        self.assertEqual(result.media_persistence["validated"], 1)
        self.assertEqual(result.media_persistence["mode"], "dry_run")
        self.assertEqual(service.get_metrics().media_records_validated, 1)
        self.assertEqual(service.camera_id_by_code["CAM_001"], "DRYRUN:CAMERA:CAM_001")

    def test_dry_run_missing_evidence_file_increments_missing_metric(self) -> None:
        service = self._build_service(
            config=PersistenceConfig(backend="dry_run", observation_batch_size=2, persist_track_media=True, track_media_roles=("BEST_OVERALL",)),
            enable_database_writes=False,
        )
        service.sync_cameras([_camera_config()])
        track = _track(observation_count=2)
        missing_path = Path(self.tempdir.name) / "RUN_TEST" / "CAM_001" / "track_000001" / "RUN_TEST_CAM_001_TRACK_1" / "best_overall.jpg"
        track.evidence_package = TrackEvidencePackage(
            run_id="RUN_TEST",
            camera_code="CAM_001",
            local_track_id=1,
            track_uuid=track.track_uuid,
            class_name=track.class_name,
            candidates={
                "best_overall": EvidenceCandidate(
                    candidate_type="best_overall",
                    frame_number=1,
                    video_time_seconds=1.0,
                    confidence=0.8,
                    original_bbox_xyxy=(1.0, 2.0, 5.0, 6.0),
                    expanded_bbox_xyxy=(0.0, 1.0, 6.0, 7.0),
                    bbox_xyxy=(1.0, 2.0, 5.0, 6.0),
                    crop_width=100,
                    crop_height=50,
                    area=5000,
                    sharpness_score=100.0,
                    visibility_score=0.9,
                    centeredness_score=0.8,
                    visible_bbox_ratio=0.95,
                    edge_penalty=0.1,
                    overall_score=0.9,
                    crop_clipped=False,
                    touches_left_edge=False,
                    touches_right_edge=False,
                    touches_top_edge=False,
                    touches_bottom_edge=False,
                    encoded_jpeg=b"jpeg",
                    file_path=str(missing_path),
                )
            },
            output_directory=str(missing_path.parent),
        )
        result = service.save_completed_track(track)
        self.assertEqual(result.status, "dry_run")
        self.assertEqual(result.media_persistence["missing_files"], 1)
        self.assertEqual(service.get_metrics().media_files_missing, 1)

    def test_finalize_run_updates_processing_run_and_camera_runs(self) -> None:
        service = self._build_service()
        service.sync_cameras([_camera_config()])
        service.finalize_run(
            {
                "camera_count": 1,
                "total_frames_processed": 10,
                "total_detections": 5,
                "total_track_observations": 5,
                "total_completed_tracks": 1,
                "cameras": {
                    "CAM_001": {
                        "frames_read": 10,
                        "frames_processed": 10,
                        "detections": 5,
                        "track_observations": 5,
                        "completed_tracks": 1,
                        "discarded_tracks": 0,
                        "resolved_source_fps": 25.0,
                        "first_frame_number": 0,
                        "last_frame_number": 9,
                        "errors": [],
                    }
                },
                "errors": [],
            }
        )
        self.assertEqual(len(service._camera_run_repository.records), 2)
        self.assertGreaterEqual(len(service._processing_run_repository.records), 2)


if __name__ == "__main__":
    unittest.main()
