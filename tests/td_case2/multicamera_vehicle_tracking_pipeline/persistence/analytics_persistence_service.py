from __future__ import annotations

import math
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from ..detection.detection_config import DetectionConfig
from ..ingestion.camera_config import CameraConfig
from ..tracking.tracking_config import TrackingConfig
from ..tracking.tracking_models import LocalVehicleTrack, TrackObservation
from .evidence_to_track_media_mapper import build_track_media_records
from .analytics_database_client import AnalyticsDatabaseClient
from .analytics_repositories import (
    AnalyticsAiModelRepository,
    AnalyticsCameraRepository,
    AnalyticsCameraRunRepository,
    AnalyticsProcessingErrorRepository,
    AnalyticsProcessingJobRepository,
    AnalyticsProcessingRunRepository,
    AnalyticsRunModelRepository,
    AnalyticsTrackObservationRepository,
    AnalyticsVehicleTrackRepository,
    AnalyticsVideoSourceRepository,
)
from .analytics_repository_base import AnalyticsRepositoryError
from .persistence_config import PersistenceConfig
from .persistence_models import (
    AiModelRecord,
    CameraRecord,
    CameraRunRecord,
    PersistenceRunMetrics,
    ProcessingErrorRecord,
    ProcessingJobRecord,
    ProcessingRunRecord,
    RunModelRecord,
    TrackObservationRecord,
    TrackPersistenceResult,
    TrackMediaRecord,
    VehicleTrackRecord,
    VideoSourceRecord,
)
from .track_media_repository import TrackMediaBatchResult, TrackMediaRepository
from .vehicle_class_mapping import is_supported_vehicle_class


class AnalyticsPersistenceValidationError(ValueError):
    """Raised when a track or observation cannot be translated to the analytics schema."""


@dataclass(slots=True)
class _CameraPersistenceContext:
    camera_id: UUID
    camera_run_id: UUID
    video_source_id: UUID
    persist_job_id: UUID | None


class AnalyticsPersistenceService:
    def __init__(
        self,
        client: AnalyticsDatabaseClient | None,
        config: PersistenceConfig,
        *,
        run_code: str,
        detection_config: DetectionConfig,
        tracking_config: TrackingConfig,
        execution_mode: str,
        pipeline_name: str = "multicamera_vehicle_tracking_pipeline",
        pipeline_version: str = "phase1",
        runtime_device: str | None = None,
        track_media_repository: TrackMediaRepository | None = None,
        artifact_root: Path | None = None,
        enable_database_writes: bool | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.run_code = str(run_code).strip()
        self.detection_config = detection_config
        self.tracking_config = tracking_config
        self.execution_mode = str(execution_mode).strip().upper() or "THREADED"
        self.pipeline_name = pipeline_name
        self.pipeline_version = pipeline_version
        self.runtime_device = runtime_device
        self.artifact_root = (artifact_root or Path("artifacts")).resolve()
        self.enable_database_writes = (config.backend == "analytics_supabase") if enable_database_writes is None else bool(enable_database_writes)
        self.metrics = PersistenceRunMetrics()
        self.camera_id_by_code: dict[str, object] = {}
        self._processing_run_id: UUID | None = None
        self._started_at: datetime | None = None
        self._camera_context_by_code: dict[str, _CameraPersistenceContext] = {}
        self._detector_model_id: UUID | None = None
        self._tracker_model_id: UUID | None = None
        self._camera_repository = AnalyticsCameraRepository(client) if self.enable_database_writes and client is not None else None
        self._video_source_repository = AnalyticsVideoSourceRepository(client) if self.enable_database_writes and client is not None else None
        self._processing_run_repository = AnalyticsProcessingRunRepository(client) if self.enable_database_writes and client is not None else None
        self._camera_run_repository = AnalyticsCameraRunRepository(client) if self.enable_database_writes and client is not None else None
        self._processing_job_repository = AnalyticsProcessingJobRepository(client) if self.enable_database_writes and client is not None else None
        self._vehicle_track_repository = AnalyticsVehicleTrackRepository(client) if self.enable_database_writes and client is not None else None
        self._track_observation_repository = AnalyticsTrackObservationRepository(client) if self.enable_database_writes and client is not None else None
        self._ai_model_repository = AnalyticsAiModelRepository(client) if self.enable_database_writes and client is not None else None
        self._run_model_repository = AnalyticsRunModelRepository(client) if self.enable_database_writes and client is not None else None
        self._processing_error_repository = AnalyticsProcessingErrorRepository(client) if self.enable_database_writes and client is not None else None
        self._track_media_repository = track_media_repository or (TrackMediaRepository(client) if self.enable_database_writes and client is not None else None)
        if self.enable_database_writes and client is None:
            raise AnalyticsPersistenceValidationError("Analytics database client is required when database writes are enabled.")

    def sync_cameras(self, camera_configs: list[CameraConfig]) -> dict[str, object]:
        self._ensure_processing_run(camera_configs)
        synced = 0
        for camera_config in camera_configs:
            camera_record = CameraRecord(
                camera_code=camera_config.camera_code,
                camera_name=camera_config.camera_name,
                timezone_name=self._camera_timezone_name(camera_config),
                enabled=bool(camera_config.enabled),
                metadata={"source_path": str(camera_config.source_path)},
            )
            camera_record.to_payload()
            camera_ref = self._dry_run_token("CAMERA", camera_config.camera_code)
            if self.enable_database_writes:
                camera_row = self._require_repository(self._camera_repository, "camera").upsert_camera(camera_record)
                camera_id = UUID(str(camera_row["id"]))
                self.camera_id_by_code[camera_config.camera_code] = camera_id
            else:
                camera_id = self._dry_run_uuid("CAMERA", camera_config.camera_code)
                self.camera_id_by_code[camera_config.camera_code] = camera_ref
            video_source_record = VideoSourceRecord(
                camera_id=camera_id,
                source_type="LOCAL_FILE",
                source_reference=str(camera_config.source_path),
                source_start_at=self._ensure_optional_timezone(camera_config.start_time),
                metadata={"camera_name": camera_config.camera_name},
            )
            video_source_record.to_payload()
            if self.enable_database_writes:
                video_source_row = self._require_repository(self._video_source_repository, "video_source").create_video_source(video_source_record)
                video_source_id = UUID(str(video_source_row["id"]))
            else:
                video_source_id = self._dry_run_uuid("VIDEO_SOURCE", camera_config.camera_code, str(camera_config.source_path))
            camera_run_record = CameraRunRecord(
                processing_run_id=self._require_processing_run_id(),
                camera_id=camera_id,
                video_source_id=video_source_id,
                status="RUNNING",
                started_at=self._started_at,
                metrics={},
            )
            camera_run_record.to_payload()
            if self.enable_database_writes:
                camera_run_row = self._require_repository(self._camera_run_repository, "camera_run").upsert_camera_run(camera_run_record)
                camera_run_id = UUID(str(camera_run_row["id"]))
            else:
                camera_run_id = self._dry_run_uuid("CAMERA_RUN", self.run_code, camera_config.camera_code)
            persist_job_id: UUID | None = None
            processing_job_record = ProcessingJobRecord(
                processing_run_id=self._require_processing_run_id(),
                camera_run_id=camera_run_id,
                job_type="PERSIST",
                status="RUNNING",
                worker_name="persistence_worker",
                started_at=self._started_at,
                input_summary={"camera_code": camera_config.camera_code},
            )
            processing_job_record.to_payload()
            if self.enable_database_writes:
                job_row = self._require_repository(self._processing_job_repository, "processing_job").create_processing_job(
                    processing_job_record
                )
                persist_job_id = UUID(str(job_row["id"]))
            elif self.config.dry_run:
                persist_job_id = self._dry_run_uuid("PROCESSING_JOB", self.run_code, camera_config.camera_code, "PERSIST")
            self._camera_context_by_code[camera_config.camera_code] = _CameraPersistenceContext(
                camera_id=camera_id,
                camera_run_id=camera_run_id,
                video_source_id=video_source_id,
                persist_job_id=persist_job_id,
            )
            synced += 1
        self.metrics.cameras_synced = synced
        return dict(self.camera_id_by_code)

    def save_completed_track(self, track: LocalVehicleTrack) -> TrackPersistenceResult:
        self.metrics.tracks_considered += 1
        try:
            result = self._save_track_internal(track)
        except Exception as exc:
            self.metrics.tracks_failed += 1
            message = f"{track.track_uuid}: {exc}"
            self.metrics.errors.append(message)
            self._record_processing_error(
                exc,
                track=track,
                stage_name="PERSIST",
                structured_context={"track_uuid": track.track_uuid, "camera_code": track.camera_code},
            )
            if self.config.fail_on_database_error:
                raise
            return TrackPersistenceResult(track_uuid=track.track_uuid, status="failed", database_track_id=None, observations_written=0, error=str(exc))
        self._apply_metrics(result)
        return result

    def save_completed_tracks(self, tracks: list[LocalVehicleTrack]) -> list[TrackPersistenceResult]:
        return [self.save_completed_track(track) for track in tracks]

    def get_metrics(self) -> PersistenceRunMetrics:
        return self.metrics

    def finalize_run(self, report: dict[str, object]) -> None:
        if self._processing_run_id is None or self.config.dry_run or not self.enable_database_writes:
            return
        completed_at = datetime.now(timezone.utc)
        camera_payloads = report.get("cameras")
        if isinstance(camera_payloads, dict):
            for camera_code, payload in camera_payloads.items():
                if not isinstance(camera_code, str) or not isinstance(payload, dict):
                    continue
                context = self._camera_context_by_code.get(camera_code)
                if context is None:
                    continue
                worker_name = payload.get("reader_worker_name")
                camera_run_status = "FAILED" if payload.get("errors") else "COMPLETED"
                frames_processed = int(payload.get("frames_processed", payload.get("frames_read", 0)) or 0)
                run_row = self._require_repository(self._camera_run_repository, "camera_run").update_camera_run_by_id(
                    context.camera_run_id,
                    CameraRunRecord(
                        processing_run_id=self._require_processing_run_id(),
                        camera_id=context.camera_id,
                        video_source_id=context.video_source_id,
                        status=camera_run_status,
                        reader_worker_name=str(worker_name) if worker_name else None,
                        resolved_source_fps=self._optional_float(payload.get("resolved_source_fps")),
                        effective_processing_fps=self._optional_processing_fps(payload),
                        first_frame_number=self._optional_int(payload.get("first_frame_number", payload.get("first_frame"))),
                        last_frame_number=self._optional_int(payload.get("last_frame_number", payload.get("last_frame"))),
                        frames_read=int(payload.get("frames_read", frames_processed) or 0),
                        frames_processed=frames_processed,
                        detections_count=int(payload.get("detections", 0) or 0),
                        track_observations_count=int(payload.get("track_observations", 0) or 0),
                        completed_tracks_count=int(payload.get("completed_tracks", 0) or 0),
                        discarded_tracks_count=int(payload.get("discarded_tracks", 0) or 0),
                        started_at=self._started_at,
                        completed_at=completed_at,
                        metrics={"report_snapshot": payload},
                    ),
                )
                if context.persist_job_id is not None:
                    self._require_repository(self._processing_job_repository, "processing_job").update_processing_job_by_id(
                        context.persist_job_id,
                        ProcessingJobRecord(
                            processing_run_id=self._require_processing_run_id(),
                            camera_run_id=context.camera_run_id,
                            job_type="PERSIST",
                            status="FAILED" if camera_run_status == "FAILED" else "COMPLETED",
                            worker_name="persistence_worker",
                            started_at=self._started_at,
                            completed_at=completed_at,
                            output_summary={"camera_run_id": run_row["id"], "camera_code": camera_code},
                            error_message="camera_report_contains_errors" if camera_run_status == "FAILED" else None,
                        )
                    )
        persistence_errors = report.get("errors")
        if isinstance(persistence_errors, list):
            for error_row in persistence_errors:
                if not isinstance(error_row, dict):
                    continue
                camera_run_id = None
                camera_code = error_row.get("camera_code")
                if isinstance(camera_code, str) and camera_code in self._camera_context_by_code:
                    camera_run_id = self._camera_context_by_code[camera_code].camera_run_id
                try:
                    self._require_repository(self._processing_error_repository, "processing_error").create_processing_error(
                        ProcessingErrorRecord(
                            processing_run_id=self._require_processing_run_id(),
                            camera_run_id=camera_run_id,
                            stage_name=str(error_row.get("worker_type") or error_row.get("worker_name") or "PIPELINE"),
                            worker_name=str(error_row.get("worker_name")) if error_row.get("worker_name") else None,
                            severity="ERROR" if error_row.get("fatal") else "WARNING",
                            exception_type=str(error_row.get("error_type")) if error_row.get("error_type") else None,
                            message=str(error_row.get("error_message") or "pipeline_error"),
                            structured_context={"error_row": error_row},
                        )
                    )
                except Exception:
                    self.metrics.errors.append(f"processing_error_finalize:{camera_code or 'global'}")
        self._require_repository(self._processing_run_repository, "processing_run").upsert_processing_run(
            ProcessingRunRecord(
                run_code=self.run_code,
                pipeline_name=self.pipeline_name,
                pipeline_version=self.pipeline_version,
                execution_mode=self.execution_mode,
                status="FAILED" if self.metrics.tracks_failed or persistence_errors else "COMPLETED",
                configured_camera_count=int(report.get("configured_camera_count", report.get("camera_count", len(self._camera_context_by_code))) or 0),
                active_camera_count=int(report.get("selected_camera_count", report.get("camera_count", len(self._camera_context_by_code))) or 0),
                started_at=self._started_at,
                completed_at=completed_at,
                total_frames_processed=int(report.get("total_frames_processed", 0) or 0),
                total_detections=int(report.get("total_detections", 0) or 0),
                total_track_observations=int(report.get("total_track_observations", 0) or 0),
                total_tracks=int(report.get("total_completed_tracks", 0) or 0),
                host_name=socket.gethostname(),
                runtime_device=self.runtime_device,
                configuration=self._configuration_payload(),
                metrics={"report_snapshot": report},
            )
        )

    def _save_track_internal(self, track: LocalVehicleTrack) -> TrackPersistenceResult:
        if track.state == "discarded" and not self.config.include_discarded_tracks:
            return TrackPersistenceResult(track_uuid=track.track_uuid, status="skipped_discarded", database_track_id=None, observations_written=0)
        if self.config.write_completed_tracks_only and track.state != "completed":
            return TrackPersistenceResult(track_uuid=track.track_uuid, status="skipped_invalid_state", database_track_id=None, observations_written=0)
        self._validate_track(track)
        context = self._require_camera_context(track.camera_code)
        dry_run_track_ref = self._dry_run_token("TRACK", track.track_uuid)
        dry_run_track_uuid = self._dry_run_uuid("TRACK", track.track_uuid)
        track_record = VehicleTrackRecord(
            processing_run_id=self._require_processing_run_id(),
            camera_run_id=context.camera_run_id,
            camera_id=context.camera_id,
            track_uuid=track.track_uuid,
            local_track_id=track.local_track_id,
            vehicle_class=track.class_name,
            first_seen_at=self._ensure_timezone(track.first_seen_at),
            last_seen_at=self._ensure_timezone(track.last_seen_at),
            first_frame_number=track.first_frame_number,
            last_frame_number=track.last_frame_number,
            first_video_time_seconds=track.first_video_time_seconds,
            last_video_time_seconds=track.last_video_time_seconds,
            observation_count=track.observation_count,
            best_detection_confidence=track.best_confidence,
            average_detection_confidence=self._average_detection_confidence(track.observations),
            lifecycle_state="DISCARDED" if track.state == "discarded" else "COMPLETED",
            completion_reason=track.state,
            tracker_backend=self.tracking_config.backend,
            tracker_configuration=self._tracking_configuration_payload(),
            searchable=True,
            metadata={"camera_name": track.camera_name, "source_path": str(track.source_path) if track.source_path else None},
        )
        track_record.to_payload()
        observation_records = self._build_observation_records(track, vehicle_track_id=dry_run_track_uuid)
        for observation_record in observation_records:
            observation_record.to_payload()
        if self.config.dry_run:
            media_result = self._persist_track_media(track=track, vehicle_track_id=dry_run_track_ref, camera_id=self._dry_run_token("CAMERA", track.camera_code), dry_run=True)
            return TrackPersistenceResult(
                track_uuid=track.track_uuid,
                status="dry_run",
                database_track_id=dry_run_track_ref,
                observations_written=len(observation_records),
                media_persistence=self._media_result_to_dict(media_result, dry_run=True),
            )
        existing = self._require_repository(self._vehicle_track_repository, "vehicle_track").get_vehicle_track_by_uuid(track.track_uuid)
        if existing is not None:
            return TrackPersistenceResult(track_uuid=track.track_uuid, status="already_exists", database_track_id=str(existing["id"]), observations_written=0)
        track_row = self._require_repository(self._vehicle_track_repository, "vehicle_track").upsert_vehicle_track(track_record)
        vehicle_track_id = UUID(str(track_row["id"]))
        persisted_rows = 0
        for start in range(0, len(observation_records), self.config.observation_batch_size):
            batch = observation_records[start : start + self.config.observation_batch_size]
            if not batch:
                continue
            persisted_batch = self._require_repository(self._track_observation_repository, "track_observation").upsert_observations_batch(
                [self._with_vehicle_track_id(item, vehicle_track_id) for item in batch]
            )
            persisted_rows += len(persisted_batch)
        media_result = self._persist_track_media(track=track, vehicle_track_id=str(vehicle_track_id), camera_id=str(context.camera_id), dry_run=False)
        return TrackPersistenceResult(
            track_uuid=track.track_uuid,
            status="inserted",
            database_track_id=str(vehicle_track_id),
            observations_written=persisted_rows,
            media_persistence=self._media_result_to_dict(media_result, dry_run=False),
        )

    def _ensure_processing_run(self, camera_configs: list[CameraConfig]) -> None:
        if self._processing_run_id is not None:
            return
        self._started_at = datetime.now(timezone.utc)
        processing_run_record = ProcessingRunRecord(
            run_code=self.run_code,
            pipeline_name=self.pipeline_name,
            pipeline_version=self.pipeline_version,
            execution_mode=self.execution_mode,
            status="RUNNING",
            configured_camera_count=len(camera_configs),
            active_camera_count=sum(1 for camera in camera_configs if camera.enabled),
            started_at=self._started_at,
            host_name=socket.gethostname(),
            runtime_device=self.runtime_device,
            configuration=self._configuration_payload(),
            metrics={},
        )
        processing_run_record.to_payload()
        if self.enable_database_writes:
            run_row = self._require_repository(self._processing_run_repository, "processing_run").upsert_processing_run(processing_run_record)
            self._processing_run_id = UUID(str(run_row["id"]))
        else:
            self._processing_run_id = self._dry_run_uuid("PROCESSING_RUN", self.run_code)
        detector_record = AiModelRecord(
            model_code=f"detector::{self.detection_config.model_path}",
            model_name=self.detection_config.model_path,
            model_type="DETECTOR",
            provider="ultralytics",
            model_reference=self.detection_config.model_path,
            configuration={
                "confidence_threshold": self.detection_config.confidence_threshold,
                "iou_threshold": self.detection_config.iou_threshold,
                "image_size": self.detection_config.image_size,
                "allowed_classes": list(self.detection_config.allowed_classes),
            },
        )
        detector_record.to_payload()
        tracker_record = AiModelRecord(
            model_code=f"tracker::{self.tracking_config.backend}",
            model_name=self.tracking_config.backend,
            model_type="TRACKER",
            provider="internal",
            model_reference=self.tracking_config.backend,
            configuration=self._tracking_configuration_payload(),
        )
        tracker_record.to_payload()
        if self.enable_database_writes:
            detector_row = self._require_repository(self._ai_model_repository, "ai_model").upsert_ai_model(detector_record)
            self._detector_model_id = UUID(str(detector_row["id"]))
            tracker_row = self._require_repository(self._ai_model_repository, "ai_model").upsert_ai_model(tracker_record)
            self._tracker_model_id = UUID(str(tracker_row["id"]))
        else:
            self._detector_model_id = self._dry_run_uuid("MODEL", f"detector::{self.detection_config.model_path}")
            self._tracker_model_id = self._dry_run_uuid("MODEL", f"tracker::{self.tracking_config.backend}")
        detector_run_model = RunModelRecord(
            processing_run_id=self._processing_run_id,
            ai_model_id=self._detector_model_id,
            stage_name="DETECT",
            device=self.runtime_device,
            resolved_configuration=self._configuration_payload()["detection"],
        )
        detector_run_model.to_payload()
        tracker_run_model = RunModelRecord(
            processing_run_id=self._processing_run_id,
            ai_model_id=self._tracker_model_id,
            stage_name="TRACK",
            device=self.runtime_device,
            resolved_configuration=self._tracking_configuration_payload(),
        )
        tracker_run_model.to_payload()
        if self.enable_database_writes:
            self._require_repository(self._run_model_repository, "run_model").upsert_run_model(detector_run_model)
            self._require_repository(self._run_model_repository, "run_model").upsert_run_model(tracker_run_model)

    def _apply_metrics(self, result: TrackPersistenceResult) -> None:
        if result.status == "inserted":
            self.metrics.tracks_inserted += 1
            self.metrics.observations_written += result.observations_written
        elif result.status == "already_exists":
            self.metrics.tracks_already_existing += 1
        elif result.status == "skipped_discarded":
            self.metrics.tracks_skipped_discarded += 1
        elif result.status == "skipped_invalid_state":
            self.metrics.tracks_skipped_invalid_state += 1
        elif result.status == "dry_run":
            self.metrics.observations_written += result.observations_written
        elif result.status == "failed":
            self.metrics.tracks_failed += 1
            if result.error:
                self.metrics.errors.append(result.error)
        if result.media_persistence is not None:
            self.metrics.media_records_attempted += int(result.media_persistence.get("attempted", 0))
            self.metrics.media_records_validated += int(result.media_persistence.get("validated", 0))
            self.metrics.media_records_inserted += int(result.media_persistence.get("inserted", 0))
            self.metrics.media_records_already_existing += int(result.media_persistence.get("already_existing", 0))
            self.metrics.media_records_failed += int(result.media_persistence.get("failed", 0))
            self.metrics.media_files_missing += int(result.media_persistence.get("missing_files", 0))

    def _validate_track(self, track: LocalVehicleTrack) -> None:
        if not str(track.track_uuid).strip():
            raise AnalyticsPersistenceValidationError("track_uuid must be present.")
        if track.camera_code not in self._camera_context_by_code:
            raise AnalyticsPersistenceValidationError(f"Unknown camera_code for analytics persistence: {track.camera_code}")
        if int(track.local_track_id) < 0:
            raise AnalyticsPersistenceValidationError("local_track_id must be non-negative.")
        if not is_supported_vehicle_class(track.class_name):
            raise AnalyticsPersistenceValidationError(f"Unsupported vehicle class: {track.class_name}")
        if track.first_seen_at is None or track.last_seen_at is None:
            raise AnalyticsPersistenceValidationError("Track timestamps must be present.")
        if track.last_seen_at < track.first_seen_at:
            raise AnalyticsPersistenceValidationError("Track last_seen_at must not be before first_seen_at.")
        if int(track.first_frame_number) > int(track.last_frame_number):
            raise AnalyticsPersistenceValidationError("first_frame_number must not be after last_frame_number.")
        if int(track.observation_count) < 0:
            raise AnalyticsPersistenceValidationError("observation_count must be non-negative.")
        if not math.isfinite(float(track.best_confidence)):
            raise AnalyticsPersistenceValidationError("best_confidence must be finite.")

    def _build_observation_records(self, track: LocalVehicleTrack, vehicle_track_id: UUID | None) -> list[TrackObservationRecord]:
        context = self._require_camera_context(track.camera_code)
        selected = self._select_observations(track.observations)
        rows: list[TrackObservationRecord] = []
        for index, observation in enumerate(selected):
            self._validate_observation(track, observation)
            x1, y1, x2, y2 = observation.bbox_xyxy
            rows.append(
                TrackObservationRecord(
                    vehicle_track_id=vehicle_track_id or UUID(int=0),
                    camera_id=context.camera_id,
                    frame_number=observation.frame_number,
                    observed_at=self._ensure_timezone(observation.camera_timestamp),
                    video_time_seconds=observation.video_time_seconds,
                    bbox_x1=x1,
                    bbox_y1=y1,
                    bbox_x2=x2,
                    bbox_y2=y2,
                    detection_confidence=observation.confidence,
                    tracker_confidence=None,
                    is_key_observation=index in {0, len(selected) - 1},
                    metadata={"track_uuid": track.track_uuid},
                )
            )
        return rows

    @staticmethod
    def _with_vehicle_track_id(record: TrackObservationRecord, vehicle_track_id: UUID) -> TrackObservationRecord:
        return TrackObservationRecord(
            vehicle_track_id=vehicle_track_id,
            camera_id=record.camera_id,
            frame_number=record.frame_number,
            observed_at=record.observed_at,
            video_time_seconds=record.video_time_seconds,
            bbox_x1=record.bbox_x1,
            bbox_y1=record.bbox_y1,
            bbox_x2=record.bbox_x2,
            bbox_y2=record.bbox_y2,
            center_x=record.center_x,
            center_y=record.center_y,
            detection_confidence=record.detection_confidence,
            tracker_confidence=record.tracker_confidence,
            is_key_observation=record.is_key_observation,
            metadata=record.metadata,
        )

    def _select_observations(self, observations: list[TrackObservation]) -> list[TrackObservation]:
        if self.config.observation_mode == "none" or not observations:
            return []
        if self.config.observation_mode == "all":
            return list(observations)
        selected: list[TrackObservation] = []
        for index, observation in enumerate(observations):
            is_first = index == 0
            is_last = index == len(observations) - 1
            if is_first or is_last or (index % self.config.observation_sample_every_n == 0):
                selected.append(observation)
        deduped: list[TrackObservation] = []
        seen_keys: set[tuple[int, float]] = set()
        for observation in selected:
            key = (observation.frame_number, observation.video_time_seconds)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(observation)
        return deduped

    def _validate_observation(self, track: LocalVehicleTrack, observation: TrackObservation) -> None:
        if int(observation.frame_number) < 0:
            raise AnalyticsPersistenceValidationError("Observation frame_number must be non-negative.")
        if observation.camera_timestamp is None:
            raise AnalyticsPersistenceValidationError("Observation camera_timestamp must be present.")
        if observation.camera_code != track.camera_code:
            raise AnalyticsPersistenceValidationError("Observation camera_code does not match parent track.")
        if int(observation.local_track_id) != int(track.local_track_id):
            raise AnalyticsPersistenceValidationError("Observation local_track_id does not match parent track.")
        if not math.isfinite(float(observation.confidence)):
            raise AnalyticsPersistenceValidationError("Observation confidence must be finite.")
        x1, y1, x2, y2 = observation.bbox_xyxy
        values = (float(x1), float(y1), float(x2), float(y2))
        if any(not math.isfinite(value) for value in values):
            raise AnalyticsPersistenceValidationError("Observation bbox values must be finite.")
        if x2 <= x1 or y2 <= y1:
            raise AnalyticsPersistenceValidationError("Observation bbox must satisfy x2 > x1 and y2 > y1.")

    def _record_processing_error(
        self,
        exc: Exception,
        *,
        track: LocalVehicleTrack | None,
        stage_name: str,
        structured_context: dict[str, object],
    ) -> None:
        if self.config.dry_run or self._processing_run_id is None or not self.enable_database_writes:
            return
        camera_run_id = None
        if track is not None and track.camera_code in self._camera_context_by_code:
            camera_run_id = self._camera_context_by_code[track.camera_code].camera_run_id
        try:
            self._require_repository(self._processing_error_repository, "processing_error").create_processing_error(
                ProcessingErrorRecord(
                    processing_run_id=self._processing_run_id,
                    camera_run_id=camera_run_id,
                    stage_name=stage_name,
                    severity="ERROR",
                    exception_type=type(exc).__name__,
                    message=str(exc),
                    structured_context=structured_context,
                )
            )
        except Exception as processing_error_exc:
            if isinstance(processing_error_exc, AnalyticsRepositoryError):
                self.metrics.errors.append(f"processing_error_write_failed:{processing_error_exc.operation}")
            else:
                self.metrics.errors.append("processing_error_write_failed")

    def _persist_track_media(
        self,
        *,
        track: LocalVehicleTrack,
        vehicle_track_id: str | None,
        camera_id: str,
        dry_run: bool,
    ) -> TrackMediaBatchResult | None:
        if not self.config.persist_track_media or track.evidence_package is None:
            return None
        try:
            records = build_track_media_records(
                evidence_package=track.evidence_package,
                vehicle_track_id=vehicle_track_id or "00000000-0000-0000-0000-000000000000",
                camera_id=camera_id,
                artifact_root=self.artifact_root,
                persist_roles=self.config.track_media_roles,
            )
        except FileNotFoundError as exc:
            self.metrics.errors.append(f"{track.track_uuid}: {exc}")
            if not dry_run:
                self._record_processing_error(
                    exc,
                    track=track,
                    stage_name="TRACK_MEDIA",
                    structured_context={"track_uuid": track.track_uuid, "camera_id": camera_id},
                )
            if self.config.fail_pipeline_on_track_media_persistence_error:
                raise
            return TrackMediaBatchResult(attempted=1, inserted=0, already_existing=0, failed=0, validated=0, missing_files=1)
        except Exception as exc:
            self.metrics.errors.append(f"{track.track_uuid}: {exc}")
            if not dry_run:
                self._record_processing_error(
                    exc,
                    track=track,
                    stage_name="TRACK_MEDIA",
                    structured_context={"track_uuid": track.track_uuid, "camera_id": camera_id},
                )
            if self.config.fail_pipeline_on_track_media_persistence_error:
                raise
            return TrackMediaBatchResult(attempted=1, inserted=0, already_existing=0, failed=1, validated=0, missing_files=0)
        if not records:
            return TrackMediaBatchResult(attempted=0, inserted=0, already_existing=0, failed=0, validated=0, missing_files=0)
        if dry_run:
            for record in records:
                record.to_payload()
            return TrackMediaBatchResult(attempted=len(records), inserted=0, already_existing=0, failed=0, validated=len(records), missing_files=0)
        try:
            return self._require_repository(self._track_media_repository, "track_media").bulk_upsert(records)
        except Exception as exc:
            self.metrics.errors.append(f"{track.track_uuid}: {exc}")
            self._record_processing_error(
                exc,
                track=track,
                stage_name="TRACK_MEDIA",
                structured_context={"track_uuid": track.track_uuid, "camera_id": camera_id},
            )
            if self.config.fail_pipeline_on_track_media_persistence_error:
                raise
            return TrackMediaBatchResult(attempted=len(records), inserted=0, already_existing=0, failed=len(records), validated=0, missing_files=0)

    @staticmethod
    def _media_result_to_dict(result: TrackMediaBatchResult | None, dry_run: bool) -> dict[str, object] | None:
        if result is None:
            return None
        return {
            "mode": "dry_run" if dry_run else "database",
            "attempted": result.attempted,
            "validated": result.validated,
            "inserted": result.inserted,
            "already_existing": result.already_existing,
            "failed": result.failed,
            "missing_files": result.missing_files,
        }

    @staticmethod
    def _require_repository(repository: Any, table_name: str) -> Any:
        if repository is None:
            raise AnalyticsPersistenceValidationError(f"Repository for analytics.{table_name} is not available in the current mode.")
        return repository

    @staticmethod
    def _dry_run_token(entity_type: str, *parts: object) -> str:
        serialized = ":".join(str(part) for part in parts if str(part) != "")
        return f"DRYRUN:{entity_type}:{serialized}"

    @classmethod
    def _dry_run_uuid(cls, entity_type: str, *parts: object) -> UUID:
        return uuid5(NAMESPACE_URL, cls._dry_run_token(entity_type, *parts))

    def _configuration_payload(self) -> dict[str, object]:
        return {
            "detection": {
                "model_path": self.detection_config.model_path,
                "fallback_model_path": self.detection_config.fallback_model_path,
                "allow_fallback": self.detection_config.allow_fallback,
                "device": self.detection_config.device,
                "confidence_threshold": self.detection_config.confidence_threshold,
                "iou_threshold": self.detection_config.iou_threshold,
                "image_size": self.detection_config.image_size,
                "allowed_classes": list(self.detection_config.allowed_classes),
            },
            "tracking": self._tracking_configuration_payload(),
            "persistence": {
                "backend": self.config.backend,
                "observation_mode": self.config.observation_mode,
                "observation_batch_size": self.config.observation_batch_size,
                "observation_sample_every_n": self.config.observation_sample_every_n,
                "write_completed_tracks_only": self.config.write_completed_tracks_only,
                "include_discarded_tracks": self.config.include_discarded_tracks,
            },
        }

    def _tracking_configuration_payload(self) -> dict[str, object]:
        return {
            "backend": self.tracking_config.backend,
            "track_high_thresh": self.tracking_config.track_high_thresh,
            "track_low_thresh": self.tracking_config.track_low_thresh,
            "new_track_thresh": self.tracking_config.new_track_thresh,
            "match_thresh": self.tracking_config.match_thresh,
            "track_buffer": self.tracking_config.track_buffer,
            "track_activation_threshold": self.tracking_config.track_activation_threshold,
            "lost_track_buffer": self.tracking_config.lost_track_buffer,
            "minimum_matching_threshold": self.tracking_config.minimum_matching_threshold,
            "frame_rate": self.tracking_config.frame_rate,
            "minimum_consecutive_frames": self.tracking_config.minimum_consecutive_frames,
            "min_confirmed_observations": self.tracking_config.min_confirmed_observations,
            "max_lost_frames": self.tracking_config.max_lost_frames,
            "preserve_state_per_camera": self.tracking_config.preserve_state_per_camera,
        }

    def _require_processing_run_id(self) -> UUID:
        if self._processing_run_id is None:
            raise AnalyticsPersistenceValidationError("Processing run has not been initialized.")
        return self._processing_run_id

    def _require_camera_context(self, camera_code: str) -> _CameraPersistenceContext:
        context = self._camera_context_by_code.get(camera_code)
        if context is None:
            raise AnalyticsPersistenceValidationError(f"Unknown camera_code for analytics persistence: {camera_code}")
        return context

    @staticmethod
    def _camera_timezone_name(camera_config: CameraConfig) -> str:
        start_time = camera_config.start_time
        if start_time is not None and start_time.tzinfo is not None:
            return str(start_time.tzinfo)
        return "Asia/Kolkata"

    @staticmethod
    def _ensure_timezone(value: datetime | None) -> datetime:
        if value is None:
            raise AnalyticsPersistenceValidationError("Timezone-aware datetime is required.")
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    @staticmethod
    def _ensure_optional_timezone(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    @staticmethod
    def _optional_float(value: object) -> float | None:
        if value in (None, ""):
            return None
        return float(value)

    @staticmethod
    def _optional_int(value: object) -> int | None:
        if value in (None, ""):
            return None
        return int(value)

    @staticmethod
    def _optional_processing_fps(payload: dict[str, object]) -> float | None:
        frames_processed = payload.get("frames_processed", payload.get("frames_read"))
        if frames_processed in (None, ""):
            return None
        return float(frames_processed)

    @staticmethod
    def _average_detection_confidence(observations: list[TrackObservation]) -> float | None:
        if not observations:
            return None
        return sum(float(item.confidence) for item in observations) / len(observations)
