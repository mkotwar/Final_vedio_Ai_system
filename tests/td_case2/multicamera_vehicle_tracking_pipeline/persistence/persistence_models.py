from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID
from pathlib import PurePosixPath, PureWindowsPath

from .track_media_types import TRACK_MEDIA_TYPES
from .vehicle_class_mapping import VehicleClass, normalize_vehicle_class


PROCESSING_RUN_EXECUTION_MODES = ("SEQUENTIAL", "THREADED", "BATCHED", "LIVE")
PROCESSING_RUN_STATUSES = ("QUEUED", "RUNNING", "COMPLETED", "PARTIAL", "FAILED", "CANCELLED")
CAMERA_RUN_STATUSES = ("PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED")
PROCESSING_JOB_TYPES = ("READ", "DETECT", "TRACK", "PERSIST", "BEST_FRAME_SELECTION", "PLATE_DETECTION", "OCR", "COLOR", "CROSS_CAMERA_MATCH", "EVENT_ANALYSIS")
PROCESSING_JOB_STATUSES = ("QUEUED", "RUNNING", "COMPLETED", "FAILED", "RETRYING", "CANCELLED")
VEHICLE_TRACK_LIFECYCLE_STATES = ("TENTATIVE", "ACTIVE", "TEMPORARILY_LOST", "COMPLETED", "DISCARDED")
VIDEO_SOURCE_TYPES = ("LOCAL_FILE", "RTSP", "LIVE_STREAM", "VMS_RECORDING", "PLAYBACK_API")
PROCESSING_ERROR_SEVERITIES = ("INFO", "WARNING", "ERROR", "CRITICAL")
PROCESSING_ERROR_RESOLUTION_STATES = ("OPEN", "ACKNOWLEDGED", "RESOLVED", "IGNORED")
TRACK_MEDIA_STORAGE_PROVIDERS = ("LOCAL", "NAS", "S3", "SUPABASE_STORAGE")
VEHICLE_ATTRIBUTE_SCOPES = ("TRACK", "GLOBAL")
VEHICLE_ATTRIBUTE_STATUSES = ("CURRENT", "HISTORICAL", "REJECTED")
PLATE_READING_STATUSES = ("VERIFIED", "PROBABLE", "PARTIAL", "UNKNOWN")


PERSISTENCE_STATUSES = ("inserted", "already_exists", "skipped_discarded", "skipped_invalid_state", "dry_run", "failed")


class PersistenceModelValidationError(ValueError):
    """Raised when a Phase 1 analytics persistence record is invalid."""


def _ensure_timezone_aware(value: datetime | None, *, field_name: str, required: bool) -> datetime | None:
    if value is None:
        if required:
            raise PersistenceModelValidationError(f"{field_name} is required.")
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise PersistenceModelValidationError(f"{field_name} must be timezone-aware.")
    return value


def _ensure_non_negative(value: int | float, *, field_name: str) -> None:
    if value < 0:
        raise PersistenceModelValidationError(f"{field_name} must be non-negative.")


def _ensure_positive(value: int | float | None, *, field_name: str) -> None:
    if value is not None and value <= 0:
        raise PersistenceModelValidationError(f"{field_name} must be positive.")


def _ensure_confidence(value: float | None, *, field_name: str) -> None:
    if value is not None and not 0 <= float(value) <= 1:
        raise PersistenceModelValidationError(f"{field_name} must be between 0 and 1.")


def _ensure_choice(value: str | None, *, field_name: str, allowed: tuple[str, ...], required: bool) -> str | None:
    if value is None:
        if required:
            raise PersistenceModelValidationError(f"{field_name} is required.")
        return None
    normalized = str(value).strip()
    if not normalized:
        if required:
            raise PersistenceModelValidationError(f"{field_name} must not be empty.")
        return None
    if normalized not in allowed:
        raise PersistenceModelValidationError(f"{field_name} must be one of: {', '.join(allowed)}")
    return normalized


def _copy_metadata(value: dict[str, Any] | None, *, field_name: str = "metadata") -> dict[str, Any]:
    payload = {} if value is None else deepcopy(value)
    if not isinstance(payload, dict):
        raise PersistenceModelValidationError(f"{field_name} must be a JSON-compatible dictionary.")
    return payload


def _serialize_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _serialize_uuid(value: UUID | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _omit_none(payload: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in payload.items() if value is not None}


def _ensure_finite(value: float | None, *, field_name: str) -> None:
    if value is not None and not float(value) == float(value):
        raise PersistenceModelValidationError(f"{field_name} must be finite.")
    if value is not None and value in (float("inf"), float("-inf")):
        raise PersistenceModelValidationError(f"{field_name} must be finite.")


def _normalize_relative_storage_uri(value: str) -> str:
    candidate = str(value).strip()
    if not candidate:
        raise PersistenceModelValidationError("storage_uri must not be empty.")
    windows_path = PureWindowsPath(candidate)
    if windows_path.drive:
        raise PersistenceModelValidationError("storage_uri must be relative, not a Windows drive path.")
    if candidate.startswith("\\\\"):
        raise PersistenceModelValidationError("storage_uri must be relative, not a UNC path.")
    normalized = candidate.replace("\\", "/")
    if normalized.startswith("/"):
        raise PersistenceModelValidationError("storage_uri must be relative, not absolute.")
    posix_path = PurePosixPath(normalized)
    if any(part == ".." for part in posix_path.parts):
        raise PersistenceModelValidationError("storage_uri must not contain path traversal.")
    return normalized


def _canonical_vehicle_class(value: VehicleClass | str) -> str:
    if isinstance(value, VehicleClass):
        return value.value
    return normalize_vehicle_class(str(value)).value


@dataclass(frozen=True, slots=True)
class CameraRecord:
    camera_code: str
    camera_name: str | None = None
    external_camera_id: str | None = None
    site_code: str | None = None
    location_name: str | None = None
    timezone_name: str = "Asia/Kolkata"
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    id: UUID | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        if not str(self.camera_code).strip():
            raise PersistenceModelValidationError("camera_code must not be empty.")
        if not str(self.timezone_name).strip():
            raise PersistenceModelValidationError("timezone_name must not be empty.")
        _ensure_timezone_aware(self.created_at, field_name="created_at", required=False)
        _ensure_timezone_aware(self.updated_at, field_name="updated_at", required=False)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_payload(self) -> dict[str, object]:
        return _omit_none(
            {
                "camera_code": self.camera_code,
                "external_camera_id": self.external_camera_id,
                "camera_name": self.camera_name,
                "site_code": self.site_code,
                "location_name": self.location_name,
                "timezone": self.timezone_name,
                "enabled": self.enabled,
                "metadata": self.metadata if self.metadata else {},
            }
        )


@dataclass(frozen=True, slots=True)
class VideoSourceRecord:
    camera_id: UUID
    source_type: str
    source_reference: str
    external_recording_id: str | None = None
    source_start_at: datetime | None = None
    source_end_at: datetime | None = None
    source_fps: float | None = None
    frame_width: int | None = None
    frame_height: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_choice(self.source_type, field_name="source_type", allowed=VIDEO_SOURCE_TYPES, required=True)
        if not str(self.source_reference).strip():
            raise PersistenceModelValidationError("source_reference must not be empty.")
        _ensure_timezone_aware(self.source_start_at, field_name="source_start_at", required=False)
        _ensure_timezone_aware(self.source_end_at, field_name="source_end_at", required=False)
        if self.source_start_at is not None and self.source_end_at is not None and self.source_start_at > self.source_end_at:
            raise PersistenceModelValidationError("source_start_at must not be after source_end_at.")
        _ensure_positive(self.source_fps, field_name="source_fps")
        _ensure_positive(self.frame_width, field_name="frame_width")
        _ensure_positive(self.frame_height, field_name="frame_height")
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_payload(self) -> dict[str, object]:
        return _omit_none(
            {
                "camera_id": _serialize_uuid(self.camera_id),
                "source_type": self.source_type,
                "external_recording_id": self.external_recording_id,
                "source_reference": self.source_reference,
                "source_start_at": _serialize_datetime(self.source_start_at),
                "source_end_at": _serialize_datetime(self.source_end_at),
                "source_fps": self.source_fps,
                "frame_width": self.frame_width,
                "frame_height": self.frame_height,
                "metadata": self.metadata if self.metadata else {},
            }
        )


@dataclass(frozen=True, slots=True)
class ProcessingRunRecord:
    run_code: str
    pipeline_name: str | None = None
    pipeline_version: str | None = None
    execution_mode: str | None = None
    status: str | None = None
    configured_camera_count: int = 0
    active_camera_count: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    total_frames_processed: int = 0
    total_detections: int = 0
    total_track_observations: int = 0
    total_tracks: int = 0
    host_name: str | None = None
    runtime_device: str | None = None
    configuration: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.run_code).strip():
            raise PersistenceModelValidationError("run_code must not be empty.")
        _ensure_choice(self.execution_mode, field_name="execution_mode", allowed=PROCESSING_RUN_EXECUTION_MODES, required=False)
        _ensure_choice(self.status, field_name="status", allowed=PROCESSING_RUN_STATUSES, required=False)
        _ensure_non_negative(self.configured_camera_count, field_name="configured_camera_count")
        _ensure_non_negative(self.active_camera_count, field_name="active_camera_count")
        if self.active_camera_count > self.configured_camera_count:
            raise PersistenceModelValidationError("active_camera_count must not exceed configured_camera_count.")
        _ensure_non_negative(self.total_frames_processed, field_name="total_frames_processed")
        _ensure_non_negative(self.total_detections, field_name="total_detections")
        _ensure_non_negative(self.total_track_observations, field_name="total_track_observations")
        _ensure_non_negative(self.total_tracks, field_name="total_tracks")
        _ensure_timezone_aware(self.started_at, field_name="started_at", required=False)
        _ensure_timezone_aware(self.completed_at, field_name="completed_at", required=False)
        if self.started_at is not None and self.completed_at is not None and self.started_at > self.completed_at:
            raise PersistenceModelValidationError("started_at must not be after completed_at.")
        object.__setattr__(self, "configuration", _copy_metadata(self.configuration, field_name="configuration"))
        object.__setattr__(self, "metrics", _copy_metadata(self.metrics, field_name="metrics"))

    def to_payload(self) -> dict[str, object]:
        payload = {
            "run_code": self.run_code,
            "pipeline_name": self.pipeline_name,
            "pipeline_version": self.pipeline_version,
            "execution_mode": self.execution_mode,
            "status": self.status,
            "configured_camera_count": self.configured_camera_count,
            "active_camera_count": self.active_camera_count,
            "started_at": _serialize_datetime(self.started_at),
            "completed_at": _serialize_datetime(self.completed_at),
            "total_frames_processed": self.total_frames_processed,
            "total_detections": self.total_detections,
            "total_track_observations": self.total_track_observations,
            "total_tracks": self.total_tracks,
            "host_name": self.host_name,
            "runtime_device": self.runtime_device,
            "configuration": self.configuration if self.configuration else {},
            "metrics": self.metrics if self.metrics else {},
        }
        return _omit_none(payload)


@dataclass(frozen=True, slots=True)
class CameraRunRecord:
    processing_run_id: UUID
    camera_id: UUID
    video_source_id: UUID | None = None
    status: str | None = None
    reader_worker_name: str | None = None
    resolved_source_fps: float | None = None
    effective_processing_fps: float | None = None
    first_frame_number: int | None = None
    last_frame_number: int | None = None
    frames_read: int = 0
    frames_processed: int = 0
    detections_count: int = 0
    track_observations_count: int = 0
    completed_tracks_count: int = 0
    discarded_tracks_count: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_choice(self.status, field_name="status", allowed=CAMERA_RUN_STATUSES, required=False)
        _ensure_positive(self.resolved_source_fps, field_name="resolved_source_fps")
        _ensure_positive(self.effective_processing_fps, field_name="effective_processing_fps")
        for name, value in (
            ("frames_read", self.frames_read),
            ("frames_processed", self.frames_processed),
            ("detections_count", self.detections_count),
            ("track_observations_count", self.track_observations_count),
            ("completed_tracks_count", self.completed_tracks_count),
            ("discarded_tracks_count", self.discarded_tracks_count),
        ):
            _ensure_non_negative(value, field_name=name)
        if self.first_frame_number is not None:
            _ensure_non_negative(self.first_frame_number, field_name="first_frame_number")
        if self.last_frame_number is not None:
            _ensure_non_negative(self.last_frame_number, field_name="last_frame_number")
        if self.first_frame_number is not None and self.last_frame_number is not None and self.first_frame_number > self.last_frame_number:
            raise PersistenceModelValidationError("first_frame_number must not be after last_frame_number.")
        _ensure_timezone_aware(self.started_at, field_name="started_at", required=False)
        _ensure_timezone_aware(self.completed_at, field_name="completed_at", required=False)
        if self.started_at is not None and self.completed_at is not None and self.started_at > self.completed_at:
            raise PersistenceModelValidationError("started_at must not be after completed_at.")
        object.__setattr__(self, "metrics", _copy_metadata(self.metrics, field_name="metrics"))

    def to_payload(self) -> dict[str, object]:
        return _omit_none(
            {
                "processing_run_id": _serialize_uuid(self.processing_run_id),
                "camera_id": _serialize_uuid(self.camera_id),
                "video_source_id": _serialize_uuid(self.video_source_id),
                "status": self.status,
                "reader_worker_name": self.reader_worker_name,
                "resolved_source_fps": self.resolved_source_fps,
                "effective_processing_fps": self.effective_processing_fps,
                "first_frame_number": self.first_frame_number,
                "last_frame_number": self.last_frame_number,
                "frames_read": self.frames_read,
                "frames_processed": self.frames_processed,
                "detections_count": self.detections_count,
                "track_observations_count": self.track_observations_count,
                "completed_tracks_count": self.completed_tracks_count,
                "discarded_tracks_count": self.discarded_tracks_count,
                "started_at": _serialize_datetime(self.started_at),
                "completed_at": _serialize_datetime(self.completed_at),
                "metrics": self.metrics if self.metrics else {},
            }
        )


@dataclass(frozen=True, slots=True)
class ProcessingJobRecord:
    processing_run_id: UUID
    job_type: str
    status: str
    camera_run_id: UUID | None = None
    priority: int = 0
    attempt_number: int = 1
    worker_name: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    processing_time_ms: int | None = None
    input_summary: dict[str, Any] | None = None
    output_summary: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        _ensure_choice(self.job_type, field_name="job_type", allowed=PROCESSING_JOB_TYPES, required=True)
        _ensure_choice(self.status, field_name="status", allowed=PROCESSING_JOB_STATUSES, required=True)
        _ensure_non_negative(self.priority, field_name="priority")
        if self.attempt_number <= 0:
            raise PersistenceModelValidationError("attempt_number must be positive.")
        if self.processing_time_ms is not None:
            _ensure_non_negative(self.processing_time_ms, field_name="processing_time_ms")
        _ensure_timezone_aware(self.started_at, field_name="started_at", required=False)
        _ensure_timezone_aware(self.completed_at, field_name="completed_at", required=False)
        if self.started_at is not None and self.completed_at is not None and self.started_at > self.completed_at:
            raise PersistenceModelValidationError("started_at must not be after completed_at.")
        object.__setattr__(self, "input_summary", _copy_metadata(self.input_summary, field_name="input_summary"))
        object.__setattr__(self, "output_summary", _copy_metadata(self.output_summary, field_name="output_summary"))

    def to_payload(self) -> dict[str, object]:
        return _omit_none(
            {
                "processing_run_id": _serialize_uuid(self.processing_run_id),
                "camera_run_id": _serialize_uuid(self.camera_run_id),
                "job_type": self.job_type,
                "status": self.status,
                "priority": self.priority,
                "attempt_number": self.attempt_number,
                "worker_name": self.worker_name,
                "started_at": _serialize_datetime(self.started_at),
                "completed_at": _serialize_datetime(self.completed_at),
                "processing_time_ms": self.processing_time_ms,
                "input_summary": self.input_summary if self.input_summary else None,
                "output_summary": self.output_summary if self.output_summary else None,
                "error_code": self.error_code,
                "error_message": self.error_message,
            }
        )


@dataclass(frozen=True, slots=True)
class VehicleTrackRecord:
    processing_run_id: UUID
    camera_run_id: UUID
    camera_id: UUID
    track_uuid: str
    local_track_id: int
    vehicle_class: VehicleClass | str
    first_seen_at: datetime
    last_seen_at: datetime
    first_frame_number: int
    last_frame_number: int
    tracker_backend: str
    first_video_time_seconds: float | None = None
    last_video_time_seconds: float | None = None
    observation_count: int = 0
    best_detection_confidence: float | None = None
    average_detection_confidence: float | None = None
    lifecycle_state: str = "COMPLETED"
    completion_reason: str | None = None
    tracker_configuration: dict[str, Any] = field(default_factory=dict)
    searchable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.track_uuid).strip():
            raise PersistenceModelValidationError("track_uuid must not be empty.")
        if not str(self.tracker_backend).strip():
            raise PersistenceModelValidationError("tracker_backend must not be empty.")
        _ensure_non_negative(self.local_track_id, field_name="local_track_id")
        _ensure_non_negative(self.observation_count, field_name="observation_count")
        _ensure_timezone_aware(self.first_seen_at, field_name="first_seen_at", required=True)
        _ensure_timezone_aware(self.last_seen_at, field_name="last_seen_at", required=True)
        if self.first_seen_at > self.last_seen_at:
            raise PersistenceModelValidationError("first_seen_at must not be after last_seen_at.")
        _ensure_non_negative(self.first_frame_number, field_name="first_frame_number")
        _ensure_non_negative(self.last_frame_number, field_name="last_frame_number")
        if self.first_frame_number > self.last_frame_number:
            raise PersistenceModelValidationError("first_frame_number must not be after last_frame_number.")
        if self.first_video_time_seconds is not None and self.last_video_time_seconds is not None and self.first_video_time_seconds > self.last_video_time_seconds:
            raise PersistenceModelValidationError("first_video_time_seconds must not be after last_video_time_seconds.")
        _ensure_confidence(self.best_detection_confidence, field_name="best_detection_confidence")
        _ensure_confidence(self.average_detection_confidence, field_name="average_detection_confidence")
        lifecycle_state = _ensure_choice(self.lifecycle_state, field_name="lifecycle_state", allowed=VEHICLE_TRACK_LIFECYCLE_STATES, required=True)
        object.__setattr__(self, "lifecycle_state", lifecycle_state)
        object.__setattr__(self, "tracker_configuration", _copy_metadata(self.tracker_configuration, field_name="tracker_configuration"))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))
        object.__setattr__(self, "vehicle_class", _canonical_vehicle_class(self.vehicle_class))

    def to_payload(self) -> dict[str, object]:
        return _omit_none(
            {
                "processing_run_id": _serialize_uuid(self.processing_run_id),
                "camera_run_id": _serialize_uuid(self.camera_run_id),
                "camera_id": _serialize_uuid(self.camera_id),
                "track_uuid": self.track_uuid,
                "local_track_id": self.local_track_id,
                "vehicle_class": self.vehicle_class,
                "first_seen_at": _serialize_datetime(self.first_seen_at),
                "last_seen_at": _serialize_datetime(self.last_seen_at),
                "first_frame_number": self.first_frame_number,
                "last_frame_number": self.last_frame_number,
                "first_video_time_seconds": self.first_video_time_seconds,
                "last_video_time_seconds": self.last_video_time_seconds,
                "observation_count": self.observation_count,
                "best_detection_confidence": self.best_detection_confidence,
                "average_detection_confidence": self.average_detection_confidence,
                "lifecycle_state": self.lifecycle_state,
                "completion_reason": self.completion_reason,
                "tracker_backend": self.tracker_backend,
                "tracker_configuration": self.tracker_configuration if self.tracker_configuration else {},
                "searchable": self.searchable,
                "metadata": self.metadata if self.metadata else {},
            }
        )


@dataclass(frozen=True, slots=True)
class TrackObservationRecord:
    vehicle_track_id: UUID
    camera_id: UUID
    frame_number: int
    observed_at: datetime
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    video_time_seconds: float | None = None
    center_x: float | None = None
    center_y: float | None = None
    detection_confidence: float | None = None
    tracker_confidence: float | None = None
    is_key_observation: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_non_negative(self.frame_number, field_name="frame_number")
        _ensure_timezone_aware(self.observed_at, field_name="observed_at", required=True)
        if self.bbox_x2 <= self.bbox_x1:
            raise PersistenceModelValidationError("bbox_x2 must be greater than bbox_x1.")
        if self.bbox_y2 <= self.bbox_y1:
            raise PersistenceModelValidationError("bbox_y2 must be greater than bbox_y1.")
        _ensure_confidence(self.detection_confidence, field_name="detection_confidence")
        _ensure_confidence(self.tracker_confidence, field_name="tracker_confidence")
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_payload(self) -> dict[str, object]:
        center_x = self.center_x if self.center_x is not None else (self.bbox_x1 + self.bbox_x2) / 2.0
        center_y = self.center_y if self.center_y is not None else (self.bbox_y1 + self.bbox_y2) / 2.0
        return _omit_none(
            {
                "vehicle_track_id": _serialize_uuid(self.vehicle_track_id),
                "camera_id": _serialize_uuid(self.camera_id),
                "frame_number": self.frame_number,
                "observed_at": _serialize_datetime(self.observed_at),
                "video_time_seconds": self.video_time_seconds,
                "bbox_x1": self.bbox_x1,
                "bbox_y1": self.bbox_y1,
                "bbox_x2": self.bbox_x2,
                "bbox_y2": self.bbox_y2,
                "center_x": center_x,
                "center_y": center_y,
                "detection_confidence": self.detection_confidence,
                "tracker_confidence": self.tracker_confidence,
                "is_key_observation": self.is_key_observation,
                "metadata": self.metadata if self.metadata else {},
            }
        )


@dataclass(frozen=True, slots=True)
class AiModelRecord:
    model_code: str
    model_name: str | None = None
    model_type: str | None = None
    provider: str | None = None
    model_reference: str | None = None
    model_version: str | None = None
    checksum: str | None = None
    configuration: dict[str, Any] = field(default_factory=dict)
    active: bool = True

    def __post_init__(self) -> None:
        if not str(self.model_code).strip():
            raise PersistenceModelValidationError("model_code must not be empty.")
        object.__setattr__(self, "configuration", _copy_metadata(self.configuration, field_name="configuration"))

    def to_payload(self) -> dict[str, object]:
        return _omit_none(
            {
                "model_code": self.model_code,
                "model_name": self.model_name,
                "model_type": self.model_type,
                "provider": self.provider,
                "model_reference": self.model_reference,
                "model_version": self.model_version,
                "checksum": self.checksum,
                "configuration": self.configuration if self.configuration else {},
                "active": self.active,
            }
        )


@dataclass(frozen=True, slots=True)
class RunModelRecord:
    processing_run_id: UUID
    ai_model_id: UUID
    stage_name: str
    device: str | None = None
    resolved_configuration: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.stage_name).strip():
            raise PersistenceModelValidationError("stage_name must not be empty.")
        object.__setattr__(self, "resolved_configuration", _copy_metadata(self.resolved_configuration, field_name="resolved_configuration"))

    def to_payload(self) -> dict[str, object]:
        return _omit_none(
            {
                "processing_run_id": _serialize_uuid(self.processing_run_id),
                "ai_model_id": _serialize_uuid(self.ai_model_id),
                "stage_name": self.stage_name,
                "device": self.device,
                "resolved_configuration": self.resolved_configuration if self.resolved_configuration else {},
            }
        )


@dataclass(frozen=True, slots=True)
class ProcessingErrorRecord:
    severity: str
    message: str
    processing_run_id: UUID | None = None
    camera_run_id: UUID | None = None
    vehicle_track_id: UUID | None = None
    processing_job_id: UUID | None = None
    stage_name: str | None = None
    worker_name: str | None = None
    exception_type: str | None = None
    error_code: str | None = None
    traceback_text: str | None = None
    frame_number: int | None = None
    structured_context: dict[str, Any] = field(default_factory=dict)
    resolution_state: str = "OPEN"

    def __post_init__(self) -> None:
        _ensure_choice(self.severity, field_name="severity", allowed=PROCESSING_ERROR_SEVERITIES, required=True)
        if not str(self.message).strip():
            raise PersistenceModelValidationError("message must not be empty.")
        _ensure_choice(self.resolution_state, field_name="resolution_state", allowed=PROCESSING_ERROR_RESOLUTION_STATES, required=True)
        if self.frame_number is not None:
            _ensure_non_negative(self.frame_number, field_name="frame_number")
        object.__setattr__(self, "structured_context", _copy_metadata(self.structured_context, field_name="structured_context"))

    def to_payload(self) -> dict[str, object]:
        return _omit_none(
            {
                "processing_run_id": _serialize_uuid(self.processing_run_id),
                "camera_run_id": _serialize_uuid(self.camera_run_id),
                "vehicle_track_id": _serialize_uuid(self.vehicle_track_id),
                "processing_job_id": _serialize_uuid(self.processing_job_id),
                "stage_name": self.stage_name,
                "worker_name": self.worker_name,
                "severity": self.severity,
                "exception_type": self.exception_type,
                "error_code": self.error_code,
                "message": self.message,
                "traceback": self.traceback_text,
                "frame_number": self.frame_number,
                "structured_context": self.structured_context if self.structured_context else {},
                "resolution_state": self.resolution_state,
            }
        )


@dataclass(frozen=True, slots=True)
class TrackMediaRecord:
    vehicle_track_id: str
    media_type: str
    storage_uri: str
    storage_provider: str = "LOCAL"
    thumbnail_uri: str | None = None
    mime_type: str | None = "image/jpeg"
    file_size_bytes: int | None = None
    checksum_sha256: str | None = None
    frame_number: int | None = None
    captured_at: datetime | None = None
    video_time_seconds: float | None = None
    bbox: dict[str, Any] | None = None
    width: int | None = None
    height: int | None = None
    quality_score: float | None = None
    sharpness_score: float | None = None
    visibility_score: float | None = None
    occlusion_score: float | None = None
    selection_rank: int | None = None
    is_primary: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.vehicle_track_id).strip():
            raise PersistenceModelValidationError("vehicle_track_id must not be empty.")
        media_type = _ensure_choice(self.media_type, field_name="media_type", allowed=TRACK_MEDIA_TYPES, required=True)
        storage_provider = _ensure_choice(self.storage_provider, field_name="storage_provider", allowed=TRACK_MEDIA_STORAGE_PROVIDERS, required=True)
        storage_uri = _normalize_relative_storage_uri(self.storage_uri)
        thumbnail_uri = None if self.thumbnail_uri is None else _normalize_relative_storage_uri(self.thumbnail_uri)
        if self.file_size_bytes is not None:
            _ensure_non_negative(self.file_size_bytes, field_name="file_size_bytes")
        if self.frame_number is not None:
            _ensure_non_negative(self.frame_number, field_name="frame_number")
        _ensure_timezone_aware(self.captured_at, field_name="captured_at", required=False)
        if self.video_time_seconds is not None:
            _ensure_non_negative(self.video_time_seconds, field_name="video_time_seconds")
        if self.width is not None:
            _ensure_positive(self.width, field_name="width")
        if self.height is not None:
            _ensure_positive(self.height, field_name="height")
        _ensure_confidence(self.quality_score, field_name="quality_score")
        _ensure_confidence(self.sharpness_score, field_name="sharpness_score")
        _ensure_confidence(self.visibility_score, field_name="visibility_score")
        _ensure_confidence(self.occlusion_score, field_name="occlusion_score")
        if self.selection_rank is not None:
            _ensure_non_negative(self.selection_rank, field_name="selection_rank")
        object.__setattr__(self, "media_type", media_type)
        object.__setattr__(self, "storage_provider", storage_provider)
        object.__setattr__(self, "storage_uri", storage_uri)
        object.__setattr__(self, "thumbnail_uri", thumbnail_uri)
        object.__setattr__(self, "bbox", None if self.bbox is None else _copy_metadata(self.bbox, field_name="bbox"))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_payload(self) -> dict[str, object]:
        return _omit_none(
            {
                "vehicle_track_id": self.vehicle_track_id,
                "media_type": self.media_type,
                "storage_provider": self.storage_provider,
                "storage_uri": self.storage_uri,
                "thumbnail_uri": self.thumbnail_uri,
                "mime_type": self.mime_type,
                "file_size_bytes": self.file_size_bytes,
                "checksum_sha256": self.checksum_sha256,
                "frame_number": self.frame_number,
                "captured_at": _serialize_datetime(self.captured_at),
                "video_time_seconds": self.video_time_seconds,
                "bbox": self.bbox,
                "width": self.width,
                "height": self.height,
                "quality_score": self.quality_score,
                "sharpness_score": self.sharpness_score,
                "visibility_score": self.visibility_score,
                "occlusion_score": self.occlusion_score,
                "selection_rank": self.selection_rank,
                "is_primary": self.is_primary,
                "metadata": self.metadata if self.metadata else {},
            }
        )


@dataclass(frozen=True, slots=True)
class VehicleAttributeRecord:
    vehicle_track_id: str
    attribute_scope: str = "TRACK"
    primary_color: str | None = None
    secondary_color: str | None = None
    color_confidence: float | None = None
    vehicle_class: str | None = None
    class_confidence: float | None = None
    attribute_source: str | None = None
    attribute_status: str = "CURRENT"
    observation_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.vehicle_track_id).strip():
            raise PersistenceModelValidationError("vehicle_track_id must not be empty.")
        _ensure_choice(self.attribute_scope, field_name="attribute_scope", allowed=VEHICLE_ATTRIBUTE_SCOPES, required=True)
        _ensure_choice(self.attribute_status, field_name="attribute_status", allowed=VEHICLE_ATTRIBUTE_STATUSES, required=True)
        _ensure_confidence(self.color_confidence, field_name="color_confidence")
        _ensure_confidence(self.class_confidence, field_name="class_confidence")
        _ensure_non_negative(self.observation_count, field_name="observation_count")
        if self.vehicle_class is not None:
            object.__setattr__(self, "vehicle_class", _canonical_vehicle_class(self.vehicle_class))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_payload(self) -> dict[str, object]:
        return _omit_none(
            {
                "vehicle_track_id": self.vehicle_track_id,
                "attribute_scope": self.attribute_scope,
                "primary_color": self.primary_color,
                "secondary_color": self.secondary_color,
                "color_confidence": self.color_confidence,
                "vehicle_class": self.vehicle_class,
                "class_confidence": self.class_confidence,
                "attribute_source": self.attribute_source,
                "attribute_status": self.attribute_status,
                "observation_count": self.observation_count,
                "metadata": self.metadata if self.metadata else {},
            }
        )


@dataclass(frozen=True, slots=True)
class PlateDetectionRecord:
    vehicle_track_id: str
    detected_at: datetime
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    track_observation_id: int | None = None
    track_media_id: str | None = None
    frame_number: int | None = None
    confidence: float | None = None
    detector_name: str | None = None
    detector_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.vehicle_track_id).strip():
            raise PersistenceModelValidationError("vehicle_track_id must not be empty.")
        _ensure_timezone_aware(self.detected_at, field_name="detected_at", required=True)
        if self.bbox_x2 <= self.bbox_x1:
            raise PersistenceModelValidationError("bbox_x2 must be greater than bbox_x1.")
        if self.bbox_y2 <= self.bbox_y1:
            raise PersistenceModelValidationError("bbox_y2 must be greater than bbox_y1.")
        if self.frame_number is not None:
            _ensure_non_negative(self.frame_number, field_name="frame_number")
        _ensure_confidence(self.confidence, field_name="confidence")
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_payload(self) -> dict[str, object]:
        return _omit_none(
            {
                "vehicle_track_id": self.vehicle_track_id,
                "track_observation_id": self.track_observation_id,
                "track_media_id": self.track_media_id,
                "detected_at": _serialize_datetime(self.detected_at),
                "frame_number": self.frame_number,
                "bbox_x1": self.bbox_x1,
                "bbox_y1": self.bbox_y1,
                "bbox_x2": self.bbox_x2,
                "bbox_y2": self.bbox_y2,
                "confidence": self.confidence,
                "detector_name": self.detector_name,
                "detector_version": self.detector_version,
                "metadata": self.metadata if self.metadata else {},
            }
        )


@dataclass(frozen=True, slots=True)
class PlateReadingRecord:
    plate_detection_id: str
    status: str
    ocr_engine: str | None = None
    ocr_version: str | None = None
    raw_text: str | None = None
    normalized_text: str | None = None
    plate_pattern: str | None = None
    confidence: float | None = None
    is_selected: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.plate_detection_id).strip():
            raise PersistenceModelValidationError("plate_detection_id must not be empty.")
        _ensure_choice(self.status, field_name="status", allowed=PLATE_READING_STATUSES, required=True)
        _ensure_confidence(self.confidence, field_name="confidence")
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_payload(self) -> dict[str, object]:
        return _omit_none(
            {
                "plate_detection_id": self.plate_detection_id,
                "ocr_engine": self.ocr_engine,
                "ocr_version": self.ocr_version,
                "raw_text": self.raw_text,
                "normalized_text": self.normalized_text,
                "plate_pattern": self.plate_pattern,
                "confidence": self.confidence,
                "status": self.status,
                "is_selected": self.is_selected,
                "metadata": self.metadata if self.metadata else {},
            }
        )


@dataclass(frozen=True, slots=True)
class PlateSummaryRecord:
    vehicle_track_id: str
    selected_plate_reading_id: str | None = None
    canonical_plate: str | None = None
    plate_pattern: str | None = None
    status: str | None = None
    confidence: float | None = None
    reading_count: int = 0

    def __post_init__(self) -> None:
        if not str(self.vehicle_track_id).strip():
            raise PersistenceModelValidationError("vehicle_track_id must not be empty.")
        _ensure_choice(self.status, field_name="status", allowed=PLATE_READING_STATUSES, required=False)
        _ensure_confidence(self.confidence, field_name="confidence")
        _ensure_non_negative(self.reading_count, field_name="reading_count")

    def to_payload(self) -> dict[str, object]:
        return _omit_none(
            {
                "vehicle_track_id": self.vehicle_track_id,
                "selected_plate_reading_id": self.selected_plate_reading_id,
                "canonical_plate": self.canonical_plate,
                "plate_pattern": self.plate_pattern,
                "status": self.status,
                "confidence": self.confidence,
                "reading_count": self.reading_count,
            }
        )


@dataclass(slots=True)
class TrackPersistenceResult:
    track_uuid: str
    status: str
    database_track_id: str | None
    observations_written: int
    error: str | None = None
    media_persistence: dict[str, object] | None = None


@dataclass(slots=True)
class PersistenceRunMetrics:
    cameras_synced: int = 0
    tracks_considered: int = 0
    tracks_inserted: int = 0
    tracks_already_existing: int = 0
    tracks_skipped_discarded: int = 0
    tracks_skipped_invalid_state: int = 0
    tracks_failed: int = 0
    observations_written: int = 0
    media_records_attempted: int = 0
    media_records_validated: int = 0
    media_records_inserted: int = 0
    media_records_already_existing: int = 0
    media_records_failed: int = 0
    media_files_missing: int = 0
    full_frame_rows_failed: int = 0
    annotated_frame_rows_failed: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "cameras_synced": self.cameras_synced,
            "tracks_considered": self.tracks_considered,
            "tracks_inserted": self.tracks_inserted,
            "tracks_already_existing": self.tracks_already_existing,
            "tracks_skipped_discarded": self.tracks_skipped_discarded,
            "tracks_skipped_invalid_state": self.tracks_skipped_invalid_state,
            "tracks_failed": self.tracks_failed,
            "observations_written": self.observations_written,
            "media_records_attempted": self.media_records_attempted,
            "media_records_validated": self.media_records_validated,
            "media_records_inserted": self.media_records_inserted,
            "media_records_already_existing": self.media_records_already_existing,
            "media_records_failed": self.media_records_failed,
            "media_files_missing": self.media_files_missing,
            "full_frame_rows_failed": self.full_frame_rows_failed,
            "annotated_frame_rows_failed": self.annotated_frame_rows_failed,
            "errors": list(self.errors),
        }
