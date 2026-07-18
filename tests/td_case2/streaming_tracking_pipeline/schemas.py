"""Dataclass schemas for the isolated streaming tracking pipeline foundation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .serialization import dataclass_to_dict
from .validation import (
    validate_enum_value,
    validate_finite_float,
    validate_non_empty_string,
    validate_non_negative_int,
    validate_positive_float,
    validate_positive_int,
    validate_probability,
)


class TrackStatus(str, Enum):
    """Lifecycle states for a tracked object."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    TEMPORARILY_LOST = "temporarily_lost"
    COMPLETED = "completed"


class TrackCompletionReason(str, Enum):
    """Reasons a track can leave the active lifecycle."""

    LOST_BUFFER_EXPIRED = "lost_buffer_expired"
    VIDEO_ENDED = "video_ended"
    STREAM_ENDED = "stream_ended"
    TRACKER_REMOVED = "tracker_removed"
    INVALID_TRACK = "invalid_track"
    MANUAL_FLUSH = "manual_flush"
    UNKNOWN = "unknown"


class TrackLifecycleEventType(str, Enum):
    """Application-level lifecycle event types emitted around tracker IDs."""

    CREATED = "created"
    OBSERVED = "observed"
    CONFIRMED = "confirmed"
    TEMPORARILY_LOST = "temporarily_lost"
    RECOVERED = "recovered"
    COMPLETED = "completed"
    CLASS_UPDATED = "class_updated"
    FLUSHED = "flushed"


class PlateStatus(str, Enum):
    """Constrained plate result states."""

    NOT_ATTEMPTED = "not_attempted"
    VERIFIED = "verified"
    WEAK = "weak"
    INVALID = "invalid"
    NOT_DETECTED = "not_detected"
    ERROR = "error"


class ColourStatus(str, Enum):
    """Constrained colour result states."""

    NOT_ATTEMPTED = "not_attempted"
    VERIFIED = "verified"
    WEAK = "weak"
    INVALID = "invalid"
    NOT_DETECTED = "not_detected"
    ERROR = "error"


@dataclass(frozen=True)
class BoundingBox:
    """XYXY bounding box in frame pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        x1 = validate_finite_float(self.x1, "x1")
        y1 = validate_finite_float(self.y1, "y1")
        x2 = validate_finite_float(self.x2, "x2")
        y2 = validate_finite_float(self.y2, "y2")
        if x2 <= x1:
            raise ValueError("x2 must be greater than x1.")
        if y2 <= y1:
            raise ValueError("y2 must be greater than y1.")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def to_xyxy(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def clip(self, frame_width: int, frame_height: int) -> "BoundingBox":
        """Return this box clipped to frame bounds."""

        width = validate_positive_int(frame_width, "frame_width")
        height = validate_positive_int(frame_height, "frame_height")
        clipped = BoundingBox(
            x1=max(0.0, min(float(width), self.x1)),
            y1=max(0.0, min(float(height), self.y1)),
            x2=max(0.0, min(float(width), self.x2)),
            y2=max(0.0, min(float(height), self.y2)),
        )
        return clipped

    def touches_frame_edge(self, frame_width: int, frame_height: int, margin_ratio: float = 0.0) -> bool:
        """Return whether the box touches a configurable frame-edge margin."""

        width = validate_positive_int(frame_width, "frame_width")
        height = validate_positive_int(frame_height, "frame_height")
        margin = validate_probability(margin_ratio, "margin_ratio") * float(min(width, height))
        return self.x1 <= margin or self.y1 <= margin or self.x2 >= (width - margin) or self.y2 >= (height - margin)


@dataclass
class FramePacket:
    """Runtime frame packet; frame content is intentionally not JSON output."""

    source_id: str
    frame_index: int
    timestamp_sec: float
    source_fps: float
    frame_width: int
    frame_height: int
    frame: Any = None

    def __post_init__(self) -> None:
        self.source_id = validate_non_empty_string(self.source_id, "source_id")
        validate_non_negative_int(self.frame_index, "frame_index")
        validate_finite_float(self.timestamp_sec, "timestamp_sec")
        if self.timestamp_sec < 0.0:
            raise ValueError("timestamp_sec must be non-negative.")
        validate_positive_float(self.source_fps, "source_fps")
        validate_positive_int(self.frame_width, "frame_width")
        validate_positive_int(self.frame_height, "frame_height")


@dataclass(frozen=True)
class DetectionRecord:
    """Detector output for one object in one frame."""

    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str
    raw_class_id: int | None = None
    raw_class_name: str | None = None
    normalized_class_name: str | None = None
    object_group: str | None = None
    detector_source: str | None = None

    def __post_init__(self) -> None:
        validate_probability(self.confidence, "confidence")
        validate_non_negative_int(self.class_id, "class_id")
        validate_non_empty_string(self.class_name, "class_name")
        if self.raw_class_id is not None:
            validate_non_negative_int(self.raw_class_id, "raw_class_id")
        if self.raw_class_name is not None:
            validate_non_empty_string(self.raw_class_name, "raw_class_name")
        if self.normalized_class_name is not None:
            validate_non_empty_string(self.normalized_class_name, "normalized_class_name")
        else:
            object.__setattr__(self, "normalized_class_name", self.class_name)
        if self.raw_class_name is None:
            object.__setattr__(self, "raw_class_name", self.class_name)
        if self.raw_class_id is None:
            object.__setattr__(self, "raw_class_id", self.class_id)
        if self.object_group is not None:
            validate_non_empty_string(self.object_group, "object_group")
        if self.detector_source is not None:
            validate_non_empty_string(self.detector_source, "detector_source")


@dataclass
class DetectionPacket:
    """Detector output packet for a frame."""

    source_id: str
    frame_index: int
    timestamp_sec: float
    frame_width: int
    frame_height: int
    detections: list[DetectionRecord] = field(default_factory=list)
    frame: Any = None

    def __post_init__(self) -> None:
        validate_non_empty_string(self.source_id, "source_id")
        validate_non_negative_int(self.frame_index, "frame_index")
        timestamp = validate_finite_float(self.timestamp_sec, "timestamp_sec")
        if timestamp < 0.0:
            raise ValueError("timestamp_sec must be non-negative.")
        validate_positive_int(self.frame_width, "frame_width")
        validate_positive_int(self.frame_height, "frame_height")


@dataclass(frozen=True)
class TrackedObject:
    """One object emitted by a tracker for one frame."""

    track_id: int
    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str
    frame_index: int
    timestamp_sec: float
    source_track_id: str | int | None = None
    raw_class_id: int | None = None
    raw_class_name: str | None = None
    normalized_class_name: str | None = None
    object_group: str | None = None
    detector_source: str | None = None

    def __post_init__(self) -> None:
        validate_non_negative_int(self.track_id, "track_id")
        validate_probability(self.confidence, "confidence")
        validate_non_negative_int(self.class_id, "class_id")
        validate_non_empty_string(self.class_name, "class_name")
        validate_non_negative_int(self.frame_index, "frame_index")
        timestamp = validate_finite_float(self.timestamp_sec, "timestamp_sec")
        if timestamp < 0.0:
            raise ValueError("timestamp_sec must be non-negative.")
        if isinstance(self.source_track_id, str):
            validate_non_empty_string(self.source_track_id, "source_track_id")
        if self.raw_class_id is not None:
            validate_non_negative_int(self.raw_class_id, "raw_class_id")
        if self.raw_class_name is not None:
            validate_non_empty_string(self.raw_class_name, "raw_class_name")
        if self.normalized_class_name is not None:
            validate_non_empty_string(self.normalized_class_name, "normalized_class_name")
        else:
            object.__setattr__(self, "normalized_class_name", self.class_name)
        if self.raw_class_name is None:
            object.__setattr__(self, "raw_class_name", self.class_name)
        if self.raw_class_id is None:
            object.__setattr__(self, "raw_class_id", self.class_id)
        if self.object_group is not None:
            validate_non_empty_string(self.object_group, "object_group")
        if self.detector_source is not None:
            validate_non_empty_string(self.detector_source, "detector_source")


@dataclass
class TrackedFramePacket:
    """Tracker output packet for a frame."""

    source_id: str
    frame_index: int
    timestamp_sec: float
    frame_width: int
    frame_height: int
    tracks: list[TrackedObject] = field(default_factory=list)
    frame: Any = None

    def __post_init__(self) -> None:
        validate_non_empty_string(self.source_id, "source_id")
        validate_non_negative_int(self.frame_index, "frame_index")
        timestamp = validate_finite_float(self.timestamp_sec, "timestamp_sec")
        if timestamp < 0.0:
            raise ValueError("timestamp_sec must be non-negative.")
        validate_positive_int(self.frame_width, "frame_width")
        validate_positive_int(self.frame_height, "frame_height")


@dataclass(frozen=True)
class CropQualityMetrics:
    """Quality inputs for later crop selection without defining a scoring formula."""

    detection_confidence: float
    bbox_area_ratio: float
    sharpness: float | None = None
    brightness: float | None = None
    edge_touching: bool = False
    occlusion_score: float | None = None
    plate_visibility_score: float | None = None
    combined_score: float | None = None
    crop_width: int | None = None
    crop_height: int | None = None
    contrast: float | None = None
    crop_completeness: float | None = None
    padding_clipped: bool = False
    preliminary_score: float | None = None

    def __post_init__(self) -> None:
        validate_probability(self.detection_confidence, "detection_confidence")
        validate_probability(self.bbox_area_ratio, "bbox_area_ratio")
        for field_name in ("occlusion_score", "plate_visibility_score", "combined_score", "crop_completeness", "preliminary_score"):
            value = getattr(self, field_name)
            if value is not None:
                validate_probability(value, field_name)
        for field_name in ("sharpness", "brightness", "contrast"):
            value = getattr(self, field_name)
            if value is not None:
                validate_finite_float(value, field_name)
        for field_name in ("crop_width", "crop_height"):
            value = getattr(self, field_name)
            if value is not None:
                validate_positive_int(value, field_name)


@dataclass(frozen=True)
class CropCandidate:
    """Serializable candidate crop reference for one track observation."""

    track_id: int
    frame_index: int
    timestamp_sec: float
    bbox: BoundingBox
    full_frame_path: str | None
    vehicle_crop_path: str | None
    quality: CropQualityMetrics
    is_primary: bool = False
    is_fallback: bool = False
    source_id: str | None = None
    track_generation: int = 0
    source_track_id: str | int | None = None
    crop_bbox: BoundingBox | None = None
    class_name: str | None = None
    detection_confidence: float | None = None
    preliminary_rank_score: float | None = None
    retention_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_non_negative_int(self.track_id, "track_id")
        validate_non_negative_int(self.frame_index, "frame_index")
        timestamp = validate_finite_float(self.timestamp_sec, "timestamp_sec")
        if timestamp < 0.0:
            raise ValueError("timestamp_sec must be non-negative.")
        if self.is_primary and self.is_fallback:
            raise ValueError("A crop cannot be both primary and fallback in this schema.")
        if self.source_id is not None:
            validate_non_empty_string(self.source_id, "source_id")
        validate_non_negative_int(self.track_generation, "track_generation")
        if isinstance(self.source_track_id, str):
            validate_non_empty_string(self.source_track_id, "source_track_id")
        if self.crop_bbox is None:
            object.__setattr__(self, "crop_bbox", self.bbox)
        if self.class_name is not None:
            validate_non_empty_string(self.class_name, "class_name")
        if self.detection_confidence is not None:
            validate_probability(self.detection_confidence, "detection_confidence")
        if self.preliminary_rank_score is not None:
            validate_probability(self.preliminary_rank_score, "preliminary_rank_score")
        if self.retention_reason is not None:
            validate_non_empty_string(self.retention_reason, "retention_reason")


@dataclass(frozen=True)
class TrackRecord:
    """Aggregated lifecycle record for one track, without tracker algorithms."""

    source_id: str
    track_id: int
    status: TrackStatus
    first_seen_frame: int
    last_seen_frame: int
    first_seen_sec: float
    last_seen_sec: float
    observation_count: int
    missed_frame_count: int
    class_votes: dict[str, int] = field(default_factory=dict)
    crop_candidates: list[CropCandidate] = field(default_factory=list)
    completion_reason: TrackCompletionReason | None = None
    source_track_id: str | int | None = None
    track_generation: int = 0
    last_bbox: BoundingBox | None = None
    last_confidence: float | None = None
    last_class_id: int | None = None
    last_class_name: str | None = None
    object_group: str | None = None

    def __post_init__(self) -> None:
        validate_non_empty_string(self.source_id, "source_id")
        validate_non_negative_int(self.track_id, "track_id")
        object.__setattr__(self, "status", validate_enum_value(self.status, TrackStatus, "status"))
        if self.completion_reason is not None:
            object.__setattr__(
                self,
                "completion_reason",
                validate_enum_value(self.completion_reason, TrackCompletionReason, "completion_reason"),
            )
        validate_non_negative_int(self.first_seen_frame, "first_seen_frame")
        validate_non_negative_int(self.last_seen_frame, "last_seen_frame")
        if self.last_seen_frame < self.first_seen_frame:
            raise ValueError("last_seen_frame must be greater than or equal to first_seen_frame.")
        first_sec = validate_finite_float(self.first_seen_sec, "first_seen_sec")
        last_sec = validate_finite_float(self.last_seen_sec, "last_seen_sec")
        if first_sec < 0.0 or last_sec < 0.0:
            raise ValueError("track timestamps must be non-negative.")
        if last_sec < first_sec:
            raise ValueError("last_seen_sec must be greater than or equal to first_seen_sec.")
        validate_non_negative_int(self.observation_count, "observation_count")
        validate_non_negative_int(self.missed_frame_count, "missed_frame_count")
        for class_name, count in self.class_votes.items():
            validate_non_empty_string(class_name, "class_votes key")
            validate_non_negative_int(count, f"class_votes[{class_name}]")
        if isinstance(self.source_track_id, str):
            validate_non_empty_string(self.source_track_id, "source_track_id")
        validate_non_negative_int(self.track_generation, "track_generation")
        if self.last_confidence is not None:
            validate_probability(self.last_confidence, "last_confidence")
        if self.last_class_id is not None:
            validate_non_negative_int(self.last_class_id, "last_class_id")
        if self.last_class_name is not None:
            validate_non_empty_string(self.last_class_name, "last_class_name")
        if self.object_group is not None:
            validate_non_empty_string(self.object_group, "object_group")

    @property
    def duration_sec(self) -> float:
        return self.last_seen_sec - self.first_seen_sec

    @property
    def dominant_class(self) -> str | None:
        if not self.class_votes:
            return None
        return sorted(self.class_votes.items(), key=lambda item: (-item[1], item[0]))[0][0]

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-safe dictionary."""

        return dataclass_to_dict(self)


@dataclass(frozen=True)
class TrackLifecycleEvent:
    """JSON-safe application lifecycle event for one normalized track ID."""

    event_type: TrackLifecycleEventType
    source_id: str
    track_id: int
    source_track_id: str | int | None
    frame_index: int
    timestamp_sec: float
    previous_status: TrackStatus | None
    new_status: TrackStatus
    observation_count: int
    missed_processed_frames: int
    reason: TrackCompletionReason | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    track_generation: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", validate_enum_value(self.event_type, TrackLifecycleEventType, "event_type"))
        validate_non_empty_string(self.source_id, "source_id")
        validate_non_negative_int(self.track_id, "track_id")
        if isinstance(self.source_track_id, str):
            validate_non_empty_string(self.source_track_id, "source_track_id")
        validate_non_negative_int(self.frame_index, "frame_index")
        timestamp = validate_finite_float(self.timestamp_sec, "timestamp_sec")
        if timestamp < 0.0:
            raise ValueError("timestamp_sec must be non-negative.")
        if self.previous_status is not None:
            object.__setattr__(self, "previous_status", validate_enum_value(self.previous_status, TrackStatus, "previous_status"))
        object.__setattr__(self, "new_status", validate_enum_value(self.new_status, TrackStatus, "new_status"))
        validate_non_negative_int(self.observation_count, "observation_count")
        validate_non_negative_int(self.missed_processed_frames, "missed_processed_frames")
        if self.reason is not None:
            object.__setattr__(self, "reason", validate_enum_value(self.reason, TrackCompletionReason, "reason"))
        validate_non_negative_int(self.track_generation, "track_generation")

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-safe dictionary."""

        return dataclass_to_dict(self)


@dataclass(frozen=True)
class PlateResult:
    """Plate OCR result container; no regex validation is performed here."""

    raw_text: str | None = None
    normalized_text: str | None = None
    confidence: float | None = None
    verified: bool = False
    status: PlateStatus = PlateStatus.NOT_ATTEMPTED
    plate_crop_path: str | None = None
    source_frame_index: int | None = None
    attempt_number: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", validate_enum_value(self.status, PlateStatus, "status"))
        if self.confidence is not None:
            validate_probability(self.confidence, "confidence")
        if self.source_frame_index is not None:
            validate_non_negative_int(self.source_frame_index, "source_frame_index")
        validate_non_negative_int(self.attempt_number, "attempt_number")


@dataclass(frozen=True)
class ColourResult:
    """Vehicle colour result container."""

    raw_text: str | None = None
    normalized_colour: str | None = None
    confidence: float | None = None
    status: ColourStatus = ColourStatus.NOT_ATTEMPTED
    source_frame_index: int | None = None
    attempt_number: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", validate_enum_value(self.status, ColourStatus, "status"))
        if self.confidence is not None:
            validate_probability(self.confidence, "confidence")
        if self.source_frame_index is not None:
            validate_non_negative_int(self.source_frame_index, "source_frame_index")
        validate_non_negative_int(self.attempt_number, "attempt_number")


@dataclass(frozen=True)
class ObjectRecord:
    """Search-index-ready object record produced after future enrichment stages."""

    source_id: str
    track_id: int
    object_class: str
    first_seen_frame: int
    last_seen_frame: int
    first_seen_sec: float
    last_seen_sec: float
    observation_count: int
    track_status: TrackStatus
    completion_reason: TrackCompletionReason | None
    primary_crops: list[CropCandidate] = field(default_factory=list)
    fallback_crop: CropCandidate | None = None
    plate: PlateResult = field(default_factory=PlateResult)
    colour: ColourResult = field(default_factory=ColourResult)
    full_frame_paths: list[str] = field(default_factory=list)
    vehicle_crop_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    source_track_id: str | int | None = None

    def __post_init__(self) -> None:
        validate_non_empty_string(self.source_id, "source_id")
        validate_non_negative_int(self.track_id, "track_id")
        validate_non_empty_string(self.object_class, "object_class")
        validate_non_negative_int(self.first_seen_frame, "first_seen_frame")
        validate_non_negative_int(self.last_seen_frame, "last_seen_frame")
        if self.last_seen_frame < self.first_seen_frame:
            raise ValueError("last_seen_frame must be greater than or equal to first_seen_frame.")
        first_sec = validate_finite_float(self.first_seen_sec, "first_seen_sec")
        last_sec = validate_finite_float(self.last_seen_sec, "last_seen_sec")
        if first_sec < 0.0 or last_sec < 0.0:
            raise ValueError("object timestamps must be non-negative.")
        if last_sec < first_sec:
            raise ValueError("last_seen_sec must be greater than or equal to first_seen_sec.")
        validate_non_negative_int(self.observation_count, "observation_count")
        object.__setattr__(self, "track_status", validate_enum_value(self.track_status, TrackStatus, "track_status"))
        if self.completion_reason is not None:
            object.__setattr__(
                self,
                "completion_reason",
                validate_enum_value(self.completion_reason, TrackCompletionReason, "completion_reason"),
            )
        for path_value in self.full_frame_paths + self.vehicle_crop_paths:
            validate_non_empty_string(path_value, "path")
        if isinstance(self.source_track_id, str):
            validate_non_empty_string(self.source_track_id, "source_track_id")

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-safe dictionary."""

        return dataclass_to_dict(self)
