from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


VALID_MATCH_DECISIONS = ("CONFIRMED", "POSSIBLE", "REJECTED", "INSUFFICIENT_EVIDENCE", "REVIEW_REQUIRED")
VALID_GLOBAL_OBJECT_STATUSES = ("ACTIVE", "CONFIRMED", "POSSIBLE", "REVIEW_REQUIRED", "INVALIDATED")
VALID_CREATION_METHODS = ("VERIFIED_PLATE", "RULE_BASED", "MANUAL", "SINGLE_TRACK")
VALID_MEMBERSHIP_STATUSES = ("CONFIRMED", "POSSIBLE", "REVIEW_REQUIRED", "REJECTED")


class GlobalMatchModelError(ValueError):
    """Raised when a cross-camera model payload is invalid."""


def _ensure_finite_unit_interval(value: float | None, *, field_name: str) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if math.isnan(numeric) or math.isinf(numeric):
        raise GlobalMatchModelError(f"{field_name} must be finite.")
    if not 0.0 <= numeric <= 1.0:
        raise GlobalMatchModelError(f"{field_name} must be between 0 and 1.")
    return numeric


def _ensure_choice(value: str, *, field_name: str, allowed: tuple[str, ...]) -> str:
    normalized = str(value).strip().upper()
    if normalized not in allowed:
        raise GlobalMatchModelError(f"{field_name} must be one of: {', '.join(allowed)}")
    return normalized


def _ensure_timezone_aware(value: datetime | None, *, field_name: str) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise GlobalMatchModelError(f"{field_name} must be timezone-aware.")
    return value


@dataclass(frozen=True, slots=True)
class TrackIdentityFeatures:
    vehicle_track_id: str
    track_uuid: str
    processing_run_id: str
    camera_id: str
    camera_code: str
    canonical_class: str | None
    canonical_colour: str | None
    colour_confidence: float | None
    normalized_plate: str | None
    plate_status: str | None
    plate_confidence: float | None
    first_seen_at: datetime | None
    last_seen_at: datetime | None
    first_video_time_seconds: float | None
    last_video_time_seconds: float | None
    primary_media_uri: str | None = None
    body_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.vehicle_track_id).strip():
            raise GlobalMatchModelError("vehicle_track_id must not be empty.")
        if not str(self.track_uuid).strip():
            raise GlobalMatchModelError("track_uuid must not be empty.")
        if not str(self.processing_run_id).strip():
            raise GlobalMatchModelError("processing_run_id must not be empty.")
        if not str(self.camera_id).strip():
            raise GlobalMatchModelError("camera_id must not be empty.")
        if not str(self.camera_code).strip():
            raise GlobalMatchModelError("camera_code must not be empty.")
        _ensure_finite_unit_interval(self.colour_confidence, field_name="colour_confidence")
        _ensure_finite_unit_interval(self.plate_confidence, field_name="plate_confidence")
        _ensure_timezone_aware(self.first_seen_at, field_name="first_seen_at")
        _ensure_timezone_aware(self.last_seen_at, field_name="last_seen_at")
        if self.first_seen_at is not None and self.last_seen_at is not None and self.first_seen_at > self.last_seen_at:
            raise GlobalMatchModelError("first_seen_at must not be after last_seen_at.")
        if self.first_video_time_seconds is not None and self.first_video_time_seconds < 0:
            raise GlobalMatchModelError("first_video_time_seconds must be non-negative.")
        if self.last_video_time_seconds is not None and self.last_video_time_seconds < 0:
            raise GlobalMatchModelError("last_video_time_seconds must be non-negative.")


@dataclass(frozen=True, slots=True)
class CrossCameraMatchResult:
    left_track_uuid: str
    right_track_uuid: str
    left_vehicle_track_id: str
    right_vehicle_track_id: str
    decision: str
    score: float
    plate_score: float
    time_score: float
    camera_route_score: float
    class_score: float
    colour_score: float
    visual_score: float
    reasons: tuple[str, ...]
    rule_version: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.left_track_uuid).strip() or not str(self.right_track_uuid).strip():
            raise GlobalMatchModelError("track UUIDs must not be empty.")
        if self.left_vehicle_track_id == self.right_vehicle_track_id:
            raise GlobalMatchModelError("left and right vehicle track IDs must be distinct.")
        object.__setattr__(self, "decision", _ensure_choice(self.decision, field_name="decision", allowed=VALID_MATCH_DECISIONS))
        for field_name in ("score", "plate_score", "time_score", "camera_route_score", "class_score", "colour_score", "visual_score"):
            _ensure_finite_unit_interval(getattr(self, field_name), field_name=field_name)
        if not str(self.rule_version).strip():
            raise GlobalMatchModelError("rule_version must not be empty.")
        object.__setattr__(self, "reasons", tuple(str(item) for item in self.reasons if str(item).strip()))


@dataclass(frozen=True, slots=True)
class GlobalObjectMembership:
    vehicle_track_id: str
    track_uuid: str
    membership_status: str
    membership_confidence: float | None
    match_method: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.vehicle_track_id).strip():
            raise GlobalMatchModelError("vehicle_track_id must not be empty.")
        if not str(self.track_uuid).strip():
            raise GlobalMatchModelError("track_uuid must not be empty.")
        object.__setattr__(
            self,
            "membership_status",
            _ensure_choice(self.membership_status, field_name="membership_status", allowed=VALID_MEMBERSHIP_STATUSES),
        )
        _ensure_finite_unit_interval(self.membership_confidence, field_name="membership_confidence")
        if not str(self.match_method).strip():
            raise GlobalMatchModelError("match_method must not be empty.")


@dataclass(frozen=True, slots=True)
class GlobalVehicleObjectProposal:
    processing_run_id: str
    global_object_code: str
    status: str
    confidence: float | None
    canonical_plate: str | None
    canonical_colour: str | None
    canonical_vehicle_class: str | None
    first_seen_at: datetime | None
    last_seen_at: datetime | None
    creation_method: str
    camera_count: int
    track_count: int
    members: tuple[GlobalObjectMembership, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.processing_run_id).strip():
            raise GlobalMatchModelError("processing_run_id must not be empty.")
        if not str(self.global_object_code).strip():
            raise GlobalMatchModelError("global_object_code must not be empty.")
        object.__setattr__(self, "status", _ensure_choice(self.status, field_name="status", allowed=VALID_GLOBAL_OBJECT_STATUSES))
        object.__setattr__(self, "creation_method", _ensure_choice(self.creation_method, field_name="creation_method", allowed=VALID_CREATION_METHODS))
        _ensure_finite_unit_interval(self.confidence, field_name="confidence")
        _ensure_timezone_aware(self.first_seen_at, field_name="first_seen_at")
        _ensure_timezone_aware(self.last_seen_at, field_name="last_seen_at")
        if self.first_seen_at is not None and self.last_seen_at is not None and self.first_seen_at > self.last_seen_at:
            raise GlobalMatchModelError("first_seen_at must not be after last_seen_at.")
        if self.camera_count < 0 or self.track_count < 0:
            raise GlobalMatchModelError("camera_count and track_count must be non-negative.")
        if not self.members:
            raise GlobalMatchModelError("members must not be empty.")
