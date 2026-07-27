from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TypedDict
from uuid import UUID, uuid4


def utc_now() -> datetime:
    return datetime.utcnow()


class PlateReading(TypedDict):
    text: str
    confidence: float


@dataclass(slots=True)
class CameraRecord:
    camera_code: str
    camera_name: str | None = None
    source_path: str | None = None
    enabled: bool = True
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class VehicleTrackRecord:
    camera_id: UUID
    local_track_id: int
    vehicle_class: str
    first_seen_at: datetime
    last_seen_at: datetime
    track_uuid: str
    first_frame_number: int | None = None
    last_frame_number: int | None = None
    observation_count: int = 0
    best_confidence: float | None = None
    best_frame_path: str | None = None
    best_crop_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class VehicleAttributeRecord:
    vehicle_track_id: UUID
    vehicle_colour: str | None = None
    colour_confidence: float | None = None
    plate_text: str | None = None
    plate_pattern: str | None = None
    plate_confidence: float | None = None
    plate_verified: bool = False
    plate_readings: list[PlateReading] = field(default_factory=list)
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class VehicleObservationRecord:
    vehicle_track_id: UUID
    frame_number: int
    observed_at: datetime
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: int | None = None
    created_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class VehicleMatchRecord:
    source_track_id: UUID
    candidate_track_id: UUID
    match_status: str
    plate_similarity: float | None = None
    colour_match: bool = False
    class_match: bool = False
    time_gap_seconds: float | None = None
    match_score: float | None = None
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class VehicleSearchFilters:
    camera_id: UUID | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    vehicle_class: str | None = None
    vehicle_colour: str | None = None
    exact_plate: str | None = None
    partial_plate: str | None = None
    plate_pattern: str | None = None
    match_statuses: tuple[str, ...] = ()
