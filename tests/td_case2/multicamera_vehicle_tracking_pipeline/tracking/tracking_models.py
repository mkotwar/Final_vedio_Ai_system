from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..evidence.evidence_models import TrackEvidencePackage


TRACK_STATES = ("tentative", "active", "temporarily_lost", "completed", "discarded")


def validate_track_state(value: str) -> str:
    if value not in TRACK_STATES:
        raise ValueError(f"Unsupported track state: {value}")
    return value


@dataclass(frozen=True, slots=True)
class TrackObservation:
    camera_code: str
    local_track_id: int
    frame_number: int
    video_time_seconds: float
    camera_timestamp: datetime | None
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    track_uuid: str = ""
    state: str = "tentative"
    raw_class_name: str | None = None

    def __post_init__(self) -> None:
        validate_track_state(self.state)


@dataclass(frozen=True, slots=True)
class ClassObservation:
    frame_number: int
    video_time_seconds: float
    camera_timestamp: datetime | None
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    raw_class_name: str | None = None


@dataclass(frozen=True, slots=True)
class TrackClassDiagnostics:
    provisional_class_name: str | None = None
    stable_class_name: str | None = None
    class_is_locked: bool = False
    class_confidence: float | None = None
    class_winner_margin: float | None = None
    class_observation_count: int = 0
    class_conflict_count: int = 0
    class_scores: dict[str, float] = field(default_factory=dict)
    class_observation_counts: dict[str, int] = field(default_factory=dict)
    class_max_confidences: dict[str, float] = field(default_factory=dict)
    raw_class_history: list[ClassObservation] = field(default_factory=list)
    latest_observation_class_name: str | None = None
    linked_track_group_id: str | None = None
    fragment_candidate_track_uuids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LocalVehicleTrack:
    track_uuid: str
    camera_code: str
    local_track_id: int
    class_name: str
    first_frame_number: int
    last_frame_number: int
    first_seen_at: datetime | None
    last_seen_at: datetime | None
    first_video_time_seconds: float
    last_video_time_seconds: float
    observation_count: int
    best_confidence: float
    state: str
    observations: list[TrackObservation] = field(default_factory=list)
    camera_name: str | None = None
    source_path: Path | None = None
    lost_frame_count: int = 0
    evidence_package: TrackEvidencePackage | None = None
    provisional_class_name: str | None = None
    stable_class_name: str | None = None
    class_is_locked: bool = False
    class_confidence: float | None = None
    class_winner_margin: float | None = None
    class_observation_count: int = 0
    class_conflict_count: int = 0
    class_scores: dict[str, float] = field(default_factory=dict)
    class_observation_counts: dict[str, int] = field(default_factory=dict)
    class_max_confidences: dict[str, float] = field(default_factory=dict)
    raw_class_history: list[ClassObservation] = field(default_factory=list)
    latest_observation_class_name: str | None = None
    linked_track_group_id: str | None = None
    fragment_candidate_track_uuids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.state = validate_track_state(self.state)
