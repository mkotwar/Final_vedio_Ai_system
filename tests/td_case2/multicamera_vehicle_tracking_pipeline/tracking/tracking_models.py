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
    native_tracker_id: int | None = None
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
    class_status: str | None = None
    final_class_reason: str | None = None
    class_is_locked: bool = False
    class_confidence: float | None = None
    class_winner_margin: float | None = None
    class_observation_count: int = 0
    class_conflict_count: int = 0
    winning_class_name: str | None = None
    winning_class_count: int = 0
    winning_class_ratio: float | None = None
    runner_up_class_name: str | None = None
    runner_up_class_count: int = 0
    runner_up_ratio: float | None = None
    winner_count_margin: int = 0
    count_winner_class_name: str | None = None
    score_winner_class_name: str | None = None
    winners_agree: bool = True
    maximum_consecutive_winner_count: int = 0
    recent_consecutive_winner_count: int = 0
    class_transition_count: int = 0
    incompatible_class_transition_count: int = 0
    recent_class_counts: dict[str, int] = field(default_factory=dict)
    recent_winning_class_name: str | None = None
    recent_winning_ratio: float | None = None
    recent_observation_count: int = 0
    strong_conflict_detected: bool = False
    split_recommended: bool = False
    mixed_identity_detected: bool = False
    mixed_identity_classes: tuple[str, ...] = ()
    mixed_identity_start_frame: int | None = None
    mixed_identity_confidence: float | None = None
    final_class_blocked_due_to_mixed_identity: bool = False
    class_scores: dict[str, float] = field(default_factory=dict)
    class_ratios: dict[str, float] = field(default_factory=dict)
    class_observation_counts: dict[str, int] = field(default_factory=dict)
    class_max_confidences: dict[str, float] = field(default_factory=dict)
    winner_confidence_sum: float | None = None
    raw_class_history: list[ClassObservation] = field(default_factory=list)
    latest_observation_class_name: str | None = None
    linked_track_group_id: str | None = None
    fragment_candidate_track_uuids: list[str] = field(default_factory=list)
    possible_identity_switch: bool = False


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
    native_tracker_ids_seen: list[int] = field(default_factory=list)
    reactivation_count: int = 0
    fragment_relink_count: int = 0
    maximum_consecutive_missing_frames: int = 0
    split_from_track_uuid: str | None = None
    completion_reason: str | None = None
    split_executed: bool = False
    split_frame: int | None = None
    split_reason_codes: list[str] = field(default_factory=list)
    source_logical_track_id: int | None = None
    new_logical_track_id: int | None = None
    split_native_tracker_id: int | None = None
    pending_conflict_observation_count: int = 0
    stable_class_before_split: str | None = None
    conflicting_class: str | None = None
    average_conflict_confidence: float | None = None
    bbox_iou_at_split: float | None = None
    center_distance_at_split: float | None = None
    normalized_center_distance_at_split: float | None = None
    area_ratio_at_split: float | None = None
    width_ratio_at_split: float | None = None
    height_ratio_at_split: float | None = None
    spatial_score_at_split: float | None = None
    provisional_class_name: str | None = None
    stable_class_name: str | None = None
    class_status: str | None = None
    final_class_reason: str | None = None
    class_is_locked: bool = False
    class_confidence: float | None = None
    class_winner_margin: float | None = None
    class_observation_count: int = 0
    class_conflict_count: int = 0
    winning_class_name: str | None = None
    winning_class_count: int = 0
    winning_class_ratio: float | None = None
    runner_up_class_name: str | None = None
    runner_up_class_count: int = 0
    runner_up_ratio: float | None = None
    winner_count_margin: int = 0
    count_winner_class_name: str | None = None
    score_winner_class_name: str | None = None
    winners_agree: bool = True
    maximum_consecutive_winner_count: int = 0
    recent_consecutive_winner_count: int = 0
    class_transition_count: int = 0
    incompatible_class_transition_count: int = 0
    recent_class_counts: dict[str, int] = field(default_factory=dict)
    recent_winning_class_name: str | None = None
    recent_winning_ratio: float | None = None
    recent_observation_count: int = 0
    strong_conflict_detected: bool = False
    split_recommended: bool = False
    mixed_identity_detected: bool = False
    mixed_identity_classes: tuple[str, ...] = ()
    mixed_identity_start_frame: int | None = None
    mixed_identity_confidence: float | None = None
    final_class_blocked_due_to_mixed_identity: bool = False
    class_scores: dict[str, float] = field(default_factory=dict)
    class_ratios: dict[str, float] = field(default_factory=dict)
    class_observation_counts: dict[str, int] = field(default_factory=dict)
    class_max_confidences: dict[str, float] = field(default_factory=dict)
    winner_confidence_sum: float | None = None
    raw_class_history: list[ClassObservation] = field(default_factory=list)
    latest_observation_class_name: str | None = None
    linked_track_group_id: str | None = None
    fragment_candidate_track_uuids: list[str] = field(default_factory=list)
    possible_identity_switch: bool = False

    def __post_init__(self) -> None:
        self.state = validate_track_state(self.state)
