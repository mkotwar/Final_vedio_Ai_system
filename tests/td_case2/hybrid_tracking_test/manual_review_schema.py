from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


OBJECT_REVIEW_STATUSES = [
    "correct_single_object",
    "fragmented_object",
    "duplicate_track",
    "false_detection",
    "wrong_class",
    "wrong_object_family",
    "track_switch",
    "partial_object_only",
    "uncertain",
]

CROP_REVIEW_STATUSES = [
    "primary_crop_good",
    "alternative_crop_better",
    "all_crops_bad",
    "crop_contains_wrong_object",
    "crop_too_blurry",
    "crop_too_small",
    "crop_too_clipped",
    "crop_uncertain",
]

TIMELINE_REVIEW_STATUSES = [
    "timeline_correct",
    "start_too_early",
    "start_too_late",
    "end_too_early",
    "end_too_late",
    "timeline_contains_drift",
    "timeline_uncertain",
]

CLASS_REVIEW_STATUSES = [
    "class_correct",
    "should_be_person",
    "should_be_car",
    "should_be_motorcycle",
    "should_be_bus",
    "should_be_truck",
    "should_be_generic_vehicle",
    "class_uncertain",
]

DOWNSTREAM_DECISIONS = [
    "ready",
    "fallback",
    "manual_review",
    "reject",
]

FALSE_DETECTION_REASONS = [
    "background region",
    "shadow",
    "partial vehicle mistaken as new object",
    "duplicate detector box",
    "wrong-class detection",
    "KCF drift",
    "frame-edge artifact",
    "unknown",
]

MERGE_REVIEW_DECISIONS = [
    "merge_correct",
    "merge_incorrect",
    "uncertain",
]

MERGE_INCORRECT_REASONS = [
    "two different simultaneous objects",
    "different appearance",
    "impossible movement",
    "wrong direction",
    "incorrect overlap duplicate",
    "class-family conflict",
    "another reason",
]

POSSIBLE_MERGE_REVIEW_DECISIONS = [
    "accept_merge",
    "reject_merge",
    "uncertain",
]

WARNING_GROUPS = {
    "critical": [
        "frozen_kcf_detected",
        "boundary_stuck_detected",
        "track_switch",
        "impossible_jump",
        "missing_valid_crop",
    ],
    "important": [
        "single_detection",
        "short_track",
        "class_instability",
        "long_kcf_only_gap_detected",
    ],
    "informational": [
        "boundary_partial",
        "lost_recovery_expired",
        "missed_refresh_termination",
    ],
}


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def normalize_int_list(values: Any) -> list[int]:
    if values is None:
        return []
    if isinstance(values, str):
        tokens = [item.strip() for item in values.replace(";", ",").split(",")]
        result: list[int] = []
        for token in tokens:
            if not token:
                continue
            try:
                result.append(int(token))
            except ValueError:
                continue
        return result
    if isinstance(values, (list, tuple, set)):
        result = []
        for value in values:
            try:
                result.append(int(value))
            except (TypeError, ValueError):
                continue
        return result
    try:
        return [int(values)]
    except (TypeError, ValueError):
        return []


@dataclass
class ObjectReview:
    local_object_id: int
    camera_id: str
    manual_real_object_id: str = ""
    object_review_status: str = "uncertain"
    crop_review_status: str = "crop_uncertain"
    timeline_review_status: str = "timeline_uncertain"
    class_review_status: str = "class_uncertain"
    downstream_decision: str = "manual_review"
    manual_class: str = ""
    same_real_object_as_local_object_ids: list[int] = field(default_factory=list)
    suggested_real_object_group: str = ""
    false_detection_reason: str = ""
    switch_timestamp_seconds: float | None = None
    switch_original_object: str = ""
    switch_new_object: str = ""
    track_should_be_split: bool | None = None
    reviewer_notes: str = ""
    reviewed_at: str = field(default_factory=now_iso)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ObjectReview":
        return cls(
            local_object_id=int(payload["local_object_id"]),
            camera_id=str(payload["camera_id"]),
            manual_real_object_id=str(payload.get("manual_real_object_id", "")),
            object_review_status=str(payload.get("object_review_status", "uncertain")),
            crop_review_status=str(payload.get("crop_review_status", "crop_uncertain")),
            timeline_review_status=str(payload.get("timeline_review_status", "timeline_uncertain")),
            class_review_status=str(payload.get("class_review_status", "class_uncertain")),
            downstream_decision=str(payload.get("downstream_decision", "manual_review")),
            manual_class=str(payload.get("manual_class", "")),
            same_real_object_as_local_object_ids=normalize_int_list(payload.get("same_real_object_as_local_object_ids")),
            suggested_real_object_group=str(payload.get("suggested_real_object_group", "")),
            false_detection_reason=str(payload.get("false_detection_reason", "")),
            switch_timestamp_seconds=(
                None
                if payload.get("switch_timestamp_seconds") in (None, "")
                else float(payload["switch_timestamp_seconds"])
            ),
            switch_original_object=str(payload.get("switch_original_object", "")),
            switch_new_object=str(payload.get("switch_new_object", "")),
            track_should_be_split=payload.get("track_should_be_split"),
            reviewer_notes=str(payload.get("reviewer_notes", "")),
            reviewed_at=str(payload.get("reviewed_at", now_iso())),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MergeReview:
    local_object_id: int
    source_track_ids: list[int]
    decision: str = "uncertain"
    incorrect_reason: str = ""
    reviewer_notes: str = ""
    reviewed_at: str = field(default_factory=now_iso)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MergeReview":
        return cls(
            local_object_id=int(payload["local_object_id"]),
            source_track_ids=normalize_int_list(payload.get("source_track_ids")),
            decision=str(payload.get("decision", "uncertain")),
            incorrect_reason=str(payload.get("incorrect_reason", "")),
            reviewer_notes=str(payload.get("reviewer_notes", "")),
            reviewed_at=str(payload.get("reviewed_at", now_iso())),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PossibleMergeReview:
    candidate_key: str
    from_track_id: int
    to_track_id: int
    decision: str = "uncertain"
    reviewer_notes: str = ""
    reviewed_at: str = field(default_factory=now_iso)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PossibleMergeReview":
        return cls(
            candidate_key=str(payload["candidate_key"]),
            from_track_id=int(payload["from_track_id"]),
            to_track_id=int(payload["to_track_id"]),
            decision=str(payload.get("decision", "uncertain")),
            reviewer_notes=str(payload.get("reviewer_notes", "")),
            reviewed_at=str(payload.get("reviewed_at", now_iso())),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
