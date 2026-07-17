from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - optional
    np = None


VALID_TRACK_STATUSES = {
    "tentative",
    "confirmed",
    "propagated",
    "propagated_unconfirmed",
    "temporarily_lost",
    "reactivated",
    "lost",
    "exited",
    "completed",
    "removed",
}


def to_json_safe(value: Any) -> Any:
    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    return value


@dataclass
class DetectionObservation:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    model_source: str
    detection_index: int = 0
    source_frame_index: int = 0
    processed_frame_index: int = 0
    timestamp_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection_index": int(self.detection_index),
            "class_id": int(self.class_id),
            "class_name": str(self.class_name),
            "confidence": round(float(self.confidence), 6),
            "bbox_xyxy": [round(float(value), 3) for value in self.bbox_xyxy],
            "model_source": str(self.model_source),
        }


@dataclass
class ValidationResult:
    valid: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float | int | bool | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": bool(self.valid),
            "reasons": list(self.reasons),
            "metrics": to_json_safe(dict(self.metrics)),
        }


@dataclass
class HybridTrack:
    track_id: int
    class_id: int
    class_name: str
    bbox_xyxy: list[float]
    previous_bbox_xyxy: list[float] | None
    bbox_source: str
    last_update_source: str
    created_frame_index: int
    created_timestamp_seconds: float
    last_update_frame_index: int
    last_update_timestamp_seconds: float
    last_detection_frame_index: int
    last_detection_timestamp_seconds: float
    last_detection_confidence: float | None
    object_family: str = "other"
    age_frames: int = 0
    detection_hits: int = 0
    propagation_hits: int = 0
    consecutive_kcf_failures: int = 0
    missed_detection_refreshes: int = 0
    consecutive_unreliable_updates: int = 0
    status: str = "tentative"
    is_confirmed: bool = False
    is_active: bool = True
    kcf_initialized: bool = False
    kcf_success: bool | None = None
    last_propagated_box_valid: bool = False
    kcf_instance: Any | None = None
    appearance_histogram: list[float] | None = None
    trajectory_history: list[dict[str, Any]] = field(default_factory=list)
    max_seconds_without_detection: float = 0.0
    creation_reason: str = "unmatched_yolo_detection"
    termination_reason: str | None = None
    quality_flags: list[str] = field(default_factory=list)
    class_votes: dict[str, float] = field(default_factory=dict)
    recent_centers: list[tuple[float, float, float]] = field(default_factory=list)
    last_valid_kcf_timestamp_seconds: float | None = None
    lost_timestamp_seconds: float | None = None
    lost_reason: str | None = None
    reactivation_count: int = 0

    def append_trajectory(
        self,
        *,
        source_frame_index: int,
        processed_frame_index: int,
        timestamp_seconds: float,
        bbox_xyxy: list[float],
        bbox_source: str,
        limit: int,
    ) -> None:
        self.trajectory_history.append(
            {
                "source_frame_index": int(source_frame_index),
                "processed_frame_index": int(processed_frame_index),
                "timestamp_seconds": round(float(timestamp_seconds), 6),
                "bbox_xyxy": [round(float(value), 3) for value in bbox_xyxy],
                "bbox_source": str(bbox_source),
            }
        )
        if len(self.trajectory_history) > limit:
            self.trajectory_history = self.trajectory_history[-limit:]
        center_x = (float(bbox_xyxy[0]) + float(bbox_xyxy[2])) / 2.0
        center_y = (float(bbox_xyxy[1]) + float(bbox_xyxy[3])) / 2.0
        self.recent_centers.append((round(center_x, 6), round(center_y, 6), round(float(timestamp_seconds), 6)))
        if len(self.recent_centers) > limit:
            self.recent_centers = self.recent_centers[-limit:]

    def seconds_since_detection(self, current_timestamp_seconds: float) -> float:
        return max(0.0, float(current_timestamp_seconds) - float(self.last_detection_timestamp_seconds))

    def seconds_since_valid_update(self, current_timestamp_seconds: float) -> float:
        reference = self.last_valid_kcf_timestamp_seconds
        if reference is None:
            reference = self.last_update_timestamp_seconds
        return max(0.0, float(current_timestamp_seconds) - float(reference))

    def frames_since_detection(self, current_processed_frame_index: int) -> int:
        return max(0, int(current_processed_frame_index) - int(self.last_detection_frame_index))

    def update_class_vote(self, class_name: str, confidence: float) -> None:
        normalized = str(class_name).lower()
        self.class_votes[normalized] = float(self.class_votes.get(normalized, 0.0)) + float(confidence)
        if self.class_votes:
            self.class_name = max(
                sorted(self.class_votes),
                key=lambda item: (float(self.class_votes[item]), item),
            )

    def estimated_velocity(self) -> tuple[float, float]:
        if len(self.recent_centers) < 2:
            return 0.0, 0.0
        x1, y1, t1 = self.recent_centers[-2]
        x2, y2, t2 = self.recent_centers[-1]
        delta_t = max(float(t2) - float(t1), 1e-6)
        return (float(x2) - float(x1)) / delta_t, (float(y2) - float(y1)) / delta_t

    def predicted_center(self, current_timestamp_seconds: float) -> tuple[float, float]:
        if not self.recent_centers:
            return 0.0, 0.0
        last_x, last_y, last_t = self.recent_centers[-1]
        velocity_x, velocity_y = self.estimated_velocity()
        elapsed = max(0.0, float(current_timestamp_seconds) - float(last_t))
        return last_x + (velocity_x * elapsed), last_y + (velocity_y * elapsed)

    def to_summary_dict(self) -> dict[str, Any]:
        start_item = self.trajectory_history[0] if self.trajectory_history else {}
        end_item = self.trajectory_history[-1] if self.trajectory_history else {}
        return {
            "track_id": int(self.track_id),
            "class_name": str(self.class_name),
            "object_family": str(self.object_family),
            "start_timestamp_seconds": round(float(self.created_timestamp_seconds), 6),
            "end_timestamp_seconds": round(float(self.last_update_timestamp_seconds), 6),
            "duration_seconds": round(float(self.last_update_timestamp_seconds - self.created_timestamp_seconds), 6),
            "first_source_frame_index": int(start_item.get("source_frame_index", self.created_frame_index)),
            "last_source_frame_index": int(end_item.get("source_frame_index", self.last_update_frame_index)),
            "detection_hits": int(self.detection_hits),
            "propagation_hits": int(self.propagation_hits),
            "kcf_failures": int(self.consecutive_kcf_failures),
            "missed_detection_refreshes": int(self.missed_detection_refreshes),
            "maximum_seconds_without_detection": round(float(self.max_seconds_without_detection), 6),
            "creation_reason": str(self.creation_reason),
            "termination_reason": self.termination_reason,
            "trajectory": list(self.trajectory_history),
            "best_detection_confidence": None if self.last_detection_confidence is None else round(float(self.last_detection_confidence), 6),
            "quality_flags": list(self.quality_flags),
            "status": str(self.status),
            "is_confirmed": bool(self.is_confirmed),
            "reactivation_count": int(self.reactivation_count),
            "class_votes": {str(key): round(float(value), 6) for key, value in self.class_votes.items()},
            "recent_centers": [list(item) for item in self.recent_centers],
            "lost_reason": self.lost_reason,
        }


@dataclass
class EventRecord:
    timestamp_seconds: float
    source_frame_index: int
    event_type: str
    track_id: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_seconds": round(float(self.timestamp_seconds), 6),
            "source_frame_index": int(self.source_frame_index),
            "event_type": str(self.event_type),
            "track_id": self.track_id,
            "details": to_json_safe(dict(self.details)),
        }
