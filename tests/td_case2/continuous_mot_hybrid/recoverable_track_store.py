from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RecoverableTrackSnapshot:
    local_object_id: int
    last_tracker_id: str
    tracker_id_history: list[str]
    object_family: str
    stable_class: str
    class_votes: dict[str, int]
    last_detector_supported_bbox: list[float]
    previous_detector_supported_bbox: list[float] | None
    last_center: tuple[float, float]
    estimated_velocity: tuple[float, float]
    last_timestamp_seconds: float
    last_detector_timestamp_seconds: float
    track_duration_seconds: float
    detector_hit_count: int
    entry_zone: str
    likely_exit_zone: str
    movement_direction: str
    bbox_width: float
    bbox_height: float
    bbox_area: float
    aspect_ratio: float
    detector_confidence: float
    recovery_expiry_timestamp: float
    recovery_reason: str
    histogram_descriptor: list[float] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "local_object_id": self.local_object_id,
            "last_tracker_id": self.last_tracker_id,
            "tracker_id_history": list(self.tracker_id_history),
            "object_family": self.object_family,
            "stable_class": self.stable_class,
            "class_votes": dict(self.class_votes),
            "last_detector_supported_bbox": list(self.last_detector_supported_bbox),
            "previous_detector_supported_bbox": None if self.previous_detector_supported_bbox is None else list(self.previous_detector_supported_bbox),
            "last_center": [round(float(value), 6) for value in self.last_center],
            "estimated_velocity": [round(float(value), 6) for value in self.estimated_velocity],
            "last_timestamp_seconds": round(self.last_timestamp_seconds, 6),
            "last_detector_timestamp_seconds": round(self.last_detector_timestamp_seconds, 6),
            "track_duration_seconds": round(self.track_duration_seconds, 6),
            "detector_hit_count": self.detector_hit_count,
            "entry_zone": self.entry_zone,
            "likely_exit_zone": self.likely_exit_zone,
            "movement_direction": self.movement_direction,
            "bbox_width": round(self.bbox_width, 6),
            "bbox_height": round(self.bbox_height, 6),
            "bbox_area": round(self.bbox_area, 6),
            "aspect_ratio": round(self.aspect_ratio, 6),
            "detector_confidence": round(self.detector_confidence, 6),
            "recovery_expiry_timestamp": round(self.recovery_expiry_timestamp, 6),
            "recovery_reason": self.recovery_reason,
            "has_histogram_descriptor": self.histogram_descriptor is not None,
        }


class RecoverableTrackStore:
    def __init__(self) -> None:
        self.entries_by_local_object_id: dict[int, RecoverableTrackSnapshot] = {}
        self.entries_created = 0
        self.expired_entries = 0

    def add(self, snapshot: RecoverableTrackSnapshot) -> None:
        self.entries_by_local_object_id[snapshot.local_object_id] = snapshot
        self.entries_created += 1

    def remove(self, local_object_id: int) -> RecoverableTrackSnapshot | None:
        return self.entries_by_local_object_id.pop(local_object_id, None)

    def get(self, local_object_id: int) -> RecoverableTrackSnapshot | None:
        return self.entries_by_local_object_id.get(local_object_id)

    def active_entries(self, *, timestamp_seconds: float) -> list[RecoverableTrackSnapshot]:
        return [
            entry
            for entry in self.entries_by_local_object_id.values()
            if float(entry.recovery_expiry_timestamp) >= float(timestamp_seconds)
        ]

    def expire(self, *, timestamp_seconds: float) -> list[RecoverableTrackSnapshot]:
        expired = [
            entry
            for entry in self.entries_by_local_object_id.values()
            if float(entry.recovery_expiry_timestamp) < float(timestamp_seconds)
        ]
        for entry in expired:
            self.entries_by_local_object_id.pop(entry.local_object_id, None)
        self.expired_entries += len(expired)
        return expired

    def build_snapshot_payload(self) -> dict[str, Any]:
        return {
            "status": "success",
            "entries_created": self.entries_created,
            "expired_entries": self.expired_entries,
            "active_entries": [entry.to_dict() for entry in self.entries_by_local_object_id.values()],
        }

