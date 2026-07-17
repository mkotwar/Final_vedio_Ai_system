from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrackObservation:
    track_id: str
    object_family: str
    class_name: str
    timestamp_seconds: float
    source_frame_index: int
    processed_frame_index: int
    bbox_xyxy: list[float]
    bbox_source: str
    detector_confidence: float | None
    detector_detection_id: str | None
    lifecycle_state: str
    confirmed: bool
    age_frames: int
    hits: int
    time_since_update_seconds: float
    track_backend_state: str
    association_cost: float | None
    observation_validity: str = "valid"
    integrity_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "object_family": self.object_family,
            "class_name": self.class_name,
            "timestamp_seconds": round(self.timestamp_seconds, 6),
            "source_frame_index": self.source_frame_index,
            "processed_frame_index": self.processed_frame_index,
            "bbox_xyxy": [round(float(value), 3) for value in self.bbox_xyxy],
            "bbox_source": self.bbox_source,
            "detector_confidence": None if self.detector_confidence is None else round(float(self.detector_confidence), 6),
            "detector_detection_id": self.detector_detection_id,
            "lifecycle_state": self.lifecycle_state,
            "confirmed": self.confirmed,
            "age_frames": self.age_frames,
            "hits": self.hits,
            "time_since_update_seconds": round(self.time_since_update_seconds, 6),
            "track_backend_state": self.track_backend_state,
            "association_cost": None if self.association_cost is None else round(float(self.association_cost), 6),
            "observation_validity": self.observation_validity,
            "integrity_flags": list(self.integrity_flags),
        }


def object_family_for_class(class_name: str) -> str:
    normalized = class_name.lower()
    if normalized == "person":
        return "person"
    if normalized in {"car", "motorcycle", "bus", "truck", "vehicle"}:
        return "vehicle"
    return "other"

