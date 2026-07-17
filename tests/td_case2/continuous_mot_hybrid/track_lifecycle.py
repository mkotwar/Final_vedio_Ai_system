from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any


LIFECYCLE_TENTATIVE = "tentative"
LIFECYCLE_CONFIRMED = "confirmed"
LIFECYCLE_ACTIVELY_TRACKED = "actively_tracked"
LIFECYCLE_VISUAL_BRIDGE = "visual_bridge"
LIFECYCLE_LOST = "lost"
LIFECYCLE_RECOVERABLE = "recoverable"
LIFECYCLE_TERMINATED = "terminated"
LIFECYCLE_WEAK_SINGLE_DETECTION = "weak_single_detection"


@dataclass(frozen=True)
class LifecycleConfig:
    min_person_confirm_hits: int
    min_vehicle_confirm_hits: int
    lost_recovery_seconds: float


def confirmation_threshold(*, object_family: str, config: LifecycleConfig) -> int:
    return config.min_person_confirm_hits if object_family == "person" else config.min_vehicle_confirm_hits


def infer_lifecycle_state(
    *,
    object_family: str,
    detector_hits: int,
    duration_seconds: float,
    bbox_source: str,
    backend_state: str,
    time_since_update_seconds: float,
    config: LifecycleConfig,
) -> str:
    threshold = confirmation_threshold(object_family=object_family, config=config)
    if bbox_source.startswith("visual_bridge"):
        return LIFECYCLE_VISUAL_BRIDGE
    if detector_hits >= threshold:
        if backend_state == "lost":
            return LIFECYCLE_RECOVERABLE if time_since_update_seconds <= config.lost_recovery_seconds else LIFECYCLE_TERMINATED
        return LIFECYCLE_ACTIVELY_TRACKED if duration_seconds > 0 else LIFECYCLE_CONFIRMED
    if detector_hits == 1 and duration_seconds <= 0.7:
        return LIFECYCLE_WEAK_SINGLE_DETECTION
    if backend_state == "lost":
        return LIFECYCLE_TERMINATED
    return LIFECYCLE_TENTATIVE


def summarize_track_histories(observations: list[dict[str, Any]], *, config: LifecycleConfig) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        grouped[str(observation["track_id"])].append(observation)
    track_rows: list[dict[str, Any]] = []
    for track_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda item: (float(item["timestamp_seconds"]), int(item["processed_frame_index"])))
        detector_hits = len([item for item in ordered if str(item["bbox_source"]) == "yolo"])
        start = float(ordered[0]["timestamp_seconds"])
        end = float(ordered[-1]["timestamp_seconds"])
        duration_seconds = max(0.0, end - start)
        object_family = str(ordered[0]["object_family"])
        lifecycle_state = infer_lifecycle_state(
            object_family=object_family,
            detector_hits=detector_hits,
            duration_seconds=duration_seconds,
            bbox_source=str(ordered[-1]["bbox_source"]),
            backend_state=str(ordered[-1]["track_backend_state"]),
            time_since_update_seconds=float(ordered[-1]["time_since_update_seconds"]),
            config=config,
        )
        track_rows.append(
            {
                "track_id": track_id,
                "object_family": object_family,
                "class_name": str(ordered[-1]["class_name"]),
                "start_timestamp_seconds": round(start, 6),
                "end_timestamp_seconds": round(end, 6),
                "duration_seconds": round(duration_seconds, 6),
                "first_source_frame_index": int(ordered[0]["source_frame_index"]),
                "last_source_frame_index": int(ordered[-1]["source_frame_index"]),
                "detector_hits": detector_hits,
                "observation_count": len(ordered),
                "confirmed": detector_hits >= confirmation_threshold(object_family=object_family, config=config),
                "lifecycle_state": lifecycle_state,
                "trajectory": ordered,
            }
        )
    return sorted(track_rows, key=lambda item: (item["start_timestamp_seconds"], item["track_id"]))

