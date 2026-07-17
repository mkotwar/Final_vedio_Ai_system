from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from statistics import median
from typing import Any


ACTIVE = "active"
TENTATIVE = "tentative"
CONFIRMED = "confirmed"
RECOVERABLE = "recoverable"
LOST = "lost"
TERMINATED = "terminated"


def detector_should_run(*, processed_frame_index: int, processing_fps: float, detector_fps: float) -> bool:
    interval = max(1, int(round(processing_fps / detector_fps)))
    return processed_frame_index % interval == 0


def classify_zone(bbox_xyxy: list[float], *, frame_width: int, frame_height: int, margin_ratio: float = 0.10) -> str:
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    margin_x = frame_width * margin_ratio
    margin_y = frame_height * margin_ratio
    if center_x <= margin_x:
        return "left"
    if center_x >= frame_width - margin_x:
        return "right"
    if center_y <= margin_y:
        return "top"
    if center_y >= frame_height - margin_y:
        return "bottom"
    return "interior"


def class_compatible(*, left: str, right: str) -> bool:
    if left == right:
        return True
    return left in {"car", "motorcycle", "bus", "truck", "vehicle"} and right in {"car", "motorcycle", "bus", "truck", "vehicle"}


def compute_new_id_reason(
    *,
    bbox_xyxy: list[float],
    class_name: str,
    family: str,
    recoverable_tracks: list[dict[str, Any]],
    frame_width: int,
    frame_height: int,
) -> str:
    if not recoverable_tracks:
        zone = classify_zone(bbox_xyxy, frame_width=frame_width, frame_height=frame_height)
        return "new_entry_object" if zone != "interior" else "unmatched_detection"
    family_matches = [item for item in recoverable_tracks if str(item.get("family")) == family]
    if not family_matches:
        return "class_family_conflict"
    class_matches = [item for item in family_matches if class_compatible(left=str(item.get("class_name")), right=class_name)]
    if not class_matches:
        return "class_family_conflict"
    return "reactivation_failed"


@dataclass
class TrackLifecycleRecord:
    track_id: str
    family: str
    class_name: str
    created_timestamp_seconds: float
    created_zone: str
    detector_hit_timestamps: list[float] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    state_history: list[dict[str, Any]] = field(default_factory=list)
    last_state: str = TENTATIVE
    last_detector_confirmation_timestamp: float = 0.0
    confirmed: bool = False
    reactivated_count: int = 0
    termination_reason: str | None = None
    terminated: bool = False


def confirmation_threshold(*, family: str) -> tuple[int, float]:
    if family == "person":
        return 3, 1.0
    return 2, 0.8


def recovery_window_seconds(*, family: str, confirmed: bool) -> float:
    if not confirmed:
        return 0.3
    return 0.8 if family == "person" else 1.0


def update_confirmation(record: TrackLifecycleRecord, *, timestamp_seconds: float) -> None:
    threshold, window = confirmation_threshold(family=record.family)
    recent = [value for value in record.detector_hit_timestamps if (timestamp_seconds - value) <= window]
    record.detector_hit_timestamps = recent
    if len(recent) >= threshold:
        record.confirmed = True
        record.last_state = ACTIVE
        record.last_detector_confirmation_timestamp = timestamp_seconds
    else:
        record.last_state = TENTATIVE


def build_validation_metrics(*, records: dict[str, TrackLifecycleRecord], per_frame_events: list[dict[str, Any]], termination_events: list[dict[str, Any]], new_id_events: list[dict[str, Any]], reactivation_events: list[dict[str, Any]]) -> dict[str, Any]:
    durations = []
    confirmed_count = 0
    tentative_count = 0
    for record in records.values():
        if record.observations:
            durations.append(max(0.0, float(record.observations[-1]["timestamp_seconds"]) - float(record.observations[0]["timestamp_seconds"])))
        if record.confirmed:
            confirmed_count += 1
        else:
            tentative_count += 1
    zone_counter = Counter(str(item.get("zone", "unknown")) for item in new_id_events)
    termination_counter = Counter(str(item.get("reason", "unknown")) for item in termination_events)
    skipped_loss_metric = sum(1 for item in per_frame_events if not bool(item["detector_ran"]) and (item["lost_track_ids"] or item["terminated_track_ids"]))
    detector_frames = len([item for item in per_frame_events if bool(item["detector_ran"])])
    failed_reactivation_reasons = {"reactivation_failed", "class_family_conflict", "spatial_conflict"}
    failed_reactivations = len([item for item in new_id_events if str(item.get("reason")) in failed_reactivation_reasons])
    successful_reactivations = len([item for item in reactivation_events if item.get("success")])
    return {
        "raw_track_ids": len(records),
        "confirmed_tracks": confirmed_count,
        "tentative_tracks": tentative_count,
        "reactivated_tracks": len([item for item in records.values() if item.reactivated_count > 0]),
        "terminated_tracks": len([item for item in records.values() if item.terminated]),
        "new_ids_created": len(new_id_events),
        "new_ids_created_near_boundaries": int(sum(zone_counter.get(key, 0) for key in ("left", "right", "top", "bottom"))),
        "new_ids_created_interior": int(zone_counter.get("interior", 0)),
        "reactivation_attempts": successful_reactivations + failed_reactivations,
        "successful_reactivations": successful_reactivations,
        "failed_reactivations": failed_reactivations,
        "tracks_under_0_5_seconds": len([value for value in durations if value < 0.5]),
        "tracks_under_1_0_seconds": len([value for value in durations if value < 1.0]),
        "average_duration_seconds": round(sum(durations) / len(durations), 6) if durations else 0.0,
        "median_duration_seconds": round(median(durations), 6) if durations else 0.0,
        "maximum_duration_seconds": round(max(durations), 6) if durations else 0.0,
        "ids_created_per_detector_frame": round(len(new_id_events) / max(detector_frames, 1), 6),
        "confirmed_track_ratio": round(confirmed_count / max(len(records), 1), 6),
        "tentative_track_ratio": round(tentative_count / max(len(records), 1), 6),
        "reactivation_success_rate": round(successful_reactivations / max(successful_reactivations + failed_reactivations, 1), 6) if (successful_reactivations or failed_reactivations) else 0.0,
        "tracks_lost_due_to_skipped_detector_frame": skipped_loss_metric,
        "termination_reasons": dict(sorted(termination_counter.items())),
    }


def build_validation_checks(*, per_frame_events: list[dict[str, Any]], new_id_events: list[dict[str, Any]], reactivation_events: list[dict[str, Any]], records: dict[str, TrackLifecycleRecord]) -> dict[str, Any]:
    warnings: list[str] = []
    passes = True
    if any((not item["detector_ran"]) and item["lost_track_ids"] for item in per_frame_events):
        passes = False
        warnings.append("A skipped detector frame marked one or more tracks lost.")
    if any((not item["detector_ran"]) and item["terminated_track_ids"] for item in per_frame_events):
        passes = False
        warnings.append("A skipped detector frame terminated one or more tracks.")
    if any((not item["detector_ran"]) and item["new_track_ids"] for item in per_frame_events):
        passes = False
        warnings.append("A skipped detector frame created a new ID.")
    if any(not bool(item.get("detector_ran", False)) for item in new_id_events):
        passes = False
        warnings.append("A new ID event did not originate from a detector frame.")
    if any(not bool(item.get("detector_ran", False)) for item in reactivation_events):
        passes = False
        warnings.append("A reactivation event did not originate from a detector frame.")
    if any(len(record.observations) >= 2 and any(float(second["timestamp_seconds"]) < float(first["timestamp_seconds"]) for first, second in zip(record.observations, record.observations[1:])) for record in records.values()):
        passes = False
        warnings.append("At least one track has non-monotonic timestamps.")
    if len(set(records.keys())) != len(records):
        passes = False
        warnings.append("Track IDs were reused.")
    return {"passed": passes, "warnings": warnings}
