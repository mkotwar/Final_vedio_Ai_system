from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, median
from typing import Any

from .recoverable_track_store import RecoverableTrackSnapshot


def _bbox_center(bbox_xyxy: list[float]) -> tuple[float, float]:
    return ((float(bbox_xyxy[0]) + float(bbox_xyxy[2])) / 2.0, (float(bbox_xyxy[1]) + float(bbox_xyxy[3])) / 2.0)


def _bbox_area(bbox_xyxy: list[float]) -> float:
    return max(0.0, float(bbox_xyxy[2]) - float(bbox_xyxy[0])) * max(0.0, float(bbox_xyxy[3]) - float(bbox_xyxy[1]))


def _aspect_ratio(bbox_xyxy: list[float]) -> float:
    width = max(1e-6, float(bbox_xyxy[2]) - float(bbox_xyxy[0]))
    height = max(1e-6, float(bbox_xyxy[3]) - float(bbox_xyxy[1]))
    return width / height


def _cosine_similarity(left: tuple[float, float], right: tuple[float, float]) -> float:
    left_norm = math.sqrt(left[0] * left[0] + left[1] * left[1])
    right_norm = math.sqrt(right[0] * right[0] + right[1] * right[1])
    if left_norm <= 1e-6 or right_norm <= 1e-6:
        return 0.0
    return ((left[0] * right[0]) + (left[1] * right[1])) / (left_norm * right_norm)


@dataclass(frozen=True)
class RecoveryScoringConfig:
    spatial_continuity_weight: float = 0.30
    time_gap_weight: float = 0.15
    scale_similarity_weight: float = 0.12
    aspect_ratio_similarity_weight: float = 0.08
    direction_compatibility_weight: float = 0.10
    class_compatibility_weight: float = 0.08
    zone_compatibility_weight: float = 0.07
    track_maturity_weight: float = 0.05
    appearance_histogram_weight: float = 0.05
    auto_reactivate_score: float = 0.78
    possible_reactivate_score: float = 0.68
    minimum_score_margin: float = 0.10


def _dynamic_distance_threshold(*, entry: RecoverableTrackSnapshot, time_gap_seconds: float) -> float:
    size_term = max(math.sqrt(max(entry.bbox_area, 1.0)), 1.0)
    speed_term = math.sqrt(entry.estimated_velocity[0] ** 2 + entry.estimated_velocity[1] ** 2) * max(time_gap_seconds, 0.0)
    return max(0.9, (0.6 * size_term + speed_term) / size_term)


def _histogram_similarity(left: list[float] | None, right: list[float] | None) -> float:
    if left is None or right is None or len(left) != len(right):
        return 0.5
    numerator = sum(float(a) * float(b) for a, b in zip(left, right))
    denominator_left = math.sqrt(sum(float(a) * float(a) for a in left))
    denominator_right = math.sqrt(sum(float(b) * float(b) for b in right))
    if denominator_left <= 1e-6 or denominator_right <= 1e-6:
        return 0.5
    return max(0.0, min(1.0, numerator / (denominator_left * denominator_right)))


def score_recovery_candidate(
    *,
    unmatched_detection: dict[str, Any],
    entry: RecoverableTrackSnapshot,
    timestamp_seconds: float,
    detection_histogram: list[float] | None,
    config: RecoveryScoringConfig,
) -> dict[str, Any]:
    hard_rejection_reasons: list[str] = []
    detection_bbox = list(unmatched_detection["bbox_xyxy"])
    detection_center = _bbox_center(detection_bbox)
    predicted_center = (
        min(max(0.0, entry.last_center[0] + entry.estimated_velocity[0] * max(timestamp_seconds - entry.last_detector_timestamp_seconds, 0.0)), unmatched_detection["frame_width"]),
        min(max(0.0, entry.last_center[1] + entry.estimated_velocity[1] * max(timestamp_seconds - entry.last_detector_timestamp_seconds, 0.0)), unmatched_detection["frame_height"]),
    )
    time_gap_seconds = max(0.0, float(timestamp_seconds) - float(entry.last_detector_timestamp_seconds))
    if entry.object_family != str(unmatched_detection["family"]):
        hard_rejection_reasons.append("family_mismatch")
    if float(entry.recovery_expiry_timestamp) < float(timestamp_seconds):
        hard_rejection_reasons.append("recovery_window_expired")
    if time_gap_seconds <= 0.0:
        hard_rejection_reasons.append("non_positive_time_gap")

    detection_area = max(_bbox_area(detection_bbox), 1.0)
    area_ratio = detection_area / max(entry.bbox_area, 1.0)
    if entry.object_family == "person":
        if area_ratio < 0.45 or area_ratio > 2.20:
            hard_rejection_reasons.append("person_area_ratio_conflict")
    else:
        if area_ratio < 0.35 or area_ratio > 2.85:
            hard_rejection_reasons.append("vehicle_area_ratio_conflict")
    aspect_ratio_change = _aspect_ratio(detection_bbox) / max(entry.aspect_ratio, 1e-6)
    if aspect_ratio_change < 0.5 or aspect_ratio_change > 2.0:
        hard_rejection_reasons.append("aspect_ratio_conflict")
    predicted_distance = math.dist(predicted_center, detection_center)
    normalized_center_distance = predicted_distance / max(math.sqrt(max(entry.bbox_area, 1.0)), 1.0)
    dynamic_threshold = _dynamic_distance_threshold(entry=entry, time_gap_seconds=time_gap_seconds)
    if normalized_center_distance > dynamic_threshold:
        hard_rejection_reasons.append("spatial_conflict")
    if not (str(unmatched_detection["class_name"]) == entry.stable_class or (entry.object_family == "vehicle" and str(unmatched_detection["family"]) == "vehicle")):
        pass
    if entry.likely_exit_zone != "interior" and str(unmatched_detection["zone"]) == "interior" and normalized_center_distance > 0.6:
        hard_rejection_reasons.append("boundary_logic_conflict")

    displacement_vector = (
        detection_center[0] - entry.last_center[0],
        detection_center[1] - entry.last_center[1],
    )
    direction_cosine = _cosine_similarity(entry.estimated_velocity, displacement_vector)
    direction_score = 1.0 if direction_cosine >= 0.25 else (0.5 if direction_cosine >= -0.1 else 0.0)
    class_score = 1.0 if str(unmatched_detection["class_name"]) == entry.stable_class else 0.7
    zone_score = 1.0 if str(unmatched_detection["zone"]) == entry.likely_exit_zone or str(unmatched_detection["zone"]) == entry.entry_zone else 0.6
    spatial_score = max(0.0, 1.0 - (normalized_center_distance / max(dynamic_threshold, 1e-6)))
    time_gap_score = max(0.0, 1.0 - (time_gap_seconds / max(entry.recovery_expiry_timestamp - entry.last_detector_timestamp_seconds, 1e-6)))
    scale_score = max(0.0, 1.0 - abs(1.0 - area_ratio))
    aspect_score = max(0.0, 1.0 - abs(1.0 - aspect_ratio_change))
    track_maturity_score = max(0.2, min(1.0, float(entry.detector_hit_count) / 6.0))
    appearance_score = _histogram_similarity(entry.histogram_descriptor, detection_histogram)
    total_score = (
        config.spatial_continuity_weight * spatial_score
        + config.time_gap_weight * time_gap_score
        + config.scale_similarity_weight * scale_score
        + config.aspect_ratio_similarity_weight * aspect_score
        + config.direction_compatibility_weight * direction_score
        + config.class_compatibility_weight * class_score
        + config.zone_compatibility_weight * zone_score
        + config.track_maturity_weight * track_maturity_score
        + config.appearance_histogram_weight * appearance_score
    )
    return {
        "timestamp_seconds": round(timestamp_seconds, 6),
        "new_tracker_id": unmatched_detection["tracker_id"],
        "proposed_local_object_id": entry.local_object_id,
        "previous_tracker_id": entry.last_tracker_id,
        "time_gap_seconds": round(time_gap_seconds, 6),
        "predicted_center": [round(float(value), 6) for value in predicted_center],
        "detected_center": [round(float(value), 6) for value in detection_center],
        "normalized_center_distance": round(normalized_center_distance, 6),
        "dynamic_center_distance_threshold": round(dynamic_threshold, 6),
        "area_ratio": round(area_ratio, 6),
        "aspect_ratio_change": round(aspect_ratio_change, 6),
        "class_compatibility": round(class_score, 6),
        "direction_compatibility": round(direction_score, 6),
        "zone_compatibility": round(zone_score, 6),
        "histogram_similarity": round(appearance_score, 6),
        "component_scores": {
            "spatial_continuity": round(spatial_score, 6),
            "time_gap": round(time_gap_score, 6),
            "scale_similarity": round(scale_score, 6),
            "aspect_ratio_similarity": round(aspect_score, 6),
            "direction_compatibility": round(direction_score, 6),
            "class_compatibility": round(class_score, 6),
            "zone_compatibility": round(zone_score, 6),
            "track_maturity": round(track_maturity_score, 6),
            "appearance_histogram": round(appearance_score, 6),
        },
        "total_score": round(total_score, 6),
        "hard_rejection_reasons": hard_rejection_reasons,
    }


def summarize_scores(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(item["total_score"]) for item in attempts]
    accepted = [float(item["total_score"]) for item in attempts if item.get("final_decision") == "accepted"]
    rejected = [float(item["total_score"]) for item in attempts if item.get("final_decision") == "rejected"]
    margins = [float(item.get("score_margin", 0.0)) for item in attempts]
    return {
        "status": "success",
        "recovery_score_mean": round(mean(scores), 6) if scores else 0.0,
        "recovery_score_median": round(median(scores), 6) if scores else 0.0,
        "accepted_score_mean": round(mean(accepted), 6) if accepted else 0.0,
        "rejected_score_mean": round(mean(rejected), 6) if rejected else 0.0,
        "average_score_margin": round(mean(margins), 6) if margins else 0.0,
    }

