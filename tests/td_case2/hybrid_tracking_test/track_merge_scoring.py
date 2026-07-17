from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .track_quality import object_family_for_class


@dataclass(frozen=True)
class MergeScoringConfig:
    maximum_merge_gap_seconds: float = 2.0
    automatic_merge_score: float = 0.78
    possible_merge_score: float = 0.62
    excellent_spatial_match_ratio: float = 0.75
    possible_spatial_match_ratio: float = 1.50
    reject_spatial_match_ratio: float = 2.25
    minimum_area_ratio: float = 0.35
    maximum_area_ratio: float = 3.0
    direction_support_threshold: float = 0.25
    direction_reject_threshold: float = -0.25


def _bbox_area(bbox_xyxy: list[float]) -> float:
    return max(0.0, float(bbox_xyxy[2]) - float(bbox_xyxy[0])) * max(0.0, float(bbox_xyxy[3]) - float(bbox_xyxy[1]))


def _bbox_center(bbox_xyxy: list[float]) -> tuple[float, float]:
    return (
        (float(bbox_xyxy[0]) + float(bbox_xyxy[2])) / 2.0,
        (float(bbox_xyxy[1]) + float(bbox_xyxy[3])) / 2.0,
    )


def _trajectory_end(track: dict[str, Any]) -> dict[str, Any]:
    trajectory = list(track.get("sanitized_valid_timeline", track.get("trajectory", [])))
    return trajectory[-1] if trajectory else {"bbox_xyxy": [0.0, 0.0, 0.0, 0.0], "timestamp_seconds": float(track.get("sanitized_end_timestamp_seconds", track.get("end_timestamp_seconds", 0.0)))}


def _trajectory_start(track: dict[str, Any]) -> dict[str, Any]:
    trajectory = list(track.get("sanitized_valid_timeline", track.get("trajectory", [])))
    return trajectory[0] if trajectory else {"bbox_xyxy": [0.0, 0.0, 0.0, 0.0], "timestamp_seconds": float(track.get("sanitized_start_timestamp_seconds", track.get("start_timestamp_seconds", 0.0)))}


def _velocity(track: dict[str, Any]) -> tuple[float, float]:
    trajectory = list(track.get("sanitized_valid_timeline", track.get("trajectory", [])))
    if len(trajectory) < 2:
        return 0.0, 0.0
    left = trajectory[-2]
    right = trajectory[-1]
    left_center = _bbox_center(list(left["bbox_xyxy"]))
    right_center = _bbox_center(list(right["bbox_xyxy"]))
    delta_t = max(float(right["timestamp_seconds"]) - float(left["timestamp_seconds"]), 1e-6)
    return ((right_center[0] - left_center[0]) / delta_t, (right_center[1] - left_center[1]) / delta_t)


def _cosine_similarity(left_vector: tuple[float, float], right_vector: tuple[float, float]) -> float:
    left_norm = math.sqrt((left_vector[0] * left_vector[0]) + (left_vector[1] * left_vector[1]))
    right_norm = math.sqrt((right_vector[0] * right_vector[0]) + (right_vector[1] * right_vector[1]))
    if left_norm <= 1e-6 or right_norm <= 1e-6:
        return 0.0
    return ((left_vector[0] * right_vector[0]) + (left_vector[1] * right_vector[1])) / (left_norm * right_norm)


def compatible_track_families(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_family = object_family_for_class(str(left.get("final_class") or left.get("class_name") or ""))
    right_family = object_family_for_class(str(right.get("final_class") or right.get("class_name") or ""))
    if left_family != right_family:
        return False
    if left_family == "vehicle":
        return True
    return str(left.get("final_class") or left.get("class_name")) == str(right.get("final_class") or right.get("class_name"))


def compute_merge_candidate(left: dict[str, Any], right: dict[str, Any], *, config: MergeScoringConfig) -> dict[str, Any]:
    left_end = _trajectory_end(left)
    right_start = _trajectory_start(right)
    left_end_ts = float(left.get("sanitized_end_timestamp_seconds", left.get("end_timestamp_seconds", left_end["timestamp_seconds"])))
    right_start_ts = float(right.get("sanitized_start_timestamp_seconds", right.get("start_timestamp_seconds", right_start["timestamp_seconds"])))
    candidate_type = str(right.get("_candidate_type", left.get("_candidate_type", "sequential_fragment")))
    overlap_seconds = max(0.0, left_end_ts - right_start_ts)
    if overlap_seconds > 0.0 and candidate_type != "overlap_duplicate":
        return {
            "compatible": False,
            "decision": "reject",
            "reasons": ["temporal_overlap_conflict"],
            "time_gap_seconds": round(-overlap_seconds, 6),
        }
    time_gap = right_start_ts - left_end_ts
    if candidate_type == "sequential_fragment" and time_gap > config.maximum_merge_gap_seconds:
        return {
            "compatible": False,
            "decision": "reject",
            "reasons": ["time_gap_too_large"],
            "time_gap_seconds": round(time_gap, 6),
        }
    if not compatible_track_families(left, right):
        return {
            "compatible": False,
            "decision": "reject",
            "reasons": ["object_family_mismatch"],
            "time_gap_seconds": round(time_gap, 6),
        }
    left_end_bbox = list(left_end["bbox_xyxy"])
    right_start_bbox = list(right_start["bbox_xyxy"])
    left_end_center = _bbox_center(left_end_bbox)
    right_start_center = _bbox_center(right_start_bbox)
    velocity = _velocity(left)
    predicted_center = (left_end_center[0] + (velocity[0] * time_gap), left_end_center[1] + (velocity[1] * time_gap))
    diagonal = max(math.sqrt(max(_bbox_area(left_end_bbox), 1.0)), 1.0)
    normalized_predicted_center_distance = math.dist(predicted_center, right_start_center) / diagonal
    if normalized_predicted_center_distance > config.reject_spatial_match_ratio:
        return {
            "compatible": False,
            "decision": "reject",
            "reasons": ["predicted_distance_too_large"],
            "time_gap_seconds": round(time_gap, 6),
            "normalized_predicted_center_distance": round(normalized_predicted_center_distance, 6),
        }
    area_ratio = _bbox_area(right_start_bbox) / max(_bbox_area(left_end_bbox), 1.0)
    if area_ratio < config.minimum_area_ratio or area_ratio > config.maximum_area_ratio:
        return {
            "compatible": False,
            "decision": "reject",
            "reasons": ["area_ratio_out_of_range"],
            "time_gap_seconds": round(time_gap, 6),
            "normalized_predicted_center_distance": round(normalized_predicted_center_distance, 6),
            "area_ratio": round(area_ratio, 6),
        }
    left_width = max(1e-6, float(left_end_bbox[2]) - float(left_end_bbox[0]))
    left_height = max(1e-6, float(left_end_bbox[3]) - float(left_end_bbox[1]))
    right_width = max(1e-6, float(right_start_bbox[2]) - float(right_start_bbox[0]))
    right_height = max(1e-6, float(right_start_bbox[3]) - float(right_start_bbox[1]))
    aspect_ratio_change = (right_width / right_height) / max(left_width / left_height, 1e-6)
    displacement_vector = (right_start_center[0] - left_end_center[0], right_start_center[1] - left_end_center[1])
    direction_cosine_similarity = _cosine_similarity(velocity, displacement_vector)
    class_compatibility_score = 1.0 if str(left.get("final_class") or left.get("class_name")) == str(right.get("final_class") or right.get("class_name")) else 0.7
    temporal_score = max(0.0, 1.0 - (time_gap / max(config.maximum_merge_gap_seconds, 1e-6)))
    spatial_score = max(0.0, 1.0 - (normalized_predicted_center_distance / max(config.reject_spatial_match_ratio, 1e-6)))
    size_score = max(0.0, 1.0 - abs(1.0 - area_ratio))
    aspect_ratio_score = max(0.0, 1.0 - abs(1.0 - aspect_ratio_change))
    if direction_cosine_similarity >= config.direction_support_threshold:
        direction_score = 1.0
    elif direction_cosine_similarity <= config.direction_reject_threshold:
        direction_score = 0.0
    else:
        direction_score = 0.5
    left_boundary = str(left.get("exit_boundary") or "")
    right_boundary = str(right.get("entry_boundary") or "")
    if left_boundary and right_boundary and left_boundary != right_boundary:
        boundary_consistency_score = 0.25
    else:
        boundary_consistency_score = 1.0
    quality_score = (float(left.get("quality_score", 0.0)) + float(right.get("quality_score", 0.0))) / 2.0
    appearance_score = 0.5
    overlap_duplicate_score = 1.0 if candidate_type == "overlap_duplicate" and overlap_seconds <= 0.5 else 0.0
    track_integrity_score = 1.0 if str(left.get("track_integrity_status", "usable")) == "usable" and str(right.get("track_integrity_status", "usable")) == "usable" else 0.5
    merge_score = (
        0.20 * temporal_score
        + 0.30 * spatial_score
        + 0.10 * size_score
        + 0.05 * aspect_ratio_score
        + 0.15 * direction_score
        + 0.10 * appearance_score
        + 0.05 * class_compatibility_score
        + 0.05 * boundary_consistency_score
        + 0.05 * track_integrity_score
        + 0.05 * overlap_duplicate_score
    )
    reasons: list[str] = []
    if (
        merge_score >= config.automatic_merge_score
        and spatial_score >= 0.25
        and (direction_score >= 0.5 or appearance_score >= 0.5 or overlap_duplicate_score >= 1.0)
    ):
        decision = "auto_merge"
    elif (
        merge_score >= config.possible_merge_score
        and normalized_predicted_center_distance <= config.possible_spatial_match_ratio
        and boundary_consistency_score >= 0.25
    ):
        decision = "possible_merge"
    else:
        decision = "reject"
        reasons.append("merge_score_below_threshold")
    return {
        "compatible": decision != "reject",
        "decision": decision,
        "reasons": reasons,
        "time_gap_seconds": round(time_gap, 6),
        "candidate_type": candidate_type,
        "overlap_seconds": round(overlap_seconds, 6),
        "normalized_predicted_center_distance": round(normalized_predicted_center_distance, 6),
        "area_ratio": round(area_ratio, 6),
        "aspect_ratio_change": round(aspect_ratio_change, 6),
        "direction_cosine_similarity": round(direction_cosine_similarity, 6),
        "temporal_score": round(temporal_score, 6),
        "spatial_score": round(spatial_score, 6),
        "size_score": round(size_score, 6),
        "aspect_ratio_score": round(aspect_ratio_score, 6),
        "direction_score": round(direction_score, 6),
        "appearance_score": round(appearance_score, 6),
        "class_compatibility_score": round(class_compatibility_score, 6),
        "boundary_consistency_score": round(boundary_consistency_score, 6),
        "track_integrity_score": round(track_integrity_score, 6),
        "overlap_duplicate_score": round(overlap_duplicate_score, 6),
        "quality_score": round(quality_score, 6),
        "merge_score": round(merge_score, 6),
    }
