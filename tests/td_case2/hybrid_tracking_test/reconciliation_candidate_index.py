from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CandidateIndexConfig:
    maximum_merge_gap_seconds: float = 2.0
    maximum_overlap_duplicate_seconds: float = 0.5
    maximum_predicted_center_distance_ratio: float = 2.5
    minimum_area_ratio: float = 0.25
    maximum_area_ratio: float = 4.0


def _bbox_area(bbox_xyxy: list[float]) -> float:
    return max(0.0, float(bbox_xyxy[2]) - float(bbox_xyxy[0])) * max(0.0, float(bbox_xyxy[3]) - float(bbox_xyxy[1]))


def _bbox_center(bbox_xyxy: list[float]) -> tuple[float, float]:
    return (
        (float(bbox_xyxy[0]) + float(bbox_xyxy[2])) / 2.0,
        (float(bbox_xyxy[1]) + float(bbox_xyxy[3])) / 2.0,
    )


def _bbox_iou(left: list[float], right: list[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = _bbox_area(left) + _bbox_area(right) - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _trajectory_endpoint(track: dict[str, Any], mode: str) -> dict[str, Any] | None:
    trajectory = list(track.get("sanitized_valid_timeline", []))
    if not trajectory:
        return None
    return trajectory[0] if mode == "start" else trajectory[-1]


def _velocity(track: dict[str, Any]) -> tuple[float, float]:
    trajectory = list(track.get("sanitized_valid_timeline", []))
    if len(trajectory) < 2:
        return 0.0, 0.0
    left = trajectory[-2]
    right = trajectory[-1]
    left_center = _bbox_center(list(left["bbox_xyxy"]))
    right_center = _bbox_center(list(right["bbox_xyxy"]))
    delta_t = max(float(right["timestamp_seconds"]) - float(left["timestamp_seconds"]), 1e-6)
    return ((right_center[0] - left_center[0]) / delta_t, (right_center[1] - left_center[1]) / delta_t)


def generate_reconciliation_candidates(
    tracks: list[dict[str, Any]],
    *,
    config: CandidateIndexConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid_tracks = [
        item
        for item in tracks
        if str(item.get("track_integrity_status")) not in {"invalid", "fallback_only"}
        and list(item.get("sanitized_valid_timeline", []))
    ]
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for track in valid_tracks:
        buckets[str(track.get("object_family", "other"))].append(track)
    candidates: list[dict[str, Any]] = []
    for family, family_tracks in buckets.items():
        ordered = sorted(
            family_tracks,
            key=lambda item: (
                float(item.get("sanitized_start_timestamp_seconds", 0.0) or 0.0),
                float(item.get("sanitized_end_timestamp_seconds", 0.0) or 0.0),
                int(item.get("track_id", 0) or 0),
            ),
        )
        for index, left in enumerate(ordered):
            left_end = _trajectory_endpoint(left, "end")
            if left_end is None:
                continue
            left_velocity = _velocity(left)
            left_end_center = _bbox_center(list(left_end["bbox_xyxy"]))
            left_diag = max(math.sqrt(max(_bbox_area(list(left_end["bbox_xyxy"])), 1.0)), 1.0)
            for right in ordered[index + 1:]:
                if int(left.get("track_id", 0)) == int(right.get("track_id", 0)):
                    continue
                right_start = _trajectory_endpoint(right, "start")
                if right_start is None:
                    continue
                overlap_seconds = float(left.get("sanitized_end_timestamp_seconds", 0.0) or 0.0) - float(right.get("sanitized_start_timestamp_seconds", 0.0) or 0.0)
                candidate_type = "sequential_fragment"
                if overlap_seconds > config.maximum_overlap_duplicate_seconds:
                    continue
                if overlap_seconds > 0.0:
                    candidate_type = "overlap_duplicate"
                time_gap_seconds = round(float(right.get("sanitized_start_timestamp_seconds", 0.0) or 0.0) - float(left.get("sanitized_end_timestamp_seconds", 0.0) or 0.0), 6)
                if candidate_type == "sequential_fragment" and not (0.0 <= time_gap_seconds <= config.maximum_merge_gap_seconds):
                    continue
                right_start_center = _bbox_center(list(right_start["bbox_xyxy"]))
                predicted_center = (
                    left_end_center[0] + (left_velocity[0] * max(time_gap_seconds, 0.0)),
                    left_end_center[1] + (left_velocity[1] * max(time_gap_seconds, 0.0)),
                )
                predicted_distance_ratio = math.dist(predicted_center, right_start_center) / left_diag
                area_ratio = _bbox_area(list(right_start["bbox_xyxy"])) / max(_bbox_area(list(left_end["bbox_xyxy"])), 1.0)
                if predicted_distance_ratio > config.maximum_predicted_center_distance_ratio:
                    continue
                if area_ratio < config.minimum_area_ratio or area_ratio > config.maximum_area_ratio:
                    continue
                candidate = {
                    "from_track_id": int(left.get("track_id", 0)),
                    "to_track_id": int(right.get("track_id", 0)),
                    "candidate_type": candidate_type,
                    "time_gap_seconds": time_gap_seconds,
                    "predicted_center_distance_ratio": round(predicted_distance_ratio, 6),
                    "area_ratio": round(area_ratio, 6),
                    "object_family": family,
                    "from_integrity_status": str(left.get("track_integrity_status")),
                    "to_integrity_status": str(right.get("track_integrity_status")),
                }
                if candidate_type == "overlap_duplicate":
                    overlap_left = list(left.get("sanitized_valid_timeline", []))
                    overlap_right = list(right.get("sanitized_valid_timeline", []))
                    overlap_ious: list[float] = []
                    indexed_right = {
                        round(float(item.get("timestamp_seconds", 0.0) or 0.0), 6): item
                        for item in overlap_right
                    }
                    for left_item in overlap_left:
                        key = round(float(left_item.get("timestamp_seconds", 0.0) or 0.0), 6)
                        right_item = indexed_right.get(key)
                        if right_item is None:
                            continue
                        overlap_ious.append(_bbox_iou(list(left_item["bbox_xyxy"]), list(right_item["bbox_xyxy"])))
                    if not overlap_ious or (sum(overlap_ious) / len(overlap_ious)) < 0.60:
                        continue
                    candidate["mean_box_iou_during_overlap"] = round(sum(overlap_ious) / len(overlap_ious), 6)
                candidates.append(candidate)
    report = {
        "status": "success",
        "candidate_count": len(candidates),
        "candidate_types": dict(
            sorted(
                {
                    "sequential_fragment": len([item for item in candidates if item["candidate_type"] == "sequential_fragment"]),
                    "overlap_duplicate": len([item for item in candidates if item["candidate_type"] == "overlap_duplicate"]),
                }.items()
            )
        ),
        "eligible_track_count": len(valid_tracks),
    }
    return candidates, report


__all__ = ["CandidateIndexConfig", "generate_reconciliation_candidates"]
