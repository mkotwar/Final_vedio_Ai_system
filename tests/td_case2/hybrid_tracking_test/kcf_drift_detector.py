from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median, pstdev
from typing import Any


@dataclass(frozen=True)
class DriftDetectionConfig:
    frozen_window_seconds: float = 0.8
    frozen_min_observations: int = 5
    frozen_max_center_motion_diagonals: float = 0.05
    frozen_max_coordinate_std_ratio: float = 0.02
    frozen_max_area_change_ratio: float = 0.05
    maximum_kcf_only_ready_gap_seconds: float = 0.6
    boundary_stuck_ratio_threshold: float = 0.75
    repeated_box_iou_threshold: float = 0.98
    boundary_tolerance_ratio: float = 0.01
    detector_kcf_disagreement_iou: float = 0.10
    detector_kcf_disagreement_center_distance: float = 1.5
    detector_kcf_disagreement_area_ratio_min: float = 0.35
    detector_kcf_disagreement_area_ratio_max: float = 3.0


def _bbox_area(bbox_xyxy: list[float]) -> float:
    return max(0.0, float(bbox_xyxy[2]) - float(bbox_xyxy[0])) * max(0.0, float(bbox_xyxy[3]) - float(bbox_xyxy[1]))


def _bbox_center(bbox_xyxy: list[float]) -> tuple[float, float]:
    return (
        (float(bbox_xyxy[0]) + float(bbox_xyxy[2])) / 2.0,
        (float(bbox_xyxy[1]) + float(bbox_xyxy[3])) / 2.0,
    )


def _bbox_diagonal(bbox_xyxy: list[float]) -> float:
    return math.sqrt(max(_bbox_area(bbox_xyxy), 1.0))


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


def _touching_boundaries(bbox_xyxy: list[float], frame_width: int, frame_height: int, tolerance_ratio: float) -> dict[str, bool]:
    tolerance_x = float(frame_width) * tolerance_ratio
    tolerance_y = float(frame_height) * tolerance_ratio
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    return {
        "left": x1 <= tolerance_x,
        "right": x2 >= float(frame_width) - tolerance_x,
        "top": y1 <= tolerance_y,
        "bottom": y2 >= float(frame_height) - tolerance_y,
    }


def detect_kcf_drift_segments(
    track: dict[str, Any],
    *,
    frame_width: int,
    frame_height: int,
    config: DriftDetectionConfig,
) -> dict[str, Any]:
    timeline = list(track.get("rebuilt_timeline", track.get("trajectory", [])))
    if not timeline:
        return {"track_id": int(track.get("track_id", 0)), "segments": [], "flags": []}
    segments: list[dict[str, Any]] = []
    active_kcf: list[dict[str, Any]] = []
    last_yolo_observation: dict[str, Any] | None = None
    recent_yolo_supported = False
    for observation in timeline:
        if str(observation.get("bbox_source")) == "yolo":
            if active_kcf:
                segments.append(_build_kcf_segment(active_kcf, last_yolo_observation, observation, frame_width, frame_height, config, recent_yolo_supported))
                active_kcf = []
            last_yolo_observation = observation
            recent_yolo_supported = True
            continue
        active_kcf.append(observation)
    if active_kcf:
        segments.append(_build_kcf_segment(active_kcf, last_yolo_observation, None, frame_width, frame_height, config, recent_yolo_supported))
    flags = sorted({flag for segment in segments for flag in list(segment.get("segment_flags", []))})
    return {
        "track_id": int(track.get("track_id", 0)),
        "segments": segments,
        "flags": flags,
    }


def _build_kcf_segment(
    observations: list[dict[str, Any]],
    last_yolo: dict[str, Any] | None,
    next_yolo: dict[str, Any] | None,
    frame_width: int,
    frame_height: int,
    config: DriftDetectionConfig,
    recent_yolo_supported: bool,
) -> dict[str, Any]:
    centers = [_bbox_center(list(item["bbox_xyxy"])) for item in observations]
    areas = [_bbox_area(list(item["bbox_xyxy"])) for item in observations]
    diagonals = [_bbox_diagonal(list(item["bbox_xyxy"])) for item in observations]
    median_diagonal = max(float(median(diagonals)) if diagonals else 1.0, 1.0)
    elapsed_time = max(0.0, float(observations[-1]["timestamp_seconds"]) - float(observations[0]["timestamp_seconds"]))
    center_displacement = math.dist(centers[0], centers[-1]) if len(centers) >= 2 else 0.0
    normalized_center_displacement = center_displacement / median_diagonal
    coordinate_std_ratio = 0.0
    if len(centers) >= 2:
        coordinate_std_ratio = max(pstdev([item[0] for item in centers]), pstdev([item[1] for item in centers])) / median_diagonal
    area_change_ratio = 0.0
    if areas:
        area_change_ratio = abs(max(areas) - min(areas)) / max(float(median(areas)) if areas else 1.0, 1.0)
    pairwise_ious = [
        _bbox_iou(list(left["bbox_xyxy"]), list(right["bbox_xyxy"]))
        for left, right in zip(observations, observations[1:])
    ]
    repeated_identical_ratio = float(sum(1 for value in pairwise_ious if value >= config.repeated_box_iou_threshold)) / max(len(pairwise_ious), 1)
    boundary_counts = {"left": 0, "right": 0, "top": 0, "bottom": 0}
    for observation in observations:
        for boundary_name, touched in _touching_boundaries(list(observation["bbox_xyxy"]), frame_width, frame_height, config.boundary_tolerance_ratio).items():
            if touched:
                boundary_counts[boundary_name] += 1
    dominant_boundary = max(boundary_counts, key=boundary_counts.get)
    dominant_boundary_ratio = float(boundary_counts[dominant_boundary]) / max(len(observations), 1)
    time_since_last_yolo = None
    if last_yolo is not None:
        time_since_last_yolo = round(float(observations[-1]["timestamp_seconds"]) - float(last_yolo["timestamp_seconds"]), 6)
    detector_disagreement = None
    detector_disagreement_flags: list[str] = []
    if next_yolo is not None and observations:
        kcf_tail = observations[-1]
        correction_iou = _bbox_iou(list(kcf_tail["bbox_xyxy"]), list(next_yolo["bbox_xyxy"]))
        kcf_center = _bbox_center(list(kcf_tail["bbox_xyxy"]))
        yolo_center = _bbox_center(list(next_yolo["bbox_xyxy"]))
        correction_distance = math.dist(kcf_center, yolo_center) / max(_bbox_diagonal(list(kcf_tail["bbox_xyxy"])), 1.0)
        correction_area_ratio = _bbox_area(list(next_yolo["bbox_xyxy"])) / max(_bbox_area(list(kcf_tail["bbox_xyxy"])), 1.0)
        detector_disagreement = {
            "correction_iou": round(correction_iou, 6),
            "correction_center_distance_normalized": round(correction_distance, 6),
            "correction_area_ratio": round(correction_area_ratio, 6),
        }
        if (
            correction_iou < config.detector_kcf_disagreement_iou
            or correction_distance > config.detector_kcf_disagreement_center_distance
            or correction_area_ratio < config.detector_kcf_disagreement_area_ratio_min
            or correction_area_ratio > config.detector_kcf_disagreement_area_ratio_max
        ):
            detector_disagreement_flags.append("detector_kcf_disagreement")
    segment_flags: list[str] = []
    if elapsed_time > config.maximum_kcf_only_ready_gap_seconds:
        segment_flags.append("long_kcf_only_segment")
    is_boundary_stuck = (
        elapsed_time >= config.frozen_window_seconds
        and dominant_boundary_ratio >= config.boundary_stuck_ratio_threshold
        and normalized_center_displacement <= config.frozen_max_center_motion_diagonals
    )
    if is_boundary_stuck:
        segment_flags.append("boundary_stuck_box")
    is_frozen = (
        len(observations) >= config.frozen_min_observations
        and elapsed_time >= config.frozen_window_seconds
        and normalized_center_displacement <= config.frozen_max_center_motion_diagonals
        and coordinate_std_ratio <= config.frozen_max_coordinate_std_ratio
        and area_change_ratio <= config.frozen_max_area_change_ratio
        and repeated_identical_ratio >= 0.8
        and (
            is_boundary_stuck
            or (time_since_last_yolo is not None and time_since_last_yolo > config.maximum_kcf_only_ready_gap_seconds)
            or not recent_yolo_supported
        )
    )
    if is_frozen:
        segment_flags.append("frozen_kcf_box")
    if repeated_identical_ratio >= 0.8 and len(observations) >= 3:
        segment_flags.append("repeated_identical_box")
    segment_flags.extend(detector_disagreement_flags)
    if time_since_last_yolo is not None and time_since_last_yolo > config.maximum_kcf_only_ready_gap_seconds:
        segment_flags.append("stale_kcf_propagation")
    return {
        "start_timestamp_seconds": round(float(observations[0]["timestamp_seconds"]), 6),
        "end_timestamp_seconds": round(float(observations[-1]["timestamp_seconds"]), 6),
        "duration_seconds": round(elapsed_time, 6),
        "start_source_frame_index": int(observations[0]["source_frame_index"]),
        "end_source_frame_index": int(observations[-1]["source_frame_index"]),
        "observation_count": len(observations),
        "segment_type": "kcf_only",
        "segment_flags": sorted(set(segment_flags)),
        "normalized_center_displacement": round(normalized_center_displacement, 6),
        "coordinate_std_ratio": round(coordinate_std_ratio, 6),
        "area_change_ratio": round(area_change_ratio, 6),
        "dominant_boundary": dominant_boundary,
        "dominant_boundary_ratio": round(dominant_boundary_ratio, 6),
        "repeated_identical_ratio": round(repeated_identical_ratio, 6),
        "time_since_last_yolo": time_since_last_yolo,
        "detector_disagreement": detector_disagreement,
        "observation_indexes": [int(item["source_frame_index"]) for item in observations],
    }


__all__ = ["DriftDetectionConfig", "detect_kcf_drift_segments"]
