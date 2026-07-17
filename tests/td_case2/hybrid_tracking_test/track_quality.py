from __future__ import annotations

import math
from collections import Counter
from statistics import mean
from typing import Any


VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "vehicle", "van", "auto", "bicycle"}


def object_family_for_class(class_name: str) -> str:
    lowered = str(class_name).lower()
    if lowered == "person":
        return "person"
    if lowered in VEHICLE_CLASSES:
        return "vehicle"
    return "other"


def _bbox_area(bbox_xyxy: list[float]) -> float:
    return max(0.0, float(bbox_xyxy[2]) - float(bbox_xyxy[0])) * max(0.0, float(bbox_xyxy[3]) - float(bbox_xyxy[1]))


def _bbox_center(bbox_xyxy: list[float]) -> tuple[float, float]:
    return (
        (float(bbox_xyxy[0]) + float(bbox_xyxy[2])) / 2.0,
        (float(bbox_xyxy[1]) + float(bbox_xyxy[3])) / 2.0,
    )


def _visible_area_ratio(bbox_xyxy: list[float], frame_width: int, frame_height: int) -> float:
    return _bbox_area(bbox_xyxy) / max(1.0, float(frame_width * frame_height))


def _boundary_info(bbox_xyxy: list[float], frame_width: int, frame_height: int) -> dict[str, Any]:
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    margin_left = max(0.0, x1) / max(float(frame_width), 1.0)
    margin_top = max(0.0, y1) / max(float(frame_height), 1.0)
    margin_right = max(0.0, float(frame_width) - x2) / max(float(frame_width), 1.0)
    margin_bottom = max(0.0, float(frame_height) - y2) / max(float(frame_height), 1.0)
    smallest_margin = min(margin_left, margin_top, margin_right, margin_bottom)
    boundary_name = None
    if smallest_margin <= 0.03:
        boundary_map = {
            margin_left: "left",
            margin_top: "top",
            margin_right: "right",
            margin_bottom: "bottom",
        }
        boundary_name = boundary_map[smallest_margin]
    return {
        "touches_boundary": smallest_margin <= 0.03,
        "boundary_name": boundary_name,
        "boundary_distance_ratio": round(smallest_margin, 6),
    }


def _trajectory_metrics(track: dict[str, Any], frame_width: int, frame_height: int) -> dict[str, Any]:
    trajectory = list(track.get("sanitized_valid_timeline", track.get("rebuilt_timeline", track.get("trajectory", []))))
    if not trajectory:
        return {
            "average_bbox_area": 0.0,
            "minimum_bbox_area": 0.0,
            "maximum_bbox_area": 0.0,
            "area_change_stability": 0.0,
            "center_motion_stability": 0.0,
            "suspicious_jump_count": 0,
            "yolo_ratio": 0.0,
            "kcf_ratio": 0.0,
            "has_detector_confirmed_crop": False,
            "maximum_detection_gap": 0.0,
            "entry_boundary": None,
            "exit_boundary": None,
            "boundary_partial": False,
        }
    areas = [_bbox_area(list(item["bbox_xyxy"])) for item in trajectory]
    yolo_count = len([item for item in trajectory if str(item.get("bbox_source")) == "yolo"])
    kcf_count = len([item for item in trajectory if str(item.get("bbox_source")) == "kcf"])
    jump_count = 0
    center_distances: list[float] = []
    area_ratios: list[float] = []
    last_yolo_timestamp: float | None = None
    maximum_detection_gap = 0.0
    for index in range(1, len(trajectory)):
        left = trajectory[index - 1]
        right = trajectory[index]
        left_bbox = list(left["bbox_xyxy"])
        right_bbox = list(right["bbox_xyxy"])
        left_center = _bbox_center(left_bbox)
        right_center = _bbox_center(right_bbox)
        center_distance = math.dist(left_center, right_center)
        diagonal = max(math.sqrt(max(_bbox_area(left_bbox), 1.0)), 1.0)
        normalized_distance = center_distance / diagonal
        center_distances.append(normalized_distance)
        area_ratios.append(_bbox_area(right_bbox) / max(_bbox_area(left_bbox), 1.0))
        if normalized_distance > 1.5:
            jump_count += 1
        if str(right.get("bbox_source")) == "yolo":
            if last_yolo_timestamp is not None:
                maximum_detection_gap = max(maximum_detection_gap, float(right["timestamp_seconds"]) - last_yolo_timestamp)
            last_yolo_timestamp = float(right["timestamp_seconds"])
    if last_yolo_timestamp is None:
        maximum_detection_gap = float(track.get("sanitized_duration_seconds", track.get("duration_seconds", 0.0)) or 0.0)
    first_boundary = _boundary_info(list(trajectory[0]["bbox_xyxy"]), frame_width, frame_height)
    last_boundary = _boundary_info(list(trajectory[-1]["bbox_xyxy"]), frame_width, frame_height)
    return {
        "average_bbox_area": round(float(mean(areas)), 6),
        "minimum_bbox_area": round(float(min(areas)), 6),
        "maximum_bbox_area": round(float(max(areas)), 6),
        "area_change_stability": round(max(0.0, 1.0 - float(mean(abs(ratio - 1.0) for ratio in area_ratios))) if area_ratios else 1.0, 6),
        "center_motion_stability": round(max(0.0, 1.0 - float(mean(center_distances))) if center_distances else 1.0, 6),
        "suspicious_jump_count": int(jump_count),
        "yolo_ratio": round(float(yolo_count) / max(len(trajectory), 1), 6),
        "kcf_ratio": round(float(kcf_count) / max(len(trajectory), 1), 6),
        "has_detector_confirmed_crop": bool(yolo_count > 0),
        "maximum_detection_gap": round(float(maximum_detection_gap), 6),
        "entry_boundary": first_boundary["boundary_name"],
        "exit_boundary": last_boundary["boundary_name"],
        "boundary_partial": bool(first_boundary["touches_boundary"] or last_boundary["touches_boundary"]),
    }


def quality_score_components(track: dict[str, Any], frame_width: int, frame_height: int) -> dict[str, float]:
    duration_seconds = float(track.get("sanitized_duration_seconds", track.get("duration_seconds", 0.0)) or 0.0)
    detection_hits = int(track.get("valid_yolo_count", track.get("detection_hits", 0)) or 0)
    propagation_hits = int(track.get("valid_kcf_count", track.get("propagation_hits", 0)) or 0)
    kcf_failures = int(track.get("kcf_failures", 0) or 0)
    metrics = _trajectory_metrics(track, frame_width, frame_height)
    confirmation_score = 1.0 if bool(track.get("is_confirmed")) else min(0.4, detection_hits / 3.0)
    duration_score = min(1.0, duration_seconds / 2.0)
    detector_evidence_score = min(1.0, detection_hits / 5.0)
    box_stability_score = max(0.0, min(1.0, (metrics["area_change_stability"] + metrics["center_motion_stability"]) / 2.0))
    kcf_reliability_score = 1.0 - min(1.0, kcf_failures / max(propagation_hits + detection_hits, 1))
    active_trajectory = list(track.get("sanitized_valid_timeline", track.get("rebuilt_timeline", track.get("trajectory", []))))
    trajectory_consistency_score = 1.0 - min(1.0, metrics["suspicious_jump_count"] / max(len(active_trajectory) - 1, 1))
    crop_availability_score = 1.0 if metrics["has_detector_confirmed_crop"] else 0.0
    return {
        "confirmation_score": round(confirmation_score, 6),
        "duration_score": round(duration_score, 6),
        "detector_evidence_score": round(detector_evidence_score, 6),
        "box_stability_score": round(box_stability_score, 6),
        "kcf_reliability_score": round(kcf_reliability_score, 6),
        "trajectory_consistency_score": round(trajectory_consistency_score, 6),
        "crop_availability_score": round(crop_availability_score, 6),
    }


def quality_score(track: dict[str, Any], frame_width: int, frame_height: int) -> tuple[float, dict[str, float]]:
    components = quality_score_components(track, frame_width, frame_height)
    score = (
        0.20 * components["confirmation_score"]
        + 0.15 * components["duration_score"]
        + 0.15 * components["detector_evidence_score"]
        + 0.15 * components["box_stability_score"]
        + 0.15 * components["kcf_reliability_score"]
        + 0.10 * components["trajectory_consistency_score"]
        + 0.10 * components["crop_availability_score"]
    )
    return round(float(score), 6), components


def _class_vote_stability(track: dict[str, Any]) -> float:
    class_votes = dict(track.get("class_votes", {}))
    if not class_votes:
        return 0.0
    values = sorted((float(value) for value in class_votes.values()), reverse=True)
    if len(values) == 1:
        return 1.0
    return round(values[0] / max(sum(values), 1e-6), 6)


def evaluate_track_quality(track: dict[str, Any], *, frame_width: int, frame_height: int) -> dict[str, Any]:
    metrics = _trajectory_metrics(track, frame_width, frame_height)
    score, components = quality_score(track, frame_width, frame_height)
    duration_seconds = float(track.get("sanitized_duration_seconds", track.get("duration_seconds", 0.0)) or 0.0)
    detection_hits = int(track.get("valid_yolo_count", track.get("detection_hits", 0)) or 0)
    sanitized_valid_observation_count = int(track.get("valid_observation_count", len(list(track.get("sanitized_valid_timeline", []))) or len(list(track.get("trajectory", [])))))
    integrity_status = str(track.get("track_integrity_status", "usable" if sanitized_valid_observation_count else "invalid"))
    frozen_kcf_detected = bool(track.get("frozen_kcf_detected", False))
    boundary_stuck_detected = bool(track.get("boundary_stuck_detected", False))
    long_kcf_only_gap_detected = bool(track.get("long_kcf_only_gap_detected", False))
    valid_crop_candidate_count = int(track.get("valid_crop_candidate_count", detection_hits) or 0)
    timeline_integrity_valid = not any(
        flag in list(track.get("integrity_flags", []))
        for flag in {"missing_frame_level_observations", "negative_duration", "non_monotonic_timestamp", "non_monotonic_source_frame"}
    )
    flags: list[str] = []
    if duration_seconds < 0.5:
        flags.append("short_track")
    if detection_hits <= 1:
        flags.append("single_detection")
    if detection_hits < 3:
        flags.append("few_detector_hits")
    if int(track.get("kcf_failures", 0) or 0) > max(int(track.get("propagation_hits", 0) or 0), 1):
        flags.append("high_kcf_failure_rate")
    if metrics["maximum_detection_gap"] > 1.0:
        flags.append("long_kcf_only_gap")
    if metrics["area_change_stability"] < 0.35:
        flags.append("unstable_bbox_scale")
    if metrics["suspicious_jump_count"] > 0:
        flags.append("large_center_jump")
    if _class_vote_stability(track) < 0.55:
        flags.append("class_instability")
    if metrics["boundary_partial"]:
        flags.append("boundary_partial")
    if not metrics["has_detector_confirmed_crop"]:
        flags.append("missing_detector_frame")
        flags.append("missing_crop")
    if str(track.get("termination_reason")) == "lost_recovery_expired":
        flags.append("lost_recovery_expired")
    if str(track.get("lost_reason")) == "missed_refresh_limit":
        flags.append("missed_refresh_termination")
    if frozen_kcf_detected:
        flags.append("frozen_kcf_detected")
    if boundary_stuck_detected:
        flags.append("boundary_stuck_detected")
    if long_kcf_only_gap_detected:
        flags.append("long_kcf_only_gap_detected")
    if integrity_status != "usable":
        flags.append(f"integrity_{integrity_status}")
    impossible_geometry = metrics["maximum_bbox_area"] <= 0.0
    if impossible_geometry:
        flags.append("likely_false_detection")
    hard_invalid = any(
        [
            impossible_geometry,
            not timeline_integrity_valid,
            integrity_status == "invalid",
            sanitized_valid_observation_count <= 0,
            detection_hits <= 0 and int(track.get("valid_kcf_count", 0) or 0) <= 0,
            frozen_kcf_detected and detection_hits == 0,
        ]
    )
    manual_review = any(
        [
            boundary_stuck_detected and detection_hits > 0,
            int(track.get("invalid_observation_count", 0) or 0) > sanitized_valid_observation_count,
            duration_seconds < 0.5 and detection_hits >= 2,
            _class_vote_stability(track) < 0.55,
            valid_crop_candidate_count <= 0,
            long_kcf_only_gap_detected,
            integrity_status in {"weak_single_detection", "fallback_only"},
        ]
    )
    if hard_invalid:
        quality_level = "invalid"
        downstream_status = "rejected"
    elif (
        score >= 0.75
        and bool(track.get("is_confirmed", track.get("confirmed", False)))
        and detection_hits >= 4
        and duration_seconds >= 1.0
        and metrics["suspicious_jump_count"] == 0
        and metrics["has_detector_confirmed_crop"]
        and not manual_review
    ):
        quality_level = "high"
        downstream_status = "ready"
    elif (
        score >= 0.50
        and bool(track.get("is_confirmed", track.get("confirmed", False)))
        and detection_hits >= 3
        and duration_seconds >= 0.5
        and metrics["has_detector_confirmed_crop"]
        and not manual_review
    ):
        quality_level = "medium"
        downstream_status = "ready"
    else:
        quality_level = "low"
        downstream_status = "manual_review" if manual_review else "fallback"
    return {
        "track_id": int(track.get("track_id", 0)),
        "object_family": object_family_for_class(str(track.get("class_name", ""))),
        "final_class": str(track.get("class_name", "unknown")),
        "duration_seconds": round(duration_seconds, 6),
        "detector_hit_count": detection_hits,
        "kcf_propagation_hit_count": int(track.get("propagation_hits", 0) or 0),
        "kcf_failure_count": int(track.get("kcf_failures", 0) or 0),
        "missed_detector_refresh_count": int(track.get("missed_detection_refreshes", 0) or 0),
        "maximum_detection_gap": metrics["maximum_detection_gap"],
        "yolo_bbox_ratio": metrics["yolo_ratio"],
        "kcf_bbox_ratio": metrics["kcf_ratio"],
        "average_bbox_area": metrics["average_bbox_area"],
        "minimum_bbox_area": metrics["minimum_bbox_area"],
        "maximum_bbox_area": metrics["maximum_bbox_area"],
        "area_change_stability": metrics["area_change_stability"],
        "center_motion_stability": metrics["center_motion_stability"],
        "suspicious_jump_count": metrics["suspicious_jump_count"],
        "touches_entry_or_exit_boundary": metrics["boundary_partial"],
        "entry_boundary": metrics["entry_boundary"],
        "exit_boundary": metrics["exit_boundary"],
        "has_detector_confirmed_crop": metrics["has_detector_confirmed_crop"],
        "confirmed": bool(track.get("is_confirmed", track.get("confirmed", False))),
        "termination_reason": track.get("termination_reason"),
        "reactivation_count": int(track.get("reactivation_count", 0) or 0),
        "class_vote_stability": _class_vote_stability(track),
        "integrity_status": integrity_status,
        "timeline_integrity_valid": timeline_integrity_valid,
        "frozen_kcf_detected": frozen_kcf_detected,
        "boundary_stuck_detected": boundary_stuck_detected,
        "long_kcf_only_gap_detected": long_kcf_only_gap_detected,
        "sanitized_valid_observation_count": sanitized_valid_observation_count,
        "sanitized_detector_hit_count": detection_hits,
        "sanitized_duration": round(duration_seconds, 6),
        "valid_crop_candidate_count": valid_crop_candidate_count,
        "quality_score": score,
        "quality_score_components": components,
        "quality_level": quality_level,
        "downstream_status": downstream_status,
        "quality_flags": sorted(set(flags)),
        "source_raw_track_ids": [int(track.get("track_id", 0))],
    }


def build_track_quality_report(
    tracks: list[dict[str, Any]],
    *,
    frame_width: int,
    frame_height: int,
) -> dict[str, Any]:
    evaluations = [evaluate_track_quality(track, frame_width=frame_width, frame_height=frame_height) for track in tracks]
    quality_breakdown = Counter(item["quality_level"] for item in evaluations)
    return {
        "status": "success",
        "raw_track_id_count": len(tracks),
        "confirmed_raw_track_segment_count": len([item for item in evaluations if item["confirmed"]]),
        "quality_breakdown": dict(sorted(quality_breakdown.items())),
        "score_formula": {
            "confirmation_score": 0.20,
            "duration_score": 0.15,
            "detector_evidence_score": 0.15,
            "box_stability_score": 0.15,
            "kcf_reliability_score": 0.15,
            "trajectory_consistency_score": 0.10,
            "crop_availability_score": 0.10,
        },
        "tracks": evaluations,
    }


def build_track_quality_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Track Quality Report",
            "",
            f"- Raw track IDs: {report['raw_track_id_count']}",
            f"- Confirmed raw track segments: {report['confirmed_raw_track_segment_count']}",
            f"- High quality: {report['quality_breakdown'].get('high', 0)}",
            f"- Medium quality: {report['quality_breakdown'].get('medium', 0)}",
            f"- Low quality: {report['quality_breakdown'].get('low', 0)}",
            f"- Invalid quality: {report['quality_breakdown'].get('invalid', 0)}",
        ]
    )
