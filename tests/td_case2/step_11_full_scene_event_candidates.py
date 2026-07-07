from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from stage_checks import format_seconds_text, read_json, write_json
from step_09_search_result_packaging import write_json_any


VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "van", "auto", "bicycle"}
PERSON_CLASSES = {"person"}
EVENT_FAMILY_MAP = {
    "possible_collision_or_near_miss": "traffic_safety",
    "sudden_stop": "traffic_safety",
    "stationary_vehicle": "traffic_flow",
    "traffic_congestion_or_dense_vehicle_activity": "traffic_flow",
    "vehicle_person_interaction": "traffic_safety",
    "unusual_motion_spike": "scene_activity",
    "object_density_spike": "scene_activity",
    "track_start_stop_activity": "scene_activity",
}
SUPPORTED_EVENT_TYPES = [
    "possible_collision_or_near_miss",
    "sudden_stop",
    "stationary_vehicle",
    "traffic_congestion_or_dense_vehicle_activity",
    "vehicle_person_interaction",
    "unusual_motion_spike",
    "object_density_spike",
    "track_start_stop_activity",
]
COMPATIBLE_EVENT_GROUPS = {
    "possible_collision_or_near_miss": {"possible_collision_or_near_miss", "sudden_stop", "unusual_motion_spike"},
    "sudden_stop": {"possible_collision_or_near_miss", "sudden_stop", "track_start_stop_activity"},
    "stationary_vehicle": {"stationary_vehicle", "traffic_congestion_or_dense_vehicle_activity"},
    "traffic_congestion_or_dense_vehicle_activity": {"stationary_vehicle", "traffic_congestion_or_dense_vehicle_activity", "object_density_spike"},
    "vehicle_person_interaction": {"vehicle_person_interaction", "unusual_motion_spike"},
    "unusual_motion_spike": {"unusual_motion_spike", "object_density_spike", "track_start_stop_activity"},
    "object_density_spike": {"unusual_motion_spike", "object_density_spike", "traffic_congestion_or_dense_vehicle_activity"},
    "track_start_stop_activity": {"track_start_stop_activity", "unusual_motion_spike", "sudden_stop"},
}
STEP02A_CANDIDATE_FILES = [
    "02A_motion_adaptive_frames.json",
    "02A_motion_adaptive_sampling.json",
    "02A_motion_selected_frames.json",
    "02A_adaptive_frames.json",
    "02A_adaptive_sampling_report.json",
]


def _safe_divide(numerator: float, denominator: float) -> float:
    """Divide safely with a zero fallback."""

    if denominator == 0:
        return 0.0
    return numerator / denominator


def _mean(values: list[float]) -> float:
    """Return a stable average."""

    if not values:
        return 0.0
    return sum(values) / len(values)


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp a float into the requested range."""

    return max(low, min(high, value))


def _resolve_step02a_path(run_dir: Path) -> Path:
    """Locate a Step 02A timeline file tolerantly."""

    for filename in STEP02A_CANDIDATE_FILES:
        candidate = run_dir / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No Step 02A adaptive frame file was found in the run directory.")


def _normalize_detection_class(class_name: str) -> str | None:
    """Map raw classes into the compact Step 11 class set."""

    normalized = str(class_name or "").strip().lower()
    if normalized in VEHICLE_CLASSES or normalized in PERSON_CLASSES:
        return normalized
    if normalized in {"bike"}:
        return "bicycle"
    if normalized in {"vehicle"}:
        return "car"
    return None


def _bbox_center_and_area(bbox_xyxy: list[float]) -> tuple[float, float, float]:
    """Return bbox center and area from xyxy coordinates."""

    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    center_x = x1 + width / 2.0
    center_y = y1 + height / 2.0
    return center_x, center_y, width * height


def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    """Calculate simple bbox IoU."""

    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return _safe_divide(intersection, union)


def _track_speed_features(track: dict[str, Any], frame_width: int, frame_height: int) -> dict[str, Any]:
    """Calculate simple motion features for one track from bbox centers."""

    detections = sorted(list(track.get("detections", [])), key=lambda item: float(item.get("timestamp_seconds", 0.0) or 0.0))
    frame_diagonal = math.sqrt(frame_width * frame_width + frame_height * frame_height) or 1.0
    center_points: list[tuple[float, float, float, float]] = []
    speeds: list[float] = []
    normalized_speeds: list[float] = []
    directions: list[float] = []
    accelerations: list[float] = []

    for detection in detections:
        bbox = list(detection.get("bbox_xyxy", []))
        if len(bbox) != 4:
            continue
        center_x, center_y, bbox_area = _bbox_center_and_area(bbox)
        center_points.append((float(detection.get("timestamp_seconds", 0.0) or 0.0), center_x, center_y, bbox_area))

    for index in range(1, len(center_points)):
        previous_timestamp, previous_x, previous_y, _previous_area = center_points[index - 1]
        current_timestamp, current_x, current_y, _current_area = center_points[index]
        delta_t = max(0.001, current_timestamp - previous_timestamp)
        delta_x = current_x - previous_x
        delta_y = current_y - previous_y
        speed = math.sqrt(delta_x * delta_x + delta_y * delta_y) / delta_t
        speeds.append(speed)
        normalized_speeds.append(speed / frame_diagonal)
        directions.append(math.degrees(math.atan2(delta_y, delta_x)))

    for index in range(1, len(speeds)):
        delta_t = max(0.001, center_points[index + 1][0] - center_points[index][0])
        accelerations.append((speeds[index] - speeds[index - 1]) / delta_t)

    direction_change_score = 0.0
    for index in range(1, len(directions)):
        delta_angle = abs(directions[index] - directions[index - 1])
        delta_angle = min(delta_angle, 360.0 - delta_angle)
        direction_change_score = max(direction_change_score, delta_angle / 180.0)

    avg_normalized_speed = _mean(normalized_speeds)
    max_normalized_speed = max(normalized_speeds) if normalized_speeds else 0.0
    stopped_score = _clamp(1.0 - (avg_normalized_speed / 0.02), 0.0, 1.0)

    sudden_stop_score = 0.0
    if speeds:
        early_speed = max(speeds[: max(1, len(speeds) // 2)])
        late_speed = _mean(speeds[max(0, len(speeds) - 3) :])
        if early_speed > 0:
            sudden_stop_score = _clamp((early_speed - late_speed) / early_speed, 0.0, 1.0)

    return {
        "center_points": center_points,
        "speed_px_per_sec_avg": round(_mean(speeds), 6),
        "speed_px_per_sec_max": round(max(speeds) if speeds else 0.0, 6),
        "normalized_speed_avg": round(avg_normalized_speed, 6),
        "normalized_speed_max": round(max_normalized_speed, 6),
        "direction_angle_avg": round(_mean(directions), 6) if directions else 0.0,
        "acceleration_estimate_max": round(max((abs(value) for value in accelerations), default=0.0), 6),
        "stopped_score": round(stopped_score, 6),
        "sudden_stop_score": round(sudden_stop_score, 6),
        "direction_change_score": round(direction_change_score, 6),
        "stationary_candidate": (
            str(track.get("track_type", "")) == "vehicle"
            and float(track.get("duration_seconds", 0.0) or 0.0) >= 3.0
            and avg_normalized_speed <= 0.01
        ),
    }


def _flatten_yolo_detections(detections_payload: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Flatten Step 03 per-frame detections into a single list."""

    flat_detections: list[dict[str, Any]] = []
    by_frame_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame_item in list(detections_payload.get("detections", [])):
        frame_id = str(frame_item.get("frame_id", "") or "")
        frame_idx = int(frame_item.get("frame_idx", 0) or 0)
        timestamp_seconds = float(frame_item.get("timestamp_seconds", 0.0) or 0.0)
        image_path = str(frame_item.get("image_path", "") or "")
        for detection in list(frame_item.get("detections", [])):
            normalized_class = _normalize_detection_class(str(detection.get("class_name", "") or ""))
            if normalized_class is None:
                continue
            item = {
                "frame_id": frame_id,
                "frame_idx": frame_idx,
                "timestamp_seconds": timestamp_seconds,
                "image_path": image_path,
                "detection_id": str(detection.get("detection_id", "") or ""),
                "class_name": normalized_class,
                "confidence": float(detection.get("confidence", 0.0) or 0.0),
                "bbox_xyxy": list(detection.get("bbox_xyxy", [])),
                "crop_path": detection.get("crop_path"),
                "annotated_frame_path": detection.get("annotated_frame_path"),
            }
            flat_detections.append(item)
            by_frame_id[frame_id].append(item)
    return flat_detections, by_frame_id


def _build_scene_windows(
    *,
    selected_frames: list[dict[str, Any]],
    yolo_by_frame_id: dict[str, list[dict[str, Any]]],
    tracks: list[dict[str, Any]],
    duration_seconds: float,
    window_seconds: float,
    stride_seconds: float,
) -> list[dict[str, Any]]:
    """Split the scene into sliding windows with aggregated evidence."""

    windows: list[dict[str, Any]] = []
    current_start = 0.0
    window_index = 1
    while current_start <= duration_seconds:
        current_end = min(duration_seconds, current_start + window_seconds)
        center_timestamp = current_start + (current_end - current_start) / 2.0
        frames_in_window = [
            frame for frame in selected_frames if current_start <= float(frame.get("timestamp_seconds", 0.0) or 0.0) <= current_end
        ]
        representative_frame = None
        if frames_in_window:
            representative_frame = min(
                frames_in_window,
                key=lambda item: abs(float(item.get("timestamp_seconds", 0.0) or 0.0) - center_timestamp),
            )

        frame_object_counts: list[int] = []
        frame_vehicle_counts: list[int] = []
        frame_person_counts: list[int] = []
        class_counts: Counter[str] = Counter()
        for frame in frames_in_window:
            frame_detections = yolo_by_frame_id.get(str(frame.get("frame_id", "") or ""), [])
            vehicle_count = sum(1 for detection in frame_detections if detection["class_name"] in VEHICLE_CLASSES)
            person_count = sum(1 for detection in frame_detections if detection["class_name"] in PERSON_CLASSES)
            frame_object_counts.append(len(frame_detections))
            frame_vehicle_counts.append(vehicle_count)
            frame_person_counts.append(person_count)
            class_counts.update(detection["class_name"] for detection in frame_detections)

        active_tracks = [
            track
            for track in tracks
            if float(track.get("start_timestamp_seconds", 0.0) or 0.0) <= current_end
            and float(track.get("end_timestamp_seconds", 0.0) or 0.0) >= current_start
        ]

        windows.append(
            {
                "window_id": f"scene_win_{window_index:06d}",
                "start_timestamp_seconds": round(current_start, 6),
                "end_timestamp_seconds": round(current_end, 6),
                "center_timestamp_seconds": round(center_timestamp, 6),
                "frame_count": len(frames_in_window),
                "representative_frame_id": representative_frame.get("frame_id") if representative_frame else None,
                "representative_frame_path": representative_frame.get("image_path") if representative_frame else None,
                "representative_frame_idx": representative_frame.get("frame_idx") if representative_frame else None,
                "motion_score_max": round(max((float(frame.get("motion_score", 0.0) or 0.0) for frame in frames_in_window), default=0.0), 6),
                "motion_score_avg": round(_mean([float(frame.get("motion_score", 0.0) or 0.0) for frame in frames_in_window]), 6),
                "motion_pixels_ratio_max": round(max((float(frame.get("motion_pixels_ratio", 0.0) or 0.0) for frame in frames_in_window), default=0.0), 6),
                "histogram_change_max": round(max((float(frame.get("histogram_change_score", 0.0) or 0.0) for frame in frames_in_window), default=0.0), 6),
                "motion_blob_count_max": int(max((int(frame.get("motion_blob_count", 0) or 0) for frame in frames_in_window), default=0)),
                "object_count_max": int(max(frame_object_counts, default=0)),
                "vehicle_count_max": int(max(frame_vehicle_counts, default=0)),
                "person_count_max": int(max(frame_person_counts, default=0)),
                "class_counts": dict(class_counts),
                "active_track_ids": [str(track.get("track_id", "") or "") for track in active_tracks],
                "active_vehicle_track_ids": [str(track.get("track_id", "") or "") for track in active_tracks if str(track.get("track_type", "")) == "vehicle"],
                "active_person_track_ids": [str(track.get("track_id", "") or "") for track in active_tracks if str(track.get("track_type", "")) == "person"],
            }
        )
        if current_end >= duration_seconds:
            break
        current_start += stride_seconds
        window_index += 1
    return windows


def _track_detection_at_time(track: dict[str, Any], center_timestamp: float, tolerance: float = 0.6) -> dict[str, Any] | None:
    """Find the closest track detection to a window center."""

    best_item = None
    best_gap = None
    for detection in list(track.get("detections", [])):
        gap = abs(float(detection.get("timestamp_seconds", 0.0) or 0.0) - center_timestamp)
        if gap <= tolerance and (best_gap is None or gap < best_gap):
            best_gap = gap
            best_item = detection
    return best_item


def _search_record_enrichment(record_by_track_id: dict[str, dict[str, Any]], track_id: str) -> dict[str, Any]:
    """Return optional Step 07 enrichment for one track."""

    return record_by_track_id.get(track_id, {})


def _candidate_score_label(score: float) -> str:
    """Map numeric score to candidate confidence label."""

    if score >= 0.80:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def _severity_label(event_type: str, score: float) -> str:
    """Map event type and score to a rough severity label."""

    if event_type == "possible_collision_or_near_miss" and score >= 0.75:
        return "high"
    if event_type in {"sudden_stop", "traffic_congestion_or_dense_vehicle_activity", "vehicle_person_interaction"} and score >= 0.55:
        return "medium"
    return "low"


def _representative_full_frame_paths(
    selected_frames: list[dict[str, Any]],
    best_timestamp_seconds: float,
    context_start_seconds: float,
    context_end_seconds: float,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Choose representative full-scene images around an event."""

    frames_in_context = [
        frame
        for frame in selected_frames
        if context_start_seconds <= float(frame.get("timestamp_seconds", 0.0) or 0.0) <= context_end_seconds
    ]
    if not frames_in_context:
        return None, []

    representative_frame = min(
        frames_in_context,
        key=lambda frame: abs(float(frame.get("timestamp_seconds", 0.0) or 0.0) - best_timestamp_seconds),
    )
    sorted_context = sorted(
        frames_in_context,
        key=lambda frame: abs(float(frame.get("timestamp_seconds", 0.0) or 0.0) - best_timestamp_seconds),
    )
    unique_paths: list[str] = []
    for frame in sorted_context[:3]:
        image_path = str(frame.get("image_path", "") or "")
        if image_path and image_path not in unique_paths:
            unique_paths.append(image_path)
    return representative_frame, unique_paths


def _raw_trigger(
    *,
    trigger_index: int,
    event_type: str,
    timestamp_seconds: float,
    window_id: str,
    score: float,
    trigger_reasons: list[str],
    involved_track_ids: list[str],
    involved_classes: list[str],
    representative_frame_path: str | None,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Build one raw trigger record."""

    return {
        "trigger_id": f"raw_evt_trigger_{trigger_index:06d}",
        "event_type": event_type,
        "timestamp_seconds": round(timestamp_seconds, 6),
        "timestamp_text": format_seconds_text(timestamp_seconds),
        "window_id": window_id,
        "score": round(_clamp(score, 0.0, 1.0), 6),
        "trigger_reasons": sorted(set(trigger_reasons)),
        "involved_track_ids": sorted(set(involved_track_ids)),
        "involved_classes": sorted(set(involved_classes)),
        "representative_frame_path": representative_frame_path,
        "evidence": evidence,
    }


def _build_raw_triggers(
    *,
    windows: list[dict[str, Any]],
    selected_frames: list[dict[str, Any]],
    tracks: list[dict[str, Any]],
    track_features: dict[str, dict[str, Any]],
    record_by_track_id: dict[str, dict[str, Any]],
    min_candidate_score: float,
) -> list[dict[str, Any]]:
    """Create raw event triggers from windows, tracks, and interactions."""

    triggers: list[dict[str, Any]] = []
    trigger_index = 1
    track_by_id = {str(track.get("track_id", "") or ""): track for track in tracks}
    motion_scores = [float(window.get("motion_score_max", 0.0) or 0.0) for window in windows]
    average_motion_peak = _mean(motion_scores)

    for window in windows:
        reasons_base: list[str] = []
        score_base = 0.0
        if float(window["motion_score_max"]) > max(0.05, average_motion_peak * 1.5):
            score_base += 0.15
            reasons_base.append("motion_spike")
        if float(window["motion_pixels_ratio_max"]) >= 0.03:
            score_base += 0.10
            reasons_base.append("motion_pixels_high")
        if float(window["histogram_change_max"]) >= 0.12:
            score_base += 0.10
            reasons_base.append("histogram_change_high")
        if int(window["object_count_max"]) >= 5:
            score_base += 0.10
            reasons_base.append("object_density_high")
        if int(window["vehicle_count_max"]) >= 4:
            score_base += 0.10
            reasons_base.append("vehicle_density_high")

        active_vehicle_ids = list(window["active_vehicle_track_ids"])
        active_person_ids = list(window["active_person_track_ids"])
        stationary_vehicle_count = sum(1 for track_id in active_vehicle_ids if track_features.get(track_id, {}).get("stationary_candidate"))
        sudden_stop_tracks = [
            track_id for track_id in active_vehicle_ids if float(track_features.get(track_id, {}).get("sudden_stop_score", 0.0) or 0.0) >= 0.55
        ]
        if stationary_vehicle_count >= 2 and int(window["vehicle_count_max"]) >= 4:
            score = score_base + 0.15 + 0.10
            reasons = reasons_base + ["stationary_vehicle_cluster", "slow_vehicle_activity"]
            if score >= min_candidate_score:
                triggers.append(
                    _raw_trigger(
                        trigger_index=trigger_index,
                        event_type="traffic_congestion_or_dense_vehicle_activity",
                        timestamp_seconds=float(window["center_timestamp_seconds"]),
                        window_id=str(window["window_id"]),
                        score=score,
                        trigger_reasons=reasons,
                        involved_track_ids=active_vehicle_ids,
                        involved_classes=[track_by_id[track_id]["dominant_class_name"] for track_id in active_vehicle_ids if track_id in track_by_id],
                        representative_frame_path=window.get("representative_frame_path"),
                        evidence={"stationary_track_count": stationary_vehicle_count, "window": window},
                    )
                )
                trigger_index += 1

        if reasons_base and int(window["object_count_max"]) >= 1:
            event_type = "unusual_motion_spike"
            if int(window["object_count_max"]) >= 6:
                event_type = "object_density_spike"
            score = score_base + (0.10 if len(reasons_base) >= 2 else 0.0)
            if score >= min_candidate_score:
                triggers.append(
                    _raw_trigger(
                        trigger_index=trigger_index,
                        event_type=event_type,
                        timestamp_seconds=float(window["center_timestamp_seconds"]),
                        window_id=str(window["window_id"]),
                        score=score,
                        trigger_reasons=reasons_base + (["multiple_trigger_reasons"] if len(reasons_base) >= 2 else []),
                        involved_track_ids=active_vehicle_ids + active_person_ids,
                        involved_classes=[track_by_id[track_id]["dominant_class_name"] for track_id in active_vehicle_ids + active_person_ids if track_id in track_by_id],
                        representative_frame_path=window.get("representative_frame_path"),
                        evidence={"window": window},
                    )
                )
                trigger_index += 1

        if len(active_vehicle_ids) + len(active_person_ids) >= 4:
            score = 0.15 + 0.10 + (0.10 if reasons_base else 0.0)
            if score >= min_candidate_score:
                triggers.append(
                    _raw_trigger(
                        trigger_index=trigger_index,
                        event_type="track_start_stop_activity",
                        timestamp_seconds=float(window["center_timestamp_seconds"]),
                        window_id=str(window["window_id"]),
                        score=score,
                        trigger_reasons=["multiple_active_tracks"] + reasons_base,
                        involved_track_ids=active_vehicle_ids + active_person_ids,
                        involved_classes=[track_by_id[track_id]["dominant_class_name"] for track_id in active_vehicle_ids + active_person_ids if track_id in track_by_id],
                        representative_frame_path=window.get("representative_frame_path"),
                        evidence={"window": window},
                    )
                )
                trigger_index += 1

        for track_id in sudden_stop_tracks:
            track = track_by_id.get(track_id)
            if track is None:
                continue
            score = 0.20 + score_base + 0.10
            if score >= min_candidate_score:
                triggers.append(
                    _raw_trigger(
                        trigger_index=trigger_index,
                        event_type="sudden_stop",
                        timestamp_seconds=float(window["center_timestamp_seconds"]),
                        window_id=str(window["window_id"]),
                        score=score,
                        trigger_reasons=reasons_base + ["sudden_speed_change", "track_level_stop_signal"],
                        involved_track_ids=[track_id],
                        involved_classes=[str(track.get("dominant_class_name", "") or "")],
                        representative_frame_path=window.get("representative_frame_path"),
                        evidence={"window": window, "track_features": track_features.get(track_id, {})},
                    )
                )
                trigger_index += 1

        for track_id in active_vehicle_ids:
            track = track_by_id.get(track_id)
            if track is None:
                continue
            feature = track_features.get(track_id, {})
            if not feature.get("stationary_candidate"):
                continue
            score = 0.15 + (0.10 if str(track.get("track_quality", "")) == "good" else 0.0)
            if score >= min_candidate_score:
                triggers.append(
                    _raw_trigger(
                        trigger_index=trigger_index,
                        event_type="stationary_vehicle",
                        timestamp_seconds=float(window["center_timestamp_seconds"]),
                        window_id=str(window["window_id"]),
                        score=score,
                        trigger_reasons=["stationary_vehicle_track", "low_average_speed"],
                        involved_track_ids=[track_id],
                        involved_classes=[str(track.get("dominant_class_name", "") or "")],
                        representative_frame_path=window.get("representative_frame_path"),
                        evidence={"window": window, "track_features": feature},
                    )
                )
                trigger_index += 1

        # Vehicle / vehicle close interaction.
        for left_index in range(len(active_vehicle_ids)):
            left_track = track_by_id.get(active_vehicle_ids[left_index])
            if left_track is None:
                continue
            left_detection = _track_detection_at_time(left_track, float(window["center_timestamp_seconds"]))
            if left_detection is None:
                continue
            for right_index in range(left_index + 1, len(active_vehicle_ids)):
                right_track = track_by_id.get(active_vehicle_ids[right_index])
                if right_track is None:
                    continue
                right_detection = _track_detection_at_time(right_track, float(window["center_timestamp_seconds"]))
                if right_detection is None:
                    continue
                left_center_x, left_center_y, _left_area = _bbox_center_and_area(list(left_detection.get("bbox_xyxy", [])))
                right_center_x, right_center_y, _right_area = _bbox_center_and_area(list(right_detection.get("bbox_xyxy", [])))
                diagonal = math.sqrt(1280 * 1280 + 720 * 720)
                center_distance_ratio = _safe_divide(
                    math.sqrt((left_center_x - right_center_x) ** 2 + (left_center_y - right_center_y) ** 2),
                    diagonal,
                )
                iou = _bbox_iou(list(left_detection.get("bbox_xyxy", [])), list(right_detection.get("bbox_xyxy", [])))
                if center_distance_ratio > 0.12 and iou <= 0.02:
                    continue
                score = 0.25
                reasons = ["vehicle_close_interaction"]
                if score_base > 0:
                    score += 0.15
                    reasons.extend(reasons_base)
                if active_vehicle_ids[left_index] in sudden_stop_tracks or active_vehicle_ids[right_index] in sudden_stop_tracks:
                    score += 0.20
                    reasons.append("sudden_speed_change")
                if iou > 0.05:
                    score += 0.10
                    reasons.append("bbox_overlap")
                if score >= min_candidate_score:
                    triggers.append(
                        _raw_trigger(
                            trigger_index=trigger_index,
                            event_type="possible_collision_or_near_miss",
                            timestamp_seconds=float(window["center_timestamp_seconds"]),
                            window_id=str(window["window_id"]),
                            score=score,
                            trigger_reasons=reasons,
                            involved_track_ids=[active_vehicle_ids[left_index], active_vehicle_ids[right_index]],
                            involved_classes=[
                                str(left_track.get("dominant_class_name", "") or ""),
                                str(right_track.get("dominant_class_name", "") or ""),
                            ],
                            representative_frame_path=window.get("representative_frame_path"),
                            evidence={
                                "window": window,
                                "close_pair_count": 1,
                                "bbox_iou": round(iou, 6),
                                "center_distance_ratio": round(center_distance_ratio, 6),
                            },
                        )
                    )
                    trigger_index += 1

        # Person / vehicle interaction.
        for vehicle_track_id in active_vehicle_ids:
            vehicle_track = track_by_id.get(vehicle_track_id)
            if vehicle_track is None:
                continue
            vehicle_detection = _track_detection_at_time(vehicle_track, float(window["center_timestamp_seconds"]))
            if vehicle_detection is None:
                continue
            vehicle_center_x, vehicle_center_y, _vehicle_area = _bbox_center_and_area(list(vehicle_detection.get("bbox_xyxy", [])))
            for person_track_id in active_person_ids:
                person_track = track_by_id.get(person_track_id)
                if person_track is None:
                    continue
                person_detection = _track_detection_at_time(person_track, float(window["center_timestamp_seconds"]))
                if person_detection is None:
                    continue
                person_center_x, person_center_y, _person_area = _bbox_center_and_area(list(person_detection.get("bbox_xyxy", [])))
                diagonal = math.sqrt(1280 * 1280 + 720 * 720)
                center_distance_ratio = _safe_divide(
                    math.sqrt((vehicle_center_x - person_center_x) ** 2 + (vehicle_center_y - person_center_y) ** 2),
                    diagonal,
                )
                if center_distance_ratio > 0.10:
                    continue
                score = 0.15 + (0.15 if score_base > 0 else 0.0)
                reasons = ["person_vehicle_proximity"] + reasons_base
                if track_features.get(vehicle_track_id, {}).get("stationary_candidate"):
                    score += 0.10
                    reasons.append("vehicle_stationary_near_person")
                if score >= min_candidate_score:
                    triggers.append(
                        _raw_trigger(
                            trigger_index=trigger_index,
                            event_type="vehicle_person_interaction",
                            timestamp_seconds=float(window["center_timestamp_seconds"]),
                            window_id=str(window["window_id"]),
                            score=score,
                            trigger_reasons=reasons,
                            involved_track_ids=[vehicle_track_id, person_track_id],
                            involved_classes=[
                                str(vehicle_track.get("dominant_class_name", "") or ""),
                                "person",
                            ],
                            representative_frame_path=window.get("representative_frame_path"),
                            evidence={"window": window, "center_distance_ratio": round(center_distance_ratio, 6)},
                        )
                    )
                    trigger_index += 1

    return triggers


def _can_merge_triggers(left: dict[str, Any], right: dict[str, Any], merge_gap_seconds: float) -> bool:
    """Return whether two triggers should be merged."""

    if abs(float(left["timestamp_seconds"]) - float(right["timestamp_seconds"])) > merge_gap_seconds:
        return False
    if right["event_type"] not in COMPATIBLE_EVENT_GROUPS.get(left["event_type"], {left["event_type"]}):
        return False
    left_tracks = set(left.get("involved_track_ids", []))
    right_tracks = set(right.get("involved_track_ids", []))
    if left_tracks & right_tracks:
        return True
    return left["window_id"] == right["window_id"]


def _merge_triggers_into_candidates(
    *,
    raw_triggers: list[dict[str, Any]],
    selected_frames: list[dict[str, Any]],
    record_by_track_id: dict[str, dict[str, Any]],
    context_before_seconds: float,
    context_after_seconds: float,
    merge_gap_seconds: float,
    max_event_seconds: float,
    video_duration_seconds: float,
    include_search_metadata: bool,
) -> list[dict[str, Any]]:
    """Merge compatible raw triggers into final candidate events."""

    sorted_triggers = sorted(raw_triggers, key=lambda item: float(item.get("timestamp_seconds", 0.0) or 0.0))
    grouped: list[list[dict[str, Any]]] = []
    for trigger in sorted_triggers:
        if not grouped or not _can_merge_triggers(grouped[-1][-1], trigger, merge_gap_seconds):
            grouped.append([trigger])
        else:
            grouped[-1].append(trigger)

    candidate_events: list[dict[str, Any]] = []
    for candidate_index, trigger_group in enumerate(grouped, start=1):
        timestamps = [float(trigger["timestamp_seconds"]) for trigger in trigger_group]
        best_trigger = max(trigger_group, key=lambda item: float(item.get("score", 0.0) or 0.0))
        merged_start = min(timestamps)
        merged_end = min(max(timestamps), merged_start + max_event_seconds)
        best_timestamp_seconds = float(best_trigger["timestamp_seconds"])
        context_start_seconds = _clamp(merged_start - context_before_seconds, 0.0, video_duration_seconds)
        context_end_seconds = _clamp(merged_end + context_after_seconds, 0.0, video_duration_seconds)
        representative_frame, full_frame_paths = _representative_full_frame_paths(
            selected_frames,
            best_timestamp_seconds,
            context_start_seconds,
            context_end_seconds,
        )
        involved_track_ids = sorted({track_id for trigger in trigger_group for track_id in trigger.get("involved_track_ids", [])})
        involved_classes = sorted({class_name for trigger in trigger_group for class_name in trigger.get("involved_classes", [])})
        merged_score = _clamp(
            max(float(trigger["score"]) for trigger in trigger_group)
            + min(0.10, 0.02 * max(0, len(trigger_group) - 1)),
            0.0,
            1.0,
        )
        trigger_reasons = sorted({reason for trigger in trigger_group for reason in trigger.get("trigger_reasons", [])})
        scene_evidence = {
            "object_count_max": max(int(trigger.get("evidence", {}).get("window", {}).get("object_count_max", 0) or 0) for trigger in trigger_group),
            "vehicle_count_max": max(int(trigger.get("evidence", {}).get("window", {}).get("vehicle_count_max", 0) or 0) for trigger in trigger_group),
            "person_count_max": max(int(trigger.get("evidence", {}).get("window", {}).get("person_count_max", 0) or 0) for trigger in trigger_group),
            "motion_score_max": round(max(float(trigger.get("evidence", {}).get("window", {}).get("motion_score_max", 0.0) or 0.0) for trigger in trigger_group), 6),
            "motion_pixels_ratio_max": round(max(float(trigger.get("evidence", {}).get("window", {}).get("motion_pixels_ratio_max", 0.0) or 0.0) for trigger in trigger_group), 6),
            "close_pair_count": sum(1 for trigger in trigger_group if trigger["event_type"] == "possible_collision_or_near_miss"),
            "stationary_track_count": sum(1 for trigger in trigger_group if trigger["event_type"] == "stationary_vehicle"),
        }

        involved_objects: list[dict[str, Any]] = []
        for track_id in involved_track_ids:
            search_record = record_by_track_id.get(track_id, {})
            object_item = {
                "track_id": track_id,
                "class_name": str(search_record.get("vehicle_class", search_record.get("dominant_class_name", "unknown")) or "unknown"),
            }
            if include_search_metadata:
                object_item["vehicle_color"] = str(search_record.get("vehicle_color", "unknown") or "unknown")
                object_item["verified_license_plate"] = str(search_record.get("verified_license_plate", "not_visible") or "not_visible")
                object_item["search_record_id"] = search_record.get("search_record_id")
                object_item["best_full_frame_path"] = search_record.get("best_full_frame_path")
            involved_objects.append(object_item)

        event_type = str(best_trigger["event_type"])
        candidate_events.append(
            {
                "candidate_event_id": f"scene_evt_{candidate_index:06d}",
                "event_type": event_type,
                "event_family": EVENT_FAMILY_MAP.get(event_type, "scene_activity"),
                "start_timestamp_seconds": round(merged_start, 6),
                "end_timestamp_seconds": round(merged_end, 6),
                "best_timestamp_seconds": round(best_timestamp_seconds, 6),
                "start_timestamp_text": format_seconds_text(merged_start),
                "end_timestamp_text": format_seconds_text(merged_end),
                "best_timestamp_text": format_seconds_text(best_timestamp_seconds),
                "context_start_seconds": round(context_start_seconds, 6),
                "context_end_seconds": round(context_end_seconds, 6),
                "context_duration_seconds": round(context_end_seconds - context_start_seconds, 6),
                "candidate_score": round(merged_score, 6),
                "confidence_label": _candidate_score_label(merged_score),
                "severity_label": _severity_label(event_type, merged_score),
                "trigger_reasons": trigger_reasons,
                "involved_track_ids": involved_track_ids,
                "involved_classes": involved_classes,
                "involved_objects": involved_objects,
                "scene_evidence": scene_evidence,
                "representative_frame": {
                    "frame_id": representative_frame.get("frame_id") if representative_frame else None,
                    "timestamp_seconds": float(representative_frame.get("timestamp_seconds", 0.0) or 0.0) if representative_frame else None,
                    "image_path": representative_frame.get("image_path") if representative_frame else None,
                },
                "full_frame_paths": full_frame_paths,
                "ready_for_step12_event_ranking": True,
                "needs_vlm_review": True,
                "final_event_truth": "unknown_candidate_only",
            }
        )
    return candidate_events


def _flat_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Create flat event candidate output."""

    return {
        "candidate_event_id": candidate["candidate_event_id"],
        "event_type": candidate["event_type"],
        "best_timestamp_text": candidate["best_timestamp_text"],
        "start_timestamp_text": candidate["start_timestamp_text"],
        "end_timestamp_text": candidate["end_timestamp_text"],
        "candidate_score": candidate["candidate_score"],
        "confidence_label": candidate["confidence_label"],
        "severity_label": candidate["severity_label"],
        "involved_track_ids": ", ".join(candidate["involved_track_ids"]),
        "involved_classes": ", ".join(candidate["involved_classes"]),
        "trigger_reasons": ", ".join(candidate["trigger_reasons"]),
        "representative_frame_path": candidate["representative_frame"]["image_path"],
        "needs_vlm_review": candidate["needs_vlm_review"],
    }


def run_full_scene_event_candidate_generation(
    *,
    run_dir: Path,
    event_config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Build rule-based full-scene event candidates from existing td_case2 outputs."""

    video_info = read_json(run_dir / "01_video_info.json")
    step02a_path = _resolve_step02a_path(run_dir)
    step02a_payload = read_json(step02a_path)
    yolo_payload = read_json(run_dir / "03_yolo_detections.json")
    tracks_payload = read_json(run_dir / "04B_tracks.json")
    tracking_report_payload = read_json(run_dir / "04B_tracking_report.json") if (run_dir / "04B_tracking_report.json").exists() else {}
    best_frames_payload = read_json(run_dir / "05_best_track_frames.json") if (run_dir / "05_best_track_frames.json").exists() else {}
    search_index_payload = read_json(run_dir / "07_vehicle_search_index.json") if (run_dir / "07_vehicle_search_index.json").exists() else {}

    selected_frames = list(step02a_payload.get("selected_frames", []))
    tracks = list(tracks_payload.get("tracks", []))
    search_records = list(search_index_payload.get("records", []))
    record_by_track_id = {str(record.get("track_id", "") or ""): record for record in search_records}

    fps = float(video_info.get("fps", 0.0) or 0.0)
    frame_count = int(video_info.get("frame_count", 0) or 0)
    duration_seconds = float(video_info.get("duration_seconds", 0.0) or 0.0)
    width = int(video_info.get("width", 0) or 0)
    height = int(video_info.get("height", 0) or 0)

    flat_detections, yolo_by_frame_id = _flatten_yolo_detections(yolo_payload)
    windows = _build_scene_windows(
        selected_frames=selected_frames,
        yolo_by_frame_id=yolo_by_frame_id,
        tracks=tracks,
        duration_seconds=duration_seconds,
        window_seconds=float(event_config["window_seconds"]),
        stride_seconds=float(event_config["window_stride_seconds"]),
    )
    track_features = {
        str(track.get("track_id", "") or ""): _track_speed_features(track, width, height)
        for track in tracks
    }
    raw_triggers = _build_raw_triggers(
        windows=windows,
        selected_frames=selected_frames,
        tracks=tracks,
        track_features=track_features,
        record_by_track_id=record_by_track_id,
        min_candidate_score=float(event_config["min_candidate_score"]),
    )
    candidate_events = _merge_triggers_into_candidates(
        raw_triggers=raw_triggers,
        selected_frames=selected_frames,
        record_by_track_id=record_by_track_id,
        context_before_seconds=float(event_config["context_before_seconds"]),
        context_after_seconds=float(event_config["context_after_seconds"]),
        merge_gap_seconds=float(event_config["merge_gap_seconds"]),
        max_event_seconds=float(event_config["max_event_seconds"]),
        video_duration_seconds=duration_seconds,
        include_search_metadata=bool(event_config["include_search_metadata"]),
    )
    candidate_events = [
        candidate
        for candidate in candidate_events
        if float(candidate.get("candidate_score", 0.0) or 0.0) >= float(event_config["min_candidate_score"])
    ]
    candidate_events.sort(key=lambda item: (-float(item["candidate_score"]), float(item["best_timestamp_seconds"])))
    flat_candidates = [_flat_candidate(candidate) for candidate in candidate_events]

    confidence_counts = Counter(candidate["confidence_label"] for candidate in candidate_events)
    severity_counts = Counter(candidate["severity_label"] for candidate in candidate_events)
    event_type_counts = Counter(candidate["event_type"] for candidate in candidate_events)
    full_event_type_counts = {event_type: int(event_type_counts.get(event_type, 0)) for event_type in SUPPORTED_EVENT_TYPES}

    summary = {
        "raw_triggers_created": len(raw_triggers),
        "candidate_events_created": len(candidate_events),
        "high_confidence_candidates": confidence_counts.get("high", 0),
        "medium_confidence_candidates": confidence_counts.get("medium", 0),
        "low_confidence_candidates": confidence_counts.get("low", 0),
        "event_type_counts": full_event_type_counts,
        "ready_for_step12_event_ranking": len(candidate_events) > 0,
    }
    output_payload = {
        "status": "success",
        "source_files": {
            "video_info": "01_video_info.json",
            "adaptive_frames": step02a_path.name,
            "yolo_detections": "03_yolo_detections.json",
            "tracks": "04B_tracks.json",
            "tracking_report": "04B_tracking_report.json" if tracking_report_payload else None,
            "best_track_frames": "05_best_track_frames.json" if best_frames_payload else None,
            "search_index": "07_vehicle_search_index.json" if search_index_payload else None,
        },
        "config": event_config,
        "summary": summary,
        "candidate_events": candidate_events,
    }
    warnings: list[str] = []
    if not candidate_events:
        warnings.append("No candidate event found; inspect thresholds or try lower min score.")
    report_payload = {
        "status": "success",
        "video_duration_seconds": duration_seconds,
        "video_fps": fps,
        "video_frame_count": frame_count,
        "windows_created": len(windows),
        "frames_used": len(selected_frames),
        "tracks_used": len(tracks),
        "yolo_detections_loaded": len(flat_detections),
        "raw_triggers_created": len(raw_triggers),
        "candidate_events_created": len(candidate_events),
        "event_type_counts": full_event_type_counts,
        "confidence_counts": {
            "high": confidence_counts.get("high", 0),
            "medium": confidence_counts.get("medium", 0),
            "low": confidence_counts.get("low", 0),
        },
        "severity_counts": {
            "high": severity_counts.get("high", 0),
            "medium": severity_counts.get("medium", 0),
            "low": severity_counts.get("low", 0),
        },
        "top_candidates": [
            {
                "candidate_event_id": candidate["candidate_event_id"],
                "event_type": candidate["event_type"],
                "best_timestamp_text": candidate["best_timestamp_text"],
                "candidate_score": candidate["candidate_score"],
                "confidence_label": candidate["confidence_label"],
                "trigger_reasons": candidate["trigger_reasons"],
            }
            for candidate in candidate_events[: int(event_config["top_k_preview"])]
        ],
        "warnings": warnings,
        "recommendation": (
            "Proceed to Step 12 Event Candidate Ranking / Top-K event selection."
            if candidate_events
            else "No candidate event found; inspect thresholds or try lower min score."
        ),
    }

    write_json(run_dir / "11_full_scene_event_candidates.json", output_payload)
    if bool(event_config["save_flat"]):
        write_json_any(run_dir / "11_full_scene_event_candidates_flat.json", flat_candidates)
    else:
        write_json_any(run_dir / "11_full_scene_event_candidates_flat.json", [])
    write_json(run_dir / "11_full_scene_event_candidate_report.json", report_payload)
    return output_payload, flat_candidates, report_payload
