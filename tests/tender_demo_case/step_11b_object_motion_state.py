from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


CLASS_GROUPS = {
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "motorcycle": "vehicle",
    "bicycle": "vehicle",
    "auto rickshaw": "vehicle",
    "scooter": "vehicle",
    "person": "person",
    "backpack": "bag",
    "handbag": "bag",
    "suitcase": "bag",
}


def _load_required_json(path: Path) -> list[dict[str, Any]] | dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Step 11B input file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _round6(value: float) -> float:
    return round(float(value), 6)


def _class_group(class_name: str) -> str | None:
    return CLASS_GROUPS.get(str(class_name or "").strip().lower())


def _iou(bbox_a: list[float], bbox_b: list[float]) -> float:
    if len(bbox_a) != 4 or len(bbox_b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def _prepare_detection(frame_item: dict[str, Any], detection: dict[str, Any]) -> dict[str, Any] | None:
    bbox = detection.get("bbox_xyxy") or detection.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [_safe_float(value) for value in bbox]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    bbox_diag = math.sqrt((width * width) + (height * height))
    class_name = str(detection.get("class_name", "")).strip().lower()
    object_group = _class_group(class_name)
    if not object_group:
        return None
    center = detection.get("bbox_center", [])
    if isinstance(center, list) and len(center) == 2:
        centroid_x, centroid_y = _safe_float(center[0]), _safe_float(center[1])
    else:
        centroid_x = x1 + (width / 2.0)
        centroid_y = y1 + (height / 2.0)
    return {
        "class_name": class_name,
        "object_group": object_group,
        "confidence": _safe_float(detection.get("confidence"), 0.0),
        "bbox_xyxy": [x1, y1, x2, y2],
        "bbox_width": width,
        "bbox_height": height,
        "bbox_diag": bbox_diag,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "frame_idx": frame_item.get("frame_idx"),
        "timestamp_seconds": _safe_float(frame_item.get("timestamp_seconds"), 0.0),
    }


def _match_frame_pair(previous_frame: dict[str, Any], current_frame: dict[str, Any]) -> list[dict[str, Any]]:
    previous_detections = [
        prepared
        for prepared in (_prepare_detection(previous_frame, det) for det in previous_frame.get("detections", []) or [])
        if prepared is not None
    ]
    current_detections = [
        prepared
        for prepared in (_prepare_detection(current_frame, det) for det in current_frame.get("detections", []) or [])
        if prepared is not None
    ]
    matches: list[dict[str, Any]] = []
    used_current: set[int] = set()

    for previous_index, previous_detection in enumerate(previous_detections):
        best_match: tuple[int, float, dict[str, Any]] | None = None
        for current_index, current_detection in enumerate(current_detections):
            if current_index in used_current:
                continue
            if current_detection["object_group"] != previous_detection["object_group"]:
                continue
            prev_diag = max(previous_detection["bbox_diag"], 1.0)
            dx = current_detection["centroid_x"] - previous_detection["centroid_x"]
            dy = current_detection["centroid_y"] - previous_detection["centroid_y"]
            pixel_displacement = math.sqrt((dx * dx) + (dy * dy))
            normalized_displacement = pixel_displacement / prev_diag
            prev_area = max(previous_detection["bbox_width"] * previous_detection["bbox_height"], 1.0)
            curr_area = max(current_detection["bbox_width"] * current_detection["bbox_height"], 1.0)
            size_similarity = min(prev_area, curr_area) / max(prev_area, curr_area)
            overlap_iou = _iou(previous_detection["bbox_xyxy"], current_detection["bbox_xyxy"])
            score = normalized_displacement - (0.75 * overlap_iou) + ((1.0 - size_similarity) * 0.5)
            if size_similarity < 0.45:
                continue
            if normalized_displacement > 4.0 and overlap_iou <= 0.0:
                continue
            if best_match is None or score < best_match[1]:
                best_match = (current_index, score, current_detection)

        if best_match is None:
            continue

        current_index, _, current_detection = best_match
        used_current.add(current_index)
        dx = current_detection["centroid_x"] - previous_detection["centroid_x"]
        dy = current_detection["centroid_y"] - previous_detection["centroid_y"]
        pixel_displacement = math.sqrt((dx * dx) + (dy * dy))
        normalized_displacement = pixel_displacement / max(previous_detection["bbox_diag"], 1.0)
        time_delta = max(current_detection["timestamp_seconds"] - previous_detection["timestamp_seconds"], 0.001)
        speed_pixels_per_second = pixel_displacement / time_delta

        if abs(dx) > abs(dy):
            direction = "left_to_right" if dx > 0 else "right_to_left"
        else:
            direction = "top_to_bottom" if dy > 0 else "bottom_to_top"

        object_group = previous_detection["object_group"]
        if object_group == "person":
            if normalized_displacement >= 0.12 or speed_pixels_per_second >= 15.0:
                motion_state = "walking_or_moving"
            elif normalized_displacement <= 0.05:
                motion_state = "standing_or_stationary"
            else:
                motion_state = "motion_unclear"
        elif object_group == "vehicle":
            if normalized_displacement >= 0.10 or speed_pixels_per_second >= 20.0:
                motion_state = "moving"
            elif normalized_displacement <= 0.04:
                motion_state = "stationary_or_parked"
            else:
                motion_state = "motion_unclear"
        else:
            if normalized_displacement >= 0.08:
                motion_state = "moving"
            elif normalized_displacement <= 0.04:
                motion_state = "stationary"
            else:
                motion_state = "motion_unclear"

        matches.append(
            {
                "class_name": previous_detection["class_name"],
                "object_group": object_group,
                "motion_state": motion_state,
                "direction": direction,
                "previous_frame_idx": previous_detection["frame_idx"],
                "current_frame_idx": current_detection["frame_idx"],
                "previous_time": previous_detection["timestamp_seconds"],
                "current_time": current_detection["timestamp_seconds"],
                "avg_confidence": _round6((previous_detection["confidence"] + current_detection["confidence"]) / 2.0),
                "pixel_displacement": _round6(pixel_displacement),
                "normalized_displacement": _round6(normalized_displacement),
                "speed_pixels_per_second": _round6(speed_pixels_per_second),
            }
        )
    return matches


def _motion_confidence(comparison_count: int, avg_norm: float, avg_speed: float) -> str:
    if comparison_count >= 3 and (avg_norm >= 0.18 or avg_speed >= 35.0):
        return "high"
    if comparison_count >= 2 and (avg_norm >= 0.08 or avg_speed >= 15.0):
        return "medium"
    return "low"


def _aggregate_motion_bucket(class_name: str, object_group: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    comparison_count = len(items)
    avg_norm = sum(_safe_float(item.get("normalized_displacement"), 0.0) for item in items) / max(comparison_count, 1)
    avg_speed = sum(_safe_float(item.get("speed_pixels_per_second"), 0.0) for item in items) / max(comparison_count, 1)
    direction_counts: dict[str, int] = {}
    for item in items:
        direction = str(item.get("direction", "unknown"))
        direction_counts[direction] = direction_counts.get(direction, 0) + 1
    direction = sorted(direction_counts.items(), key=lambda pair: (-pair[1], pair[0]))[0][0] if direction_counts else "unknown"
    confidence = _motion_confidence(comparison_count, avg_norm, avg_speed)
    evidence_frames = sorted(
        {
            _safe_int(item.get("previous_frame_idx"), 0)
            for item in items
        }
        | {
            _safe_int(item.get("current_frame_idx"), 0)
            for item in items
        }
    )
    evidence_times = sorted(
        {
            _round6(_safe_float(item.get("previous_time"), 0.0))
            for item in items
        }
        | {
            _round6(_safe_float(item.get("current_time"), 0.0))
            for item in items
        }
    )
    motion_state = str(items[0].get("motion_state", "motion_unclear"))
    return {
        "class_name": class_name,
        "object_group": object_group,
        "motion_state": motion_state,
        "direction": direction,
        "confidence": confidence,
        "comparison_count": comparison_count,
        "evidence_frames": evidence_frames[:6],
        "evidence_times": evidence_times[:6],
        "avg_normalized_displacement": _round6(avg_norm),
        "avg_speed_pixels_per_second": _round6(avg_speed),
    }


def _clip_motion_summary(objects_in_motion: list[dict[str, Any]], stationary_objects: list[dict[str, Any]]) -> str:
    motion_parts: list[str] = []
    for item in objects_in_motion[:3]:
        class_name = str(item.get("class_name", "object"))
        direction = str(item.get("direction", "")).replace("_", " ")
        if direction and direction != "unknown":
            motion_parts.append(f"a {class_name} appears to be moving {direction}")
        else:
            motion_parts.append(f"a {class_name} appears to be moving")
    for item in stationary_objects[:2]:
        class_name = str(item.get("class_name", "object"))
        motion_parts.append(f"a {class_name} appears stationary")
    if not motion_parts:
        return "Object motion is unclear in this clip."
    if len(motion_parts) == 1:
        return motion_parts[0].capitalize() + "."
    return (", ".join(motion_parts[:-1]) + " while " + motion_parts[-1]).capitalize() + "."


def _clip_flags(objects_in_motion: list[dict[str, Any]], stationary_objects: list[dict[str, Any]]) -> dict[str, bool]:
    moving_vehicle = any(item.get("object_group") == "vehicle" for item in objects_in_motion)
    moving_person = any(item.get("object_group") == "person" for item in objects_in_motion)
    stationary_vehicle = any(item.get("object_group") == "vehicle" for item in stationary_objects)
    unclear_motion = not objects_in_motion and not stationary_objects
    return {
        "has_moving_vehicle": moving_vehicle,
        "has_moving_person": moving_person,
        "has_stationary_vehicle": stationary_vehicle,
        "has_unclear_motion": unclear_motion,
    }


def estimate_object_motion_states(run_dir: Path) -> dict[str, Any]:
    print("[tender-demo] Starting Step 11B: object motion state estimation")

    yolo_detections = _load_required_json(run_dir / "10_yolo_detections.json")
    sampled_frames = _load_required_json(run_dir / "02_sampled_frames.json")
    clip_path = run_dir / "06_expanded_clips.json"
    if not clip_path.exists():
        clip_path = run_dir / "05_candidate_clips.json"
    clip_items = _load_required_json(clip_path)
    video_info = _load_required_json(run_dir / "01_video_info.json") if (run_dir / "01_video_info.json").exists() else {}

    if not isinstance(yolo_detections, list):
        raise ValueError("Expected a list in 10_yolo_detections.json")
    if not isinstance(sampled_frames, list):
        raise ValueError("Expected a list in 02_sampled_frames.json")
    if not isinstance(clip_items, list):
        raise ValueError(f"Expected a list in {clip_path.name}")
    if not isinstance(video_info, dict):
        video_info = {}

    detections_by_frame_idx = {
        _safe_int(item.get("frame_idx"), -1): item
        for item in yolo_detections
        if isinstance(item, dict) and item.get("frame_idx") is not None
    }
    ordered_frames = [
        detections_by_frame_idx[_safe_int(frame.get("frame_idx"), -1)]
        for frame in sorted(sampled_frames, key=lambda entry: _safe_float(entry.get("timestamp_seconds"), 0.0))
        if isinstance(frame, dict) and _safe_int(frame.get("frame_idx"), -1) in detections_by_frame_idx
    ]

    pair_matches: list[dict[str, Any]] = []
    for previous_frame, current_frame in zip(ordered_frames, ordered_frames[1:]):
        pair_matches.extend(_match_frame_pair(previous_frame, current_frame))

    clip_motion_states: list[dict[str, Any]] = []
    clips_with_moving_vehicle = 0
    clips_with_stationary_vehicle = 0
    clips_with_moving_person = 0
    clips_with_unclear_motion = 0
    motion_state_counts: dict[str, int] = {}

    for clip in clip_items:
        if not isinstance(clip, dict):
            continue
        clip_id = str(clip.get("clip_id", "")).strip()
        start_time = _safe_float(clip.get("expanded_start_time", clip.get("start_time")), 0.0)
        end_time = _safe_float(clip.get("expanded_end_time", clip.get("end_time")), start_time)
        clip_matches = [
            item
            for item in pair_matches
            if start_time <= _safe_float(item.get("previous_time"), -1.0) <= end_time
            or start_time <= _safe_float(item.get("current_time"), -1.0) <= end_time
        ]

        grouped_matches: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for match in clip_matches:
            key = (
                str(match.get("class_name", "")),
                str(match.get("object_group", "")),
                str(match.get("motion_state", "")),
            )
            grouped_matches.setdefault(key, []).append(match)

        moving_objects: list[dict[str, Any]] = []
        stationary_objects: list[dict[str, Any]] = []
        for (class_name, object_group, motion_state), items in grouped_matches.items():
            aggregate = _aggregate_motion_bucket(class_name, object_group, items)
            motion_state_counts[motion_state] = motion_state_counts.get(motion_state, 0) + 1
            if motion_state in {"moving", "walking_or_moving"}:
                moving_objects.append(aggregate)
            elif motion_state in {"standing_or_stationary", "stationary", "stationary_or_parked"}:
                if object_group == "vehicle":
                    strong_stationary = aggregate["comparison_count"] >= 2 or (
                        aggregate["avg_normalized_displacement"] <= 0.02 and aggregate["avg_speed_pixels_per_second"] <= 5.0
                    )
                    if not strong_stationary:
                        aggregate["motion_state"] = "motion_unclear"
                        motion_state_counts["motion_unclear"] = motion_state_counts.get("motion_unclear", 0) + 1
                        continue
                stationary_objects.append(aggregate)

        moving_objects.sort(key=lambda item: (-_safe_float(item.get("avg_normalized_displacement"), 0.0), item.get("class_name", "")))
        stationary_objects.sort(key=lambda item: (-_safe_int(item.get("comparison_count"), 0), item.get("class_name", "")))
        flags = _clip_flags(moving_objects, stationary_objects)

        clip_motion_confidence = "low"
        if moving_objects:
            clip_motion_confidence = sorted(
                [str(item.get("confidence", "low")) for item in moving_objects],
                key=lambda value: {"high": 2, "medium": 1, "low": 0}.get(value, 0),
                reverse=True,
            )[0]
        elif stationary_objects:
            clip_motion_confidence = sorted(
                [str(item.get("confidence", "low")) for item in stationary_objects],
                key=lambda value: {"high": 2, "medium": 1, "low": 0}.get(value, 0),
                reverse=True,
            )[0]

        clip_motion_state = {
            "clip_id": clip_id,
            "start_time": _safe_float(clip.get("start_time"), start_time),
            "end_time": _safe_float(clip.get("end_time"), end_time),
            "expanded_start_time": start_time,
            "expanded_end_time": end_time,
            "objects_in_motion": moving_objects,
            "stationary_objects": stationary_objects,
            "motion_summary": _clip_motion_summary(moving_objects, stationary_objects),
            "motion_confidence": clip_motion_confidence,
            **flags,
        }
        if flags["has_moving_vehicle"]:
            clips_with_moving_vehicle += 1
        if flags["has_stationary_vehicle"]:
            clips_with_stationary_vehicle += 1
        if flags["has_moving_person"]:
            clips_with_moving_person += 1
        if flags["has_unclear_motion"]:
            clips_with_unclear_motion += 1
        clip_motion_states.append(clip_motion_state)

    motion_output = {
        "video_name": video_info.get("video_name"),
        "total_frame_pairs_analyzed": max(0, len(ordered_frames) - 1),
        "clip_motion_states": clip_motion_states,
    }
    report_output = {
        "video_name": video_info.get("video_name"),
        "clips_with_moving_vehicle": clips_with_moving_vehicle,
        "clips_with_stationary_vehicle": clips_with_stationary_vehicle,
        "clips_with_moving_person": clips_with_moving_person,
        "clips_with_unclear_motion": clips_with_unclear_motion,
        "most_common_motion_states": [
            {"motion_state": key, "count": value}
            for key, value in sorted(motion_state_counts.items(), key=lambda pair: (-pair[1], pair[0]))
        ],
    }

    output_path = run_dir / "11b_object_motion_states.json"
    report_path = run_dir / "11b_object_motion_state_report.json"
    output_path.write_text(json.dumps(motion_output, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report_output, indent=2), encoding="utf-8")

    print(f"[tender-demo] Total frame pairs analyzed: {motion_output['total_frame_pairs_analyzed']}")
    print(f"[tender-demo] Clips with moving vehicle: {clips_with_moving_vehicle}")
    print(f"[tender-demo] Clips with stationary vehicle: {clips_with_stationary_vehicle}")
    print(f"[tender-demo] Clips with moving person: {clips_with_moving_person}")
    print(f"[tender-demo] Clips with unclear motion: {clips_with_unclear_motion}")
    print(f"[tender-demo] Object motion states output path: {output_path}")
    print(f"[tender-demo] Object motion state report path: {report_path}")
    return {
        "motion_output": motion_output,
        "report_output": report_output,
        "output_path": str(output_path),
        "report_path": str(report_path),
    }
