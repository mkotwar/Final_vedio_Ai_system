from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

from stage_checks import read_json, write_json


VEHICLE_CLASS_GROUP = {"car", "motorcycle", "bus", "truck", "bicycle", "auto", "van", "vehicle"}
PERSON_CLASS_GROUP = {"person"}


@dataclass
class TrackState:
    """Mutable deterministic offline track state."""

    track_id: str
    track_type: str
    detections: list[dict[str, Any]] = field(default_factory=list)
    class_counts: Counter[str] = field(default_factory=Counter)
    confidence_values: list[float] = field(default_factory=list)
    bbox_area_ratio_values: list[float] = field(default_factory=list)
    time_gaps: list[float] = field(default_factory=list)
    center_distance_ratios: list[float] = field(default_factory=list)
    class_switches: list[dict[str, Any]] = field(default_factory=list)
    last_bbox_xyxy: list[float] = field(default_factory=list)
    last_timestamp_seconds: float = 0.0
    last_detection_id: str = ""
    last_class_name: str = ""


def _read_frame_dimensions(run_dir: Path) -> tuple[int, int]:
    """Read width and height from step 01 metadata."""

    video_info = read_json(run_dir / "01_video_info.json")
    return int(video_info.get("width", 0) or 0), int(video_info.get("height", 0) or 0)


def _safe_stats(values: list[float]) -> dict[str, float]:
    """Return min/max/avg stats with zero defaults."""

    if not values:
        return {"min": 0.0, "max": 0.0, "avg": 0.0}
    return {
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "avg": round(sum(values) / len(values), 6),
    }


def _class_group(class_name: str) -> str | None:
    """Map a class name into tracking groups."""

    normalized = class_name.lower()
    if normalized in PERSON_CLASS_GROUP:
        return "person"
    if normalized in VEHICLE_CLASS_GROUP:
        return "vehicle"
    return None


def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute bbox IoU for xyxy boxes."""

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def _bbox_center(box: list[float]) -> tuple[float, float]:
    """Return bbox center."""

    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_area(box: list[float]) -> float:
    """Return bbox area."""

    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _center_distance_ratio(box_a: list[float], box_b: list[float], image_diagonal: float) -> float:
    """Return normalized center distance against image diagonal."""

    ax, ay = _bbox_center(box_a)
    bx, by = _bbox_center(box_b)
    distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    return distance / image_diagonal if image_diagonal > 0 else 0.0


def _area_change_ratio(box_a: list[float], box_b: list[float]) -> float:
    """Return symmetric area change ratio."""

    area_a = _bbox_area(box_a)
    area_b = _bbox_area(box_b)
    if area_a <= 0 or area_b <= 0:
        return float("inf")
    return max(area_a, area_b) / min(area_a, area_b)


def _area_similarity_score(box_a: list[float], box_b: list[float]) -> float:
    """Convert area difference into a 0..1 similarity score."""

    ratio = _area_change_ratio(box_a, box_b)
    if ratio == float("inf"):
        return 0.0
    return max(0.0, min(1.0, 1.0 - ((ratio - 1.0) / ratio)))


def _dominant_class(track: TrackState) -> str:
    """Return current dominant class."""

    return track.class_counts.most_common(1)[0][0] if track.class_counts else track.track_type


def _can_allow_class_switch(track_class: str, detection_class: str, tracking_config: dict[str, Any]) -> bool:
    """Decide whether different vehicle classes can match strongly."""

    if track_class == detection_class:
        return True
    if not bool(tracking_config["allow_vehicle_class_switch"]):
        return False
    # Keep motorcycles isolated unless an explicit future setting is added.
    if "motorcycle" in {track_class, detection_class}:
        return False
    if track_class in VEHICLE_CLASS_GROUP and detection_class in VEHICLE_CLASS_GROUP:
        return True
    return False


def _class_match_kind(track_class: str, detection_class: str, tracking_config: dict[str, Any]) -> str | None:
    """Return exact/same_group/none for class compatibility."""

    if track_class == detection_class:
        return "exact"
    if _can_allow_class_switch(track_class, detection_class, tracking_config):
        return "same_group"
    return None


def _compute_match_score(
    *,
    track: TrackState,
    detection: dict[str, Any],
    image_diagonal: float,
    tracking_config: dict[str, Any],
) -> tuple[bool, float]:
    """Compute stricter deterministic tracking match score."""

    detection_class = str(detection["class_name"]).lower()
    detection_group = _class_group(detection_class)
    if detection_group is None or detection_group != track.track_type:
        return False, 0.0

    previous_box = track.last_bbox_xyxy
    current_box = detection["bbox_xyxy"]
    time_gap = float(detection["timestamp_seconds"]) - float(track.last_timestamp_seconds)
    if time_gap > float(tracking_config["max_time_gap_seconds"]):
        return False, 0.0

    iou_score = _bbox_iou(previous_box, current_box)
    center_distance_ratio = _center_distance_ratio(previous_box, current_box, image_diagonal)
    area_change_ratio = _area_change_ratio(previous_box, current_box)
    if center_distance_ratio > float(tracking_config["max_center_distance_ratio"]):
        return False, 0.0
    if area_change_ratio > float(tracking_config["max_area_change_ratio"]):
        return False, 0.0

    dominant_class = _dominant_class(track)
    match_kind = _class_match_kind(dominant_class, detection_class, tracking_config)
    if match_kind is None:
        return False, 0.0

    if match_kind == "same_group":
        if iou_score < float(tracking_config["class_switch_min_iou"]):
            return False, 0.0
        if center_distance_ratio > float(tracking_config["class_switch_max_center_distance_ratio"]):
            return False, 0.0
        if time_gap > float(tracking_config["class_switch_max_time_gap_seconds"]):
            return False, 0.0
        if area_change_ratio > 1.8:
            return False, 0.0

    if iou_score < float(tracking_config["min_iou"]) and center_distance_ratio > (float(tracking_config["max_center_distance_ratio"]) * 0.35):
        return False, 0.0

    center_distance_score = max(0.0, 1.0 - center_distance_ratio)
    area_similarity_score = _area_similarity_score(previous_box, current_box)
    confidence_score = float(detection["confidence"])

    if match_kind == "exact":
        score = (
            0.50 * iou_score
            + 0.25 * center_distance_score
            + 0.15 * area_similarity_score
            + 0.10 * confidence_score
        )
    else:
        score = (
            0.65 * iou_score
            + 0.20 * center_distance_score
            + 0.10 * area_similarity_score
            + 0.05 * confidence_score
        )
    return True, round(score, 6)


def _border_metrics(box: list[float], image_width: int, image_height: int) -> tuple[bool, float]:
    """Return whether a box touches the border and how strongly."""

    x1, y1, x2, y2 = box
    min_dimension = max(1.0, float(min(image_width, image_height)))
    nearest_border = min(x1, y1, max(0.0, image_width - x2), max(0.0, image_height - y2))
    normalized_margin = nearest_border / (0.05 * min_dimension)
    border_touch_ratio = round(max(0.0, min(1.0, 1.0 - normalized_margin)), 6)
    border_touching = border_touch_ratio > 0.0
    return border_touching, border_touch_ratio


def _track_quality(track: TrackState, min_track_length: int, class_consistency_ratio: float) -> tuple[str, float]:
    """Assign refined track quality categories."""

    detection_count = len(track.detections)
    start_time = float(track.detections[0]["timestamp_seconds"])
    end_time = float(track.detections[-1]["timestamp_seconds"])
    duration_seconds = max(0.0, end_time - start_time)
    avg_confidence = sum(track.confidence_values) / len(track.confidence_values) if track.confidence_values else 0.0
    max_gap = max(track.time_gaps) if track.time_gaps else 0.0
    avg_center_jump = sum(track.center_distance_ratios) / len(track.center_distance_ratios) if track.center_distance_ratios else 0.0

    if detection_count == 1:
        return "single_frame", 0.1
    if detection_count < min_track_length:
        return "short", 0.2
    if avg_confidence < 0.35:
        return "weak", 0.3
    if class_consistency_ratio < 0.75:
        return "class_mixed", 0.35
    if max_gap > 1.5 or avg_center_jump > 0.18 or (duration_seconds < 0.8 and detection_count >= 4):
        return "fragmented", 0.45
    if detection_count >= 4 and duration_seconds >= 1.0 and avg_confidence >= 0.4 and class_consistency_ratio >= 0.85:
        return "good", 0.9
    return "fragmented", 0.5


def _best_detection(track: TrackState, image_width: int, image_height: int) -> dict[str, Any]:
    """Choose the best detection for later best-frame selection and OCR/color."""

    dominant_class_name = _dominant_class(track)

    for detection in track.detections:
        border_touching, border_touch_ratio = _border_metrics(detection["bbox_xyxy"], image_width, image_height)
        detection["border_touching"] = border_touching
        detection["border_touch_ratio"] = border_touch_ratio
        bbox_area_score = min(1.0, float(detection["bbox_area_ratio"]) / 0.20)
        class_consistency_bonus = 1.0 if detection["class_name"] == dominant_class_name else 0.0
        not_border_touching_bonus = 0.0 if border_touching else 1.0
        detection["best_frame_score"] = round(
            0.45 * float(detection["confidence"])
            + 0.35 * bbox_area_score
            + 0.10 * class_consistency_bonus
            + 0.10 * not_border_touching_bonus,
            6,
        )

    return max(track.detections, key=lambda item: float(item["best_frame_score"]))


def _draw_preview(
    *,
    run_dir: Path,
    preview_dir: Path,
    frame_item: dict[str, Any],
    tracked_detections: list[dict[str, Any]],
) -> bool:
    """Render simple preview frames with track IDs."""

    image_path = Path(str(frame_item.get("image_path", "")))
    absolute_image_path = image_path if image_path.is_absolute() else (run_dir / image_path).resolve()
    if not absolute_image_path.exists():
        return False
    image = cv2.imread(str(absolute_image_path))
    if image is None:
        return False

    for detection in tracked_detections:
        x1, y1, x2, y2 = [int(round(float(value))) for value in detection["bbox_xyxy"]]
        track_id = str(detection["track_id"])
        color_seed = sum(ord(char) for char in track_id)
        color = (
            50 + (color_seed * 29) % 205,
            50 + (color_seed * 47) % 205,
            50 + (color_seed * 61) % 205,
        )
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            track_id,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    preview_output_path = preview_dir / f"{frame_item['frame_id']}.jpg"
    return bool(cv2.imwrite(str(preview_output_path), image))


def run_tracking(
    *,
    run_dir: Path,
    tracking_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Create stricter deterministic offline tracks from YOLO detections."""

    yolo_payload = read_json(run_dir / "03_yolo_detections.json")
    frame_items = list(yolo_payload.get("detections", []))
    image_width, image_height = _read_frame_dimensions(run_dir)
    image_diagonal = (image_width**2 + image_height**2) ** 0.5

    tracking_classes = {item.lower() for item in tracking_config["tracking_classes"]}
    min_confidence = float(tracking_config["min_confidence"])
    min_person_confidence = float(tracking_config["min_person_confidence"])
    min_vehicle_confidence = float(tracking_config["min_vehicle_confidence"])
    min_track_length = int(tracking_config["min_track_length"])
    save_preview = bool(tracking_config["save_preview"])
    preview_limit = int(tracking_config["preview_limit"])

    tracks: list[TrackState] = []
    track_counters = {"person": 0, "vehicle": 0}
    assignments: list[dict[str, Any]] = []
    preview_dir = run_dir / "04B_tracking_preview_frames"
    if save_preview:
        preview_dir.mkdir(parents=True, exist_ok=True)

    detections_total_from_yolo = 0
    detections_considered = 0
    detections_ignored_by_class = 0
    detections_ignored_by_confidence = 0
    preview_written = 0
    same_frame_multi_assignment_prevented_count = 0

    for frame_item in frame_items:
        raw_detections = list(frame_item.get("detections", []))
        detections_total_from_yolo += len(raw_detections)
        filtered_detections: list[dict[str, Any]] = []

        for detection in raw_detections:
            class_name = str(detection.get("class_name", "")).lower()
            confidence = float(detection.get("confidence", 0.0) or 0.0)
            bbox_xyxy = [float(value) for value in list(detection.get("bbox_xyxy", []))]
            if class_name not in tracking_classes:
                detections_ignored_by_class += 1
                continue

            class_group = _class_group(class_name)
            class_specific_threshold = min_confidence
            if class_group == "person":
                class_specific_threshold = max(min_confidence, min_person_confidence)
            elif class_group == "vehicle":
                class_specific_threshold = max(min_confidence, min_vehicle_confidence)
            if confidence < class_specific_threshold:
                detections_ignored_by_confidence += 1
                continue
            if len(bbox_xyxy) != 4 or bbox_xyxy[2] <= bbox_xyxy[0] or bbox_xyxy[3] <= bbox_xyxy[1]:
                continue

            crop_path_value = str(detection.get("crop_path", ""))
            crop_path = (run_dir / Path(crop_path_value)).resolve() if crop_path_value else None
            filtered_detections.append(
                {
                    "frame_id": str(frame_item.get("frame_id", "")),
                    "frame_idx": int(frame_item.get("frame_idx", 0) or 0),
                    "timestamp_seconds": float(frame_item.get("timestamp_seconds", 0.0) or 0.0),
                    "timestamp_text": str(frame_item.get("timestamp_text", "")),
                    "image_path": str(frame_item.get("image_path", "")),
                    "detection_id": str(detection.get("detection_id", "")),
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox_xyxy": bbox_xyxy,
                    "bbox_area_ratio": float(detection.get("bbox_area_ratio", 0.0) or 0.0),
                    "crop_path": crop_path_value,
                    "crop_exists": bool(crop_path and crop_path.exists()),
                    "absolute_crop_path": crop_path,
                }
            )

        filtered_detections.sort(
            key=lambda item: (float(item["confidence"]), float(item["bbox_area_ratio"])),
            reverse=True,
        )
        detections_considered += len(filtered_detections)
        tracked_preview_detections: list[dict[str, Any]] = []
        assigned_track_ids_in_frame: set[str] = set()

        # Adaptive sampling creates variable time gaps, so matching must allow sparse timestamps while keeping one assignment per frame.
        for detection in filtered_detections:
            best_track: TrackState | None = None
            best_score = -1.0

            for track in tracks:
                if track.track_id in assigned_track_ids_in_frame:
                    is_match_allowed, _ = _compute_match_score(
                        track=track,
                        detection=detection,
                        image_diagonal=image_diagonal,
                        tracking_config=tracking_config,
                    )
                    if is_match_allowed:
                        same_frame_multi_assignment_prevented_count += 1
                    continue

                is_match_allowed, score = _compute_match_score(
                    track=track,
                    detection=detection,
                    image_diagonal=image_diagonal,
                    tracking_config=tracking_config,
                )
                if is_match_allowed and score > best_score:
                    best_track = track
                    best_score = score

            if best_track is None:
                track_type = _class_group(detection["class_name"])
                if track_type is None:
                    continue
                track_counters[track_type] += 1
                track_id = f"{track_type}_track_{track_counters[track_type]:04d}"
                best_track = TrackState(
                    track_id=track_id,
                    track_type=track_type,
                    last_bbox_xyxy=list(detection["bbox_xyxy"]),
                    last_timestamp_seconds=float(detection["timestamp_seconds"]),
                    last_detection_id=str(detection["detection_id"]),
                    last_class_name=str(detection["class_name"]),
                )
                tracks.append(best_track)
                match_status = "new_track"
                match_score = 0.0
                matched_from_previous_track_detection_id = None
            else:
                match_status = "matched"
                match_score = best_score
                matched_from_previous_track_detection_id = best_track.last_detection_id
                time_gap = float(detection["timestamp_seconds"]) - best_track.last_timestamp_seconds
                if time_gap > 0:
                    best_track.time_gaps.append(time_gap)
                best_track.center_distance_ratios.append(
                    _center_distance_ratio(best_track.last_bbox_xyxy, detection["bbox_xyxy"], image_diagonal)
                )
                if best_track.last_class_name and best_track.last_class_name != detection["class_name"]:
                    best_track.class_switches.append(
                        {
                            "from_class": best_track.last_class_name,
                            "to_class": detection["class_name"],
                            "timestamp_seconds": detection["timestamp_seconds"],
                            "detection_id": detection["detection_id"],
                        }
                    )

            detection_record = {
                "frame_id": detection["frame_id"],
                "frame_idx": detection["frame_idx"],
                "timestamp_seconds": detection["timestamp_seconds"],
                "detection_id": detection["detection_id"],
                "class_name": detection["class_name"],
                "confidence": round(detection["confidence"], 6),
                "bbox_xyxy": [round(value, 3) for value in detection["bbox_xyxy"]],
                "bbox_area_ratio": round(detection["bbox_area_ratio"], 6),
                "crop_path": detection["crop_path"],
                "crop_exists": detection["crop_exists"],
                "match_score": round(match_score, 6),
                "matched_from_previous_track_detection_id": matched_from_previous_track_detection_id,
            }
            best_track.detections.append(detection_record)
            best_track.class_counts[detection["class_name"]] += 1
            best_track.confidence_values.append(detection["confidence"])
            best_track.bbox_area_ratio_values.append(detection["bbox_area_ratio"])
            best_track.last_bbox_xyxy = list(detection["bbox_xyxy"])
            best_track.last_timestamp_seconds = float(detection["timestamp_seconds"])
            best_track.last_detection_id = detection["detection_id"]
            best_track.last_class_name = detection["class_name"]
            assigned_track_ids_in_frame.add(best_track.track_id)

            assignments.append(
                {
                    "detection_id": detection["detection_id"],
                    "frame_id": detection["frame_id"],
                    "timestamp_seconds": detection["timestamp_seconds"],
                    "class_name": detection["class_name"],
                    "track_id": best_track.track_id,
                    "match_status": match_status,
                    "match_score": round(match_score, 6),
                }
            )
            tracked_preview_detections.append({"track_id": best_track.track_id, "bbox_xyxy": detection["bbox_xyxy"]})

        if save_preview and tracked_preview_detections and preview_written < preview_limit:
            if _draw_preview(run_dir=run_dir, preview_dir=preview_dir, frame_item=frame_item, tracked_detections=tracked_preview_detections):
                preview_written += 1

    track_payloads: list[dict[str, Any]] = []
    track_type_counts: Counter[str] = Counter()
    dominant_class_track_counts: Counter[str] = Counter()
    track_quality_counts: Counter[str] = Counter()
    unusable_reason_counts: Counter[str] = Counter()
    track_lengths: list[float] = []
    track_durations: list[float] = []
    usable_vehicle_tracks_for_ocr_color = 0
    usable_person_tracks = 0
    unusable_tracks = 0

    for track in tracks:
        dominant_class_name = _dominant_class(track)
        detection_count = len(track.detections)
        start_timestamp_seconds = float(track.detections[0]["timestamp_seconds"])
        end_timestamp_seconds = float(track.detections[-1]["timestamp_seconds"])
        duration_seconds = round(max(0.0, end_timestamp_seconds - start_timestamp_seconds), 6)
        avg_confidence = round(sum(track.confidence_values) / len(track.confidence_values), 6) if track.confidence_values else 0.0
        max_confidence = round(max(track.confidence_values), 6) if track.confidence_values else 0.0
        avg_bbox_area_ratio = round(sum(track.bbox_area_ratio_values) / len(track.bbox_area_ratio_values), 6) if track.bbox_area_ratio_values else 0.0
        max_bbox_area_ratio = round(max(track.bbox_area_ratio_values), 6) if track.bbox_area_ratio_values else 0.0
        class_consistency_ratio = round(track.class_counts[dominant_class_name] / detection_count, 6) if detection_count > 0 else 0.0
        class_switch_count = len(track.class_switches)
        track_quality, track_quality_score = _track_quality(track, min_track_length, class_consistency_ratio)
        best_detection = _best_detection(track, image_width, image_height)

        track_type_counts[track.track_type] += 1
        dominant_class_track_counts[dominant_class_name] += 1
        track_quality_counts[track_quality] += 1
        track_lengths.append(float(detection_count))
        track_durations.append(duration_seconds)

        usable_for_next_step = bool(
            track.track_type == "vehicle"
            and track_quality == "good"
            and detection_count >= 4
            and avg_confidence >= 0.4
            and class_consistency_ratio >= 0.85
            and bool(best_detection.get("crop_exists"))
        )
        if usable_for_next_step:
            usable_vehicle_tracks_for_ocr_color += 1
        elif track.track_type == "person" and track_quality == "good":
            usable_person_tracks += 1
        else:
            unusable_tracks += 1
            unusable_reason_counts[track_quality] += 1

        track_payloads.append(
            {
                "track_id": track.track_id,
                "track_type": track.track_type,
                "dominant_class_name": dominant_class_name,
                "class_counts": dict(sorted(track.class_counts.items())),
                "class_consistency_ratio": class_consistency_ratio,
                "class_switch_count": class_switch_count,
                "class_switches": track.class_switches,
                "start_timestamp_seconds": start_timestamp_seconds,
                "end_timestamp_seconds": end_timestamp_seconds,
                "duration_seconds": duration_seconds,
                "first_frame_id": track.detections[0]["frame_id"],
                "last_frame_id": track.detections[-1]["frame_id"],
                "detection_count": detection_count,
                "avg_confidence": avg_confidence,
                "max_confidence": max_confidence,
                "avg_bbox_area_ratio": avg_bbox_area_ratio,
                "max_bbox_area_ratio": max_bbox_area_ratio,
                "track_quality": track_quality,
                "track_quality_score": track_quality_score,
                "best_detection_id": best_detection["detection_id"],
                "best_crop_path": best_detection["crop_path"],
                "usable_for_next_step": usable_for_next_step,
                "detections": track.detections,
            }
        )

    track_payloads.sort(key=lambda item: item["start_timestamp_seconds"])
    usable_tracks_for_next_step = [item["track_id"] for item in track_payloads if item["usable_for_next_step"]]

    assignments_payload = {"status": "success", "assignments": assignments}
    tracks_payload = {
        "status": "success",
        "input_yolo_detections_file": "03_yolo_detections.json",
        "tracking_config": {
            "enabled": bool(tracking_config["enabled"]),
            "tracking_classes": sorted(tracking_classes),
            "min_confidence": float(tracking_config["min_confidence"]),
            "min_person_confidence": float(tracking_config["min_person_confidence"]),
            "min_vehicle_confidence": float(tracking_config["min_vehicle_confidence"]),
            "min_iou": float(tracking_config["min_iou"]),
            "max_time_gap_seconds": float(tracking_config["max_time_gap_seconds"]),
            "max_center_distance_ratio": float(tracking_config["max_center_distance_ratio"]),
            "max_area_change_ratio": float(tracking_config["max_area_change_ratio"]),
            "allow_vehicle_class_switch": bool(tracking_config["allow_vehicle_class_switch"]),
            "class_switch_min_iou": float(tracking_config["class_switch_min_iou"]),
            "class_switch_max_center_distance_ratio": float(tracking_config["class_switch_max_center_distance_ratio"]),
            "class_switch_max_time_gap_seconds": float(tracking_config["class_switch_max_time_gap_seconds"]),
            "min_track_length": min_track_length,
            "save_preview": bool(tracking_config["save_preview"]),
            "preview_limit": int(tracking_config["preview_limit"]),
        },
        "frames_processed": len(frame_items),
        "detections_considered": detections_considered,
        "detections_tracked": len(assignments),
        "tracks_created": len(track_payloads),
        "same_frame_multi_assignment_prevented_count": same_frame_multi_assignment_prevented_count,
        "usable_tracks_for_next_step": usable_tracks_for_next_step,
        "tracks": track_payloads,
    }

    top_tracks_by_detection_count = sorted(track_payloads, key=lambda item: item["detection_count"], reverse=True)[:10]
    top_tracks_by_duration = sorted(track_payloads, key=lambda item: item["duration_seconds"], reverse=True)[:10]
    top_tracks_by_confidence = sorted(track_payloads, key=lambda item: item["avg_confidence"], reverse=True)[:10]
    report_payload = {
        "status": "success",
        "frames_processed": len(frame_items),
        "detections_total_from_yolo": detections_total_from_yolo,
        "detections_considered_for_tracking": detections_considered,
        "detections_ignored_by_class": detections_ignored_by_class,
        "detections_ignored_by_confidence": detections_ignored_by_confidence,
        "tracks_created": len(track_payloads),
        "active_track_count_end": len(track_payloads),
        "same_frame_multi_assignment_prevented_count": same_frame_multi_assignment_prevented_count,
        "track_type_counts": dict(sorted(track_type_counts.items())),
        "dominant_class_track_counts": dict(sorted(dominant_class_track_counts.items())),
        "track_length_stats": _safe_stats(track_lengths),
        "track_duration_stats": _safe_stats(track_durations),
        "track_quality_counts": dict(sorted(track_quality_counts.items())),
        "usable_vehicle_tracks_for_ocr_color": usable_vehicle_tracks_for_ocr_color,
        "usable_person_tracks": usable_person_tracks,
        "unusable_tracks": unusable_tracks,
        "unusable_reason_counts": dict(sorted(unusable_reason_counts.items())),
        "top_tracks_by_detection_count": [{"track_id": item["track_id"], "detection_count": item["detection_count"]} for item in top_tracks_by_detection_count],
        "top_tracks_by_duration": [{"track_id": item["track_id"], "duration_seconds": item["duration_seconds"]} for item in top_tracks_by_duration],
        "top_tracks_by_confidence": [{"track_id": item["track_id"], "avg_confidence": item["avg_confidence"]} for item in top_tracks_by_confidence],
        "recommendation": (
            "Use only usable good vehicle tracks for next OCR/color stages."
            if usable_vehicle_tracks_for_ocr_color > 0
            else "Tracking finished, but no usable vehicle tracks were found. Review class thresholds or YOLO quality."
        ),
    }
    quality_report_payload = {
        "status": "success",
        "total_tracks": len(track_payloads),
        "good_tracks": track_quality_counts.get("good", 0),
        "fragmented_tracks": track_quality_counts.get("fragmented", 0),
        "short_tracks": track_quality_counts.get("short", 0),
        "weak_tracks": track_quality_counts.get("weak", 0),
        "class_mixed_tracks": track_quality_counts.get("class_mixed", 0),
        "single_frame_tracks": track_quality_counts.get("single_frame", 0),
        "usable_vehicle_tracks_for_ocr_color": usable_vehicle_tracks_for_ocr_color,
        "top_bad_tracks_by_reason": [
            {"track_id": item["track_id"], "track_quality": item["track_quality"], "detection_count": item["detection_count"]}
            for item in sorted(track_payloads, key=lambda item: (item["track_quality"] == "good", item["track_quality_score"]))[:10]
        ],
        "top_class_mixed_tracks": [
            {"track_id": item["track_id"], "class_consistency_ratio": item["class_consistency_ratio"], "class_counts": item["class_counts"]}
            for item in sorted(
                [track for track in track_payloads if track["track_quality"] == "class_mixed"],
                key=lambda item: item["class_consistency_ratio"],
            )[:10]
        ],
        "top_fragmented_tracks": [
            {"track_id": item["track_id"], "duration_seconds": item["duration_seconds"], "detection_count": item["detection_count"]}
            for item in sorted(
                [track for track in track_payloads if track["track_quality"] == "fragmented"],
                key=lambda item: item["duration_seconds"],
            )[:10]
        ],
        "recommendation": report_payload["recommendation"],
    }

    write_json(run_dir / "04B_tracks.json", tracks_payload)
    write_json(run_dir / "04B_detection_track_assignments.json", assignments_payload)
    write_json(run_dir / "04B_tracking_report.json", report_payload)
    write_json(run_dir / "04B_tracking_quality_report.json", quality_report_payload)
    return tracks_payload, assignments_payload, report_payload, quality_report_payload
