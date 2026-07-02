from __future__ import annotations

import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tests.final_demo.services.chunk_planner import read_json
from tests.final_demo.services.video_io import current_timestamp, write_json


EXCLUDED_CLASSES = {
    "person",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "auto",
    "rickshaw",
    "vehicle",
}
COLOR_FAMILY_MAP = {
    "white": ["white", "light"],
    "gray": ["gray", "grey", "light", "white_possible"],
    "black": ["black", "dark"],
    "brown": ["brown", "dark"],
    "red": ["red"],
    "orange": ["orange"],
    "yellow": ["yellow"],
    "green": ["green"],
    "blue": ["blue"],
    "purple": ["purple"],
    "pink": ["pink"],
    "unknown": [],
}
NORMALIZED_OBJECT_TYPES = {
    "cell phone": "phone",
    "mobile phone": "phone",
    "phone": "phone",
    "handbag": "bag",
    "backpack": "backpack",
    "suitcase": "suitcase",
    "laptop": "laptop",
    "bottle": "bottle",
}
SMALL_OBJECT_LARGE_BBOX_THRESHOLDS = {
    "suitcase": 0.20,
    "backpack": 0.20,
    "handbag": 0.20,
    "bag": 0.20,
    "laptop": 0.10,
    "cell phone": 0.10,
    "phone": 0.10,
    "bottle": 0.10,
}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "auto", "rickshaw", "vehicle"}
MIN_OBJECT_CROP_WIDTH = 18
MIN_OBJECT_CROP_HEIGHT = 18
MIN_OBJECT_CROP_AREA = 600
LOW_CONFIDENCE_THRESHOLD = 0.55
DEFAULT_MAX_RECORDS = 100
ENV_FINAL_DEMO_OBJECT_ATTR_MAX_RECORDS = "FINAL_DEMO_OBJECT_ATTR_MAX_RECORDS"


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def to_repo_relative_path(path: Path) -> str:
    return path.resolve().relative_to(get_repo_root()).as_posix()


def resolve_image_path(path_value: Any) -> Path | None:
    text = str(path_value or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else (get_repo_root() / path)


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def read_optional_positive_int_env(env_name: str, default_value: int) -> int:
    raw_value = os.environ.get(env_name, str(default_value))
    try:
        value = int(str(raw_value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Environment variable {env_name} must be a valid integer. Received: {raw_value!r}"
        ) from exc
    if value <= 0:
        raise ValueError(
            f"Environment variable {env_name} must be greater than 0. Received: {value}"
        )
    return value


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number):
        return default
    return number


def compute_bbox_area(bbox_xyxy: list[float]) -> float:
    return max(0.0, float(bbox_xyxy[2]) - float(bbox_xyxy[0])) * max(
        0.0,
        float(bbox_xyxy[3]) - float(bbox_xyxy[1]),
    )


def compute_bbox_center(bbox_xyxy: list[float]) -> tuple[float, float]:
    return (
        (float(bbox_xyxy[0]) + float(bbox_xyxy[2])) / 2.0,
        (float(bbox_xyxy[1]) + float(bbox_xyxy[3])) / 2.0,
    )


def compute_center_distance(box_a: list[float], box_b: list[float]) -> float:
    return math.dist(compute_bbox_center(box_a), compute_bbox_center(box_b))


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    intersection_x1 = max(float(box_a[0]), float(box_b[0]))
    intersection_y1 = max(float(box_a[1]), float(box_b[1]))
    intersection_x2 = min(float(box_a[2]), float(box_b[2]))
    intersection_y2 = min(float(box_a[3]), float(box_b[3]))
    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection = intersection_width * intersection_height
    union = compute_bbox_area(box_a) + compute_bbox_area(box_b) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def crop_bbox_from_image(image_path: Path, bbox_xyxy: list[float]) -> Any | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox_xyxy[:4]]
    height, width = image.shape[:2]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def normalize_color_name(color_name: str) -> str:
    mapping = {
        "grey": "gray",
        "dark gray": "gray",
        "light gray": "gray",
        "navy": "blue",
        "maroon": "red",
    }
    return mapping.get(color_name.strip().lower(), color_name.strip().lower())


def normalize_object_type(class_name: str) -> str:
    normalized = str(class_name or "").strip().lower()
    return NORMALIZED_OBJECT_TYPES.get(normalized, normalized)


def object_bbox_area_ratio_threshold(class_name: str) -> float | None:
    normalized = str(class_name or "").strip().lower()
    return SMALL_OBJECT_LARGE_BBOX_THRESHOLDS.get(normalized)


def score_crop_size(width: int, height: int) -> float:
    area = float(width * height)
    if width < MIN_OBJECT_CROP_WIDTH or height < MIN_OBJECT_CROP_HEIGHT or area < MIN_OBJECT_CROP_AREA:
        return 0.25
    return min(1.0, max(0.35, area / 5000.0))


def focus_object_region(region: Any | None) -> Any | None:
    if region is None or getattr(region, "size", 0) == 0:
        return None
    height, width = region.shape[:2]
    border_x = int(round(width * 0.08))
    border_y = int(round(height * 0.08))
    x1 = min(max(0, border_x), max(0, width - 1))
    y1 = min(max(0, border_y), max(0, height - 1))
    x2 = max(x1 + 1, width - border_x)
    y2 = max(y1 + 1, height - border_y)
    focused = region[y1:y2, x1:x2]
    if getattr(focused, "size", 0) == 0:
        return region
    return focused


def classify_color_pixels(region: Any | None) -> tuple[str, float]:
    if region is None or getattr(region, "size", 0) == 0:
        return "unknown", 0.0
    focused = focus_object_region(region)
    if focused is None or getattr(focused, "size", 0) == 0:
        return "unknown", 0.0
    height, width = focused.shape[:2]
    hsv = cv2.cvtColor(focused, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    masks = {
        "black": v < 50,
        "white": (s < 28) & (v >= 200),
        "gray": (s < 42) & (v >= 50) & (v < 200),
        "brown": (h >= 5) & (h < 22) & (s >= 60) & (v >= 40) & (v < 160),
        "red": (((h < 10) | (h >= 170)) & (s >= 55) & (v >= 60)),
        "orange": (h >= 10) & (h < 22) & (s >= 70) & (v >= 160),
        "yellow": (h >= 22) & (h < 35) & (s >= 60) & (v >= 90),
        "green": (h >= 35) & (h < 85) & (s >= 50) & (v >= 50),
        "blue": (h >= 85) & (h < 130) & (s >= 50) & (v >= 40),
        "purple": (h >= 130) & (h < 160) & (s >= 45) & (v >= 45),
        "pink": (h >= 160) & (h < 170) & (s >= 35) & (v >= 90),
    }
    counts = {color_name: int(np.count_nonzero(mask)) for color_name, mask in masks.items()}
    total_pixels = max(1, int(width * height))
    dominant_color, dominant_count = max(counts.items(), key=lambda item: item[1])
    if dominant_count <= 0:
        return "unknown", 0.0
    dominant_ratio = dominant_count / total_pixels
    if dominant_ratio < 0.12:
        return "unknown", round(max(0.0, dominant_ratio), 3)
    saturation_quality = min(1.0, float(np.mean(s)) / 140.0)
    brightness_mean = float(np.mean(v))
    if brightness_mean < 35 or brightness_mean > 245:
        brightness_quality = 0.45
    elif brightness_mean < 60 or brightness_mean > 220:
        brightness_quality = 0.7
    else:
        brightness_quality = 1.0
    size_quality = score_crop_size(width, height)
    confidence = (
        dominant_ratio * 0.48
        + size_quality * 0.18
        + saturation_quality * 0.14
        + brightness_quality * 0.12
    )
    return normalize_color_name(dominant_color), round(min(1.0, confidence), 3)


def build_frame_lookup(frames_index_payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    if not isinstance(frames_index_payload, dict):
        return lookup
    for frame in list(frames_index_payload.get("frames") or []):
        if not isinstance(frame, dict):
            continue
        frame_id = str(frame.get("frame_id") or "")
        if frame_id:
            lookup[frame_id] = frame
    return lookup


def frame_dimensions_for_detection(
    detection: dict[str, Any],
    frame_lookup: dict[str, dict[str, Any]],
) -> tuple[int, int]:
    frame_id = str(detection.get("frame_id") or "")
    frame = frame_lookup.get(frame_id) if frame_id else None
    width = 0
    height = 0
    if isinstance(frame, dict):
        width = int(frame.get("width") or 0)
        height = int(frame.get("height") or 0)
    if width > 0 and height > 0:
        return width, height
    image_path = resolve_image_path(detection.get("image_path"))
    if image_path is not None and image_path.exists():
        image = cv2.imread(str(image_path))
        if image is not None:
            read_height, read_width = image.shape[:2]
            return int(read_width), int(read_height)
    return 0, 0


def build_object_detections(detections_payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    object_detections: list[dict[str, Any]] = []
    if not isinstance(detections_payload, dict):
        return object_detections
    for detection in list(detections_payload.get("detections") or []):
        if not isinstance(detection, dict):
            continue
        class_name = str(detection.get("class_name") or "").strip().lower()
        if not class_name or class_name in EXCLUDED_CLASSES:
            continue
        bbox_xyxy = detection.get("bbox_xyxy")
        if not isinstance(bbox_xyxy, list) or len(bbox_xyxy) < 4:
            continue
        object_detections.append(
            {
                **detection,
                "class_name": class_name,
                "detection_id": str(detection.get("detection_id") or ""),
                "frame_id": str(detection.get("frame_id") or ""),
                "image_path": str(detection.get("image_path") or ""),
                "global_timestamp_seconds": round(
                    as_float(detection.get("global_timestamp_seconds"), 0.0),
                    3,
                ),
                "confidence": round(as_float(detection.get("confidence"), 0.0), 4),
                "bbox_xyxy": [round(float(value), 3) for value in bbox_xyxy[:4]],
            }
        )
    object_detections.sort(
        key=lambda item: (
            str(item["class_name"]),
            float(item["global_timestamp_seconds"]),
            str(item["frame_id"]),
            str(item["detection_id"]),
        )
    )
    return object_detections


def build_vehicle_detections(detections_payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    vehicle_detections: list[dict[str, Any]] = []
    if not isinstance(detections_payload, dict):
        return vehicle_detections
    for detection in list(detections_payload.get("detections") or []):
        if not isinstance(detection, dict):
            continue
        class_name = str(detection.get("class_name") or "").strip().lower()
        if class_name not in VEHICLE_CLASSES:
            continue
        bbox_xyxy = detection.get("bbox_xyxy")
        if not isinstance(bbox_xyxy, list) or len(bbox_xyxy) < 4:
            continue
        vehicle_detections.append(
            {
                "class_name": class_name,
                "frame_id": str(detection.get("frame_id") or ""),
                "global_timestamp_seconds": round(
                    as_float(detection.get("global_timestamp_seconds"), 0.0),
                    3,
                ),
                "bbox_xyxy": [round(float(value), 3) for value in bbox_xyxy[:4]],
            }
        )
    return vehicle_detections


def build_vehicle_track_windows(clean_tracks_payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    if not isinstance(clean_tracks_payload, dict):
        return windows
    for track in list(clean_tracks_payload.get("clean_tracks") or []):
        if not isinstance(track, dict):
            continue
        class_name = str(track.get("class_name") or "").strip().lower()
        if class_name not in VEHICLE_CLASSES:
            continue
        bbox_sequence = list(track.get("bbox_sequence") or [])
        normalized_sequence: list[dict[str, Any]] = []
        for item in bbox_sequence:
            if not isinstance(item, dict):
                continue
            bbox_xyxy = item.get("bbox_xyxy")
            if not isinstance(bbox_xyxy, list) or len(bbox_xyxy) < 4:
                continue
            normalized_sequence.append(
                {
                    "timestamp": round(as_float(item.get("timestamp"), 0.0), 3),
                    "frame_id": str(item.get("frame_id") or ""),
                    "bbox_xyxy": [round(float(value), 3) for value in bbox_xyxy[:4]],
                }
            )
        if not normalized_sequence:
            continue
        windows.append(
            {
                "class_name": class_name,
                "start_time": round(as_float(track.get("start_time"), normalized_sequence[0]["timestamp"]), 3),
                "end_time": round(as_float(track.get("end_time"), normalized_sequence[-1]["timestamp"]), 3),
                "bbox_sequence": normalized_sequence,
            }
        )
    return windows


def detections_should_group(current: dict[str, Any], candidate: dict[str, Any]) -> bool:
    if str(current.get("class_name")) != str(candidate.get("class_name")):
        return False
    current_time = as_float(current.get("global_timestamp_seconds"), 0.0)
    candidate_time = as_float(candidate.get("global_timestamp_seconds"), 0.0)
    if abs(current_time - candidate_time) > 1.0:
        return False
    current_bbox = list(current.get("bbox_xyxy") or [])
    candidate_bbox = list(candidate.get("bbox_xyxy") or [])
    if len(current_bbox) < 4 or len(candidate_bbox) < 4:
        return False
    iou = compute_iou(current_bbox, candidate_bbox)
    distance = compute_center_distance(current_bbox, candidate_bbox)
    current_area = compute_bbox_area(current_bbox)
    candidate_area = compute_bbox_area(candidate_bbox)
    area_ratio = min(current_area, candidate_area) / max(current_area, candidate_area, 1.0)
    scale = max(
        1.0,
        float(current_bbox[2]) - float(current_bbox[0]),
        float(current_bbox[3]) - float(current_bbox[1]),
        float(candidate_bbox[2]) - float(candidate_bbox[0]),
        float(candidate_bbox[3]) - float(candidate_bbox[1]),
    )
    return iou >= 0.35 or (distance <= scale * 0.55 and area_ratio >= 0.55)


def group_object_detections(object_detections: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for detection in object_detections:
        placed = False
        for group in groups:
            if detections_should_group(group[-1], detection):
                group.append(detection)
                placed = True
                break
        if not placed:
            groups.append([detection])
    return groups


def save_object_crop(crop: Any | None, output_dir: Path, object_attribute_id: str) -> str | None:
    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{object_attribute_id}.jpg"
    if not cv2.imwrite(str(output_path), crop):
        return None
    return to_repo_relative_path(output_path)


def build_review_flags(
    crop: Any | None,
    object_color: str,
    object_color_confidence: float,
    detection_confidence: float,
) -> tuple[bool, str, str, float]:
    if crop is None or getattr(crop, "size", 0) == 0:
        return True, "low_quality_crop", "crop_missing", 0.0
    height, width = crop.shape[:2]
    crop_area = int(width * height)
    if width < MIN_OBJECT_CROP_WIDTH or height < MIN_OBJECT_CROP_HEIGHT or crop_area < MIN_OBJECT_CROP_AREA:
        return True, "low_quality_crop", "crop_too_small", round(score_crop_size(width, height), 3)
    crop_quality = round(score_crop_size(width, height), 3)
    reasons: list[str] = []
    if object_color == "unknown":
        reasons.append("object_color_unclear")
    if 0.0 < object_color_confidence < LOW_CONFIDENCE_THRESHOLD:
        reasons.append("object_color_low_confidence")
    if detection_confidence < 0.25:
        reasons.append("low_detection_confidence")
    if reasons:
        return True, "partial", "; ".join(reasons), crop_quality
    return False, "extracted", "", crop_quality


def center_inside_bbox(center: tuple[float, float], bbox_xyxy: list[float]) -> bool:
    return (
        float(bbox_xyxy[0]) <= center[0] <= float(bbox_xyxy[2])
        and float(bbox_xyxy[1]) <= center[1] <= float(bbox_xyxy[3])
    )


def collect_vehicle_overlap_info(
    *,
    representative: dict[str, Any],
    bbox: list[float],
    vehicle_detections: list[dict[str, Any]],
    vehicle_track_windows: list[dict[str, Any]],
) -> tuple[bool, list[str], float]:
    representative_time = as_float(representative.get("global_timestamp_seconds"), 0.0)
    representative_frame_id = str(representative.get("frame_id") or "")
    object_center = compute_bbox_center(bbox)
    possible_vehicle_types: list[str] = []
    max_overlap_score = 0.0

    def consider_vehicle(candidate_class_name: str, candidate_bbox: list[float]) -> None:
        nonlocal max_overlap_score
        iou = compute_iou(bbox, candidate_bbox)
        inside = center_inside_bbox(object_center, candidate_bbox)
        if iou > 0.30 or inside:
            if candidate_class_name not in possible_vehicle_types:
                possible_vehicle_types.append(candidate_class_name)
            max_overlap_score = max(max_overlap_score, iou if iou > 0 else (0.35 if inside else 0.0))

    for vehicle_detection in vehicle_detections:
        same_frame = representative_frame_id and representative_frame_id == str(vehicle_detection.get("frame_id") or "")
        close_time = abs(representative_time - as_float(vehicle_detection.get("global_timestamp_seconds"), 0.0)) <= 0.6
        if not same_frame and not close_time:
            continue
        candidate_bbox = list(vehicle_detection.get("bbox_xyxy") or [])
        if len(candidate_bbox) < 4:
            continue
        consider_vehicle(str(vehicle_detection.get("class_name") or ""), candidate_bbox)

    for track_window in vehicle_track_windows:
        if representative_time < as_float(track_window.get("start_time"), 0.0) - 0.8:
            continue
        if representative_time > as_float(track_window.get("end_time"), 0.0) + 0.8:
            continue
        for bbox_item in list(track_window.get("bbox_sequence") or []):
            same_frame = representative_frame_id and representative_frame_id == str(bbox_item.get("frame_id") or "")
            close_time = abs(representative_time - as_float(bbox_item.get("timestamp"), 0.0)) <= 0.6
            if not same_frame and not close_time:
                continue
            candidate_bbox = list(bbox_item.get("bbox_xyxy") or [])
            if len(candidate_bbox) < 4:
                continue
            consider_vehicle(str(track_window.get("class_name") or ""), candidate_bbox)
            break

    return bool(possible_vehicle_types), possible_vehicle_types or [], round(max_overlap_score, 3)


def build_object_attribute_record(
    *,
    object_attribute_id: str,
    group: list[dict[str, Any]],
    frame_lookup: dict[str, dict[str, Any]],
    vehicle_detections: list[dict[str, Any]],
    vehicle_track_windows: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, bool]]:
    representative = max(
        group,
        key=lambda item: (
            float(item.get("confidence") or 0.0),
            compute_bbox_area(list(item.get("bbox_xyxy") or [0.0, 0.0, 0.0, 0.0])),
            -float(item.get("global_timestamp_seconds") or 0.0),
        ),
    )
    best_frame_id = str(representative.get("frame_id") or "")
    best_image_path = str(representative.get("image_path") or "")
    if not best_image_path and best_frame_id in frame_lookup:
        best_image_path = str(frame_lookup[best_frame_id].get("image_path") or "")
    bbox = list(representative.get("bbox_xyxy") or [])
    crop = None
    absolute_path = resolve_image_path(best_image_path)
    if absolute_path is not None and absolute_path.exists():
        crop = crop_bbox_from_image(absolute_path, bbox)
    crop_path = save_object_crop(crop, output_dir, object_attribute_id)
    object_color, object_color_confidence = classify_color_pixels(crop)
    normalized_color = normalize_color_name(object_color) if object_color != "unknown" else ""
    needs_review, attribute_status, review_reason, crop_quality = build_review_flags(
        crop,
        object_color,
        object_color_confidence,
        as_float(representative.get("confidence"), 0.0),
    )
    object_attribute_confidence = round(
        (
            crop_quality * 0.35
            + object_color_confidence * 0.45
            + as_float(representative.get("confidence"), 0.0) * 0.20
        ),
        3,
    )
    class_name = str(representative.get("class_name") or "").lower()
    normalized_object_type = normalize_object_type(class_name)
    frame_width, frame_height = frame_dimensions_for_detection(representative, frame_lookup)
    frame_area = float(frame_width * frame_height) if frame_width > 0 and frame_height > 0 else 0.0
    bbox_area = compute_bbox_area(bbox)
    bbox_area_ratio = round((bbox_area / frame_area), 4) if frame_area > 0 else 0.0
    review_reasons = [reason for reason in str(review_reason or "").split("; ") if reason]
    possible_actual_family = ""
    possible_actual_types: list[str] = []
    object_class_needs_review = bool(needs_review)
    false_positive_risk_score = 0.0

    large_bbox_flag = False
    bbox_ratio_threshold = object_bbox_area_ratio_threshold(class_name)
    if bbox_ratio_threshold is not None and bbox_area_ratio > bbox_ratio_threshold:
        large_bbox_flag = True
        object_class_needs_review = True
        needs_review = True
        false_positive_risk_score = max(false_positive_risk_score, min(1.0, bbox_area_ratio / max(bbox_ratio_threshold, 0.001)))
        if "object_bbox_too_large_for_class" not in review_reasons:
            review_reasons.append("object_bbox_too_large_for_class")
        if attribute_status not in {"low_quality_crop", "possible_vehicle_misclassification"}:
            attribute_status = "possible_false_positive"

    vehicle_overlap_flag, overlap_vehicle_types, vehicle_overlap_score = collect_vehicle_overlap_info(
        representative=representative,
        bbox=bbox,
        vehicle_detections=vehicle_detections,
        vehicle_track_windows=vehicle_track_windows,
    )
    if vehicle_overlap_flag:
        object_class_needs_review = True
        needs_review = True
        possible_actual_family = "vehicle"
        possible_actual_types = overlap_vehicle_types or ["unknown_vehicle"]
        false_positive_risk_score = max(false_positive_risk_score, max(0.65, vehicle_overlap_score))
        if "object_overlaps_vehicle_detection" not in review_reasons:
            review_reasons.append("object_overlaps_vehicle_detection")
        attribute_status = "possible_vehicle_misclassification"
        if large_bbox_flag:
            normalized_object_type = "uncertain_object"
            if "unknown_vehicle" not in possible_actual_types:
                possible_actual_types.append("unknown_vehicle")

    detection_confidence = as_float(representative.get("confidence"), 0.0)
    if detection_confidence < 0.25 and "low_detection_confidence" not in review_reasons:
        review_reasons.append("low_detection_confidence")
        object_class_needs_review = True
        needs_review = True
        false_positive_risk_score = max(false_positive_risk_score, 0.4)

    if object_class_needs_review and large_bbox_flag and not possible_actual_family:
        possible_actual_family = "object"

    review_reason = "; ".join(dict.fromkeys(review_reasons))
    record = {
        "object_attribute_id": object_attribute_id,
        "record_source": "detection_fallback",
        "source_detection_id": str(representative.get("detection_id") or ""),
        "source_track_id": "",
        "attribute_track_id": "",
        "class_name": class_name,
        "entity_family": "object",
        "object_type": class_name,
        "normalized_object_type": normalized_object_type,
        "start_time": round(as_float(group[0].get("global_timestamp_seconds"), 0.0), 3),
        "end_time": round(as_float(group[-1].get("global_timestamp_seconds"), 0.0), 3),
        "representative_timestamp": round(as_float(representative.get("global_timestamp_seconds"), 0.0), 3),
        "duration_seconds": round(
            max(
                0.0,
                as_float(group[-1].get("global_timestamp_seconds"), 0.0)
                - as_float(group[0].get("global_timestamp_seconds"), 0.0),
            ),
            3,
        ),
        "best_frame_id": best_frame_id,
        "best_image_path": best_image_path or None,
        "bbox": [round(float(value), 3) for value in bbox[:4]],
        "crop_path": crop_path,
        "object_color": object_color,
        "object_color_confidence": round(object_color_confidence, 3),
        "normalized_color": normalized_color,
        "color_family": list(COLOR_FAMILY_MAP.get(object_color, [])),
        "object_attribute_confidence": object_attribute_confidence,
        "needs_review": needs_review,
        "review_reason": review_reason,
        "attribute_status": attribute_status,
        "possible_actual_family": possible_actual_family,
        "possible_actual_types": possible_actual_types,
        "false_positive_risk_score": round(min(1.0, false_positive_risk_score), 3),
        "object_class_needs_review": object_class_needs_review,
        "bbox_area_ratio": bbox_area_ratio,
        "nearby_person_ids": [],
        "nearby_person_detection_ids": [],
        "nearby_person_possible": False,
        "association_status": "not_computed",
    }
    return record, {
        "low_quality_crop": attribute_status == "low_quality_crop",
        "possible_false_positive": attribute_status == "possible_false_positive",
        "possible_vehicle_misclassification": attribute_status == "possible_vehicle_misclassification",
        "large_bbox_reviewed": large_bbox_flag,
        "vehicle_overlap_reviewed": vehicle_overlap_flag,
    }


def summarize_counts(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        value = str(record.get(key) or "")
        if value:
            counts[value] += 1
    return dict(sorted(counts.items()))


def build_object_attribute_outputs(run_dir: Path) -> dict[str, Any]:
    detections_path = run_dir / "04_yolo_detections.json"
    frames_index_path = run_dir / "03_sampled_frames_index.json"
    clean_tracks_path = run_dir / "05B_clean_tracks.json"
    detections_payload = read_optional_json(detections_path)
    if detections_payload is None:
        raise FileNotFoundError(f"Missing required Step 11 input: {detections_path}")
    frames_index_payload = read_optional_json(frames_index_path)
    if frames_index_payload is None:
        raise FileNotFoundError(f"Missing required Step 11 input: {frames_index_path}")

    frame_lookup = build_frame_lookup(frames_index_payload)
    object_detections = build_object_detections(detections_payload)
    vehicle_detections = build_vehicle_detections(detections_payload)
    vehicle_track_windows = build_vehicle_track_windows(read_optional_json(clean_tracks_path))
    warnings: list[str] = []
    recommendations: list[str] = []
    max_records = read_optional_positive_int_env(
        ENV_FINAL_DEMO_OBJECT_ATTR_MAX_RECORDS,
        DEFAULT_MAX_RECORDS,
    )

    if not object_detections:
        warnings.append("Current video has no generic object detections.")

    grouped_candidates = group_object_detections(object_detections)
    fallback_candidates_before_dedup = len(object_detections)
    fallback_candidates_after_dedup = len(grouped_candidates)
    limited_groups = grouped_candidates[:max_records]
    records_limited = len(grouped_candidates) > max_records

    output_dir = run_dir / "11_object_attribute_crops"
    records: list[dict[str, Any]] = []
    low_quality_crop_count = 0
    possible_false_positive_count = 0
    possible_vehicle_misclassification_count = 0
    large_bbox_reviewed_count = 0
    vehicle_overlap_reviewed_count = 0
    for index, group in enumerate(limited_groups, start=1):
        object_attribute_id = f"object_attr_{index:06d}"
        record, flags = build_object_attribute_record(
            object_attribute_id=object_attribute_id,
            group=group,
            frame_lookup=frame_lookup,
            vehicle_detections=vehicle_detections,
            vehicle_track_windows=vehicle_track_windows,
            output_dir=output_dir,
        )
        records.append(record)
        if flags["low_quality_crop"]:
            low_quality_crop_count += 1
        if flags["possible_false_positive"]:
            possible_false_positive_count += 1
        if flags["possible_vehicle_misclassification"]:
            possible_vehicle_misclassification_count += 1
        if flags["large_bbox_reviewed"]:
            large_bbox_reviewed_count += 1
        if flags["vehicle_overlap_reviewed"]:
            vehicle_overlap_reviewed_count += 1

    if low_quality_crop_count >= max(3, math.ceil(len(records) * 0.4)) and records:
        recommendations.append(
            "Many object crops were low quality. Consider higher resolution or better object crops."
        )
    if fallback_candidates_before_dedup - fallback_candidates_after_dedup >= 10:
        warnings.append(
            f"Object deduplication collapsed {fallback_candidates_before_dedup - fallback_candidates_after_dedup} repeated detections."
        )
    unique_classes = {str(item.get('class_name') or '') for item in object_detections}
    if unique_classes and len(unique_classes) <= 2:
        recommendations.append(
            "Generic object classes are limited in this run. Consider an object-specific model if needed."
        )
    if possible_vehicle_misclassification_count > 0:
        warnings.append(
            f"{possible_vehicle_misclassification_count} object records overlap vehicle evidence and were downgraded for review."
        )

    created_at = current_timestamp()
    report_payload = {
        "overall_status": "completed",
        "object_detections_loaded": len(object_detections),
        "object_attribute_records": len(records),
        "records_by_object_type": summarize_counts(records, "object_type"),
        "records_by_normalized_object_type": summarize_counts(records, "normalized_object_type"),
        "records_by_color": summarize_counts(records, "object_color"),
        "records_by_normalized_color": summarize_counts(records, "normalized_color"),
        "records_needing_review": sum(1 for record in records if bool(record.get("needs_review"))),
        "records_with_missing_color": sum(
            1 for record in records if str(record.get("object_color") or "") in {"", "unknown"}
        ),
        "records_low_quality_crop": low_quality_crop_count,
        "possible_false_positive_records": possible_false_positive_count,
        "possible_vehicle_misclassification_records": possible_vehicle_misclassification_count,
        "large_bbox_rejected_or_reviewed": large_bbox_reviewed_count,
        "vehicle_overlap_reviewed": vehicle_overlap_reviewed_count,
        "records_by_attribute_status": summarize_counts(records, "attribute_status"),
        "fallback_candidates_before_dedup": fallback_candidates_before_dedup,
        "fallback_candidates_after_dedup": fallback_candidates_after_dedup,
        "records_limited": records_limited,
        "warnings": warnings,
        "recommendations": recommendations,
        "created_at": created_at,
    }
    attributes_payload = {
        "created_at": created_at,
        "object_attribute_count": len(records),
        "record_source_mode": "detection_fallback",
        "object_attributes": records,
    }
    return {
        "attributes_payload": attributes_payload,
        "report_payload": report_payload,
    }


def update_run_manifest_for_object_attributes(run_manifest_path: Path) -> dict[str, Any]:
    run_manifest = read_json(run_manifest_path)
    completed_steps = list(run_manifest.get("completed_steps") or [])
    if "11_object_attribute_extraction" not in completed_steps:
        completed_steps.append("11_object_attribute_extraction")
    run_manifest["completed_steps"] = completed_steps
    run_manifest["next_step"] = "12_entity_association"
    write_json(run_manifest_path, run_manifest)
    return run_manifest
