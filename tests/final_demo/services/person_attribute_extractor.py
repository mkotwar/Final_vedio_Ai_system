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


PERSON_CLASS = "person"
CARRYING_OBJECT_CLASSES = {
    "backpack",
    "handbag",
    "suitcase",
    "bag",
    "laptop",
    "cell phone",
    "bottle",
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
MIN_PERSON_CROP_WIDTH = 24
MIN_PERSON_CROP_HEIGHT = 48
MIN_PERSON_CROP_AREA = 1800
LOW_CONFIDENCE_THRESHOLD = 0.55
HIGH_CONFIDENCE_THRESHOLD = 0.70
DEFAULT_MAX_FALLBACK_RECORDS = 25
ENV_FINAL_DEMO_PERSON_ATTR_MAX_FALLBACK_RECORDS = "FINAL_DEMO_PERSON_ATTR_MAX_FALLBACK_RECORDS"


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


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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
    center_a = compute_bbox_center(box_a)
    center_b = compute_bbox_center(box_b)
    return math.dist(center_a, center_b)


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
    return crop_bbox_from_array(image, bbox_xyxy)


def crop_bbox_from_array(image: Any, bbox_xyxy: list[float]) -> Any | None:
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


def normalize_bbox_sequence(raw_bbox_sequence: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_bbox_sequence, list):
        return []
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(raw_bbox_sequence):
        if not isinstance(item, dict):
            continue
        bbox_xyxy = item.get("bbox_xyxy")
        if not isinstance(bbox_xyxy, list) or len(bbox_xyxy) < 4:
            continue
        normalized.append(
            {
                "timestamp": round(as_float(item.get("timestamp"), float(index)), 3),
                "frame_id": str(item.get("frame_id") or ""),
                "bbox_xyxy": [round(float(value), 3) for value in bbox_xyxy[:4]],
                "confidence": round(as_float(item.get("confidence"), 0.0), 4),
                "detection_id": str(item.get("detection_id") or ""),
            }
        )
    normalized.sort(key=lambda item: (float(item["timestamp"]), str(item["frame_id"]), str(item["detection_id"])))
    return normalized


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


def build_detection_lookup(
    detections_payload: dict[str, Any] | None,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_detections: list[dict[str, Any]] = []
    if not isinstance(detections_payload, dict):
        return by_frame, all_detections
    for detection in list(detections_payload.get("detections") or []):
        if not isinstance(detection, dict):
            continue
        normalized = dict(detection)
        normalized["class_name"] = str(detection.get("class_name") or "").lower()
        normalized["frame_id"] = str(detection.get("frame_id") or "")
        normalized["detection_id"] = str(detection.get("detection_id") or "")
        normalized["global_timestamp_seconds"] = round(
            as_float(detection.get("global_timestamp_seconds"), 0.0),
            3,
        )
        bbox_xyxy = detection.get("bbox_xyxy")
        normalized["bbox_xyxy"] = (
            [round(float(value), 3) for value in bbox_xyxy[:4]]
            if isinstance(bbox_xyxy, list) and len(bbox_xyxy) >= 4
            else []
        )
        by_frame[normalized["frame_id"]].append(normalized)
        all_detections.append(normalized)
    return by_frame, all_detections


def normalize_color_name(color_name: str) -> str:
    mapping = {
        "grey": "gray",
        "dark gray": "gray",
        "light gray": "gray",
        "navy": "blue",
        "maroon": "red",
    }
    return mapping.get(color_name.strip().lower(), color_name.strip().lower())


def select_representative_bbox_item(
    track: dict[str, Any],
    bbox_sequence: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not bbox_sequence:
        return None
    best_frame_id = str(track.get("best_frame_id") or "")
    if best_frame_id:
        for item in bbox_sequence:
            if str(item.get("frame_id") or "") == best_frame_id:
                return item
    return max(
        bbox_sequence,
        key=lambda item: (
            compute_bbox_area(list(item.get("bbox_xyxy") or [0.0, 0.0, 0.0, 0.0])),
            float(item.get("confidence") or 0.0),
            -float(item.get("timestamp") or 0.0),
        ),
    )


def select_matching_detection_id(
    *,
    track: dict[str, Any],
    representative_item: dict[str, Any],
    detection_lookup: dict[str, list[dict[str, Any]]],
) -> str:
    detection_id = str(representative_item.get("detection_id") or "")
    if detection_id:
        return detection_id

    explicit_detection_ids = [
        str(item)
        for item in list(track.get("detection_ids") or [])
        if str(item or "").strip()
    ]
    if len(explicit_detection_ids) == 1:
        return explicit_detection_ids[0]

    frame_id = str(representative_item.get("frame_id") or "")
    if not frame_id:
        return explicit_detection_ids[0] if explicit_detection_ids else ""

    bbox = list(representative_item.get("bbox_xyxy") or [])
    best_match_id = ""
    best_score = -1.0
    allowed_detection_ids = set(explicit_detection_ids)
    for detection in detection_lookup.get(frame_id, []):
        candidate_id = str(detection.get("detection_id") or "")
        if allowed_detection_ids and candidate_id not in allowed_detection_ids:
            continue
        candidate_bbox = list(detection.get("bbox_xyxy") or [])
        if len(candidate_bbox) < 4 or len(bbox) < 4:
            continue
        score = compute_iou(bbox, candidate_bbox)
        if score > best_score:
            best_score = score
            best_match_id = candidate_id
    if best_match_id:
        return best_match_id
    return explicit_detection_ids[0] if explicit_detection_ids else ""


def resolve_track_source_ids(track: dict[str, Any]) -> tuple[str, str]:
    clean_track_id = str(track.get("clean_track_id") or "").strip()
    explicit_source_track_id = str(track.get("source_track_id") or "").strip()
    source_track_ids = [
        str(item).strip()
        for item in list(track.get("source_track_ids") or [])
        if str(item or "").strip()
    ]
    attribute_track_id = (
        str(track.get("attribute_track_id") or "").strip()
        or clean_track_id
        or explicit_source_track_id
        or (source_track_ids[0] if source_track_ids else "")
    )
    source_track_id = explicit_source_track_id or (source_track_ids[0] if source_track_ids else "") or clean_track_id
    return source_track_id, attribute_track_id


def select_supporting_bbox_items(bbox_sequence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not bbox_sequence:
        return []
    indices = {0, len(bbox_sequence) // 2, len(bbox_sequence) - 1}
    ranked = sorted(
        range(len(bbox_sequence)),
        key=lambda index: (
            compute_bbox_area(list(bbox_sequence[index].get("bbox_xyxy") or [0.0, 0.0, 0.0, 0.0])),
            float(bbox_sequence[index].get("confidence") or 0.0),
        ),
        reverse=True,
    )
    indices.update(ranked[:2])
    return [bbox_sequence[index] for index in sorted(indices) if 0 <= index < len(bbox_sequence)]


def focus_clothing_region(region: Any | None) -> Any | None:
    if region is None or getattr(region, "size", 0) == 0:
        return None
    height, width = region.shape[:2]
    x_margin = int(round(width * 0.12))
    y_margin = int(round(height * 0.06))
    x1 = min(max(0, x_margin), max(0, width - 1))
    y1 = min(max(0, y_margin), max(0, height - 1))
    x2 = max(x1 + 1, width - x_margin)
    y2 = max(y1 + 1, height - y_margin)
    focused = region[y1:y2, x1:x2]
    if getattr(focused, "size", 0) == 0:
        return region
    return focused


def score_crop_size(width: int, height: int) -> float:
    area = float(width * height)
    if width < MIN_PERSON_CROP_WIDTH or height < MIN_PERSON_CROP_HEIGHT or area < MIN_PERSON_CROP_AREA:
        return 0.25
    return min(1.0, max(0.35, area / 9000.0))


def classify_color_pixels(region: Any | None) -> tuple[str, float, list[str]]:
    if region is None or getattr(region, "size", 0) == 0:
        return "unknown", 0.0, []

    focused = focus_clothing_region(region)
    if focused is None or getattr(focused, "size", 0) == 0:
        return "unknown", 0.0, []

    height, width = focused.shape[:2]
    hsv = cv2.cvtColor(focused, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    masks: dict[str, Any] = {}
    masks["black"] = v < 50
    masks["white"] = (s < 28) & (v >= 200)
    masks["gray"] = (s < 42) & (v >= 50) & (v < 200)
    masks["brown"] = (h >= 5) & (h < 22) & (s >= 60) & (v >= 40) & (v < 160)
    masks["red"] = (((h < 10) | (h >= 170)) & (s >= 55) & (v >= 60))
    masks["orange"] = (h >= 10) & (h < 22) & (s >= 70) & (v >= 160)
    masks["yellow"] = (h >= 22) & (h < 35) & (s >= 60) & (v >= 90)
    masks["green"] = (h >= 35) & (h < 85) & (s >= 50) & (v >= 50)
    masks["blue"] = (h >= 85) & (h < 130) & (s >= 50) & (v >= 40)
    masks["purple"] = (h >= 130) & (h < 160) & (s >= 45) & (v >= 45)
    masks["pink"] = (h >= 160) & (h < 170) & (s >= 35) & (v >= 90)

    counts = {color_name: int(np.count_nonzero(mask)) for color_name, mask in masks.items()}
    total_pixels = max(1, int(width * height))
    top_items = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    dominant_color, dominant_count = top_items[0]
    if dominant_count <= 0:
        return "unknown", 0.0, []

    dominant_ratio = dominant_count / total_pixels
    if dominant_ratio < 0.12:
        return "unknown", round(max(0.0, dominant_ratio), 3), [
            color_name for color_name, count in top_items[:3] if count > 0
        ]

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
        dominant_ratio * 0.52
        + size_quality * 0.20
        + saturation_quality * 0.14
        + brightness_quality * 0.14
    )
    return normalize_color_name(dominant_color), round(min(1.0, confidence), 3), [
        normalize_color_name(color_name)
        for color_name, count in top_items[:3]
        if count > 0
    ]


def split_person_crop(person_crop: Any | None) -> tuple[Any | None, Any | None]:
    if person_crop is None or getattr(person_crop, "size", 0) == 0:
        return None, None
    height, width = person_crop.shape[:2]
    if width <= 0 or height <= 0:
        return None, None
    top_y1 = int(round(height * 0.15))
    top_y2 = int(round(height * 0.55))
    bottom_y1 = int(round(height * 0.55))
    bottom_y2 = int(round(height * 0.95))
    return person_crop[top_y1:top_y2, :], person_crop[bottom_y1:bottom_y2, :]


def combine_color_families(*color_names: str) -> list[str]:
    family: list[str] = []
    for color_name in color_names:
        for token in COLOR_FAMILY_MAP.get(color_name, []):
            if token not in family:
                family.append(token)
    return family


def build_overall_color(
    top_color: str,
    top_confidence: float,
    bottom_color: str,
    bottom_confidence: float,
) -> tuple[str, float]:
    candidates = [
        (top_color, top_confidence),
        (bottom_color, bottom_confidence),
    ]
    known = [(color_name, confidence) for color_name, confidence in candidates if color_name and color_name != "unknown"]
    if not known:
        return "unknown", 0.0
    if len(known) == 2 and known[0][0] == known[1][0]:
        return known[0][0], round((known[0][1] + known[1][1]) / 2.0, 3)
    best_color, best_confidence = max(known, key=lambda item: item[1])
    confidence = best_confidence
    if len(known) == 2:
        confidence = round((best_confidence + min(item[1] for item in known)) / 2.0, 3)
    return best_color, confidence


def evaluate_person_crop_quality(crop: Any | None) -> tuple[bool, list[str], float]:
    reasons: list[str] = []
    if crop is None or getattr(crop, "size", 0) == 0:
        return False, ["crop_missing"], 0.0
    height, width = crop.shape[:2]
    area = int(width * height)
    if width < MIN_PERSON_CROP_WIDTH:
        reasons.append("crop_too_narrow")
    if height < MIN_PERSON_CROP_HEIGHT:
        reasons.append("crop_too_short")
    if area < MIN_PERSON_CROP_AREA:
        reasons.append("crop_area_too_small")
    return len(reasons) == 0, reasons, round(score_crop_size(width, height), 3)


def normalize_helper_class_name(class_name: str) -> str:
    normalized = str(class_name or "").strip().lower()
    if normalized in {"backpack", "handbag", "suitcase", "laptop", "cell phone", "bottle"}:
        return normalized
    if "bag" in normalized:
        return "bag"
    return normalized


def evaluate_helper_candidates(
    *,
    person_bbox: list[float],
    frame_id: str,
    timestamp: float,
    detection_lookup: dict[str, list[dict[str, Any]]],
    all_detections: list[dict[str, Any]],
) -> tuple[bool, str, float, list[str], list[str], bool]:
    person_center = compute_bbox_center(person_bbox)
    person_width = max(1.0, float(person_bbox[2]) - float(person_bbox[0]))
    person_height = max(1.0, float(person_bbox[3]) - float(person_bbox[1]))

    candidate_pool = list(detection_lookup.get(frame_id, []))
    if not candidate_pool:
        candidate_pool = [
            item
            for item in all_detections
            if abs(float(item.get("global_timestamp_seconds") or 0.0) - timestamp) <= 0.5
        ]

    helper_candidates: list[dict[str, Any]] = []
    nearby_object_ids: list[str] = []
    nearby_object_types: list[str] = []
    for detection in candidate_pool:
        class_name = normalize_helper_class_name(str(detection.get("class_name") or ""))
        if class_name not in CARRYING_OBJECT_CLASSES:
            continue
        bbox_xyxy = list(detection.get("bbox_xyxy") or [])
        if len(bbox_xyxy) < 4:
            continue
        detection_id = str(detection.get("detection_id") or "")
        if detection_id and detection_id not in nearby_object_ids:
            nearby_object_ids.append(detection_id)
        if class_name and class_name not in nearby_object_types:
            nearby_object_types.append(class_name)

        object_center = compute_bbox_center(bbox_xyxy)
        distance = math.dist(person_center, object_center)
        proximity = max(0.0, 1.0 - (distance / max(person_width, person_height, 1.0)))
        overlap = compute_iou(person_bbox, bbox_xyxy)
        torso_side_zone = (
            float(person_bbox[0]) - person_width * 0.25 <= object_center[0] <= float(person_bbox[2]) + person_width * 0.25
            and float(person_bbox[1]) + person_height * 0.12 <= object_center[1] <= float(person_bbox[3]) + person_height * 0.20
        )
        region_bonus = 1.0 if torso_side_zone else 0.45
        score = round(min(1.0, overlap * 0.45 + proximity * 0.40 + region_bonus * 0.15), 3)
        helper_candidates.append(
            {
                "class_name": class_name,
                "detection_id": detection_id,
                "confidence": score,
            }
        )

    if not helper_candidates:
        return False, "", 0.0, nearby_object_ids, nearby_object_types, False

    helper_candidates.sort(key=lambda item: (float(item["confidence"]), item["class_name"]), reverse=True)
    best_candidate = helper_candidates[0]
    carrying_possible = float(best_candidate["confidence"]) >= 0.35
    carrying_confidence = round(float(best_candidate["confidence"]), 3) if carrying_possible else 0.0
    needs_review = carrying_possible and carrying_confidence < 0.7
    return (
        carrying_possible,
        str(best_candidate["class_name"]),
        carrying_confidence,
        nearby_object_ids,
        nearby_object_types,
        needs_review,
    )


def select_image_path(
    *,
    explicit_image_path: Any,
    frame_id: str,
    frame_lookup: dict[str, dict[str, Any]],
) -> str | None:
    if explicit_image_path:
        return str(explicit_image_path)
    if frame_id and frame_id in frame_lookup:
        image_path = frame_lookup[frame_id].get("image_path")
        if image_path:
            return str(image_path)
    return None


def save_person_crop(
    *,
    crop: Any | None,
    output_dir: Path,
    person_attribute_id: str,
) -> str | None:
    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{person_attribute_id}.jpg"
    if not cv2.imwrite(str(output_path), crop):
        return None
    return to_repo_relative_path(output_path)


def build_review_flags(
    *,
    crop_ok: bool,
    crop_reasons: list[str],
    top_color: str,
    top_confidence: float,
    bottom_color: str,
    bottom_confidence: float,
    carrying_object_needs_review: bool,
) -> tuple[bool, str, str]:
    reasons: list[str] = []
    if not crop_ok:
        reasons.extend(crop_reasons or ["low_quality_crop"])
        return True, "low_quality_crop", "; ".join(dict.fromkeys(reasons))
    if top_color == "unknown":
        reasons.append("top_color_unclear")
    if bottom_color == "unknown":
        reasons.append("bottom_color_unclear")
    if 0.0 < top_confidence < LOW_CONFIDENCE_THRESHOLD:
        reasons.append("top_color_low_confidence")
    if 0.0 < bottom_confidence < LOW_CONFIDENCE_THRESHOLD:
        reasons.append("bottom_color_low_confidence")
    if carrying_object_needs_review:
        reasons.append("nearby_object_association_needs_review")
    if not reasons:
        return False, "extracted", ""
    return True, "partial", "; ".join(dict.fromkeys(reasons))


def build_person_attribute_record(
    *,
    person_attribute_id: str,
    record_source: str,
    source_track_id: str | None,
    attribute_track_id: str | None,
    source_detection_id: str | None,
    start_time: float,
    end_time: float,
    representative_timestamp: float,
    best_frame_id: str,
    best_image_path: str | None,
    bbox: list[float],
    crop_path: str | None,
    crop: Any | None,
    detection_lookup: dict[str, list[dict[str, Any]]],
    all_detections: list[dict[str, Any]],
) -> dict[str, Any]:
    crop_ok, crop_reasons, crop_quality = evaluate_person_crop_quality(crop)
    top_region, bottom_region = split_person_crop(crop)
    top_color, top_confidence, _top_candidates = classify_color_pixels(top_region)
    bottom_color, bottom_confidence, _bottom_candidates = classify_color_pixels(bottom_region)
    overall_color, overall_confidence = build_overall_color(
        top_color,
        top_confidence,
        bottom_color,
        bottom_confidence,
    )

    (
        carrying_object_possible,
        carrying_object_type,
        carrying_object_confidence,
        nearby_object_ids,
        nearby_object_types,
        carrying_object_needs_review,
    ) = evaluate_helper_candidates(
        person_bbox=bbox,
        frame_id=best_frame_id,
        timestamp=representative_timestamp,
        detection_lookup=detection_lookup,
        all_detections=all_detections,
    )

    needs_review, attribute_status, review_reason = build_review_flags(
        crop_ok=crop_ok,
        crop_reasons=crop_reasons,
        top_color=top_color,
        top_confidence=top_confidence,
        bottom_color=bottom_color,
        bottom_confidence=bottom_confidence,
        carrying_object_needs_review=carrying_object_needs_review,
    )

    confidence_parts = [
        crop_quality,
        top_confidence if top_color != "unknown" else 0.0,
        bottom_confidence if bottom_color != "unknown" else 0.0,
    ]
    if carrying_object_possible:
        confidence_parts.append(carrying_object_confidence)
    person_attribute_confidence = round(
        sum(confidence_parts) / max(1, len(confidence_parts)),
        3,
    )

    return {
        "person_attribute_id": person_attribute_id,
        "record_source": record_source,
        "source_track_id": source_track_id,
        "attribute_track_id": attribute_track_id,
        "source_detection_id": source_detection_id,
        "class_name": PERSON_CLASS,
        "entity_family": "person",
        "start_time": round(start_time, 3),
        "end_time": round(end_time, 3),
        "representative_timestamp": round(representative_timestamp, 3),
        "duration_seconds": round(max(0.0, end_time - start_time), 3),
        "frame_id": best_frame_id,
        "timestamp": round(representative_timestamp, 3),
        "best_frame_id": best_frame_id,
        "best_image_path": best_image_path,
        "bbox": [round(float(value), 3) for value in bbox[:4]],
        "crop_path": crop_path,
        "top_clothing_color": top_color,
        "top_clothing_color_confidence": round(top_confidence, 3),
        "bottom_clothing_color": bottom_color,
        "bottom_clothing_color_confidence": round(bottom_confidence, 3),
        "overall_clothing_color": overall_color,
        "overall_clothing_color_confidence": round(overall_confidence, 3),
        "normalized_top_color": normalize_color_name(top_color) if top_color != "unknown" else "",
        "normalized_bottom_color": normalize_color_name(bottom_color) if bottom_color != "unknown" else "",
        "clothing_color_family": combine_color_families(top_color, bottom_color),
        "carrying_object_possible": carrying_object_possible,
        "carrying_object_type": carrying_object_type,
        "carrying_object_confidence": round(carrying_object_confidence, 3),
        "nearby_object_ids": nearby_object_ids,
        "nearby_object_types": nearby_object_types,
        "person_attribute_confidence": person_attribute_confidence,
        "needs_review": needs_review,
        "review_reason": review_reason,
        "attribute_status": attribute_status,
    }


def build_track_based_records(
    *,
    run_dir: Path,
    person_tracks: list[dict[str, Any]],
    frame_lookup: dict[str, dict[str, Any]],
    detection_lookup: dict[str, list[dict[str, Any]]],
    all_detections: list[dict[str, Any]],
    warnings: list[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    crop_dir = run_dir / "10_person_attribute_crops"
    for index, track in enumerate(person_tracks, start=1):
        person_attribute_id = f"person_attr_{index:06d}"
        bbox_sequence = normalize_bbox_sequence(track.get("bbox_sequence"))
        representative_item = select_representative_bbox_item(track, bbox_sequence)
        if representative_item is None:
            warnings.append(f"Person track {track.get('source_track_id')} does not have bbox_sequence.")
            continue

        best_frame_id = str(representative_item.get("frame_id") or track.get("best_frame_id") or "")
        best_image_path = select_image_path(
            explicit_image_path=track.get("best_image_path"),
            frame_id=best_frame_id,
            frame_lookup=frame_lookup,
        )
        crop = None
        if best_image_path:
            absolute_image_path = resolve_image_path(best_image_path)
            if absolute_image_path is not None and absolute_image_path.exists():
                crop = crop_bbox_from_image(absolute_image_path, list(representative_item["bbox_xyxy"]))
        crop_path = save_person_crop(crop=crop, output_dir=crop_dir, person_attribute_id=person_attribute_id)
        if crop_path is None:
            warnings.append(f"Person crop could not be saved for track {track.get('source_track_id')}.")
        source_track_id, attribute_track_id = resolve_track_source_ids(track)
        source_detection_id = select_matching_detection_id(
            track=track,
            representative_item=representative_item,
            detection_lookup=detection_lookup,
        )

        record = build_person_attribute_record(
            person_attribute_id=person_attribute_id,
            record_source="track_based",
            source_track_id=source_track_id,
            attribute_track_id=attribute_track_id,
            source_detection_id=source_detection_id,
            start_time=as_float(track.get("start_time"), as_float(representative_item.get("timestamp"), 0.0)),
            end_time=as_float(track.get("end_time"), as_float(representative_item.get("timestamp"), 0.0)),
            representative_timestamp=as_float(representative_item.get("timestamp"), 0.0),
            best_frame_id=best_frame_id,
            best_image_path=best_image_path,
            bbox=list(representative_item["bbox_xyxy"]),
            crop_path=crop_path,
            crop=crop,
            detection_lookup=detection_lookup,
            all_detections=all_detections,
        )
        records.append(record)
    return records


def detection_covered_by_track(
    detection: dict[str, Any],
    track: dict[str, Any],
    *,
    time_tolerance: float = 0.6,
    iou_threshold: float = 0.2,
    center_distance_ratio: float = 0.6,
) -> bool:
    detection_bbox = list(detection.get("bbox_xyxy") or [])
    if len(detection_bbox) < 4:
        return False
    detection_frame_id = str(detection.get("frame_id") or "")
    detection_time = as_float(detection.get("global_timestamp_seconds"), 0.0)
    track_bbox_sequence = normalize_bbox_sequence(track.get("bbox_sequence"))
    if not track_bbox_sequence:
        return False

    for bbox_item in track_bbox_sequence:
        bbox_frame_id = str(bbox_item.get("frame_id") or "")
        bbox_time = as_float(bbox_item.get("timestamp"), 0.0)
        if detection_frame_id and bbox_frame_id and detection_frame_id == bbox_frame_id:
            candidate_bbox = list(bbox_item.get("bbox_xyxy") or [])
        elif abs(detection_time - bbox_time) <= time_tolerance:
            candidate_bbox = list(bbox_item.get("bbox_xyxy") or [])
        else:
            continue
        if len(candidate_bbox) < 4:
            continue
        if compute_iou(detection_bbox, candidate_bbox) >= iou_threshold:
            return True
        distance = compute_center_distance(detection_bbox, candidate_bbox)
        detection_scale = max(
            1.0,
            float(detection_bbox[2]) - float(detection_bbox[0]),
            float(detection_bbox[3]) - float(detection_bbox[1]),
        )
        candidate_scale = max(
            1.0,
            float(candidate_bbox[2]) - float(candidate_bbox[0]),
            float(candidate_bbox[3]) - float(candidate_bbox[1]),
        )
        if distance <= max(detection_scale, candidate_scale) * center_distance_ratio:
            return True
    return False


def dedupe_fallback_detections(
    detections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    sorted_detections = sorted(
        detections,
        key=lambda item: (
            as_float(item.get("global_timestamp_seconds"), 0.0),
            -as_float(item.get("confidence"), 0.0),
            str(item.get("detection_id") or ""),
        ),
    )
    for detection in sorted_detections:
        bbox = list(detection.get("bbox_xyxy") or [])
        if len(bbox) < 4:
            continue
        detection_time = as_float(detection.get("global_timestamp_seconds"), 0.0)
        area = compute_bbox_area(bbox)
        is_duplicate = False
        for kept in deduped:
            kept_bbox = list(kept.get("bbox_xyxy") or [])
            if len(kept_bbox) < 4:
                continue
            kept_time = as_float(kept.get("global_timestamp_seconds"), 0.0)
            if abs(detection_time - kept_time) > 0.75:
                continue
            iou = compute_iou(bbox, kept_bbox)
            distance = compute_center_distance(bbox, kept_bbox)
            kept_area = compute_bbox_area(kept_bbox)
            area_ratio = min(area, kept_area) / max(area, kept_area, 1.0)
            scale = max(
                1.0,
                float(bbox[2]) - float(bbox[0]),
                float(bbox[3]) - float(bbox[1]),
                float(kept_bbox[2]) - float(kept_bbox[0]),
                float(kept_bbox[3]) - float(kept_bbox[1]),
            )
            if iou >= 0.55 or (distance <= scale * 0.45 and area_ratio >= 0.65):
                is_duplicate = True
                break
        if not is_duplicate:
            deduped.append(detection)
    return deduped


def build_detection_fallback_records(
    *,
    run_dir: Path,
    person_detections: list[dict[str, Any]],
    detection_lookup: dict[str, list[dict[str, Any]]],
    all_detections: list[dict[str, Any]],
    warnings: list[str],
    start_index: int = 1,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    crop_dir = run_dir / "10_person_attribute_crops"
    for index, detection in enumerate(person_detections, start=start_index):
        person_attribute_id = f"person_attr_{index:06d}"
        best_image_path = str(detection.get("image_path") or "") or None
        crop = None
        absolute_image_path = resolve_image_path(best_image_path)
        bbox = list(detection.get("bbox_xyxy") or [])
        if absolute_image_path is not None and absolute_image_path.exists() and len(bbox) >= 4:
            crop = crop_bbox_from_image(absolute_image_path, bbox)
        crop_path = save_person_crop(crop=crop, output_dir=crop_dir, person_attribute_id=person_attribute_id)
        if crop_path is None:
            warnings.append(f"Person crop could not be saved for detection {detection.get('detection_id')}.")

        timestamp = as_float(detection.get("global_timestamp_seconds"), 0.0)
        records.append(
            build_person_attribute_record(
                person_attribute_id=person_attribute_id,
                record_source="detection_fallback",
                source_track_id=None,
                attribute_track_id=None,
                source_detection_id=str(detection.get("detection_id") or ""),
                start_time=timestamp,
                end_time=timestamp,
                representative_timestamp=timestamp,
                best_frame_id=str(detection.get("frame_id") or ""),
                best_image_path=best_image_path,
                bbox=bbox[:4],
                crop_path=crop_path,
                crop=crop,
                detection_lookup=detection_lookup,
                all_detections=all_detections,
            )
        )
    return records


def summarize_records(records: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    top_counts: Counter[str] = Counter()
    bottom_counts: Counter[str] = Counter()
    for record in records:
        top_color = str(record.get("top_clothing_color") or "")
        bottom_color = str(record.get("bottom_clothing_color") or "")
        if top_color and top_color != "unknown":
            top_counts[top_color] += 1
        if bottom_color and bottom_color != "unknown":
            bottom_counts[bottom_color] += 1
    return dict(sorted(top_counts.items())), dict(sorted(bottom_counts.items()))


def build_person_attribute_outputs(run_dir: Path) -> dict[str, Any]:
    clean_tracks_path = run_dir / "05B_clean_tracks.json"
    detections_path = run_dir / "04_yolo_detections.json"
    frames_index_path = run_dir / "03_sampled_frames_index.json"

    detections_payload = read_optional_json(detections_path)
    if detections_payload is None:
        raise FileNotFoundError(f"Missing required Step 10 input: {detections_path}")
    frames_index_payload = read_optional_json(frames_index_path)
    if frames_index_payload is None:
        raise FileNotFoundError(f"Missing required Step 10 input: {frames_index_path}")

    clean_tracks_payload = read_optional_json(clean_tracks_path)
    frame_lookup = build_frame_lookup(frames_index_payload)
    detection_lookup, all_detections = build_detection_lookup(detections_payload)
    warnings: list[str] = []
    recommendations: list[str] = []
    max_fallback_records = read_optional_positive_int_env(
        ENV_FINAL_DEMO_PERSON_ATTR_MAX_FALLBACK_RECORDS,
        DEFAULT_MAX_FALLBACK_RECORDS,
    )

    person_tracks = []
    if isinstance(clean_tracks_payload, dict):
        person_tracks = [
            track
            for track in list(clean_tracks_payload.get("clean_tracks") or [])
            if isinstance(track, dict)
            and str(track.get("class_name") or "").lower() == PERSON_CLASS
            and str(track.get("cleanup_status") or "") != "noise_short_track"
        ]

    person_detections = [
        detection
        for detection in all_detections
        if str(detection.get("class_name") or "").lower() == PERSON_CLASS
    ]

    if not person_tracks and person_detections:
        warnings.append("No person tracks found; detection fallback was used for person attributes.")

    records: list[dict[str, Any]] = []
    if person_tracks:
        records = build_track_based_records(
            run_dir=run_dir,
            person_tracks=person_tracks,
            frame_lookup=frame_lookup,
            detection_lookup=detection_lookup,
            all_detections=all_detections,
            warnings=warnings,
        )
    else:
        records = []

    covered_detection_ids: set[str] = set()
    covered_detection_count = 0
    uncovered_person_detections: list[dict[str, Any]] = []
    for detection in person_detections:
        detection_id = str(detection.get("detection_id") or "")
        covered = any(detection_covered_by_track(detection, track) for track in person_tracks)
        if covered:
            covered_detection_count += 1
            if detection_id:
                covered_detection_ids.add(detection_id)
            continue
        uncovered_person_detections.append(detection)

    fallback_candidates_before_dedup = len(uncovered_person_detections)
    deduped_fallback_candidates = dedupe_fallback_detections(uncovered_person_detections)
    fallback_candidates_after_dedup = len(deduped_fallback_candidates)
    limited_fallback_candidates = deduped_fallback_candidates[:max_fallback_records]
    fallback_records_limited = max(0, len(deduped_fallback_candidates) - len(limited_fallback_candidates))

    if person_tracks:
        fallback_records = build_detection_fallback_records(
            run_dir=run_dir,
            person_detections=limited_fallback_candidates,
            detection_lookup=detection_lookup,
            all_detections=all_detections,
            warnings=warnings,
            start_index=len(records) + 1,
        )
        records.extend(fallback_records)
    else:
        records = build_detection_fallback_records(
            run_dir=run_dir,
            person_detections=limited_fallback_candidates,
            detection_lookup=detection_lookup,
            all_detections=all_detections,
            warnings=warnings,
        )

    if not records:
        warnings.append("No person records found in current video/index.")

    records_with_missing_source_ids = sum(
        1
        for record in records
        if str(record.get("record_source") or "") == "track_based"
        and (
            not str(record.get("source_track_id") or "").strip()
            or not str(record.get("attribute_track_id") or "").strip()
        )
    )
    if records_with_missing_source_ids > 0:
        warnings.append("Some track-based person attribute records are missing source track IDs.")
    if len(uncovered_person_detections) >= max(10, max(1, len(person_detections) // 4)):
        warnings.append("Many person detections were not covered by person tracks.")
    if any(str(record.get("record_source") or "") == "detection_fallback" for record in records):
        warnings.append("Detection-fallback person records were created and may contain duplicates.")

    top_counts, bottom_counts = summarize_records(records)
    low_quality_count = sum(1 for record in records if str(record.get("attribute_status") or "") == "low_quality_crop")
    carrying_object_count = sum(1 for record in records if bool(record.get("carrying_object_possible")))
    needs_review_count = sum(1 for record in records if bool(record.get("needs_review")))
    helper_object_detections = sum(
        1 for detection in all_detections if normalize_helper_class_name(str(detection.get("class_name") or "")) in CARRYING_OBJECT_CLASSES
    )

    if records and low_quality_count >= max(2, math.ceil(len(records) * 0.5)):
        recommendations.append(
            "Many person crops were low quality. Consider higher frame sampling or better person tracking."
        )
    if helper_object_detections == 0:
        recommendations.append(
            "Carrying object detection is limited in this run. Consider a stronger object association step."
        )

    created_at = current_timestamp()
    track_based_records = sum(1 for record in records if str(record.get("record_source")) == "track_based")
    detection_fallback_records = sum(
        1 for record in records if str(record.get("record_source")) == "detection_fallback"
    )
    if track_based_records > 0 and detection_fallback_records > 0:
        record_source_mode = "hybrid"
    elif track_based_records > 0:
        record_source_mode = "track_based"
    else:
        record_source_mode = "detection_fallback"
    attributes_payload = {
        "created_at": created_at,
        "person_attribute_count": len(records),
        "record_source_mode": record_source_mode,
        "person_attributes": records,
    }
    report_payload = {
        "overall_status": "completed",
        "person_tracks_loaded": len(person_tracks),
        "person_detections_loaded": len(person_detections),
        "person_detections_covered_by_tracks": covered_detection_count,
        "person_detections_uncovered": len(uncovered_person_detections),
        "person_attribute_records": len(records),
        "track_based_records": track_based_records,
        "detection_fallback_records": detection_fallback_records,
        "fallback_candidates_before_dedup": fallback_candidates_before_dedup,
        "fallback_candidates_after_dedup": fallback_candidates_after_dedup,
        "fallback_records_limited": fallback_records_limited,
        "records_with_missing_source_ids": records_with_missing_source_ids,
        "records_with_top_color": sum(
            1 for record in records if str(record.get("top_clothing_color") or "") not in {"", "unknown"}
        ),
        "records_with_bottom_color": sum(
            1 for record in records if str(record.get("bottom_clothing_color") or "") not in {"", "unknown"}
        ),
        "records_with_carrying_object_possible": carrying_object_count,
        "records_needing_review": needs_review_count,
        "records_by_top_color": top_counts,
        "records_by_bottom_color": bottom_counts,
        "warnings": warnings,
        "recommendations": recommendations,
        "created_at": created_at,
    }
    return {
        "attributes_payload": attributes_payload,
        "report_payload": report_payload,
    }


def update_run_manifest_for_person_attributes(run_manifest_path: Path) -> dict[str, Any]:
    run_manifest = read_json(run_manifest_path)
    completed_steps = list(run_manifest.get("completed_steps") or [])
    if "10_person_attribute_extraction" not in completed_steps:
        completed_steps.append("10_person_attribute_extraction")
    run_manifest["completed_steps"] = completed_steps
    run_manifest["next_step"] = "11_object_attribute_extraction"
    write_json(run_manifest_path, run_manifest)
    return run_manifest
