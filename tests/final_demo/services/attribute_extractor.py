from __future__ import annotations

import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tests.final_demo.services.chunk_planner import read_json
from tests.final_demo.services.tracker_adapter import (
    crop_bbox_from_image,
    to_absolute_repo_path,
    to_repo_relative_path,
)
from tests.final_demo.services.video_io import current_timestamp, write_json


ENV_FINAL_DEMO_ATTRIBUTE_STATIONARY_DISTANCE_RATIO = "FINAL_DEMO_ATTRIBUTE_STATIONARY_DISTANCE_RATIO"
ENV_FINAL_DEMO_ATTRIBUTE_SPEED_SLOW_PX_PER_SEC = "FINAL_DEMO_ATTRIBUTE_SPEED_SLOW_PX_PER_SEC"
ENV_FINAL_DEMO_ATTRIBUTE_SPEED_FAST_PX_PER_SEC = "FINAL_DEMO_ATTRIBUTE_SPEED_FAST_PX_PER_SEC"

DEFAULT_STATIONARY_DISTANCE_RATIO = 0.03
DEFAULT_SPEED_SLOW_PX_PER_SEC = 30.0
DEFAULT_SPEED_FAST_PX_PER_SEC = 150.0

PERSON_CLASSES = {"person"}
VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "bus", "truck"}
ANIMAL_CLASSES = {"dog", "cat", "horse", "sheep", "cow"}
HELPER_CLASSES = {"backpack", "handbag", "suitcase"}


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def read_non_negative_float_env(env_name: str, default_value: float) -> float:
    raw_value = os.environ.get(env_name, str(default_value))
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {env_name} must be a valid number. Received: {raw_value!r}"
        ) from exc
    if value < 0:
        raise ValueError(
            f"Environment variable {env_name} must be greater than or equal to 0. Received: {value}"
        )
    return value


def read_positive_float_env(env_name: str, default_value: float) -> float:
    value = read_non_negative_float_env(env_name, default_value)
    if value <= 0:
        raise ValueError(
            f"Environment variable {env_name} must be greater than 0. Received: {value}"
        )
    return value


def choose_number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number):
        return default
    return number


def choose_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def compute_bbox_area(bbox_xyxy: list[float]) -> float:
    return max(0.0, float(bbox_xyxy[2]) - float(bbox_xyxy[0])) * max(
        0.0, float(bbox_xyxy[3]) - float(bbox_xyxy[1])
    )


def compute_center_distance(center_a: list[float], center_b: list[float]) -> float:
    return math.dist(center_a, center_b)


def detect_track_source(
    run_dir: Path,
    warnings: list[str],
) -> tuple[str, list[dict[str, Any]], dict[str, Any] | None]:
    clean_tracks_path = run_dir / "05B_clean_tracks.json"
    if clean_tracks_path.exists():
        payload = read_json(clean_tracks_path)
        return "05B_clean_tracks", list(payload.get("clean_tracks") or []), payload

    raw_tracks_path = run_dir / "05_tracks.json"
    if raw_tracks_path.exists():
        warnings.append("05B clean tracks not found; using raw Step 5 tracks for attributes.")
        payload = read_json(raw_tracks_path)
        return "05_tracks", list(payload.get("tracks") or []), payload

    raise FileNotFoundError(
        "Missing required Step 6 input. Expected 05B_clean_tracks.json or fallback 05_tracks.json."
    )


def build_frame_lookup(frames_index_payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    if not isinstance(frames_index_payload, dict):
        return lookup
    for frame in list(frames_index_payload.get("frames") or []):
        if isinstance(frame, dict):
            frame_id = str(frame.get("frame_id") or "")
            if frame_id:
                lookup[frame_id] = frame
    return lookup


def build_detection_lookup(detections_payload: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    lookup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not isinstance(detections_payload, dict):
        return lookup
    for detection in list(detections_payload.get("detections") or []):
        if not isinstance(detection, dict):
            continue
        frame_id = str(detection.get("frame_id") or "")
        if frame_id:
            lookup[frame_id].append(detection)
    return lookup


def normalize_bbox_sequence(raw_bbox_sequence: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_bbox_sequence, list):
        return []
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(raw_bbox_sequence):
        if isinstance(item, dict):
            bbox_xyxy = item.get("bbox_xyxy")
            if not isinstance(bbox_xyxy, list) or len(bbox_xyxy) < 4:
                continue
            normalized.append(
                {
                    "timestamp": round(choose_number(item.get("timestamp"), float(index)), 3),
                    "frame_id": str(item.get("frame_id") or ""),
                    "bbox_xyxy": [round(float(value), 3) for value in bbox_xyxy[:4]],
                    "confidence": round(choose_number(item.get("confidence"), 0.0), 4),
                }
            )
        elif isinstance(item, list) and len(item) >= 4:
            normalized.append(
                {
                    "timestamp": round(float(index), 3),
                    "frame_id": "",
                    "bbox_xyxy": [round(float(value), 3) for value in item[:4]],
                    "confidence": 0.0,
                }
            )
    normalized.sort(key=lambda entry: (float(entry["timestamp"]), str(entry["frame_id"])))
    return normalized


def normalize_center_sequence(
    raw_center_sequence: Any,
    bbox_sequence: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    normalized: list[dict[str, Any]] = []
    if isinstance(raw_center_sequence, list):
        for index, item in enumerate(raw_center_sequence):
            if isinstance(item, dict):
                center = item.get("center")
                if isinstance(center, list) and len(center) >= 2:
                    normalized.append(
                        {
                            "timestamp": round(choose_number(item.get("timestamp"), float(index)), 3),
                            "center": [round(float(center[0]), 3), round(float(center[1]), 3)],
                        }
                    )
            elif isinstance(item, list) and len(item) >= 2:
                normalized.append(
                    {
                        "timestamp": round(float(index), 3),
                        "center": [round(float(item[0]), 3), round(float(item[1]), 3)],
                    }
                )
    if normalized:
        normalized.sort(key=lambda entry: float(entry["timestamp"]))
        return normalized, False

    reconstructed: list[dict[str, Any]] = []
    for bbox_item in bbox_sequence:
        x1, y1, x2, y2 = [float(value) for value in bbox_item["bbox_xyxy"]]
        reconstructed.append(
            {
                "timestamp": round(float(bbox_item["timestamp"]), 3),
                "center": [round((x1 + x2) / 2.0, 3), round((y1 + y2) / 2.0, 3)],
            }
        )
    return reconstructed, True


def select_best_bbox_item(track: dict[str, Any], bbox_sequence: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not bbox_sequence:
        return None
    best_frame_id = str(track.get("best_frame_id") or "")
    for item in bbox_sequence:
        if str(item.get("frame_id") or "") == best_frame_id:
            return item
    return max(
        bbox_sequence,
        key=lambda item: (
            float(item.get("confidence", 0.0)),
            compute_bbox_area(list(item.get("bbox_xyxy") or [0.0, 0.0, 0.0, 0.0])),
            -float(item.get("timestamp", 0.0)),
        ),
    )


def normalize_track(
    raw_track: dict[str, Any],
    *,
    frame_lookup: dict[str, dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    bbox_sequence = normalize_bbox_sequence(raw_track.get("bbox_sequence"))
    center_sequence, centers_reconstructed = normalize_center_sequence(
        raw_track.get("center_sequence"),
        bbox_sequence,
    )
    if centers_reconstructed and bbox_sequence:
        warnings.append("Center data was reconstructed from bbox_sequence for some tracks.")
    if not bbox_sequence:
        warnings.append("Missing bbox_sequence for one or more tracks.")
    if not center_sequence:
        warnings.append("Missing center_sequence for one or more tracks.")

    source_track_id = str(
        raw_track.get("clean_track_id")
        or raw_track.get("local_track_id")
        or raw_track.get("track_id")
        or ""
    )
    best_frame_id = str(raw_track.get("best_frame_id") or raw_track.get("first_frame_id") or "")
    best_image_path = raw_track.get("best_image_path")
    if not best_image_path and best_frame_id in frame_lookup:
        best_image_path = frame_lookup[best_frame_id].get("image_path")

    start_time = round(
        choose_number(raw_track.get("start_time"), bbox_sequence[0]["timestamp"] if bbox_sequence else 0.0),
        3,
    )
    end_time = round(
        choose_number(raw_track.get("end_time"), bbox_sequence[-1]["timestamp"] if bbox_sequence else start_time),
        3,
    )
    duration_seconds = round(
        choose_number(raw_track.get("duration_seconds"), max(0.0, end_time - start_time)),
        3,
    )
    detection_ids = [str(item) for item in list(raw_track.get("detection_ids") or [])]
    detection_count = choose_int(raw_track.get("detection_count"), 0)
    if detection_count <= 0:
        detection_count = len(detection_ids) or len(bbox_sequence) or len(center_sequence)

    return {
        "source_track_id": source_track_id,
        "class_name": str(raw_track.get("class_name") or "unknown").lower(),
        "dominant_class_name": str(raw_track.get("dominant_class_name") or raw_track.get("class_name") or "unknown").lower(),
        "dominant_class_confidence": round(choose_number(raw_track.get("dominant_class_confidence"), 0.0), 4),
        "class_votes": {
            str(key).lower(): choose_int(value, 0)
            for key, value in dict(raw_track.get("class_votes") or {}).items()
            if str(key).strip()
        },
        "class_confidence_sum": {
            str(key).lower(): round(choose_number(value, 0.0), 4)
            for key, value in dict(raw_track.get("class_confidence_sum") or {}).items()
            if str(key).strip()
        },
        "class_consistency_score": round(choose_number(raw_track.get("class_consistency_score"), 1.0), 4),
        "class_conflict": bool(raw_track.get("class_conflict", False)),
        "chunk_id": str(raw_track.get("chunk_id") or "unknown_chunk"),
        "chunk_index": choose_int(raw_track.get("chunk_index"), 0),
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": max(0.0, duration_seconds),
        "detection_count": max(0, detection_count),
        "cleanup_status": str(raw_track.get("cleanup_status") or "unknown"),
        "count_for_summary": raw_track.get("count_for_summary", True),
        "needs_review": bool(raw_track.get("needs_review", False)),
        "best_frame_id": best_frame_id,
        "best_image_path": str(best_image_path) if best_image_path else None,
        "thumbnail_path": raw_track.get("thumbnail_path"),
        "bbox_sequence": bbox_sequence,
        "center_sequence": center_sequence,
        "detection_ids": detection_ids,
        "best_bbox_item": select_best_bbox_item(raw_track, bbox_sequence),
        "max_confidence": round(choose_number(raw_track.get("max_confidence"), 0.0), 4),
    }


def infer_frame_dimensions(
    track: dict[str, Any],
    frame_lookup: dict[str, dict[str, Any]],
    video_info_payload: dict[str, Any] | None,
) -> tuple[int, int]:
    best_frame_id = str(track.get("best_frame_id") or "")
    frame_item = frame_lookup.get(best_frame_id, {})
    width = choose_int(frame_item.get("width"), 0)
    height = choose_int(frame_item.get("height"), 0)
    if width > 0 and height > 0:
        return width, height
    width = choose_int(video_info_payload.get("width") if isinstance(video_info_payload, dict) else 0, 0)
    height = choose_int(video_info_payload.get("height") if isinstance(video_info_payload, dict) else 0, 0)
    return width, height


def build_attribute_crop(
    attribute_track_id: str,
    track: dict[str, Any],
    *,
    objects_dir: Path,
    warnings: list[str],
) -> tuple[str | None, str, Any | None]:
    best_bbox_item = track.get("best_bbox_item")
    best_image_path = track.get("best_image_path")
    if not isinstance(best_bbox_item, dict) or not best_image_path:
        warnings.append(f"Failed image crop for track {attribute_track_id}.")
        return None, "crop_unavailable", None

    crop = crop_bbox_from_image(
        to_absolute_repo_path(str(best_image_path)),
        list(best_bbox_item["bbox_xyxy"]),
    )
    if crop is None:
        warnings.append(f"Failed image crop for track {attribute_track_id}.")
        return None, "crop_failed", None

    objects_dir.mkdir(parents=True, exist_ok=True)
    output_path = objects_dir / f"{attribute_track_id}.jpg"
    if not cv2.imwrite(str(output_path), crop):
        warnings.append(f"Failed image crop for track {attribute_track_id}.")
        return None, "crop_write_failed", crop
    return to_repo_relative_path(output_path), "crop_saved", crop


def crop_center_region(crop: Any | None, x_ratio: float = 0.1, y_ratio: float = 0.1) -> Any | None:
    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    height, width = crop.shape[:2]
    x_margin = int(round(width * x_ratio))
    y_margin = int(round(height * y_ratio))
    if width <= x_margin * 2 or height <= y_margin * 2:
        return crop
    return crop[y_margin : height - y_margin, x_margin : width - x_margin]


def color_palette_for_mode(mode: str) -> list[tuple[str, tuple[int, int, int]]]:
    if mode == "vehicle":
        return [
            ("black", (20, 20, 20)),
            ("white", (235, 235, 235)),
            ("gray", (128, 128, 128)),
            ("silver", (190, 190, 190)),
            ("red", (200, 45, 45)),
            ("orange", (220, 120, 40)),
            ("yellow", (225, 215, 60)),
            ("green", (60, 150, 60)),
            ("blue", (60, 90, 190)),
            ("brown", (120, 80, 55)),
        ]
    return [
        ("black", (20, 20, 20)),
        ("white", (235, 235, 235)),
        ("gray", (128, 128, 128)),
        ("red", (200, 45, 45)),
        ("orange", (220, 120, 40)),
        ("yellow", (225, 215, 60)),
        ("green", (60, 150, 60)),
        ("blue", (60, 90, 190)),
        ("purple", (120, 70, 150)),
        ("pink", (220, 140, 170)),
        ("brown", (120, 80, 55)),
    ]


def color_distance(rgb_a: list[int], rgb_b: tuple[int, int, int]) -> float:
    return math.sqrt(
        sum((float(a) - float(b)) ** 2 for a, b in zip(rgb_a, rgb_b))
    )


def extract_color_candidates(
    crop: Any | None,
    *,
    mode: str,
    max_candidates: int = 3,
) -> tuple[str, list[int] | None, float, list[str]]:
    if crop is None or getattr(crop, "size", 0) == 0:
        return "unknown", None, 0.0, []

    working_crop = crop_center_region(crop, x_ratio=0.08, y_ratio=0.08)
    if working_crop is None or getattr(working_crop, "size", 0) == 0:
        return "unknown", None, 0.0, []

    mean_bgr = working_crop.reshape(-1, 3).mean(axis=0)
    mean_rgb = [
        int(round(float(mean_bgr[2]))),
        int(round(float(mean_bgr[1]))),
        int(round(float(mean_bgr[0]))),
    ]

    hsv_pixel = cv2.cvtColor(
        np.uint8([[[mean_bgr[0], mean_bgr[1], mean_bgr[2]]]]),
        cv2.COLOR_BGR2HSV,
    )[0][0]
    hue, saturation, value = int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2])

    if value < 35:
        return "black", mean_rgb, 0.92, ["black"]
    if mode == "vehicle":
        if saturation < 18 and value > 225:
            return "white", mean_rgb, 0.9, ["white", "silver", "gray"]
        if saturation < 30:
            if value > 175:
                return "silver", mean_rgb, 0.74, ["silver", "gray", "white"]
            return "gray", mean_rgb, 0.72, ["gray", "silver", "black"]
    else:
        if saturation < 20 and value > 225:
            return "white", mean_rgb, 0.9, ["white", "gray"]
        if saturation < 35:
            if value < 70:
                return "black", mean_rgb, 0.8, ["black", "gray"]
            return "gray", mean_rgb, 0.72, ["gray", "white", "black"]

    palette = color_palette_for_mode(mode)
    ranked = sorted(
        ((name, color_distance(mean_rgb, rgb)) for name, rgb in palette),
        key=lambda item: item[1],
    )
    candidates = [name for name, _ in ranked[:max_candidates]]
    dominant = candidates[0] if candidates else "unknown"
    best_distance = ranked[0][1] if ranked else 255.0
    confidence = round(max(0.2, min(0.95, 1.0 - best_distance / 255.0)), 3)

    if mode == "vehicle" and dominant in {"white", "silver", "gray"}:
        merged_candidates = ["white", "silver", "gray"]
        return dominant, mean_rgb, confidence, merged_candidates[:max_candidates]
    return dominant, mean_rgb, confidence, candidates


def split_person_crop(person_crop: Any | None) -> tuple[Any | None, Any | None]:
    if person_crop is None or getattr(person_crop, "size", 0) == 0:
        return None, None
    height, width = person_crop.shape[:2]
    if height < 20 or width < 10:
        return None, None
    upper_start = int(round(height * 0.15))
    upper_end = int(round(height * 0.55))
    lower_start = int(round(height * 0.55))
    lower_end = int(round(height * 0.95))
    upper_crop = person_crop[upper_start:upper_end, :]
    lower_crop = person_crop[lower_start:lower_end, :]
    if upper_crop.size == 0:
        upper_crop = None
    if lower_crop.size == 0:
        lower_crop = None
    return upper_crop, lower_crop


def build_plate_candidate_crop(
    attribute_track_id: str,
    track: dict[str, Any],
    *,
    plate_dir: Path,
    warnings: list[str],
) -> tuple[str | None, str, bool]:
    def score_plate_region(region: Any | None) -> float:
        if region is None or getattr(region, "size", 0) == 0:
            return -1.0
        region_height, region_width = region.shape[:2]
        if region_height < 12 or region_width < 24:
            return -1.0
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        edge_map = cv2.Canny(gray, 80, 180)
        edge_density = float(edge_map.mean()) / 255.0
        aspect_ratio = region_width / max(1.0, float(region_height))
        if 2.0 <= aspect_ratio <= 6.5:
            aspect_score = 1.0
        elif 1.6 <= aspect_ratio <= 8.0:
            aspect_score = 0.65
        else:
            aspect_score = 0.2
        horizontal_band_score = 1.0 if region_height <= region_width * 0.45 else 0.55
        return round(edge_density * 0.55 + aspect_score * 0.30 + horizontal_band_score * 0.15, 4)

    best_bbox_item = track.get("best_bbox_item")
    best_image_path = track.get("best_image_path")
    if not isinstance(best_bbox_item, dict) or not best_image_path:
        warnings.append(f"Plate crop not available for vehicle track {attribute_track_id}.")
        return None, "not_available", False

    vehicle_crop = crop_bbox_from_image(
        to_absolute_repo_path(str(best_image_path)),
        list(best_bbox_item["bbox_xyxy"]),
    )
    if vehicle_crop is None:
        warnings.append(f"Plate crop not available for vehicle track {attribute_track_id}.")
        return None, "not_available", False

    crop_height, crop_width = vehicle_crop.shape[:2]
    if crop_height < 20 or crop_width < 30:
        warnings.append(f"Plate crop not available for vehicle track {attribute_track_id}.")
        return None, "not_available", False

    region_specs = [
        ("lower_center_tight", 0.58, 0.82, 0.28, 0.72),
        ("lower_center_wide", 0.55, 0.86, 0.20, 0.80),
        ("bumper_low", 0.66, 0.92, 0.24, 0.76),
        ("rear_mid", 0.48, 0.76, 0.26, 0.74),
    ]
    best_region_score = -1.0
    plate_crop = None
    for _, y1_ratio, y2_ratio, x1_ratio, x2_ratio in region_specs:
        y1 = int(round(crop_height * y1_ratio))
        y2 = int(round(crop_height * y2_ratio))
        x1 = int(round(crop_width * x1_ratio))
        x2 = int(round(crop_width * x2_ratio))
        if y2 <= y1 or x2 <= x1:
            continue
        candidate_region = vehicle_crop[y1:y2, x1:x2]
        candidate_score = score_plate_region(candidate_region)
        if candidate_score > best_region_score:
            best_region_score = candidate_score
            plate_crop = candidate_region

    if plate_crop is None or getattr(plate_crop, "size", 0) == 0:
        warnings.append(f"Plate crop not available for vehicle track {attribute_track_id}.")
        return None, "not_available", False

    plate_dir.mkdir(parents=True, exist_ok=True)
    output_path = plate_dir / f"{attribute_track_id}_plate_candidate.jpg"
    if not cv2.imwrite(str(output_path), plate_crop):
        warnings.append(f"Plate crop not available for vehicle track {attribute_track_id}.")
        return None, "not_available", False
    return to_repo_relative_path(output_path), "candidate_saved", True


def detect_object_group(class_name: str) -> str:
    if class_name in PERSON_CLASSES:
        return "person"
    if class_name in VEHICLE_CLASSES:
        return "vehicle"
    if class_name in ANIMAL_CLASSES:
        return "animal"
    return "object"


def classify_direction(
    dx: float,
    dy: float,
    movement_distance: float,
    stationary_threshold: float,
) -> tuple[str, float]:
    if movement_distance <= 0:
        return "unknown", 0.0
    if movement_distance < stationary_threshold:
        return "mostly_stationary", 0.85

    abs_dx = abs(dx)
    abs_dy = abs(dy)
    confidence = round(min(1.0, max(abs_dx, abs_dy) / max(1.0, movement_distance)), 3)
    if abs_dx >= abs_dy * 1.5:
        return ("left_to_right" if dx > 0 else "right_to_left"), confidence
    if abs_dy >= abs_dx * 1.5:
        return ("top_to_bottom" if dy > 0 else "bottom_to_top"), confidence
    if dx > 0 and dy > 0:
        return "diagonal_down_right", confidence
    if dx < 0 and dy > 0:
        return "diagonal_down_left", confidence
    if dx > 0 and dy < 0:
        return "diagonal_up_right", confidence
    if dx < 0 and dy < 0:
        return "diagonal_up_left", confidence
    return "unknown", 0.2


def compute_speed_level(
    speed_pixels_per_second: float | None,
    slow_threshold: float,
    fast_threshold: float,
) -> str:
    if speed_pixels_per_second is None:
        return "unknown"
    if speed_pixels_per_second <= 0:
        return "stationary"
    if speed_pixels_per_second < slow_threshold:
        return "slow"
    if speed_pixels_per_second < fast_threshold:
        return "medium"
    return "fast"


def compute_stationary_ratio(
    center_sequence: list[dict[str, Any]],
    stationary_threshold: float,
) -> float:
    if len(center_sequence) < 2:
        return 1.0
    total_steps = 0
    stationary_steps = 0
    for previous, current in zip(center_sequence, center_sequence[1:]):
        total_steps += 1
        distance = compute_center_distance(list(previous["center"]), list(current["center"]))
        if distance < stationary_threshold:
            stationary_steps += 1
    if total_steps == 0:
        return 1.0
    return round(stationary_steps / total_steps, 3)


def classify_edge(center: list[float] | None, width: int, height: int) -> str:
    if center is None or width <= 0 or height <= 0:
        return "unknown"
    x, y = float(center[0]), float(center[1])
    x_margin = width * 0.1
    y_margin = height * 0.1
    if x <= x_margin:
        return "left_edge"
    if x >= width - x_margin:
        return "right_edge"
    if y <= y_margin:
        return "top_edge"
    if y >= height - y_margin:
        return "bottom_edge"
    return "inside_frame"


def compute_bbox_area_stats(
    bbox_sequence: list[dict[str, Any]],
    width: int,
    height: int,
    class_name: str,
) -> dict[str, Any]:
    if not bbox_sequence:
        return {
            "bbox_area_min": 0.0,
            "bbox_area_max": 0.0,
            "bbox_area_mean": 0.0,
            "approximate_size_level": "unknown",
        }
    areas = [compute_bbox_area(list(item["bbox_xyxy"])) for item in bbox_sequence]
    mean_area = sum(areas) / len(areas)
    frame_area = float(width * height) if width > 0 and height > 0 else 0.0

    if class_name == "bicycle" or class_name == "motorcycle":
        size_level = "small_vehicle" if class_name in VEHICLE_CLASSES else "small"
    elif class_name == "car":
        size_level = "medium_vehicle"
    elif class_name in {"bus", "truck"}:
        size_level = "heavy_vehicle"
    elif frame_area <= 0:
        size_level = "unknown"
    else:
        ratio = mean_area / frame_area
        if ratio < 0.03:
            size_level = "small"
        elif ratio < 0.12:
            size_level = "medium"
        else:
            size_level = "large"

    return {
        "bbox_area_min": round(min(areas), 3),
        "bbox_area_max": round(max(areas), 3),
        "bbox_area_mean": round(mean_area, 3),
        "approximate_size_level": size_level,
    }


def infer_attribute_confidence(track: dict[str, Any]) -> float:
    count_for_summary = track.get("count_for_summary")
    if count_for_summary == "review" and bool(track.get("needs_review")):
        return 0.6
    if count_for_summary is True:
        return 0.85
    return 0.35


def build_helper_candidates(
    track: dict[str, Any],
    *,
    detection_lookup: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    carried_candidates: list[dict[str, Any]] = []
    nearby_candidates: list[dict[str, Any]] = []
    bbox_by_frame = {
        str(item.get("frame_id") or ""): item
        for item in list(track.get("bbox_sequence") or [])
        if str(item.get("frame_id") or "")
    }
    for frame_id, bbox_item in bbox_by_frame.items():
        person_box = list(bbox_item.get("bbox_xyxy") or [])
        if len(person_box) < 4:
            continue
        px1, py1, px2, py2 = [float(value) for value in person_box]
        person_area = max(1.0, compute_bbox_area(person_box))
        person_center = [(px1 + px2) / 2.0, (py1 + py2) / 2.0]
        for detection in detection_lookup.get(frame_id, []):
            class_name = str(detection.get("class_name") or "").lower()
            if class_name not in HELPER_CLASSES:
                continue
            bbox_xyxy = detection.get("bbox_xyxy")
            if not isinstance(bbox_xyxy, list) or len(bbox_xyxy) < 4:
                continue
            dx1, dy1, dx2, dy2 = [float(value) for value in bbox_xyxy[:4]]
            det_center = [(dx1 + dx2) / 2.0, (dy1 + dy2) / 2.0]
            distance = compute_center_distance(person_center, det_center)
            horizontal_close = abs(det_center[0] - person_center[0]) <= max(px2 - px1, dx2 - dx1) * 0.8
            vertical_close = abs(det_center[1] - person_center[1]) <= max(py2 - py1, dy2 - dy1) * 0.8
            overlaps = not (dx2 < px1 or dx1 > px2 or dy2 < py1 or dy1 > py2)
            if not overlaps and not (horizontal_close and vertical_close):
                continue

            helper_area = compute_bbox_area([dx1, dy1, dx2, dy2])
            overlap_area = max(0.0, min(px2, dx2) - max(px1, dx1)) * max(0.0, min(py2, dy2) - max(py1, dy1))
            overlap_ratio = overlap_area / max(1.0, min(person_area, helper_area))
            proximity_score = max(0.1, min(1.0, 1.0 - distance / max(px2 - px1, py2 - py1, 1.0)))
            confidence = round(max(overlap_ratio, proximity_score), 3)
            candidate = {
                "label": f"possible_{class_name}",
                "confidence": confidence,
                "frame_id": frame_id,
            }
            if candidate["label"] not in {item["label"] for item in carried_candidates}:
                carried_candidates.append(candidate)
            nearby_candidate = {
                "class_name": class_name,
                "confidence": confidence,
                "frame_id": frame_id,
            }
            if nearby_candidate["class_name"] not in {item["class_name"] for item in nearby_candidates}:
                nearby_candidates.append(nearby_candidate)
    return carried_candidates, nearby_candidates


def build_motion_attributes(
    track: dict[str, Any],
    *,
    width: int,
    height: int,
    stationary_distance_ratio: float,
    slow_speed_threshold: float,
    fast_speed_threshold: float,
    warnings: list[str],
) -> dict[str, Any]:
    center_sequence = list(track.get("center_sequence") or [])
    if not center_sequence:
        warnings.append("Missing center_sequence for one or more tracks.")
        return {
            "movement_direction": "unknown",
            "movement_direction_confidence": 0.0,
            "speed_pixels_per_second": None,
            "speed_level": "unknown",
            "motion_status": "unknown",
            "start_edge": "unknown",
            "end_edge": "unknown",
            "path_summary": "unknown_path",
            "start_center": None,
            "end_center": None,
            "movement_distance_pixels": 0.0,
            "stationary_ratio": 1.0,
        }

    start_center = list(center_sequence[0]["center"])
    end_center = list(center_sequence[-1]["center"])
    dx = round(float(end_center[0]) - float(start_center[0]), 3)
    dy = round(float(end_center[1]) - float(start_center[1]), 3)
    movement_distance = round(compute_center_distance(start_center, end_center), 3)
    frame_diagonal = math.hypot(width, height) if width > 0 and height > 0 else 0.0
    stationary_threshold = (
        frame_diagonal * stationary_distance_ratio
        if frame_diagonal > 0
        else max(15.0, stationary_distance_ratio * 100.0)
    )
    if frame_diagonal <= 0:
        warnings.append("Missing frame size for one or more tracks.")
    movement_direction, movement_direction_confidence = classify_direction(
        dx,
        dy,
        movement_distance,
        stationary_threshold,
    )
    duration_seconds = float(track.get("duration_seconds") or 0.0)
    speed_pixels_per_second = (
        round(movement_distance / duration_seconds, 3) if duration_seconds > 0 else None
    )
    stationary_ratio = compute_stationary_ratio(center_sequence, stationary_threshold)
    speed_level = compute_speed_level(speed_pixels_per_second, slow_speed_threshold, fast_speed_threshold)
    motion_status = "stationary" if movement_direction == "mostly_stationary" else "moving"
    start_edge = classify_edge(start_center, width, height)
    end_edge = classify_edge(end_center, width, height)
    return {
        "movement_direction": movement_direction,
        "movement_direction_confidence": movement_direction_confidence,
        "speed_pixels_per_second": speed_pixels_per_second,
        "speed_level": speed_level,
        "motion_status": motion_status,
        "start_edge": start_edge,
        "end_edge": end_edge,
        "path_summary": f"{start_edge}_to_{end_edge}",
        "start_center": [round(float(start_center[0]), 3), round(float(start_center[1]), 3)],
        "end_center": [round(float(end_center[0]), 3), round(float(end_center[1]), 3)],
        "movement_distance_pixels": movement_distance,
        "stationary_ratio": stationary_ratio,
    }


def build_common_attribute(
    *,
    attribute_track_id: str,
    track: dict[str, Any],
    object_group: str,
    crop_path: str | None,
    crop_status: str,
    motion_attributes: dict[str, Any],
    bbox_area_stats: dict[str, Any],
) -> dict[str, Any]:
    return {
        "attribute_track_id": attribute_track_id,
        "source_track_id": track["source_track_id"],
        "class_name": track["class_name"],
        "class_evidence": {
            "dominant_class_name": track.get("dominant_class_name"),
            "dominant_class_confidence": track.get("dominant_class_confidence"),
            "class_votes": dict(track.get("class_votes") or {}),
            "class_confidence_sum": dict(track.get("class_confidence_sum") or {}),
            "class_consistency_score": track.get("class_consistency_score"),
            "class_conflict": bool(track.get("class_conflict")),
        },
        "object_group": object_group,
        "chunk_id": track["chunk_id"],
        "start_time": track["start_time"],
        "end_time": track["end_time"],
        "duration_seconds": track["duration_seconds"],
        "detection_count": track["detection_count"],
        "cleanup_status": track["cleanup_status"],
        "count_for_summary": track["count_for_summary"],
        "needs_review": bool(track.get("needs_review")),
        "attribute_confidence": infer_attribute_confidence(track),
        "best_frame_id": track["best_frame_id"],
        "best_image_path": track["best_image_path"],
        "thumbnail_path": track["thumbnail_path"],
        "attribute_crop_path": crop_path,
        "crop_status": crop_status,
        "movement_direction": motion_attributes["movement_direction"],
        "movement_direction_confidence": motion_attributes["movement_direction_confidence"],
        "speed_pixels_per_second": motion_attributes["speed_pixels_per_second"],
        "speed_level": motion_attributes["speed_level"],
        "motion_status": motion_attributes["motion_status"],
        "start_edge": motion_attributes["start_edge"],
        "end_edge": motion_attributes["end_edge"],
        "path_summary": motion_attributes["path_summary"],
        "bbox_area_min": bbox_area_stats["bbox_area_min"],
        "bbox_area_max": bbox_area_stats["bbox_area_max"],
        "bbox_area_mean": bbox_area_stats["bbox_area_mean"],
        "approximate_size_level": bbox_area_stats["approximate_size_level"],
    }


def build_person_attributes(
    *,
    person_crop: Any | None,
    person_crop_path: str | None,
    detection_lookup: dict[str, list[dict[str, Any]]],
    track: dict[str, Any],
    warnings: list[str],
) -> tuple[dict[str, Any], dict[str, int]]:
    upper_crop, lower_crop = split_person_crop(person_crop)
    upper_color, _, upper_conf, upper_candidates = extract_color_candidates(upper_crop, mode="person")
    lower_color, _, lower_conf, lower_candidates = extract_color_candidates(lower_crop, mode="person")
    if upper_color == "unknown" or lower_color == "unknown":
        warnings.append(f"Color extraction failed for person track {track['source_track_id']}.")

    carried_candidates, nearby_candidates = build_helper_candidates(
        track,
        detection_lookup=detection_lookup,
    )
    dominant_colors = []
    for color_name in [upper_color, lower_color]:
        if color_name != "unknown" and color_name not in dominant_colors:
            dominant_colors.append(color_name)
    possible_clothing_colors = []
    for candidate in upper_candidates + lower_candidates:
        if candidate != "unknown" and candidate not in possible_clothing_colors:
            possible_clothing_colors.append(candidate)

    person_attributes = {
        "upper_clothing_color": upper_color,
        "upper_clothing_color_confidence": upper_conf,
        "lower_clothing_color": lower_color,
        "lower_clothing_color_confidence": lower_conf,
        "dominant_person_colors": dominant_colors,
        "possible_clothing_colors": possible_clothing_colors[:3],
        "clothing_color_method": "upper_lower_crop_split",
        "carried_object_candidates": carried_candidates,
        "nearby_object_candidates": nearby_candidates,
        "person_crop_path": person_crop_path,
        "attribute_notes": [],
    }
    return person_attributes, {
        "color_success": 1 if upper_color != "unknown" or lower_color != "unknown" else 0,
        "color_failed": 1 if upper_color == "unknown" and lower_color == "unknown" else 0,
        "carried_object_candidate_count": len(carried_candidates),
    }


def vehicle_type_category(class_name: str) -> tuple[str, str]:
    mapping = {
        "bicycle": ("bicycle", "two_wheeler_non_motor"),
        "motorcycle": ("motorcycle", "two_wheeler_motor"),
        "car": ("car", "light_vehicle"),
        "bus": ("bus", "heavy_passenger_vehicle"),
        "truck": ("truck", "heavy_goods_vehicle"),
    }
    return mapping.get(class_name, (class_name, "unknown"))


def vehicle_size_category(class_name: str, bbox_size_level: str) -> str:
    if class_name in {"bicycle", "motorcycle"}:
        return "small_vehicle"
    if class_name == "car":
        return "medium_vehicle"
    if class_name in {"bus", "truck"}:
        return "heavy_vehicle"
    return bbox_size_level


def build_vehicle_attributes(
    *,
    vehicle_crop: Any | None,
    track: dict[str, Any],
    attribute_track_id: str,
    plate_dir: Path,
    motion_direction: str,
    bbox_size_level: str,
    warnings: list[str],
) -> tuple[dict[str, Any], dict[str, int]]:
    vehicle_focus_crop = crop_center_region(vehicle_crop, x_ratio=0.18, y_ratio=0.18)
    vehicle_color, _, vehicle_color_conf, vehicle_color_candidates = extract_color_candidates(
        vehicle_focus_crop,
        mode="vehicle",
    )
    if vehicle_color == "unknown":
        warnings.append(f"Color extraction failed for vehicle track {attribute_track_id}.")

    plate_crop_path, plate_crop_status, anpr_ready = build_plate_candidate_crop(
        attribute_track_id,
        track,
        plate_dir=plate_dir,
        warnings=warnings,
    )
    vehicle_type, vehicle_category = vehicle_type_category(str(track["class_name"]))
    vehicle_attributes = {
        "vehicle_type": vehicle_type,
        "vehicle_category": vehicle_category,
        "vehicle_color": vehicle_color,
        "vehicle_color_confidence": vehicle_color_conf,
        "vehicle_color_candidates": vehicle_color_candidates[:3],
        "vehicle_size_category": vehicle_size_category(str(track["class_name"]), bbox_size_level),
        "vehicle_motion_direction": motion_direction,
        "possible_plate_crop_path": plate_crop_path,
        "plate_crop_status": plate_crop_status,
        "plate_ocr_status": "not_run",
        "anpr_ready": anpr_ready,
        "attribute_notes": [],
    }
    return vehicle_attributes, {
        "color_success": 1 if vehicle_color != "unknown" else 0,
        "color_failed": 1 if vehicle_color == "unknown" else 0,
        "plate_candidate_crop_count": 1 if anpr_ready else 0,
    }


def build_basic_attributes(
    *,
    object_crop: Any | None,
    track: dict[str, Any],
    warnings: list[str],
) -> tuple[dict[str, Any], dict[str, int]]:
    dominant_color, _, _, color_candidates = extract_color_candidates(object_crop, mode="person")
    if dominant_color == "unknown":
        warnings.append(f"Color extraction failed for track {track['source_track_id']}.")
    basic_attributes = {
        "dominant_color": dominant_color,
        "color_candidates": color_candidates[:3],
        "motion_direction": "unknown",
        "speed_level": "unknown",
        "attribute_notes": [],
    }
    return basic_attributes, {
        "color_success": 1 if dominant_color != "unknown" else 0,
        "color_failed": 1 if dominant_color == "unknown" else 0,
    }


def build_attribute_for_track(
    index: int,
    track: dict[str, Any],
    *,
    frame_lookup: dict[str, dict[str, Any]],
    detection_lookup: dict[str, list[dict[str, Any]]],
    video_info_payload: dict[str, Any] | None,
    objects_dir: Path,
    plate_dir: Path,
    settings: dict[str, float],
    warnings: list[str],
) -> tuple[dict[str, Any], dict[str, int]]:
    attribute_track_id = (
        f"attr_{track['source_track_id']}" if track["source_track_id"] else f"attr_track_{index:06d}"
    )
    width, height = infer_frame_dimensions(track, frame_lookup, video_info_payload)
    object_group = detect_object_group(str(track["class_name"]))
    crop_path, crop_status, object_crop = build_attribute_crop(
        attribute_track_id,
        track,
        objects_dir=objects_dir,
        warnings=warnings,
    )
    motion_attributes = build_motion_attributes(
        track,
        width=width,
        height=height,
        stationary_distance_ratio=settings["stationary_distance_ratio"],
        slow_speed_threshold=settings["slow_speed_threshold"],
        fast_speed_threshold=settings["fast_speed_threshold"],
        warnings=warnings,
    )
    bbox_area_stats = compute_bbox_area_stats(
        list(track.get("bbox_sequence") or []),
        width,
        height,
        str(track["class_name"]),
    )

    attribute = build_common_attribute(
        attribute_track_id=attribute_track_id,
        track=track,
        object_group=object_group,
        crop_path=crop_path,
        crop_status=crop_status,
        motion_attributes=motion_attributes,
        bbox_area_stats=bbox_area_stats,
    )

    counters = {
        "person_attribute_count": 0,
        "vehicle_attribute_count": 0,
        "basic_attribute_count": 0,
        "person_color_success_count": 0,
        "vehicle_color_success_count": 0,
        "plate_candidate_crop_count": 0,
        "carried_object_candidate_count": 0,
        "unknown_color_count": 0,
    }

    if str(track["class_name"]) == "person":
        person_attributes, person_counters = build_person_attributes(
            person_crop=object_crop,
            person_crop_path=crop_path,
            detection_lookup=detection_lookup,
            track=track,
            warnings=warnings,
        )
        attribute["person_attributes"] = person_attributes
        counters["person_attribute_count"] = 1
        counters["person_color_success_count"] = person_counters["color_success"]
        counters["carried_object_candidate_count"] = person_counters["carried_object_candidate_count"]
        if person_counters["color_failed"] > 0:
            counters["unknown_color_count"] += 1
    elif str(track["class_name"]) in VEHICLE_CLASSES:
        vehicle_attributes, vehicle_counters = build_vehicle_attributes(
            vehicle_crop=object_crop,
            track=track,
            attribute_track_id=attribute_track_id,
            plate_dir=plate_dir,
            motion_direction=motion_attributes["movement_direction"],
            bbox_size_level=bbox_area_stats["approximate_size_level"],
            warnings=warnings,
        )
        attribute["vehicle_attributes"] = vehicle_attributes
        counters["vehicle_attribute_count"] = 1
        counters["vehicle_color_success_count"] = vehicle_counters["color_success"]
        counters["plate_candidate_crop_count"] = vehicle_counters["plate_candidate_crop_count"]
        if vehicle_counters["color_failed"] > 0:
            counters["unknown_color_count"] += 1
    else:
        basic_attributes, basic_counters = build_basic_attributes(
            object_crop=object_crop,
            track=track,
            warnings=warnings,
        )
        basic_attributes["motion_direction"] = motion_attributes["movement_direction"]
        basic_attributes["speed_level"] = motion_attributes["speed_level"]
        attribute["basic_attributes"] = basic_attributes
        counters["basic_attribute_count"] = 1
        if basic_counters["color_failed"] > 0:
            counters["unknown_color_count"] += 1

    summary_ready = (
        track.get("count_for_summary") is True
        or (track.get("count_for_summary") == "review" and bool(track.get("needs_review")))
    )
    attribute["summary_ready"] = summary_ready
    return attribute, counters


def determine_next_step(attributes: list[dict[str, Any]]) -> str:
    has_person = any(
        str(item.get("class_name")) == "person" and bool(item.get("summary_ready"))
        for item in attributes
    )
    has_vehicle = any(
        str(item.get("object_group")) == "vehicle" and bool(item.get("summary_ready"))
        for item in attributes
    )
    if has_person and has_vehicle:
        return "07_plate_or_text_ocr_and_event_candidate_generation"
    if has_vehicle:
        return "07_plate_or_text_ocr"
    return "07_event_candidate_generation"


def update_run_manifest_for_attributes(
    run_manifest_path: Path,
    attributes: list[dict[str, Any]],
) -> dict[str, Any]:
    run_manifest = read_json(run_manifest_path)
    completed_steps = list(run_manifest.get("completed_steps", []))
    if "06_attribute_extraction" not in completed_steps:
        completed_steps.append("06_attribute_extraction")
    run_manifest["completed_steps"] = completed_steps
    run_manifest["next_step"] = determine_next_step(attributes)
    write_json(run_manifest_path, run_manifest)
    return run_manifest


def build_attribute_outputs(run_dir: Path) -> dict[str, Any]:
    warnings: list[str] = []
    selected_track_source, source_tracks, _ = detect_track_source(run_dir, warnings)
    cleanup_report_payload = read_optional_json(run_dir / "05B_track_cleanup_report.json")
    tracking_focus_payload = read_optional_json(run_dir / "05_tracking_focus.json")
    detections_payload = read_optional_json(run_dir / "04_yolo_detections.json")
    frames_index_payload = read_optional_json(run_dir / "03_sampled_frames_index.json")
    video_info_payload = read_optional_json(run_dir / "01_video_info.json")

    if cleanup_report_payload is None:
        warnings.append("Optional file missing: 05B_track_cleanup_report.json")
    if tracking_focus_payload is None:
        warnings.append("Optional file missing: 05_tracking_focus.json")
    if detections_payload is None:
        warnings.append("Optional file missing: 04_yolo_detections.json")
    if frames_index_payload is None:
        warnings.append("Optional file missing: 03_sampled_frames_index.json")
    if video_info_payload is None:
        warnings.append("Optional file missing: 01_video_info.json")

    frame_lookup = build_frame_lookup(frames_index_payload)
    detection_lookup = build_detection_lookup(detections_payload)

    settings = {
        "stationary_distance_ratio": round(
            read_non_negative_float_env(
                ENV_FINAL_DEMO_ATTRIBUTE_STATIONARY_DISTANCE_RATIO,
                DEFAULT_STATIONARY_DISTANCE_RATIO,
            ),
            3,
        ),
        "slow_speed_threshold": round(
            read_positive_float_env(
                ENV_FINAL_DEMO_ATTRIBUTE_SPEED_SLOW_PX_PER_SEC,
                DEFAULT_SPEED_SLOW_PX_PER_SEC,
            ),
            3,
        ),
        "fast_speed_threshold": round(
            read_positive_float_env(
                ENV_FINAL_DEMO_ATTRIBUTE_SPEED_FAST_PX_PER_SEC,
                DEFAULT_SPEED_FAST_PX_PER_SEC,
            ),
            3,
        ),
    }

    crops_root = run_dir / "06_attribute_crops"
    objects_dir = crops_root / "objects"
    plate_dir = crops_root / "plate_candidates"
    objects_dir.mkdir(parents=True, exist_ok=True)
    plate_dir.mkdir(parents=True, exist_ok=True)

    normalized_tracks = [
        normalize_track(track, frame_lookup=frame_lookup, warnings=warnings)
        for track in source_tracks
        if isinstance(track, dict)
    ]

    attributes: list[dict[str, Any]] = []
    attributes_by_class: dict[str, int] = defaultdict(int)
    attributes_by_object_group: dict[str, int] = defaultdict(int)
    summary_ready_track_count = 0
    review_track_count = 0
    noise_track_count = 0
    person_attribute_count = 0
    vehicle_attribute_count = 0
    basic_attribute_count = 0
    person_color_success_count = 0
    vehicle_color_success_count = 0
    plate_candidate_crop_count = 0
    carried_object_candidate_count = 0
    unknown_color_count = 0

    for index, track in enumerate(normalized_tracks, start=1):
        attribute, counters = build_attribute_for_track(
            index,
            track,
            frame_lookup=frame_lookup,
            detection_lookup=detection_lookup,
            video_info_payload=video_info_payload,
            objects_dir=objects_dir,
            plate_dir=plate_dir,
            settings=settings,
            warnings=warnings,
        )
        attributes.append(attribute)
        attributes_by_class[str(attribute["class_name"])] += 1
        attributes_by_object_group[str(attribute["object_group"])] += 1
        if bool(attribute["summary_ready"]):
            summary_ready_track_count += 1
        if attribute["count_for_summary"] == "review":
            review_track_count += 1
        if attribute["count_for_summary"] is False:
            noise_track_count += 1
        person_attribute_count += counters["person_attribute_count"]
        vehicle_attribute_count += counters["vehicle_attribute_count"]
        basic_attribute_count += counters["basic_attribute_count"]
        person_color_success_count += counters["person_color_success_count"]
        vehicle_color_success_count += counters["vehicle_color_success_count"]
        plate_candidate_crop_count += counters["plate_candidate_crop_count"]
        carried_object_candidate_count += counters["carried_object_candidate_count"]
        unknown_color_count += counters["unknown_color_count"]

    if summary_ready_track_count == 0:
        warnings.append("No summary-ready tracks were available for Step 6.")

    recommendations = [
        "Use 05B clean tracks for all downstream summaries.",
        "Use zone configuration later for dwell/area analytics.",
        "For people search, use upper/lower clothing colors and carried object candidates.",
        "For vehicle search, use vehicle_type, vehicle_color, vehicle_category, and later OCR plate text.",
        "Do not use Step 6 for confirmed gender/sex identification.",
    ]
    if plate_candidate_crop_count > 0:
        recommendations.append("Run OCR/ANPR in the next step for vehicle plate crops.")
    if review_track_count > 0:
        recommendations.append("Review tracks with count_for_summary = \"review\".")

    attributes_payload = {
        "created_at": current_timestamp(),
        "selected_track_source": selected_track_source,
        "tracking_focus": tracking_focus_payload or {},
        "total_tracks_input": len(normalized_tracks),
        "total_attributes_created": len(attributes),
        "attributes": attributes,
        "warnings": list(dict.fromkeys(warnings)),
    }

    attribute_report_payload = {
        "created_at": current_timestamp(),
        "selected_track_source": selected_track_source,
        "tracking_focus_profile": (
            tracking_focus_payload.get("selected_focus_profile")
            if isinstance(tracking_focus_payload, dict)
            else None
        ),
        "total_tracks_input": len(normalized_tracks),
        "total_attributes_created": len(attributes),
        "attributes_by_class": dict(sorted(attributes_by_class.items())),
        "attributes_by_object_group": dict(sorted(attributes_by_object_group.items())),
        "summary_ready_track_count": summary_ready_track_count,
        "review_track_count": review_track_count,
        "noise_track_count": noise_track_count,
        "person_attribute_count": person_attribute_count,
        "vehicle_attribute_count": vehicle_attribute_count,
        "basic_attribute_count": basic_attribute_count,
        "person_color_success_count": person_color_success_count,
        "vehicle_color_success_count": vehicle_color_success_count,
        "plate_candidate_crop_count": plate_candidate_crop_count,
        "carried_object_candidate_count": carried_object_candidate_count,
        "unknown_color_count": unknown_color_count,
        "sensitive_attributes_estimated": False,
        "warnings": list(dict.fromkeys(warnings)),
        "recommendations": list(dict.fromkeys(recommendations)),
        "crop_output_dirs": {
            "objects": to_repo_relative_path(objects_dir),
            "plate_candidates": to_repo_relative_path(plate_dir),
        },
    }

    return {
        "attributes_payload": attributes_payload,
        "report_payload": attribute_report_payload,
        "crops_root": crops_root,
    }
