from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tests.final_demo.services.chunk_planner import read_json
from tests.final_demo.services.tracker_adapter import to_absolute_repo_path, to_repo_relative_path
from tests.final_demo.services.video_io import current_timestamp, write_json


ENV_FINAL_DEMO_PLATE_FRAMES_PER_TRACK = "FINAL_DEMO_PLATE_FRAMES_PER_TRACK"
ENV_FINAL_DEMO_PLATE_MAX_TRACKS = "FINAL_DEMO_PLATE_MAX_TRACKS"
ENV_FINAL_DEMO_PLATE_MIN_CANDIDATE_SCORE = "FINAL_DEMO_PLATE_MIN_CANDIDATE_SCORE"
ENV_FINAL_DEMO_PLATE_SAVE_DEBUG = "FINAL_DEMO_PLATE_SAVE_DEBUG"
ENV_FINAL_DEMO_PLATE_USE_ORIGINAL_FRAME = "FINAL_DEMO_PLATE_USE_ORIGINAL_FRAME"
ENV_FINAL_DEMO_PLATE_DETECTOR_MODEL = "FINAL_DEMO_PLATE_DETECTOR_MODEL"
ENV_FINAL_DEMO_PLATE_SCAN_MODE = "FINAL_DEMO_PLATE_SCAN_MODE"
ENV_FINAL_DEMO_PLATE_FRAME_SCAN_MAX_FRAMES = "FINAL_DEMO_PLATE_FRAME_SCAN_MAX_FRAMES"
ENV_FINAL_DEMO_PLATE_FRAME_SCAN_EVERY_N = "FINAL_DEMO_PLATE_FRAME_SCAN_EVERY_N"
ENV_FINAL_DEMO_PLATE_FRAME_SCAN_CLASSES = "FINAL_DEMO_PLATE_FRAME_SCAN_CLASSES"
ENV_FINAL_DEMO_PLATE_FRAME_SCAN_MIN_VEHICLE_CONF = "FINAL_DEMO_PLATE_FRAME_SCAN_MIN_VEHICLE_CONF"
ENV_FINAL_DEMO_PLATE_FRAME_SCAN_TOPK_PER_FRAME = "FINAL_DEMO_PLATE_FRAME_SCAN_TOPK_PER_FRAME"
ENV_FINAL_DEMO_PLATE_FRAME_SCAN_DEDUP_IOU = "FINAL_DEMO_PLATE_FRAME_SCAN_DEDUP_IOU"
ENV_FINAL_DEMO_PLATE_FRAME_SCAN_TRACK_ASSOC_IOU = "FINAL_DEMO_PLATE_FRAME_SCAN_TRACK_ASSOC_IOU"
ENV_FINAL_DEMO_PLATE_FRAME_SCAN_TRACK_ASSOC_MAX_TIME_DELTA = "FINAL_DEMO_PLATE_FRAME_SCAN_TRACK_ASSOC_MAX_TIME_DELTA"

DEFAULT_PLATE_FRAMES_PER_TRACK = 5
DEFAULT_PLATE_MIN_CANDIDATE_SCORE = 0.45
DEFAULT_PLATE_SAVE_DEBUG = True
DEFAULT_PLATE_USE_ORIGINAL_FRAME = False
DEFAULT_PLATE_SCAN_MODE = "track"
DEFAULT_PLATE_FRAME_SCAN_MAX_FRAMES = 150
DEFAULT_PLATE_FRAME_SCAN_EVERY_N = 1
DEFAULT_PLATE_FRAME_SCAN_CLASSES = "car,bus,truck,motorcycle"
DEFAULT_PLATE_FRAME_SCAN_MIN_VEHICLE_CONF = 0.10
DEFAULT_PLATE_FRAME_SCAN_TOPK_PER_FRAME = 3
DEFAULT_PLATE_FRAME_SCAN_DEDUP_IOU = 0.50
DEFAULT_PLATE_FRAME_SCAN_TRACK_ASSOC_IOU = 0.30
DEFAULT_PLATE_FRAME_SCAN_TRACK_ASSOC_MAX_TIME_DELTA = 0.50
VEHICLE_CLASSES = {"bicycle", "motorcycle", "car", "bus", "truck"}
ALLOWED_SCAN_MODES = {"track", "frame", "hybrid"}


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def read_bool_env(env_name: str, default_value: bool) -> bool:
    raw_value = os.environ.get(env_name)
    if raw_value is None or raw_value.strip() == "":
        return default_value
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Environment variable {env_name} must be boolean-like. Received: {raw_value!r}")


def read_positive_int_env(env_name: str, default_value: int) -> int:
    raw_value = os.environ.get(env_name, str(default_value))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {env_name} must be a valid integer. Received: {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"Environment variable {env_name} must be greater than 0. Received: {value}")
    return value


def read_optional_positive_int_env(env_name: str) -> int | None:
    raw_value = os.environ.get(env_name)
    if raw_value is None or raw_value.strip() == "":
        return None
    return read_positive_int_env(env_name, 1)


def read_non_negative_float_env(env_name: str, default_value: float) -> float:
    raw_value = os.environ.get(env_name, str(default_value))
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {env_name} must be a valid number. Received: {raw_value!r}") from exc
    if value < 0:
        raise ValueError(f"Environment variable {env_name} must be greater than or equal to 0. Received: {value}")
    return value


def read_detector_settings() -> dict[str, Any]:
    scan_mode = os.environ.get(ENV_FINAL_DEMO_PLATE_SCAN_MODE, DEFAULT_PLATE_SCAN_MODE).strip().lower()
    if scan_mode not in ALLOWED_SCAN_MODES:
        raise ValueError(
            f"Environment variable {ENV_FINAL_DEMO_PLATE_SCAN_MODE} must be one of "
            f"{sorted(ALLOWED_SCAN_MODES)}. Received: {scan_mode!r}"
        )
    frame_scan_classes = [
        item.strip().lower()
        for item in os.environ.get(
            ENV_FINAL_DEMO_PLATE_FRAME_SCAN_CLASSES,
            DEFAULT_PLATE_FRAME_SCAN_CLASSES,
        ).split(",")
        if item.strip()
    ]
    return {
        "frames_per_track": read_positive_int_env(
            ENV_FINAL_DEMO_PLATE_FRAMES_PER_TRACK,
            DEFAULT_PLATE_FRAMES_PER_TRACK,
        ),
        "max_tracks": read_optional_positive_int_env(ENV_FINAL_DEMO_PLATE_MAX_TRACKS),
        "min_candidate_score": round(
            read_non_negative_float_env(
                ENV_FINAL_DEMO_PLATE_MIN_CANDIDATE_SCORE,
                DEFAULT_PLATE_MIN_CANDIDATE_SCORE,
            ),
            3,
        ),
        "save_debug": read_bool_env(ENV_FINAL_DEMO_PLATE_SAVE_DEBUG, DEFAULT_PLATE_SAVE_DEBUG),
        "use_original_frame": read_bool_env(
            ENV_FINAL_DEMO_PLATE_USE_ORIGINAL_FRAME,
            DEFAULT_PLATE_USE_ORIGINAL_FRAME,
        ),
        "detector_model_path": os.environ.get(ENV_FINAL_DEMO_PLATE_DETECTOR_MODEL, "").strip(),
        "plate_scan_mode": scan_mode,
        "frame_scan_max_frames": read_positive_int_env(
            ENV_FINAL_DEMO_PLATE_FRAME_SCAN_MAX_FRAMES,
            DEFAULT_PLATE_FRAME_SCAN_MAX_FRAMES,
        ),
        "frame_scan_every_n": read_positive_int_env(
            ENV_FINAL_DEMO_PLATE_FRAME_SCAN_EVERY_N,
            DEFAULT_PLATE_FRAME_SCAN_EVERY_N,
        ),
        "frame_scan_classes": frame_scan_classes,
        "frame_scan_min_vehicle_conf": round(
            read_non_negative_float_env(
                ENV_FINAL_DEMO_PLATE_FRAME_SCAN_MIN_VEHICLE_CONF,
                DEFAULT_PLATE_FRAME_SCAN_MIN_VEHICLE_CONF,
            ),
            3,
        ),
        "frame_scan_topk_per_frame": read_positive_int_env(
            ENV_FINAL_DEMO_PLATE_FRAME_SCAN_TOPK_PER_FRAME,
            DEFAULT_PLATE_FRAME_SCAN_TOPK_PER_FRAME,
        ),
        "frame_scan_dedup_iou": round(
            read_non_negative_float_env(
                ENV_FINAL_DEMO_PLATE_FRAME_SCAN_DEDUP_IOU,
                DEFAULT_PLATE_FRAME_SCAN_DEDUP_IOU,
            ),
            3,
        ),
        "frame_scan_track_assoc_iou": round(
            read_non_negative_float_env(
                ENV_FINAL_DEMO_PLATE_FRAME_SCAN_TRACK_ASSOC_IOU,
                DEFAULT_PLATE_FRAME_SCAN_TRACK_ASSOC_IOU,
            ),
            3,
        ),
        "frame_scan_track_assoc_max_time_delta": round(
            read_non_negative_float_env(
                ENV_FINAL_DEMO_PLATE_FRAME_SCAN_TRACK_ASSOC_MAX_TIME_DELTA,
                DEFAULT_PLATE_FRAME_SCAN_TRACK_ASSOC_MAX_TIME_DELTA,
            ),
            3,
        ),
    }


def build_track_lookup(tracks_payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    if not isinstance(tracks_payload, dict):
        return lookup
    for track in list(tracks_payload.get("clean_tracks") or tracks_payload.get("tracks") or []):
        if not isinstance(track, dict):
            continue
        track_id = str(track.get("clean_track_id") or track.get("local_track_id") or track.get("source_track_id") or "")
        if track_id:
            lookup[track_id] = track
    return lookup


def build_frame_lookup(frames_payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    if not isinstance(frames_payload, dict):
        return lookup
    for frame in list(frames_payload.get("frames") or []):
        if not isinstance(frame, dict):
            continue
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
    for frame_id in list(lookup.keys()):
        lookup[frame_id].sort(
            key=lambda item: float(item.get("confidence") or 0.0),
            reverse=True,
        )
    return lookup


def build_clean_track_lookup(tracks_payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    if not isinstance(tracks_payload, dict):
        return lookup
    for track in list(tracks_payload.get("clean_tracks") or tracks_payload.get("tracks") or []):
        if not isinstance(track, dict):
            continue
        track_id = str(track.get("clean_track_id") or track.get("local_track_id") or track.get("source_track_id") or "")
        if track_id:
            lookup[track_id] = track
    return lookup


def build_attribute_lookup(attributes_payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    if not isinstance(attributes_payload, dict):
        return lookup
    for attribute in list(attributes_payload.get("attributes") or []):
        if not isinstance(attribute, dict):
            continue
        source_track_id = str(attribute.get("source_track_id") or "")
        if source_track_id:
            lookup[source_track_id] = attribute
    return lookup


def to_float_bbox(values: list[Any] | None) -> list[float]:
    if not values or len(values) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(values[0]), float(values[1]), float(values[2]), float(values[3])]


def compute_iou(box_a: list[float] | None, box_b: list[float] | None) -> float:
    if not box_a or not box_b or len(box_a) < 4 or len(box_b) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a[:4]
    bx1, by1, bx2, by2 = box_b[:4]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return round(intersection / union, 6)


def offset_bbox(bbox_xyxy: list[int], offset_x: float, offset_y: float) -> list[int]:
    return [
        int(round(float(bbox_xyxy[0]) + offset_x)),
        int(round(float(bbox_xyxy[1]) + offset_y)),
        int(round(float(bbox_xyxy[2]) + offset_x)),
        int(round(float(bbox_xyxy[3]) + offset_y)),
    ]


def find_nearest_track_bbox(
    track: dict[str, Any],
    *,
    timestamp: float,
    max_time_delta: float,
) -> tuple[list[float] | None, float | None]:
    best_bbox = None
    best_delta = None
    for item in list(track.get("bbox_sequence") or []):
        if not isinstance(item, dict):
            continue
        bbox_xyxy = item.get("bbox_xyxy")
        if not isinstance(bbox_xyxy, list) or len(bbox_xyxy) < 4:
            continue
        item_timestamp = float(item.get("timestamp") or 0.0)
        delta = abs(item_timestamp - timestamp)
        if delta > max_time_delta:
            continue
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_bbox = [float(value) for value in bbox_xyxy[:4]]
    return best_bbox, best_delta


def match_frame_scan_detection_to_track(
    *,
    detection: dict[str, Any],
    timestamp: float,
    clean_track_lookup: dict[str, dict[str, Any]],
    attribute_lookup: dict[str, dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    best_match = {
        "matched_source_track_id": None,
        "matched_attribute_track_id": None,
        "matched_track_class_name": None,
        "matched_track_vehicle_type": None,
        "matched_track_iou": 0.0,
        "matched_track_time_delta": None,
        "class_source": "source_yolo_detection",
    }
    detection_bbox = to_float_bbox(list(detection.get("bbox_xyxy") or []))
    detection_chunk_id = str(detection.get("chunk_id") or "")
    best_score = 0.0
    for track_id, track in clean_track_lookup.items():
        if str(track.get("chunk_id") or "") != detection_chunk_id:
            continue
        track_bbox, time_delta = find_nearest_track_bbox(
            track,
            timestamp=timestamp,
            max_time_delta=float(settings["frame_scan_track_assoc_max_time_delta"]),
        )
        if track_bbox is None or time_delta is None:
            continue
        iou = compute_iou(detection_bbox, track_bbox)
        if iou < float(settings["frame_scan_track_assoc_iou"]):
            continue
        if iou > best_score:
            attribute = attribute_lookup.get(track_id, {})
            vehicle_attributes = dict(attribute.get("vehicle_attributes") or {})
            best_score = iou
            best_match = {
                "matched_source_track_id": track_id,
                "matched_attribute_track_id": str(attribute.get("attribute_track_id") or "") or None,
                "matched_track_class_name": str(track.get("class_name") or "").lower() or None,
                "matched_track_vehicle_type": str(vehicle_attributes.get("vehicle_type") or track.get("class_name") or "").lower() or None,
                "matched_track_iou": round(iou, 4),
                "matched_track_time_delta": round(float(time_delta), 3),
                "class_source": "matched_clean_track",
            }
    return best_match


def select_track_frames(
    attribute: dict[str, Any],
    track_lookup: dict[str, dict[str, Any]],
    frame_lookup: dict[str, dict[str, Any]],
    frames_per_track: int,
) -> list[dict[str, Any]]:
    source_track_id = str(attribute.get("source_track_id") or "")
    track = track_lookup.get(source_track_id, {})
    bbox_sequence = list(track.get("bbox_sequence") or [])
    if not bbox_sequence:
        best_frame_id = str(attribute.get("best_frame_id") or "")
        best_image_path = attribute.get("best_image_path")
        if best_frame_id and best_image_path:
            return [{
                "frame_id": best_frame_id,
                "timestamp": round(float(attribute.get("start_time") or 0.0), 3),
                "bbox_xyxy": None,
                "image_path": str(best_image_path),
            }]
        return []

    preferred_indices = {0, len(bbox_sequence) // 2, len(bbox_sequence) - 1}
    step = max(1, len(bbox_sequence) // max(1, frames_per_track - 1))
    preferred_indices.update(range(0, len(bbox_sequence), step))

    selected: list[dict[str, Any]] = []
    seen_frame_ids: set[str] = set()
    for index in sorted(preferred_indices):
        if index < 0 or index >= len(bbox_sequence):
            continue
        item = bbox_sequence[index]
        frame_id = str(item.get("frame_id") or "")
        if not frame_id or frame_id in seen_frame_ids:
            continue
        seen_frame_ids.add(frame_id)
        frame_item = frame_lookup.get(frame_id, {})
        image_path = frame_item.get("image_path") or track.get("best_image_path") or attribute.get("best_image_path")
        selected.append(
            {
                "frame_id": frame_id,
                "timestamp": round(float(item.get("timestamp") or 0.0), 3),
                "bbox_xyxy": list(item.get("bbox_xyxy") or [])[:4],
                "image_path": str(image_path) if image_path else None,
            }
        )
        if len(selected) >= frames_per_track:
            break
    return selected


def load_frame_image(
    frame_info: dict[str, Any],
    *,
    settings: dict[str, Any],
    video_info_payload: dict[str, Any] | None,
    warnings: list[str],
) -> Any | None:
    if settings["use_original_frame"] and isinstance(video_info_payload, dict):
        video_path = Path(str(video_info_payload.get("video_path") or ""))
        if video_path.exists():
            capture = cv2.VideoCapture(str(video_path))
            try:
                capture.set(cv2.CAP_PROP_POS_MSEC, float(frame_info["timestamp"]) * 1000.0)
                success, frame = capture.read()
                if success and frame is not None:
                    return frame
                warnings.append(
                    f"Original frame read failed at {float(frame_info['timestamp']):.3f}s; using sampled frame fallback."
                )
            finally:
                capture.release()

    image_path = frame_info.get("image_path")
    if not image_path:
        return None
    return cv2.imread(str(to_absolute_repo_path(str(image_path))))


def crop_vehicle_from_frame(frame: Any, bbox_xyxy: list[Any] | None) -> Any | None:
    if frame is None or bbox_xyxy is None or len(bbox_xyxy) < 4:
        return None
    height, width = frame.shape[:2]
    x1 = max(0, min(width, int(round(float(bbox_xyxy[0])))))
    y1 = max(0, min(height, int(round(float(bbox_xyxy[1])))))
    x2 = max(0, min(width, int(round(float(bbox_xyxy[2])))))
    y2 = max(0, min(height, int(round(float(bbox_xyxy[3])))))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    return crop


def score_contour_candidate(
    contour: Any,
    *,
    vehicle_crop: Any,
    gray: Any,
) -> tuple[float, list[int], str] | None:
    x, y, w, h = cv2.boundingRect(contour)
    if w < 24 or h < 10:
        return None

    crop_height, crop_width = vehicle_crop.shape[:2]
    aspect_ratio = w / max(1.0, float(h))
    if aspect_ratio < 2.0 or aspect_ratio > 6.5:
        return None

    area_ratio = (w * h) / max(1.0, float(crop_width * crop_height))
    if area_ratio < 0.005 or area_ratio > 0.25:
        return None

    contour_area = cv2.contourArea(contour)
    rectangle_score = contour_area / max(1.0, float(w * h))

    aspect_score = 1.0 - min(abs(aspect_ratio - 4.2) / 4.2, 1.0)
    region = gray[y : y + h, x : x + w]
    contrast_score = min(1.0, float(region.std()) / 64.0) if region.size else 0.0
    size_score = min(1.0, area_ratio / 0.08)
    center_y = (y + h / 2.0) / max(1.0, float(crop_height))
    position_score = 1.0 - min(abs(center_y - 0.70) / 0.70, 1.0)
    plate_candidate_score = round(
        aspect_score * 0.30
        + rectangle_score * 0.25
        + contrast_score * 0.20
        + size_score * 0.15
        + position_score * 0.10,
        3,
    )
    reason = "wide rectangular high-contrast region"
    return plate_candidate_score, [int(x), int(y), int(x + w), int(y + h)], reason


def detect_candidates_opencv(vehicle_crop: Any) -> list[dict[str, Any]]:
    gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
    if max(vehicle_crop.shape[:2]) < 220:
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        vehicle_crop = cv2.resize(vehicle_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.bilateralFilter(gray, 9, 50, 50)
    edges = cv2.Canny(blurred, 75, 180)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(morph, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[dict[str, Any]] = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(approx) < 4 or len(approx) > 8:
            continue
        scored = score_contour_candidate(contour, vehicle_crop=vehicle_crop, gray=gray)
        if scored is None:
            continue
        score_value, bbox_xyxy, reason = scored
        candidates.append(
            {
                "plate_candidate_score": score_value,
                "bbox_in_vehicle_crop": bbox_xyxy,
                "reason": reason,
                "method": "opencv_plate_heuristic",
            }
        )
    candidates.sort(key=lambda item: float(item["plate_candidate_score"]), reverse=True)
    return candidates[:10]


def maybe_load_plate_detector(settings: dict[str, Any], warnings: list[str]) -> tuple[str, Any | None]:
    raw_model_path = str(settings["detector_model_path"] or "").strip()
    if not raw_model_path:
        return "opencv_plate_heuristic", None
    model_path = Path(raw_model_path).expanduser()
    if not model_path.exists():
        warnings.append(
            f"Plate detector model path does not exist: {model_path}. Falling back to OpenCV heuristic."
        )
        return "opencv_plate_heuristic", None
    if not model_path.is_file():
        warnings.append(
            f"Plate detector model path is not a file: {model_path}. Falling back to OpenCV heuristic."
        )
        return "opencv_plate_heuristic", None
    allowed_suffixes = {".pt", ".onnx", ".engine", ".xml", ".tflite", ".pb", ".pth"}
    if model_path.suffix.lower() not in allowed_suffixes:
        warnings.append(
            f"Plate detector model file does not look like a supported model artifact: {model_path.name}. "
            "Falling back to OpenCV heuristic."
        )
        return "opencv_plate_heuristic", None
    try:
        from ultralytics import YOLO  # type: ignore

        return "yolo_plate_detector", YOLO(str(model_path), task="detect")
    except Exception as exc:
        warnings.append(f"Plate detector model could not be loaded, falling back to OpenCV heuristic: {exc}")
        return "opencv_plate_heuristic", None


def detect_candidates_with_model(vehicle_crop: Any, model: Any) -> list[dict[str, Any]]:
    results = model.predict(source=vehicle_crop, verbose=False)
    candidates: list[dict[str, Any]] = []
    for result in results or []:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy = getattr(boxes, "xyxy", None)
        conf = getattr(boxes, "conf", None)
        if xyxy is None:
            continue
        for index in range(len(xyxy)):
            bbox = [int(round(float(value))) for value in xyxy[index].tolist()[:4]]
            score_value = round(float(conf[index].item()) if conf is not None else 0.5, 3)
            candidates.append(
                {
                    "plate_candidate_score": score_value,
                    "bbox_in_vehicle_crop": bbox,
                    "reason": "detected by dedicated plate detector",
                    "method": "yolo_plate_detector",
                }
            )
    candidates.sort(key=lambda item: float(item["plate_candidate_score"]), reverse=True)
    return candidates[:10]


def draw_debug_boxes(vehicle_crop: Any, candidates: list[dict[str, Any]]) -> Any:
    debug_image = vehicle_crop.copy()
    for candidate in candidates:
        x1, y1, x2, y2 = [int(value) for value in candidate["bbox_in_vehicle_crop"]]
        cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            debug_image,
            f"{float(candidate['plate_candidate_score']):.2f}",
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return debug_image


def build_candidate_payload(
    *,
    candidate_id: str,
    candidate_source: str,
    attribute_track_id: str | None,
    source_track_id: str | None,
    source_detection_id: str | None,
    class_name: str,
    vehicle_type: str,
    vehicle_color: str,
    frame_id: str,
    timestamp: float,
    vehicle_image_path: str | None,
    plate_crop_output_path: Path,
    debug_image_path: Path | None,
    plate_candidate_score: float,
    method: str,
    bbox_in_vehicle_crop: list[int],
    bbox_in_frame: list[int] | None,
    reason: str,
    source_detection_class_name: str | None = None,
    matched_source_track_id: str | None = None,
    matched_attribute_track_id: str | None = None,
    matched_track_class_name: str | None = None,
    matched_track_vehicle_type: str | None = None,
    matched_track_iou: float | None = None,
    matched_track_time_delta: float | None = None,
    class_source: str | None = None,
) -> dict[str, Any]:
    return {
        "plate_candidate_id": candidate_id,
        "candidate_source": candidate_source,
        "attribute_track_id": attribute_track_id,
        "source_track_id": source_track_id,
        "source_detection_id": source_detection_id,
        "source_detection_class_name": source_detection_class_name,
        "class_name": class_name,
        "vehicle_type": vehicle_type,
        "vehicle_color": vehicle_color,
        "matched_source_track_id": matched_source_track_id,
        "matched_attribute_track_id": matched_attribute_track_id,
        "matched_track_class_name": matched_track_class_name,
        "matched_track_vehicle_type": matched_track_vehicle_type,
        "matched_track_iou": round(float(matched_track_iou), 4) if matched_track_iou is not None else None,
        "matched_track_time_delta": round(float(matched_track_time_delta), 3) if matched_track_time_delta is not None else None,
        "class_source": class_source,
        "frame_id": frame_id,
        "timestamp": round(timestamp, 3),
        "vehicle_crop_path": vehicle_image_path,
        "plate_candidate_crop_path": to_repo_relative_path(plate_crop_output_path),
        "debug_image_path": to_repo_relative_path(debug_image_path) if debug_image_path else None,
        "plate_candidate_score": round(float(plate_candidate_score), 3),
        "candidate_status": "plate_like_region",
        "method": method,
        "bbox_in_vehicle_crop": [int(value) for value in bbox_in_vehicle_crop[:4]],
        "bbox_in_frame": [int(value) for value in bbox_in_frame[:4]] if bbox_in_frame else None,
        "reason": reason,
    }


def save_vehicle_debug_image(
    *,
    debug_root: Path,
    group_name: str,
    frame_id: str,
    vehicle_crop: Any,
    candidates: list[dict[str, Any]],
) -> Path:
    debug_output_dir = debug_root / group_name
    debug_output_dir.mkdir(parents=True, exist_ok=True)
    debug_image_path = debug_output_dir / f"{frame_id}.jpg"
    cv2.imwrite(str(debug_image_path), draw_debug_boxes(vehicle_crop, candidates[:5]))
    return debug_image_path


def detect_plate_candidates_from_vehicle_crop(
    *,
    vehicle_crop: Any,
    detector_mode: str,
    plate_detector: Any,
    settings: dict[str, Any],
    warnings: list[str],
) -> list[dict[str, Any]]:
    try:
        raw_candidates = (
            detect_candidates_with_model(vehicle_crop, plate_detector)
            if detector_mode == "yolo_plate_detector" and plate_detector is not None
            else detect_candidates_opencv(vehicle_crop)
        )
    except Exception as exc:
        warnings.append(f"Plate candidate detector failed on one vehicle crop: {exc}")
        return []
    return [
        candidate
        for candidate in raw_candidates
        if float(candidate.get("plate_candidate_score") or 0.0) >= float(settings["min_candidate_score"])
    ]


def select_frame_scan_frames(
    frame_lookup: dict[str, dict[str, Any]],
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    frames = sorted(
        frame_lookup.values(),
        key=lambda item: (
            float(item.get("global_timestamp_seconds") or 0.0),
            str(item.get("frame_id") or ""),
        ),
    )
    every_n = max(1, int(settings["frame_scan_every_n"]))
    limited = frames[: int(settings["frame_scan_max_frames"])]
    return [frame for index, frame in enumerate(limited) if index % every_n == 0]


def dedupe_plate_candidates(
    candidates: list[dict[str, Any]],
    *,
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    threshold = float(settings["frame_scan_dedup_iou"])
    for candidate in sorted(
        candidates,
        key=lambda item: (
            str(item.get("frame_id") or ""),
            -float(item.get("plate_candidate_score") or 0.0),
            0 if str(item.get("candidate_source") or "") == "track_based" else 1,
        ),
    ):
        keep = True
        for existing in deduped:
            if str(existing.get("frame_id") or "") != str(candidate.get("frame_id") or ""):
                continue
            if compute_iou(
                to_float_bbox(existing.get("bbox_in_frame")),
                to_float_bbox(candidate.get("bbox_in_frame")),
            ) < threshold:
                continue
            existing_score = float(existing.get("plate_candidate_score") or 0.0)
            candidate_score = float(candidate.get("plate_candidate_score") or 0.0)
            existing_track = str(existing.get("candidate_source") or "") == "track_based"
            candidate_track = str(candidate.get("candidate_source") or "") == "track_based"
            if candidate_track and (not existing_track or candidate_score + 0.03 >= existing_score):
                deduped.remove(existing)
                break
            if candidate_score > existing_score + 0.03:
                deduped.remove(existing)
                break
            keep = False
            break
        if keep:
            deduped.append(candidate)
    return sorted(
        deduped,
        key=lambda item: (
            float(item.get("timestamp") or 0.0),
            str(item.get("frame_id") or ""),
            -float(item.get("plate_candidate_score") or 0.0),
        ),
    )


def update_run_manifest_for_plate_candidates(run_manifest_path: Path) -> dict[str, Any]:
    run_manifest = read_json(run_manifest_path)
    completed_steps = list(run_manifest.get("completed_steps") or [])
    if "06A_plate_candidate_detection" not in completed_steps:
        completed_steps.append("06A_plate_candidate_detection")
    run_manifest["completed_steps"] = completed_steps
    run_manifest["next_step"] = "07A_plate_ocr"
    write_json(run_manifest_path, run_manifest)
    return run_manifest


def build_plate_candidate_outputs(run_dir: Path) -> dict[str, Any]:
    attributes_path = run_dir / "06_track_attributes.json"
    if not attributes_path.exists():
        raise FileNotFoundError(f"Missing required Step 6A input: {attributes_path}")

    warnings: list[str] = []
    settings = read_detector_settings()
    attributes_payload = read_json(attributes_path)
    tracks_payload = read_optional_json(run_dir / "05B_clean_tracks.json")
    clean_track_lookup = build_clean_track_lookup(tracks_payload)
    attribute_lookup = build_attribute_lookup(attributes_payload)
    frames_payload = read_optional_json(run_dir / "03_sampled_frames_index.json")
    detections_payload = read_optional_json(run_dir / "04_yolo_detections.json")
    video_info_payload = read_optional_json(run_dir / "01_video_info.json")
    track_lookup = build_track_lookup(tracks_payload)
    frame_lookup = build_frame_lookup(frames_payload)
    detection_lookup = build_detection_lookup(detections_payload)
    detector_mode, plate_detector = maybe_load_plate_detector(settings, warnings)

    attributes = [
        item
        for item in list(attributes_payload.get("attributes") or [])
        if isinstance(item, dict)
        and isinstance(item.get("vehicle_attributes"), dict)
        and str(item.get("class_name") or "") in VEHICLE_CLASSES
        and item.get("count_for_summary") in {True, "review"}
    ]
    total_vehicle_tracks = len(attributes)
    if settings["max_tracks"] is not None:
        attributes = attributes[: settings["max_tracks"]]
        warnings.append(
            f"Plate candidate detection limited to the first {settings['max_tracks']} vehicle tracks by {ENV_FINAL_DEMO_PLATE_MAX_TRACKS}."
        )

    crops_root = run_dir / "06A_plate_candidate_crops"
    debug_root = run_dir / "06A_plate_candidate_debug"
    crops_root.mkdir(parents=True, exist_ok=True)
    debug_root.mkdir(parents=True, exist_ok=True)

    frames_inspected = 0
    candidates_saved = 0
    tracks_processed = 0
    tracks_with_plate_candidates = 0
    frame_scan_frames_checked = 0
    frame_scan_vehicle_crops_checked = 0
    track_based_candidates_count = 0
    frame_scan_candidates_count = 0
    candidates_by_vehicle_type: dict[str, int] = defaultdict(int)
    combined_candidates: list[dict[str, Any]] = []

    run_track_scan = str(settings["plate_scan_mode"]) in {"track", "hybrid"}
    run_frame_scan = str(settings["plate_scan_mode"]) in {"frame", "hybrid"}

    if run_track_scan:
        for attribute in attributes:
            tracks_processed += 1
            attribute_track_id = str(attribute.get("attribute_track_id") or "")
            source_track_id = str(attribute.get("source_track_id") or "")
            vehicle_attributes = dict(attribute.get("vehicle_attributes") or {})
            vehicle_type = str(vehicle_attributes.get("vehicle_type") or attribute.get("class_name") or "unknown")
            vehicle_color = str(vehicle_attributes.get("vehicle_color") or "")
            selected_frames = select_track_frames(
                attribute,
                track_lookup,
                frame_lookup,
                int(settings["frames_per_track"]),
            )

            per_track_candidates: list[dict[str, Any]] = []
            for frame_info in selected_frames:
                bbox_in_frame = list(frame_info.get("bbox_xyxy") or [])
                frame_image = load_frame_image(
                    frame_info,
                    settings=settings,
                    video_info_payload=video_info_payload,
                    warnings=warnings,
                )
                vehicle_crop = crop_vehicle_from_frame(frame_image, bbox_in_frame)
                if vehicle_crop is None:
                    continue
                frames_inspected += 1
                raw_candidates = detect_plate_candidates_from_vehicle_crop(
                    vehicle_crop=vehicle_crop,
                    detector_mode=detector_mode,
                    plate_detector=plate_detector,
                    settings=settings,
                    warnings=warnings,
                )
                if not raw_candidates:
                    continue

                debug_image_path = None
                if settings["save_debug"]:
                    debug_image_path = save_vehicle_debug_image(
                        debug_root=debug_root,
                        group_name=attribute_track_id or "track_unknown",
                        frame_id=str(frame_info["frame_id"]),
                        vehicle_crop=vehicle_crop,
                        candidates=raw_candidates,
                    )

                for candidate in raw_candidates:
                    x1, y1, x2, y2 = [int(value) for value in candidate["bbox_in_vehicle_crop"]]
                    plate_crop = vehicle_crop[y1:y2, x1:x2]
                    if plate_crop is None or getattr(plate_crop, "size", 0) == 0:
                        continue
                    candidate_index = len(per_track_candidates) + 1
                    candidate_id = f"platecand_{attribute_track_id}_{candidate_index:03d}"
                    track_crop_dir = crops_root / (attribute_track_id or "track_unknown")
                    track_crop_dir.mkdir(parents=True, exist_ok=True)
                    crop_output_path = track_crop_dir / f"{candidate_id}.jpg"
                    cv2.imwrite(str(crop_output_path), plate_crop)
                    full_frame_bbox = None
                    if bbox_in_frame and len(bbox_in_frame) >= 4:
                        full_frame_bbox = offset_bbox(
                            [x1, y1, x2, y2],
                            float(bbox_in_frame[0]),
                            float(bbox_in_frame[1]),
                        )
                    per_track_candidates.append(
                        build_candidate_payload(
                            candidate_id=candidate_id,
                            candidate_source="track_based",
                            attribute_track_id=attribute_track_id or None,
                            source_track_id=source_track_id or None,
                            source_detection_id=None,
                            source_detection_class_name=None,
                            class_name=str(attribute.get("class_name") or ""),
                            vehicle_type=vehicle_type,
                            vehicle_color=vehicle_color,
                            matched_source_track_id=source_track_id or None,
                            matched_attribute_track_id=attribute_track_id or None,
                            matched_track_class_name=str(attribute.get("class_name") or "").lower() or None,
                            matched_track_vehicle_type=vehicle_type,
                            matched_track_iou=1.0 if full_frame_bbox else None,
                            matched_track_time_delta=0.0 if full_frame_bbox else None,
                            class_source="track_based_attribute",
                            frame_id=str(frame_info.get("frame_id") or ""),
                            timestamp=float(frame_info.get("timestamp") or 0.0),
                            vehicle_image_path=str(frame_info.get("image_path") or ""),
                            plate_crop_output_path=crop_output_path,
                            debug_image_path=debug_image_path,
                            plate_candidate_score=float(candidate["plate_candidate_score"]),
                            method=str(candidate["method"]),
                            bbox_in_vehicle_crop=[x1, y1, x2, y2],
                            bbox_in_frame=full_frame_bbox,
                            reason=str(candidate["reason"]),
                        )
                    )

            per_track_candidates.sort(key=lambda item: float(item["plate_candidate_score"]), reverse=True)
            kept_candidates = per_track_candidates[:3]
            if kept_candidates:
                tracks_with_plate_candidates += 1
                track_based_candidates_count += len(kept_candidates)
                combined_candidates.extend(kept_candidates)

    if run_frame_scan:
        if not frame_lookup:
            warnings.append("Frame-scan plate mode requested, but 03_sampled_frames_index.json is missing or empty.")
        if not detection_lookup:
            warnings.append("Frame-scan plate mode requested, but 04_yolo_detections.json is missing or empty.")
        selected_frame_infos = select_frame_scan_frames(frame_lookup, settings)
        allowed_classes = set(settings["frame_scan_classes"]) & VEHICLE_CLASSES
        if not allowed_classes:
            warnings.append("Frame-scan vehicle class list is empty after filtering allowed classes.")
        for frame_info in selected_frame_infos:
            frame_id = str(frame_info.get("frame_id") or "")
            detections = [
                detection for detection in list(detection_lookup.get(frame_id, []))
                if str(detection.get("class_name") or "").lower() in allowed_classes
                and float(detection.get("confidence") or 0.0) >= float(settings["frame_scan_min_vehicle_conf"])
            ]
            if not detections:
                continue
            frame_scan_frames_checked += 1
            frame_image = load_frame_image(
                frame_info,
                settings=settings,
                video_info_payload=video_info_payload,
                warnings=warnings,
            )
            if frame_image is None:
                continue
            for detection in detections[: int(settings["frame_scan_topk_per_frame"])]:
                timestamp = float(frame_info.get("global_timestamp_seconds") or frame_info.get("timestamp") or 0.0)
                matched_track_info = match_frame_scan_detection_to_track(
                    detection=detection,
                    timestamp=timestamp,
                    clean_track_lookup=clean_track_lookup,
                    attribute_lookup=attribute_lookup,
                    settings=settings,
                )
                vehicle_crop = crop_vehicle_from_frame(frame_image, list(detection.get("bbox_xyxy") or []))
                if vehicle_crop is None:
                    continue
                frame_scan_vehicle_crops_checked += 1
                raw_candidates = detect_plate_candidates_from_vehicle_crop(
                    vehicle_crop=vehicle_crop,
                    detector_mode=detector_mode,
                    plate_detector=plate_detector,
                    settings=settings,
                    warnings=warnings,
                )
                if not raw_candidates:
                    continue
                debug_image_path = None
                detection_id = str(detection.get("detection_id") or "")
                if settings["save_debug"]:
                    debug_image_path = save_vehicle_debug_image(
                        debug_root=debug_root,
                        group_name="frame_scan",
                        frame_id=f"{frame_id}_{detection_id or 'det'}",
                        vehicle_crop=vehicle_crop,
                        candidates=raw_candidates,
                    )
                frame_scan_dir = crops_root / "frame_scan" / frame_id
                frame_scan_dir.mkdir(parents=True, exist_ok=True)
                for index, candidate in enumerate(raw_candidates, start=1):
                    x1, y1, x2, y2 = [int(value) for value in candidate["bbox_in_vehicle_crop"]]
                    plate_crop = vehicle_crop[y1:y2, x1:x2]
                    if plate_crop is None or getattr(plate_crop, "size", 0) == 0:
                        continue
                    candidate_id = f"platecand_framescan_{frame_id}_{index:03d}_{detection_id or 'det'}"
                    crop_output_path = frame_scan_dir / f"{candidate_id}.jpg"
                    cv2.imwrite(str(crop_output_path), plate_crop)
                    source_detection_class_name = str(detection.get("class_name") or "vehicle").lower()
                    resolved_class_name = str(
                        matched_track_info.get("matched_track_class_name")
                        or source_detection_class_name
                    ).lower()
                    resolved_vehicle_type = str(
                        matched_track_info.get("matched_track_vehicle_type")
                        or resolved_class_name
                    ).lower()
                    combined_candidates.append(
                        build_candidate_payload(
                            candidate_id=candidate_id,
                            candidate_source="frame_scan",
                            attribute_track_id=matched_track_info.get("matched_attribute_track_id"),
                            source_track_id=matched_track_info.get("matched_source_track_id"),
                            source_detection_id=detection_id or None,
                            source_detection_class_name=source_detection_class_name,
                            class_name=resolved_class_name,
                            vehicle_type=resolved_vehicle_type,
                            vehicle_color="",
                            matched_source_track_id=matched_track_info.get("matched_source_track_id"),
                            matched_attribute_track_id=matched_track_info.get("matched_attribute_track_id"),
                            matched_track_class_name=matched_track_info.get("matched_track_class_name"),
                            matched_track_vehicle_type=matched_track_info.get("matched_track_vehicle_type"),
                            matched_track_iou=matched_track_info.get("matched_track_iou"),
                            matched_track_time_delta=matched_track_info.get("matched_track_time_delta"),
                            class_source=str(matched_track_info.get("class_source") or "source_yolo_detection"),
                            frame_id=frame_id,
                            timestamp=timestamp,
                            vehicle_image_path=str(frame_info.get("image_path") or ""),
                            plate_crop_output_path=crop_output_path,
                            debug_image_path=debug_image_path,
                            plate_candidate_score=float(candidate["plate_candidate_score"]),
                            method=str(candidate["method"]),
                            bbox_in_vehicle_crop=[x1, y1, x2, y2],
                            bbox_in_frame=offset_bbox(
                                [x1, y1, x2, y2],
                                float(list(detection.get("bbox_xyxy") or [0, 0, 0, 0])[0]),
                                float(list(detection.get("bbox_xyxy") or [0, 0, 0, 0])[1]),
                            ),
                            reason=str(candidate["reason"]),
                        )
                    )
                    frame_scan_candidates_count += 1

    total_candidates_before_dedup = len(combined_candidates)
    all_candidates = dedupe_plate_candidates(combined_candidates, settings=settings)
    for candidate in all_candidates:
        candidates_saved += 1
        candidates_by_vehicle_type[str(candidate.get("vehicle_type") or "unknown")] += 1

    overall_status = "completed" if all_candidates else "completed_no_plate_candidates"
    recommendations: list[str] = []
    if not all_candidates:
        recommendations.append("No reliable plate-like regions found. Use a dedicated licence plate detector model for ANPR.")

    candidates_payload = {
        "created_at": current_timestamp(),
        "selected_track_source": "06_track_attributes",
        "plate_scan_mode": settings["plate_scan_mode"],
        "plate_detector_mode": detector_mode,
        "total_vehicle_tracks": total_vehicle_tracks,
        "tracks_processed": tracks_processed,
        "tracks_with_plate_candidates": tracks_with_plate_candidates,
        "total_plate_candidates": len(all_candidates),
        "candidates": all_candidates,
        "warnings": list(dict.fromkeys(warnings)),
    }
    report_payload = {
        "created_at": current_timestamp(),
        "overall_status": overall_status,
        "plate_scan_mode": settings["plate_scan_mode"],
        "plate_detector_mode": detector_mode,
        "total_vehicle_tracks": total_vehicle_tracks,
        "tracks_processed": tracks_processed,
        "tracks_with_plate_candidates": tracks_with_plate_candidates,
        "total_plate_candidates": len(all_candidates),
        "frames_inspected": frames_inspected,
        "track_based_candidates": track_based_candidates_count,
        "frame_scan_candidates": frame_scan_candidates_count,
        "total_frame_scan_frames_checked": frame_scan_frames_checked,
        "total_frame_scan_vehicle_crops_checked": frame_scan_vehicle_crops_checked,
        "total_plate_candidates_before_dedup": total_candidates_before_dedup,
        "total_plate_candidates_after_dedup": len(all_candidates),
        "candidates_by_source": {
            "track_based": len([item for item in all_candidates if str(item.get("candidate_source")) == "track_based"]),
            "frame_scan": len([item for item in all_candidates if str(item.get("candidate_source")) == "frame_scan"]),
        },
        "candidate_crops_saved": candidates_saved,
        "candidates_by_vehicle_class": dict(sorted(candidates_by_vehicle_type.items())),
        "candidates_by_vehicle_type": dict(sorted(candidates_by_vehicle_type.items())),
        "candidate_score_threshold": float(settings["min_candidate_score"]),
        "warnings": list(dict.fromkeys(warnings)),
        "recommendations": recommendations,
    }
    return {
        "candidates_payload": candidates_payload,
        "report_payload": report_payload,
        "crops_root": crops_root,
        "debug_root": debug_root,
    }
