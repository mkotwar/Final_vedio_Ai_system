from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_RUN_YOLO = "true"
DEFAULT_YOLO_MODEL = "yolov8n.pt"
DEFAULT_YOLO_CONF = 0.25
DEFAULT_YOLO_IMGSZ = 640
DEFAULT_YOLO_INPUT_SCOPE = "motion_candidates"

INPUT_SCOPE_TO_FILENAME = {
    "motion_candidates": "04_motion_candidates.json",
    "sampled_frames": "02_sampled_frames.json",
}

VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "bicycle"}
IMPORTANT_OBJECT_CLASSES = {
    "backpack",
    "handbag",
    "suitcase",
    "cell phone",
    "knife",
    "sports ball",
    "bottle",
    "box",
}


def _read_env_bool_str(name: str, default: str) -> bool:
    raw_value = os.environ.get(name, default).strip().lower()
    return raw_value == "true"


def _read_env_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name, str(default)).strip()
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a valid float. Received: {raw_value!r}") from exc


def _read_env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name, str(default)).strip()
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a valid integer. Received: {raw_value!r}") from exc


def _read_input_scope() -> str:
    scope = os.environ.get("TENDER_DEMO_YOLO_INPUT_SCOPE", DEFAULT_YOLO_INPUT_SCOPE).strip().lower()
    if scope not in INPUT_SCOPE_TO_FILENAME:
        allowed_values = ", ".join(sorted(INPUT_SCOPE_TO_FILENAME))
        raise ValueError(
            "Environment variable TENDER_DEMO_YOLO_INPUT_SCOPE must be one of "
            f"{allowed_values}. Received: {scope!r}"
        )
    return scope


def _load_input_items(run_dir: Path, input_scope: str) -> list[dict[str, Any]]:
    input_path = run_dir / INPUT_SCOPE_TO_FILENAME[input_scope]
    if not input_path.exists():
        raise FileNotFoundError(f"Missing YOLO input file for scope '{input_scope}': {input_path}")

    items = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError(f"Expected a list in YOLO input file: {input_path}")
    return items


def _to_abs_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / path


def _safe_round(value: float) -> float:
    return round(float(value), 6)


def _build_detection_entry(box: Any, class_names: dict[int, str]) -> dict[str, Any]:
    class_id = int(box.cls[0].item())
    class_name = class_names.get(class_id, str(class_id))
    confidence = float(box.conf[0].item())
    xyxy = [float(coord) for coord in box.xyxy[0].tolist()]
    x1, y1, x2, y2 = xyxy
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    center_x = x1 + (width / 2.0)
    center_y = y1 + (height / 2.0)
    area = width * height

    return {
        "class_id": class_id,
        "class_name": class_name,
        "confidence": _safe_round(confidence),
        "bbox_xyxy": [_safe_round(coord) for coord in xyxy],
        "bbox_center": [_safe_round(center_x), _safe_round(center_y)],
        "bbox_width": _safe_round(width),
        "bbox_height": _safe_round(height),
        "bbox_area": _safe_round(area),
    }


def _count_top_classes(frame_results: list[dict[str, Any]]) -> list[str]:
    class_counts: dict[str, int] = {}
    for item in frame_results:
        for class_name, count in item.get("object_counts", {}).items():
            class_counts[class_name] = class_counts.get(class_name, 0) + int(count)
    return [
        class_name
        for class_name, _ in sorted(class_counts.items(), key=lambda pair: (-pair[1], pair[0]))[:10]
    ]


def run_yolo_detection_on_selected_frames(run_dir: Path) -> list[dict[str, Any]]:
    if not _read_env_bool_str("TENDER_DEMO_RUN_YOLO", DEFAULT_RUN_YOLO):
        print("[tender-demo] Step 10 skipped: TENDER_DEMO_RUN_YOLO is not 'true'")
        return []

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is required for Step 10 YOLO detection. Install it with: pip install ultralytics"
        ) from exc

    input_scope = _read_input_scope()
    yolo_model_name = os.environ.get("TENDER_DEMO_YOLO_MODEL", DEFAULT_YOLO_MODEL).strip() or DEFAULT_YOLO_MODEL
    yolo_conf = _read_env_float("TENDER_DEMO_YOLO_CONF", DEFAULT_YOLO_CONF)
    yolo_imgsz = _read_env_int("TENDER_DEMO_YOLO_IMGSZ", DEFAULT_YOLO_IMGSZ)
    input_items = _load_input_items(run_dir=run_dir, input_scope=input_scope)

    print("[tender-demo] Starting Step 10: YOLO object detection on selected frames")
    try:
        model = YOLO(yolo_model_name)
    except Exception as exc:
        raise RuntimeError(f"Failed to load YOLO model '{yolo_model_name}': {exc}") from exc

    frame_results: list[dict[str, Any]] = []
    processed_frames = 0
    skipped_frames = 0
    total_detections = 0
    frames_with_person = 0
    frames_with_vehicle = 0
    frames_with_important_object = 0

    for index, item in enumerate(input_items, start=1):
        frame_path_value = item.get("frame_path")
        if not frame_path_value:
            skipped_frames += 1
            print(f"[tender-demo] Warning: missing frame_path for input item {index}; skipping")
            continue

        frame_path = _to_abs_path(str(frame_path_value))
        if not frame_path.exists():
            skipped_frames += 1
            print(f"[tender-demo] Warning: frame path does not exist; skipping {frame_path}")
            continue

        try:
            predictions = model.predict(
                source=str(frame_path),
                conf=yolo_conf,
                imgsz=yolo_imgsz,
                verbose=False,
            )
        except Exception as exc:
            skipped_frames += 1
            print(f"[tender-demo] Warning: YOLO failed on {frame_path}: {exc}")
            continue

        detections: list[dict[str, Any]] = []
        object_counts: dict[str, int] = {}
        class_names = predictions[0].names if predictions else {}

        if predictions:
            for box in predictions[0].boxes:
                detection = _build_detection_entry(box=box, class_names=class_names)
                detections.append(detection)
                class_name = detection["class_name"]
                object_counts[class_name] = object_counts.get(class_name, 0) + 1

        processed_frames += 1
        detection_count = len(detections)
        total_detections += detection_count
        object_classes_present = sorted(object_counts.keys())
        person_count = object_counts.get("person", 0)
        vehicle_count = sum(object_counts.get(name, 0) for name in VEHICLE_CLASSES)
        important_object_count = sum(object_counts.get(name, 0) for name in IMPORTANT_OBJECT_CLASSES)
        has_person = person_count > 0
        has_vehicle = vehicle_count > 0
        has_important_object = important_object_count > 0

        if has_person:
            frames_with_person += 1
        if has_vehicle:
            frames_with_vehicle += 1
        if has_important_object:
            frames_with_important_object += 1

        frame_result = {
            **item,
            "frame_id": item.get("frame_id", item.get("sample_id", item.get("candidate_id", f"frame_{index:06d}"))),
            "sample_id": item.get("sample_id"),
            "candidate_id": item.get("candidate_id"),
            "frame_idx": item.get("frame_idx"),
            "timestamp_seconds": item.get("timestamp_seconds"),
            "frame_path": item.get("frame_path"),
            "motion_score_norm": item.get("motion_score_norm"),
            "motion_level": item.get("motion_level"),
            "detections": detections,
            "detection_count": detection_count,
            "person_count": person_count,
            "vehicle_count": vehicle_count,
            "important_object_count": important_object_count,
            "object_classes_present": object_classes_present,
            "object_counts": object_counts,
            "has_person": has_person,
            "has_vehicle": has_vehicle,
            "has_important_object": has_important_object,
        }
        frame_results.append(frame_result)

    output_path = run_dir / "10_yolo_detections.json"
    output_path.write_text(json.dumps(frame_results, indent=2), encoding="utf-8")

    print(f"[tender-demo] YOLO model: {yolo_model_name}")
    print(f"[tender-demo] Confidence threshold: {yolo_conf}")
    print(f"[tender-demo] Image size: {yolo_imgsz}")
    print(f"[tender-demo] Input scope: {input_scope}")
    print(f"[tender-demo] Total input frames: {len(input_items)}")
    print(f"[tender-demo] Total frames processed by YOLO: {processed_frames}")
    print(f"[tender-demo] Total skipped frames: {skipped_frames}")
    print(f"[tender-demo] Total detections: {total_detections}")
    print(f"[tender-demo] Frames with person: {frames_with_person}")
    print(f"[tender-demo] Frames with vehicle: {frames_with_vehicle}")
    print(f"[tender-demo] Frames with important objects: {frames_with_important_object}")
    print(f"[tender-demo] Top detected classes: {_count_top_classes(frame_results)}")
    print(f"[tender-demo] YOLO detections output path: {output_path}")
    return frame_results
