from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2

from tests.final_demo.services.chunk_planner import read_json
from tests.final_demo.services.video_io import current_timestamp, write_json


ENV_FINAL_DEMO_YOLO_MODEL = "FINAL_DEMO_YOLO_MODEL"
ENV_FINAL_DEMO_YOLO_CONF = "FINAL_DEMO_YOLO_CONF"
ENV_FINAL_DEMO_YOLO_IOU = "FINAL_DEMO_YOLO_IOU"
ENV_FINAL_DEMO_YOLO_DEVICE = "FINAL_DEMO_YOLO_DEVICE"
ENV_FINAL_DEMO_YOLO_MAX_FRAMES = "FINAL_DEMO_YOLO_MAX_FRAMES"
ENV_FINAL_DEMO_SAVE_ANNOTATED_FRAMES = "FINAL_DEMO_SAVE_ANNOTATED_FRAMES"

DEFAULT_YOLO_MODEL = "yolo11n.pt"
DEFAULT_YOLO_CONF = 0.25
DEFAULT_YOLO_IOU = 0.45
DEFAULT_YOLO_IMGSZ = 640
DEFAULT_SAVE_ANNOTATED_FRAMES = True

TENDER_RELEVANT_CLASS_NAMES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "backpack",
    "handbag",
    "suitcase",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
}


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def to_absolute_repo_path(relative_repo_path: str) -> Path:
    return get_repo_root() / Path(relative_repo_path)


def to_repo_relative_path(path: Path) -> str:
    return path.resolve().relative_to(get_repo_root()).as_posix()


def read_float_env(env_name: str, default_value: float) -> float:
    raw_value = os.environ.get(env_name, str(default_value))
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {env_name} must be a valid number. Received: {raw_value!r}"
        ) from exc
    return value


def read_positive_float_env(env_name: str, default_value: float) -> float:
    value = read_float_env(env_name, default_value)
    if value <= 0:
        raise ValueError(
            f"Environment variable {env_name} must be greater than 0. Received: {value}"
        )
    return value


def read_non_negative_float_env(env_name: str, default_value: float) -> float:
    value = read_float_env(env_name, default_value)
    if value < 0:
        raise ValueError(
            f"Environment variable {env_name} must be greater than or equal to 0. Received: {value}"
        )
    return value


def read_optional_positive_int_env(env_name: str) -> int | None:
    raw_value = os.environ.get(env_name)
    if raw_value is None or raw_value == "":
        return None

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {env_name} must be a valid integer. Received: {raw_value!r}"
        ) from exc

    if value <= 0:
        raise ValueError(
            f"Environment variable {env_name} must be greater than 0. Received: {value}"
        )
    return value


def read_bool_env(env_name: str, default_value: bool) -> bool:
    raw_value = os.environ.get(env_name)
    if raw_value is None or raw_value == "":
        return default_value

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False

    raise ValueError(
        f"Environment variable {env_name} must be a boolean-like value. Received: {raw_value!r}"
    )


def read_yolo_settings() -> dict[str, Any]:
    return {
        "model_name": os.environ.get(ENV_FINAL_DEMO_YOLO_MODEL, DEFAULT_YOLO_MODEL),
        "confidence_threshold": round(
            read_positive_float_env(ENV_FINAL_DEMO_YOLO_CONF, DEFAULT_YOLO_CONF), 3
        ),
        "iou_threshold": round(
            read_non_negative_float_env(ENV_FINAL_DEMO_YOLO_IOU, DEFAULT_YOLO_IOU), 3
        ),
        "device": os.environ.get(ENV_FINAL_DEMO_YOLO_DEVICE, "").strip(),
        "max_frames": read_optional_positive_int_env(ENV_FINAL_DEMO_YOLO_MAX_FRAMES),
        "save_annotated_frames": read_bool_env(
            ENV_FINAL_DEMO_SAVE_ANNOTATED_FRAMES,
            DEFAULT_SAVE_ANNOTATED_FRAMES,
        ),
    }


def normalize_model_names(model_names: Any) -> dict[int, str]:
    if isinstance(model_names, dict):
        return {int(key): str(value) for key, value in model_names.items()}
    if isinstance(model_names, list):
        return {index: str(value) for index, value in enumerate(model_names)}
    return {}


def determine_allowed_class_ids(model_names: dict[int, str]) -> tuple[set[int] | None, list[str]]:
    if not model_names:
        return None, ["Model class names were not available; keeping all detections."]

    normalized_name_map = {
        class_id: class_name.strip().lower()
        for class_id, class_name in model_names.items()
    }
    matched_class_ids = {
        class_id
        for class_id, class_name in normalized_name_map.items()
        if class_name in TENDER_RELEVANT_CLASS_NAMES
    }

    if matched_class_ids:
        return matched_class_ids, []

    return None, [
        "Tender-relevant class names were not found in the model label map; keeping all detections."
    ]


def annotate_frame(
    image: Any,
    *,
    class_name: str,
    confidence: float,
    bbox_xyxy: list[float],
) -> Any:
    annotated = image.copy()
    x1, y1, x2, y2 = [int(round(value)) for value in bbox_xyxy]
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{class_name} {confidence:.2f}"
    label_origin = (x1, max(20, y1 - 10))
    cv2.putText(
        annotated,
        label,
        label_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return annotated


def update_chunk_manifest_for_yolo(
    chunk_manifest: dict[str, Any],
    processed_frames: list[dict[str, Any]],
    detections: list[dict[str, Any]],
    annotated_root: Path | None,
) -> dict[str, Any]:
    frames_by_chunk: dict[str, int] = defaultdict(int)
    detections_by_chunk: dict[str, int] = defaultdict(int)

    for frame_item in processed_frames:
        frames_by_chunk[str(frame_item["chunk_id"])] += 1

    for detection in detections:
        detections_by_chunk[str(detection["chunk_id"])] += 1

    for chunk in list(chunk_manifest.get("chunks", [])):
        chunk_id = str(chunk.get("chunk_id", ""))
        if chunk_id not in frames_by_chunk:
            continue

        steps_completed = list(chunk.get("steps_completed", []))
        if "04_yolo_detection" not in steps_completed:
            steps_completed.append("04_yolo_detection")

        chunk["steps_completed"] = steps_completed
        chunk["yolo_frame_count"] = frames_by_chunk[chunk_id]
        chunk["yolo_detection_count"] = detections_by_chunk.get(chunk_id, 0)
        if str(chunk.get("status", "")) == "sampled":
            chunk["status"] = "detected"
        if annotated_root is not None:
            chunk["yolo_annotated_frames_dir"] = to_repo_relative_path(annotated_root / chunk_id)

    return chunk_manifest


def update_run_manifest_for_yolo(run_manifest_path: Path) -> dict[str, Any]:
    run_manifest = read_json(run_manifest_path)
    completed_steps = list(run_manifest.get("completed_steps", []))
    if "04_yolo_detection" not in completed_steps:
        completed_steps.append("04_yolo_detection")

    run_manifest["completed_steps"] = completed_steps
    run_manifest["next_step"] = "05_object_tracking"
    write_json(run_manifest_path, run_manifest)
    return run_manifest


def run_yolo_detection_on_sampled_frames(
    *,
    run_dir: Path,
    frames_index_payload: dict[str, Any],
    chunk_manifest: dict[str, Any],
) -> dict[str, Any]:
    from ultralytics import YOLO

    settings = read_yolo_settings()
    frames = list(frames_index_payload.get("frames", []))
    if not frames:
        raise ValueError("Sampled frames index does not contain any frames.")

    total_frames_input = len(frames)
    selected_frames = frames[: settings["max_frames"]] if settings["max_frames"] else frames
    warnings: list[str] = []

    if settings["max_frames"] is not None and settings["max_frames"] < total_frames_input:
        warnings.append(
            f"YOLO processing limited to first {settings['max_frames']} frames by {ENV_FINAL_DEMO_YOLO_MAX_FRAMES}."
        )

    model = YOLO(settings["model_name"])
    model_names = normalize_model_names(getattr(model, "names", {}))
    allowed_class_ids, class_warnings = determine_allowed_class_ids(model_names)
    warnings.extend(class_warnings)

    annotated_root: Path | None = None
    if settings["save_annotated_frames"]:
        annotated_root = run_dir / "04_yolo_annotated_frames"
        annotated_root.mkdir(parents=True, exist_ok=True)

    detections: list[dict[str, Any]] = []
    detections_by_class: dict[str, int] = defaultdict(int)
    detections_by_chunk: dict[str, int] = defaultdict(int)
    frames_with_detections = 0
    frames_without_detections = 0
    detection_counter = 1

    for frame_item in selected_frames:
        image_path = to_absolute_repo_path(str(frame_item.get("image_path", "")))
        if not image_path.exists():
            warnings.append(f"Sampled frame image is missing: {image_path}")
            frames_without_detections += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            warnings.append(f"Failed to read sampled frame image: {image_path}")
            frames_without_detections += 1
            continue

        predict_kwargs: dict[str, Any] = {
            "source": str(image_path),
            "conf": settings["confidence_threshold"],
            "iou": settings["iou_threshold"],
            "imgsz": DEFAULT_YOLO_IMGSZ,
            "verbose": False,
        }
        if settings["device"]:
            predict_kwargs["device"] = settings["device"]

        results = model.predict(**predict_kwargs)
        result = results[0]
        boxes = result.boxes
        frame_detection_count = 0
        annotated_image = image.copy()

        if boxes is not None:
            class_ids = boxes.cls.tolist() if boxes.cls is not None else []
            confidences = boxes.conf.tolist() if boxes.conf is not None else []
            bbox_xyxy_values = boxes.xyxy.tolist() if boxes.xyxy is not None else []
            bbox_xywh_values = boxes.xywh.tolist() if boxes.xywh is not None else []

            for class_id_raw, confidence_raw, bbox_xyxy_raw, bbox_xywh_raw in zip(
                class_ids,
                confidences,
                bbox_xyxy_values,
                bbox_xywh_values,
            ):
                class_id = int(class_id_raw)
                if allowed_class_ids is not None and class_id not in allowed_class_ids:
                    continue

                class_name = model_names.get(class_id, str(class_id))
                confidence = round(float(confidence_raw), 4)
                bbox_xyxy = [round(float(value), 3) for value in bbox_xyxy_raw]
                bbox_xywh = [round(float(value), 3) for value in bbox_xywh_raw]
                bbox_center = [
                    round((bbox_xyxy[0] + bbox_xyxy[2]) / 2.0, 3),
                    round((bbox_xyxy[1] + bbox_xyxy[3]) / 2.0, 3),
                ]
                bbox_area_pixels = round(
                    max(0.0, bbox_xyxy[2] - bbox_xyxy[0]) * max(0.0, bbox_xyxy[3] - bbox_xyxy[1]),
                    3,
                )

                detections.append(
                    {
                        "detection_id": f"det_{detection_counter:08d}",
                        "frame_id": str(frame_item["frame_id"]),
                        "chunk_id": str(frame_item["chunk_id"]),
                        "chunk_index": int(frame_item["chunk_index"]),
                        "global_timestamp_seconds": round(
                            float(frame_item["global_timestamp_seconds"]),
                            3,
                        ),
                        "image_path": str(frame_item["image_path"]),
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox_xyxy": bbox_xyxy,
                        "bbox_xywh": bbox_xywh,
                        "bbox_center": bbox_center,
                        "bbox_area_pixels": bbox_area_pixels,
                        "frame_width": int(frame_item["width"]),
                        "frame_height": int(frame_item["height"]),
                    }
                )
                detection_counter += 1
                frame_detection_count += 1
                detections_by_class[class_name] += 1
                detections_by_chunk[str(frame_item["chunk_id"])] += 1

                if annotated_root is not None:
                    annotated_image = annotate_frame(
                        annotated_image,
                        class_name=class_name,
                        confidence=confidence,
                        bbox_xyxy=bbox_xyxy,
                    )

        if frame_detection_count > 0:
            frames_with_detections += 1
        else:
            frames_without_detections += 1

        if annotated_root is not None:
            chunk_output_dir = annotated_root / str(frame_item["chunk_id"])
            chunk_output_dir.mkdir(parents=True, exist_ok=True)
            annotated_path = chunk_output_dir / f"annotated_{Path(str(frame_item['image_path'])).name}"
            write_success = cv2.imwrite(str(annotated_path), annotated_image)
            if not write_success:
                warnings.append(f"Failed to write annotated frame: {annotated_path}")

    updated_chunk_manifest = update_chunk_manifest_for_yolo(
        chunk_manifest=chunk_manifest,
        processed_frames=selected_frames,
        detections=detections,
        annotated_root=annotated_root,
    )

    detections_payload = {
        "model_name": settings["model_name"],
        "confidence_threshold": settings["confidence_threshold"],
        "iou_threshold": settings["iou_threshold"],
        "device": settings["device"] or "",
        "total_frames_input": total_frames_input,
        "total_frames_processed": len(selected_frames),
        "total_detections": len(detections),
        "created_at": current_timestamp(),
        "detections": detections,
    }

    report_payload = {
        "model_name": settings["model_name"],
        "total_frames_input": total_frames_input,
        "total_frames_processed": len(selected_frames),
        "total_detections": len(detections),
        "detections_by_class": dict(sorted(detections_by_class.items())),
        "detections_by_chunk": dict(sorted(detections_by_chunk.items())),
        "frames_with_detections": frames_with_detections,
        "frames_without_detections": frames_without_detections,
        "warnings": warnings,
        "created_at": current_timestamp(),
    }

    return {
        "settings": settings,
        "detections_payload": detections_payload,
        "report_payload": report_payload,
        "updated_chunk_manifest": updated_chunk_manifest,
    }
