from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from device_manager import cuda_memory_allocated_mb, cuda_memory_reserved_mb
from stage_checks import read_json, write_json


def _load_yolo_class() -> Any:
    """Import ultralytics YOLO lazily so failures can be recorded cleanly."""

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(f"Failed to import ultralytics.YOLO: {exc}") from exc
    return YOLO


def _resolve_frame_path(run_dir: Path, item: dict[str, Any]) -> Path:
    """Resolve adaptive frame image paths safely relative to the run directory."""

    image_path = Path(str(item.get("image_path", "")))
    if image_path.is_absolute():
        return image_path
    return (run_dir / image_path).resolve()


def _safe_stats(values: list[float]) -> dict[str, float]:
    """Return min/max/avg stats with zero defaults."""

    if not values:
        return {"min": 0.0, "max": 0.0, "avg": 0.0}
    return {
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "avg": round(sum(values) / len(values), 6),
    }


def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU for duplicate-check metadata."""

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def _safe_model_device(model: Any) -> str | None:
    try:
        inner_model = getattr(model, "model", None)
        parameters = getattr(inner_model, "parameters", None)
        if parameters is None:
            return None
        first_parameter = next(parameters())
        return str(first_parameter.device)
    except Exception:
        return None


def _safe_result_device(result: Any) -> str | None:
    try:
        boxes = getattr(result, "boxes", None)
        data = getattr(boxes, "data", None)
        device = getattr(data, "device", None)
        return str(device) if device is not None else None
    except Exception:
        return None


def run_yolo_detection(
    *,
    run_dir: Path,
    audit_payload: dict[str, Any],
    model_specs: list[dict[str, Any]],
    conf_threshold: float,
    iou_threshold: float,
    device: str,
    save_annotated: bool,
    save_crops: bool,
    max_frames: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run YOLO detection on all adaptive frames using the successfully audited models."""

    adaptive_manifest = read_json(run_dir / "02A_adaptive_frames.json")
    adaptive_frames = list(adaptive_manifest.get("selected_frames", []))
    if max_frames is not None:
        adaptive_frames = adaptive_frames[:max_frames]

    audit_models = list(audit_payload.get("models", []))
    ready_roles = {
        str(item.get("model_role")): item
        for item in audit_models
        if str(item.get("load_status")) == "success"
    }
    active_specs = [item for item in model_specs if str(item["model_role"]) in ready_roles]
    if not active_specs:
        raise RuntimeError("No YOLO models were successfully audited, so Step 03B cannot continue.")

    YOLO = _load_yolo_class()
    loaded_models: dict[str, Any] = {}
    class_name_lookup: dict[str, dict[str, str]] = {}
    for model_spec in active_specs:
        model_role = str(model_spec["model_role"])
        model = YOLO(str(model_spec["model_path"]))
        loaded_models[model_role] = model
        raw_names = getattr(model, "names", {})
        if isinstance(raw_names, dict):
            class_name_lookup[model_role] = {str(key): str(value) for key, value in raw_names.items()}
        elif isinstance(raw_names, (list, tuple)):
            class_name_lookup[model_role] = {str(index): str(value) for index, value in enumerate(raw_names)}
        else:
            class_name_lookup[model_role] = {}

    model_device_map = {model_role: _safe_model_device(model) for model_role, model in loaded_models.items()}
    inference_device_map: dict[str, str | None] = {model_role: None for model_role in loaded_models}

    annotated_dir = run_dir / "03_yolo_annotated_frames"
    crops_dir = run_dir / "03_yolo_object_crops"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    frame_results: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    model_role_counts: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    confidence_values: list[float] = []
    bbox_area_ratio_values: list[float] = []
    top_detection_frames: list[dict[str, Any]] = []

    total_detections = 0
    frames_with_detections = 0

    for frame_number, frame_item in enumerate(adaptive_frames, start=1):
        frame_path = _resolve_frame_path(run_dir, frame_item)
        frame_payload = {
            "frame_id": str(frame_item.get("frame_id", "")),
            "frame_idx": int(frame_item.get("frame_idx", 0) or 0),
            "timestamp_seconds": float(frame_item.get("timestamp_seconds", 0.0) or 0.0),
            "timestamp_text": str(frame_item.get("timestamp_text", "")),
            "image_path": str(frame_item.get("image_path", "")),
            "adaptive_keep_reason": list(frame_item.get("keep_reason", [])),
            "seconds_since_previous_selected": frame_item.get("seconds_since_previous_selected"),
            "motion_score": frame_item.get("motion_score"),
            "motion_pixels_ratio": frame_item.get("motion_pixels_ratio"),
            "histogram_change_score": frame_item.get("histogram_change_score"),
            "detections": [],
        }

        try:
            original_image = cv2.imread(str(frame_path))
            if original_image is None:
                raise RuntimeError(f"Failed to read adaptive frame image: {frame_path}")
            annotated_image = original_image.copy()
            frame_height, frame_width = original_image.shape[:2]

            for model_spec in active_specs:
                model_role = str(model_spec["model_role"])
                model_path = Path(str(model_spec["model_path"]))
                model = loaded_models[model_role]
                results = model.predict(
                    source=str(frame_path),
                    conf=conf_threshold,
                    iou=iou_threshold,
                    device=device,
                    verbose=False,
                )
                if not results:
                    continue

                result = results[0]
                inference_device_map[model_role] = _safe_result_device(result)
                plotted_image = result.plot()
                annotated_image = plotted_image
                boxes = getattr(result, "boxes", None)
                if boxes is None or getattr(boxes, "xyxy", None) is None:
                    continue

                xyxy_list = boxes.xyxy.tolist()
                xywh_list = boxes.xywh.tolist() if getattr(boxes, "xywh", None) is not None else []
                cls_list = boxes.cls.tolist() if getattr(boxes, "cls", None) is not None else []
                conf_list = boxes.conf.tolist() if getattr(boxes, "conf", None) is not None else []
                role_class_names = class_name_lookup.get(model_role, {})

                for index, bbox_xyxy in enumerate(xyxy_list):
                    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
                    bbox_width = max(0.0, x2 - x1)
                    bbox_height = max(0.0, y2 - y1)
                    bbox_area = bbox_width * bbox_height
                    bbox_area_ratio = bbox_area / float(frame_width * frame_height) if frame_width > 0 and frame_height > 0 else 0.0
                    class_id = int(cls_list[index]) if index < len(cls_list) else -1
                    class_name = role_class_names.get(str(class_id), str(class_id))
                    confidence = float(conf_list[index]) if index < len(conf_list) else 0.0
                    bbox_xywh = [float(value) for value in xywh_list[index]] if index < len(xywh_list) else [0.0, 0.0, 0.0, 0.0]
                    bbox_center_xy = [round(float(bbox_xywh[0]), 3), round(float(bbox_xywh[1]), 3)]

                    crop_path = ""
                    crop_bbox_xyxy = [x1, y1, x2, y2]
                    crop_padding_ratio = 0.0
                    if save_crops:
                        crop_padding_ratio = 0.05 if class_name.lower() in {
                            "car", "truck", "bus", "motorcycle", "bicycle", "auto", "van", "vehicle"
                        } else 0.0
                        pad_x = bbox_width * crop_padding_ratio
                        pad_y = bbox_height * crop_padding_ratio
                        ix1 = max(0, int(round(x1 - pad_x)))
                        iy1 = max(0, int(round(y1 - pad_y)))
                        ix2 = min(frame_width, int(round(x2 + pad_x)))
                        iy2 = min(frame_height, int(round(y2 + pad_y)))
                        crop_bbox_xyxy = [float(ix1), float(iy1), float(ix2), float(iy2)]
                        if ix2 > ix1 and iy2 > iy1:
                            crop = original_image[iy1:iy2, ix1:ix2]
                            crop_name = f"{frame_payload['frame_id']}_{model_role}_{index + 1:03d}_{class_name}.jpg"
                            crop_output_path = crops_dir / crop_name
                            if cv2.imwrite(str(crop_output_path), crop):
                                crop_path = str(Path("03_yolo_object_crops") / crop_name).replace("\\", "/")

                    detection_payload = {
                        "detection_id": f"{frame_payload['frame_id']}_{model_role}_{index + 1:03d}",
                        "model_role": model_role,
                        "model_path": str(model_path),
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": round(confidence, 6),
                        "bbox_xyxy": [round(value, 3) for value in [x1, y1, x2, y2]],
                        "bbox_xywh": [round(float(value), 3) for value in bbox_xywh],
                        "bbox_center_xy": bbox_center_xy,
                        "bbox_area": round(bbox_area, 3),
                        "bbox_area_ratio": round(bbox_area_ratio, 6),
                        "crop_path": crop_path,
                        "crop_bbox_xyxy": [round(value, 3) for value in crop_bbox_xyxy],
                        "crop_padding_ratio": crop_padding_ratio,
                        "annotated_frame_path": "",
                        "duplicate_check_enabled": True,
                        "possible_duplicate_group_id": None,
                    }
                    frame_payload["detections"].append(detection_payload)
                    class_counts[class_name] += 1
                    model_role_counts[model_role] += 1
                    confidence_values.append(confidence)
                    bbox_area_ratio_values.append(bbox_area_ratio)
                    total_detections += 1

            if frame_payload["detections"]:
                frames_with_detections += 1

            if save_annotated and frame_payload["detections"]:
                annotated_name = f"{frame_payload['frame_id']}.jpg"
                annotated_output_path = annotated_dir / annotated_name
                if not cv2.imwrite(str(annotated_output_path), annotated_image):
                    raise RuntimeError(f"Failed to write annotated detection frame: {annotated_output_path}")
                relative_annotated_path = str(Path("03_yolo_annotated_frames") / annotated_name).replace("\\", "/")
                for detection_payload in frame_payload["detections"]:
                    detection_payload["annotated_frame_path"] = relative_annotated_path

            duplicate_group_counter = 0
            for left_index in range(len(frame_payload["detections"])):
                left_detection = frame_payload["detections"][left_index]
                for right_index in range(left_index + 1, len(frame_payload["detections"])):
                    right_detection = frame_payload["detections"][right_index]
                    overlap = _bbox_iou(left_detection["bbox_xyxy"], right_detection["bbox_xyxy"])
                    if overlap >= 0.70:
                        duplicate_group_counter += 1
                        group_id = f"{frame_payload['frame_id']}_dup_{duplicate_group_counter:03d}"
                        if left_detection["possible_duplicate_group_id"] is None:
                            left_detection["possible_duplicate_group_id"] = group_id
                        if right_detection["possible_duplicate_group_id"] is None:
                            right_detection["possible_duplicate_group_id"] = group_id

            frame_results.append(frame_payload)
            top_detection_frames.append(
                {
                    "frame_id": frame_payload["frame_id"],
                    "timestamp_seconds": frame_payload["timestamp_seconds"],
                    "detection_count": len(frame_payload["detections"]),
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "frame_id": frame_payload["frame_id"],
                    "image_path": frame_payload["image_path"],
                    "error_message": str(exc),
                }
            )
            frame_results.append(frame_payload)

    frames_processed = len(frame_results)
    frames_without_detections = max(0, frames_processed - frames_with_detections)

    detection_payload = {
        "status": "success",
        "input_manifest": "02A_adaptive_frames.json",
        "models_used": [
            {
                "model_role": str(item["model_role"]),
                "model_path": str(item["model_path"]),
                "model_device": model_device_map.get(str(item["model_role"])),
                "inference_output_device": inference_device_map.get(str(item["model_role"])),
            }
            for item in active_specs
        ],
        "yolo_conf_threshold": conf_threshold,
        "yolo_iou_threshold": iou_threshold,
        "device_used": device,
        "cuda_memory_allocated_mb": cuda_memory_allocated_mb(),
        "cuda_memory_reserved_mb": cuda_memory_reserved_mb(),
        "input_frame_count": len(adaptive_frames),
        "frames_processed": frames_processed,
        "frames_with_detections": frames_with_detections,
        "frames_without_detections": frames_without_detections,
        "total_detections": total_detections,
        "class_counts": dict(sorted(class_counts.items())),
        "model_role_counts": dict(sorted(model_role_counts.items())),
        "detections": frame_results,
    }

    top_detection_frames = sorted(
        top_detection_frames,
        key=lambda item: item["detection_count"],
        reverse=True,
    )[:10]

    report_payload = {
        "status": "success",
        "source_manifest_used": "02A_adaptive_frames.json",
        "frames_processed": frames_processed,
        "frames_with_detections": frames_with_detections,
        "frames_without_detections": frames_without_detections,
        "total_detections": total_detections,
        "class_counts": dict(sorted(class_counts.items())),
        "model_role_counts": dict(sorted(model_role_counts.items())),
        "avg_detections_per_frame": round(total_detections / frames_processed, 6) if frames_processed > 0 else 0.0,
        "confidence_stats": _safe_stats(confidence_values),
        "bbox_area_ratio_stats": _safe_stats(bbox_area_ratio_values),
        "top_detection_frames": top_detection_frames,
        "output_folders": {
            "annotated_frames": "03_yolo_annotated_frames",
            "object_crops": "03_yolo_object_crops",
        },
        "failures": failures,
    }

    write_json(run_dir / "03_yolo_detections.json", detection_payload)
    write_json(run_dir / "03_yolo_detection_report.json", report_payload)
    return detection_payload, report_payload
