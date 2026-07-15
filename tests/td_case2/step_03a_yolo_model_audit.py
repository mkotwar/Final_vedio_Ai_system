from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from device_manager import cuda_memory_allocated_mb, cuda_memory_reserved_mb
from stage_checks import read_json, write_json


def _load_yolo_class() -> Any:
    """Import ultralytics YOLO lazily so audit reports can fail gracefully."""

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:  # pragma: no cover - import failure path depends on env
        raise RuntimeError(f"Failed to import ultralytics.YOLO: {exc}") from exc
    return YOLO


def _normalize_class_names(raw_names: Any) -> dict[str, str]:
    """Normalize YOLO class-name mappings into a JSON-friendly dictionary."""

    if isinstance(raw_names, dict):
        return {str(key): str(value) for key, value in raw_names.items()}
    if isinstance(raw_names, (list, tuple)):
        return {str(index): str(value) for index, value in enumerate(raw_names)}
    return {}


def _resolve_frame_path(run_dir: Path, item: dict[str, Any]) -> Path:
    """Resolve an adaptive frame image path relative to the run directory."""

    image_path = Path(str(item.get("image_path", "")))
    if image_path.is_absolute():
        return image_path
    return (run_dir / image_path).resolve()


def _render_annotated_frame(result: Any) -> Any:
    """Render an annotated YOLO result into an OpenCV image."""

    try:
        return result.plot()
    except Exception as exc:
        raise RuntimeError(f"Failed to render annotated YOLO frame: {exc}") from exc


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


def run_yolo_model_audit(
    *,
    run_dir: Path,
    model_specs: list[dict[str, Any]],
    audit_frame_limit: int,
    conf_threshold: float,
    iou_threshold: float,
    device: str,
    save_annotated: bool,
) -> dict[str, Any]:
    """Audit configured YOLO models before full detection."""

    adaptive_manifest = read_json(run_dir / "02A_adaptive_frames.json")
    adaptive_frames = list(adaptive_manifest.get("selected_frames", []))
    audit_frames = adaptive_frames[:audit_frame_limit]
    annotated_dir = run_dir / "03A_yolo_audit_annotated_frames"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    audit_models: list[dict[str, Any]] = []
    overall_ready_for_detection = False

    for model_spec in model_specs:
        model_role = str(model_spec["model_role"])
        model_path = Path(str(model_spec["model_path"]))
        model_payload = {
            "model_role": model_role,
            "model_path": str(model_path),
            "path_exists": model_path.exists(),
            "load_status": "failed",
            "class_names": {},
            "test_frames_processed": 0,
            "test_total_detections": 0,
            "test_class_counts": {},
            "model_device": None,
            "inference_output_device": None,
            "cuda_memory_allocated_mb": None,
            "cuda_memory_reserved_mb": None,
            "error_message": None,
        }

        if not model_path.exists():
            model_payload["error_message"] = f"Model path does not exist: {model_path}"
            audit_models.append(model_payload)
            continue

        try:
            YOLO = _load_yolo_class()
            model = YOLO(str(model_path))
            model_payload["model_device"] = _safe_model_device(model)
            class_names = _normalize_class_names(getattr(model, "names", {}))
            class_counter: Counter[str] = Counter()
            total_detections = 0

            for frame_index, frame_item in enumerate(audit_frames, start=1):
                frame_path = _resolve_frame_path(run_dir, frame_item)
                if not frame_path.exists():
                    continue
                results = model.predict(
                    source=str(frame_path),
                    conf=conf_threshold,
                    iou=iou_threshold,
                    device=device,
                    verbose=False,
                )
                if not results:
                    model_payload["test_frames_processed"] += 1
                    continue

                result = results[0]
                model_payload["inference_output_device"] = _safe_result_device(result)
                boxes = getattr(result, "boxes", None)
                if boxes is not None and getattr(boxes, "cls", None) is not None:
                    class_ids = boxes.cls.tolist()
                    for class_id_value in class_ids:
                        class_name = class_names.get(str(int(class_id_value)), str(int(class_id_value)))
                        class_counter[class_name] += 1
                    total_detections += len(class_ids)

                if save_annotated:
                    annotated_image = _render_annotated_frame(result)
                    annotated_output_path = annotated_dir / f"{model_role}_{frame_index:03d}_{frame_path.name}"
                    if not cv2.imwrite(str(annotated_output_path), annotated_image):
                        raise RuntimeError(f"Failed to write audit annotated frame: {annotated_output_path}")

                model_payload["test_frames_processed"] += 1

            model_payload["load_status"] = "success"
            model_payload["class_names"] = class_names
            model_payload["test_total_detections"] = total_detections
            model_payload["test_class_counts"] = dict(sorted(class_counter.items()))
            model_payload["cuda_memory_allocated_mb"] = cuda_memory_allocated_mb()
            model_payload["cuda_memory_reserved_mb"] = cuda_memory_reserved_mb()
            overall_ready_for_detection = True
        except Exception as exc:
            model_payload["error_message"] = str(exc)

        audit_models.append(model_payload)

    audit_payload = {
        "status": "success",
        "input_manifest": "02A_adaptive_frames.json",
        "audit_frame_limit": audit_frame_limit,
        "models": audit_models,
        "overall_ready_for_detection": overall_ready_for_detection,
    }
    write_json(run_dir / "03A_yolo_model_audit.json", audit_payload)
    return audit_payload
