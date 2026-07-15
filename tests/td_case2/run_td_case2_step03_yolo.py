from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import (
    DEFAULT_YOLO_AUDIT_FRAME_LIMIT,
    DEFAULT_YOLO_CONF_THRESHOLD,
    DEFAULT_YOLO_IOU_THRESHOLD,
    DEFAULT_YOLO_SAVE_ANNOTATED,
    DEFAULT_YOLO_SAVE_CROPS,
    ENV_OBJECT_YOLO_MODEL_PATH,
    ENV_PERSON_YOLO_MODEL_PATH,
    ENV_RUN_DIR,
    ENV_YOLO_AUDIT_FRAME_LIMIT,
    ENV_YOLO_CONF_THRESHOLD,
    ENV_YOLO_DEVICE,
    ENV_YOLO_IOU_THRESHOLD,
    ENV_YOLO_MAX_FRAMES,
    ENV_YOLO_MODEL_PATH,
    ENV_YOLO_SAVE_ANNOTATED,
    ENV_YOLO_SAVE_CROPS,
    resolve_case_path,
)
from device_manager import record_stage_device, resolve_device
from run_td_case2_step01_02 import log
from stage_checks import build_failure_payload, read_json, update_stage_gate_report
from step_03a_yolo_model_audit import run_yolo_model_audit
from step_03b_yolo_detection import run_yolo_detection


DEFAULT_PERSON_MODEL_PATH = Path(r"C:\Mukul K\vinfo1\video-search-engine\object\Person_detection (1)\Person_detection.pt")
DEFAULT_OBJECT_MODEL_PATH = Path(r"C:\Mukul K\vinfo1\video-search-engine\object\vehical_detection")


@dataclass(frozen=True)
class YoloStepConfig:
    """Runtime configuration for isolated td_case2 YOLO audit and detection."""

    run_dir: Path
    model_specs: list[dict[str, Any]]
    conf_threshold: float
    iou_threshold: float
    device: str
    device_reason: str
    audit_frame_limit: int
    save_annotated: bool
    save_crops: bool
    max_frames: int | None


def _read_bool(env_name: str, default_value: bool) -> bool:
    """Read a permissive boolean-like environment flag."""

    raw_value = os.environ.get(env_name)
    if raw_value is None or raw_value.strip() == "":
        return default_value
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Environment variable {env_name} must be boolean-like. Received: {raw_value!r}")


def _read_positive_float(env_name: str, default_value: float) -> float:
    """Read a positive float from the environment."""

    raw_value = os.environ.get(env_name, str(default_value)).strip()
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {env_name} must be a valid number. Received: {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"Environment variable {env_name} must be greater than 0. Received: {value}")
    return value


def _read_positive_int(env_name: str, default_value: int) -> int:
    """Read a positive integer from the environment."""

    raw_value = os.environ.get(env_name, str(default_value)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {env_name} must be a valid integer. Received: {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"Environment variable {env_name} must be greater than 0. Received: {value}")
    return value


def _read_optional_positive_int(env_name: str) -> int | None:
    """Read an optional positive integer limit."""

    raw_value = os.environ.get(env_name, "").strip()
    if raw_value == "":
        return None
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {env_name} must be a valid integer. Received: {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"Environment variable {env_name} must be greater than 0. Received: {value}")
    return value


def _resolve_model_path(raw_value: str | None, default_path: Path | None) -> Path | None:
    """Resolve a model path from env or optional local default."""

    if raw_value and raw_value.strip():
        candidate = Path(raw_value.strip()).expanduser()
    elif default_path is not None and default_path.exists():
        candidate = default_path
    else:
        return None
    if not candidate.is_absolute():
        candidate = resolve_case_path(str(candidate))
    return candidate


def read_config() -> YoloStepConfig:
    """Read configuration for the isolated Step 03 YOLO pipeline."""

    raw_run_dir = os.environ.get(ENV_RUN_DIR, "").strip()
    if not raw_run_dir:
        raise ValueError(
            f"Environment variable {ENV_RUN_DIR} is required. "
            "Set it to an existing td_case2 run directory before running Step 03."
        )

    run_dir = Path(raw_run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"TD_CASE2_RUN_DIR does not point to an existing directory: {run_dir}")

    required_inputs = [
        run_dir / "01_video_info.json",
        run_dir / "02A_adaptive_frames.json",
        run_dir / "02_sampled_frames",
    ]
    for required_path in required_inputs:
        if not required_path.exists():
            raise FileNotFoundError(f"Required Step 03 input is missing: {required_path}")

    person_model_path = _resolve_model_path(os.environ.get(ENV_PERSON_YOLO_MODEL_PATH), DEFAULT_PERSON_MODEL_PATH)
    object_model_path = _resolve_model_path(os.environ.get(ENV_OBJECT_YOLO_MODEL_PATH), DEFAULT_OBJECT_MODEL_PATH)
    fallback_model_path = _resolve_model_path(os.environ.get(ENV_YOLO_MODEL_PATH), None)

    model_specs: list[dict[str, Any]] = []
    if person_model_path is not None:
        model_specs.append({"model_role": "person", "model_path": str(person_model_path)})
    if object_model_path is not None:
        model_specs.append({"model_role": "object_vehicle", "model_path": str(object_model_path)})
    if not model_specs and fallback_model_path is not None:
        model_specs.append({"model_role": "combined", "model_path": str(fallback_model_path)})

    if not model_specs:
        raise ValueError(
            "At least one YOLO model path must be provided via "
            f"{ENV_PERSON_YOLO_MODEL_PATH}, {ENV_OBJECT_YOLO_MODEL_PATH}, or {ENV_YOLO_MODEL_PATH}."
        )

    device_decision = resolve_device(component_name="Step 03 YOLO", override_env_names=(ENV_YOLO_DEVICE,))

    return YoloStepConfig(
        run_dir=run_dir.resolve(),
        model_specs=model_specs,
        conf_threshold=_read_positive_float(ENV_YOLO_CONF_THRESHOLD, DEFAULT_YOLO_CONF_THRESHOLD),
        iou_threshold=_read_positive_float(ENV_YOLO_IOU_THRESHOLD, DEFAULT_YOLO_IOU_THRESHOLD),
        device=device_decision.ultralytics_device,
        device_reason=device_decision.reason,
        audit_frame_limit=_read_positive_int(ENV_YOLO_AUDIT_FRAME_LIMIT, DEFAULT_YOLO_AUDIT_FRAME_LIMIT),
        save_annotated=_read_bool(ENV_YOLO_SAVE_ANNOTATED, DEFAULT_YOLO_SAVE_ANNOTATED),
        save_crops=_read_bool(ENV_YOLO_SAVE_CROPS, DEFAULT_YOLO_SAVE_CROPS),
        max_frames=_read_optional_positive_int(ENV_YOLO_MAX_FRAMES),
    )


def main() -> None:
    """Run the isolated td_case2 Step 03A/03B YOLO flow."""

    config = read_config()
    log(f"Run directory: {config.run_dir}")
    log("Input manifest used: 02A_adaptive_frames.json")
    log(f"YOLO device selection: {config.device} ({config.device_reason})")
    person_paths = [item["model_path"] for item in config.model_specs if item["model_role"] == "person"]
    object_paths = [item["model_path"] for item in config.model_specs if item["model_role"] == "object_vehicle"]
    log(f"Person model path: {person_paths[0] if person_paths else 'not provided'}")
    log(f"Object model path: {object_paths[0] if object_paths else 'not provided'}")

    audit_payload: dict[str, Any]
    try:
        audit_payload = run_yolo_model_audit(
            run_dir=config.run_dir,
            model_specs=config.model_specs,
            audit_frame_limit=config.audit_frame_limit,
            conf_threshold=config.conf_threshold,
            iou_threshold=config.iou_threshold,
            device=config.device,
            save_annotated=config.save_annotated,
        )
        models_loaded = sum(1 for item in audit_payload["models"] if item.get("load_status") == "success")
        update_stage_gate_report(
            config.run_dir,
            "03A_yolo_model_audit",
            {
                "status": "success",
                "models_checked": len(audit_payload["models"]),
                "models_loaded": models_loaded,
                "overall_ready_for_detection": bool(audit_payload["overall_ready_for_detection"]),
            },
        )
        log(f"Model audit status: ready={audit_payload['overall_ready_for_detection']}")
        for item in audit_payload["models"]:
            log(
                f"Model {item['model_role']}: load_status={item['load_status']} "
                f"class_names={item.get('class_names', {})}"
            )
            if item.get("load_status") == "success":
                actual_device = str(item.get("inference_output_device") or item.get("model_device") or config.device)
                record_stage_device(
                    run_dir=config.run_dir,
                    stage_name="03A_yolo_model_audit",
                    component_name=f"yolo_{item['model_role']}",
                    supports_gpu=True,
                    selected_device=actual_device,
                    actual_device=actual_device,
                    model_device=item.get("model_device"),
                    output_device=item.get("inference_output_device"),
                    inference_device=item.get("inference_output_device"),
                    postprocess_device="cpu",
                    preprocessing_device="ultralytics_internal",
                    vram_allocated_mb=item.get("cuda_memory_allocated_mb"),
                    vram_reserved_mb=item.get("cuda_memory_reserved_mb"),
                    reason=config.device_reason,
                    notes=["Ultralytics handles tensor preprocessing internally before model inference."],
                )
    except Exception as exc:
        update_stage_gate_report(config.run_dir, "03A_yolo_model_audit", build_failure_payload(exc))
        log(f"Step 03A failed: {exc}")
        log(f"Run directory: {config.run_dir}")
        raise

    if not audit_payload.get("overall_ready_for_detection"):
        message = "No YOLO models loaded successfully during Step 03A, so Step 03B was not started."
        update_stage_gate_report(
            config.run_dir,
            "03B_yolo_detection",
            {
                "status": "failed",
                "error_message": message,
                "traceback_short": message,
            },
        )
        raise RuntimeError(message)

    try:
        detections_payload, detection_report = run_yolo_detection(
            run_dir=config.run_dir,
            audit_payload=audit_payload,
            model_specs=config.model_specs,
            conf_threshold=config.conf_threshold,
            iou_threshold=config.iou_threshold,
            device=config.device,
            save_annotated=config.save_annotated,
            save_crops=config.save_crops,
            max_frames=config.max_frames,
        )
        update_stage_gate_report(
            config.run_dir,
            "03B_yolo_detection",
            {
                "status": "success",
                "input_frame_count": detections_payload["input_frame_count"],
                "frames_processed": detections_payload["frames_processed"],
                "frames_with_detections": detections_payload["frames_with_detections"],
                "total_detections": detections_payload["total_detections"],
                "class_counts": detections_payload["class_counts"],
                "model_role_counts": detections_payload["model_role_counts"],
                "annotated_frames_folder_exists": (config.run_dir / "03_yolo_annotated_frames").exists(),
                "object_crops_folder_exists": (config.run_dir / "03_yolo_object_crops").exists(),
            },
        )
        log(f"Frames processed: {detections_payload['frames_processed']}")
        log(f"Total detections: {detections_payload['total_detections']}")
        log(f"Class counts: {detections_payload['class_counts']}")
        log(f"Output paths: {config.run_dir / '03_yolo_annotated_frames'} | {config.run_dir / '03_yolo_object_crops'}")
        for item in detections_payload.get("models_used", []):
            actual_device = str(item.get("inference_output_device") or item.get("model_device") or config.device)
            record_stage_device(
                run_dir=config.run_dir,
                stage_name="03B_yolo_detection",
                component_name=f"yolo_{item['model_role']}",
                supports_gpu=True,
                selected_device=actual_device,
                actual_device=actual_device,
                model_device=item.get("model_device"),
                output_device=item.get("inference_output_device"),
                inference_device=item.get("inference_output_device"),
                postprocess_device="cpu",
                preprocessing_device="ultralytics_internal",
                vram_allocated_mb=detections_payload.get("cuda_memory_allocated_mb"),
                vram_reserved_mb=detections_payload.get("cuda_memory_reserved_mb"),
                reason=config.device_reason,
                notes=["Annotated image rendering and crop JPEG writes remain on CPU via OpenCV."],
            )
    except Exception as exc:
        update_stage_gate_report(config.run_dir, "03B_yolo_detection", build_failure_payload(exc))
        log(f"Step 03B failed: {exc}")
        log(f"Run directory: {config.run_dir}")
        raise


if __name__ == "__main__":
    main()
