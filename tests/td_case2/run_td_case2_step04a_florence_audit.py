from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from config import (
    DEFAULT_FLORENCE_AUDIT_LIMIT,
    DEFAULT_FLORENCE_AUDIT_NATIVE_TASKS,
    DEFAULT_FLORENCE_DEVICE,
    DEFAULT_FLORENCE_MAX_NEW_TOKENS,
    DEFAULT_FLORENCE_NUM_BEAMS,
    DEFAULT_FLORENCE_RUN_JSON_PROMPT_TEST,
    DEFAULT_FLORENCE_SAVE_AUDIT_INPUTS,
    DEFAULT_FLORENCE_TASK_MODE,
    ENV_FLORENCE_ADAPTER_PATH,
    ENV_FLORENCE_AUDIT_LIMIT,
    ENV_FLORENCE_AUDIT_NATIVE_TASKS,
    ENV_FLORENCE_DEVICE,
    ENV_FLORENCE_MAX_NEW_TOKENS,
    ENV_FLORENCE_MODEL_PATH,
    ENV_FLORENCE_NUM_BEAMS,
    ENV_FLORENCE_RUN_JSON_PROMPT_TEST,
    ENV_FLORENCE_SAVE_AUDIT_INPUTS,
    ENV_FLORENCE_TASK_MODE,
    ENV_PLATE_DETECTOR_MODEL_PATH,
    ENV_RUN_DIR,
    read_local_env_path,
    resolve_case_path,
)
from device_manager import record_stage_device, resolve_device
from run_td_case2_step01_02 import log
from stage_checks import build_failure_payload, update_stage_gate_report, write_json
from step_04a_florence_model_audit import (
    ADAPTER_EXPECTED_FILES,
    BASE_MODEL_REQUIRED_FILES,
    inspect_path_files,
    run_florence_audit,
)


DEFAULT_FLORENCE_ADAPTER_PATH = Path(r"C:\Mukul K\vinfo1\video-search-engine\ocr_colour\adaptor_florance_baseFT")
SUPPORTED_TASK_MODES = {"ocr", "color", "ocr_and_color"}


@dataclass(frozen=True)
class FlorenceAuditConfig:
    """Runtime settings for the isolated td_case2 Florence audit."""

    run_dir: Path
    florence_model_path: Path
    florence_adapter_path: Path | None
    device: str
    device_reason: str
    audit_limit: int
    task_mode: str
    max_new_tokens: int
    num_beams: int
    native_tasks: list[str]
    run_json_prompt_test: bool
    save_audit_inputs: bool
    plate_detector_model_path: Path | None


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


def _resolve_optional_path(raw_value: str | None, default_path: Path | None = None) -> Path | None:
    """Resolve an optional path from env or default."""

    if raw_value and raw_value.strip():
        candidate = Path(raw_value.strip()).expanduser()
    elif default_path is not None and default_path.exists():
        candidate = default_path
    else:
        return None
    if not candidate.is_absolute():
        candidate = resolve_case_path(str(candidate))
    return candidate


def _read_native_tasks() -> list[str]:
    """Read comma-separated Florence native task prompts."""

    raw_value = os.environ.get(ENV_FLORENCE_AUDIT_NATIVE_TASKS, DEFAULT_FLORENCE_AUDIT_NATIVE_TASKS)
    requested_tasks = [item.strip() for item in raw_value.split(",") if item.strip()]
    tasks = [item for item in requested_tasks if item in {"<OCR>", "<CAPTION>"}]
    if not tasks:
        raise ValueError(
            f"Environment variable {ENV_FLORENCE_AUDIT_NATIVE_TASKS} must contain <OCR> and/or <CAPTION>."
        )
    return tasks


def read_config() -> FlorenceAuditConfig:
    """Read config for the isolated Step 04A Florence model audit."""

    raw_run_dir = os.environ.get(ENV_RUN_DIR, "").strip()
    if not raw_run_dir:
        raise ValueError(f"Environment variable {ENV_RUN_DIR} is required for Step 04A.")
    run_dir = Path(raw_run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"TD_CASE2_RUN_DIR does not point to an existing directory: {run_dir}")

    required_inputs = [
        run_dir / "03_yolo_detections.json",
        run_dir / "03_yolo_object_crops",
    ]
    for required_path in required_inputs:
        if not required_path.exists():
            raise FileNotFoundError(f"Required Step 04A input is missing: {required_path}")

    raw_model_path = os.environ.get(ENV_FLORENCE_MODEL_PATH, "").strip()
    if not raw_model_path:
        raise ValueError(
            f"Environment variable {ENV_FLORENCE_MODEL_PATH} is required. "
            'Suggested setting: TD_CASE2_FLORENCE_MODEL_PATH="C:\\Mukul K\\models\\Florence-2-base-ft"'
        )
    florence_model_path = resolve_case_path(raw_model_path)
    if not florence_model_path.exists():
        local_env_model_path = read_local_env_path(ENV_FLORENCE_MODEL_PATH)
        if local_env_model_path is not None and local_env_model_path.exists():
            florence_model_path = local_env_model_path
            os.environ[ENV_FLORENCE_MODEL_PATH] = str(florence_model_path)

    florence_adapter_path = _resolve_optional_path(
        os.environ.get(ENV_FLORENCE_ADAPTER_PATH),
        DEFAULT_FLORENCE_ADAPTER_PATH,
    )
    plate_detector_model_path = _resolve_optional_path(os.environ.get(ENV_PLATE_DETECTOR_MODEL_PATH))

    task_mode = os.environ.get(ENV_FLORENCE_TASK_MODE, DEFAULT_FLORENCE_TASK_MODE).strip().lower() or DEFAULT_FLORENCE_TASK_MODE
    if task_mode not in SUPPORTED_TASK_MODES:
        raise ValueError(
            f"Environment variable {ENV_FLORENCE_TASK_MODE} must be one of {sorted(SUPPORTED_TASK_MODES)}. "
            f"Received: {task_mode!r}"
        )

    device_decision = resolve_device(component_name="Step 04A Florence", override_env_names=(ENV_FLORENCE_DEVICE,))

    return FlorenceAuditConfig(
        run_dir=run_dir.resolve(),
        florence_model_path=florence_model_path.resolve(),
        florence_adapter_path=florence_adapter_path.resolve() if florence_adapter_path is not None else None,
        device=device_decision.torch_device,
        device_reason=device_decision.reason,
        audit_limit=_read_positive_int(ENV_FLORENCE_AUDIT_LIMIT, DEFAULT_FLORENCE_AUDIT_LIMIT),
        task_mode=task_mode,
        max_new_tokens=_read_positive_int(ENV_FLORENCE_MAX_NEW_TOKENS, DEFAULT_FLORENCE_MAX_NEW_TOKENS),
        num_beams=_read_positive_int(ENV_FLORENCE_NUM_BEAMS, DEFAULT_FLORENCE_NUM_BEAMS),
        native_tasks=_read_native_tasks(),
        run_json_prompt_test=_read_bool(ENV_FLORENCE_RUN_JSON_PROMPT_TEST, DEFAULT_FLORENCE_RUN_JSON_PROMPT_TEST),
        save_audit_inputs=_read_bool(ENV_FLORENCE_SAVE_AUDIT_INPUTS, DEFAULT_FLORENCE_SAVE_AUDIT_INPUTS),
        plate_detector_model_path=plate_detector_model_path.resolve() if plate_detector_model_path is not None else None,
    )


def _write_failed_reports(
    *,
    run_dir: Path,
    florence_model_path: Path | None,
    florence_adapter_path: Path | None,
    task_mode: str | None,
    audit_limit: int | None,
    error_message: str,
) -> None:
    """Write failure JSON artifacts when Step 04A cannot proceed."""

    base_model_files = inspect_path_files(florence_model_path, BASE_MODEL_REQUIRED_FILES)
    adapter_files_summary = inspect_path_files(florence_adapter_path, ADAPTER_EXPECTED_FILES)
    audit_summary = {
        "status": "failed",
        "run_dir": str(run_dir),
        "florence_model_path": str(florence_model_path) if florence_model_path is not None else None,
        "florence_adapter_path": str(florence_adapter_path) if florence_adapter_path is not None else None,
        "model_path_exists": base_model_files["path_exists"],
        "base_model_required_files": base_model_files["found_files"],
        "base_model_missing_files": base_model_files["missing_files"],
        "adapter_path_exists": adapter_files_summary["path_exists"],
        "adapter_files_summary": adapter_files_summary,
        "model_load_status": "failed",
        "adapter_load_status": "not_provided" if florence_adapter_path is None else ("failed" if adapter_files_summary["path_exists"] else "not_provided"),
        "device_used": None,
        "task_mode": task_mode,
        "audit_crop_limit": audit_limit,
        "selected_crop_count": 0,
        "successful_outputs": 0,
        "failed_outputs": 0,
        "avg_seconds_per_crop": 0.0,
        "native_task_success_count": 0,
        "ocr_native_success_count": 0,
        "caption_success_count": 0,
        "detailed_caption_success_count": 0,
        "json_prompt_success_count": 0,
        "plate_detector_load_status": "not_provided",
        "plate_crop_count": 0,
        "plate_ocr_success_count": 0,
        "ready_for_full_ocr_color_after_tracking": False,
        "recommendation": error_message,
        "error_message": error_message,
    }
    audit_results = {
        "status": "failed",
        "input_yolo_detections_file": "03_yolo_detections.json",
        "selected_crop_count": 0,
        "results": [],
        "error_message": error_message,
    }
    write_json(run_dir / "04A_florence_model_audit.json", audit_summary)
    write_json(run_dir / "04A_florence_audit_results.json", audit_results)


def main() -> None:
    """Run the isolated Florence model audit for td_case2."""

    config = read_config()
    base_model_files = inspect_path_files(config.florence_model_path, BASE_MODEL_REQUIRED_FILES)
    adapter_files_summary = inspect_path_files(config.florence_adapter_path, ADAPTER_EXPECTED_FILES)

    log(f"Run directory: {config.run_dir}")
    log(f"Florence model path: {config.florence_model_path}")
    log(f"Florence device selection: {config.device} ({config.device_reason})")
    log(
        "Model required files status: "
        f"found={base_model_files['found_files']} missing={base_model_files['missing_files']}"
    )
    log(f"Adapter path: {config.florence_adapter_path if config.florence_adapter_path else 'not provided'}")
    log(
        "Adapter files status: "
        f"found={adapter_files_summary['found_files']} missing={adapter_files_summary['missing_files']}"
    )

    outputs_written = False
    try:
        audit_summary, _audit_results = run_florence_audit(
            run_dir=config.run_dir,
            florence_model_path=config.florence_model_path,
            florence_adapter_path=config.florence_adapter_path,
            task_mode=config.task_mode,
            audit_limit=config.audit_limit,
            max_new_tokens=config.max_new_tokens,
            num_beams=config.num_beams,
            native_tasks=config.native_tasks,
            run_json_prompt_test=config.run_json_prompt_test,
            save_audit_inputs=config.save_audit_inputs,
            device=config.device,
            plate_detector_model_path=config.plate_detector_model_path,
        )
        outputs_written = True
        if audit_summary["selected_crop_count"] > 0 and not audit_summary["ready_for_full_ocr_color_after_tracking"]:
            raise RuntimeError(
                "Florence loaded but inference failed for every audit crop. "
                "Inspect 04A_florence_audit_results.json before continuing."
            )
        update_stage_gate_report(
            config.run_dir,
            "04A_florence_model_audit",
            {
                "status": "success",
                "model_load_status": audit_summary["model_load_status"],
                "adapter_load_status": audit_summary["adapter_load_status"],
                "selected_crop_count": audit_summary["selected_crop_count"],
                "successful_outputs": audit_summary["successful_outputs"],
                "failed_outputs": audit_summary["failed_outputs"],
                "ready_for_full_ocr_color_after_tracking": audit_summary["ready_for_full_ocr_color_after_tracking"],
            },
        )
        log(f"Model load status: {audit_summary['model_load_status']}")
        log(f"Device used: {audit_summary['device_used']}")
        log(f"Native OCR successes: {audit_summary['ocr_native_success_count']}")
        log(f"Caption successes: {audit_summary['caption_success_count']}")
        log(f"Plate detector status: {audit_summary['plate_detector_load_status']}")
        log(f"Final recommendation: {audit_summary['recommendation']}")
        log(f"Report paths: {config.run_dir / '04A_florence_model_audit.json'} | {config.run_dir / '04A_florence_audit_results.json'}")
        record_stage_device(
            run_dir=config.run_dir,
            stage_name="04A_florence_model_audit",
            component_name="florence",
            supports_gpu=True,
            selected_device=str(audit_summary.get("device_used") or config.device),
            actual_device=str(audit_summary.get("device_used") or config.device),
            model_device=audit_summary.get("model_device"),
            input_device=str(audit_summary.get("device_used") or config.device),
            output_device=str(audit_summary.get("device_used") or config.device),
            preprocessing_device="cpu",
            inference_device=str(audit_summary.get("device_used") or config.device),
            postprocess_device="cpu",
            vram_allocated_mb=audit_summary.get("cuda_memory_allocated_mb"),
            vram_reserved_mb=audit_summary.get("cuda_memory_reserved_mb"),
            reason=config.device_reason,
            notes=["Crop decode and Florence post-processing stay on CPU; tensors are moved onto the resolved torch device for generation."],
        )
        if audit_summary.get("plate_detector_load_status") == "success":
            record_stage_device(
                run_dir=config.run_dir,
                stage_name="04A_florence_model_audit",
                component_name="plate_detector",
                supports_gpu=True,
                selected_device=str(audit_summary.get("device_used") or config.device),
                actual_device=str(audit_summary.get("device_used") or config.device),
                inference_device=str(audit_summary.get("device_used") or config.device),
                postprocess_device="cpu",
                preprocessing_device="cpu",
                reason="Plate detector inherits the same centralized device decision as Florence during audit.",
                notes=["Plate crops are extracted and written back through OpenCV on CPU."],
            )
    except Exception as exc:
        if not outputs_written:
            _write_failed_reports(
                run_dir=config.run_dir,
                florence_model_path=config.florence_model_path,
                florence_adapter_path=config.florence_adapter_path,
                task_mode=config.task_mode,
                audit_limit=config.audit_limit,
                error_message=str(exc),
            )
        update_stage_gate_report(config.run_dir, "04A_florence_model_audit", build_failure_payload(exc))
        log(f"Step 04A failed: {exc}")
        log(f"Final recommendation: {exc}")
        log(f"Run directory: {config.run_dir}")
        raise


if __name__ == "__main__":
    main()
