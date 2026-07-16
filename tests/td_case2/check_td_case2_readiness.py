from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from config import (
    DEFAULT_QWEN_API_MODEL,
    DEFAULT_QWEN_API_PROVIDER,
    DEFAULT_STEP11_5_MODEL_PATH,
    DEFAULT_STEP14_MODEL_PATH,
    DEFAULT_VLM_BACKEND,
    ENV_DEVICE,
    ENV_DEVICE_INDEX,
    ENV_FLORENCE_ADAPTER_PATH,
    ENV_FLORENCE_MODEL_PATH,
    ENV_INPUT_VIDEO,
    ENV_OBJECT_YOLO_MODEL_PATH,
    ENV_OUTPUT_ROOT,
    ENV_PERSON_YOLO_MODEL_PATH,
    ENV_PLATE_DETECTOR_MODEL_PATH,
    ENV_QWEN_API_KEY,
    ENV_QWEN_API_MODEL,
    ENV_QWEN_API_PROVIDER,
    ENV_RUN_DIR,
    ENV_STEP11_5_MODEL_PATH,
    ENV_STEP14_MODEL_PATH,
    ENV_VLM_BACKEND,
    ENV_YOLO_MODEL_PATH,
    CASE_ENV_PATH,
    default_output_root,
    read_local_env_path,
    repo_root,
    resolve_case_path,
)
from device_manager import get_runtime_info
from qwen_4bit import validate_prequantized_nf4_checkpoint
from run_td_case2_step03_yolo import DEFAULT_OBJECT_MODEL_PATH, DEFAULT_PERSON_MODEL_PATH
from run_td_case2_step04a_florence_audit import DEFAULT_FLORENCE_ADAPTER_PATH
from run_td_case2_step06_ocr_color import DEFAULT_PLATE_DETECTOR_MODEL_PATH
from stage_checks import write_json
from step_04a_florence_model_audit import ADAPTER_EXPECTED_FILES, BASE_MODEL_REQUIRED_FILES, inspect_path_files


HF_OFFLINE_ENV_NAMES = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
DOCUMENTED_FLORENCE_PATH = Path(r"C:\Mukul K\models\Florence-2-base-ft")
PACKAGE_CHECKS = [
    {
        "package": "python-dotenv",
        "module": "dotenv",
        "required_by": "config.py local .env loading",
        "required": True,
    },
    {
        "package": "opencv-python",
        "module": "cv2",
        "required_by": "Steps 01, 02A, 03A, 03B, 04A, 04B, 05, 06, 13, 16",
        "required": True,
    },
    {
        "package": "numpy",
        "module": "numpy",
        "required_by": "Steps 02A, 05, 13, 16, vehicle colour helpers",
        "required": True,
    },
    {
        "package": "torch",
        "module": "torch",
        "required_by": "device_manager, Florence, local Qwen stages",
        "required": True,
    },
    {
        "package": "ultralytics",
        "module": "ultralytics",
        "required_by": "YOLO detection, plate detector",
        "required": True,
    },
    {
        "package": "transformers",
        "module": "transformers",
        "required_by": "Florence, local Qwen stages",
        "required": True,
    },
    {
        "package": "bitsandbytes",
        "module": "bitsandbytes",
        "required_by": "Step 11.5 / Step 14 local 4-bit Qwen only",
        "required": False,
    },
    {
        "package": "qwen-vl-utils",
        "module": "qwen_vl_utils",
        "required_by": "Step 11.5 / Step 14 local Qwen vision preprocessing",
        "required": False,
    },
    {
        "package": "streamlit",
        "module": "streamlit",
        "required_by": "Optional td_case2 UI pages only",
        "required": False,
    },
    {
        "package": "pandas",
        "module": "pandas",
        "required_by": "Optional td_case2 UI pages only",
        "required": False,
    },
]


def _safe_bool_env(name: str) -> bool:
    return bool(os.environ.get(name, "").strip())


def _module_installed(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def _package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _resolve_optional_path(raw_value: str | None, default_path: Path | None = None) -> Path | None:
    if raw_value and raw_value.strip():
        candidate = Path(raw_value.strip()).expanduser()
        if not candidate.is_absolute():
            candidate = resolve_case_path(str(candidate))
        return candidate.resolve()
    if default_path is not None and default_path.exists():
        return default_path.resolve()
    return None


def _env_path(name: str) -> Path | None:
    raw_value = os.environ.get(name, "").strip()
    if raw_value:
        return _resolve_optional_path(raw_value)
    return read_local_env_path(name)


def _ensure_writable_dir(path: Path) -> tuple[bool, str | None]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe_path = path / ".td_case2_write_probe"
        probe_path.write_text("ok", encoding="utf-8")
        probe_path.unlink()
        return True, None
    except Exception as exc:
        return False, str(exc)


def _resolve_report_dir(case_dir: Path) -> tuple[Path, str]:
    preferred = case_dir / "readiness_reports"
    preferred_ok, _preferred_error = _ensure_writable_dir(preferred)
    if preferred_ok:
        return preferred, "preferred"
    fallback = repo_root() / "debug_runs" / "td_case2_readiness_reports"
    fallback_ok, fallback_error = _ensure_writable_dir(fallback)
    if fallback_ok:
        return fallback, "fallback"
    raise PermissionError(
        "Unable to create a writable readiness-report directory. "
        f"Preferred={preferred} Fallback={fallback} LastError={fallback_error}"
    )


def _probe_yolo_model(path: Path | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "resolved_path": str(path) if path is not None else None,
        "exists": bool(path and path.exists()),
        "load_status": "not_checked",
        "class_names_count": None,
        "error": None,
    }
    if path is None:
        result["load_status"] = "missing"
        return result
    if not path.exists():
        result["load_status"] = "missing"
        return result
    try:
        from ultralytics import YOLO  # type: ignore

        model = YOLO(str(path))
        names = getattr(model, "names", {})
        result["load_status"] = "success"
        result["class_names_count"] = len(names) if hasattr(names, "__len__") else None
    except Exception as exc:
        result["load_status"] = "failed"
        result["error"] = str(exc)
    return result


def _probe_florence(path: Path | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "resolved_path": str(path) if path is not None else None,
        "exists": bool(path and path.exists()),
        "required_files": inspect_path_files(path, BASE_MODEL_REQUIRED_FILES) if path is not None else None,
        "processor_load": "not_checked",
        "config_load": "not_checked",
        "error": None,
    }
    if path is None or not path.exists():
        result["processor_load"] = "missing"
        result["config_load"] = "missing"
        return result
    try:
        from transformers import AutoConfig, AutoProcessor

        AutoProcessor.from_pretrained(str(path), trust_remote_code=True, local_files_only=True)
        result["processor_load"] = "success"
        AutoConfig.from_pretrained(str(path), trust_remote_code=True, local_files_only=True)
        result["config_load"] = "success"
    except Exception as exc:
        result["processor_load"] = "failed"
        result["config_load"] = "failed"
        result["error"] = str(exc)
    return result


def _probe_adapter(path: Path | None) -> dict[str, Any]:
    summary = inspect_path_files(path, ADAPTER_EXPECTED_FILES) if path is not None else None
    return {
        "resolved_path": str(path) if path is not None else None,
        "exists": bool(path and path.exists()),
        "required_files": summary,
    }


def _probe_qwen_checkpoint(path: Path | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "resolved_path": str(path) if path is not None else None,
        "exists": bool(path and path.exists()),
        "nf4_validation": "not_checked",
        "processor_load": "not_checked",
        "error": None,
    }
    if path is None or not path.exists():
        result["nf4_validation"] = "missing"
        result["processor_load"] = "missing"
        return result
    try:
        validate_prequantized_nf4_checkpoint(path)
        result["nf4_validation"] = "success"
    except Exception as exc:
        result["nf4_validation"] = "failed"
        result["error"] = str(exc)
        return result
    try:
        from transformers import AutoProcessor

        AutoProcessor.from_pretrained(str(path), trust_remote_code=True, local_files_only=True)
        result["processor_load"] = "success"
    except Exception as exc:
        result["processor_load"] = "failed"
        result["error"] = str(exc)
    return result


def _video_check(video_path: str | None) -> dict[str, Any]:
    if not video_path:
        return {
            "provided": False,
            "resolved_path": None,
            "exists": None,
            "is_file": None,
            "status": "warning",
            "message": "No video path provided yet.",
        }
    candidate = Path(video_path).expanduser()
    if not candidate.is_absolute():
        candidate = candidate.resolve()
    return {
        "provided": True,
        "resolved_path": str(candidate),
        "exists": candidate.exists(),
        "is_file": candidate.is_file(),
        "status": "pass" if candidate.exists() and candidate.is_file() else "fail",
        "message": "Video path is valid." if candidate.exists() and candidate.is_file() else "Video path is missing or not a file.",
    }


def _build_active_pipeline_summary() -> dict[str, Any]:
    return {
        "search_ready_pipeline": {
            "runner": "tests/td_case2/run_td_case2_search_ready_pipeline.py",
            "stage_order": [
                "run_td_case2_step01_02_02a.py",
                "run_td_case2_step03_yolo.py",
                "run_td_case2_step04a_florence_audit.py",
                "run_td_case2_step04b_tracking.py",
                "run_td_case2_step05_best_frames.py",
                "run_td_case2_step06_ocr_color.py",
                "run_td_case2_step07b_traffic_object_search_index.py",
                "run_td_case2_step08b_dynamic_search_validation.py",
                "run_td_case2_step09b_universal_search_cards.py",
                "run_td_case2_step10b_universal_search_demo.py",
            ],
        },
        "event_vlm_pipeline": {
            "runner": "tests/td_case2/run_td_case2_vlm_event_pipeline.py",
            "stage_order": [
                "run_td_case2_step11_event_candidates.py",
                "run_td_case2_step11_5_vlm_filter.py",
                "run_td_case2_step12_event_ranking.py",
                "run_td_case2_step13_vlm_inputs.py",
                "run_td_case2_step14_vlm_review.py",
                "run_td_case2_step15_searchable_events.py",
                "run_td_case2_step16_evidence_video.py",
            ],
        },
        "complete_end_to_end_runner": {
            "runner": None,
            "notes": "No single checked-in Python runner executes Steps 01 through 16 end to end. The active flow is split into search-ready and VLM/event pipeline runners.",
        },
        "deprecated_or_older_runners": [
            "tests/td_case2/run_td_case2_step01_02.py",
            "tests/td_case2/run_td_case2_step07_search_index.py",
            "tests/td_case2/run_td_case2_step08_query_validation.py",
            "tests/td_case2/run_td_case2_step09_result_packaging.py",
            "tests/td_case2/run_td_case2_step10_search_demo.py",
            "tests/td_case2/experiments/dynamic_yolo_tracking/*",
            "tests/td_case2/experiments/step04b_bytetrack/*",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight checker for the active td_case2 pipeline on this machine.")
    parser.add_argument("--video-path", help="Optional absolute video path to validate without starting the pipeline.")
    args = parser.parse_args()

    repo = repo_root()
    case_dir = repo / "tests" / "td_case2"
    output_root = Path(os.environ.get(ENV_OUTPUT_ROOT, "").strip()).expanduser() if os.environ.get(ENV_OUTPUT_ROOT, "").strip() else default_output_root()
    if not output_root.is_absolute():
        output_root = (Path.cwd() / output_root).resolve()
    readiness_dir, readiness_dir_mode = _resolve_report_dir(case_dir)

    runtime = get_runtime_info()
    status_items: list[dict[str, Any]] = []
    blocking: list[str] = []
    warnings: list[str] = []
    passes: list[str] = []

    def record(status: str, category: str, message: str, **extra: Any) -> None:
        item = {"status": status, "category": category, "message": message}
        if extra:
            item.update(extra)
        status_items.append(item)
        if status == "FAIL":
            blocking.append(message)
        elif status == "WARNING":
            warnings.append(message)
        else:
            passes.append(message)

    record("PASS", "environment", f"Repository root resolved to {repo}.")
    record("PASS", "environment", f"Using Python executable {sys.executable}.", python_version=sys.version.split()[0], platform=platform.platform())

    if sys.version_info[:2] == (3, 11):
        record("PASS", "environment", "Python 3.11 is in use, matching the td_case2 setup guidance.")
    else:
        record("FAIL", "environment", f"Python {sys.version.split()[0]} is not the expected 3.11 runtime for td_case2.")

    writable, writable_error = _ensure_writable_dir(output_root)
    if writable:
        record("PASS", "paths", f"Output root is writable: {output_root}")
    else:
        record("FAIL", "paths", f"Output root is not writable: {output_root}", error=writable_error)

    current_backend = os.environ.get(ENV_VLM_BACKEND, DEFAULT_VLM_BACKEND).strip().lower() or DEFAULT_VLM_BACKEND
    record("PASS", "environment", f"VLM backend resolved to {current_backend}.")

    for package_check in PACKAGE_CHECKS:
        version = _package_version(package_check["package"])
        installed = version is not None and _module_installed(package_check["module"])
        required = bool(package_check["required"])
        if installed:
            record(
                "PASS",
                "dependency",
                f"{package_check['package']} is installed.",
                package=package_check["package"],
                required_by=package_check["required_by"],
                version=version,
            )
        elif required:
            record(
                "FAIL",
                "dependency",
                f"{package_check['package']} is missing but required by the active pipeline.",
                package=package_check["package"],
                required_by=package_check["required_by"],
            )
        else:
            record(
                "WARNING",
                "dependency",
                f"{package_check['package']} is not installed; it is only needed for optional pipeline modes.",
                package=package_check["package"],
                required_by=package_check["required_by"],
            )

    external_tools = {
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
        "git": shutil.which("git"),
        "nvidia-smi": shutil.which("nvidia-smi"),
    }
    for tool_name, tool_path in external_tools.items():
        if tool_path:
            record("PASS", "external_tool", f"{tool_name} is on PATH.", tool=tool_name, path=tool_path)
        else:
            level = "WARNING" if tool_name in {"ffmpeg", "ffprobe", "git"} else "FAIL"
            record(level, "external_tool", f"{tool_name} is not on PATH.", tool=tool_name)

    if runtime.cuda_available:
        record(
            "PASS",
            "gpu",
            f"CUDA is available on {runtime.cuda_device_name}.",
            total_vram_mb=runtime.cuda_total_vram_mb,
            bf16_supported=runtime.cuda_bf16_supported,
            opencv_cuda_available=runtime.opencv_cuda_available,
        )
    else:
        record("WARNING", "gpu", "CUDA is not available. GPU-first stages would fall back to CPU or fail when CUDA is required.")

    if _module_installed("bitsandbytes"):
        record("PASS", "gpu", "bitsandbytes imports successfully on this machine.")
    else:
        record("WARNING", "gpu", "bitsandbytes is unavailable. Local 4-bit Qwen backends will not run.")

    if _module_installed("qwen_vl_utils"):
        record("PASS", "gpu", "qwen_vl_utils imports successfully for local Qwen stages.")
    else:
        record("WARNING", "gpu", "qwen_vl_utils is unavailable. Local Qwen stages will not run.")

    for offline_name in HF_OFFLINE_ENV_NAMES:
        if _safe_bool_env(offline_name):
            record("PASS", "offline_mode", f"{offline_name}=1 is set.")
        else:
            record("WARNING", "offline_mode", f"{offline_name} is not set. Local model code uses local_files_only, but explicit offline mode is still recommended.")

    person_model_path = _env_path(ENV_PERSON_YOLO_MODEL_PATH) or (DEFAULT_PERSON_MODEL_PATH.resolve() if DEFAULT_PERSON_MODEL_PATH.exists() else None)
    object_model_path = _env_path(ENV_OBJECT_YOLO_MODEL_PATH) or (DEFAULT_OBJECT_MODEL_PATH.resolve() if DEFAULT_OBJECT_MODEL_PATH.exists() else None)
    combined_model_path = _env_path(ENV_YOLO_MODEL_PATH) or ((repo / "yolo11m.pt").resolve() if (repo / "yolo11m.pt").exists() else None)
    florence_model_path = _env_path(ENV_FLORENCE_MODEL_PATH) or (DOCUMENTED_FLORENCE_PATH.resolve() if DOCUMENTED_FLORENCE_PATH.exists() else None)
    florence_adapter_path = _env_path(ENV_FLORENCE_ADAPTER_PATH) or (DEFAULT_FLORENCE_ADAPTER_PATH.resolve() if DEFAULT_FLORENCE_ADAPTER_PATH.exists() else None)
    plate_detector_model_path = _env_path(ENV_PLATE_DETECTOR_MODEL_PATH) or (DEFAULT_PLATE_DETECTOR_MODEL_PATH.resolve() if DEFAULT_PLATE_DETECTOR_MODEL_PATH.exists() else None)
    step11_5_model_path = _env_path(ENV_STEP11_5_MODEL_PATH) or Path(DEFAULT_STEP11_5_MODEL_PATH).resolve()
    step14_model_path = _env_path(ENV_STEP14_MODEL_PATH) or Path(DEFAULT_STEP14_MODEL_PATH).resolve()

    yolo_person_probe = _probe_yolo_model(person_model_path)
    yolo_object_probe = _probe_yolo_model(object_model_path)
    yolo_combined_probe = _probe_yolo_model(combined_model_path)
    plate_detector_probe = _probe_yolo_model(plate_detector_model_path)
    florence_probe = _probe_florence(florence_model_path)
    florence_adapter_probe = _probe_adapter(florence_adapter_path)
    step11_5_probe = _probe_qwen_checkpoint(step11_5_model_path)
    step14_probe = _probe_qwen_checkpoint(step14_model_path)

    if yolo_person_probe["load_status"] == "success":
        record("PASS", "model", f"Person YOLO model loads: {yolo_person_probe['resolved_path']}")
    else:
        record("FAIL", "model", f"Person YOLO model is not usable: {yolo_person_probe['resolved_path']}", details=yolo_person_probe)

    if yolo_object_probe["load_status"] == "success":
        record("PASS", "model", f"Object/vehicle YOLO model loads: {yolo_object_probe['resolved_path']}")
    else:
        record("WARNING", "model", f"Configured/default object YOLO path is not usable: {yolo_object_probe['resolved_path']}", details=yolo_object_probe)

    if yolo_combined_probe["load_status"] == "success":
        record("PASS", "model", f"Combined fallback YOLO model loads: {yolo_combined_probe['resolved_path']}")
    else:
        record("FAIL", "model", f"Combined fallback YOLO model is not usable: {yolo_combined_probe['resolved_path']}", details=yolo_combined_probe)

    if plate_detector_probe["load_status"] == "success":
        record("PASS", "model", f"Plate detector model loads: {plate_detector_probe['resolved_path']}")
    else:
        record("FAIL", "model", f"Plate detector model is not usable: {plate_detector_probe['resolved_path']}", details=plate_detector_probe)

    if florence_probe["processor_load"] == "success" and florence_probe["config_load"] == "success":
        record("PASS", "model", f"Florence base model passes offline processor/config loading: {florence_probe['resolved_path']}")
    else:
        record("FAIL", "model", f"Florence base model is not ready for offline loading: {florence_probe['resolved_path']}", details=florence_probe)

    if florence_adapter_probe["exists"]:
        record("PASS", "model", f"Florence adapter directory exists: {florence_adapter_probe['resolved_path']}", details=florence_adapter_probe)
    else:
        record("WARNING", "model", "Florence adapter path is not configured or missing. The base Florence model can still run if the adapter is optional.", details=florence_adapter_probe)

    if current_backend == "local_qwen":
        if step11_5_probe["processor_load"] == "success":
            record("PASS", "model", f"Step 11.5 local Qwen checkpoint is ready: {step11_5_probe['resolved_path']}")
        else:
            record("FAIL", "model", f"Step 11.5 local Qwen checkpoint is not ready: {step11_5_probe['resolved_path']}", details=step11_5_probe)
        if step14_probe["processor_load"] == "success":
            record("PASS", "model", f"Step 14 local Qwen checkpoint is ready: {step14_probe['resolved_path']}")
        else:
            record("FAIL", "model", f"Step 14 local Qwen checkpoint is not ready: {step14_probe['resolved_path']}", details=step14_probe)
    elif current_backend == "api_qwen":
        if _safe_bool_env(ENV_QWEN_API_KEY):
            record("PASS", "environment", "TD_CASE2_QWEN_API_KEY is set for api_qwen mode.")
        else:
            record("FAIL", "environment", "TD_CASE2_QWEN_API_KEY is required when TD_CASE2_VLM_BACKEND=api_qwen.")
    else:
        record("PASS", "environment", "TD_CASE2_VLM_BACKEND=disabled avoids local/API Qwen requirements for the event pipeline.")

    if not os.environ.get(ENV_FLORENCE_MODEL_PATH, "").strip() and not read_local_env_path(ENV_FLORENCE_MODEL_PATH):
        record("WARNING", "environment", f"{ENV_FLORENCE_MODEL_PATH} is not explicitly set. The checker found a documented Florence directory, but your run command should still set it.")

    if not os.environ.get(ENV_OBJECT_YOLO_MODEL_PATH, "").strip():
        record("WARNING", "environment", f"{ENV_OBJECT_YOLO_MODEL_PATH} is not explicitly set. The code default points at a directory that does not load under Ultralytics; use the actual .pt file instead.")

    if not os.environ.get(ENV_VLM_BACKEND, "").strip():
        record("WARNING", "environment", f"{ENV_VLM_BACKEND} is not explicitly set. The code default is {DEFAULT_VLM_BACKEND}, which requires local Qwen checkpoints that are not present here.")

    if not CASE_ENV_PATH.exists():
        record("WARNING", "environment", f"Local case env file is missing: {CASE_ENV_PATH}")
    else:
        record("PASS", "environment", f"Local case env file exists: {CASE_ENV_PATH}")

    video_result = _video_check(args.video_path or os.environ.get(ENV_INPUT_VIDEO, "").strip() or None)
    if video_result["status"] == "pass":
        record("PASS", "video", video_result["message"], details=video_result)
    elif video_result["status"] == "fail":
        record("FAIL", "video", video_result["message"], details=video_result)
    else:
        record("WARNING", "video", video_result["message"], details=video_result)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "repository_root": str(repo),
        "case_root": str(case_dir),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "active_virtual_env": os.environ.get("VIRTUAL_ENV", ""),
        "pip_executable": str((Path(sys.executable).parent / "pip.exe").resolve()) if (Path(sys.executable).parent / "pip.exe").exists() else None,
        "runtime": {
            "cuda_available": runtime.cuda_available,
            "cuda_device_count": runtime.cuda_device_count,
            "cuda_device_name": runtime.cuda_device_name,
            "cuda_total_vram_mb": runtime.cuda_total_vram_mb,
            "cuda_bf16_supported": runtime.cuda_bf16_supported,
            "opencv_cuda_available": runtime.opencv_cuda_available,
            "onnxruntime_cuda_available": runtime.onnxruntime_cuda_available,
            "tensorrt_available": runtime.tensorrt_available,
        },
        "active_pipeline": _build_active_pipeline_summary(),
        "report_directory_mode": readiness_dir_mode,
        "external_tools": external_tools,
        "models": {
            "person_yolo": yolo_person_probe,
            "object_yolo": yolo_object_probe,
            "combined_yolo": yolo_combined_probe,
            "plate_detector": plate_detector_probe,
            "florence_base": florence_probe,
            "florence_adapter": florence_adapter_probe,
            "step11_5_local_qwen": step11_5_probe,
            "step14_local_qwen": step14_probe,
        },
        "env": {
            ENV_DEVICE: os.environ.get(ENV_DEVICE, ""),
            ENV_DEVICE_INDEX: os.environ.get(ENV_DEVICE_INDEX, ""),
            ENV_INPUT_VIDEO: os.environ.get(ENV_INPUT_VIDEO, ""),
            ENV_OUTPUT_ROOT: os.environ.get(ENV_OUTPUT_ROOT, ""),
            ENV_RUN_DIR: os.environ.get(ENV_RUN_DIR, ""),
            ENV_PERSON_YOLO_MODEL_PATH: os.environ.get(ENV_PERSON_YOLO_MODEL_PATH, ""),
            ENV_OBJECT_YOLO_MODEL_PATH: os.environ.get(ENV_OBJECT_YOLO_MODEL_PATH, ""),
            ENV_YOLO_MODEL_PATH: os.environ.get(ENV_YOLO_MODEL_PATH, ""),
            ENV_FLORENCE_MODEL_PATH: os.environ.get(ENV_FLORENCE_MODEL_PATH, ""),
            ENV_FLORENCE_ADAPTER_PATH: os.environ.get(ENV_FLORENCE_ADAPTER_PATH, ""),
            ENV_PLATE_DETECTOR_MODEL_PATH: os.environ.get(ENV_PLATE_DETECTOR_MODEL_PATH, ""),
            ENV_VLM_BACKEND: current_backend,
            ENV_STEP11_5_MODEL_PATH: os.environ.get(ENV_STEP11_5_MODEL_PATH, DEFAULT_STEP11_5_MODEL_PATH),
            ENV_STEP14_MODEL_PATH: os.environ.get(ENV_STEP14_MODEL_PATH, DEFAULT_STEP14_MODEL_PATH),
            ENV_QWEN_API_PROVIDER: os.environ.get(ENV_QWEN_API_PROVIDER, DEFAULT_QWEN_API_PROVIDER),
            ENV_QWEN_API_MODEL: os.environ.get(ENV_QWEN_API_MODEL, DEFAULT_QWEN_API_MODEL),
            ENV_QWEN_API_KEY: "<set>" if _safe_bool_env(ENV_QWEN_API_KEY) else "",
            HF_OFFLINE_ENV_NAMES[0]: os.environ.get(HF_OFFLINE_ENV_NAMES[0], ""),
            HF_OFFLINE_ENV_NAMES[1]: os.environ.get(HF_OFFLINE_ENV_NAMES[1], ""),
        },
        "status_items": status_items,
        "summary": {
            "pass_count": len(passes),
            "warning_count": len(warnings),
            "fail_count": len(blocking),
            "passes": passes,
            "warnings": warnings,
            "blocking_failures": blocking,
            "ready": len(blocking) == 0,
        },
    }

    report_path = readiness_dir / f"td_case2_readiness_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    write_json(report_path, report)

    print(f"Readiness report: {report_path}")
    print(f"PASS: {len(passes)}")
    print(f"WARNING: {len(warnings)}")
    print(f"FAIL: {len(blocking)}")
    for item in status_items:
        print(f"{item['status']}: [{item['category']}] {item['message']}")

    return 0 if not blocking else 1


if __name__ == "__main__":
    raise SystemExit(main())
