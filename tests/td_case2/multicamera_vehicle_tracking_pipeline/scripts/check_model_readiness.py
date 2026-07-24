from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..detection.detection_config import load_detection_config
from ..enrichment.anpr_config import load_anpr_config
from ..enrichment.florence_config import load_florence_config
from ..models.florence_runtime import _ensure_writable_huggingface_modules_cache
from ..models.florence_runtime_factory import (
    DEFAULT_FLORENCE_ADAPTER_PATH,
    DEFAULT_FLORENCE_MODEL_PATH,
    DEFAULT_FLORENCE_PROCESSOR_PATH,
)
from ..models.model_path_resolver import ModelPathResolutionError, resolve_model_path_with_source
from ..models.plate_detector_runtime_factory import DEFAULT_PLATE_DETECTOR_MODEL_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_VEHICLE_DETECTOR_MODEL_PATH = PROJECT_ROOT / "models" / "vehicle_detection" / "best_old.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check multicamera pipeline model readiness.")
    parser.add_argument("--vehicle-detector-model-path", default=None)
    parser.add_argument("--plate-detector-model-path", default=None)
    parser.add_argument("--florence-model-path", default=None)
    parser.add_argument("--florence-processor-path", default=None)
    parser.add_argument("--florence-adapter-path", default=None)
    parser.add_argument("--load-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detection_config_path = PROJECT_ROOT / "tests" / "td_case2" / "multicamera_vehicle_tracking_pipeline" / "config" / "detection.yaml"
    florence_config_path = PROJECT_ROOT / "tests" / "td_case2" / "multicamera_vehicle_tracking_pipeline" / "config" / "florence.yaml"
    anpr_config_path = PROJECT_ROOT / "tests" / "td_case2" / "multicamera_vehicle_tracking_pipeline" / "config" / "anpr.yaml"

    detection_config = load_detection_config(detection_config_path)
    florence_config = load_florence_config(florence_config_path)
    anpr_config = load_anpr_config(anpr_config_path)

    required_ok = True
    report: dict[str, object] = {"project_root": str(PROJECT_ROOT), "models": []}
    checks = [
        {
            "name": "vehicle_detector",
            "cli_value": args.vehicle_detector_model_path,
            "environment_variable": "VEHICLE_DETECTOR_MODEL_PATH",
            "config_value": detection_config.model_path,
            "default_value": DEFAULT_VEHICLE_DETECTOR_MODEL_PATH,
            "expect_directory": False,
            "required_files": [],
        },
        {
            "name": "plate_detector",
            "cli_value": args.plate_detector_model_path,
            "environment_variable": anpr_config.plate_detector.model_path_env,
            "config_value": anpr_config.plate_detector.model_path,
            "default_value": DEFAULT_PLATE_DETECTOR_MODEL_PATH,
            "expect_directory": False,
            "required_files": [],
        },
        {
            "name": "florence_base",
            "cli_value": args.florence_model_path,
            "environment_variable": florence_config.model_path_env,
            "config_value": florence_config.model_path,
            "default_value": DEFAULT_FLORENCE_MODEL_PATH,
            "expect_directory": True,
            "required_files": [
                "config.json",
                "configuration_florence2.py",
                "modeling_florence2.py",
                "processing_florence2.py",
                "preprocessor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
        },
        {
            "name": "florence_processor",
            "cli_value": args.florence_processor_path,
            "environment_variable": florence_config.processor_path_env,
            "config_value": florence_config.processor_path,
            "default_value": DEFAULT_FLORENCE_PROCESSOR_PATH,
            "expect_directory": True,
            "required_files": ["processing_florence2.py"],
        },
        {
            "name": "florence_adapter",
            "cli_value": args.florence_adapter_path,
            "environment_variable": florence_config.adapter_path_env,
            "config_value": florence_config.adapter_path,
            "default_value": DEFAULT_FLORENCE_ADAPTER_PATH,
            "expect_directory": True,
            "required_files": ["adapter_config.json", "adapter_model.safetensors"],
        },
    ]

    resolved_for_load: dict[str, Path] = {}
    for check in checks:
        entry: dict[str, object] = {"name": check["name"]}
        try:
            resolution = resolve_model_path_with_source(
                cli_value=check["cli_value"],
                environment_variable=check["environment_variable"],
                config_value=check["config_value"],
                default_value=check["default_value"],
                project_root=PROJECT_ROOT,
                required=True,
                expect_directory=check["expect_directory"],
            )
            entry["resolved_from"] = resolution.source
            entry["path"] = str(resolution.path)
            entry["exists"] = bool(resolution.path and resolution.path.exists())
            entry["kind"] = "directory" if resolution.path and resolution.path.is_dir() else "file"
            entry["size_bytes"] = _path_size(resolution.path) if resolution.path is not None else None
            entry["required_supporting_files"] = _supporting_file_report(resolution.path, check["required_files"])
            entry["ready"] = _supporting_files_ready(entry["required_supporting_files"])
            if resolution.path is not None:
                resolved_for_load[str(check["name"])] = resolution.path
            if not bool(entry["ready"]):
                required_ok = False
        except ModelPathResolutionError as exc:
            entry["resolved_from"] = None
            entry["path"] = None
            entry["exists"] = False
            entry["kind"] = None
            entry["size_bytes"] = None
            entry["required_supporting_files"] = []
            entry["ready"] = False
            entry["error"] = str(exc)
            required_ok = False
        models = report["models"]
        assert isinstance(models, list)
        models.append(entry)

    if args.load_check and required_ok:
        report["load_check"] = run_load_check(resolved_for_load)
        if not bool(report["load_check"]["ready"]):  # type: ignore[index]
            required_ok = False

    print(json.dumps(report, indent=2))
    raise SystemExit(0 if required_ok else 1)


def run_load_check(resolved_paths: dict[str, Path]) -> dict[str, object]:
    result: dict[str, object] = {"ready": False}
    try:
        cache_path = _ensure_writable_huggingface_modules_cache()
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoProcessor
        from ultralytics import YOLO

        vehicle_model = YOLO(str(resolved_paths["vehicle_detector"]))
        plate_model = YOLO(str(resolved_paths["plate_detector"]))
        processor = AutoProcessor.from_pretrained(
            str(resolved_paths["florence_processor"]),
            trust_remote_code=True,
            local_files_only=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            str(resolved_paths["florence_base"]),
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="eager",
        )
        adapted_model = PeftModel.from_pretrained(
            base_model,
            str(resolved_paths["florence_adapter"]),
            local_files_only=True,
        )
        result.update(
            {
                "ready": True,
                "vehicle_detector_loaded": vehicle_model is not None,
                "plate_detector_loaded": plate_model is not None,
                "florence_processor_loaded": processor is not None,
                "florence_base_loaded": base_model is not None,
                "florence_adapter_applied": adapted_model is not None,
                "hf_modules_cache": str(cache_path),
            }
        )
    except Exception as exc:  # pragma: no cover
        result["error"] = str(exc)
    return result


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _supporting_file_report(path: Path | None, required_files: list[str]) -> list[dict[str, object]]:
    if path is None:
        return []
    report: list[dict[str, object]] = []
    for relative_name in required_files:
        candidate = path / relative_name
        report.append({"name": relative_name, "exists": candidate.exists(), "path": str(candidate)})
    return report


def _supporting_files_ready(items: object) -> bool:
    if not isinstance(items, list):
        return False
    return all(bool(item.get("exists")) for item in items if isinstance(item, dict))


if __name__ == "__main__":
    main()
