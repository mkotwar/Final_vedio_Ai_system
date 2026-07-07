from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import (
    DEFAULT_FLORENCE_MAX_NEW_TOKENS,
    DEFAULT_FLORENCE_NUM_BEAMS,
    DEFAULT_STEP06_DEVICE,
    DEFAULT_STEP06_FALLBACK_LIMIT,
    DEFAULT_STEP06_MAX_NEW_TOKENS,
    DEFAULT_STEP06_MIN_PLATE_CONFIDENCE,
    DEFAULT_STEP06_NUM_BEAMS,
    DEFAULT_STEP06_PRIMARY_LIMIT,
    DEFAULT_STEP06_PROCESS_GROUPS,
    DEFAULT_STEP06_REUSE_EXISTING_RAW_RESULTS,
    DEFAULT_STEP06_RUN_CLEANING_TESTS,
    DEFAULT_STEP06_SAVE_DEBUG_IMAGES,
    DEFAULT_STEP06_SAVE_PLATE_CROPS,
    ENV_FLORENCE_ADAPTER_PATH,
    ENV_FLORENCE_MODEL_PATH,
    ENV_PLATE_DETECTOR_MODEL_PATH,
    ENV_RUN_DIR,
    ENV_STEP06_DEVICE,
    ENV_STEP06_FALLBACK_LIMIT,
    ENV_STEP06_MAX_NEW_TOKENS,
    ENV_STEP06_MIN_PLATE_CONFIDENCE,
    ENV_STEP06_NUM_BEAMS,
    ENV_STEP06_PRIMARY_LIMIT,
    ENV_STEP06_PROCESS_GROUPS,
    ENV_STEP06_REUSE_EXISTING_RAW_RESULTS,
    ENV_STEP06_RUN_CLEANING_TESTS,
    ENV_STEP06_SAVE_DEBUG_IMAGES,
    ENV_STEP06_SAVE_PLATE_CROPS,
)
from run_td_case2_step01_02 import log
from stage_checks import build_failure_payload, update_stage_gate_report, write_json
from step_06_ocr_color_enrichment import (
    SUPPORTED_PROCESS_GROUPS,
    run_ocr_color_enrichment,
    test_clean_plate_text_examples,
    test_verified_plate_classification_examples,
    verify_existing_step06_outputs,
)


SUPPORTED_DEVICE_VALUES = {"auto", "cpu", "cuda"}
DEFAULT_PLATE_DETECTOR_MODEL_PATH = Path(r"C:\Mukul K\vinfo1\video-search-engine\ocr_colour\license_plate_weights.pt")


@dataclass(frozen=True)
class Step06Config:
    """Runtime settings for isolated Step 06 OCR/color enrichment."""

    run_dir: Path
    florence_model_path: Path
    florence_adapter_path: Path | None
    plate_detector_model_path: Path | None
    process_groups: list[str]
    primary_limit: int
    fallback_limit: int
    device: str
    max_new_tokens: int
    num_beams: int
    save_plate_crops: bool
    save_debug_images: bool
    min_plate_confidence: float
    run_cleaning_tests: bool
    reuse_existing_raw_results: bool


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


def _read_non_negative_int(env_name: str, default_value: int) -> int:
    """Read a non-negative integer from the environment."""

    raw_value = os.environ.get(env_name, str(default_value)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {env_name} must be a valid integer. Received: {raw_value!r}") from exc
    if value < 0:
        raise ValueError(f"Environment variable {env_name} must be 0 or greater. Received: {value}")
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


def _read_non_negative_float(env_name: str, default_value: float) -> float:
    """Read a non-negative float from the environment."""

    raw_value = os.environ.get(env_name, str(default_value)).strip()
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {env_name} must be a valid number. Received: {raw_value!r}") from exc
    if value < 0:
        raise ValueError(f"Environment variable {env_name} must be 0 or greater. Received: {value}")
    return value


def _resolve_optional_path(raw_value: str | None) -> Path | None:
    """Resolve an optional path from the environment."""

    if not raw_value or not raw_value.strip():
        return None
    candidate = Path(raw_value.strip()).expanduser()
    if not candidate.is_absolute():
        candidate = candidate.resolve()
    return candidate


def _read_process_groups() -> list[str]:
    """Read and validate the Step 06 process groups."""

    raw_value = os.environ.get(ENV_STEP06_PROCESS_GROUPS, DEFAULT_STEP06_PROCESS_GROUPS)
    groups = [item.strip().lower() for item in raw_value.split(",") if item.strip()]
    if not groups:
        raise ValueError(f"Environment variable {ENV_STEP06_PROCESS_GROUPS} must include at least one process group.")
    invalid = [item for item in groups if item not in SUPPORTED_PROCESS_GROUPS]
    if invalid:
        raise ValueError(
            f"Environment variable {ENV_STEP06_PROCESS_GROUPS} contains unsupported values: {invalid}. "
            f"Supported values: {sorted(SUPPORTED_PROCESS_GROUPS)}"
        )
    return groups


def read_config() -> Step06Config:
    """Read configuration for isolated Step 06 OCR/color enrichment."""

    raw_run_dir = os.environ.get(ENV_RUN_DIR, "").strip()
    if not raw_run_dir:
        raise ValueError(f"Environment variable {ENV_RUN_DIR} is required for Step 06.")
    run_dir = Path(raw_run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"TD_CASE2_RUN_DIR does not point to an existing directory: {run_dir}")

    required_inputs = [
        run_dir / "05_best_track_frames.json",
        run_dir / "05_best_track_frames_report.json",
        run_dir / "05_selected_track_crops",
    ]
    for required_path in required_inputs:
        if not required_path.exists():
            raise FileNotFoundError(f"Required Step 06 input is missing: {required_path}")

    florence_model_path: Path
    raw_model_path = os.environ.get(ENV_FLORENCE_MODEL_PATH, "").strip()
    if raw_model_path:
        florence_model_path = Path(raw_model_path).expanduser()
        if not florence_model_path.is_absolute():
            florence_model_path = florence_model_path.resolve()
    else:
        florence_model_path = Path(".")

    florence_adapter_path = _resolve_optional_path(os.environ.get(ENV_FLORENCE_ADAPTER_PATH))
    plate_detector_model_path = _resolve_optional_path(os.environ.get(ENV_PLATE_DETECTOR_MODEL_PATH))
    if plate_detector_model_path is None and DEFAULT_PLATE_DETECTOR_MODEL_PATH.exists():
        plate_detector_model_path = DEFAULT_PLATE_DETECTOR_MODEL_PATH

    device = os.environ.get(ENV_STEP06_DEVICE, DEFAULT_STEP06_DEVICE).strip().lower() or DEFAULT_STEP06_DEVICE
    if device not in SUPPORTED_DEVICE_VALUES:
        raise ValueError(
            f"Environment variable {ENV_STEP06_DEVICE} must be one of {sorted(SUPPORTED_DEVICE_VALUES)}. "
            f"Received: {device!r}"
        )

    return Step06Config(
        run_dir=run_dir.resolve(),
        florence_model_path=florence_model_path.resolve(),
        florence_adapter_path=florence_adapter_path.resolve() if florence_adapter_path is not None else None,
        plate_detector_model_path=plate_detector_model_path.resolve() if plate_detector_model_path is not None else None,
        process_groups=_read_process_groups(),
        primary_limit=_read_non_negative_int(ENV_STEP06_PRIMARY_LIMIT, DEFAULT_STEP06_PRIMARY_LIMIT),
        fallback_limit=_read_non_negative_int(ENV_STEP06_FALLBACK_LIMIT, DEFAULT_STEP06_FALLBACK_LIMIT),
        device=device,
        max_new_tokens=_read_positive_int(ENV_STEP06_MAX_NEW_TOKENS, DEFAULT_STEP06_MAX_NEW_TOKENS or DEFAULT_FLORENCE_MAX_NEW_TOKENS),
        num_beams=_read_positive_int(ENV_STEP06_NUM_BEAMS, DEFAULT_STEP06_NUM_BEAMS or DEFAULT_FLORENCE_NUM_BEAMS),
        save_plate_crops=_read_bool(ENV_STEP06_SAVE_PLATE_CROPS, DEFAULT_STEP06_SAVE_PLATE_CROPS),
        save_debug_images=_read_bool(ENV_STEP06_SAVE_DEBUG_IMAGES, DEFAULT_STEP06_SAVE_DEBUG_IMAGES),
        min_plate_confidence=_read_non_negative_float(ENV_STEP06_MIN_PLATE_CONFIDENCE, DEFAULT_STEP06_MIN_PLATE_CONFIDENCE),
        run_cleaning_tests=_read_bool(ENV_STEP06_RUN_CLEANING_TESTS, DEFAULT_STEP06_RUN_CLEANING_TESTS),
        reuse_existing_raw_results=_read_bool(
            ENV_STEP06_REUSE_EXISTING_RAW_RESULTS,
            DEFAULT_STEP06_REUSE_EXISTING_RAW_RESULTS,
        ),
    )


def _write_failed_reports(run_dir: Path, error_message: str) -> None:
    """Write Step 06 failure JSON artifacts."""

    results_payload = {
        "status": "failed",
        "input_best_frames_file": "05_best_track_frames.json",
        "selected_crop_count": 0,
        "processed_crop_count": 0,
        "successful_crop_count": 0,
        "failed_crop_count": 0,
        "track_results": [],
        "error_message": error_message,
    }
    report_payload = {
        "status": "failed",
        "selected_crop_count": 0,
        "processed_crop_count": 0,
        "primary_processed_count": 0,
        "fallback_processed_count": 0,
        "successful_crop_count": 0,
        "failed_crop_count": 0,
        "plate_detector_available": False,
        "plate_crops_found": 0,
        "tracks_with_plate_text": 0,
        "tracks_with_vehicle_color": 0,
        "primary_tracks_with_plate_text": 0,
        "fallback_tracks_with_plate_text": 0,
        "primary_tracks_with_color": 0,
        "fallback_tracks_with_color": 0,
        "colour_counts": {},
        "top_plate_results": [],
        "failed_examples": [],
        "avg_processing_seconds_per_crop": 0.0,
        "recommendation": error_message,
        "error_message": error_message,
    }
    write_json(run_dir / "06_ocr_color_results.json", results_payload)
    write_json(run_dir / "06_ocr_color_report.json", report_payload)


def main() -> None:
    """Run isolated Step 06 OCR and colour enrichment."""

    config = read_config()
    log(f"Run directory: {config.run_dir}")
    log("Input selected crops file: 05_best_track_frames.json")
    log(f"reuse_existing_raw_results: {str(config.reuse_existing_raw_results).lower()}")

    if config.run_cleaning_tests:
        warnings = test_clean_plate_text_examples()
        for warning in warnings:
            log(f"Cleaning test warning: {warning}")
        verification_warnings = test_verified_plate_classification_examples()
        for warning in verification_warnings:
            log(f"Verification test warning: {warning}")

    try:
        cleaned_source_path = config.run_dir / "06_ocr_color_results_cleaned.json"
        raw_source_path = config.run_dir / "06_ocr_color_results.json"
        if config.reuse_existing_raw_results or cleaned_source_path.exists() or raw_source_path.exists():
            source_results_path = cleaned_source_path if cleaned_source_path.exists() else raw_source_path
            log(f"Source cleaned results file: {source_results_path}")
            results_payload, report_payload, results_path, report_path = verify_existing_step06_outputs(config.run_dir)
        else:
            if str(config.florence_model_path) == ".":
                raise ValueError(
                    f"Environment variable {ENV_FLORENCE_MODEL_PATH} is required when rerunning Step 06 from images."
                )
            results_payload, report_payload, results_path, report_path = run_ocr_color_enrichment(
                run_dir=config.run_dir,
                florence_model_path=config.florence_model_path,
                florence_adapter_path=config.florence_adapter_path,
                plate_detector_model_path=config.plate_detector_model_path,
                process_groups=config.process_groups,
                primary_limit=config.primary_limit,
                fallback_limit=config.fallback_limit,
                device=config.device,
                max_new_tokens=config.max_new_tokens,
                num_beams=config.num_beams,
                save_plate_crops=config.save_plate_crops,
                save_debug_images=config.save_debug_images,
                min_plate_confidence=config.min_plate_confidence,
                reuse_existing_raw_results=config.reuse_existing_raw_results,
            )
        update_stage_gate_report(
            config.run_dir,
            "06_ocr_color_enrichment",
            {
                "status": results_payload["status"],
                "processed_crop_count": report_payload["processed_crop_count"],
                "primary_processed_count": report_payload["primary_processed_count"],
                "fallback_processed_count": report_payload["fallback_processed_count"],
                "plate_crops_found": report_payload["plate_crops_found"],
                "tracks_with_raw_plate_text": report_payload["tracks_with_raw_plate_text"],
                "tracks_with_valid_plate_text": report_payload["tracks_with_valid_plate_text"],
                "tracks_with_verified_license_plate": report_payload["tracks_with_verified_license_plate"],
                "verified_unique_license_plate_count": report_payload["verified_unique_license_plate_count"],
                "tracks_with_vehicle_color": report_payload["tracks_with_vehicle_color"],
                "ready_for_search_index_enrichment": report_payload["processed_crop_count"] > 0,
            },
        )
        update_stage_gate_report(
            config.run_dir,
            "06_plate_verification",
            {
                "status": results_payload["status"],
                "verified_unique_license_plate_count": report_payload["verified_unique_license_plate_count"],
                "tracks_with_verified_license_plate": report_payload["tracks_with_verified_license_plate"],
                "possible_unique_license_plate_count": report_payload["possible_unique_license_plate_count"],
                "ready_for_step07_search_index": report_payload["processed_crop_count"] > 0,
            },
        )
        log(f"Tracks processed: {len(results_payload.get('track_results', []))}")
        log(f"Verified plate tracks: {report_payload['tracks_with_verified_license_plate']}")
        log(f"Verified unique plates: {report_payload['verified_unique_license_plate_count']}")
        log(f"Possible plate tracks: {report_payload['tracks_with_possible_license_plate_only']}")
        log(f"Possible unique plates: {report_payload['possible_unique_license_plate_count']}")
        log(f"Output verified results path: {results_path}")
        log(f"Output verified report path: {report_path}")
    except Exception as exc:
        _write_failed_reports(config.run_dir, str(exc))
        update_stage_gate_report(config.run_dir, "06_ocr_color_enrichment", build_failure_payload(exc))
        log(f"Step 06 failed: {exc}")
        log(f"Run directory: {config.run_dir}")
        raise


if __name__ == "__main__":
    main()
