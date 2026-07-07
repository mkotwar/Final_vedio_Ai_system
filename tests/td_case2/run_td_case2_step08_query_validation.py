from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import (
    DEFAULT_STEP08_ALLOW_WEAK_OCR_SEARCH,
    DEFAULT_STEP08_FAIL_ON_CRITICAL_TEST_FAILURE,
    DEFAULT_STEP08_SAVE_MATCH_PREVIEW,
    DEFAULT_STEP08_TIME_TOLERANCE_SECONDS,
    DEFAULT_STEP08_TOP_K,
    ENV_RUN_DIR,
    ENV_STEP08_ALLOW_WEAK_OCR_SEARCH,
    ENV_STEP08_FAIL_ON_CRITICAL_TEST_FAILURE,
    ENV_STEP08_SAVE_MATCH_PREVIEW,
    ENV_STEP08_TIME_TOLERANCE_SECONDS,
    ENV_STEP08_TOP_K,
)
from run_td_case2_step01_02 import log
from stage_checks import build_failure_payload, update_stage_gate_report, write_json
from step_08_query_search_validation import run_query_search_validation


@dataclass(frozen=True)
class Step08Config:
    """Runtime settings for isolated Step 08 query validation."""

    run_dir: Path
    validation_config: dict[str, Any]


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


def read_config() -> Step08Config:
    """Read configuration for isolated Step 08 query validation."""

    raw_run_dir = os.environ.get(ENV_RUN_DIR, "").strip()
    if not raw_run_dir:
        raise ValueError(f"Environment variable {ENV_RUN_DIR} is required for Step 08.")
    run_dir = Path(raw_run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"TD_CASE2_RUN_DIR does not point to an existing directory: {run_dir}")

    required_inputs = [
        run_dir / "07_vehicle_search_index.json",
        run_dir / "07_vehicle_search_index_flat.json",
        run_dir / "07_vehicle_search_index_report.json",
    ]
    for required_path in required_inputs:
        if not required_path.exists():
            raise FileNotFoundError(f"Required Step 08 input is missing: {required_path}")

    return Step08Config(
        run_dir=run_dir.resolve(),
        validation_config={
            "allow_weak_ocr_search": _read_bool(ENV_STEP08_ALLOW_WEAK_OCR_SEARCH, DEFAULT_STEP08_ALLOW_WEAK_OCR_SEARCH),
            "time_tolerance_seconds": _read_positive_float(ENV_STEP08_TIME_TOLERANCE_SECONDS, DEFAULT_STEP08_TIME_TOLERANCE_SECONDS),
            "top_k": _read_positive_int(ENV_STEP08_TOP_K, DEFAULT_STEP08_TOP_K),
            "fail_on_critical_test_failure": _read_bool(
                ENV_STEP08_FAIL_ON_CRITICAL_TEST_FAILURE,
                DEFAULT_STEP08_FAIL_ON_CRITICAL_TEST_FAILURE,
            ),
            "save_match_preview": _read_bool(ENV_STEP08_SAVE_MATCH_PREVIEW, DEFAULT_STEP08_SAVE_MATCH_PREVIEW),
        },
    )


def _write_failed_reports(run_dir: Path, validation_config: dict[str, Any], error_message: str) -> None:
    """Write Step 08 failure JSON artifacts."""

    results_payload = {
        "status": "failed",
        "source_index_file": "07_vehicle_search_index.json",
        "test_config": validation_config,
        "validation_summary": {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "warning_tests": 0,
            "pass_rate": 0.0,
        },
        "test_results": [],
        "error_message": error_message,
    }
    matches_payload = {"status": "failed", "queries": [], "error_message": error_message}
    report_payload = {
        "status": "failed",
        "total_records_loaded": 0,
        "records_with_verified_plate": 0,
        "unique_verified_plate_count": 0,
        "records_with_color": 0,
        "records_with_full_frame": 0,
        "total_tests": 0,
        "passed_tests": 0,
        "failed_tests": 0,
        "pass_rate": 0.0,
        "exact_plate_tests": {"passed": 0, "failed": 0},
        "color_class_tests": {"passed": 0, "failed": 0},
        "timestamp_tests": {"passed": 0, "failed": 0},
        "combined_tests": {"passed": 0, "failed": 0},
        "weak_ocr_tests": {"passed": 0, "failed": 0},
        "invalid_ocr_blocking_tests": {"passed": 0, "failed": 0},
        "path_availability_tests": {"passed": 0, "failed": 0},
        "example_passed_queries": [],
        "failed_tests_list": [],
        "recommendation": error_message,
        "error_message": error_message,
    }
    write_json(run_dir / "08_query_validation_results.json", results_payload)
    write_json(run_dir / "08_query_validation_matches.json", matches_payload)
    write_json(run_dir / "08_query_validation_report.json", report_payload)


def main() -> None:
    """Run isolated Step 08 query validation."""

    config = read_config()
    log(f"Run directory: {config.run_dir}")
    log("Loaded search index: 07_vehicle_search_index.json")

    try:
        results_payload, _matches_payload, report_payload = run_query_search_validation(
            run_dir=config.run_dir,
            validation_config=config.validation_config,
        )
        update_stage_gate_report(
            config.run_dir,
            "08_query_search_validation",
            {
                "status": results_payload["status"],
                "total_tests": report_payload["total_tests"],
                "passed_tests": report_payload["passed_tests"],
                "failed_tests": report_payload["failed_tests"],
                "pass_rate": report_payload["pass_rate"],
                "exact_plate_search_ready": report_payload["exact_plate_tests"]["failed"] == 0,
                "color_class_search_ready": report_payload["color_class_tests"]["failed"] == 0,
                "timestamp_search_ready": report_payload["timestamp_tests"]["failed"] == 0,
                "weak_ocr_search_ready": report_payload["weak_ocr_tests"]["failed"] == 0,
                "invalid_ocr_blocking_ready": report_payload["invalid_ocr_blocking_tests"]["failed"] == 0,
                "result_paths_ready": report_payload["path_availability_tests"]["failed"] == 0,
                "ready_for_step09_result_packaging": (
                    report_payload["exact_plate_tests"]["failed"] == 0
                    and report_payload["color_class_tests"]["failed"] == 0
                    and report_payload["invalid_ocr_blocking_tests"]["failed"] == 0
                    and report_payload["path_availability_tests"]["failed"] == 0
                ),
            },
        )
        log(f"Records loaded: {report_payload['total_records_loaded']}")
        log(f"Total tests: {report_payload['total_tests']}")
        log(f"Passed tests: {report_payload['passed_tests']}")
        log(f"Failed tests: {report_payload['failed_tests']}")
        log(f"Exact plate status: {report_payload['exact_plate_tests']}")
        log(f"Color/class status: {report_payload['color_class_tests']}")
        log(f"Invalid OCR blocking status: {report_payload['invalid_ocr_blocking_tests']}")
        log(f"Full-frame path status: {report_payload['path_availability_tests']}")
        log(
            "Output files: "
            f"{config.run_dir / '08_query_validation_results.json'} | "
            f"{config.run_dir / '08_query_validation_matches.json'} | "
            f"{config.run_dir / '08_query_validation_report.json'}"
        )
    except Exception as exc:
        _write_failed_reports(config.run_dir, config.validation_config, str(exc))
        update_stage_gate_report(config.run_dir, "08_query_search_validation", build_failure_payload(exc))
        log(f"Step 08 failed: {exc}")
        log(f"Run directory: {config.run_dir}")
        raise


if __name__ == "__main__":
    main()
