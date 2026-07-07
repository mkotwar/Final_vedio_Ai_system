from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import (
    DEFAULT_STEP09_BUILD_DEMO_QUERIES,
    DEFAULT_STEP09_INCLUDE_DEBUG_PATHS,
    DEFAULT_STEP09_INCLUDE_INVALID_DEBUG_FIELDS,
    DEFAULT_STEP09_INCLUDE_WEAK_OCR,
    DEFAULT_STEP09_RESULT_CARD_VERSION,
    DEFAULT_STEP09_TOP_K,
    DEFAULT_STEP09_VALIDATE_PATH_STRINGS,
    ENV_RUN_DIR,
    ENV_STEP09_BUILD_DEMO_QUERIES,
    ENV_STEP09_INCLUDE_DEBUG_PATHS,
    ENV_STEP09_INCLUDE_INVALID_DEBUG_FIELDS,
    ENV_STEP09_INCLUDE_WEAK_OCR,
    ENV_STEP09_RESULT_CARD_VERSION,
    ENV_STEP09_TOP_K,
    ENV_STEP09_VALIDATE_PATH_STRINGS,
)
from run_td_case2_step01_02 import log
from stage_checks import build_failure_payload, update_stage_gate_report, write_json
from step_09_search_result_packaging import write_json_any, run_search_result_packaging


@dataclass(frozen=True)
class Step09Config:
    """Runtime settings for isolated Step 09 result packaging."""

    run_dir: Path
    packaging_config: dict[str, Any]


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


def read_config() -> Step09Config:
    """Read configuration for isolated Step 09 result packaging."""

    raw_run_dir = os.environ.get(ENV_RUN_DIR, "").strip()
    if not raw_run_dir:
        raise ValueError(f"Environment variable {ENV_RUN_DIR} is required for Step 09.")
    run_dir = Path(raw_run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"TD_CASE2_RUN_DIR does not point to an existing directory: {run_dir}")

    required_inputs = [
        run_dir / "07_vehicle_search_index.json",
        run_dir / "07_vehicle_search_index_flat.json",
        run_dir / "07_vehicle_search_index_report.json",
        run_dir / "08_query_validation_results.json",
        run_dir / "08_query_validation_report.json",
    ]
    for required_path in required_inputs:
        if not required_path.exists():
            raise FileNotFoundError(f"Required Step 09 input is missing: {required_path}")

    return Step09Config(
        run_dir=run_dir.resolve(),
        packaging_config={
            "top_k": _read_positive_int(ENV_STEP09_TOP_K, DEFAULT_STEP09_TOP_K),
            "include_weak_ocr": _read_bool(ENV_STEP09_INCLUDE_WEAK_OCR, DEFAULT_STEP09_INCLUDE_WEAK_OCR),
            "include_invalid_debug_fields": _read_bool(
                ENV_STEP09_INCLUDE_INVALID_DEBUG_FIELDS,
                DEFAULT_STEP09_INCLUDE_INVALID_DEBUG_FIELDS,
            ),
            "include_debug_paths": _read_bool(ENV_STEP09_INCLUDE_DEBUG_PATHS, DEFAULT_STEP09_INCLUDE_DEBUG_PATHS),
            "validate_path_strings": _read_bool(
                ENV_STEP09_VALIDATE_PATH_STRINGS,
                DEFAULT_STEP09_VALIDATE_PATH_STRINGS,
            ),
            "build_demo_queries": _read_bool(ENV_STEP09_BUILD_DEMO_QUERIES, DEFAULT_STEP09_BUILD_DEMO_QUERIES),
            "result_card_version": os.environ.get(ENV_STEP09_RESULT_CARD_VERSION, DEFAULT_STEP09_RESULT_CARD_VERSION).strip()
            or DEFAULT_STEP09_RESULT_CARD_VERSION,
        },
    )


def _write_failed_reports(run_dir: Path, packaging_config: dict[str, Any], error_message: str) -> None:
    """Write Step 09 failure JSON artifacts."""

    cards_payload = {
        "status": "failed",
        "schema_version": str(packaging_config.get("result_card_version", "v1")),
        "source_index_file": "07_vehicle_search_index.json",
        "source_validation_file": "08_query_validation_matches.json",
        "summary": {
            "total_query_packages": 0,
            "total_cards_created": 0,
            "cards_with_verified_plate": 0,
            "cards_with_weak_ocr": 0,
            "cards_with_full_frame": 0,
            "cards_with_crop": 0,
            "high_confidence_cards": 0,
            "medium_confidence_cards": 0,
            "low_confidence_cards": 0,
        },
        "cards": [],
        "error_message": error_message,
    }
    report_payload = {
        "status": "failed",
        "source_records_loaded": 0,
        "source_validation_queries_loaded": 0,
        "total_query_packages": 0,
        "total_cards_created": 0,
        "unique_tracks_packaged": 0,
        "cards_with_verified_plate": 0,
        "cards_with_weak_ocr": 0,
        "cards_with_full_frame": 0,
        "cards_with_crop": 0,
        "confidence_badge_counts": {"high": 0, "medium": 0, "low": 0},
        "result_badge_counts": {"verified_plate": 0, "weak_ocr": 0, "color_class": 0, "basic_track": 0},
        "path_availability": {
            "cards_with_crop": 0,
            "cards_missing_crop": 0,
            "cards_with_full_frame": 0,
            "cards_missing_full_frame": 0,
            "cards_with_contact_sheet": 0,
            "cards_missing_contact_sheet": 0,
        },
        "example_cards": [],
        "warnings": [],
        "recommendation": error_message,
        "error_message": error_message,
    }
    schema_payload = {
        "schema_name": "td_case2_search_result_card",
        "schema_version": str(packaging_config.get("result_card_version", "v1")),
        "field_definitions": {},
        "example_card": {},
        "error_message": error_message,
    }
    query_packages_payload = {
        "status": "failed",
        "schema_version": str(packaging_config.get("result_card_version", "v1")),
        "top_k": int(packaging_config.get("top_k", 0) or 0),
        "query_packages": [],
        "error_message": error_message,
    }
    write_json(run_dir / "09_search_result_cards.json", cards_payload)
    write_json_any(run_dir / "09_search_result_cards_flat.json", [])
    write_json(run_dir / "09_demo_query_result_packages.json", query_packages_payload)
    write_json(run_dir / "09_search_result_card_schema.json", schema_payload)
    write_json(run_dir / "09_search_result_packaging_report.json", report_payload)


def main() -> None:
    """Run isolated Step 09 search result packaging."""

    config = read_config()
    log(f"Run directory: {config.run_dir}")
    log("Source files loaded: 07_vehicle_search_index.json | 08_query_validation_results.json | 08_query_validation_report.json")

    try:
        cards_payload, _flat_cards, query_packages_payload, _schema_payload, report_payload = run_search_result_packaging(
            run_dir=config.run_dir,
            packaging_config=config.packaging_config,
        )
        update_stage_gate_report(
            config.run_dir,
            "09_search_result_packaging",
            {
                "status": cards_payload["status"],
                "total_query_packages": report_payload["total_query_packages"],
                "total_cards_created": report_payload["total_cards_created"],
                "cards_with_verified_plate": report_payload["cards_with_verified_plate"],
                "cards_with_full_frame": report_payload["cards_with_full_frame"],
                "cards_with_crop": report_payload["cards_with_crop"],
                "high_confidence_cards": report_payload["confidence_badge_counts"]["high"],
                "ready_for_step10_search_demo_runner": (
                    cards_payload["status"] in {"success", "partial_success"}
                    and report_payload["cards_with_full_frame"] > 0
                    and report_payload["cards_with_crop"] > 0
                ),
            },
        )
        log(f"Query packages created: {report_payload['total_query_packages']}")
        log(f"Cards created: {report_payload['total_cards_created']}")
        log(f"Cards with verified plate: {report_payload['cards_with_verified_plate']}")
        log(f"Cards with full frame: {report_payload['cards_with_full_frame']}")
        log(f"Cards with crop: {report_payload['cards_with_crop']}")
        log(
            "Output paths: "
            f"{config.run_dir / '09_search_result_cards.json'} | "
            f"{config.run_dir / '09_search_result_cards_flat.json'} | "
            f"{config.run_dir / '09_demo_query_result_packages.json'} | "
            f"{config.run_dir / '09_search_result_card_schema.json'} | "
            f"{config.run_dir / '09_search_result_packaging_report.json'}"
        )
    except Exception as exc:
        _write_failed_reports(config.run_dir, config.packaging_config, str(exc))
        update_stage_gate_report(config.run_dir, "09_search_result_packaging", build_failure_payload(exc))
        log(f"Step 09 failed: {exc}")
        log(f"Run directory: {config.run_dir}")
        raise


if __name__ == "__main__":
    main()
