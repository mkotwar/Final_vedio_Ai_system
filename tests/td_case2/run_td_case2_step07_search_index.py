from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import (
    DEFAULT_STEP07_INCLUDE_FALLBACK,
    DEFAULT_STEP07_INCLUDE_POSSIBLE_OCR,
    DEFAULT_STEP07_MIN_CONFIDENCE_FOR_SEARCH,
    DEFAULT_STEP07_REQUIRE_COLOR_FOR_COLOR_INDEX,
    DEFAULT_STEP07_REQUIRE_VERIFIED_PLATE_FOR_PLATE_INDEX,
    DEFAULT_STEP07_SAVE_DEBUG_SUMMARY,
    DEFAULT_STEP07_SAVE_FLAT_INDEX,
    ENV_RUN_DIR,
    ENV_STEP07_INCLUDE_FALLBACK,
    ENV_STEP07_INCLUDE_POSSIBLE_OCR,
    ENV_STEP07_MIN_CONFIDENCE_FOR_SEARCH,
    ENV_STEP07_REQUIRE_COLOR_FOR_COLOR_INDEX,
    ENV_STEP07_REQUIRE_VERIFIED_PLATE_FOR_PLATE_INDEX,
    ENV_STEP07_SAVE_DEBUG_SUMMARY,
    ENV_STEP07_SAVE_FLAT_INDEX,
)
from run_td_case2_step01_02 import log
from stage_checks import build_failure_payload, update_stage_gate_report, write_json
from step_07_search_index_enrichment import run_search_index_enrichment


@dataclass(frozen=True)
class Step07Config:
    """Runtime settings for isolated Step 07 search index enrichment."""

    run_dir: Path
    index_config: dict[str, Any]


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


def read_config() -> Step07Config:
    """Read configuration for isolated Step 07 search index enrichment."""

    raw_run_dir = os.environ.get(ENV_RUN_DIR, "").strip()
    if not raw_run_dir:
        raise ValueError(f"Environment variable {ENV_RUN_DIR} is required for Step 07.")
    run_dir = Path(raw_run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"TD_CASE2_RUN_DIR does not point to an existing directory: {run_dir}")

    required_inputs = [
        run_dir / "06_ocr_color_results_verified.json",
        run_dir / "06_ocr_color_report_verified.json",
        run_dir / "05_best_track_frames.json",
        run_dir / "04B_tracks.json",
    ]
    for required_path in required_inputs:
        if not required_path.exists():
            raise FileNotFoundError(f"Required Step 07 input is missing: {required_path}")

    return Step07Config(
        run_dir=run_dir.resolve(),
        index_config={
            "include_fallback": _read_bool(ENV_STEP07_INCLUDE_FALLBACK, DEFAULT_STEP07_INCLUDE_FALLBACK),
            "include_possible_ocr": _read_bool(ENV_STEP07_INCLUDE_POSSIBLE_OCR, DEFAULT_STEP07_INCLUDE_POSSIBLE_OCR),
            "min_confidence_for_search": _read_non_negative_float(
                ENV_STEP07_MIN_CONFIDENCE_FOR_SEARCH,
                DEFAULT_STEP07_MIN_CONFIDENCE_FOR_SEARCH,
            ),
            "require_color_for_color_index": _read_bool(
                ENV_STEP07_REQUIRE_COLOR_FOR_COLOR_INDEX,
                DEFAULT_STEP07_REQUIRE_COLOR_FOR_COLOR_INDEX,
            ),
            "require_verified_plate_for_plate_index": _read_bool(
                ENV_STEP07_REQUIRE_VERIFIED_PLATE_FOR_PLATE_INDEX,
                DEFAULT_STEP07_REQUIRE_VERIFIED_PLATE_FOR_PLATE_INDEX,
            ),
            "save_flat_index": _read_bool(ENV_STEP07_SAVE_FLAT_INDEX, DEFAULT_STEP07_SAVE_FLAT_INDEX),
            "save_debug_summary": _read_bool(ENV_STEP07_SAVE_DEBUG_SUMMARY, DEFAULT_STEP07_SAVE_DEBUG_SUMMARY),
        },
    )


def _write_failed_reports(run_dir: Path, index_config: dict[str, Any], error_message: str) -> None:
    """Write Step 07 failure JSON artifacts."""

    index_payload = {
        "status": "failed",
        "source_results_file": "06_ocr_color_results_verified.json",
        "source_report_file": "06_ocr_color_report_verified.json",
        "index_config": index_config,
        "summary": {
            "total_vehicle_records": 0,
            "searchable_records": 0,
            "primary_records": 0,
            "fallback_records": 0,
            "records_with_verified_plate": 0,
            "unique_verified_plate_count": 0,
            "records_with_color": 0,
            "records_with_possible_ocr": 0,
            "records_with_crop": 0,
            "records_with_full_frame": 0,
        },
        "records": [],
        "error_message": error_message,
    }
    report_payload = {
        "status": "failed",
        "total_vehicle_records": 0,
        "searchable_records": 0,
        "primary_records": 0,
        "fallback_records": 0,
        "records_with_verified_plate": 0,
        "unique_verified_plate_count": 0,
        "verified_license_plates": [],
        "records_with_color": 0,
        "records_with_full_frame": 0,
        "records_missing_full_frame": 0,
        "color_counts": {},
        "vehicle_class_counts": {},
        "quality_group_counts": {},
        "search_confidence_stats": {"min": 0.0, "max": 0.0, "avg": 0.0},
        "invalid_ocr_terms_removed_from_search_text_count": 0,
        "invalid_ocr_terms_removed_examples": [],
        "search_text_policy": {
            "verified_plate": "included_for_exact_search",
            "possible_ocr": "included_for_weak_search",
            "invalid_ocr": "excluded_from_search_text",
        },
        "example_exact_plate_searches": [],
        "example_color_searches": [],
        "recommendation": error_message,
        "error_message": error_message,
    }
    write_json(run_dir / "07_vehicle_search_index.json", index_payload)
    write_json(run_dir / "07_vehicle_search_index_report.json", report_payload)
    if bool(index_config.get("save_flat_index", True)):
        (run_dir / "07_vehicle_search_index_flat.json").write_text("[]", encoding="utf-8")


def main() -> None:
    """Run isolated Step 07 search index enrichment."""

    config = read_config()
    log(f"Run directory: {config.run_dir}")
    log("Loaded verified OCR results: 06_ocr_color_results_verified.json")

    try:
        index_payload, report_payload, _flat_payload = run_search_index_enrichment(
            run_dir=config.run_dir,
            index_config=config.index_config,
        )
        update_stage_gate_report(
            config.run_dir,
            "07_search_index_enrichment",
            {
                "status": index_payload["status"],
                "total_vehicle_records": report_payload["total_vehicle_records"],
                "searchable_records": report_payload["searchable_records"],
                "records_with_verified_plate": report_payload["records_with_verified_plate"],
                "unique_verified_plate_count": report_payload["unique_verified_plate_count"],
                "records_with_color": report_payload["records_with_color"],
                "records_with_full_frame": report_payload["records_with_full_frame"],
                "invalid_ocr_terms_removed_from_search_text_count": report_payload["invalid_ocr_terms_removed_from_search_text_count"],
                "primary_records": report_payload["primary_records"],
                "fallback_records": report_payload["fallback_records"],
                "ready_for_step08_search_validation": report_payload["searchable_records"] > 0,
            },
        )
        log(f"Records loaded: {report_payload['total_vehicle_records']}")
        log(f"Records created: {report_payload['searchable_records']}")
        log(f"Full frame paths linked: {report_payload['records_with_full_frame']}")
        log(f"Full frame paths missing: {report_payload['records_missing_full_frame']}")
        log(f"Invalid OCR terms removed from search text: {report_payload['invalid_ocr_terms_removed_from_search_text_count']}")
        log(f"Records with verified plates: {report_payload['records_with_verified_plate']}")
        log(f"Records with color: {report_payload['records_with_color']}")
        log(f"Unique verified plates: {report_payload['unique_verified_plate_count']}")
        log(
            "Output paths: "
            f"{config.run_dir / '07_vehicle_search_index.json'} | "
            f"{config.run_dir / '07_vehicle_search_index_report.json'}"
        )
    except Exception as exc:
        _write_failed_reports(config.run_dir, config.index_config, str(exc))
        update_stage_gate_report(config.run_dir, "07_search_index_enrichment", build_failure_payload(exc))
        log(f"Step 07 failed: {exc}")
        log(f"Run directory: {config.run_dir}")
        raise


if __name__ == "__main__":
    main()
