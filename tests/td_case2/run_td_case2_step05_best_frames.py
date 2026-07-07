from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import (
    DEFAULT_STEP05_AVOID_NEAR_DUPLICATES,
    DEFAULT_STEP05_FALLBACK_TOP_K_PER_TRACK,
    DEFAULT_STEP05_INCLUDE_ALL_VEHICLE_TRACKS,
    DEFAULT_STEP05_MAX_TRACKS,
    DEFAULT_STEP05_MIN_FALLBACK_SCORE,
    DEFAULT_STEP05_MIN_PRIMARY_SCORE,
    DEFAULT_STEP05_MIN_TIME_GAP_BETWEEN_SELECTED_SECONDS,
    DEFAULT_STEP05_PRIMARY_TOP_K_PER_TRACK,
    DEFAULT_STEP05_REQUIRE_CROP_EXISTS,
    DEFAULT_STEP05_SAVE_CONTACT_SHEETS,
    DEFAULT_STEP05_SAVE_SELECTED_CROPS,
    DEFAULT_STEP05_SAVE_SELECTED_FULL_FRAMES,
    DEFAULT_STEP05_TRACK_TYPES,
    ENV_RUN_DIR,
    ENV_STEP05_AVOID_NEAR_DUPLICATES,
    ENV_STEP05_FALLBACK_TOP_K_PER_TRACK,
    ENV_STEP05_INCLUDE_ALL_VEHICLE_TRACKS,
    ENV_STEP05_MAX_TRACKS,
    ENV_STEP05_MIN_FALLBACK_SCORE,
    ENV_STEP05_MIN_PRIMARY_SCORE,
    ENV_STEP05_MIN_TIME_GAP_BETWEEN_SELECTED_SECONDS,
    ENV_STEP05_PRIMARY_TOP_K_PER_TRACK,
    ENV_STEP05_REQUIRE_CROP_EXISTS,
    ENV_STEP05_SAVE_CONTACT_SHEETS,
    ENV_STEP05_SAVE_SELECTED_CROPS,
    ENV_STEP05_SAVE_SELECTED_FULL_FRAMES,
    ENV_STEP05_TRACK_TYPES,
)
from run_td_case2_step01_02 import log
from stage_checks import build_failure_payload, update_stage_gate_report, write_json
from step_05_best_track_frame_selector import run_best_track_frame_selector


@dataclass(frozen=True)
class Step05Config:
    """Runtime settings for isolated Step 05 best frame selection."""

    run_dir: Path
    selection_config: dict[str, Any]


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


def read_config() -> Step05Config:
    """Read configuration for isolated Step 05 best-frame selection."""

    raw_run_dir = os.environ.get(ENV_RUN_DIR, "").strip()
    if not raw_run_dir:
        raise ValueError(f"Environment variable {ENV_RUN_DIR} is required for Step 05.")

    run_dir = Path(raw_run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"TD_CASE2_RUN_DIR does not point to an existing directory: {run_dir}")

    required_inputs = [
        run_dir / "04B_tracks.json",
        run_dir / "04B_tracking_report.json",
        run_dir / "03_yolo_detections.json",
    ]
    for required_path in required_inputs:
        if not required_path.exists():
            raise FileNotFoundError(f"Required Step 05 input is missing: {required_path}")

    track_types = [item.strip().lower() for item in os.environ.get(ENV_STEP05_TRACK_TYPES, DEFAULT_STEP05_TRACK_TYPES).split(",") if item.strip()]
    if not track_types:
        raise ValueError(f"Environment variable {ENV_STEP05_TRACK_TYPES} must include at least one track type.")

    return Step05Config(
        run_dir=run_dir.resolve(),
        selection_config={
            "track_types": track_types,
            "include_all_vehicle_tracks": _read_bool(
                ENV_STEP05_INCLUDE_ALL_VEHICLE_TRACKS,
                DEFAULT_STEP05_INCLUDE_ALL_VEHICLE_TRACKS,
            ),
            "primary_top_k_per_track": max(
                1,
                _read_non_negative_int(
                    ENV_STEP05_PRIMARY_TOP_K_PER_TRACK,
                    DEFAULT_STEP05_PRIMARY_TOP_K_PER_TRACK,
                ),
            ),
            "fallback_top_k_per_track": max(
                1,
                _read_non_negative_int(
                    ENV_STEP05_FALLBACK_TOP_K_PER_TRACK,
                    DEFAULT_STEP05_FALLBACK_TOP_K_PER_TRACK,
                ),
            ),
            "min_primary_score": _read_non_negative_float(
                ENV_STEP05_MIN_PRIMARY_SCORE,
                DEFAULT_STEP05_MIN_PRIMARY_SCORE,
            ),
            "min_fallback_score": _read_non_negative_float(
                ENV_STEP05_MIN_FALLBACK_SCORE,
                DEFAULT_STEP05_MIN_FALLBACK_SCORE,
            ),
            "require_crop_exists": _read_bool(
                ENV_STEP05_REQUIRE_CROP_EXISTS,
                DEFAULT_STEP05_REQUIRE_CROP_EXISTS,
            ),
            "save_selected_crops": _read_bool(
                ENV_STEP05_SAVE_SELECTED_CROPS,
                DEFAULT_STEP05_SAVE_SELECTED_CROPS,
            ),
            "save_selected_full_frames": _read_bool(
                ENV_STEP05_SAVE_SELECTED_FULL_FRAMES,
                DEFAULT_STEP05_SAVE_SELECTED_FULL_FRAMES,
            ),
            "save_contact_sheets": _read_bool(
                ENV_STEP05_SAVE_CONTACT_SHEETS,
                DEFAULT_STEP05_SAVE_CONTACT_SHEETS,
            ),
            "max_tracks": _read_non_negative_int(
                ENV_STEP05_MAX_TRACKS,
                DEFAULT_STEP05_MAX_TRACKS,
            ),
            "avoid_near_duplicates": _read_bool(
                ENV_STEP05_AVOID_NEAR_DUPLICATES,
                DEFAULT_STEP05_AVOID_NEAR_DUPLICATES,
            ),
            "min_time_gap_between_selected_seconds": _read_non_negative_float(
                ENV_STEP05_MIN_TIME_GAP_BETWEEN_SELECTED_SECONDS,
                DEFAULT_STEP05_MIN_TIME_GAP_BETWEEN_SELECTED_SECONDS,
            ),
        },
    )


def _write_failed_reports(run_dir: Path, selection_config: dict[str, Any], error_message: str) -> None:
    """Write Step 05 failure JSON artifacts."""

    frames_payload = {
        "status": "failed",
        "input_tracks_file": "04B_tracks.json",
        "selection_config": selection_config,
        "vehicle_track_count_total": 0,
        "primary_vehicle_track_count": 0,
        "fallback_vehicle_track_count": 0,
        "selected_track_count": 0,
        "skipped_track_count": 0,
        "total_selected_detections": 0,
        "primary_selected_detections": 0,
        "fallback_selected_detections": 0,
        "tracks": [],
        "skipped_tracks": [],
        "error_message": error_message,
    }
    report_payload = {
        "status": "failed",
        "vehicle_track_count_total": 0,
        "primary_vehicle_track_count": 0,
        "fallback_vehicle_track_count": 0,
        "selected_track_count": 0,
        "primary_selected_track_count": 0,
        "fallback_selected_track_count": 0,
        "skipped_track_count": 0,
        "total_selected_detections": 0,
        "primary_selected_detections": 0,
        "fallback_selected_detections": 0,
        "selected_crops_saved": 0,
        "selected_full_frames_saved": 0,
        "missing_crop_count": 0,
        "missing_full_frame_count": 0,
        "contact_sheets_saved": 0,
        "skipped_no_valid_crop_count": 0,
        "skipped_track_ids_no_valid_crop": [],
        "selection_group_counts": {"primary": 0, "fallback": 0},
        "track_quality_counts_selected": {},
        "class_counts_selected": {},
        "avg_primary_selected_score": 0.0,
        "avg_fallback_selected_score": 0.0,
        "top_primary_selected_tracks": [],
        "top_fallback_selected_tracks": [],
        "recommendation": error_message,
        "error_message": error_message,
    }
    write_json(run_dir / "05_best_track_frames.json", frames_payload)
    write_json(run_dir / "05_best_track_frames_report.json", report_payload)


def main() -> None:
    """Run isolated Step 05 best frame/crop selection."""

    config = read_config()
    log(f"Run directory: {config.run_dir}")
    log("Input tracks file: 04B_tracks.json")

    try:
        frames_payload, report_payload = run_best_track_frame_selector(
            run_dir=config.run_dir,
            selection_config=config.selection_config,
        )
        update_stage_gate_report(
            config.run_dir,
            "05_best_track_frame_selection",
            {
                "status": frames_payload["status"],
                "vehicle_track_count_total": report_payload["vehicle_track_count_total"],
                "primary_vehicle_track_count": report_payload["primary_vehicle_track_count"],
                "fallback_vehicle_track_count": report_payload["fallback_vehicle_track_count"],
                "selected_track_count": report_payload["selected_track_count"],
                "primary_selected_track_count": report_payload["primary_selected_track_count"],
                "fallback_selected_track_count": report_payload["fallback_selected_track_count"],
                "skipped_track_count": report_payload["skipped_track_count"],
                "total_selected_detections": report_payload["total_selected_detections"],
                "selected_crops_saved": report_payload["selected_crops_saved"],
                "selected_full_frames_saved": report_payload["selected_full_frames_saved"],
                "contact_sheets_saved": report_payload["contact_sheets_saved"],
                "ready_for_step06_ocr_color": report_payload["total_selected_detections"] > 0,
            },
        )
        log(f"Total vehicle tracks found: {report_payload['vehicle_track_count_total']}")
        log(f"Primary vehicle tracks: {report_payload['primary_vehicle_track_count']}")
        log(f"Fallback vehicle tracks: {report_payload['fallback_vehicle_track_count']}")
        log(f"Selected tracks: {report_payload['selected_track_count']}")
        log(f"Skipped tracks: {report_payload['skipped_track_count']}")
        log(f"Total selected detections: {report_payload['total_selected_detections']}")
        log(f"Selected crops saved: {report_payload['selected_crops_saved']}")
        log(f"Selected full frames saved: {report_payload['selected_full_frames_saved']}")
        log(f"Contact sheets saved: {report_payload['contact_sheets_saved']}")
        log(
            "Report files: "
            f"{config.run_dir / '05_best_track_frames.json'} | "
            f"{config.run_dir / '05_best_track_frames_report.json'}"
        )
    except Exception as exc:
        _write_failed_reports(config.run_dir, config.selection_config, str(exc))
        update_stage_gate_report(config.run_dir, "05_best_track_frame_selection", build_failure_payload(exc))
        log(f"Step 05 failed: {exc}")
        log(f"Run directory: {config.run_dir}")
        raise


if __name__ == "__main__":
    main()
