from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import (
    DEFAULT_TRACKING_CLASSES,
    DEFAULT_TRACKING_ALLOW_VEHICLE_CLASS_SWITCH,
    DEFAULT_TRACKING_CLASS_SWITCH_MAX_CENTER_DISTANCE_RATIO,
    DEFAULT_TRACKING_CLASS_SWITCH_MAX_TIME_GAP_SECONDS,
    DEFAULT_TRACKING_CLASS_SWITCH_MIN_IOU,
    DEFAULT_TRACKING_ENABLED,
    DEFAULT_TRACKING_MAX_AREA_CHANGE_RATIO,
    DEFAULT_TRACKING_MAX_CENTER_DISTANCE_RATIO,
    DEFAULT_TRACKING_MAX_TIME_GAP_SECONDS,
    DEFAULT_TRACKING_MIN_CONFIDENCE,
    DEFAULT_TRACKING_MIN_IOU,
    DEFAULT_TRACKING_MIN_PERSON_CONFIDENCE,
    DEFAULT_TRACKING_MIN_TRACK_LENGTH,
    DEFAULT_TRACKING_MIN_VEHICLE_CONFIDENCE,
    DEFAULT_TRACKING_PREVIEW_LIMIT,
    DEFAULT_TRACKING_SAVE_PREVIEW,
    ENV_RUN_DIR,
    ENV_TRACKING_ALLOW_VEHICLE_CLASS_SWITCH,
    ENV_TRACKING_CLASS_SWITCH_MAX_CENTER_DISTANCE_RATIO,
    ENV_TRACKING_CLASS_SWITCH_MAX_TIME_GAP_SECONDS,
    ENV_TRACKING_CLASS_SWITCH_MIN_IOU,
    ENV_TRACKING_CLASSES,
    ENV_TRACKING_ENABLED,
    ENV_TRACKING_MAX_AREA_CHANGE_RATIO,
    ENV_TRACKING_MAX_CENTER_DISTANCE_RATIO,
    ENV_TRACKING_MAX_TIME_GAP_SECONDS,
    ENV_TRACKING_MIN_CONFIDENCE,
    ENV_TRACKING_MIN_IOU,
    ENV_TRACKING_MIN_PERSON_CONFIDENCE,
    ENV_TRACKING_MIN_TRACK_LENGTH,
    ENV_TRACKING_MIN_VEHICLE_CONFIDENCE,
    ENV_TRACKING_PREVIEW_LIMIT,
    ENV_TRACKING_SAVE_PREVIEW,
)
from run_td_case2_step01_02 import log
from stage_checks import build_failure_payload, update_stage_gate_report, write_json
from step_04b_tracking import run_tracking


@dataclass(frozen=True)
class TrackingConfig:
    """Runtime settings for isolated td_case2 tracking."""

    run_dir: Path
    tracking_config: dict[str, Any]


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


def read_config() -> TrackingConfig:
    """Read configuration for isolated Step 04B tracking."""

    raw_run_dir = os.environ.get(ENV_RUN_DIR, "").strip()
    if not raw_run_dir:
        raise ValueError(f"Environment variable {ENV_RUN_DIR} is required for Step 04B.")
    run_dir = Path(raw_run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"TD_CASE2_RUN_DIR does not point to an existing directory: {run_dir}")

    required_inputs = [
        run_dir / "03_yolo_detections.json",
        run_dir / "02A_adaptive_frames.json",
    ]
    for required_path in required_inputs:
        if not required_path.exists():
            raise FileNotFoundError(f"Required Step 04B input is missing: {required_path}")

    tracking_classes = [item.strip().lower() for item in os.environ.get(ENV_TRACKING_CLASSES, DEFAULT_TRACKING_CLASSES).split(",") if item.strip()]
    return TrackingConfig(
        run_dir=run_dir.resolve(),
        tracking_config={
            "enabled": _read_bool(ENV_TRACKING_ENABLED, DEFAULT_TRACKING_ENABLED),
            "tracking_classes": tracking_classes,
            "min_confidence": _read_positive_float(ENV_TRACKING_MIN_CONFIDENCE, DEFAULT_TRACKING_MIN_CONFIDENCE),
            "min_iou": _read_positive_float(ENV_TRACKING_MIN_IOU, DEFAULT_TRACKING_MIN_IOU),
            "max_time_gap_seconds": _read_positive_float(ENV_TRACKING_MAX_TIME_GAP_SECONDS, DEFAULT_TRACKING_MAX_TIME_GAP_SECONDS),
            "max_center_distance_ratio": _read_positive_float(
                ENV_TRACKING_MAX_CENTER_DISTANCE_RATIO,
                DEFAULT_TRACKING_MAX_CENTER_DISTANCE_RATIO,
            ),
            "max_area_change_ratio": _read_positive_float(
                ENV_TRACKING_MAX_AREA_CHANGE_RATIO,
                DEFAULT_TRACKING_MAX_AREA_CHANGE_RATIO,
            ),
            "allow_vehicle_class_switch": _read_bool(
                ENV_TRACKING_ALLOW_VEHICLE_CLASS_SWITCH,
                DEFAULT_TRACKING_ALLOW_VEHICLE_CLASS_SWITCH,
            ),
            "class_switch_min_iou": _read_positive_float(
                ENV_TRACKING_CLASS_SWITCH_MIN_IOU,
                DEFAULT_TRACKING_CLASS_SWITCH_MIN_IOU,
            ),
            "class_switch_max_center_distance_ratio": _read_positive_float(
                ENV_TRACKING_CLASS_SWITCH_MAX_CENTER_DISTANCE_RATIO,
                DEFAULT_TRACKING_CLASS_SWITCH_MAX_CENTER_DISTANCE_RATIO,
            ),
            "class_switch_max_time_gap_seconds": _read_positive_float(
                ENV_TRACKING_CLASS_SWITCH_MAX_TIME_GAP_SECONDS,
                DEFAULT_TRACKING_CLASS_SWITCH_MAX_TIME_GAP_SECONDS,
            ),
            "min_person_confidence": _read_positive_float(
                ENV_TRACKING_MIN_PERSON_CONFIDENCE,
                DEFAULT_TRACKING_MIN_PERSON_CONFIDENCE,
            ),
            "min_vehicle_confidence": _read_positive_float(
                ENV_TRACKING_MIN_VEHICLE_CONFIDENCE,
                DEFAULT_TRACKING_MIN_VEHICLE_CONFIDENCE,
            ),
            "min_track_length": _read_positive_int(ENV_TRACKING_MIN_TRACK_LENGTH, DEFAULT_TRACKING_MIN_TRACK_LENGTH),
            "save_preview": _read_bool(ENV_TRACKING_SAVE_PREVIEW, DEFAULT_TRACKING_SAVE_PREVIEW),
            "preview_limit": _read_positive_int(ENV_TRACKING_PREVIEW_LIMIT, DEFAULT_TRACKING_PREVIEW_LIMIT),
        },
    )


def _write_failed_reports(run_dir: Path, tracking_config: dict[str, Any], error_message: str) -> None:
    """Write failure JSON artifacts if Step 04B cannot proceed."""

    tracks_payload = {
        "status": "failed",
        "input_yolo_detections_file": "03_yolo_detections.json",
        "tracking_config": tracking_config,
        "frames_processed": 0,
        "detections_considered": 0,
        "detections_tracked": 0,
        "tracks_created": 0,
        "same_frame_multi_assignment_prevented_count": 0,
        "usable_tracks_for_next_step": [],
        "tracks": [],
        "error_message": error_message,
    }
    assignments_payload = {
        "status": "failed",
        "assignments": [],
        "error_message": error_message,
    }
    report_payload = {
        "status": "failed",
        "frames_processed": 0,
        "detections_total_from_yolo": 0,
        "detections_considered_for_tracking": 0,
        "detections_ignored_by_class": 0,
        "detections_ignored_by_confidence": 0,
        "tracks_created": 0,
        "active_track_count_end": 0,
        "same_frame_multi_assignment_prevented_count": 0,
        "track_type_counts": {},
        "dominant_class_track_counts": {},
        "track_length_stats": {"min": 0.0, "max": 0.0, "avg": 0.0},
        "track_duration_stats": {"min": 0.0, "max": 0.0, "avg": 0.0},
        "track_quality_counts": {},
        "usable_vehicle_tracks_for_ocr_color": 0,
        "usable_person_tracks": 0,
        "unusable_tracks": 0,
        "unusable_reason_counts": {},
        "top_tracks_by_detection_count": [],
        "top_tracks_by_duration": [],
        "top_tracks_by_confidence": [],
        "recommendation": error_message,
        "error_message": error_message,
    }
    quality_payload = {
        "status": "failed",
        "total_tracks": 0,
        "good_tracks": 0,
        "fragmented_tracks": 0,
        "short_tracks": 0,
        "weak_tracks": 0,
        "class_mixed_tracks": 0,
        "single_frame_tracks": 0,
        "usable_vehicle_tracks_for_ocr_color": 0,
        "top_bad_tracks_by_reason": [],
        "top_class_mixed_tracks": [],
        "top_fragmented_tracks": [],
        "recommendation": error_message,
        "error_message": error_message,
    }
    write_json(run_dir / "04B_tracks.json", tracks_payload)
    write_json(run_dir / "04B_detection_track_assignments.json", assignments_payload)
    write_json(run_dir / "04B_tracking_report.json", report_payload)
    write_json(run_dir / "04B_tracking_quality_report.json", quality_payload)


def main() -> None:
    """Run isolated Step 04B tracking."""

    config = read_config()
    log(f"Run directory: {config.run_dir}")
    log("Input YOLO file: 03_yolo_detections.json")
    log(f"Tracking classes: {config.tracking_config['tracking_classes']}")

    try:
        tracks_payload, _assignments_payload, report_payload, quality_report_payload = run_tracking(
            run_dir=config.run_dir,
            tracking_config=config.tracking_config,
        )
        update_stage_gate_report(
            config.run_dir,
            "04B_tracking",
            {
                "status": "success",
                "frames_processed": report_payload["frames_processed"],
                "detections_considered_for_tracking": report_payload["detections_considered_for_tracking"],
                "tracks_created": report_payload["tracks_created"],
                "track_type_counts": report_payload["track_type_counts"],
                "dominant_class_track_counts": report_payload["dominant_class_track_counts"],
                "good_tracks": report_payload["track_quality_counts"].get("good", 0),
                "fragmented_tracks": report_payload["track_quality_counts"].get("fragmented", 0),
                "short_tracks": report_payload["track_quality_counts"].get("short", 0),
                "class_mixed_tracks": report_payload["track_quality_counts"].get("class_mixed", 0),
                "usable_vehicle_tracks_for_ocr_color": report_payload["usable_vehicle_tracks_for_ocr_color"],
                "same_frame_multi_assignment_prevented_count": report_payload["same_frame_multi_assignment_prevented_count"],
                "tracking_preview_exists": (config.run_dir / "04B_tracking_preview_frames").exists(),
            },
        )
        log(f"Frames processed: {report_payload['frames_processed']}")
        log(f"Detections considered: {report_payload['detections_considered_for_tracking']}")
        log(f"Tracks created: {report_payload['tracks_created']}")
        log(f"Good tracks: {quality_report_payload['good_tracks']}")
        log(f"Fragmented tracks: {quality_report_payload['fragmented_tracks']}")
        log(f"Class mixed tracks: {quality_report_payload['class_mixed_tracks']}")
        log(f"Usable vehicle tracks for OCR/color: {report_payload['usable_vehicle_tracks_for_ocr_color']}")
        log(f"Track type counts: {report_payload['track_type_counts']}")
        log(f"Quality report: {config.run_dir / '04B_tracking_quality_report.json'}")
        log(
            "Output files: "
            f"{config.run_dir / '04B_tracks.json'} | "
            f"{config.run_dir / '04B_tracking_report.json'} | "
            f"{config.run_dir / '04B_detection_track_assignments.json'} | "
            f"{config.run_dir / '04B_tracking_quality_report.json'}"
        )
    except Exception as exc:
        _write_failed_reports(config.run_dir, config.tracking_config, str(exc))
        update_stage_gate_report(config.run_dir, "04B_tracking", build_failure_payload(exc))
        log(f"Step 04B failed: {exc}")
        log(f"Run directory: {config.run_dir}")
        raise


if __name__ == "__main__":
    main()
