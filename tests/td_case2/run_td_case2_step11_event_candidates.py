from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import (
    DEFAULT_STEP11_CONTEXT_AFTER_SECONDS,
    DEFAULT_STEP11_CONTEXT_BEFORE_SECONDS,
    DEFAULT_STEP11_INCLUDE_SEARCH_METADATA,
    DEFAULT_STEP11_MAX_EVENT_SECONDS,
    DEFAULT_STEP11_MERGE_GAP_SECONDS,
    DEFAULT_STEP11_MIN_CANDIDATE_SCORE,
    DEFAULT_STEP11_SAVE_FLAT,
    DEFAULT_STEP11_TOP_K_PREVIEW,
    DEFAULT_STEP11_WINDOW_SECONDS,
    DEFAULT_STEP11_WINDOW_STRIDE_SECONDS,
    ENV_RUN_DIR,
    ENV_STEP11_CONTEXT_AFTER_SECONDS,
    ENV_STEP11_CONTEXT_BEFORE_SECONDS,
    ENV_STEP11_INCLUDE_SEARCH_METADATA,
    ENV_STEP11_MAX_EVENT_SECONDS,
    ENV_STEP11_MERGE_GAP_SECONDS,
    ENV_STEP11_MIN_CANDIDATE_SCORE,
    ENV_STEP11_SAVE_FLAT,
    ENV_STEP11_TOP_K_PREVIEW,
    ENV_STEP11_WINDOW_SECONDS,
    ENV_STEP11_WINDOW_STRIDE_SECONDS,
)
from run_td_case2_step01_02 import log
from stage_checks import build_failure_payload, update_stage_gate_report, write_json
from step_11_full_scene_event_candidates import STEP02A_CANDIDATE_FILES, run_full_scene_event_candidate_generation
from step_09_search_result_packaging import write_json_any


@dataclass(frozen=True)
class Step11Config:
    """Runtime settings for isolated Step 11 event candidate generation."""

    run_dir: Path
    event_config: dict[str, Any]


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


def _has_step02a_file(run_dir: Path) -> bool:
    """Return whether any accepted Step 02A file exists."""

    return any((run_dir / filename).exists() for filename in STEP02A_CANDIDATE_FILES)


def read_config() -> Step11Config:
    """Read configuration for isolated Step 11 event candidate generation."""

    raw_run_dir = os.environ.get(ENV_RUN_DIR, "").strip()
    if not raw_run_dir:
        raise ValueError(f"Environment variable {ENV_RUN_DIR} is required for Step 11.")
    run_dir = Path(raw_run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"TD_CASE2_RUN_DIR does not point to an existing directory: {run_dir}")

    missing_files: list[str] = []
    for required_path in [
        run_dir / "01_video_info.json",
        run_dir / "03_yolo_detections.json",
        run_dir / "04B_tracks.json",
    ]:
        if not required_path.exists():
            missing_files.append(str(required_path))
    if not _has_step02a_file(run_dir):
        missing_files.append("One of the Step 02A adaptive frame files is required.")
    if missing_files:
        raise FileNotFoundError("; ".join(missing_files))

    return Step11Config(
        run_dir=run_dir.resolve(),
        event_config={
            "window_seconds": _read_positive_float(ENV_STEP11_WINDOW_SECONDS, DEFAULT_STEP11_WINDOW_SECONDS),
            "window_stride_seconds": _read_positive_float(ENV_STEP11_WINDOW_STRIDE_SECONDS, DEFAULT_STEP11_WINDOW_STRIDE_SECONDS),
            "merge_gap_seconds": _read_positive_float(ENV_STEP11_MERGE_GAP_SECONDS, DEFAULT_STEP11_MERGE_GAP_SECONDS),
            "max_event_seconds": _read_positive_float(ENV_STEP11_MAX_EVENT_SECONDS, DEFAULT_STEP11_MAX_EVENT_SECONDS),
            "context_before_seconds": _read_positive_float(
                ENV_STEP11_CONTEXT_BEFORE_SECONDS,
                DEFAULT_STEP11_CONTEXT_BEFORE_SECONDS,
            ),
            "context_after_seconds": _read_positive_float(
                ENV_STEP11_CONTEXT_AFTER_SECONDS,
                DEFAULT_STEP11_CONTEXT_AFTER_SECONDS,
            ),
            "min_candidate_score": _read_positive_float(
                ENV_STEP11_MIN_CANDIDATE_SCORE,
                DEFAULT_STEP11_MIN_CANDIDATE_SCORE,
            ),
            "top_k_preview": _read_positive_int(ENV_STEP11_TOP_K_PREVIEW, DEFAULT_STEP11_TOP_K_PREVIEW),
            "save_flat": _read_bool(ENV_STEP11_SAVE_FLAT, DEFAULT_STEP11_SAVE_FLAT),
            "include_search_metadata": _read_bool(
                ENV_STEP11_INCLUDE_SEARCH_METADATA,
                DEFAULT_STEP11_INCLUDE_SEARCH_METADATA,
            ),
        },
    )


def _write_failed_reports(run_dir: Path, event_config: dict[str, Any], error_message: str) -> None:
    """Write Step 11 failure JSON artifacts."""

    output_payload = {
        "status": "failed",
        "source_files": {},
        "config": event_config,
        "summary": {
            "raw_triggers_created": 0,
            "candidate_events_created": 0,
            "high_confidence_candidates": 0,
            "medium_confidence_candidates": 0,
            "low_confidence_candidates": 0,
            "event_type_counts": {},
            "ready_for_step12_event_ranking": False,
        },
        "candidate_events": [],
        "error_message": error_message,
    }
    report_payload = {
        "status": "failed",
        "video_duration_seconds": 0.0,
        "windows_created": 0,
        "frames_used": 0,
        "tracks_used": 0,
        "raw_triggers_created": 0,
        "candidate_events_created": 0,
        "event_type_counts": {},
        "confidence_counts": {"high": 0, "medium": 0, "low": 0},
        "search_index": {
            "requested": True,
            "status": "load_failed",
            "source_type": "none",
            "source_filename": None,
            "legacy_fallback_used": False,
            "records_loaded": 0,
            "records_normalized": 0,
            "records_with_track_id": 0,
            "records_with_verified_plate": 0,
            "records_with_color": 0,
            "candidate_events_total": 0,
            "candidate_events_enriched": 0,
            "candidate_events_without_enrichment": 0,
            "matched_track_ids": 0,
            "unmatched_candidate_track_ids": [],
            "warnings": [error_message],
        },
        "severity_counts": {"high": 0, "medium": 0, "low": 0},
        "top_candidates": [],
        "warnings": [],
        "recommendation": error_message,
        "missing_files": error_message,
        "error_message": error_message,
    }
    diagnostics_payload = {
        "raw_trigger_type_counts": {},
        "raw_trigger_reason_counts": {},
        "candidate_reason_counts": {},
        "candidate_confidence_breakdown_by_event_type": {},
        "involved_track_quality_counts": {},
        "event_type_track_quality_breakdown": {},
        "collision_candidate_diagnostics": {},
        "rejected_reason_counts": {},
        "rejected_event_type_counts": {},
        "top_noisy_candidate_examples": [],
        "error_message": error_message,
    }
    write_json(run_dir / "11_full_scene_event_candidates.json", output_payload)
    write_json_any(run_dir / "11_full_scene_event_candidates_flat.json", [])
    write_json(run_dir / "11_full_scene_event_candidate_report.json", report_payload)
    write_json(run_dir / "11_full_scene_event_candidate_diagnostics.json", diagnostics_payload)


def main() -> None:
    """Run isolated Step 11 full-scene event candidate generation."""

    config = read_config()
    log(f"Run directory: {config.run_dir}")

    try:
        output_payload, _flat_payload, report_payload, _diagnostics_payload = run_full_scene_event_candidate_generation(
            run_dir=config.run_dir,
            event_config=config.event_config,
        )
        update_stage_gate_report(
            config.run_dir,
            "11_full_scene_event_candidate_generation",
            {
                "status": output_payload["status"],
                "raw_triggers_created": output_payload["summary"]["raw_triggers_created"],
                "candidate_events_created": output_payload["summary"]["candidate_events_created"],
                "high_confidence_candidates": output_payload["summary"]["high_confidence_candidates"],
                "medium_confidence_candidates": output_payload["summary"]["medium_confidence_candidates"],
                "low_confidence_candidates": output_payload["summary"]["low_confidence_candidates"],
                "ready_for_step12_event_ranking": output_payload["summary"]["ready_for_step12_event_ranking"],
            },
        )
        log(f"Video duration: {report_payload['video_duration_seconds']}")
        log(f"Frames loaded: {report_payload['frames_used']}")
        log(f"YOLO detections loaded: {report_payload['yolo_detections_loaded']}")
        log(f"Tracks loaded: {report_payload['tracks_used']}")
        log(f"Windows created: {report_payload['windows_created']}")
        log(f"Raw triggers created: {report_payload['raw_triggers_created']}")
        log(f"Candidate events created: {report_payload['candidate_events_created']}")
        log(f"Event type counts: {report_payload['event_type_counts']}")
        search_index = dict(report_payload.get("search_index", {}))
        source_type = str(search_index.get("source_type", "none") or "none")
        if search_index.get("legacy_fallback_used"):
            log("[step11] Search enrichment: active 07B missing; using legacy Step 07 index.")
        elif source_type == "none":
            log("[step11] Search enrichment unavailable; continuing without search index.")
        else:
            log(
                "[step11] Search enrichment: "
                f"source={source_type}, records={search_index.get('records_normalized', 0)}, "
                f"enriched_candidates={search_index.get('candidate_events_enriched', 0)}/"
                f"{search_index.get('candidate_events_total', 0)}"
            )
        log(
            "Output paths: "
            f"{config.run_dir / '11_full_scene_event_candidates.json'} | "
            f"{config.run_dir / '11_full_scene_event_candidates_flat.json'} | "
            f"{config.run_dir / '11_full_scene_event_candidate_report.json'} | "
            f"{config.run_dir / '11_full_scene_event_candidate_diagnostics.json'}"
        )
    except Exception as exc:
        _write_failed_reports(config.run_dir, config.event_config, str(exc))
        update_stage_gate_report(config.run_dir, "11_full_scene_event_candidate_generation", build_failure_payload(exc))
        log(f"Step 11 failed: {exc}")
        log(f"Run directory: {config.run_dir}")
        raise


if __name__ == "__main__":
    main()
