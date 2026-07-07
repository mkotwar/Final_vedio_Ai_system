from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import (
    DEFAULT_STEP12_INCLUDE_LOW_CONFIDENCE,
    DEFAULT_STEP12_MAX_PER_EVENT_TYPE,
    DEFAULT_STEP12_MAX_PER_TIME_CLUSTER,
    DEFAULT_STEP12_MIN_RANKING_SCORE,
    DEFAULT_STEP12_MIN_TEMPORAL_GAP_SECONDS,
    DEFAULT_STEP12_PREFER_TRAFFIC_SAFETY,
    DEFAULT_STEP12_REQUIRE_FULL_FRAME_PATH,
    DEFAULT_STEP12_SAVE_FLAT,
    DEFAULT_STEP12_TOP_K,
    ENV_RUN_DIR,
    ENV_STEP12_INCLUDE_LOW_CONFIDENCE,
    ENV_STEP12_MAX_PER_EVENT_TYPE,
    ENV_STEP12_MAX_PER_TIME_CLUSTER,
    ENV_STEP12_MIN_RANKING_SCORE,
    ENV_STEP12_MIN_TEMPORAL_GAP_SECONDS,
    ENV_STEP12_PREFER_TRAFFIC_SAFETY,
    ENV_STEP12_REQUIRE_FULL_FRAME_PATH,
    ENV_STEP12_SAVE_FLAT,
    ENV_STEP12_TOP_K,
)
from run_td_case2_step01_02 import log
from stage_checks import build_failure_payload, update_stage_gate_report, write_json
from step_09_search_result_packaging import write_json_any
from step_12_event_candidate_ranking import run_event_candidate_ranking


@dataclass(frozen=True)
class Step12Config:
    """Runtime settings for isolated Step 12 event candidate ranking."""

    run_dir: Path
    ranking_config: dict[str, Any]


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


def read_config() -> Step12Config:
    """Read configuration for isolated Step 12 event candidate ranking."""

    raw_run_dir = os.environ.get(ENV_RUN_DIR, "").strip()
    if not raw_run_dir:
        raise ValueError(f"Environment variable {ENV_RUN_DIR} is required for Step 12.")
    run_dir = Path(raw_run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"TD_CASE2_RUN_DIR does not point to an existing directory: {run_dir}")

    required_inputs = [
        run_dir / "11_full_scene_event_candidates.json",
        run_dir / "11_full_scene_event_candidates_flat.json",
        run_dir / "11_full_scene_event_candidate_report.json",
    ]
    for required_path in required_inputs:
        if not required_path.exists():
            raise FileNotFoundError(f"Required Step 12 input is missing: {required_path}")

    return Step12Config(
        run_dir=run_dir.resolve(),
        ranking_config={
            "top_k": _read_positive_int(ENV_STEP12_TOP_K, DEFAULT_STEP12_TOP_K),
            "min_ranking_score": _read_positive_float(ENV_STEP12_MIN_RANKING_SCORE, DEFAULT_STEP12_MIN_RANKING_SCORE),
            "min_temporal_gap_seconds": _read_positive_float(
                ENV_STEP12_MIN_TEMPORAL_GAP_SECONDS,
                DEFAULT_STEP12_MIN_TEMPORAL_GAP_SECONDS,
            ),
            "max_per_event_type": _read_positive_int(
                ENV_STEP12_MAX_PER_EVENT_TYPE,
                DEFAULT_STEP12_MAX_PER_EVENT_TYPE,
            ),
            "max_per_time_cluster": _read_positive_int(
                ENV_STEP12_MAX_PER_TIME_CLUSTER,
                DEFAULT_STEP12_MAX_PER_TIME_CLUSTER,
            ),
            "prefer_traffic_safety": _read_bool(
                ENV_STEP12_PREFER_TRAFFIC_SAFETY,
                DEFAULT_STEP12_PREFER_TRAFFIC_SAFETY,
            ),
            "include_low_confidence": _read_bool(
                ENV_STEP12_INCLUDE_LOW_CONFIDENCE,
                DEFAULT_STEP12_INCLUDE_LOW_CONFIDENCE,
            ),
            "save_flat": _read_bool(ENV_STEP12_SAVE_FLAT, DEFAULT_STEP12_SAVE_FLAT),
            "require_full_frame_path": _read_bool(
                ENV_STEP12_REQUIRE_FULL_FRAME_PATH,
                DEFAULT_STEP12_REQUIRE_FULL_FRAME_PATH,
            ),
        },
    )


def _write_failed_reports(run_dir: Path, ranking_config: dict[str, Any], error_message: str) -> None:
    """Write Step 12 failure JSON artifacts."""

    ranked_payload = {
        "status": "failed",
        "source_file": "11_full_scene_event_candidates.json",
        "config": ranking_config,
        "summary": {
            "input_candidate_count": 0,
            "ranked_candidate_count": 0,
            "selected_top_k_count": 0,
            "rejected_candidate_count": 0,
            "temporal_cluster_count": 0,
            "ready_for_step13_vlm_input_generation": False,
        },
        "ranked_candidates": [],
        "error_message": error_message,
    }
    selected_payload = {
        "status": "failed",
        "source_file": "11_full_scene_event_candidates.json",
        "top_k": int(ranking_config.get("top_k", 0) or 0),
        "selected_count": 0,
        "selected_candidates": [],
        "error_message": error_message,
    }
    report_payload = {
        "status": "failed",
        "input_candidate_count": 0,
        "ranked_candidate_count": 0,
        "selected_top_k_count": 0,
        "temporal_cluster_count": 0,
        "selected_event_type_counts": {},
        "selected_confidence_counts": {},
        "selected_severity_counts": {},
        "ranking_label_counts": {},
        "vlm_priority_counts": {},
        "top_selected_candidates": [],
        "suppression_summary": {
            "candidates_suppressed_by_temporal_cluster": 0,
            "candidates_suppressed_by_event_type_cap": 0,
            "candidates_below_min_ranking_score": 0,
        },
        "warnings": [],
        "recommendation": error_message,
        "missing_files": error_message,
        "error_message": error_message,
    }
    write_json(run_dir / "12_ranked_event_candidates.json", ranked_payload)
    write_json(run_dir / "12_selected_top_event_candidates.json", selected_payload)
    write_json(run_dir / "12_event_candidate_ranking_report.json", report_payload)
    write_json_any(run_dir / "12_selected_event_candidates_flat.json", [])


def main() -> None:
    """Run isolated Step 12 event candidate ranking."""

    config = read_config()
    log(f"Run directory: {config.run_dir}")

    try:
        ranked_payload, selected_payload, report_payload, _flat_payload = run_event_candidate_ranking(
            run_dir=config.run_dir,
            ranking_config=config.ranking_config,
        )
        update_stage_gate_report(
            config.run_dir,
            "12_event_candidate_ranking",
            {
                "status": ranked_payload["status"],
                "input_candidate_count": ranked_payload["summary"]["input_candidate_count"],
                "ranked_candidate_count": ranked_payload["summary"]["ranked_candidate_count"],
                "selected_top_k_count": ranked_payload["summary"]["selected_top_k_count"],
                "temporal_cluster_count": ranked_payload["summary"]["temporal_cluster_count"],
                "ready_for_step13_vlm_input_generation": ranked_payload["summary"]["ready_for_step13_vlm_input_generation"],
            },
        )
        log(f"Input candidate count: {report_payload['input_candidate_count']}")
        log(f"Ranked candidate count: {report_payload['ranked_candidate_count']}")
        log(f"Temporal cluster count: {report_payload['temporal_cluster_count']}")
        log(f"Selected Top-K count: {report_payload['selected_top_k_count']}")
        log(f"Selected event type counts: {report_payload['selected_event_type_counts']}")
        log(
            "Output paths: "
            f"{config.run_dir / '12_ranked_event_candidates.json'} | "
            f"{config.run_dir / '12_selected_top_event_candidates.json'} | "
            f"{config.run_dir / '12_event_candidate_ranking_report.json'} | "
            f"{config.run_dir / '12_selected_event_candidates_flat.json'}"
        )
    except Exception as exc:
        _write_failed_reports(config.run_dir, config.ranking_config, str(exc))
        update_stage_gate_report(config.run_dir, "12_event_candidate_ranking", build_failure_payload(exc))
        log(f"Step 12 failed: {exc}")
        log(f"Run directory: {config.run_dir}")
        raise


if __name__ == "__main__":
    main()
