from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import (
    DEFAULT_STEP13_ADD_LABELS,
    DEFAULT_STEP13_CONTEXT_AFTER_SECONDS,
    DEFAULT_STEP13_CONTEXT_BEFORE_SECONDS,
    DEFAULT_STEP13_MAX_GROUP_DURATION_SECONDS,
    DEFAULT_STEP13_MAX_INPUTS,
    DEFAULT_STEP13_MERGE_GAP_SECONDS,
    DEFAULT_STEP13_MERGE_NEARBY_SELECTED,
    DEFAULT_STEP13_REQUIRE_FULL_FRAME_EXISTS,
    DEFAULT_STEP13_SAVE_CONTACT_SHEET,
    DEFAULT_STEP13_STRIP_MODE,
    DEFAULT_STEP13_STRIP_PANEL_HEIGHT,
    DEFAULT_STEP13_STRIP_WIDTH,
    ENV_RUN_DIR,
    ENV_STEP13_ADD_LABELS,
    ENV_STEP13_CONTEXT_AFTER_SECONDS,
    ENV_STEP13_CONTEXT_BEFORE_SECONDS,
    ENV_STEP13_MAX_GROUP_DURATION_SECONDS,
    ENV_STEP13_MAX_INPUTS,
    ENV_STEP13_MERGE_GAP_SECONDS,
    ENV_STEP13_MERGE_NEARBY_SELECTED,
    ENV_STEP13_REQUIRE_FULL_FRAME_EXISTS,
    ENV_STEP13_SAVE_CONTACT_SHEET,
    ENV_STEP13_STRIP_MODE,
    ENV_STEP13_STRIP_PANEL_HEIGHT,
    ENV_STEP13_STRIP_WIDTH,
)
from run_td_case2_step01_02 import log
from stage_checks import build_failure_payload, update_stage_gate_report, write_json
from step_09_search_result_packaging import write_json_any
from step_13_vlm_input_generation import ALLOWED_STRIP_MODES, run_vlm_input_generation


@dataclass(frozen=True)
class Step13Config:
    """Runtime settings for isolated Step 13 VLM input generation."""

    run_dir: Path
    vlm_config: dict[str, Any]


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


def read_config() -> Step13Config:
    """Read configuration for isolated Step 13 VLM input generation."""

    raw_run_dir = os.environ.get(ENV_RUN_DIR, "").strip()
    if not raw_run_dir:
        raise ValueError(f"Environment variable {ENV_RUN_DIR} is required for Step 13.")
    run_dir = Path(raw_run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"TD_CASE2_RUN_DIR does not point to an existing directory: {run_dir}")

    required_inputs = [
        run_dir / "12_selected_top_event_candidates.json",
        run_dir / "12_selected_event_candidates_flat.json",
        run_dir / "12_event_candidate_ranking_report.json",
        run_dir / "11_full_scene_event_candidates.json",
        run_dir / "01_video_info.json",
        run_dir / "02_sampled_frames",
    ]
    for required_path in required_inputs:
        if not required_path.exists():
            raise FileNotFoundError(f"Required Step 13 input is missing: {required_path}")

    strip_mode = os.environ.get(ENV_STEP13_STRIP_MODE, DEFAULT_STEP13_STRIP_MODE).strip().lower() or DEFAULT_STEP13_STRIP_MODE
    if strip_mode not in ALLOWED_STRIP_MODES:
        raise ValueError(f"{ENV_STEP13_STRIP_MODE} must be one of {sorted(ALLOWED_STRIP_MODES)}. Received: {strip_mode!r}")

    return Step13Config(
        run_dir=run_dir.resolve(),
        vlm_config={
            "merge_nearby_selected": _read_bool(ENV_STEP13_MERGE_NEARBY_SELECTED, DEFAULT_STEP13_MERGE_NEARBY_SELECTED),
            "merge_gap_seconds": _read_positive_float(ENV_STEP13_MERGE_GAP_SECONDS, DEFAULT_STEP13_MERGE_GAP_SECONDS),
            "max_group_duration_seconds": _read_positive_float(
                ENV_STEP13_MAX_GROUP_DURATION_SECONDS,
                DEFAULT_STEP13_MAX_GROUP_DURATION_SECONDS,
            ),
            "context_before_seconds": _read_positive_float(
                ENV_STEP13_CONTEXT_BEFORE_SECONDS,
                DEFAULT_STEP13_CONTEXT_BEFORE_SECONDS,
            ),
            "context_after_seconds": _read_positive_float(
                ENV_STEP13_CONTEXT_AFTER_SECONDS,
                DEFAULT_STEP13_CONTEXT_AFTER_SECONDS,
            ),
            "strip_mode": strip_mode,
            "strip_width": _read_positive_int(ENV_STEP13_STRIP_WIDTH, DEFAULT_STEP13_STRIP_WIDTH),
            "strip_panel_height": _read_positive_int(
                ENV_STEP13_STRIP_PANEL_HEIGHT,
                DEFAULT_STEP13_STRIP_PANEL_HEIGHT,
            ),
            "add_labels": _read_bool(ENV_STEP13_ADD_LABELS, DEFAULT_STEP13_ADD_LABELS),
            "save_contact_sheet": _read_bool(
                ENV_STEP13_SAVE_CONTACT_SHEET,
                DEFAULT_STEP13_SAVE_CONTACT_SHEET,
            ),
            "require_full_frame_exists": _read_bool(
                ENV_STEP13_REQUIRE_FULL_FRAME_EXISTS,
                DEFAULT_STEP13_REQUIRE_FULL_FRAME_EXISTS,
            ),
            "max_inputs": _read_positive_int(ENV_STEP13_MAX_INPUTS, DEFAULT_STEP13_MAX_INPUTS),
        },
    )


def _write_failed_reports(run_dir: Path, vlm_config: dict[str, Any], error_message: str) -> None:
    """Write Step 13 failure JSON artifacts."""

    output_payload = {
        "status": "failed",
        "source_file": "12_selected_top_event_candidates.json",
        "config": vlm_config,
        "summary": {
            "selected_candidates_loaded": 0,
            "merged_vlm_input_groups": 0,
            "vlm_inputs_created": 0,
            "temporal_strips_created": 0,
            "contact_sheets_created": 0,
            "inputs_ready_for_vlm": 0,
            "inputs_skipped": 0,
            "ready_for_step14_vlm_event_review": False,
        },
        "vlm_inputs": [],
        "error_message": error_message,
    }
    report_payload = {
        "status": "failed",
        "selected_candidates_loaded": 0,
        "merged_groups_created": 0,
        "vlm_inputs_created": 0,
        "temporal_strips_created": 0,
        "contact_sheets_created": 0,
        "inputs_ready_for_vlm": 0,
        "inputs_skipped": 0,
        "merge_summary": {
            "merged_candidate_count": 0,
            "unmerged_candidate_count": 0,
            "largest_group_size": 0,
            "groups": [],
        },
        "priority_counts": {},
        "warnings": [],
        "recommendation": error_message,
        "missing_files": error_message,
        "error_message": error_message,
    }
    write_json(run_dir / "13_vlm_event_inputs.json", output_payload)
    write_json_any(run_dir / "13_vlm_event_inputs_flat.json", [])
    write_json(run_dir / "13_vlm_event_input_report.json", report_payload)


def main() -> None:
    """Run isolated Step 13 VLM input generation."""

    config = read_config()
    log(f"Run directory: {config.run_dir}")

    try:
        output_payload, _flat_payload, report_payload = run_vlm_input_generation(
            run_dir=config.run_dir,
            vlm_config=config.vlm_config,
        )
        update_stage_gate_report(
            config.run_dir,
            "13_vlm_input_generation",
            {
                "status": output_payload["status"],
                "selected_candidates_loaded": output_payload["summary"]["selected_candidates_loaded"],
                "vlm_inputs_created": output_payload["summary"]["vlm_inputs_created"],
                "temporal_strips_created": output_payload["summary"]["temporal_strips_created"],
                "inputs_ready_for_vlm": output_payload["summary"]["inputs_ready_for_vlm"],
                "ready_for_step14_vlm_event_review": output_payload["summary"]["ready_for_step14_vlm_event_review"],
            },
        )
        log(f"Selected candidates loaded: {report_payload['selected_candidates_loaded']}")
        log(f"Merge enabled: {config.vlm_config['merge_nearby_selected']}")
        log(f"Merged groups created: {report_payload['merged_groups_created']}")
        log(f"Temporal strips created: {report_payload['temporal_strips_created']}")
        log(f"Contact sheets created: {report_payload['contact_sheets_created']}")
        log(f"Inputs ready for VLM: {report_payload['inputs_ready_for_vlm']}")
        log(
            "Output paths: "
            f"{config.run_dir / '13_vlm_event_inputs.json'} | "
            f"{config.run_dir / '13_vlm_event_inputs_flat.json'} | "
            f"{config.run_dir / '13_vlm_event_input_report.json'}"
        )
    except Exception as exc:
        _write_failed_reports(config.run_dir, config.vlm_config, str(exc))
        update_stage_gate_report(config.run_dir, "13_vlm_input_generation", build_failure_payload(exc))
        log(f"Step 13 failed: {exc}")
        log(f"Run directory: {config.run_dir}")
        raise


if __name__ == "__main__":
    main()
