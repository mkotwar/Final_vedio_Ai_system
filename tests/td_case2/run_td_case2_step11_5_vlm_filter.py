from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import (
    DEFAULT_QWEN_API_MODEL,
    DEFAULT_QWEN_API_PROVIDER,
    DEFAULT_STEP11_5_ALLOW_NORMAL_CONTEXT_BACKFILL,
    DEFAULT_STEP11_5_ALLOW_UNCERTAIN_BACKFILL,
    DEFAULT_STEP11_5_DEVICE,
    DEFAULT_STEP11_5_MAX_CANDIDATES_TO_CHECK,
    DEFAULT_STEP11_5_MAX_FILTERED_EVENTS,
    DEFAULT_STEP11_5_MAX_NEW_TOKENS,
    DEFAULT_STEP11_5_MIN_FILTERED_EVENTS,
    DEFAULT_STEP11_5_MODEL_PATH,
    DEFAULT_STEP11_5_USE_CACHE,
    DEFAULT_VLM_BACKEND,
    ENV_RUN_DIR,
    ENV_QWEN_API_MODEL,
    ENV_QWEN_API_PROVIDER,
    ENV_STEP11_5_ALLOW_NORMAL_CONTEXT_BACKFILL,
    ENV_STEP11_5_ALLOW_UNCERTAIN_BACKFILL,
    ENV_STEP11_5_DEVICE,
    ENV_STEP11_5_MAX_CANDIDATES_TO_CHECK,
    ENV_STEP11_5_MAX_FILTERED_EVENTS,
    ENV_STEP11_5_MAX_NEW_TOKENS,
    ENV_STEP11_5_MIN_FILTERED_EVENTS,
    ENV_STEP11_5_MODEL_PATH,
    ENV_STEP11_5_USE_CACHE,
    ENV_VLM_BACKEND,
)
from run_td_case2_step01_02 import log
from stage_checks import build_failure_payload, update_stage_gate_report, write_json
from step_09_search_result_packaging import write_json_any
from step_11_5_lightweight_vlm_filter import run_lightweight_vlm_filter


@dataclass(frozen=True)
class Step11_5Config:
    run_dir: Path
    filter_config: dict[str, Any]


def _read_bool(env_name: str, default_value: bool) -> bool:
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
    raw_value = os.environ.get(env_name, str(default_value)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {env_name} must be a valid integer. Received: {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"Environment variable {env_name} must be greater than 0. Received: {value}")
    return value


def read_config() -> Step11_5Config:
    raw_run_dir = os.environ.get(ENV_RUN_DIR, "").strip()
    if not raw_run_dir:
        raise ValueError(f"Environment variable {ENV_RUN_DIR} is required for Step 11.5.")
    run_dir = Path(raw_run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"TD_CASE2_RUN_DIR does not point to an existing directory: {run_dir}")

    for required_path in [run_dir / "11_full_scene_event_candidates.json", run_dir / "01_video_info.json"]:
        if not required_path.exists():
            raise FileNotFoundError(f"Required Step 11.5 input is missing: {required_path}")

    min_filtered_events = _read_positive_int(ENV_STEP11_5_MIN_FILTERED_EVENTS, DEFAULT_STEP11_5_MIN_FILTERED_EVENTS)
    max_filtered_events = _read_positive_int(ENV_STEP11_5_MAX_FILTERED_EVENTS, DEFAULT_STEP11_5_MAX_FILTERED_EVENTS)
    max_candidates_to_check = _read_positive_int(
        ENV_STEP11_5_MAX_CANDIDATES_TO_CHECK,
        DEFAULT_STEP11_5_MAX_CANDIDATES_TO_CHECK,
    )
    if min_filtered_events > max_filtered_events:
        raise ValueError("TD_CASE2_STEP11_5_MIN_FILTERED_EVENTS must be <= TD_CASE2_STEP11_5_MAX_FILTERED_EVENTS.")

    return Step11_5Config(
        run_dir=run_dir.resolve(),
        filter_config={
            "vlm_backend": os.environ.get(ENV_VLM_BACKEND, DEFAULT_VLM_BACKEND).strip().lower() or DEFAULT_VLM_BACKEND,
            "api_provider": os.environ.get(ENV_QWEN_API_PROVIDER, DEFAULT_QWEN_API_PROVIDER).strip() or DEFAULT_QWEN_API_PROVIDER,
            "api_model": os.environ.get(ENV_QWEN_API_MODEL, DEFAULT_QWEN_API_MODEL).strip() or DEFAULT_QWEN_API_MODEL,
            "model_path": os.environ.get(ENV_STEP11_5_MODEL_PATH, DEFAULT_STEP11_5_MODEL_PATH).strip() or DEFAULT_STEP11_5_MODEL_PATH,
            "max_candidates_to_check": max_candidates_to_check,
            "min_filtered_events": min_filtered_events,
            "max_filtered_events": max_filtered_events,
            "allow_uncertain_backfill": _read_bool(
                ENV_STEP11_5_ALLOW_UNCERTAIN_BACKFILL,
                DEFAULT_STEP11_5_ALLOW_UNCERTAIN_BACKFILL,
            ),
            "allow_normal_context_backfill": _read_bool(
                ENV_STEP11_5_ALLOW_NORMAL_CONTEXT_BACKFILL,
                DEFAULT_STEP11_5_ALLOW_NORMAL_CONTEXT_BACKFILL,
            ),
            "max_new_tokens": _read_positive_int(ENV_STEP11_5_MAX_NEW_TOKENS, DEFAULT_STEP11_5_MAX_NEW_TOKENS),
            "use_cache": _read_bool(ENV_STEP11_5_USE_CACHE, DEFAULT_STEP11_5_USE_CACHE),
            "device": os.environ.get(ENV_STEP11_5_DEVICE, DEFAULT_STEP11_5_DEVICE).strip() or DEFAULT_STEP11_5_DEVICE,
        },
    )


def _write_failed_reports(run_dir: Path, filter_config: dict[str, Any], error_message: str) -> None:
    output_payload = {
        "status": "failed",
        "source_file": "11_full_scene_event_candidates.json",
        "filter_model": "Qwen2.5-VL-3B-Instruct",
        "config": filter_config,
        "summary": {
            "input_candidate_count": 0,
            "candidates_checked_by_vlm": 0,
            "accepted_yes_count": 0,
            "uncertain_count": 0,
            "rejected_no_count": 0,
            "fallback_normal_context_count": 0,
            "final_filtered_candidate_count": 0,
            "ready_for_step12_event_ranking": False,
        },
        "candidate_events": [],
        "error_message": error_message,
    }
    report_payload = {
        "status": "failed",
        "source_file": "11_full_scene_event_candidates.json",
        "input_candidate_count": 0,
        "vlm_backend": str(filter_config.get("vlm_backend", DEFAULT_VLM_BACKEND)),
        "api_provider": filter_config.get("api_provider"),
        "api_model": filter_config.get("api_model"),
        "candidates_checked_by_vlm": 0,
        "accepted_yes_count": 0,
        "uncertain_count": 0,
        "rejected_no_count": 0,
        "fallback_normal_context_count": 0,
        "final_filtered_candidate_count": 0,
        "selected_for_step12_count": 0,
        "min_filtered_events": int(filter_config.get("min_filtered_events", 0) or 0),
        "max_filtered_events": int(filter_config.get("max_filtered_events", 0) or 0),
        "max_candidates_to_check": int(filter_config.get("max_candidates_to_check", 0) or 0),
        "model_path": str(filter_config.get("model_path", "")),
        "model_load_time_seconds": 0.0,
        "api_success_count": 0,
        "api_failed_count": 0,
        "total_api_latency_seconds": 0.0,
        "average_api_latency_seconds": 0.0,
        "total_inference_time_seconds": 0.0,
        "average_inference_time_seconds": 0.0,
        "fallback_used": False,
        "errors_summary": [],
        "cuda_memory_allocated_after_mb": None,
        "decision_counts": {},
        "event_type_counts_before": {},
        "event_type_counts_after": {},
        "top_filtered_candidates": [],
        "rejected_examples": [],
        "warnings": [],
        "ready_for_step12_event_ranking": False,
        "error_message": error_message,
    }
    write_json(run_dir / "11_5_vlm_filtered_event_candidates.json", output_payload)
    write_json(run_dir / "11_5_vlm_filter_report.json", report_payload)
    write_json_any(run_dir / "11_5_vlm_filter_results_flat.json", [])


def main() -> None:
    config = read_config()
    log(f"Run directory: {config.run_dir}")
    try:
        output_payload, report_payload, _flat_payload = run_lightweight_vlm_filter(
            run_dir=config.run_dir,
            filter_config=config.filter_config,
        )
        update_stage_gate_report(
            config.run_dir,
            "11_5_lightweight_vlm_candidate_filter",
            {
                "status": output_payload["status"],
                "input_candidate_count": output_payload["summary"]["input_candidate_count"],
                "candidates_checked_by_vlm": output_payload["summary"]["candidates_checked_by_vlm"],
                "final_filtered_candidate_count": output_payload["summary"]["final_filtered_candidate_count"],
                "accepted_yes_count": output_payload["summary"]["accepted_yes_count"],
                "uncertain_count": output_payload["summary"]["uncertain_count"],
                "rejected_no_count": output_payload["summary"]["rejected_no_count"],
                "ready_for_step12_event_ranking": output_payload["summary"]["ready_for_step12_event_ranking"],
            },
        )
        log(f"Input candidate count: {report_payload['input_candidate_count']}")
        log(f"Candidates checked by VLM: {report_payload['candidates_checked_by_vlm']}")
        log(f"Accepted yes count: {report_payload['accepted_yes_count']}")
        log(f"Uncertain count: {report_payload['uncertain_count']}")
        log(f"Rejected no count: {report_payload['rejected_no_count']}")
        log(f"Fallback normal context count: {report_payload['fallback_normal_context_count']}")
        log(f"Final filtered candidate count: {report_payload['final_filtered_candidate_count']}")
        log(
            "Output paths: "
            f"{config.run_dir / '11_5_vlm_filtered_event_candidates.json'} | "
            f"{config.run_dir / '11_5_vlm_filter_report.json'} | "
            f"{config.run_dir / '11_5_vlm_filter_results_flat.json'}"
        )
    except Exception as exc:
        _write_failed_reports(config.run_dir, config.filter_config, str(exc))
        update_stage_gate_report(config.run_dir, "11_5_lightweight_vlm_candidate_filter", build_failure_payload(exc))
        log(f"Step 11.5 failed: {exc}")
        log(f"Run directory: {config.run_dir}")
        raise


if __name__ == "__main__":
    main()
