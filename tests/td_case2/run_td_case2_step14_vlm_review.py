from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import (
    DEFAULT_QWEN_API_MODEL,
    DEFAULT_QWEN_API_PROVIDER,
    DEFAULT_STEP14_DEVICE,
    DEFAULT_STEP14_MAX_INPUTS,
    DEFAULT_STEP14_MAX_NEW_TOKENS,
    DEFAULT_STEP14_MODEL_PATH,
    DEFAULT_STEP14_REQUIRE_STRIP,
    DEFAULT_STEP14_USE_CACHE,
    DEFAULT_VLM_BACKEND,
    ENV_RUN_DIR,
    ENV_QWEN_API_MODEL,
    ENV_QWEN_API_PROVIDER,
    ENV_STEP14_DEVICE,
    ENV_STEP14_MAX_INPUTS,
    ENV_STEP14_MAX_NEW_TOKENS,
    ENV_STEP14_MODEL_PATH,
    ENV_STEP14_REQUIRE_STRIP,
    ENV_STEP14_USE_CACHE,
    ENV_VLM_BACKEND,
)
from run_td_case2_step01_02 import log
from stage_checks import build_failure_payload, update_stage_gate_report, write_json
from step_09_search_result_packaging import write_json_any
from step_14_vlm_event_review import run_vlm_event_review


@dataclass(frozen=True)
class Step14Config:
    run_dir: Path
    review_config: dict[str, Any]


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


def read_config() -> Step14Config:
    raw_run_dir = os.environ.get(ENV_RUN_DIR, "").strip()
    if not raw_run_dir:
        raise ValueError(f"Environment variable {ENV_RUN_DIR} is required for Step 14.")
    run_dir = Path(raw_run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"TD_CASE2_RUN_DIR does not point to an existing directory: {run_dir}")

    required_inputs = [
        run_dir / "13_vlm_event_inputs.json",
        run_dir / "13_vlm_event_input_report.json",
        run_dir / "12_selected_top_event_candidates.json",
        run_dir / "01_video_info.json",
    ]
    for required_path in required_inputs:
        if not required_path.exists():
            raise FileNotFoundError(f"Required Step 14 input is missing: {required_path}")

    return Step14Config(
        run_dir=run_dir.resolve(),
        review_config={
            "vlm_backend": os.environ.get(ENV_VLM_BACKEND, DEFAULT_VLM_BACKEND).strip().lower() or DEFAULT_VLM_BACKEND,
            "api_provider": os.environ.get(ENV_QWEN_API_PROVIDER, DEFAULT_QWEN_API_PROVIDER).strip() or DEFAULT_QWEN_API_PROVIDER,
            "api_model": os.environ.get(ENV_QWEN_API_MODEL, DEFAULT_QWEN_API_MODEL).strip() or DEFAULT_QWEN_API_MODEL,
            "model_path": os.environ.get(ENV_STEP14_MODEL_PATH, DEFAULT_STEP14_MODEL_PATH).strip() or DEFAULT_STEP14_MODEL_PATH,
            "max_inputs": _read_positive_int(ENV_STEP14_MAX_INPUTS, DEFAULT_STEP14_MAX_INPUTS),
            "max_new_tokens": _read_positive_int(ENV_STEP14_MAX_NEW_TOKENS, DEFAULT_STEP14_MAX_NEW_TOKENS),
            "use_cache": _read_bool(ENV_STEP14_USE_CACHE, DEFAULT_STEP14_USE_CACHE),
            "device": os.environ.get(ENV_STEP14_DEVICE, DEFAULT_STEP14_DEVICE).strip() or DEFAULT_STEP14_DEVICE,
            "require_strip": _read_bool(ENV_STEP14_REQUIRE_STRIP, DEFAULT_STEP14_REQUIRE_STRIP),
        },
    )


def _write_failed_reports(run_dir: Path, review_config: dict[str, Any], error_message: str) -> None:
    output_payload = {
        "status": "failed",
        "source_file": "13_vlm_event_inputs.json",
        "model": "Qwen2.5-VL-7B-Instruct",
        "config": review_config,
        "summary": {
            "inputs_loaded": 0,
            "inputs_reviewed": 0,
            "inputs_skipped": 0,
            "event_visible_count": 0,
            "normal_context_count": 0,
            "uncertain_count": 0,
            "used_temporal_strip_count": 0,
            "used_contact_sheet_count": 0,
            "ready_for_demo_report_ui": False,
        },
        "reviews": [],
        "error_message": error_message,
    }
    final_summary = {
        "overall_status": "step14_failed",
        "headline": "Step 14 review failed",
        "summary": error_message,
        "event_count": 0,
        "normal_context_count": 0,
        "uncertain_count": 0,
        "high_risk_event_count": 0,
        "medium_risk_event_count": 0,
        "low_risk_event_count": 0,
        "recommended_action": "Fix Step 14 failure before using this review output.",
    }
    report_payload = {
        "status": "failed",
        "vlm_backend": str(review_config.get("vlm_backend", DEFAULT_VLM_BACKEND)),
        "api_provider": review_config.get("api_provider"),
        "api_model": review_config.get("api_model"),
        "inputs_loaded": 0,
        "inputs_reviewed": 0,
        "inputs_skipped": 0,
        "event_visible_count": 0,
        "normal_context_count": 0,
        "uncertain_count": 0,
        "risk_counts": {},
        "event_type_counts": {},
        "model_path": str(review_config.get("model_path", "")),
        "model_load_time_seconds": 0.0,
        "api_success_count": 0,
        "api_failed_count": 0,
        "parse_success_count": 0,
        "parse_failed_count": 0,
        "average_latency_seconds": 0.0,
        "total_latency_seconds": 0.0,
        "estimated_cost_usd": None,
        "total_inference_time_seconds": 0.0,
        "average_inference_time_seconds": 0.0,
        "cuda_memory_allocated_after_mb": None,
        "cache_used_count": 0,
        "used_temporal_strip_count": 0,
        "used_contact_sheet_count": 0,
        "warnings": [],
        "recommendation": error_message,
        "error_message": error_message,
    }
    write_json(run_dir / "14_vlm_event_reviews.json", output_payload)
    write_json_any(run_dir / "14_vlm_event_reviews_flat.json", [])
    write_json(run_dir / "14_final_video_summary.json", final_summary)
    write_json(run_dir / "14_vlm_event_review_report.json", report_payload)


def main() -> None:
    config = read_config()
    log(f"Run directory: {config.run_dir}")
    try:
        output_payload, _flat_payload, final_summary_payload, report_payload = run_vlm_event_review(
            run_dir=config.run_dir,
            review_config=config.review_config,
        )
        update_stage_gate_report(
            config.run_dir,
            "14_vlm_event_review",
            {
                "status": output_payload["status"],
                "inputs_loaded": output_payload["summary"]["inputs_loaded"],
                "inputs_reviewed": output_payload["summary"]["inputs_reviewed"],
                "event_visible_count": output_payload["summary"]["event_visible_count"],
                "normal_context_count": output_payload["summary"]["normal_context_count"],
                "uncertain_count": output_payload["summary"]["uncertain_count"],
                "overall_status": final_summary_payload["overall_status"],
            },
        )
        log(f"Inputs loaded: {report_payload['inputs_loaded']}")
        log(f"Inputs reviewed: {report_payload['inputs_reviewed']}")
        log(f"Inputs skipped: {report_payload['inputs_skipped']}")
        log(f"Event visible count: {report_payload['event_visible_count']}")
        log(f"Normal context count: {report_payload['normal_context_count']}")
        log(f"Uncertain count: {report_payload['uncertain_count']}")
        log(
            "Output paths: "
            f"{config.run_dir / '14_vlm_event_reviews.json'} | "
            f"{config.run_dir / '14_vlm_event_reviews_flat.json'} | "
            f"{config.run_dir / '14_final_video_summary.json'} | "
            f"{config.run_dir / '14_vlm_event_review_report.json'}"
        )
    except Exception as exc:
        _write_failed_reports(config.run_dir, config.review_config, str(exc))
        update_stage_gate_report(config.run_dir, "14_vlm_event_review", build_failure_payload(exc))
        log(f"Step 14 failed: {exc}")
        log(f"Run directory: {config.run_dir}")
        raise


if __name__ == "__main__":
    main()
