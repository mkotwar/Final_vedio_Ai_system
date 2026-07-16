from __future__ import annotations

import json
import os
from pathlib import Path

from config import (
    DEFAULT_QWEN_API_MODEL,
    DEFAULT_QWEN_API_PROVIDER,
    DEFAULT_VLM_BACKEND,
    ENV_QWEN_API_MODEL,
    ENV_QWEN_API_PROVIDER,
    ENV_RUN_DIR,
    ENV_VLM_BACKEND,
)
from device_manager import record_stage_device
from run_td_case2_step01_02 import log
from run_td_case2_step11_event_candidates import main as step11_main
from run_td_case2_step11_5_vlm_filter import main as step11_5_main
from run_td_case2_step12_event_ranking import main as step12_main
from run_td_case2_step13_vlm_inputs import main as step13_main
from run_td_case2_step14_vlm_review import main as step14_main
from run_td_case2_step15_searchable_events import main as step15_main
from run_td_case2_step16_evidence_video import main as step16_main
from stage_checks import read_json, write_json


def _record_cpu_stage(run_dir: Path, stage_name: str, component_name: str, reason: str, notes: list[str] | None = None) -> None:
    record_stage_device(
        run_dir=run_dir,
        stage_name=stage_name,
        component_name=component_name,
        supports_gpu=False,
        selected_device="cpu",
        actual_device="cpu",
        preprocessing_device="cpu",
        inference_device="cpu",
        postprocess_device="cpu",
        reason=reason,
        notes=notes,
    )


def _pipeline_status_path(run_dir: Path) -> Path:
    return run_dir / "pipeline_status.json"


def _write_pipeline_status(run_dir: Path, *, vlm_status: str, latest_completed_step: str, error_message: str | None = None) -> None:
    existing: dict[str, object] = {}
    status_path = _pipeline_status_path(run_dir)
    if status_path.exists():
        try:
            existing = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    backend = os.environ.get(ENV_VLM_BACKEND, DEFAULT_VLM_BACKEND).strip().lower() or DEFAULT_VLM_BACKEND
    existing.update(
        {
            "vlm_status": vlm_status,
            "vlm_backend": backend,
            "api_provider": os.environ.get(ENV_QWEN_API_PROVIDER, DEFAULT_QWEN_API_PROVIDER).strip() or DEFAULT_QWEN_API_PROVIDER,
            "api_model": os.environ.get(ENV_QWEN_API_MODEL, DEFAULT_QWEN_API_MODEL).strip() or DEFAULT_QWEN_API_MODEL,
            "latest_completed_step": latest_completed_step,
            "error_message": error_message,
        }
    )
    write_json(status_path, existing)


def main() -> None:
    raw_run_dir = os.environ.get(ENV_RUN_DIR, "").strip()
    if not raw_run_dir:
        raise ValueError(f"Environment variable {ENV_RUN_DIR} is required for the VLM pipeline.")
    run_dir = Path(raw_run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"TD_CASE2_RUN_DIR does not point to an existing directory: {run_dir}")

    backend = os.environ.get(ENV_VLM_BACKEND, DEFAULT_VLM_BACKEND).strip().lower() or DEFAULT_VLM_BACKEND
    api_provider = os.environ.get(ENV_QWEN_API_PROVIDER, DEFAULT_QWEN_API_PROVIDER).strip() or DEFAULT_QWEN_API_PROVIDER
    api_model = os.environ.get(ENV_QWEN_API_MODEL, DEFAULT_QWEN_API_MODEL).strip() or DEFAULT_QWEN_API_MODEL

    log(f"Run directory: {run_dir}")
    log(f"VLM backend: {backend}")
    log(f"Qwen API provider: {api_provider}")
    log(f"Qwen API model: {api_model}")

    _write_pipeline_status(run_dir, vlm_status="running", latest_completed_step="starting")
    try:
        step11_main()
        _record_cpu_stage(
            run_dir,
            "11_full_scene_event_candidates",
            "event_candidate_generation",
            "Step 11 is rule-based temporal aggregation over existing JSON artifacts; no GPU model runs here.",
        )
        _write_pipeline_status(run_dir, vlm_status="running", latest_completed_step="step11")
        step11_5_main()
        _write_pipeline_status(run_dir, vlm_status="running", latest_completed_step="step11_5")
        step12_main()
        _record_cpu_stage(
            run_dir,
            "12_event_candidate_ranking",
            "event_ranking",
            "Step 12 is deterministic scoring and selection logic on CPU.",
        )
        _write_pipeline_status(run_dir, vlm_status="running", latest_completed_step="step12")
        step13_main()
        _record_cpu_stage(
            run_dir,
            "13_vlm_event_inputs",
            "vlm_input_generation",
            "Step 13 builds temporal strips and contact sheets with OpenCV/image I/O on CPU.",
            notes=["Image composition is CPU-bound in the current td_case2 implementation."],
        )
        _write_pipeline_status(run_dir, vlm_status="running", latest_completed_step="step13")
        step14_main()
        _write_pipeline_status(run_dir, vlm_status="running", latest_completed_step="step14")
        step15_main()
        _record_cpu_stage(
            run_dir,
            "15_searchable_event_generation",
            "searchable_event_generation",
            "Step 15 normalizes reviewed scene events into searchable event JSON on CPU.",
        )
        _write_pipeline_status(run_dir, vlm_status="running", latest_completed_step="step15")
        step16_main()
        _record_cpu_stage(
            run_dir,
            "16_evidence_video_generation",
            "evidence_video_generation",
            "Step 16 reuses existing JSON artifacts and sampled frames to build a final evidence MP4 on CPU with OpenCV encoding.",
            notes=["No AI models are re-run in this stage."],
        )

        step14_report = read_json(run_dir / "14_vlm_event_review_report.json")
        final_status = "skipped" if step14_report.get("status") == "skipped" else "ready"
        _write_pipeline_status(run_dir, vlm_status=final_status, latest_completed_step="step16")
    except Exception as exc:
        _write_pipeline_status(run_dir, vlm_status="failed", latest_completed_step="failed", error_message=str(exc))
        raise


if __name__ == "__main__":
    main()
