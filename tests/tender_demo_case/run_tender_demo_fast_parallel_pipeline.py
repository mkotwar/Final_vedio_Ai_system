from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from run_tender_demo_pipeline import (
    DEFAULT_CLIP_OVERLAP_SECONDS,
    DEFAULT_CONTEXT_AFTER_SECONDS,
    DEFAULT_CONTEXT_BEFORE_SECONDS,
    DEFAULT_MAX_CLIP_SECONDS,
    DEFAULT_MAX_GAP_SECONDS,
    DEFAULT_MIN_EXPANDED_CLIP_SECONDS,
    ENV_CLIP_OVERLAP_SECONDS,
    ENV_CONTEXT_AFTER_SECONDS,
    ENV_CONTEXT_BEFORE_SECONDS,
    ENV_MAX_CLIP_SECONDS,
    ENV_MAX_GAP_SECONDS,
    ENV_MIN_EXPANDED_CLIP_SECONDS,
    _create_candidate_clips,
    _create_debug_run_dir,
    _expand_candidate_clips,
    _extract_video_info,
    _read_motion_threshold,
    _read_positive_float_env,
    _read_sample_every_seconds,
    _read_video_path,
    _sample_base_frames,
    _score_motion_on_sampled_frames,
    _select_motion_candidates,
    _write_video_info,
)
from step_00_runtime_metrics import (
    build_parallel_branch_result,
    build_parallel_section_result,
    build_step_result,
    compute_slowest_steps,
    now_seconds,
    write_runtime_metrics,
)
from step_10_yolo_detection import run_yolo_detection_on_selected_frames
from step_11_yolo_object_scoring import run_yolo_object_scoring
from step_11b_object_motion_state import estimate_object_motion_states
from step_02b_adaptive_sampling import run_adaptive_sampling
from step_02c_frame_candidate_pool import create_frame_candidate_pool
from step_13_rank_candidate_clips import rank_candidate_clips
from step_14_select_topk_clips import select_topk_clips_for_qwen
from step_14b_incident_coverage_guardrails import apply_incident_coverage_guardrails
from step_15_create_topk_vlm_inputs import create_topk_vlm_inputs
from step_16_run_topk_qwen import run_qwen_on_topk_vlm_inputs
from step_17_topk_final_summary import create_topk_final_summary
from step_18_export_event_clips import export_event_clips
from step_19_create_demo_report import create_demo_report_html

try:
    from step_16b_incident_recheck import run_incident_recheck_reasoning
except Exception:  # pragma: no cover - optional isolated step
    run_incident_recheck_reasoning = None


FAST_DEFAULTS = {
    "TENDER_DEMO_SAMPLE_EVERY_SECONDS": "3.0",
    "TENDER_DEMO_TOP_K_CLIPS": "5",
    "TENDER_DEMO_QWEN_BATCH_SIZE": "1",
    "TENDER_DEMO_QWEN_MAX_NEW_TOKENS": "256",
    "TENDER_DEMO_TOP_K_MAX_CLIPS": "25",
    "TENDER_DEMO_MOTION_THRESHOLD": "0.20",
    "TENDER_DEMO_YOLO_IMGSZ": "416",
    "TENDER_DEMO_YOLO_CONF": "0.35",
    "TENDER_DEMO_CREATE_COMPILED_REVIEW_VIDEO": "true",
    "TENDER_DEMO_COMPILE_NORMAL_IF_NO_EVENTS": "true",
    "TENDER_DEMO_FAST_PARALLEL_BRANCHES": "true",
    "TENDER_DEMO_ANALYSIS_SENSITIVITY_MODE": "Fast demo",
    "TENDER_DEMO_ENABLE_INCIDENT_RECHECK": "false",
    "TENDER_DEMO_INCIDENT_RECHECK_ALL_TOPK": "false",
    "TENDER_DEMO_INCIDENT_FALLBACK_PASS": "false",
    "TENDER_DEMO_PIPELINE_ENGINE": "fast_parallel_topk",
}
SKIPPED_STEPS = [7, 8, 9, 12]


def set_default_env(name: str, value: str) -> None:
    if name not in os.environ:
        os.environ[name] = value


def _read_env_bool(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() == "true"


def _read_env_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name, str(default)).strip()
    try:
        return float(raw_value)
    except ValueError:
        return default


def _sensitive_mode_defaults(mode: str) -> dict[str, Any]:
    normalized = str(mode or "").strip().lower()
    if normalized == "sensitive incident review":
        return {
            "adaptive_sampling_enabled": True,
            "adaptive_base_interval_seconds": 1.0,
            "adaptive_max_frame_gap_seconds": 4.0,
            "coverage_guardrails_enabled": True,
            "vlm_input_strategy": "multi_focus",
            "max_vlm_inputs": 40,
            "yolo_input_scope": "frame_candidate_pool",
        }
    if normalized in {"high accuracy", "high accuracy review"}:
        return {
            "adaptive_sampling_enabled": True,
            "adaptive_base_interval_seconds": 0.5,
            "adaptive_max_frame_gap_seconds": 3.0,
            "coverage_guardrails_enabled": True,
            "vlm_input_strategy": "multi_focus",
            "max_vlm_inputs": 50,
            "yolo_input_scope": "frame_candidate_pool",
        }
    return {
        "adaptive_sampling_enabled": False,
        "adaptive_base_interval_seconds": 1.0,
        "adaptive_max_frame_gap_seconds": 4.0,
        "coverage_guardrails_enabled": False,
        "vlm_input_strategy": "center_only",
        "max_vlm_inputs": 25,
        "yolo_input_scope": "motion_candidates",
    }


def _apply_mode_defaults() -> None:
    mode = os.environ.get("TENDER_DEMO_ANALYSIS_SENSITIVITY_MODE", FAST_DEFAULTS["TENDER_DEMO_ANALYSIS_SENSITIVITY_MODE"])
    defaults = _sensitive_mode_defaults(mode)
    set_default_env("TENDER_DEMO_ENABLE_ADAPTIVE_SAMPLING", "true" if defaults["adaptive_sampling_enabled"] else "false")
    set_default_env("TENDER_DEMO_ADAPTIVE_BASE_INTERVAL_SECONDS", str(defaults["adaptive_base_interval_seconds"]))
    set_default_env("TENDER_DEMO_ADAPTIVE_MOTION_THRESHOLD", "0.08")
    set_default_env("TENDER_DEMO_ADAPTIVE_HIST_THRESHOLD", "0.12")
    set_default_env("TENDER_DEMO_ADAPTIVE_SIMILARITY_THRESHOLD", "0.92")
    set_default_env("TENDER_DEMO_ADAPTIVE_MAX_FRAME_GAP_SECONDS", str(defaults["adaptive_max_frame_gap_seconds"]))
    set_default_env("TENDER_DEMO_ADAPTIVE_TARGET_WINDOW_SECONDS", "3.0")
    set_default_env("TENDER_DEMO_ENABLE_COVERAGE_GUARDRAILS", "true" if defaults["coverage_guardrails_enabled"] else "false")
    set_default_env("TENDER_DEMO_CRITICAL_WINDOW_SECONDS", "8")
    set_default_env("TENDER_DEMO_VLM_INPUT_STRATEGY", str(defaults["vlm_input_strategy"]))
    set_default_env("TENDER_DEMO_MAX_VLM_INPUTS", str(defaults["max_vlm_inputs"]))
    set_default_env("TENDER_DEMO_YOLO_INPUT_SCOPE", str(defaults["yolo_input_scope"]))


def _runtime_settings_snapshot() -> dict[str, Any]:
    return {
        "sample_every_seconds": float(os.environ.get("TENDER_DEMO_SAMPLE_EVERY_SECONDS", FAST_DEFAULTS["TENDER_DEMO_SAMPLE_EVERY_SECONDS"])),
        "top_k_clips": int(os.environ.get("TENDER_DEMO_TOP_K_CLIPS", FAST_DEFAULTS["TENDER_DEMO_TOP_K_CLIPS"])),
        "top_k_max": int(os.environ.get("TENDER_DEMO_TOP_K_MAX_CLIPS", FAST_DEFAULTS["TENDER_DEMO_TOP_K_MAX_CLIPS"])),
        "motion_threshold": float(os.environ.get("TENDER_DEMO_MOTION_THRESHOLD", FAST_DEFAULTS["TENDER_DEMO_MOTION_THRESHOLD"])),
        "qwen_batch_size": int(os.environ.get("TENDER_DEMO_QWEN_BATCH_SIZE", FAST_DEFAULTS["TENDER_DEMO_QWEN_BATCH_SIZE"])),
        "qwen_max_new_tokens": int(os.environ.get("TENDER_DEMO_QWEN_MAX_NEW_TOKENS", FAST_DEFAULTS["TENDER_DEMO_QWEN_MAX_NEW_TOKENS"])),
        "yolo_imgsz": int(os.environ.get("TENDER_DEMO_YOLO_IMGSZ", FAST_DEFAULTS["TENDER_DEMO_YOLO_IMGSZ"])),
        "yolo_conf": float(os.environ.get("TENDER_DEMO_YOLO_CONF", FAST_DEFAULTS["TENDER_DEMO_YOLO_CONF"])),
        "analysis_sensitivity_mode": os.environ.get("TENDER_DEMO_ANALYSIS_SENSITIVITY_MODE", FAST_DEFAULTS["TENDER_DEMO_ANALYSIS_SENSITIVITY_MODE"]),
        "incident_fallback_pass": _read_env_bool(
            "TENDER_DEMO_INCIDENT_FALLBACK_PASS",
            FAST_DEFAULTS["TENDER_DEMO_INCIDENT_FALLBACK_PASS"].strip().lower() == "true",
        ),
        "incident_recheck_enabled": _read_env_bool(
            "TENDER_DEMO_ENABLE_INCIDENT_RECHECK",
            FAST_DEFAULTS["TENDER_DEMO_ENABLE_INCIDENT_RECHECK"].strip().lower() == "true",
        ),
        "incident_recheck_all_topk": _read_env_bool(
            "TENDER_DEMO_INCIDENT_RECHECK_ALL_TOPK",
            FAST_DEFAULTS["TENDER_DEMO_INCIDENT_RECHECK_ALL_TOPK"].strip().lower() == "true",
        ),
        "adaptive_sampling_enabled": _read_env_bool("TENDER_DEMO_ENABLE_ADAPTIVE_SAMPLING", False),
        "coverage_guardrails_enabled": _read_env_bool("TENDER_DEMO_ENABLE_COVERAGE_GUARDRAILS", False),
        "vlm_input_strategy": os.environ.get("TENDER_DEMO_VLM_INPUT_STRATEGY", "center_only"),
        "max_vlm_inputs": int(os.environ.get("TENDER_DEMO_MAX_VLM_INPUTS", "25")),
        "critical_timestamps": [part.strip() for part in os.environ.get("TENDER_DEMO_CRITICAL_TIMESTAMPS", "").split(",") if part.strip()],
        "yolo_input_scope": os.environ.get("TENDER_DEMO_YOLO_INPUT_SCOPE", "motion_candidates"),
    }


def _analysis_settings_snapshot() -> dict[str, Any]:
    sample_every_seconds = _read_env_float(
        "TENDER_DEMO_SAMPLE_EVERY_SECONDS",
        float(FAST_DEFAULTS["TENDER_DEMO_SAMPLE_EVERY_SECONDS"]),
    )
    return {
        "mode": os.environ.get("TENDER_DEMO_ANALYSIS_SENSITIVITY_MODE", FAST_DEFAULTS["TENDER_DEMO_ANALYSIS_SENSITIVITY_MODE"]),
        "sample_every_seconds": sample_every_seconds,
        "approx_sampled_fps": round(1.0 / sample_every_seconds, 3) if sample_every_seconds > 0 else 0.0,
        "top_k_clips": int(os.environ.get("TENDER_DEMO_TOP_K_CLIPS", FAST_DEFAULTS["TENDER_DEMO_TOP_K_CLIPS"])),
        "top_k_max": int(os.environ.get("TENDER_DEMO_TOP_K_MAX_CLIPS", FAST_DEFAULTS["TENDER_DEMO_TOP_K_MAX_CLIPS"])),
        "motion_threshold": _read_env_float(
            "TENDER_DEMO_MOTION_THRESHOLD",
            float(FAST_DEFAULTS["TENDER_DEMO_MOTION_THRESHOLD"]),
        ),
        "yolo_imgsz": int(os.environ.get("TENDER_DEMO_YOLO_IMGSZ", FAST_DEFAULTS["TENDER_DEMO_YOLO_IMGSZ"])),
        "yolo_conf": _read_env_float("TENDER_DEMO_YOLO_CONF", float(FAST_DEFAULTS["TENDER_DEMO_YOLO_CONF"])),
        "qwen_max_new_tokens": int(os.environ.get("TENDER_DEMO_QWEN_MAX_NEW_TOKENS", FAST_DEFAULTS["TENDER_DEMO_QWEN_MAX_NEW_TOKENS"])),
        "incident_fallback_pass": _read_env_bool(
            "TENDER_DEMO_INCIDENT_FALLBACK_PASS",
            FAST_DEFAULTS["TENDER_DEMO_INCIDENT_FALLBACK_PASS"].strip().lower() == "true",
        ),
        "enable_incident_recheck": _read_env_bool(
            "TENDER_DEMO_ENABLE_INCIDENT_RECHECK",
            FAST_DEFAULTS["TENDER_DEMO_ENABLE_INCIDENT_RECHECK"].strip().lower() == "true",
        ),
        "incident_recheck_all_topk": _read_env_bool(
            "TENDER_DEMO_INCIDENT_RECHECK_ALL_TOPK",
            FAST_DEFAULTS["TENDER_DEMO_INCIDENT_RECHECK_ALL_TOPK"].strip().lower() == "true",
        ),
        "adaptive_sampling_enabled": _read_env_bool("TENDER_DEMO_ENABLE_ADAPTIVE_SAMPLING", False),
        "coverage_guardrails_enabled": _read_env_bool("TENDER_DEMO_ENABLE_COVERAGE_GUARDRAILS", False),
        "vlm_input_strategy": os.environ.get("TENDER_DEMO_VLM_INPUT_STRATEGY", "center_only"),
        "max_vlm_inputs": int(os.environ.get("TENDER_DEMO_MAX_VLM_INPUTS", "25")),
        "critical_timestamps": [part.strip() for part in os.environ.get("TENDER_DEMO_CRITICAL_TIMESTAMPS", "").split(",") if part.strip()],
        "yolo_input_scope": os.environ.get("TENDER_DEMO_YOLO_INPUT_SCOPE", "motion_candidates"),
    }


def _run_step(
    step_id: int,
    step_name: str,
    action,
    step_metrics: list[dict[str, Any]],
):
    print(f"[tender-demo-fast] Starting Step {step_id}: {step_name}")
    started_at = now_seconds()
    try:
        result = action()
    except Exception:
        step_metrics.append(build_step_result(step_id, step_name, started_at, status="failed"))
        raise
    step_result = build_step_result(step_id, step_name, started_at, status="success")
    step_metrics.append(step_result)
    print(f"[tender-demo-fast] Finished Step {step_id} in {step_result['duration_seconds']:.2f}s")
    return result


def _run_clip_branch(
    run_dir: Path,
    motion_candidates: list[dict[str, object]],
    video_info: dict[str, object],
) -> dict[str, Any]:
    branch_started = now_seconds()
    branch_steps: list[dict[str, Any]] = []
    print("[tender-demo-fast] Starting clip branch: Steps 5-6")
    step_5_started = now_seconds()
    _create_candidate_clips(
        motion_candidates=motion_candidates,
        run_dir=run_dir,
        max_gap_seconds=_read_positive_float_env(ENV_MAX_GAP_SECONDS, DEFAULT_MAX_GAP_SECONDS, "max gap seconds"),
        max_clip_seconds=_read_positive_float_env(ENV_MAX_CLIP_SECONDS, DEFAULT_MAX_CLIP_SECONDS, "max clip seconds"),
        overlap_seconds=_read_positive_float_env(ENV_CLIP_OVERLAP_SECONDS, DEFAULT_CLIP_OVERLAP_SECONDS, "clip overlap seconds"),
    )
    branch_steps.append(build_step_result(5, "candidate clips", step_5_started, status="success"))
    candidate_clips = []
    candidate_path = run_dir / "05_candidate_clips.json"
    if candidate_path.exists():
        import json
        candidate_clips = json.loads(candidate_path.read_text(encoding="utf-8"))

    step_6_started = now_seconds()
    _expand_candidate_clips(
        candidate_clips=candidate_clips,
        video_info=video_info,
        run_dir=run_dir,
        context_before_seconds=_read_positive_float_env(ENV_CONTEXT_BEFORE_SECONDS, DEFAULT_CONTEXT_BEFORE_SECONDS, "context before seconds"),
        context_after_seconds=_read_positive_float_env(ENV_CONTEXT_AFTER_SECONDS, DEFAULT_CONTEXT_AFTER_SECONDS, "context after seconds"),
        min_expanded_clip_seconds=_read_positive_float_env(
            ENV_MIN_EXPANDED_CLIP_SECONDS,
            DEFAULT_MIN_EXPANDED_CLIP_SECONDS,
            "minimum expanded clip seconds",
        ),
    )
    branch_steps.append(build_step_result(6, "expanded clips", step_6_started, status="success"))
    return {
        "branch_metrics": build_parallel_branch_result("clip_branch", [5, 6], branch_started, status="success"),
        "step_metrics": branch_steps,
    }


def _run_yolo_branch(run_dir: Path) -> dict[str, Any]:
    branch_started = now_seconds()
    branch_steps: list[dict[str, Any]] = []
    print("[tender-demo-fast] Starting YOLO branch: Steps 10-11-11B")
    step_10_started = now_seconds()
    run_yolo_detection_on_selected_frames(run_dir)
    branch_steps.append(build_step_result(10, "YOLO detection", step_10_started, status="success"))
    step_11_started = now_seconds()
    run_yolo_object_scoring(run_dir)
    branch_steps.append(build_step_result(11, "YOLO object scoring", step_11_started, status="success"))
    step_11b_started = now_seconds()
    estimate_object_motion_states(run_dir)
    branch_steps.append(build_step_result("11B", "object motion state estimation", step_11b_started, status="success"))
    return {
        "branch_metrics": build_parallel_branch_result("yolo_branch", [10, 11, "11B"], branch_started, status="success"),
        "step_metrics": branch_steps,
    }


def _run_parallel_or_sequential(
    run_dir: Path,
    motion_candidates: list[dict[str, object]],
    video_info: dict[str, object],
    step_metrics: list[dict[str, Any]],
    parallel_sections: list[dict[str, Any]],
    parallel_enabled: bool,
) -> None:
    section_started = now_seconds()
    print("[tender-demo-fast] Starting parallel section: clips + YOLO")

    if not parallel_enabled:
        clip_result = _run_clip_branch(run_dir, motion_candidates, video_info)
        yolo_result = _run_yolo_branch(run_dir)
        step_metrics.extend(clip_result.get("step_metrics", []))
        step_metrics.extend(yolo_result.get("step_metrics", []))
        parallel_sections.append(
            build_parallel_section_result(
                "clip_branch_and_yolo_branch",
                section_started,
                [clip_result["branch_metrics"], yolo_result["branch_metrics"]],
            )
        )
        print(f"[tender-demo-fast] Finished parallel section in {round(now_seconds() - section_started, 2)}s")
        return

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(_run_clip_branch, run_dir, motion_candidates, video_info): "clip_branch",
            executor.submit(_run_yolo_branch, run_dir): "yolo_branch",
        }
        branch_results: list[dict[str, Any]] = []
        branch_step_metrics: list[dict[str, Any]] = []
        for future, branch_name in futures.items():
            try:
                branch_result = future.result()
                branch_results.append(branch_result["branch_metrics"])
                branch_step_metrics.extend(branch_result.get("step_metrics", []))
            except Exception as exc:
                print(f"[tender-demo-fast] Parallel branch failed: {branch_name}")
                raise RuntimeError(f"Parallel branch failed: {branch_name}: {exc}") from exc

    step_metrics.extend(branch_step_metrics)
    parallel_sections.append(
        build_parallel_section_result(
            "clip_branch_and_yolo_branch",
            section_started,
            branch_results,
        )
    )
    print(f"[tender-demo-fast] Finished parallel section in {round(now_seconds() - section_started, 2)}s")


def _load_summary_counts(run_dir: Path) -> dict[str, int]:
    summary_path = run_dir / "17_topk_final_summary.json"
    if not summary_path.exists():
        return {
            "priority_suspicious_events": 0,
            "possible_review_clips": 0,
        }
    import json

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {
            "priority_suspicious_events": 0,
            "possible_review_clips": 0,
        }
    return {
        "priority_suspicious_events": len(payload.get("priority_suspicious_events", [])) if isinstance(payload.get("priority_suspicious_events"), list) else 0,
        "possible_review_clips": len(payload.get("possible_review_clips", [])) if isinstance(payload.get("possible_review_clips"), list) else 0,
    }


def main() -> None:
    for name, value in FAST_DEFAULTS.items():
        set_default_env(name, value)
    _apply_mode_defaults()

    pipeline_started = now_seconds()
    step_metrics: list[dict[str, Any]] = []
    parallel_sections: list[dict[str, Any]] = []
    parallel_enabled = _read_env_bool("TENDER_DEMO_FAST_PARALLEL_BRANCHES", True)
    max_video_seconds = _read_env_float("TENDER_DEMO_MAX_VIDEO_SECONDS", 0.0)

    print("[tender-demo-fast] Fast mode enabled: skipping old all-clip VLM path.")
    print("[tender-demo-fast] Qwen will run only on Top-K selected clips.")

    video_path = _read_video_path()
    run_dir = _create_debug_run_dir(video_path)

    try:
        def _step_1_video_info() -> dict[str, object]:
            video_info = _extract_video_info(video_path)
            if max_video_seconds > 0:
                original_total_frames = int(video_info.get("total_frames", 0) or 0)
                fps = float(video_info.get("fps", 0.0) or 0.0)
                original_duration_seconds = float(video_info.get("duration_seconds", 0.0) or 0.0)
                capped_duration = min(original_duration_seconds, max_video_seconds)
                if fps > 0 and original_total_frames > 0:
                    capped_total_frames = min(original_total_frames, int(round(capped_duration * fps)))
                    video_info["total_frames"] = capped_total_frames
                    video_info["duration_seconds"] = round(capped_duration, 3)
                    video_info["processing_duration_seconds"] = round(capped_duration, 3)
                    video_info["original_total_frames"] = original_total_frames
                    video_info["original_duration_seconds"] = round(original_duration_seconds, 3)
                    print(f"[tender-demo-fast] Processing first {capped_duration:.1f}s only due to TENDER_DEMO_MAX_VIDEO_SECONDS.")
            _write_video_info(run_dir, video_info)
            return video_info

        video_info = _run_step(1, "video info", _step_1_video_info, step_metrics)

        sample_every_seconds = _read_sample_every_seconds()
        _, _, sampled_frames = _run_step(
            2,
            "frame sampling",
            lambda: _sample_base_frames(
                video_path=video_path,
                run_dir=run_dir,
                fps=float(video_info["fps"]),
                total_frames=int(video_info["total_frames"]),
                sample_every_seconds=sample_every_seconds,
            ),
            step_metrics,
        )

        adaptive_sampling_enabled = _read_env_bool("TENDER_DEMO_ENABLE_ADAPTIVE_SAMPLING", False)
        if adaptive_sampling_enabled:
            _run_step("02B", "adaptive sampling", lambda: run_adaptive_sampling(run_dir), step_metrics)
            _run_step("02C", "frame candidate pool", lambda: create_frame_candidate_pool(run_dir), step_metrics)

        _, motion_scores = _run_step(
            3,
            "motion scoring",
            lambda: _score_motion_on_sampled_frames(sampled_frames=sampled_frames, run_dir=run_dir),
            step_metrics,
        )
        motion_threshold = _read_motion_threshold()
        _, motion_candidates = _run_step(
            4,
            "motion candidate selection",
            lambda: _select_motion_candidates(
                motion_scores=motion_scores,
                run_dir=run_dir,
                motion_threshold=motion_threshold,
            ),
            step_metrics,
        )

        _run_parallel_or_sequential(
            run_dir=run_dir,
            motion_candidates=motion_candidates,
            video_info=video_info,
            step_metrics=step_metrics,
            parallel_sections=parallel_sections,
            parallel_enabled=parallel_enabled,
        )

        _run_step(13, "rank candidate clips", lambda: rank_candidate_clips(run_dir), step_metrics)

        def _run_topk_tail() -> None:
            _run_step(14, "select Top-K clips", lambda: select_topk_clips_for_qwen(run_dir), step_metrics)
            if _read_env_bool("TENDER_DEMO_ENABLE_COVERAGE_GUARDRAILS", False):
                _run_step("14B", "incident coverage guardrails", lambda: apply_incident_coverage_guardrails(run_dir), step_metrics)
            _run_step(15, "create Top-K VLM inputs", lambda: create_topk_vlm_inputs(run_dir), step_metrics)
            if (run_dir / "15_vlm_coverage_audit.json").exists():
                _run_step("15B", "VLM coverage audit", lambda: (run_dir / "15_vlm_coverage_audit.json").read_text(encoding="utf-8"), step_metrics)
            _run_step(16, "Qwen on Top-K only", lambda: run_qwen_on_topk_vlm_inputs(run_dir), step_metrics)
            incident_recheck_enabled = _read_env_bool(
                "TENDER_DEMO_ENABLE_INCIDENT_RECHECK",
                FAST_DEFAULTS["TENDER_DEMO_ENABLE_INCIDENT_RECHECK"].strip().lower() == "true",
            )
            if incident_recheck_enabled:
                if run_incident_recheck_reasoning is not None:
                    _run_step("16B", "incident recheck reasoning", lambda: run_incident_recheck_reasoning(run_dir), step_metrics)
                else:
                    print("[tender-demo] Incident recheck requested but Step 16B is not available yet.")
            _run_step(17, "final summary", lambda: create_topk_final_summary(run_dir), step_metrics)
            _run_step(18, "export/compile review video", lambda: export_event_clips(run_dir), step_metrics)
            _run_step(19, "HTML report", lambda: create_demo_report_html(run_dir), step_metrics)

        _run_topk_tail()

        incident_fallback_enabled = _read_env_bool(
            "TENDER_DEMO_INCIDENT_FALLBACK_PASS",
            FAST_DEFAULTS["TENDER_DEMO_INCIDENT_FALLBACK_PASS"].strip().lower() == "true",
        )
        summary_counts = _load_summary_counts(run_dir)
        current_top_k = max(1, int(os.environ.get("TENDER_DEMO_TOP_K_CLIPS", FAST_DEFAULTS["TENDER_DEMO_TOP_K_CLIPS"])))
        top_k_max = max(1, int(os.environ.get("TENDER_DEMO_TOP_K_MAX_CLIPS", FAST_DEFAULTS["TENDER_DEMO_TOP_K_MAX_CLIPS"])))
        if (
            incident_fallback_enabled
            and summary_counts["priority_suspicious_events"] == 0
            and summary_counts["possible_review_clips"] == 0
            and current_top_k < top_k_max
        ):
            new_top_k = min(current_top_k + 5, top_k_max)
            os.environ["TENDER_DEMO_TOP_K_CLIPS"] = str(new_top_k)
            os.environ["TENDER_DEMO_INCIDENT_FALLBACK_PASS_USED"] = "true"
            os.environ["TENDER_DEMO_INCIDENT_FALLBACK_REASON"] = (
                f"No priority/review clips found in sensitive mode, expanded Top-K from {current_top_k} to {new_top_k}"
            )
            print(f"[tender-demo] {os.environ['TENDER_DEMO_INCIDENT_FALLBACK_REASON']}")
            _run_topk_tail()
        else:
            os.environ["TENDER_DEMO_INCIDENT_FALLBACK_PASS_USED"] = "false"
            if incident_fallback_enabled:
                os.environ["TENDER_DEMO_INCIDENT_FALLBACK_REASON"] = ""

        total_runtime_seconds = round(now_seconds() - pipeline_started, 3)
        video_duration_seconds = float(video_info.get("duration_seconds", 0.0) or 0.0)
        runtime_ratio = round(total_runtime_seconds / video_duration_seconds, 3) if video_duration_seconds > 0 else 0.0

        metrics = {
            "pipeline_name": "fast_parallel_topk",
            "pipeline_mode": "fast_parallel",
            "video_name": video_info.get("video_name"),
            "video_duration_seconds": video_duration_seconds,
            "total_runtime_seconds": total_runtime_seconds,
            "runtime_ratio_to_video": runtime_ratio,
            "parallel_branches_enabled": parallel_enabled,
            "steps": step_metrics,
            "parallel_sections": parallel_sections,
            "slowest_steps": compute_slowest_steps(step_metrics, limit=5),
            "settings": _runtime_settings_snapshot(),
            "analysis_settings": _analysis_settings_snapshot(),
            "skipped_steps": SKIPPED_STEPS,
        }
        metrics_path = write_runtime_metrics(run_dir, metrics)
        print(f"[tender-demo-fast] Total runtime: {total_runtime_seconds}s")
        print(f"[tender-demo-fast] Runtime/video ratio: {runtime_ratio}x")
        print(f"[tender-demo-fast] Runtime metrics path: {metrics_path}")
        print(f"[tender-demo-fast] Debug run directory: {run_dir}")
    except Exception:
        total_runtime_seconds = round(now_seconds() - pipeline_started, 3)
        if run_dir.exists():
            metrics = {
                "pipeline_name": "fast_parallel_topk",
                "pipeline_mode": "fast_parallel",
                "video_name": video_path.name,
                "video_duration_seconds": 0.0,
                "total_runtime_seconds": total_runtime_seconds,
                "runtime_ratio_to_video": 0.0,
                "parallel_branches_enabled": parallel_enabled,
                "steps": step_metrics,
                "parallel_sections": parallel_sections,
                "slowest_steps": compute_slowest_steps(step_metrics, limit=5),
                "settings": _runtime_settings_snapshot(),
                "analysis_settings": _analysis_settings_snapshot(),
                "skipped_steps": SKIPPED_STEPS,
            }
            write_runtime_metrics(run_dir, metrics)
            print(f"[tender-demo-fast] Debug run directory: {run_dir}")
        raise


if __name__ == "__main__":
    main()
