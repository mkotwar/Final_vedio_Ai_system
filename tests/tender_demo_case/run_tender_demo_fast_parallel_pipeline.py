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
from step_13_rank_candidate_clips import rank_candidate_clips
from step_14_select_topk_clips import select_topk_clips_for_qwen
from step_15_create_topk_vlm_inputs import create_topk_vlm_inputs
from step_16_run_topk_qwen import run_qwen_on_topk_vlm_inputs
from step_17_topk_final_summary import create_topk_final_summary
from step_18_export_event_clips import export_event_clips
from step_19_create_demo_report import create_demo_report_html


FAST_DEFAULTS = {
    "TENDER_DEMO_SAMPLE_EVERY_SECONDS": "2.0",
    "TENDER_DEMO_TOP_K_CLIPS": "5",
    "TENDER_DEMO_QWEN_BATCH_SIZE": "1",
    "TENDER_DEMO_QWEN_MAX_NEW_TOKENS": "256",
    "TENDER_DEMO_YOLO_IMGSZ": "416",
    "TENDER_DEMO_YOLO_CONF": "0.35",
    "TENDER_DEMO_CREATE_COMPILED_REVIEW_VIDEO": "true",
    "TENDER_DEMO_COMPILE_NORMAL_IF_NO_EVENTS": "true",
    "TENDER_DEMO_FAST_PARALLEL_BRANCHES": "true",
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


def _runtime_settings_snapshot() -> dict[str, Any]:
    return {
        "sample_every_seconds": float(os.environ.get("TENDER_DEMO_SAMPLE_EVERY_SECONDS", FAST_DEFAULTS["TENDER_DEMO_SAMPLE_EVERY_SECONDS"])),
        "top_k_clips": int(os.environ.get("TENDER_DEMO_TOP_K_CLIPS", FAST_DEFAULTS["TENDER_DEMO_TOP_K_CLIPS"])),
        "qwen_batch_size": int(os.environ.get("TENDER_DEMO_QWEN_BATCH_SIZE", FAST_DEFAULTS["TENDER_DEMO_QWEN_BATCH_SIZE"])),
        "qwen_max_new_tokens": int(os.environ.get("TENDER_DEMO_QWEN_MAX_NEW_TOKENS", FAST_DEFAULTS["TENDER_DEMO_QWEN_MAX_NEW_TOKENS"])),
        "yolo_imgsz": int(os.environ.get("TENDER_DEMO_YOLO_IMGSZ", FAST_DEFAULTS["TENDER_DEMO_YOLO_IMGSZ"])),
        "yolo_conf": float(os.environ.get("TENDER_DEMO_YOLO_CONF", FAST_DEFAULTS["TENDER_DEMO_YOLO_CONF"])),
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
    print("[tender-demo-fast] Starting clip branch: Steps 5-6")
    _create_candidate_clips(
        motion_candidates=motion_candidates,
        run_dir=run_dir,
        max_gap_seconds=_read_positive_float_env(ENV_MAX_GAP_SECONDS, DEFAULT_MAX_GAP_SECONDS, "max gap seconds"),
        max_clip_seconds=_read_positive_float_env(ENV_MAX_CLIP_SECONDS, DEFAULT_MAX_CLIP_SECONDS, "max clip seconds"),
        overlap_seconds=_read_positive_float_env(ENV_CLIP_OVERLAP_SECONDS, DEFAULT_CLIP_OVERLAP_SECONDS, "clip overlap seconds"),
    )
    candidate_clips = []
    candidate_path = run_dir / "05_candidate_clips.json"
    if candidate_path.exists():
        import json
        candidate_clips = json.loads(candidate_path.read_text(encoding="utf-8"))

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
    return {
        "branch_metrics": build_parallel_branch_result("clip_branch", [5, 6], branch_started, status="success"),
    }


def _run_yolo_branch(run_dir: Path) -> dict[str, Any]:
    branch_started = now_seconds()
    print("[tender-demo-fast] Starting YOLO branch: Steps 10-11")
    run_yolo_detection_on_selected_frames(run_dir)
    run_yolo_object_scoring(run_dir)
    return {
        "branch_metrics": build_parallel_branch_result("yolo_branch", [10, 11], branch_started, status="success"),
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
        for future, branch_name in futures.items():
            try:
                branch_results.append(future.result()["branch_metrics"])
            except Exception as exc:
                print(f"[tender-demo-fast] Parallel branch failed: {branch_name}")
                raise RuntimeError(f"Parallel branch failed: {branch_name}: {exc}") from exc

    parallel_sections.append(
        build_parallel_section_result(
            "clip_branch_and_yolo_branch",
            section_started,
            branch_results,
        )
    )
    print(f"[tender-demo-fast] Finished parallel section in {round(now_seconds() - section_started, 2)}s")


def main() -> None:
    for name, value in FAST_DEFAULTS.items():
        set_default_env(name, value)

    pipeline_started = now_seconds()
    step_metrics: list[dict[str, Any]] = []
    parallel_sections: list[dict[str, Any]] = []
    parallel_enabled = _read_env_bool("TENDER_DEMO_FAST_PARALLEL_BRANCHES", True)

    video_path = _read_video_path()
    run_dir = _create_debug_run_dir(video_path)

    try:
        def _step_1_video_info() -> dict[str, object]:
            video_info = _extract_video_info(video_path)
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
        _run_step(14, "select Top-K clips", lambda: select_topk_clips_for_qwen(run_dir), step_metrics)
        _run_step(15, "create Top-K VLM inputs", lambda: create_topk_vlm_inputs(run_dir), step_metrics)
        _run_step(16, "Qwen on Top-K only", lambda: run_qwen_on_topk_vlm_inputs(run_dir), step_metrics)
        _run_step(17, "final summary", lambda: create_topk_final_summary(run_dir), step_metrics)
        _run_step(18, "export/compile review video", lambda: export_event_clips(run_dir), step_metrics)
        _run_step(19, "HTML report", lambda: create_demo_report_html(run_dir), step_metrics)

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
                "skipped_steps": SKIPPED_STEPS,
            }
            write_runtime_metrics(run_dir, metrics)
            print(f"[tender-demo-fast] Debug run directory: {run_dir}")
        raise


if __name__ == "__main__":
    main()
