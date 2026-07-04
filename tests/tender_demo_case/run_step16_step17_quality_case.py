from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPORT_JSON_NAME = "21_step16_step17_quality_case.json"
REPORT_MD_NAME = "21_step16_step17_quality_case.md"
DEFAULT_PROMPT_PATH = "tests/tender_demo_case/prompts/temporal_strip_observation_prompt.txt"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _debug_runs_root() -> Path:
    return Path(__file__).resolve().parent / "debug_runs"


def _resolve_path(path_value: str) -> Path:
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (_repo_root() / candidate).resolve()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("```json", " ").replace("```", " ").split()).strip()


def _format_seconds(seconds: float | int | None) -> str:
    if seconds is None:
        return "unknown"
    total = float(seconds)
    if total < 0:
        return "unknown"
    minutes = int(total // 60)
    remaining = total - minutes * 60
    if float(remaining).is_integer():
        return f"00:{minutes:02d}:{int(remaining):02d}"
    return f"00:{minutes:02d}:{remaining:04.1f}"


def _load_step17_module():
    module_path = Path(__file__).resolve().parent / "step_17_topk_final_summary.py"
    spec = importlib.util.spec_from_file_location("tender_demo_step17_quality", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Step 17 module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_env_updates(args: argparse.Namespace) -> dict[str, str]:
    return {
        "TENDER_DEMO_INPUT_VIDEO": str(_resolve_path(args.video)),
        "TENDER_DEMO_VLM_BACKEND": args.vlm_backend,
        "TENDER_DEMO_QWEN_LOCAL_FILES_ONLY": "true" if args.qwen_local_files_only else "false",
        "TENDER_DEMO_STEP16_PROMPT_FILE": str(_resolve_path(args.prompt_file)),
        "TENDER_DEMO_ANALYSIS_SENSITIVITY_MODE": args.mode,
        "TENDER_DEMO_SAMPLE_EVERY_SECONDS": str(args.sample_every_seconds),
        "TENDER_DEMO_TOP_K_CLIPS": str(args.top_k_clips),
        "TENDER_DEMO_TOP_K_MAX_CLIPS": str(args.top_k_max_clips),
        "TENDER_DEMO_MOTION_THRESHOLD": str(args.motion_threshold),
        "TENDER_DEMO_QWEN_MAX_NEW_TOKENS": str(args.max_new_tokens),
        "TENDER_DEMO_YOLO_IMGSZ": str(args.yolo_imgsz),
        "TENDER_DEMO_YOLO_CONF": str(args.yolo_conf),
        "TENDER_DEMO_ENABLE_INCIDENT_RECHECK": "true" if args.enable_incident_recheck else "false",
        "TENDER_DEMO_ENABLE_ADAPTIVE_SAMPLING": "true" if args.enable_adaptive_sampling else "false",
        "TENDER_DEMO_ENABLE_COVERAGE_GUARDRAILS": "true" if args.enable_coverage_guardrails else "false",
        "TENDER_DEMO_VLM_INPUT_STRATEGY": args.vlm_input_strategy,
        "TENDER_DEMO_MAX_VLM_INPUTS": str(args.max_vlm_inputs),
        "TENDER_DEMO_CREATE_COMPILED_REVIEW_VIDEO": "true" if args.create_compiled_review_video else "false",
        "TENDER_DEMO_MAX_VIDEO_SECONDS": str(args.max_video_seconds),
    }


def _detect_new_run_dir(before: set[str], after: set[str]) -> Path | None:
    new_names = sorted(after - before)
    if not new_names:
        return None
    return _debug_runs_root() / new_names[-1]


def _run_fast_pipeline(args: argparse.Namespace) -> Path:
    debug_root = _debug_runs_root()
    debug_root.mkdir(parents=True, exist_ok=True)
    before = {entry.name for entry in debug_root.iterdir() if entry.is_dir()}
    env = os.environ.copy()
    env.update(_build_env_updates(args))

    script_path = Path(__file__).resolve().parent / "run_tender_demo_fast_parallel_pipeline.py"
    command = [sys.executable, str(script_path)]
    completed = subprocess.run(command, env=env, cwd=str(_repo_root()), capture_output=True, text=True)

    if completed.returncode != 0:
        raise RuntimeError(
            "Tender-demo fast pipeline quality testcase failed.\n"
            f"STDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}"
        )

    for line in reversed(completed.stdout.splitlines()):
        marker = "Debug run directory:"
        if marker in line:
            run_dir = Path(line.split(marker, 1)[1].strip())
            if run_dir.exists():
                return run_dir

    after = {entry.name for entry in debug_root.iterdir() if entry.is_dir()}
    detected = _detect_new_run_dir(before, after)
    if detected and detected.exists():
        return detected
    raise RuntimeError("Pipeline completed, but the new debug run folder could not be detected.")


def _build_clip_observations(step17_payload: dict[str, Any]) -> list[dict[str, Any]]:
    clips: list[dict[str, Any]] = []
    for section_name in ["priority_suspicious_events", "possible_review_clips", "normal_activity_clips", "uncertain_clips"]:
        items = step17_payload.get(section_name, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            clips.append(
                {
                    "category": section_name,
                    "clip_id": item.get("clip_id"),
                    "time_range": item.get("time_range"),
                    "event_label": item.get("primary_event_label") or item.get("event_label"),
                    "scene_type": item.get("scene_type"),
                    "scene_description": item.get("scene_description", ""),
                    "temporal_change_summary": item.get("temporal_change_summary", ""),
                    "description": item.get("description", ""),
                    "objects": item.get("object_names", []),
                    "activities": item.get("activity_descriptions", []),
                    "parse_success": bool(item.get("parse_success")),
                    "fallback_used": bool(item.get("fallback_used")),
                }
            )
    clips.sort(key=lambda row: (_safe_float(row.get("time_range", "0").split(" - ")[0].replace(":", "").replace(".", ""), 0.0), str(row.get("clip_id", ""))))
    return clips


def _build_quality_findings(step16_payload: dict[str, Any], step17_payload: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    successful_parses = _safe_int(step16_payload.get("successful_outputs"), 0)
    fallback_outputs = _safe_int(step16_payload.get("fallback_outputs"), 0)
    overall_summary = _clean_text(step17_payload.get("overall_summary", ""))
    if successful_parses > 0 and fallback_outputs == 0:
        findings.append("Step 16 returned parseable JSON for all selected Top-K strips in this run.")
    elif successful_parses > 0:
        findings.append(f"Step 16 returned parseable JSON for {successful_parses} selected strips, with {fallback_outputs} fallback parse(s).")
    else:
        findings.append("Step 16 did not produce any parseable JSON outputs in this run.")
    if overall_summary:
        findings.append("Step 17 produced a human-readable summary from the selected strips.")
    if "scene_type" in overall_summary.lower() or "{ \"" in overall_summary:
        findings.append("Step 17 summary still contains schema-like noise and should be cleaned further.")
    else:
        findings.append("Step 17 summary no longer leaks raw schema fragments into the main summary text.")
    return findings


def _build_report_payload(run_dir: Path, prompt_path: Path) -> dict[str, Any]:
    step16_payload = _load_json(run_dir / "16_topk_vlm_outputs.json")
    step17_payload = _load_json(run_dir / "17_topk_final_summary.json")
    video_info = _load_json(run_dir / "01_video_info.json") if (run_dir / "01_video_info.json").exists() else {}
    runtime_metrics = _load_json(run_dir / "20_runtime_metrics.json") if (run_dir / "20_runtime_metrics.json").exists() else {}

    clip_observations = _build_clip_observations(step17_payload if isinstance(step17_payload, dict) else {})
    slowest_steps = runtime_metrics.get("slowest_steps", []) if isinstance(runtime_metrics, dict) else []
    slowest_step_name = ""
    if isinstance(slowest_steps, list) and slowest_steps:
        first_step = slowest_steps[0]
        if isinstance(first_step, dict):
            slowest_step_name = str(first_step.get("step_name", "")).strip()
    payload = {
        "test_case": "step16_step17_quality_case",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "prompt_file": str(prompt_path),
        "video_info": video_info if isinstance(video_info, dict) else {},
        "step16_summary": {
            "vlm_backend": step16_payload.get("vlm_backend"),
            "requested_vlm_backend": step16_payload.get("requested_vlm_backend"),
            "total_inputs": step16_payload.get("total_inputs"),
            "successful_outputs": step16_payload.get("successful_outputs"),
            "failed_outputs": step16_payload.get("failed_outputs"),
            "fallback_outputs": step16_payload.get("fallback_outputs"),
            "empty_outputs": step16_payload.get("empty_outputs"),
        },
        "step17_summary": {
            "overall_summary": step17_payload.get("overall_summary", ""),
            "descriptive_summary": step17_payload.get("descriptive_summary", ""),
            "scene_overview": step17_payload.get("scene_overview", {}),
            "processing_summary": step17_payload.get("processing_summary", {}),
        },
        "runtime_summary": {
            "total_runtime_seconds": runtime_metrics.get("total_runtime_seconds") if isinstance(runtime_metrics, dict) else None,
            "runtime_video_ratio": runtime_metrics.get("runtime_ratio_to_video") if isinstance(runtime_metrics, dict) else None,
            "slowest_step": slowest_step_name or None,
        },
        "quality_findings": _build_quality_findings(
            step16_payload if isinstance(step16_payload, dict) else {},
            step17_payload if isinstance(step17_payload, dict) else {},
        ),
        "clip_observations": clip_observations,
    }
    return payload


def _render_report_markdown(payload: dict[str, Any]) -> str:
    video_info = payload.get("video_info", {}) if isinstance(payload.get("video_info"), dict) else {}
    step16_summary = payload.get("step16_summary", {}) if isinstance(payload.get("step16_summary"), dict) else {}
    step17_summary = payload.get("step17_summary", {}) if isinstance(payload.get("step17_summary"), dict) else {}
    runtime_summary = payload.get("runtime_summary", {}) if isinstance(payload.get("runtime_summary"), dict) else {}
    clip_observations = payload.get("clip_observations", []) if isinstance(payload.get("clip_observations"), list) else []

    lines = [
        "# Step 16 / Step 17 Quality Case",
        "",
        f"- Run dir: `{payload.get('run_dir', '')}`",
        f"- Prompt file: `{payload.get('prompt_file', '')}`",
        f"- Video: `{video_info.get('video_name', 'unknown')}`",
        f"- Duration: `{video_info.get('duration_seconds', 'unknown')}` seconds",
        "",
        "## Step 16",
        "",
        f"- Backend: `{step16_summary.get('vlm_backend', 'unknown')}`",
        f"- Total inputs: `{step16_summary.get('total_inputs', 0)}`",
        f"- Successful outputs: `{step16_summary.get('successful_outputs', 0)}`",
        f"- Fallback outputs: `{step16_summary.get('fallback_outputs', 0)}`",
        "",
        "## Step 17",
        "",
        step17_summary.get("overall_summary", "") or "No Step 17 summary available.",
        "",
        "## Findings",
        "",
    ]
    for finding in payload.get("quality_findings", []):
        lines.append(f"- {finding}")

    lines.extend(
        [
            "",
            "## Runtime",
            "",
            f"- Total runtime seconds: `{runtime_summary.get('total_runtime_seconds', 'unknown')}`",
            f"- Runtime/video ratio: `{runtime_summary.get('runtime_video_ratio', 'unknown')}`",
            f"- Slowest step: `{runtime_summary.get('slowest_step', 'unknown')}`",
            "",
            "## Clip Observations",
            "",
        ]
    )
    for clip in clip_observations:
        lines.extend(
            [
                f"### {clip.get('clip_id', 'unknown')} | {clip.get('time_range', 'unknown')}",
                "",
                f"- Category: `{clip.get('category', 'unknown')}`",
                f"- Event label: `{clip.get('event_label', 'unknown')}`",
                f"- Scene description: {clip.get('scene_description', '') or 'n/a'}",
                f"- Temporal change: {clip.get('temporal_change_summary', '') or 'n/a'}",
                f"- Description: {clip.get('description', '') or 'n/a'}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def _write_report(run_dir: Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    json_path = run_dir / REPORT_JSON_NAME
    md_path = run_dir / REPORT_MD_NAME
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_render_report_markdown(payload), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or evaluate a tender-demo Step 16 / Step 17 quality testcase.")
    parser.add_argument("--video", help="Video path to run through the fast tender-demo pipeline.")
    parser.add_argument("--run-dir", help="Existing tender-demo debug run directory to evaluate without rerunning the pipeline.")
    parser.add_argument("--prompt-file", default=DEFAULT_PROMPT_PATH, help="Step 16 prompt file to use for the testcase.")
    parser.add_argument("--mode", default="Balanced")
    parser.add_argument("--sample-every-seconds", type=float, default=1.0)
    parser.add_argument("--top-k-clips", type=int, default=6)
    parser.add_argument("--top-k-max-clips", type=int, default=20)
    parser.add_argument("--motion-threshold", type=float, default=0.15)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--yolo-imgsz", type=int, default=416)
    parser.add_argument("--yolo-conf", type=float, default=0.35)
    parser.add_argument("--max-video-seconds", type=float, default=90.0)
    parser.add_argument("--vlm-input-strategy", default="center_only")
    parser.add_argument("--max-vlm-inputs", type=int, default=20)
    parser.add_argument("--vlm-backend", default="qwen")
    parser.add_argument("--enable-incident-recheck", action="store_true")
    parser.add_argument("--enable-adaptive-sampling", action="store_true")
    parser.add_argument("--enable-coverage-guardrails", action="store_true")
    parser.add_argument("--create-compiled-review-video", action="store_true")
    parser.add_argument("--qwen-local-files-only", action="store_true", default=True)
    args = parser.parse_args()

    if not args.video and not args.run_dir:
        raise ValueError("Provide either --video to run a testcase or --run-dir to evaluate an existing tender-demo run.")

    prompt_path = _resolve_path(args.prompt_file)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    if args.run_dir:
        run_dir = _resolve_path(args.run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run dir not found: {run_dir}")
    else:
        run_dir = _run_fast_pipeline(args)

    step17_module = _load_step17_module()
    step17_module.create_topk_final_summary(run_dir)
    payload = _build_report_payload(run_dir, prompt_path)
    json_path, md_path = _write_report(run_dir, payload)

    print(f"[tender-demo-testcase] Run dir: {run_dir}")
    print(f"[tender-demo-testcase] Report JSON: {json_path}")
    print(f"[tender-demo-testcase] Report MD: {md_path}")
    print(f"[tender-demo-testcase] Summary: {payload.get('step17_summary', {}).get('overall_summary', '')}")


if __name__ == "__main__":
    main()
