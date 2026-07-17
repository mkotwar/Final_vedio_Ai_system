from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

try:
    from .comparison_schema import PipelineArtifacts, PipelineCommand
except ImportError:  # pragma: no cover
    from comparison_schema import PipelineArtifacts, PipelineCommand


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_dotenv(dotenv_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not dotenv_path.exists():
        return values
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def run_hybrid_pipeline(
    *,
    repo_root: Path,
    run_dir: Path,
    video_path: Path,
    camera_id: str,
    camera_group: str,
    camera_timezone: str,
    logs_dir: Path,
) -> PipelineArtifacts:
    python_exe = repo_root / "tests" / "td_case2" / ".venv" / "Scripts" / "python.exe"
    hybrid_script = repo_root / "tests" / "td_case2" / "hybrid_tracking_test" / "run_hybrid_tracking_test.py"
    post_script = repo_root / "tests" / "td_case2" / "hybrid_tracking_test" / "run_post_tracking_integrity_fix.py"
    dotenv_values = _load_dotenv(repo_root / "tests" / "td_case2" / ".env")
    logs_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env_overrides = {
        "TD_CASE2_RUN_DIR": str(run_dir),
        "TD_CASE2_VIDEO_PATH": str(video_path),
        "TD_CASE2_CAMERA_ID": camera_id,
        "TD_CASE2_CAMERA_GROUP": camera_group,
        "TD_CASE2_CAMERA_TIMEZONE": camera_timezone,
    }
    if dotenv_values.get("TD_CASE2_YOLO_MODEL_PATH"):
        env_overrides["TD_CASE2_YOLO_MODEL_PATH"] = dotenv_values["TD_CASE2_YOLO_MODEL_PATH"]
    env.update(env_overrides)

    hybrid_stdout = logs_dir / "hybrid_stdout.log"
    hybrid_stderr = logs_dir / "hybrid_stderr.log"
    hybrid_command = [str(python_exe), str(hybrid_script), "--video-path", str(video_path), "--run-dir", str(run_dir)]
    tracking_cmd = PipelineCommand(
        label="hybrid_tracking",
        command=hybrid_command,
        cwd=str(repo_root),
        env_overrides=env_overrides,
        stdout_path=str(hybrid_stdout),
        stderr_path=str(hybrid_stderr),
    )
    started = time.perf_counter()
    with hybrid_stdout.open("w", encoding="utf-8") as stdout_handle, hybrid_stderr.open("w", encoding="utf-8") as stderr_handle:
        result = subprocess.run(
            hybrid_command,
            cwd=str(repo_root),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            check=False,
        )
    tracking_cmd.exit_code = int(result.returncode)
    tracking_cmd.finished_at = datetime_now_iso()
    if result.returncode != 0:
        raise RuntimeError(f"Hybrid tracking failed with exit code {result.returncode}. See {hybrid_stderr}")

    post_stdout = logs_dir / "hybrid_post_stdout.log"
    post_stderr = logs_dir / "hybrid_post_stderr.log"
    post_command = [str(python_exe), str(post_script), "--run-dir", str(run_dir), "--camera-id", camera_id, "--camera-group", camera_group, "--camera-timezone", camera_timezone]
    post_cmd = PipelineCommand(
        label="hybrid_post_tracking",
        command=post_command,
        cwd=str(repo_root),
        env_overrides=env_overrides,
        stdout_path=str(post_stdout),
        stderr_path=str(post_stderr),
    )
    post_started = time.perf_counter()
    with post_stdout.open("w", encoding="utf-8") as stdout_handle, post_stderr.open("w", encoding="utf-8") as stderr_handle:
        result = subprocess.run(
            post_command,
            cwd=str(repo_root),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            check=False,
        )
    post_cmd.exit_code = int(result.returncode)
    post_cmd.finished_at = datetime_now_iso()
    if result.returncode != 0:
        raise RuntimeError(f"Hybrid post-tracking failed with exit code {result.returncode}. See {post_stderr}")

    total_runtime = time.perf_counter() - started
    post_runtime = time.perf_counter() - post_started
    metrics = load_hybrid_outputs(run_dir)
    metrics["adapter_total_runtime_seconds"] = total_runtime
    metrics["adapter_post_runtime_seconds"] = post_runtime
    tracking_runtime = float(metrics["tracking_report"]["processing_speed"]["total_runtime_seconds"])
    return PipelineArtifacts(
        run_dir=str(run_dir),
        tracking_runtime_seconds=tracking_runtime,
        post_processing_runtime_seconds=post_runtime,
        total_runtime_seconds=total_runtime,
        tracker_name="yolo_plus_kcf_hybrid",
        output_files=metrics["output_files"],
        metrics=metrics,
        command={
            "tracking": tracking_cmd.to_dict(),
            "post_tracking": post_cmd.to_dict(),
        },
        config_snapshot={"dotenv_values": dotenv_values, "env_overrides": env_overrides},
    )


def datetime_now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).replace(microsecond=0).isoformat()


def load_hybrid_outputs(run_dir: Path) -> dict[str, Any]:
    hybrid_dir = run_dir / "hybrid_tracking_test"
    post_dir = hybrid_dir / "post_tracking_v2"
    tracking_report = _read_json(hybrid_dir / "04c_hybrid_tracking_report.json")
    track_summary = _read_json(hybrid_dir / "04c_hybrid_track_summary.json")
    frame_metrics = _read_json(hybrid_dir / "04c_hybrid_frame_metrics.json")
    failures = _read_json(hybrid_dir / "04c_hybrid_failures.json")
    reconciled_tracks = _read_json(post_dir / "04d2_reconciled_tracks.json")
    quality_report = _read_json(post_dir / "04d2_track_quality_report.json")
    merge_events = _read_json(post_dir / "04d2_track_merge_events.json")
    candidates = _read_json(post_dir / "04d2_reconciliation_candidates.json")
    representative_report = _read_json(post_dir / "05v2_representative_frames_report.json")
    crop_failures = _read_json(post_dir / "05v2_crop_failures.json")
    invalid_crop_candidates = _read_json(post_dir / "05v2_invalid_crop_candidates.json")
    package_report = _read_json(post_dir / "05v2_local_identity_package_report.json")
    packages = _read_json(post_dir / "05v2_local_identity_packages.json")

    manual_review_summary = {}
    manual_review_progress = {}
    manual_review_dir = post_dir / "manual_review"
    if (manual_review_dir / "manual_review_summary.json").exists():
        manual_review_summary = _read_json(manual_review_dir / "manual_review_summary.json")
    if (manual_review_dir / "manual_review_progress.json").exists():
        manual_review_progress = _read_json(manual_review_dir / "manual_review_progress.json")
    if not manual_review_summary or not manual_review_progress:
        fallback_dir = run_dir.parents[1] / "hybrid_test_run_v2" / "hybrid_tracking_test" / "post_tracking_v2" / "manual_review"
        if (fallback_dir / "manual_review_summary.json").exists() and not manual_review_summary:
            manual_review_summary = _read_json(fallback_dir / "manual_review_summary.json")
        if (fallback_dir / "manual_review_progress.json").exists() and not manual_review_progress:
            manual_review_progress = _read_json(fallback_dir / "manual_review_progress.json")

    output_files = {
        "tracking_report": str(hybrid_dir / "04c_hybrid_tracking_report.json"),
        "track_summary": str(hybrid_dir / "04c_hybrid_track_summary.json"),
        "frame_metrics": str(hybrid_dir / "04c_hybrid_frame_metrics.json"),
        "reconciled_tracks": str(post_dir / "04d2_reconciled_tracks.json"),
        "quality_report": str(post_dir / "04d2_track_quality_report.json"),
        "merge_events": str(post_dir / "04d2_track_merge_events.json"),
        "representative_report": str(post_dir / "05v2_representative_frames_report.json"),
        "package_report": str(post_dir / "05v2_local_identity_package_report.json"),
    }
    return {
        "tracking_report": tracking_report,
        "track_summary": track_summary,
        "frame_metrics": frame_metrics,
        "failures": failures,
        "reconciled_tracks": reconciled_tracks,
        "quality_report": quality_report,
        "merge_events": merge_events,
        "candidates": candidates,
        "representative_report": representative_report,
        "crop_failures": crop_failures,
        "invalid_crop_candidates": invalid_crop_candidates,
        "package_report": package_report,
        "packages": packages,
        "manual_review_summary": manual_review_summary,
        "manual_review_progress": manual_review_progress,
        "output_files": output_files,
    }
