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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_subdir(root: Path, previous: set[str]) -> Path:
    candidates = [item for item in root.iterdir() if item.is_dir() and item.name not in previous]
    if not candidates:
        raise FileNotFoundError(f"No new run directory was created under {root}")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def run_td_case2_pipeline(
    *,
    repo_root: Path,
    video_path: Path,
    output_root: Path,
    logs_dir: Path,
) -> PipelineArtifacts:
    python_exe = repo_root / "tests" / "td_case2" / ".venv" / "Scripts" / "python.exe"
    script_path = repo_root / "tests" / "td_case2" / "run_td_case2_search_ready_pipeline.py"
    dotenv_values = _load_dotenv(repo_root / "tests" / "td_case2" / ".env")
    env = os.environ.copy()
    env_overrides = {
        "TD_CASE2_INPUT_VIDEO": str(video_path),
        "TD_CASE2_OUTPUT_ROOT": str(output_root),
    }
    for key, value in dotenv_values.items():
        if value and key not in env_overrides:
            env_overrides[key] = value
    env.update(env_overrides)

    command = [str(python_exe), str(script_path)]
    stdout_path = logs_dir / "td_case2_stdout.log"
    stderr_path = logs_dir / "td_case2_stderr.log"
    previous = {item.name for item in output_root.iterdir() if item.is_dir()} if output_root.exists() else set()
    logs_dir.mkdir(parents=True, exist_ok=True)
    pipeline_command = PipelineCommand(
        label="td_case2",
        command=command,
        cwd=str(repo_root),
        env_overrides=env_overrides,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        result = subprocess.run(
            command,
            cwd=str(repo_root),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            check=False,
        )
    pipeline_command.finished_at = datetime_now_iso()
    pipeline_command.exit_code = int(result.returncode)
    if result.returncode != 0:
        raise RuntimeError(f"td_case2 pipeline failed with exit code {result.returncode}. See {stderr_path}")
    run_dir = _latest_subdir(output_root, previous)
    total_runtime = time.perf_counter() - started
    metrics = load_td_case2_outputs(run_dir)
    metrics["adapter_total_runtime_seconds"] = total_runtime
    tracking_runtime = metrics["timings"].get("tracking_runtime_seconds")
    crop_runtime = metrics["timings"].get("step05_runtime_seconds")
    post_runtime = None
    if tracking_runtime is not None and crop_runtime is not None:
        post_runtime = crop_runtime
    return PipelineArtifacts(
        run_dir=str(run_dir),
        tracking_runtime_seconds=tracking_runtime,
        post_processing_runtime_seconds=post_runtime,
        total_runtime_seconds=total_runtime,
        tracker_name="td_case2_step04b_tracker",
        output_files=metrics["output_files"],
        metrics=metrics,
        command=pipeline_command.to_dict(),
        config_snapshot={"dotenv_values": dotenv_values, "env_overrides": env_overrides},
    )


def datetime_now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).replace(microsecond=0).isoformat()


def load_td_case2_outputs(run_dir: Path) -> dict[str, Any]:
    stage_gate = _read_json(run_dir / "00_stage_gate_report.json")
    video_info = _read_json(run_dir / "01_video_info.json")
    sampled_frames = _read_json(run_dir / "02_sampled_frames.json")
    yolo_detections = _read_json(run_dir / "03_yolo_detections.json")
    yolo_report = _read_json(run_dir / "03_yolo_detection_report.json")
    tracks_payload = _read_json(run_dir / "04B_tracks.json")
    tracking_report = _read_json(run_dir / "04B_tracking_report.json")
    tracking_quality = _read_json(run_dir / "04B_tracking_quality_report.json")
    best_frames = _read_json(run_dir / "05_best_track_frames.json")
    best_frames_report = _read_json(run_dir / "05_best_track_frames_report.json")
    step06_report = _read_json(run_dir / "06_ocr_color_report.json")
    step07_report = _read_json(run_dir / "07B_traffic_object_search_index_report.json")

    processed_frames = int(stage_gate.get("steps", {}).get("03B_yolo_detection", {}).get("frames_processed", yolo_report.get("frames_processed", 0)) or 0)
    source_duration = float(video_info.get("duration_seconds", 0.0) or 0.0)
    step03_runtime = yolo_report.get("runtime_seconds")
    tracking_runtime = tracking_report.get("runtime_seconds")
    step05_runtime = best_frames_report.get("total_processing_seconds") or best_frames_report.get("runtime_seconds")
    step06_runtime = step06_report.get("total_processing_seconds") or step06_report.get("runtime_seconds")
    step07_runtime = step07_report.get("runtime_seconds")

    confirmed_tracks = list(tracks_payload.get("usable_tracks_for_next_step", []))
    all_tracks = list(tracks_payload.get("tracks", []))
    class_counts = dict(tracking_report.get("dominant_class_track_counts", {}))
    output_files = {
        "video_info": str(run_dir / "01_video_info.json"),
        "sampled_frames": str(run_dir / "02_sampled_frames.json"),
        "yolo_detections": str(run_dir / "03_yolo_detections.json"),
        "tracking_report": str(run_dir / "04B_tracking_report.json"),
        "tracking_quality": str(run_dir / "04B_tracking_quality_report.json"),
        "best_frames": str(run_dir / "05_best_track_frames.json"),
        "best_frames_report": str(run_dir / "05_best_track_frames_report.json"),
        "ocr_color_report": str(run_dir / "06_ocr_color_report.json"),
        "search_index_report": str(run_dir / "07B_traffic_object_search_index_report.json"),
    }
    timings = {
        "step03_runtime_seconds": step03_runtime,
        "tracking_runtime_seconds": tracking_runtime,
        "step05_runtime_seconds": step05_runtime,
        "step06_runtime_seconds": step06_runtime,
        "step07_runtime_seconds": step07_runtime,
    }
    total_known_runtime = sum(float(item or 0.0) for item in timings.values())
    return {
        "video_info": video_info,
        "processed_frames": processed_frames,
        "timings": timings,
        "known_stage_runtime_seconds": total_known_runtime,
        "source_duration_seconds": source_duration,
        "source_frame_count": int(video_info.get("frame_count", 0) or 0),
        "yolo_calls": _derive_td_case2_yolo_calls(yolo_detections, yolo_report),
        "raw_tracks": len(all_tracks),
        "confirmed_tracks": "not_available",
        "reconciled_objects": None,
        "vehicles": int(tracking_report.get("track_type_counts", {}).get("vehicle", 0) or 0),
        "persons": int(tracking_report.get("track_type_counts", {}).get("person", 0) or 0),
        "short_tracks_under_0_5_seconds": _count_tracks_shorter_than(all_tracks, 0.5),
        "short_tracks_under_1_0_second": _count_tracks_shorter_than(all_tracks, 1.0),
        "track_duration_stats": dict(tracking_report.get("track_duration_stats", {})),
        "track_quality_counts": dict(tracking_report.get("track_quality_counts", {})),
        "class_counts": class_counts,
        "representative_crops": int(best_frames_report.get("selected_crops_saved", 0) or 0),
        "tracks_with_primary_crop": int(best_frames_report.get("selected_track_count", 0) or 0),
        "tracks_with_three_representative_crops": _count_tracks_with_three(best_frames),
        "objects_with_full_scene_frame": int(best_frames_report.get("selected_full_frames_saved", 0) or 0),
        "fallback_crop_count": int(best_frames_report.get("fallback_selected_detections", 0) or 0),
        "invalid_crop_candidates": 0,
        "crop_failures": int(best_frames_report.get("missing_crop_count", 0) or 0),
        "yolo_selected_crops": _count_yolo_selected_crops(best_frames),
        "kcf_selected_crops": 0,
        "plate_candidates": int(step06_report.get("tracks_with_plate_candidates", 0) or 0),
        "warnings": tracking_quality.get("top_bad_tracks_by_reason", []),
        "failures": [],
        "tracker_name": "td_case2_step04b_tracker",
        "stage_gate": stage_gate,
        "tracking_report": tracking_report,
        "tracking_quality": tracking_quality,
        "best_frames": best_frames,
        "best_frames_report": best_frames_report,
        "step06_report": step06_report,
        "step07_report": step07_report,
        "output_files": output_files,
    }


def _derive_td_case2_yolo_calls(yolo_detections: dict[str, Any], yolo_report: dict[str, Any]) -> int | str:
    explicit = yolo_report.get("frames_processed")
    frame_records = list(yolo_detections.get("frames", []))
    if isinstance(explicit, int) and explicit > 0 and frame_records:
        return explicit
    if frame_records:
        return len(frame_records)
    return "not_available"


def _count_tracks_shorter_than(tracks: list[dict[str, Any]], seconds: float) -> int:
    count = 0
    for item in tracks:
        duration = float(item.get("duration_seconds", 0.0) or 0.0)
        if duration < float(seconds):
            count += 1
    return count


def _count_tracks_with_three(best_frames: dict[str, Any]) -> int:
    count = 0
    for item in best_frames.get("tracks", []):
        if len(list(item.get("selected_detections", []))) >= 3:
            count += 1
    return count


def _count_yolo_selected_crops(best_frames: dict[str, Any]) -> int:
    count = 0
    for item in best_frames.get("tracks", []):
        for detection in item.get("selected_detections", []):
            if str(detection.get("bbox_source", "yolo")) == "yolo":
                count += 1
    return count
