from __future__ import annotations

import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency in UI only
    cv2 = None


SUPPORTED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm", "m4v"}
STANDARD_STAGE_WEIGHTS = {
    1: 3,
    2: 5,
    "02B": 5,
    "02C": 3,
    3: 5,
    4: 3,
    5: 4,
    6: 3,
    7: 6,
    10: 15,
    11: 8,
    "11B": 4,
    12: 4,
    13: 4,
    14: 3,
    "14B": 3,
    15: 5,
    "15B": 2,
    16: 25,
    "16B": 12,
    17: 4,
    18: 6,
    19: 2,
}
STANDARD_STAGE_LABELS = {
    1: "Reading video information",
    2: "Sampling frames",
    "02B": "Adaptive sampling",
    "02C": "Building frame candidate pool",
    3: "Scoring motion",
    4: "Selecting motion candidates",
    5: "Grouping motion into clips",
    6: "Expanding clips with context",
    7: "Creating VLM temporal strips",
    10: "Running YOLO detection",
    11: "Scoring YOLO object evidence",
    "11B": "Estimating object motion state",
    12: "Fusing motion + YOLO + VLM evidence",
    13: "Ranking candidate clips",
    14: "Selecting Top-K + guardrail clips",
    "14B": "Applying incident coverage guardrails",
    15: "Creating Top-K VLM inputs",
    "15B": "Auditing VLM coverage",
    16: "Running Qwen on Top-K clips",
    "16B": "Incident recheck reasoning",
    17: "Creating final summary",
    18: "Exporting/compiling review video",
    19: "Creating HTML demo report",
}
STANDARD_STAGE_PROGRESS_PERCENT = {
    1: 3,
    2: 8,
    "02B": 12,
    "02C": 15,
    3: 13,
    4: 18,
    5: 22,
    6: 26,
    7: 31,
    10: 45,
    11: 55,
    "11B": 58,
    12: 60,
    13: 66,
    14: 70,
    "14B": 73,
    15: 75,
    "15B": 78,
    16: 85,
    "16B": 90,
    17: 92,
    18: 97,
    19: 99,
}
FAST_STAGE_WEIGHTS = {
    1: 5,
    2: 5,
    "02B": 5,
    "02C": 3,
    3: 5,
    4: 5,
    56: 35,
    13: 7,
    14: 6,
    "14B": 4,
    15: 5,
    "15B": 2,
    16: 15,
    "16B": 10,
    17: 5,
    18: 4,
    19: 3,
}
FAST_STAGE_PROGRESS_PERCENT = {
    1: 5,
    2: 10,
    "02B": 13,
    "02C": 16,
    3: 15,
    4: 20,
    56: 55,
    13: 62,
    14: 68,
    "14B": 71,
    15: 73,
    "15B": 77,
    16: 88,
    "16B": 91,
    17: 93,
    18: 97,
    19: 99,
}
PIPELINE_ENGINES = [
    "Fast parallel Top-K pipeline",
    "Standard demo pipeline",
]
PIPELINE_ENGINE_MAP = {
    "Fast parallel Top-K pipeline": {
        "engine_id": "fast_parallel_topk",
        "script_path": "tests/tender_demo_case/run_tender_demo_fast_parallel_pipeline.py",
        "description": "Runs optimized Top-K flow. Skips old full-VLM path and uses parallel clip/YOLO branches.",
    },
    "Standard demo pipeline": {
        "engine_id": "standard_demo",
        "script_path": "tests/tender_demo_case/run_tender_demo_pipeline.py",
        "description": "Runs the existing complete demo flow. Best for compatibility, slower.",
    },
}
PROCESSING_PRESETS = {
    "Quick scan": {
        "sample_every_seconds": 4.0,
        "top_k": 3,
        "qwen_max_new_tokens": 192,
        "qwen_batch_size": 1,
        "yolo_imgsz": 416,
        "yolo_conf": 0.40,
        "motion_threshold": 0.25,
        "parallel_branches": True,
        "enable_incident_recheck": False,
        "incident_recheck_all_topk": False,
        "incident_fallback_pass": False,
        "incident_focus": "general",
        "adaptive_sampling_enabled": False,
        "adaptive_base_interval_seconds": 1.0,
        "adaptive_max_frame_gap_seconds": 4.0,
        "coverage_guardrails_enabled": False,
        "critical_timestamps": "",
        "critical_window_seconds": 8.0,
        "vlm_input_strategy": "center_only",
        "max_vlm_inputs": 25,
        "yolo_input_scope": "motion_candidates",
    },
    "Fast demo": {
        "sample_every_seconds": 3.0,
        "top_k": 5,
        "qwen_max_new_tokens": 256,
        "qwen_batch_size": 1,
        "yolo_imgsz": 416,
        "yolo_conf": 0.35,
        "motion_threshold": 0.20,
        "parallel_branches": True,
        "enable_incident_recheck": False,
        "incident_recheck_all_topk": False,
        "incident_fallback_pass": False,
        "incident_focus": "general",
        "adaptive_sampling_enabled": False,
        "adaptive_base_interval_seconds": 1.0,
        "adaptive_max_frame_gap_seconds": 4.0,
        "coverage_guardrails_enabled": False,
        "critical_timestamps": "",
        "critical_window_seconds": 8.0,
        "vlm_input_strategy": "center_only",
        "max_vlm_inputs": 25,
        "yolo_input_scope": "motion_candidates",
    },
    "Balanced": {
        "sample_every_seconds": 2.0,
        "top_k": 8,
        "qwen_max_new_tokens": 384,
        "qwen_batch_size": 1,
        "yolo_imgsz": 416,
        "yolo_conf": 0.35,
        "motion_threshold": 0.15,
        "parallel_branches": True,
        "enable_incident_recheck": True,
        "incident_recheck_all_topk": False,
        "incident_fallback_pass": False,
        "incident_focus": "general",
        "adaptive_sampling_enabled": True,
        "adaptive_base_interval_seconds": 1.0,
        "adaptive_max_frame_gap_seconds": 4.0,
        "coverage_guardrails_enabled": True,
        "critical_timestamps": "",
        "critical_window_seconds": 8.0,
        "vlm_input_strategy": "multi_focus",
        "max_vlm_inputs": 8,
        "yolo_input_scope": "frame_candidate_pool",
    },
    "Jewelry shop robbery demo": {
        "sample_every_seconds": 1.0,
        "top_k": 8,
        "qwen_max_new_tokens": 384,
        "qwen_batch_size": 1,
        "yolo_imgsz": 416,
        "yolo_conf": 0.35,
        "motion_threshold": 0.15,
        "parallel_branches": True,
        "enable_incident_recheck": True,
        "incident_recheck_all_topk": False,
        "incident_fallback_pass": False,
        "incident_focus": "theft",
        "adaptive_sampling_enabled": True,
        "adaptive_base_interval_seconds": 1.0,
        "adaptive_max_frame_gap_seconds": 4.0,
        "coverage_guardrails_enabled": True,
        "critical_timestamps": "",
        "critical_window_seconds": 8.0,
        "vlm_input_strategy": "multi_focus",
        "max_vlm_inputs": 8,
        "yolo_input_scope": "frame_candidate_pool",
    },
    "Sensitive Incident Review": {
        "sample_every_seconds": 1.0,
        "top_k": 20,
        "qwen_max_new_tokens": 512,
        "qwen_batch_size": 1,
        "yolo_imgsz": 640,
        "yolo_conf": 0.25,
        "motion_threshold": 0.10,
        "parallel_branches": True,
        "enable_incident_recheck": True,
        "incident_recheck_all_topk": True,
        "incident_fallback_pass": True,
        "incident_focus": "general",
        "adaptive_sampling_enabled": True,
        "adaptive_base_interval_seconds": 1.0,
        "adaptive_max_frame_gap_seconds": 4.0,
        "coverage_guardrails_enabled": True,
        "critical_timestamps": "",
        "critical_window_seconds": 8.0,
        "vlm_input_strategy": "multi_focus",
        "max_vlm_inputs": 40,
        "yolo_input_scope": "frame_candidate_pool",
    },
    "High accuracy review": {
        "sample_every_seconds": 0.5,
        "top_k": 25,
        "qwen_max_new_tokens": 512,
        "qwen_batch_size": 1,
        "yolo_imgsz": 640,
        "yolo_conf": 0.20,
        "motion_threshold": 0.08,
        "parallel_branches": True,
        "enable_incident_recheck": True,
        "incident_recheck_all_topk": True,
        "incident_fallback_pass": True,
        "incident_focus": "general",
        "adaptive_sampling_enabled": True,
        "adaptive_base_interval_seconds": 0.5,
        "adaptive_max_frame_gap_seconds": 3.0,
        "coverage_guardrails_enabled": True,
        "critical_timestamps": "",
        "critical_window_seconds": 8.0,
        "vlm_input_strategy": "multi_focus",
        "max_vlm_inputs": 50,
        "yolo_input_scope": "frame_candidate_pool",
    },
    "Custom": {
        "sample_every_seconds": 2.0,
        "top_k": 8,
        "qwen_max_new_tokens": 384,
        "qwen_batch_size": 1,
        "yolo_imgsz": 416,
        "yolo_conf": 0.35,
        "motion_threshold": 0.15,
        "parallel_branches": True,
        "enable_incident_recheck": True,
        "incident_recheck_all_topk": False,
        "incident_fallback_pass": False,
        "incident_focus": "general",
        "adaptive_sampling_enabled": True,
        "adaptive_base_interval_seconds": 1.0,
        "adaptive_max_frame_gap_seconds": 4.0,
        "coverage_guardrails_enabled": True,
        "critical_timestamps": "",
        "critical_window_seconds": 8.0,
        "vlm_input_strategy": "multi_focus",
        "max_vlm_inputs": 8,
        "yolo_input_scope": "frame_candidate_pool",
    },
}
INCIDENT_FOCUS_OPTIONS = {
    "General incident review": "general",
    "Robbery / weapon / assault": "robbery",
    "Theft / shoplifting": "theft",
    "Traffic / collision": "traffic",
    "Fall / injury": "fall",
    "Intrusion / restricted area": "intrusion",
}
INPUT_MODE_OPTIONS = [
    "Use existing local/server video path",
    "Upload video file",
    "Select from import folder",
]
QUICK_RESULT_SETTINGS = {
    "sample_every_seconds": 4.0,
    "top_k": 3,
    "qwen_max_new_tokens": 192,
    "yolo_imgsz": 416,
    "yolo_conf": 0.40,
    "motion_threshold": 0.25,
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def safe_filename(name: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*]+', "_", str(name or "").strip())
    sanitized = re.sub(r"\s+", "_", sanitized)
    sanitized = sanitized.strip("._")
    return sanitized or "uploaded_video"


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def resolve_media_path(run_dir: Path, path_value: str | None) -> Path | None:
    if not path_value:
        return None

    raw_value = str(path_value).strip()
    path = Path(raw_value)
    if path.is_absolute():
        return path if path.exists() else None

    run_candidate = run_dir / path
    if run_candidate.exists():
        return run_candidate

    root = project_root()
    root_candidate = root / path
    if root_candidate.exists():
        return root_candidate

    marker = f"tests/tender_demo_case/debug_runs/{run_dir.name}/"
    normalized = raw_value.replace("\\", "/")
    if marker in normalized:
        relative_part = normalized.split(marker, 1)[1]
        run_marker_candidate = run_dir / relative_part
        if run_marker_candidate.exists():
            return run_marker_candidate

    return None


def find_ffmpeg() -> str | None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    fallback_candidates = [
        Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe",
    ]
    for candidate in fallback_candidates:
        try:
            if candidate.exists():
                return str(candidate)
        except PermissionError:
            return str(candidate)
    return None


def convert_avi_to_browser_mp4(avi_path: Path, web_mp4_path: Path) -> tuple[bool, str]:
    ffmpeg_path = find_ffmpeg()
    if ffmpeg_path is None:
        return False, "FFmpeg not found. Install FFmpeg and restart Streamlit."
    completed = subprocess.run(
        [
            ffmpeg_path,
            "-y",
            "-i",
            str(avi_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(web_mp4_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        error_text = (completed.stderr or completed.stdout or "").strip() or f"ffmpeg exited with code {completed.returncode}"
        return False, error_text
    return True, "Browser MP4 created."


def media_exists(path: Path | None) -> bool:
    return bool(path and path.exists() and path.is_file())


def get_video_duration_seconds(video_path: Path) -> float | None:
    if cv2 is None or not video_path.exists():
        return None
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return None
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        if fps <= 0 or frame_count <= 0:
            return None
        return frame_count / fps
    finally:
        capture.release()


def format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "unknown"
    total_seconds = int(max(0, round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def format_file_size(num_bytes: int | float | None) -> str:
    try:
        size = float(num_bytes or 0)
    except (TypeError, ValueError):
        return "unknown"
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    return f"{size:.1f}{units[unit_index]}"


def find_latest_debug_run() -> Path | None:
    debug_runs_dir = project_root() / "tests" / "tender_demo_case" / "debug_runs"
    if not debug_runs_dir.exists():
        return None
    candidates = [path for path in debug_runs_dir.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def get_active_run_dir() -> Path | None:
    value = st.session_state.get("active_run_dir", "")
    if not value:
        return None
    path = Path(value)
    if not path.exists() or not path.is_dir():
        return None
    return path


def save_uploaded_video(uploaded_file) -> Path:
    uploads_root = project_root() / "tests" / "tender_demo_case" / "ui_uploads"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = safe_filename(uploaded_file.name)
    safe_stem = safe_filename(Path(uploaded_file.name).stem)
    target_dir = uploads_root / f"{timestamp}_{safe_stem}"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / safe_name
    target_path.write_bytes(uploaded_file.getbuffer())
    return target_path


def import_folder_path() -> Path:
    folder = project_root() / "tests" / "tender_demo_case" / "video_imports"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def list_import_folder_videos() -> list[Path]:
    folder = import_folder_path()
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower().lstrip(".") in SUPPORTED_EXTENSIONS
        ],
        key=lambda path: path.name.lower(),
    )


def build_pipeline_env(settings: dict, selected_video_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env_updates = {
        "TENDER_DEMO_INPUT_VIDEO": str(selected_video_path),
        "TENDER_DEMO_PIPELINE_ENGINE": str(settings["pipeline_engine_id"]),
        "TENDER_DEMO_ANALYSIS_SENSITIVITY_MODE": str(settings["analysis_sensitivity_mode"]),
        "TENDER_DEMO_SAMPLE_EVERY_SECONDS": str(settings["sample_every_seconds"]),
        "TENDER_DEMO_TOP_K_CLIPS": str(settings["top_k"]),
        "TENDER_DEMO_TOP_K_MAX_CLIPS": str(settings["top_k_max"]),
        "TENDER_DEMO_MOTION_THRESHOLD": str(settings["motion_threshold"]),
        "TENDER_DEMO_QWEN_MODEL_ID": str(settings["qwen_model_id"]),
        "TENDER_DEMO_QWEN_BATCH_SIZE": str(settings["qwen_batch_size"]),
        "TENDER_DEMO_QWEN_MAX_NEW_TOKENS": str(settings["qwen_max_new_tokens"]),
        "TENDER_DEMO_RUN_YOLO": "true" if settings["run_yolo"] else "false",
        "TENDER_DEMO_YOLO_MODEL": str(settings["yolo_model"]),
        "TENDER_DEMO_YOLO_CONF": str(settings["yolo_conf"]),
        "TENDER_DEMO_YOLO_IMGSZ": str(settings["yolo_imgsz"]),
        "TENDER_DEMO_FAST_PARALLEL_BRANCHES": "true" if settings["parallel_branches"] else "false",
        "TENDER_DEMO_QUICK_RESULT_MODE": "true" if settings.get("quick_result_mode") else "false",
        "TENDER_DEMO_ENABLE_INCIDENT_RECHECK": "true" if settings.get("enable_incident_recheck") else "false",
        "TENDER_DEMO_INCIDENT_RECHECK_ALL_TOPK": "true" if settings.get("incident_recheck_all_topk") else "false",
        "TENDER_DEMO_INCIDENT_FALLBACK_PASS": "true" if settings.get("incident_fallback_pass") else "false",
        "TENDER_DEMO_INCIDENT_FOCUS": str(settings.get("incident_focus", "general")),
        "TENDER_DEMO_ENABLE_ADAPTIVE_SAMPLING": "true" if settings.get("adaptive_sampling_enabled") else "false",
        "TENDER_DEMO_ADAPTIVE_BASE_INTERVAL_SECONDS": str(settings.get("adaptive_base_interval_seconds", 1.0)),
        "TENDER_DEMO_ADAPTIVE_MAX_FRAME_GAP_SECONDS": str(settings.get("adaptive_max_frame_gap_seconds", 4.0)),
        "TENDER_DEMO_ENABLE_COVERAGE_GUARDRAILS": "true" if settings.get("coverage_guardrails_enabled") else "false",
        "TENDER_DEMO_CRITICAL_TIMESTAMPS": str(settings.get("critical_timestamps", "")),
        "TENDER_DEMO_CRITICAL_WINDOW_SECONDS": str(settings.get("critical_window_seconds", 8.0)),
        "TENDER_DEMO_VLM_INPUT_STRATEGY": str(settings.get("vlm_input_strategy", "center_only")),
        "TENDER_DEMO_MAX_VLM_INPUTS": str(settings.get("max_vlm_inputs", 25)),
        "TENDER_DEMO_YOLO_INPUT_SCOPE": str(settings.get("yolo_input_scope", "motion_candidates")),
        "TENDER_DEMO_CREATE_COMPILED_REVIEW_VIDEO": "true" if settings["create_compiled_review_video"] else "false",
        "TENDER_DEMO_COMPILED_VIDEO_FPS": str(settings["compiled_video_fps"]),
        "TENDER_DEMO_SECONDS_PER_FRAME": str(settings["seconds_per_frame"]),
        "TENDER_DEMO_SECONDS_PER_TITLE_CARD": str(settings["seconds_per_title_card"]),
    }
    max_video_seconds = str(settings.get("max_video_seconds", "")).strip()
    if max_video_seconds:
        env_updates["TENDER_DEMO_MAX_VIDEO_SECONDS"] = max_video_seconds
    env.update(env_updates)
    return env


def stop_pipeline_process(pid: int) -> bool:
    try:
        if os.name == "nt":
            completed = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
                check=False,
            )
            return completed.returncode == 0
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        return True
    except Exception:
        return False


def detect_stage_from_line(line: str, pipeline_engine: str) -> tuple[str | None, int | None]:
    standard_stage_map = [
        (["Starting Step 1", "01_video_info.json"], "Reading video information", 5),
        (["Starting Step 2", "02_sampled_frames.json"], "Sampling frames", 10),
        (["Starting Step 02B", "02b_adaptive_sampling_report.json"], "Adaptive sampling", 13),
        (["Starting Step 02C", "02c_frame_candidate_pool.json"], "Building frame candidate pool", 16),
        (["Starting Step 3", "03_motion_scores.json"], "Scoring motion", 15),
        (["Starting Step 4", "04_motion_candidates.json"], "Selecting motion candidates", 20),
        (["Starting Step 5", "05_candidate_clips.json"], "Grouping candidate clips", 25),
        (["Starting Step 6", "06_expanded_clips.json"], "Expanding clips", 30),
        (["Starting Step 10", "10_yolo_detections.json"], "Running YOLO object detection", 45),
        (["Starting Step 11", "11_yolo_object_scores.json"], "Scoring YOLO evidence", 55),
        (["Starting Step 11B", "11b_object_motion_states.json"], "Estimating object motion state", 58),
        (["Starting Step 13", "13_ranked_clips.json"], "Ranking candidate clips", 62),
        (["Starting Step 14", "14_selected_top_clips.json"], "Selecting Top-K + guardrail clips", 68),
        (["Starting Step 14B", "14b_coverage_selected_clips.json"], "Applying incident coverage guardrails", 71),
        (["Starting Step 15", "15_topk_vlm_inputs.json"], "Creating Top-K VLM inputs", 73),
        (["15_vlm_coverage_audit.json"], "Auditing VLM coverage", 77),
        (["Starting Step 16", "16_topk_vlm_outputs.json"], "Running Qwen on selected clips", 88),
        (["Starting Step 16B", "16b_incident_recheck_outputs.json"], "Incident recheck reasoning", 90),
        (["Starting Step 17", "17_topk_final_summary.json", "17_topk_final_summary.md"], "Creating final summary", 93),
        (["Starting Step 18", "18_compiled_review_video.json", "18_exported_clips.json"], "Creating compiled review video", 97),
        (["Starting Step 19", "19_demo_report.html"], "Creating HTML report", 99),
    ]
    fast_stage_map = [
        (["Starting Step 1", "01_video_info.json"], "Reading video information", 5),
        (["Starting Step 2", "02_sampled_frames.json"], "Sampling frames", 10),
        (["Starting Step 02B", "02b_adaptive_sampling_report.json"], "Adaptive sampling", 13),
        (["Starting Step 02C", "02c_frame_candidate_pool.json"], "Building frame candidate pool", 16),
        (["Starting Step 3", "03_motion_scores.json"], "Scoring motion", 15),
        (["Starting Step 4", "04_motion_candidates.json"], "Selecting motion candidates", 20),
        (["Starting Step 5", "05_candidate_clips.json"], "Grouping candidate clips", 25),
        (["Starting Step 6", "06_expanded_clips.json"], "Expanding clips", 30),
        (["Starting parallel section", "Starting clip branch", "Starting YOLO branch"], "Building clips and running YOLO evidence", 55),
        (["Starting Step 10", "10_yolo_detections.json"], "Running YOLO object detection", 45),
        (["Starting Step 11", "11_yolo_object_scores.json"], "Scoring YOLO evidence", 55),
        (["Starting Step 11B", "11b_object_motion_states.json"], "Estimating object motion state", 58),
        (["Starting Step 13", "13_ranked_clips.json"], "Ranking candidate clips", 62),
        (["Starting Step 14", "14_selected_top_clips.json"], "Selecting Top-K + guardrail clips", 68),
        (["Starting Step 14B", "14b_coverage_selected_clips.json"], "Applying incident coverage guardrails", 71),
        (["Starting Step 15", "15_topk_vlm_inputs.json"], "Creating Top-K VLM inputs", 73),
        (["15_vlm_coverage_audit.json"], "Auditing VLM coverage", 77),
        (["Starting Step 16", "16_topk_vlm_outputs.json"], "Running Qwen on selected clips", 88),
        (["Starting Step 16B", "16b_incident_recheck_outputs.json"], "Incident recheck reasoning", 91),
        (["Starting Step 17", "17_topk_final_summary.json", "17_topk_final_summary.md"], "Creating final summary", 93),
        (["Starting Step 18", "18_compiled_review_video.json", "18_exported_clips.json"], "Creating compiled review video", 97),
        (["Starting Step 19", "19_demo_report.html", "Runtime metrics path"], "Creating HTML report", 99),
    ]
    stage_map = standard_stage_map if pipeline_engine == "Standard demo pipeline" else fast_stage_map
    for tokens, label, progress in stage_map:
        if any(token in line for token in tokens):
            return label, progress
    return None, None


def filter_user_friendly_log_line(line: str) -> str | None:
    mappings = {
        "Starting Step 1": "Reading video information...",
        "Starting Step 2": "Sampling frames...",
        "Starting Step 02B": "Running adaptive sampling...",
        "Starting Step 02C": "Building frame candidate pool...",
        "Starting Step 3": "Scoring motion...",
        "Starting Step 4": "Selecting motion candidates...",
        "Starting Step 5": "Grouping motion into clips...",
        "Starting Step 6": "Expanding clips with context...",
        "Starting Step 7": "Creating VLM temporal strips...",
        "Starting Step 10": "Running YOLO object detection...",
        "Starting Step 11": "Scoring YOLO object evidence...",
        "Starting Step 11B": "Estimating object motion state...",
        "Starting Step 12": "Fusing motion + YOLO + VLM evidence...",
        "Starting Step 13": "Ranking candidate clips...",
        "Starting Step 14": "Selecting Top-K + guardrail clips...",
        "Starting Step 14B": "Applying incident coverage guardrails...",
        "Starting Step 15": "Creating Top-K VLM input strips...",
        "15_vlm_coverage_audit.json": "Auditing VLM coverage...",
        "Starting Step 16": "Running Qwen on selected Top-K clips...",
        "Starting Step 16B": "Running incident recheck reasoning...",
        "Starting Step 17": "Creating final summary...",
        "Starting Step 18": "Creating compiled review video...",
        "Starting Step 19": "Creating HTML demo report...",
        "Starting parallel section": "Building clips and running YOLO evidence...",
        "Starting clip branch": "Building clip evidence branch...",
        "Starting YOLO branch": "Building YOLO evidence branch...",
    }
    for marker, clean_message in mappings.items():
        if marker in line:
            return clean_message

    important_terms = [
        "Successful parses",
        "Failed parses",
        "Priority suspicious events",
        "Possible review clips",
        "Incident recheck priority clips",
        "Incident recheck review clips",
        "Rechecked clips",
        "Output path",
        "Compiled video path",
        "HTML report path",
        "Debug run directory",
    ]
    for term in important_terms:
        if term in line:
            return line.strip()
    return None


def find_new_run_dir(before_dirs: set[Path], debug_runs_dir: Path) -> Path | None:
    if not debug_runs_dir.exists():
        return None
    after_dirs = {path for path in debug_runs_dir.iterdir() if path.is_dir()}
    new_dirs = list(after_dirs - before_dirs)
    if new_dirs:
        return max(new_dirs, key=lambda path: path.stat().st_mtime)
    candidates = list(after_dirs)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def run_pipeline_with_live_logs(command, env, cwd, placeholders, stage_weights) -> dict:
    debug_runs_dir = project_root() / "tests" / "tender_demo_case" / "debug_runs"
    before_dirs = {path for path in debug_runs_dir.iterdir() if path.is_dir()} if debug_runs_dir.exists() else set()
    total_weight = sum(stage_weights.values())
    stage_order = list(stage_weights.keys())
    log_lines: list[str] = []
    detected_run_dir: Path | None = None
    current_stage_label = "Waiting to start"
    latest_clean_message = "Preparing pipeline..."
    latest_output_hint = ""
    stage_start_time = time.time()
    start_time = time.time()
    estimated_seconds = max(120.0, float(placeholders["estimated_seconds"]))
    st.session_state["current_stage"] = "Starting pipeline..."
    st.session_state["progress_percent"] = 0

    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0,
    )
    st.session_state["pipeline_process_pid"] = process.pid
    st.session_state["pipeline_running"] = True

    try:
        for raw_line in iter(process.stdout.readline, ""):
            line = raw_line.rstrip()
            log_lines.append(line)
            stage_label, detected_progress = detect_stage_from_line(
                line,
                st.session_state.get("pipeline_engine", PIPELINE_ENGINES[0]),
            )
            if stage_label is not None:
                current_stage_label = stage_label or current_stage_label
                stage_start_time = time.time()
                st.session_state["current_stage"] = current_stage_label
                st.session_state["progress_percent"] = max(
                    int(st.session_state.get("progress_percent", 0)),
                    int(detected_progress or 0),
                )

            if "debug_runs" in line:
                match = re.search(r"([A-Za-z]:\\[^\\r\\n]*debug_runs[^\\r\\n]*)", line)
                if match:
                    candidate = Path(match.group(1).strip())
                    if candidate.exists():
                        detected_run_dir = candidate if candidate.is_dir() else candidate.parent

            clean_message = filter_user_friendly_log_line(line)
            if clean_message:
                if "HTML report path" in line:
                    clean_message = "HTML report created."
                elif "Output path for 17_topk_final_summary.json" in line:
                    clean_message = "Final summary created."
                elif "compiled review video" in line.lower():
                    clean_message = "Compiled review video created."
                latest_clean_message = clean_message
                if "Output path" in clean_message or "Compiled video path" in clean_message or "HTML report path" in clean_message:
                    latest_output_hint = clean_message

            elapsed = time.time() - start_time
            current_progress_percent = max(int(st.session_state.get("progress_percent", 0)), 0)
            progress_value = min(max(current_progress_percent / 100.0, 0.0), 0.99)

            if current_progress_percent >= 99:
                remaining_caption = "finishing..."
            else:
                estimated_remaining_seconds = elapsed * (100 - current_progress_percent) / max(current_progress_percent, 1)
                estimated_remaining_seconds = max(10.0, estimated_remaining_seconds)
                remaining_caption = f"{format_duration(estimated_remaining_seconds)} approximate"

            placeholders["status_placeholder"].info(f"Current stage: {current_stage_label}")
            placeholders["message_placeholder"].success(latest_clean_message)
            display_progress = max(progress_value, current_progress_percent / 100.0)
            placeholders["progress_bar"].progress(min(max(display_progress, 0.0), 1.0))
            placeholders["eta_placeholder"].caption(
                f"Elapsed: {format_duration(elapsed)} | Estimated remaining: {remaining_caption} | "
                f"Progress: {max(int(display_progress * 100), current_progress_percent)}%\n"
                "Estimated time is approximate and depends mainly on video length, GPU speed, YOLO, and Qwen."
            )
            if latest_output_hint:
                placeholders["output_placeholder"].caption(f"Latest output: {latest_output_hint}")
    finally:
        process.wait()
        st.session_state["pipeline_running"] = False
        st.session_state["pipeline_process_pid"] = None

    elapsed_seconds = time.time() - start_time
    if detected_run_dir is None:
        detected_run_dir = find_new_run_dir(before_dirs, debug_runs_dir)

    if process.returncode == 0:
        st.session_state["current_stage"] = "Pipeline completed"
        st.session_state["progress_percent"] = 100
    else:
        st.session_state["current_stage"] = "Pipeline failed"

    return {
        "return_code": process.returncode,
        "logs": "\n".join(log_lines),
        "detected_run_dir": str(detected_run_dir) if detected_run_dir else None,
        "elapsed_seconds": elapsed_seconds,
        "latest_clean_message": latest_clean_message,
    }


def _extract_compiled_video_path(run_dir: Path, results: dict[str, Any]) -> Path | None:
    compiled_info = results.get("compiled_video", {}) if isinstance(results.get("compiled_video"), dict) else {}
    export_info = results.get("exported_clips", {}).get("compiled_review_video", {}) if isinstance(results.get("exported_clips"), dict) else {}
    candidate_values = [
        compiled_info.get("browser_playable_video_path"),
        compiled_info.get("playback_recommended_file"),
        compiled_info.get("compiled_video_path"),
        compiled_info.get("fallback_video_path"),
        export_info.get("browser_playable_video_path"),
        export_info.get("playback_recommended_file"),
        export_info.get("output_path"),
        str(run_dir / "18_exported_clips" / "18_compiled_review_video_web.mp4"),
        str(run_dir / "18_exported_clips" / "18_compiled_review_video.mp4"),
        str(run_dir / "18_exported_clips" / "18_compiled_review_video_fallback.avi"),
    ]
    for compiled_path_value in candidate_values:
        compiled_path = resolve_media_path(run_dir, compiled_path_value)
        if media_exists(compiled_path):
            return compiled_path
    return resolve_media_path(run_dir, next((value for value in candidate_values if value), None))


def _timeline_description(item: dict[str, Any]) -> str:
    for key in ["best_event_description", "caption", "description"]:
        value = str(item.get(key, "")).strip()
        if value:
            return value
    activities = item.get("activity_descriptions", [])
    if isinstance(activities, list) and activities:
        return str(activities[0])
    raw_text = str(item.get("raw_vlm_output", "")).strip()
    if raw_text:
        return raw_text[:220]
    return "No useful description was produced for this clip."


@st.cache_data(show_spinner=False)
def load_all_search_records(debug_runs_dir_str: str, scope: str, current_run_dir_str: str, cache_buster: str = "") -> list[dict]:
    debug_runs_dir = Path(debug_runs_dir_str)
    current_run_dir = Path(current_run_dir_str) if current_run_dir_str else None
    target_runs: list[Path] = []

    if scope == "Current selected run" and current_run_dir and current_run_dir.exists():
        target_runs = [current_run_dir]
    elif debug_runs_dir.exists():
        target_runs = [path for path in debug_runs_dir.iterdir() if path.is_dir()]

    all_records: list[dict] = []
    for run_dir in target_runs:
        all_records.extend(build_or_load_search_index_for_run(run_dir))
    return all_records


def _compute_search_cache_buster(debug_runs_dir: Path, scope: str, current_run_dir: Path | None) -> str:
    if scope == "Current selected run" and current_run_dir and current_run_dir.exists():
        candidates = [
            current_run_dir / "20_search_index.json",
            current_run_dir / "17_topk_final_summary.json",
            current_run_dir / "16_topk_vlm_outputs.json",
            current_run_dir / "11_yolo_object_scores.json",
        ]
        mtimes = [f"{path.name}:{path.stat().st_mtime_ns}" for path in candidates if path.exists()]
        return "|".join(mtimes) or current_run_dir.name

    if debug_runs_dir.exists():
        parts: list[str] = []
        for run_dir in sorted([path for path in debug_runs_dir.iterdir() if path.is_dir()], key=lambda path: path.name):
            cache_path = run_dir / "20_search_index.json"
            if cache_path.exists():
                parts.append(f"{run_dir.name}:{cache_path.stat().st_mtime_ns}")
            else:
                parts.append(f"{run_dir.name}:{run_dir.stat().st_mtime_ns}")
        return "|".join(parts)
    return "no-runs"


def build_or_load_search_index_for_run(run_dir: Path, force_rebuild=False) -> list[dict]:
    cache_path = run_dir / "20_search_index.json"
    summary_path = run_dir / "17_topk_final_summary.json"
    vlm_path = run_dir / "16_topk_vlm_outputs.json"
    yolo_scores_path = run_dir / "11_yolo_object_scores.json"
    if not summary_path.exists():
        return []

    if not force_rebuild and cache_path.exists():
        cache_mtime = cache_path.stat().st_mtime
        summary_mtime = summary_path.stat().st_mtime
        vlm_mtime = vlm_path.stat().st_mtime if vlm_path.exists() else 0.0
        yolo_mtime = yolo_scores_path.stat().st_mtime if yolo_scores_path.exists() else 0.0
        if cache_mtime >= max(summary_mtime, vlm_mtime, yolo_mtime):
            cached = load_json(cache_path, default=[])
            if isinstance(cached, list):
                return cached

    records = build_search_records_for_run(run_dir)
    cache_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


def _build_object_search_records_for_run(
    run_dir: Path,
    video_info: dict[str, Any],
    yolo_scores: list[dict[str, Any]],
    compiled_video_path: str | None,
) -> list[dict]:
    records: list[dict] = []
    for score_item in yolo_scores if isinstance(yolo_scores, list) else []:
        if not isinstance(score_item, dict):
            continue
        frame_idx = int(score_item.get("frame_idx", 0) or 0)
        timestamp_seconds = float(score_item.get("timestamp_seconds", 0.0) or 0.0)
        annotated_frame_path = score_item.get("annotated_frame_path")
        detections = score_item.get("detections", [])
        if not isinstance(detections, list):
            continue
        for detection_index, detection in enumerate(detections):
            if not isinstance(detection, dict):
                continue
            class_name = str(detection.get("class_name", "")).strip().lower()
            if not class_name:
                continue
            appearance_terms = [str(term).strip().lower() for term in detection.get("appearance_terms", []) if str(term).strip()]
            crop_path = detection.get("crop_path")
            description = " ".join(
                part
                for part in [
                    "Detected",
                    class_name,
                    "at",
                    f"{timestamp_seconds:.1f}s.",
                    f"Appearance: {', '.join(appearance_terms[:4])}." if appearance_terms else "",
                ]
                if part
            ).strip()
            raw_search_text = " ".join(
                [
                    str(video_info.get("video_name", run_dir.name)),
                    f"object {class_name}",
                    " ".join(appearance_terms),
                    str(score_item.get("motion_level", "")),
                    str(score_item.get("selection_reason", "")),
                    str(score_item.get("motion_score_norm", "")),
                ]
            ).lower()
            records.append(
                {
                    "record_type": "object_evidence",
                    "run_dir": str(run_dir),
                    "run_name": run_dir.name,
                    "video_name": str(video_info.get("video_name", run_dir.name)),
                    "clip_id": f"frame_{frame_idx:06d}_{detection_index:02d}_{class_name}",
                    "time_range": f"{timestamp_seconds:.1f}s",
                    "start_time": timestamp_seconds,
                    "end_time": timestamp_seconds,
                    "final_category": "object_evidence",
                    "event_label": f"object_detected:{class_name}",
                    "risk_level": "unknown",
                    "confidence": f"{float(detection.get('confidence', 0.0) or 0.0):.2f}",
                    "caption": description,
                    "description": description,
                    "incident_category": "",
                    "incident_event_label": "",
                    "secondary_event_labels": [],
                    "incident_score": 0.0,
                    "evidence_strength": "object_detection",
                    "weapon_visible": "",
                    "weapon_description": "",
                    "person_grabbing_or_restraining": "",
                    "grabbing_description": "",
                    "person_threatened_or_controlled": "",
                    "threat_description": "",
                    "taking_items_visible": "",
                    "item_taking_description": "",
                    "display_or_counter_interaction": "",
                    "display_interaction_description": "",
                    "incident_visible_evidence": [],
                    "incident_review_reason": "",
                    "incident_suspicious_explanation": "",
                    "incident_normal_explanation": "",
                    "motion_summary": str(score_item.get("motion_level", "")),
                    "moving_objects": [class_name] if str(score_item.get("motion_level", "")).lower() in {"medium", "high"} else [],
                    "stationary_objects": [class_name] if str(score_item.get("motion_level", "")).lower() not in {"medium", "high"} else [],
                    "traffic_state": "",
                    "people_count": 1 if class_name == "person" else 0,
                    "person_descriptions": appearance_terms if class_name == "person" else [],
                    "person_clothing_text": appearance_terms if class_name == "person" else [],
                    "object_names": [class_name],
                    "yolo_object_classes": [class_name],
                    "vehicle_terms": appearance_terms if class_name in {"car", "truck", "bus", "motorcycle", "bicycle"} else [],
                    "activity_types": [],
                    "event_types": [],
                    "keywords": appearance_terms[:6],
                    "selection_reasons": [str(score_item.get("selection_reason", "")).strip()] if str(score_item.get("selection_reason", "")).strip() else [],
                    "ranking_reasons": [],
                    "motion_score": score_item.get("motion_score_norm", 0.0),
                    "ranked_clip_score": score_item.get("object_importance_score", 0.0),
                    "strip_path": crop_path,
                    "top_annotated_frame_path": annotated_frame_path,
                    "compiled_video_path": compiled_video_path,
                    "qwen_parsed_json": None,
                    "raw_vlm_output": None,
                    "parse_success": None,
                    "parse_error": None,
                    "frame_path": score_item.get("frame_path"),
                    "object_crop_path": crop_path,
                    "object_confidence": float(detection.get("confidence", 0.0) or 0.0),
                    "object_class_name": class_name,
                    "appearance_terms": appearance_terms,
                    "raw_search_text": raw_search_text,
                }
            )
    return records


def build_search_records_for_run(run_dir: Path) -> list[dict]:
    summary = load_json(run_dir / "17_topk_final_summary.json", default={}) or {}
    if not isinstance(summary, dict):
        return []

    vlm_outputs = load_json(run_dir / "16_topk_vlm_outputs.json", default={}) or {}
    video_info = load_json(run_dir / "01_video_info.json", default={}) or {}
    yolo_scores = load_json(run_dir / "11_yolo_object_scores.json", default=[]) or []
    export_manifest = load_json(run_dir / "18_exported_clips.json", default={}) or {}
    compiled_manifest = (
        load_json(run_dir / "18_compiled_review_video.json", default=None)
        or load_json(run_dir / "18_exported_clips" / "18_compiled_review_video.json", default={})
        or {}
    )

    qwen_by_clip_id = {}
    for item in vlm_outputs.get("items", []) if isinstance(vlm_outputs, dict) else []:
        if isinstance(item, dict):
            clip_id = str(item.get("source_clip_id", "")).strip()
            if clip_id:
                qwen_by_clip_id[clip_id] = item

    export_by_clip_id = {}
    for item in export_manifest.get("exported_clips", []) if isinstance(export_manifest, dict) else []:
        if isinstance(item, dict):
            clip_id = str(item.get("clip_id", "")).strip()
            if clip_id:
                export_by_clip_id[clip_id] = item

    records: list[dict] = []
    for event in summary.get("event_timeline", []) if isinstance(summary.get("event_timeline"), list) else []:
        if not isinstance(event, dict):
            continue
        clip_id = str(event.get("clip_id", "")).strip()
        qwen_item = qwen_by_clip_id.get(clip_id, {})
        parsed_json = qwen_item.get("parsed_json", {}) if isinstance(qwen_item, dict) else {}
        if not isinstance(parsed_json, dict):
            parsed_json = {}

        visible_people = parsed_json.get("visible_people", [])
        objects = parsed_json.get("objects", [])
        activities = parsed_json.get("activities", [])
        events = parsed_json.get("events", [])
        keywords = parsed_json.get("keywords", [])

        person_descriptions = []
        person_clothing_text = []
        for person in visible_people if isinstance(visible_people, list) else []:
            if not isinstance(person, dict):
                continue
            appearance = str(person.get("appearance", "")).strip()
            pose = str(person.get("pose_or_action", "")).strip()
            location = str(person.get("location", "")).strip()
            combined = " ".join(part for part in [appearance, pose, location] if part)
            if combined:
                person_descriptions.append(combined)
            if appearance:
                person_clothing_text.append(appearance)

        object_names = []
        vehicle_terms = []
        for obj in objects if isinstance(objects, list) else []:
            if not isinstance(obj, dict):
                continue
            name = str(obj.get("name", "")).strip()
            if name:
                object_names.append(name)
                if any(term in name.lower() for term in ["car", "bike", "truck", "vehicle", "bus"]):
                    vehicle_terms.append(name)

        activity_types = [str(activity.get("activity_type", "")).strip() for activity in activities if isinstance(activity, dict)]
        event_types = [str(event_item.get("event_type", "")).strip() for event_item in events if isinstance(event_item, dict)]

        yolo_object_classes = list(event.get("yolo_top_classes", []) or [])
        start_time = float(event.get("start_time", 0.0) or 0.0)
        end_time = float(event.get("end_time", start_time) or start_time)
        if isinstance(yolo_scores, list):
            for score_item in yolo_scores:
                if not isinstance(score_item, dict):
                    continue
                timestamp = float(score_item.get("timestamp_seconds", -1.0) or -1.0)
                if start_time <= timestamp <= end_time:
                    for class_name in score_item.get("object_classes_present", []) or []:
                        class_name_str = str(class_name).strip()
                        if class_name_str and class_name_str not in yolo_object_classes:
                            yolo_object_classes.append(class_name_str)

        description = str(event.get("best_event_description", "")).strip()
        caption = str(event.get("caption", "")).strip()
        incident_category = str(event.get("incident_category", "")).strip()
        incident_event_label = str(event.get("incident_event_label", "")).strip()
        secondary_event_labels = [str(item).strip() for item in event.get("secondary_event_labels", []) if str(item).strip()] if isinstance(event.get("secondary_event_labels"), list) else []
        incident_score = float(event.get("incident_score", 0.0) or 0.0)
        evidence_strength = str(event.get("evidence_strength", "")).strip()
        weapon_visible = str(event.get("weapon_visible", "")).strip()
        weapon_description = str(event.get("weapon_description", "")).strip()
        person_grabbing_or_restraining = str(event.get("person_grabbing_or_restraining", "")).strip()
        grabbing_description = str(event.get("grabbing_description", "")).strip()
        person_threatened_or_controlled = str(event.get("person_threatened_or_controlled", "")).strip()
        threat_description = str(event.get("threat_description", "")).strip()
        taking_items_visible = str(event.get("taking_items_visible", "")).strip()
        item_taking_description = str(event.get("item_taking_description", "")).strip()
        display_or_counter_interaction = str(event.get("display_or_counter_interaction", "")).strip()
        display_interaction_description = str(event.get("display_interaction_description", "")).strip()
        incident_visible_evidence = [str(item).strip() for item in event.get("incident_visible_evidence", []) if str(item).strip()] if isinstance(event.get("incident_visible_evidence"), list) else []
        incident_review_reason = str(event.get("incident_review_reason", "")).strip()
        incident_suspicious_explanation = str(event.get("incident_suspicious_explanation", "")).strip()
        incident_normal_explanation = str(event.get("incident_normal_explanation", "")).strip()
        motion_summary = str(event.get("motion_summary", "")).strip()
        moving_objects = [str(item).strip() for item in event.get("moving_objects", []) if str(item).strip()] if isinstance(event.get("moving_objects"), list) else []
        stationary_objects = [str(item).strip() for item in event.get("stationary_objects", []) if str(item).strip()] if isinstance(event.get("stationary_objects"), list) else []
        traffic_state = str(event.get("traffic_state", "")).strip()
        motion_aliases: list[str] = []
        for name in moving_objects:
            lowered = name.lower()
            motion_aliases.append(f"moving {lowered}")
            if lowered in {"car", "truck", "bus", "vehicle", "motorcycle", "bicycle", "scooter", "auto rickshaw"}:
                motion_aliases.append("moving vehicle")
                motion_aliases.append("traffic activity")
            if lowered == "person":
                motion_aliases.append("walking person")
                motion_aliases.append("moving person")
        for name in stationary_objects:
            lowered = name.lower()
            motion_aliases.append(f"stationary {lowered}")
            if lowered in {"car", "truck", "bus", "vehicle", "motorcycle", "bicycle", "scooter", "auto rickshaw"}:
                motion_aliases.append("parked car")
                motion_aliases.append("stationary vehicle")
        raw_text_parts = [
            str(video_info.get("video_name", "")),
            clip_id,
            str(event.get("time_range", "")),
            str(event.get("final_category", "")),
            str(event.get("event_label", "")),
            str(event.get("risk_level", "")),
            str(event.get("confidence", "")),
            caption,
            description,
            incident_category,
            incident_event_label,
            " ".join(secondary_event_labels),
            str(incident_score),
            evidence_strength,
            weapon_visible,
            weapon_description,
            person_grabbing_or_restraining,
            grabbing_description,
            person_threatened_or_controlled,
            threat_description,
            taking_items_visible,
            item_taking_description,
            display_or_counter_interaction,
            display_interaction_description,
            incident_review_reason,
            incident_suspicious_explanation,
            incident_normal_explanation,
            " ".join(incident_visible_evidence),
            "possible robbery weapon gun knife grabbing restraint employee threat controlled display case counter taking items theft from display group robbery",
            motion_summary,
            traffic_state,
            " ".join(moving_objects),
            " ".join(stationary_objects),
            " ".join(motion_aliases),
            " ".join(person_descriptions),
            " ".join(person_clothing_text),
            " ".join(object_names),
            " ".join(yolo_object_classes),
            " ".join(vehicle_terms),
            " ".join(activity_types),
            " ".join(event_types),
            " ".join(str(keyword) for keyword in keywords if str(keyword).strip()),
            " ".join(str(reason) for reason in event.get("selection_reasons", []) or []),
            " ".join(str(reason) for reason in event.get("ranking_reasons", []) or []),
        ]
        raw_search_text = " ".join(part for part in raw_text_parts if part).lower()

        export_item = export_by_clip_id.get(clip_id, {})
        compiled_video_path = None
        if isinstance(compiled_manifest, dict) or isinstance(export_manifest, dict):
            compiled_results = {
                "compiled_video": compiled_manifest if isinstance(compiled_manifest, dict) else {},
                "exported_clips": export_manifest if isinstance(export_manifest, dict) else {},
            }
            resolved_compiled = _extract_compiled_video_path(run_dir, compiled_results)
            compiled_video_path = str(resolved_compiled) if resolved_compiled else None

        records.append(
            {
                "record_type": "event_timeline",
                "run_dir": str(run_dir),
                "run_name": run_dir.name,
                "video_name": str(video_info.get("video_name", run_dir.name)),
                "clip_id": clip_id,
                "time_range": str(event.get("time_range", "")),
                "start_time": start_time,
                "end_time": end_time,
                "final_category": str(event.get("final_category", "")),
                "event_label": str(event.get("event_label", "")),
                "risk_level": str(event.get("risk_level", "")),
                "confidence": str(event.get("confidence", "")),
                "caption": caption,
                "description": description,
                "incident_category": incident_category,
                "incident_event_label": incident_event_label,
                "secondary_event_labels": secondary_event_labels,
                "incident_score": incident_score,
                "evidence_strength": evidence_strength,
                "weapon_visible": weapon_visible,
                "weapon_description": weapon_description,
                "person_grabbing_or_restraining": person_grabbing_or_restraining,
                "grabbing_description": grabbing_description,
                "person_threatened_or_controlled": person_threatened_or_controlled,
                "threat_description": threat_description,
                "taking_items_visible": taking_items_visible,
                "item_taking_description": item_taking_description,
                "display_or_counter_interaction": display_or_counter_interaction,
                "display_interaction_description": display_interaction_description,
                "incident_visible_evidence": incident_visible_evidence,
                "incident_review_reason": incident_review_reason,
                "incident_suspicious_explanation": incident_suspicious_explanation,
                "incident_normal_explanation": incident_normal_explanation,
                "motion_summary": motion_summary,
                "moving_objects": moving_objects,
                "stationary_objects": stationary_objects,
                "traffic_state": traffic_state,
                "people_count": int(event.get("people_count", 0) or 0),
                "person_descriptions": person_descriptions,
                "person_clothing_text": person_clothing_text,
                "object_names": object_names,
                "yolo_object_classes": yolo_object_classes,
                "vehicle_terms": vehicle_terms,
                "activity_types": [item for item in activity_types if item],
                "event_types": [item for item in event_types if item],
                "keywords": [str(keyword).strip() for keyword in keywords if str(keyword).strip()],
                "selection_reasons": event.get("selection_reasons", []) or [],
                "ranking_reasons": event.get("ranking_reasons", []) or [],
                "motion_score": event.get("motion_score", 0.0),
                "ranked_clip_score": event.get("ranked_clip_score", 0.0),
                "strip_path": event.get("strip_path"),
                "top_annotated_frame_path": event.get("top_annotated_frame_path"),
                "compiled_video_path": compiled_video_path,
                "qwen_parsed_json": parsed_json,
                "raw_vlm_output": qwen_item.get("raw_vlm_output") if isinstance(qwen_item, dict) else None,
                "parse_success": qwen_item.get("parse_success") if isinstance(qwen_item, dict) else None,
                "parse_error": qwen_item.get("parse_error") if isinstance(qwen_item, dict) else None,
                "raw_search_text": raw_search_text,
            }
        )
    compiled_video_path = None
    if isinstance(compiled_manifest, dict) or isinstance(export_manifest, dict):
        compiled_results = {
            "compiled_video": compiled_manifest if isinstance(compiled_manifest, dict) else {},
            "exported_clips": export_manifest if isinstance(export_manifest, dict) else {},
        }
        resolved_compiled = _extract_compiled_video_path(run_dir, compiled_results)
        compiled_video_path = str(resolved_compiled) if resolved_compiled else None
    records.extend(_build_object_search_records_for_run(run_dir, video_info, yolo_scores, compiled_video_path))
    return records


def search_records(records: list[dict], filters: dict) -> list[dict]:
    query = str(filters.get("query", "")).strip().lower()
    query_terms = [term for term in re.split(r"\s+", query) if term]
    category_filter = filters.get("category", "All")
    risk_filter = filters.get("risk_level", "All")
    event_filter = str(filters.get("event_type", "All")).strip().lower()
    object_filter = str(filters.get("object_type", "")).strip().lower()
    appearance_filter = str(filters.get("person_appearance", "")).strip().lower()
    vehicle_filter = str(filters.get("vehicle", "")).strip().lower()
    selected_videos = filters.get("selected_videos", ["All videos"])
    if not isinstance(selected_videos, list):
        selected_videos = ["All videos"]
    time_start = filters.get("time_start")
    time_end = filters.get("time_end")

    category_map = {
        "Priority suspicious event": "priority_suspicious_event",
        "Possible review clip": "possible_review_clip",
        "Normal activity": "normal_activity",
        "Uncertain activity": "uncertain_activity",
        "Object evidence": "object_evidence",
    }

    filtered: list[dict] = []
    for record in records:
        if category_filter != "All" and record.get("final_category") != category_map.get(category_filter):
            continue
        if risk_filter != "All" and str(record.get("risk_level", "")).lower() != str(risk_filter).lower():
            continue
        if event_filter != "all":
            searchable_event_text = " ".join(
                [
                    str(record.get("event_label", "")),
                    " ".join(record.get("activity_types", [])),
                    " ".join(record.get("event_types", [])),
                    str(record.get("description", "")),
                ]
            ).lower()
            if event_filter not in searchable_event_text:
                continue
        if object_filter:
            object_text = " ".join(record.get("object_names", []) + record.get("yolo_object_classes", []) + record.get("appearance_terms", [])).lower()
            if object_filter not in object_text:
                continue
        if appearance_filter:
            appearance_text = " ".join(record.get("person_descriptions", []) + record.get("person_clothing_text", [])).lower()
            if appearance_filter not in appearance_text:
                continue
        if vehicle_filter:
            vehicle_text = " ".join(record.get("vehicle_terms", []) + record.get("object_names", []) + record.get("appearance_terms", [])).lower()
            if vehicle_filter not in vehicle_text:
                continue
        if "All videos" not in selected_videos:
            record_video_keys = {record.get("video_name"), record.get("run_name")}
            if not any(selected_video in record_video_keys for selected_video in selected_videos):
                continue
        if time_start is not None and float(record.get("start_time", 0.0) or 0.0) < float(time_start):
            continue
        if time_end is not None and float(record.get("end_time", 0.0) or 0.0) > float(time_end):
            continue

        score = 0
        raw_search_text = str(record.get("raw_search_text", "")).lower()
        matched_terms: list[str] = []
        if query:
            if query in raw_search_text:
                score += 10
                matched_terms.append(query)
            for term in query_terms:
                if term in raw_search_text:
                    score += 2
                    matched_terms.append(term)
                if term in str(record.get("event_label", "")).lower():
                    score += 4
                if term in str(record.get("description", "")).lower():
                    score += 4
                if term in " ".join(record.get("object_names", []) + record.get("yolo_object_classes", [])).lower():
                    score += 5
                if term in " ".join(record.get("appearance_terms", [])).lower():
                    score += 5
                if term in " ".join(record.get("person_descriptions", []) + record.get("person_clothing_text", [])).lower():
                    score += 5
        if record.get("final_category") == "priority_suspicious_event":
            score += 2
        if record.get("final_category") == "object_evidence":
            score += 1
        if str(record.get("risk_level", "")).lower() == "high":
            score += 2
        elif str(record.get("risk_level", "")).lower() == "medium":
            score += 1

        record_copy = dict(record)
        record_copy["search_score"] = score
        record_copy["matched_terms"] = sorted(set(matched_terms))
        filtered.append(record_copy)

    if query:
        category_priority = {
            "priority_suspicious_event": 0,
            "possible_review_clip": 1,
            "normal_activity": 2,
            "uncertain_activity": 3,
        }
        filtered.sort(
            key=lambda item: (
                -int(item.get("search_score", 0)),
                category_priority.get(str(item.get("final_category", "")), 9),
                float(item.get("start_time", 0.0) or 0.0),
            )
        )
    else:
        category_priority = {
            "priority_suspicious_event": 0,
            "possible_review_clip": 1,
            "normal_activity": 2,
            "uncertain_activity": 3,
        }
        filtered.sort(
            key=lambda item: (
                category_priority.get(str(item.get("final_category", "")), 9),
                float(item.get("start_time", 0.0) or 0.0),
            )
        )
    return filtered


SEARCH_OBJECT_TERMS = [
    "display case",
    "backpack",
    "handbag",
    "suitcase",
    "motorcycle",
    "bicycle",
    "vehicle",
    "person",
    "people",
    "woman",
    "child",
    "truck",
    "phone",
    "laptop",
    "jewelry",
    "bottle",
    "bag",
    "bike",
    "bus",
    "car",
    "man",
]

SEARCH_APPEARANCE_TERMS = [
    "person with bag",
    "carrying bag",
    "dark clothing",
    "black shirt",
    "white shirt",
    "blue shirt",
    "pink shirt",
    "red shirt",
    "red cap",
    "black car",
    "white car",
    "grey car",
    "blue car",
    "red car",
]

SEARCH_EVENT_TERMS = [
    "person object interaction",
    "suspicious reaching",
    "normal activity",
    "collision",
    "intrusion",
    "crowding",
    "accident",
    "standing",
    "robbery",
    "walking",
    "reaching",
    "bending",
    "theft",
    "fight",
    "fall",
]


def _parse_nl_search_time_seconds(text: str) -> float | None:
    normalized = str(text or "").strip().lower()
    match = re.search(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b", normalized)
    if not match:
        return None
    first = int(match.group(1))
    second = int(match.group(2))
    third = match.group(3)
    if third is not None:
        return float((first * 3600) + (second * 60) + int(third))
    return float((first * 60) + second)


def interpret_nl_search_query(query: str) -> dict[str, Any]:
    normalized = str(query or "").strip().lower()
    interpreted = {
        "intent": "general_search",
        "query": query,
        "object_type": "",
        "person_appearance": "",
        "vehicle": "",
        "event_type": "All",
        "time_start": None,
        "time_end": None,
        "search_terms": [],
    }
    if not normalized:
        return interpreted

    matched_objects = [term for term in SEARCH_OBJECT_TERMS if term in normalized]
    matched_appearance = [term for term in SEARCH_APPEARANCE_TERMS if term in normalized]
    matched_events = [term for term in SEARCH_EVENT_TERMS if term in normalized]

    if any(term in normalized for term in ["show", "find", "search", "look for", "where is", "track"]):
        interpreted["intent"] = "find_object"
    if any(term in normalized for term in ["robbery", "theft", "fight", "fall", "collision", "accident", "intrusion"]):
        interpreted["intent"] = "find_incident"

    if matched_objects:
        preferred_object = max(matched_objects, key=len)
        object_aliases = {
            "man": "person",
            "woman": "person",
            "child": "person",
            "people": "person",
            "bike": "motorcycle",
        }
        interpreted["object_type"] = object_aliases.get(preferred_object, preferred_object)

    if matched_appearance:
        interpreted["person_appearance"] = max(matched_appearance, key=len)

    vehicle_terms = [
        term for term in matched_objects + matched_appearance
        if any(token in term for token in ["car", "truck", "bus", "bike", "bicycle", "motorcycle", "vehicle"])
    ]
    if vehicle_terms:
        interpreted["vehicle"] = max(vehicle_terms, key=len)

    if matched_events:
        interpreted["event_type"] = max(matched_events, key=len)

    start_match = re.search(r"\b(?:after|from|starting at)\s+(\d{1,2}:\d{2}(?::\d{2})?)", normalized)
    end_match = re.search(r"\b(?:before|until|ending at)\s+(\d{1,2}:\d{2}(?::\d{2})?)", normalized)
    between_match = re.search(
        r"\bbetween\s+(\d{1,2}:\d{2}(?::\d{2})?)\s+and\s+(\d{1,2}:\d{2}(?::\d{2})?)",
        normalized,
    )
    if between_match:
        interpreted["time_start"] = _parse_nl_search_time_seconds(between_match.group(1))
        interpreted["time_end"] = _parse_nl_search_time_seconds(between_match.group(2))
    else:
        if start_match:
            interpreted["time_start"] = _parse_nl_search_time_seconds(start_match.group(1))
        if end_match:
            interpreted["time_end"] = _parse_nl_search_time_seconds(end_match.group(1))
        if interpreted["time_start"] is None and interpreted["time_end"] is None:
            standalone_time = _parse_nl_search_time_seconds(normalized)
            if standalone_time is not None:
                interpreted["time_start"] = standalone_time

    search_terms: list[str] = []
    for value in [
        interpreted["object_type"],
        interpreted["person_appearance"],
        interpreted["vehicle"],
        interpreted["event_type"] if interpreted["event_type"] != "All" else "",
    ]:
        value_str = str(value).strip()
        if value_str and value_str not in search_terms:
            search_terms.append(value_str)
    interpreted["search_terms"] = search_terms
    return interpreted


def _render_search_result(record: dict) -> None:
    record_type = str(record.get("record_type", "event_timeline")).strip().lower()
    if record_type == "object_evidence":
        class_name = str(record.get("object_class_name", "") or record.get("event_label", "")).replace("object_detected:", "")
        st.markdown(
            f"### {class_name or 'object'} | {record.get('time_range', 'unknown')} | "
            f"{record.get('video_name', 'unknown video')}"
        )
    else:
        st.markdown(
            f"### {record.get('clip_id', 'unknown')} | {record.get('time_range', 'unknown')} | "
            f"{record.get('video_name', 'unknown video')}"
        )

    run_dir = Path(str(record.get("run_dir")))
    media_cols = st.columns(2)
    strip_path = resolve_media_path(run_dir, record.get("object_crop_path") or record.get("strip_path"))
    yolo_path = resolve_media_path(run_dir, record.get("top_annotated_frame_path"))
    if record_type == "object_evidence":
        metric_cols = st.columns(5)
        metric_cols[0].metric("Class", str(record.get("object_class_name", "unknown")))
        metric_cols[1].metric("Timestamp", str(record.get("time_range", "unknown")))
        metric_cols[2].metric("Confidence", str(record.get("confidence", "unknown")))
        metric_cols[3].metric("Run", str(record.get("run_name", "unknown")))
        metric_cols[4].metric("Score", str(record.get("search_score", 0)))

        appearance_terms = [str(term).strip() for term in record.get("appearance_terms", []) if str(term).strip()]
        if appearance_terms:
            st.caption("Appearance terms")
            st.write(", ".join(appearance_terms))

        description = str(record.get("description", "")).strip()
        if description:
            st.write(description)

        with media_cols[0]:
            if media_exists(strip_path):
                st.image(str(strip_path), caption="Object crop", width="stretch")
            else:
                st.warning("Object crop not found.")
        with media_cols[1]:
            if media_exists(yolo_path):
                st.image(str(yolo_path), caption="Annotated frame", width="stretch")
            else:
                st.warning("Annotated frame not found.")

        with st.expander("Object metadata", expanded=False):
            st.write(
                {
                    "run_name": record.get("run_name"),
                    "video_name": record.get("video_name"),
                    "record_type": record.get("record_type"),
                    "object_class_name": record.get("object_class_name"),
                    "time_range": record.get("time_range"),
                    "confidence": record.get("confidence"),
                    "appearance_terms": record.get("appearance_terms", []),
                    "motion_summary": record.get("motion_summary"),
                    "matched_terms": record.get("matched_terms", []),
                }
            )
    else:
        st.write(
            {
                "run_name": record.get("run_name"),
                "final_category": record.get("final_category"),
                "risk_level": record.get("risk_level"),
                "confidence": record.get("confidence"),
                "event_label": record.get("event_label"),
                "description": record.get("description"),
                "incident_category": record.get("incident_category"),
                "incident_event_label": record.get("incident_event_label"),
                "secondary_event_labels": record.get("secondary_event_labels"),
                "incident_score": record.get("incident_score"),
                "evidence_strength": record.get("evidence_strength"),
                "weapon_visible": record.get("weapon_visible"),
                "weapon_description": record.get("weapon_description"),
                "person_grabbing_or_restraining": record.get("person_grabbing_or_restraining"),
                "grabbing_description": record.get("grabbing_description"),
                "person_threatened_or_controlled": record.get("person_threatened_or_controlled"),
                "threat_description": record.get("threat_description"),
                "taking_items_visible": record.get("taking_items_visible"),
                "item_taking_description": record.get("item_taking_description"),
                "display_or_counter_interaction": record.get("display_or_counter_interaction"),
                "display_interaction_description": record.get("display_interaction_description"),
                "incident_review_reason": record.get("incident_review_reason"),
                "incident_visible_evidence": record.get("incident_visible_evidence"),
                "motion_summary": record.get("motion_summary"),
                "moving_objects": record.get("moving_objects"),
                "stationary_objects": record.get("stationary_objects"),
                "traffic_state": record.get("traffic_state"),
                "people_count": record.get("people_count"),
                "yolo_object_classes": record.get("yolo_object_classes"),
                "person_descriptions": record.get("person_descriptions"),
                "activity_types": record.get("activity_types"),
                "event_types": record.get("event_types"),
                "matched_terms": record.get("matched_terms", []),
                "record_type": record.get("record_type"),
                "appearance_terms": record.get("appearance_terms", []),
            }
        )

        with media_cols[0]:
            if media_exists(strip_path):
                st.image(str(strip_path), caption="Incident image / temporal strip", width="stretch")
            else:
                st.warning("Object/incident image not found.")
        with media_cols[1]:
            if media_exists(yolo_path):
                st.image(str(yolo_path), caption="YOLO annotated frame", width="stretch")
            else:
                st.warning("YOLO annotated frame not found.")

    if record.get("qwen_parsed_json"):
        with st.expander("Qwen parsed JSON"):
            st.json(record.get("qwen_parsed_json"))
    with st.expander("Evidence file paths"):
        st.write(
            {
                "run_dir": record.get("run_dir"),
                "strip_path": record.get("strip_path"),
                "object_crop_path": record.get("object_crop_path"),
                "top_annotated_frame_path": record.get("top_annotated_frame_path"),
                "compiled_video_path": record.get("compiled_video_path"),
            }
        )
    with st.expander("Open run folder path text"):
        st.code(str(record.get("run_dir", "")))
    if record.get("compiled_video_path"):
        st.write(f"Compiled review video path: `{record.get('compiled_video_path')}`")


def _render_json_details(label: str, payload: Any) -> None:
    if payload is None:
        return
    with st.expander(label):
        st.json(payload)


def _event_sort_key(event: dict) -> tuple[int, float, int, int, int, float]:
    category_priority = {
        "priority_suspicious_event": 0,
        "possible_review_clip": 1,
        "uncertain_activity": 2,
        "normal_activity": 3,
    }
    evidence_priority = {"strong": 0, "medium": 1, "weak": 2, "none": 3, "unknown": 4}
    risk_priority = {"high": 0, "medium": 1, "low": 2, "unknown": 3}
    confidence_priority = {"high": 0, "medium": 1, "low": 2, "unknown": 3}
    return (
        category_priority.get(str(event.get("final_category", "")), 9),
        -float(event.get("incident_score", 0.0) or 0.0),
        evidence_priority.get(str(event.get("evidence_strength", "unknown")).lower(), 9),
        risk_priority.get(str(event.get("risk_level", "unknown")).lower(), 9),
        confidence_priority.get(str(event.get("confidence", "unknown")).lower(), 9),
        float(event.get("start_time", 0.0) or 0.0),
    )


def render_event_card(event, qwen_by_clip_id, run_dir, show_raw_qwen=False):
    clip_id = str(event.get("clip_id", "unknown_clip"))
    qwen_item = qwen_by_clip_id.get(clip_id, {})
    event_title = str(event.get("title", "")).strip()
    event_description = str(event.get("description", "")).strip() or str(event.get("best_event_description", "")).strip()

    st.markdown(f"**{event_title or clip_id}**")
    if event_title:
        st.caption(clip_id)
    left, right = st.columns(2)
    with left:
        st.write(f"Time: `{event.get('time_range', 'unknown')}`")
        st.write(f"Event label: `{event.get('event_label', 'unknown')}`")
        st.write(f"Primary event label: `{event.get('primary_event_label', event.get('incident_event_label', 'unknown'))}`")
        st.write(f"Secondary event labels: {', '.join(event.get('secondary_event_labels', [])) if isinstance(event.get('secondary_event_labels'), list) and event.get('secondary_event_labels') else 'n/a'}")
        st.write(f"Risk: `{event.get('risk_level', 'unknown')}`")
        st.write(f"Confidence: `{event.get('confidence', 'unknown')}`")
        st.write(f"Incident score: `{event.get('incident_score', 'n/a')}`")
        st.write(f"Evidence strength: `{event.get('evidence_strength', 'n/a')}`")
        st.write(f"Description: {event_description or 'n/a'}")
        st.write(f"Selection reasons: {', '.join(event.get('selection_reasons', [])) or 'n/a'}")
        st.write(f"Why selected: {event.get('why_selected', 'n/a')}")
        st.write(f"Review note: {event.get('review_note', 'n/a')}")
    with right:
        st.write(f"Ranked clip score: `{event.get('ranked_clip_score', 'n/a')}`")
        st.write(f"Motion score: `{event.get('motion_score', 'n/a')}`")
        st.write(f"YOLO person max: `{event.get('yolo_person_max', 'n/a')}`")
        st.write(f"YOLO classes: {', '.join(event.get('yolo_top_classes', [])) or 'n/a'}")
        st.write(f"Traffic state: `{event.get('traffic_state', 'unclear') or 'unclear'}`")

    incident_category = str(event.get("incident_category", "")).strip()
    incident_event_label = str(event.get("incident_event_label", "")).strip()
    incident_visible_evidence = [str(item).strip() for item in event.get("incident_visible_evidence", []) if str(item).strip()] if isinstance(event.get("incident_visible_evidence"), list) else []
    incident_suspicious_explanation = str(event.get("incident_suspicious_explanation", "")).strip()
    incident_normal_explanation = str(event.get("incident_normal_explanation", "")).strip()
    incident_review_reason = str(event.get("incident_review_reason", "")).strip()
    if incident_category or incident_event_label or incident_visible_evidence:
        st.write("Incident recheck")
        st.write(f"Incident category: `{incident_category or 'n/a'}`")
        st.write(f"Incident event label: `{incident_event_label or 'n/a'}`")
        st.write(f"Incident confidence: `{event.get('confidence', 'n/a')}`")
        st.write(f"Weapon visible: `{event.get('weapon_visible', 'unclear')}`")
        st.write(f"Weapon description: {event.get('weapon_description', 'n/a') or 'n/a'}")
        st.write(f"Grabbing/restraint: `{event.get('person_grabbing_or_restraining', 'unclear')}`")
        st.write(f"Grabbing description: {event.get('grabbing_description', 'n/a') or 'n/a'}")
        st.write(f"Threat/control: `{event.get('person_threatened_or_controlled', 'unclear')}`")
        st.write(f"Threat description: {event.get('threat_description', 'n/a') or 'n/a'}")
        st.write(f"Taking items visible: `{event.get('taking_items_visible', 'unclear')}`")
        st.write(f"Item-taking description: {event.get('item_taking_description', 'n/a') or 'n/a'}")
        st.write(f"Display/counter interaction: `{event.get('display_or_counter_interaction', 'unclear')}`")
        st.write(f"Display interaction description: {event.get('display_interaction_description', 'n/a') or 'n/a'}")
        st.write(f"Visible evidence: {', '.join(incident_visible_evidence) if incident_visible_evidence else 'n/a'}")
        st.write(f"Suspicious explanation: {incident_suspicious_explanation or 'n/a'}")
        st.write(f"Normal explanation: {incident_normal_explanation or 'n/a'}")
        st.write(f"Review reason: {incident_review_reason or 'n/a'}")

    motion_summary = str(event.get("motion_summary", "")).strip()
    moving_objects = [str(item).strip() for item in event.get("moving_objects", []) if str(item).strip()] if isinstance(event.get("moving_objects"), list) else []
    stationary_objects = [str(item).strip() for item in event.get("stationary_objects", []) if str(item).strip()] if isinstance(event.get("stationary_objects"), list) else []
    if motion_summary or moving_objects or stationary_objects:
        st.write("Motion evidence")
        if motion_summary:
            st.write(f"Summary: {motion_summary}")
        st.write(f"Moving objects: {', '.join(moving_objects) if moving_objects else 'n/a'}")
        st.write(f"Stationary objects: {', '.join(stationary_objects) if stationary_objects else 'n/a'}")

    strip_path = resolve_media_path(run_dir, event.get("strip_path"))
    yolo_path = resolve_media_path(run_dir, event.get("top_annotated_frame_path"))

    media_cols = st.columns(2)
    with media_cols[0]:
        if media_exists(strip_path):
            st.image(str(strip_path), caption=f"{clip_id} temporal strip", width="stretch")
        else:
            st.warning("Temporal strip media not found.")
    with media_cols[1]:
        if media_exists(yolo_path):
            st.image(str(yolo_path), caption=f"{clip_id} annotated YOLO frame", width="stretch")
        else:
            st.warning("Annotated YOLO frame not found.")

    if qwen_item:
        _render_json_details("Qwen Parsed Output", qwen_item.get("parsed_json"))
        if show_raw_qwen:
            with st.expander("Raw Qwen Output"):
                st.code(str(qwen_item.get("raw_vlm_output", "")), language="json")
        if qwen_item.get("parse_error"):
            st.warning(f"Qwen parse error: {qwen_item.get('parse_error')}")


def _load_run_results(run_dir: Path) -> dict[str, Any]:
    summary = load_json(run_dir / "17_topk_final_summary.json", default=None)
    if summary is None:
        st.error("17_topk_final_summary.json is missing for this run.")
        st.stop()

    return {
        "video_info": load_json(run_dir / "01_video_info.json", default={}) or {},
        "summary": summary,
        "adaptive_sampling": load_json(run_dir / "02b_adaptive_sampling_report.json", default={}) or {},
        "frame_candidate_pool": load_json(run_dir / "02c_frame_candidate_pool.json", default={}) or {},
        "vlm_outputs": load_json(run_dir / "16_topk_vlm_outputs.json", default={}) or {},
        "coverage_guardrails": load_json(run_dir / "14b_coverage_guardrail_report.json", default={}) or {},
        "vlm_coverage_audit": load_json(run_dir / "15_vlm_coverage_audit.json", default={}) or {},
        "exported_clips": load_json(run_dir / "18_exported_clips.json", default={}) or {},
        "compiled_video": load_json(run_dir / "18_compiled_review_video.json", default=None)
        or load_json(run_dir / "18_exported_clips" / "18_compiled_review_video.json", default={})
        or {},
        "runtime_metrics": load_json(run_dir / "20_runtime_metrics.json", default={}) or {},
    }


def _render_results_summary(run_dir: Path, results: dict[str, Any]) -> None:
    summary = results["summary"]
    processing_summary = summary.get("processing_summary", {})
    runtime_metrics = results.get("runtime_metrics", {}) if isinstance(results.get("runtime_metrics"), dict) else {}
    priority_count = int(processing_summary.get("priority_suspicious_events", 0) or 0)
    review_count = int(processing_summary.get("possible_review_clips", 0) or 0)
    scene_overview = summary.get("scene_overview", {}) if isinstance(summary.get("scene_overview"), dict) else {}
    descriptive_summary = (
        summary.get("descriptive_summary")
        or summary.get("final_summary_text")
        or summary.get("overall_summary")
        or "Summary not available."
    )
    incident_recheck_summary = summary.get("incident_recheck_summary", {}) if isinstance(summary.get("incident_recheck_summary"), dict) else {}
    analysis_settings = summary.get("analysis_settings", {}) if isinstance(summary.get("analysis_settings"), dict) else {}
    adaptive_sampling = results.get("adaptive_sampling", {}) if isinstance(results.get("adaptive_sampling"), dict) else {}
    frame_candidate_pool = results.get("frame_candidate_pool", {}) if isinstance(results.get("frame_candidate_pool"), dict) else {}
    coverage_guardrails = results.get("coverage_guardrails", {}) if isinstance(results.get("coverage_guardrails"), dict) else {}
    vlm_coverage_audit = results.get("vlm_coverage_audit", {}) if isinstance(results.get("vlm_coverage_audit"), dict) else {}
    review_clusters = summary.get("review_clusters", []) if isinstance(summary.get("review_clusters"), list) else []
    if not analysis_settings:
        analysis_settings = runtime_metrics.get("analysis_settings", {}) if isinstance(runtime_metrics.get("analysis_settings"), dict) else {}

    st.subheader("Final Summary")
    st.write(descriptive_summary)
    st.info("This summary is generated from selected Top-K clips. For richer summary, use the descriptive summary fields from Step 17.")
    st.caption("Incident Recheck improves detection of subtle events like theft, shoplifting, fight, fall, and collision, but increases runtime.")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Top-K Clips", processing_summary.get("topk_inputs", 0))
    metric_cols[1].metric("Successful Parses", processing_summary.get("successful_parses", 0))
    metric_cols[2].metric("Priority Events", priority_count)
    metric_cols[3].metric("Review Clips", review_count)

    st.subheader("Processing Performance")
    engine_name = str(runtime_metrics.get("pipeline_name") or runtime_metrics.get("pipeline_mode") or "standard_demo")
    if engine_name == "fast_parallel_topk":
        st.caption("Fast Parallel Top-K Pipeline")
    else:
        st.caption("Standard Demo Pipeline")
    perf_cols = st.columns(4)
    perf_cols[0].write(f"Pipeline engine: `{engine_name}`")
    perf_cols[1].write(f"Total runtime: `{runtime_metrics.get('total_runtime_seconds', 'unavailable')}`")
    perf_cols[2].write(f"Video duration: `{runtime_metrics.get('video_duration_seconds', 'unavailable')}`")
    perf_cols[3].write(f"Runtime/video ratio: `{runtime_metrics.get('runtime_ratio_to_video', 'unavailable')}`")
    perf_cols_2 = st.columns(4)
    perf_cols_2[0].write(f"Parallel branches enabled: `{runtime_metrics.get('parallel_branches_enabled', 'unavailable')}`")
    perf_cols_2[1].write(f"Skipped steps: `{runtime_metrics.get('skipped_steps', [])}`")
    slowest_steps = runtime_metrics.get("slowest_steps", []) if isinstance(runtime_metrics.get("slowest_steps"), list) else []
    if slowest_steps:
        perf_cols_2[2].write(
            f"Slowest step: `{slowest_steps[0].get('step_name', 'unknown')} ({slowest_steps[0].get('duration_seconds', 0.0)}s)`"
        )
        perf_cols_2[3].write(
            "Top 5 slowest steps: `"
            + ", ".join(f"{item.get('step_name', 'unknown')} ({item.get('duration_seconds', 0.0)}s)" for item in slowest_steps[:5])
            + "`"
        )
    else:
        perf_cols_2[2].write("Slowest step: `unavailable`")
        perf_cols_2[3].write("Top 5 slowest steps: `unavailable`")
    st.write(f"Top-K clips sent to Qwen: `{processing_summary.get('topk_inputs', 'unavailable')}`")
    runtime_ratio = runtime_metrics.get("runtime_ratio_to_video")
    try:
        runtime_ratio_value = float(runtime_ratio)
    except (TypeError, ValueError):
        runtime_ratio_value = None
    if runtime_ratio_value is not None:
        if runtime_ratio_value > 1.0:
            st.warning("Processing took longer than video length. To improve speed, use Quick Result Mode, increase sample interval, reduce Top-K, reduce Qwen max tokens, or use a smaller/faster VLM.")
        else:
            st.success("Processing completed faster than real time.")

    st.subheader("Incident Recheck")
    incident_cols = st.columns(5)
    incident_cols[0].write(f"Enabled: `{incident_recheck_summary.get('enabled', False)}`")
    incident_cols[1].write(f"Rechecked clips: `{incident_recheck_summary.get('rechecked_clips', 0)}`")
    incident_cols[2].write(f"Priority: `{incident_recheck_summary.get('priority_suspicious_events', 0)}`")
    incident_cols[3].write(f"Review: `{incident_recheck_summary.get('possible_review_clips', 0)}`")
    incident_cols[4].write(f"Normal: `{incident_recheck_summary.get('normal_activity', 0)}`")
    incident_cols_2 = st.columns(5)
    incident_cols_2[0].write(f"Uncertain: `{incident_recheck_summary.get('uncertain_activity', 0)}`")
    incident_cols_2[1].write(f"Weapon-visible clips: `{incident_recheck_summary.get('weapon_visible_clips', 0)}`")
    incident_cols_2[2].write(f"Grabbing/restraint clips: `{incident_recheck_summary.get('grabbing_or_restraint_clips', 0)}`")
    incident_cols_2[3].write(f"Threat/control clips: `{incident_recheck_summary.get('threat_or_control_clips', 0)}`")
    incident_cols_2[4].write(f"Taking-items clips: `{incident_recheck_summary.get('taking_items_clips', 0)}`")
    st.write(f"Display-interaction clips: `{incident_recheck_summary.get('display_interaction_clips', 0)}`")

    st.subheader("Analysis Settings")
    analysis_cols = st.columns(3)
    analysis_cols[0].write(f"Mode: `{analysis_settings.get('mode', 'unavailable')}`")
    analysis_cols[1].write(f"Sample interval: `{analysis_settings.get('sample_every_seconds', 'unavailable')}`")
    analysis_cols[2].write(f"Approx sampled FPS: `{analysis_settings.get('approx_sampled_fps', 'unavailable')}`")
    analysis_cols_2 = st.columns(3)
    analysis_cols_2[0].write(f"Top-K clips sent to Qwen: `{analysis_settings.get('top_k_clips', processing_summary.get('topk_inputs', 'unavailable'))}`")
    analysis_cols_2[1].write(f"Top-K max cap: `{analysis_settings.get('top_k_max', 'unavailable')}`")
    analysis_cols_2[2].write(f"Motion threshold: `{analysis_settings.get('motion_threshold', 'unavailable')}`")
    analysis_cols_3 = st.columns(3)
    analysis_cols_3[0].write(f"YOLO image size: `{analysis_settings.get('yolo_imgsz', 'unavailable')}`")
    analysis_cols_3[1].write(f"YOLO confidence: `{analysis_settings.get('yolo_conf', 'unavailable')}`")
    analysis_cols_3[2].write(f"Qwen max tokens: `{analysis_settings.get('qwen_max_new_tokens', 'unavailable')}`")
    analysis_cols_4 = st.columns(3)
    analysis_cols_4[0].write(f"Incident fallback pass enabled: `{analysis_settings.get('incident_fallback_pass_enabled', analysis_settings.get('incident_fallback_pass', 'unavailable'))}`")
    analysis_cols_4[1].write(f"Incident fallback pass used: `{summary.get('incident_fallback_pass_used', analysis_settings.get('incident_fallback_pass_used', False))}`")
    analysis_cols_4[2].write(f"Enable incident recheck: `{analysis_settings.get('enable_incident_recheck', 'unavailable')}`")
    st.write(f"Incident focus: `{analysis_settings.get('incident_focus', 'general')}`")
    analysis_cols_5 = st.columns(4)
    analysis_cols_5[0].write(f"Adaptive sampling enabled: `{analysis_settings.get('adaptive_sampling_enabled', adaptive_sampling.get('enabled', False))}`")
    analysis_cols_5[1].write(f"Coverage guardrails enabled: `{analysis_settings.get('coverage_guardrails_enabled', coverage_guardrails.get('enabled', False))}`")
    analysis_cols_5[2].write(f"VLM input strategy: `{analysis_settings.get('vlm_input_strategy', 'unavailable')}`")
    analysis_cols_5[3].write(f"Max VLM inputs: `{analysis_settings.get('max_vlm_inputs', 'unavailable')}`")

    st.subheader("Adaptive Coverage")
    st.write(f"Adaptive retained frames: `{adaptive_sampling.get('retained_frames', 0)}`")

    if review_clusters:
        st.subheader("Review Clusters")
        for cluster in review_clusters:
            st.markdown(f"**{cluster.get('cluster_id', 'review_cluster')}**")
            st.write(f"Time range: `{cluster.get('display_time', 'unknown')}`")
            st.write(f"Primary event label: `{cluster.get('primary_event_label', 'unknown')}`")
            st.write(f"Secondary event labels: {', '.join(cluster.get('secondary_event_labels', [])) if isinstance(cluster.get('secondary_event_labels'), list) and cluster.get('secondary_event_labels') else 'n/a'}")
            st.write(f"Risk level: `{cluster.get('risk_level', 'unknown')}`")
            st.write(f"Max incident score: `{cluster.get('max_incident_score', 'n/a')}`")
            st.write(f"Key evidence: {', '.join(cluster.get('key_evidence', [])) if isinstance(cluster.get('key_evidence'), list) and cluster.get('key_evidence') else 'n/a'}")
            st.write(f"Review reason: {cluster.get('review_reason', 'n/a') or 'n/a'}")
            st.caption(f"Clip IDs: {', '.join(cluster.get('clip_ids', [])) if isinstance(cluster.get('clip_ids'), list) else 'n/a'}")

    if scene_overview:
        st.subheader("Scene Overview")
        scene_cols = st.columns(4)
        people_counts = scene_overview.get("people_count_observed", {}) if isinstance(scene_overview.get("people_count_observed"), dict) else {}
        scene_cols[0].write(f"Dominant scene type: `{scene_overview.get('dominant_scene_type', 'unknown')}`")
        scene_cols[1].write(
            "Common activities: "
            + (", ".join(scene_overview.get("common_activities", [])) if isinstance(scene_overview.get("common_activities"), list) and scene_overview.get("common_activities") else "unavailable")
        )
        scene_cols[2].write(
            "Common objects: "
            + (", ".join(scene_overview.get("common_objects", [])) if isinstance(scene_overview.get("common_objects"), list) and scene_overview.get("common_objects") else "unavailable")
        )
        scene_cols[3].write(
            f"People count observed: `{people_counts.get('min', 0)} to {people_counts.get('max', 0)}`"
        )

    st.subheader("What Happened In This Video?")
    timeline_items = summary.get("event_timeline", []) if isinstance(summary.get("event_timeline"), list) else []
    if timeline_items:
        for item in timeline_items[:8]:
            st.write(
                f"- {item.get('time_range', 'unknown time')}: {_timeline_description(item)} "
                f"({item.get('final_category', 'unknown')})"
            )
    else:
        st.caption("No selected clips are available in the event timeline.")

    st.subheader("Detection Outcome")
    if priority_count > 0:
        st.error("Priority suspicious activity detected.")
    elif review_count > 0:
        st.warning("No priority suspicious event confirmed. Some clips are marked for review.")
    else:
        st.success("No priority suspicious event detected in the selected clips. The video mainly shows routine activity.")

    compiled_info = results["compiled_video"] if isinstance(results["compiled_video"], dict) else {}
    export_info = results["exported_clips"].get("compiled_review_video", {}) if isinstance(results["exported_clips"], dict) else {}
    compiled_path = _extract_compiled_video_path(run_dir, results)
    st.subheader("Compiled Review Video")
    if media_exists(compiled_path):
        if compiled_path.suffix.lower() == ".mp4":
            try:
                st.video(str(compiled_path))
            except Exception:
                with open(compiled_path, "rb") as video_file:
                    st.video(video_file.read())
        else:
            st.warning("Fallback AVI was created and is readable by OpenCV, but browsers may not play AVI. Install FFmpeg or convert to MP4 for browser playback.")
            st.caption(f"Fallback AVI video path: `{compiled_path}`")
            with open(compiled_path, "rb") as avi_file:
                st.download_button(
                    "Download fallback AVI",
                    data=avi_file.read(),
                    file_name=compiled_path.name,
                    mime="video/x-msvideo",
                )
    else:
        fallback_candidate = resolve_media_path(run_dir, str(run_dir / "18_exported_clips" / "18_compiled_review_video_fallback.avi"))
        if media_exists(fallback_candidate):
            st.info(f"Fallback AVI video created and readable by OpenCV. Open it from this path: `{fallback_candidate}`")
        else:
            st.warning("Compiled review video not found. Re-run Step 18. For normal-only videos, enable TENDER_DEMO_COMPILE_NORMAL_IF_NO_EVENTS=true.")
            st.caption("Run Step 18 again to generate compiled review video.")

    fallback_candidate = resolve_media_path(
        run_dir,
        compiled_info.get("fallback_video_path")
        or export_info.get("fallback_video_path")
        or str(run_dir / "18_exported_clips" / "18_compiled_review_video_fallback.avi"),
    )
    browser_mp4_candidate = resolve_media_path(
        run_dir,
        compiled_info.get("browser_playable_video_path")
        or export_info.get("browser_playable_video_path")
        or str(run_dir / "18_exported_clips" / "18_compiled_review_video_web.mp4"),
    )
    if media_exists(fallback_candidate) and not media_exists(browser_mp4_candidate):
        if st.button("Convert AVI to browser MP4"):
            success, message = convert_avi_to_browser_mp4(
                fallback_candidate,
                run_dir / "18_exported_clips" / "18_compiled_review_video_web.mp4",
            )
            if success:
                st.success(message)
                st.rerun()
            else:
                st.warning(message)

    verification = compiled_info.get("playback_recommended_verification") or compiled_info.get("video_verification", {})
    backend_value = compiled_info.get("compiled_video_backend") or export_info.get("backend") or "unavailable"
    fps_value = verification.get("fps", "unavailable")
    frame_count_value = verification.get("frame_count", "unavailable")
    readable_value = "yes" if verification.get("readable_by_opencv") is True else "unavailable"
    info_cols = st.columns(4)
    info_cols[0].write(f"Backend: `{backend_value}`")
    info_cols[1].write(f"FPS: `{fps_value}`")
    info_cols[2].write(f"Frame count: `{frame_count_value}`")
    info_cols[3].write(f"Readable: `{readable_value}`")
    with st.expander("Compiled video manifest details"):
        _render_json_details("Compiled Video Manifest", compiled_info or export_info)

    report_path = run_dir / "19_demo_report.html"
    st.write(f"19_demo_report.html path: `{report_path}`")


def _render_success_panel(run_dir: Path) -> None:
    st.success("Pipeline completed successfully.")
    st.write("Detected run folder:")
    st.code(str(run_dir))
    if st.button("Copy detected run to existing-run field instructions"):
        st.info("Copy the detected run folder shown above and paste it into Existing run folder if needed.")
    st.write(
        {
            "17_topk_final_summary.json": (run_dir / "17_topk_final_summary.json").exists(),
            "18_compiled_review_video.json": (run_dir / "18_compiled_review_video.json").exists(),
            "19_demo_report.html": (run_dir / "19_demo_report.html").exists(),
        }
    )
    st.info("Open the Results Summary, Events, or Search tab to view results.")


def _render_events_tab(run_dir: Path, results: dict[str, Any]) -> None:
    summary = results["summary"]
    vlm_outputs = results["vlm_outputs"]
    items = vlm_outputs.get("items", []) if isinstance(vlm_outputs, dict) else []
    qwen_by_clip_id = {
        str(item.get("source_clip_id")): item
        for item in items
        if isinstance(item, dict) and str(item.get("source_clip_id", "")).strip()
    }
    show_raw_qwen = st.checkbox("Show raw Qwen output", value=False)
    review_clusters = summary.get("review_clusters", []) if isinstance(summary.get("review_clusters"), list) else []

    if review_clusters:
        st.subheader("Review Clusters")
        for cluster in review_clusters:
            with st.expander(
                f"{cluster.get('cluster_id', 'review_cluster')} | {cluster.get('display_time', 'unknown')} | {cluster.get('primary_event_label', 'unknown')}",
                expanded=False,
            ):
                st.write(
                    {
                        "time_range": cluster.get("display_time"),
                        "primary_event_label": cluster.get("primary_event_label"),
                        "secondary_event_labels": cluster.get("secondary_event_labels", []),
                        "risk_level": cluster.get("risk_level"),
                        "max_incident_score": cluster.get("max_incident_score"),
                        "key_evidence": cluster.get("key_evidence", []),
                        "review_reason": cluster.get("review_reason"),
                        "clip_ids": cluster.get("clip_ids", []),
                    }
                )

    st.subheader("Priority Suspicious Events")
    for event in sorted(summary.get("priority_suspicious_events", []), key=_event_sort_key):
        with st.expander(
            f"{event.get('title', event.get('clip_id', 'unknown'))} | {event.get('time_range', 'unknown')} | "
            f"{event.get('risk_level', 'unknown')} | {event.get('confidence', 'unknown')}",
            expanded=True,
        ):
            render_event_card(event, qwen_by_clip_id, run_dir, show_raw_qwen=show_raw_qwen)

    st.subheader("Possible Review Clips")
    for event in sorted(summary.get("possible_review_clips", []), key=_event_sort_key):
        with st.expander(
            f"{event.get('title', event.get('clip_id', 'unknown'))} | {event.get('time_range', 'unknown')} | "
            f"{event.get('risk_level', 'unknown')} | {event.get('confidence', 'unknown')}",
            expanded=False,
        ):
            render_event_card(event, qwen_by_clip_id, run_dir, show_raw_qwen=show_raw_qwen)

    uncertain_items = summary.get("uncertain_clips", []) if isinstance(summary.get("uncertain_clips"), list) else []
    if uncertain_items:
        st.subheader("Uncertain Clips")
        for event in sorted(uncertain_items, key=_event_sort_key):
            with st.expander(
                f"{event.get('title', event.get('clip_id', 'unknown'))} | {event.get('time_range', 'unknown')} | uncertain",
                expanded=False,
            ):
                render_event_card(event, qwen_by_clip_id, run_dir, show_raw_qwen=False)

    st.subheader("Normal Activity Clips")
    for event in sorted(summary.get("normal_activity_clips", []), key=_event_sort_key):
        with st.expander(f"{event.get('title', event.get('clip_id', 'unknown'))} | {event.get('time_range', 'unknown')}", expanded=False):
            render_event_card(event, qwen_by_clip_id, run_dir, show_raw_qwen=False)


def _render_timeline_tab(results: dict[str, Any]) -> None:
    summary = results["summary"]
    event_timeline = summary.get("event_timeline", [])
    rows = [
        {
            "time_range": item.get("time_range"),
            "clip_id": item.get("clip_id"),
            "final_category": item.get("final_category"),
            "event_label": item.get("event_label"),
            "risk_level": item.get("risk_level"),
            "confidence": item.get("confidence"),
            "best_event_description": item.get("best_event_description"),
        }
        for item in sorted(event_timeline, key=lambda entry: float(entry.get("start_time", 0.0) or 0.0))
    ]
    st.dataframe(rows, width="stretch")

    counts = {}
    for item in rows:
        counts[item["final_category"]] = counts.get(item["final_category"], 0) + 1
    st.write(counts)


def _render_files_tab(run_dir: Path, logs: str) -> None:
    st.subheader("Evidence File Status")
    evidence_files = [
        run_dir / "01_video_info.json",
        run_dir / "13_ranked_clips.json",
        run_dir / "13_ranked_clips_report.json",
        run_dir / "14_selected_top_clips.json",
        run_dir / "14_selected_top_clips_report.json",
        run_dir / "15_topk_vlm_inputs.json",
        run_dir / "16_topk_vlm_outputs.json",
        run_dir / "17_topk_final_summary.json",
        run_dir / "17_topk_final_summary.md",
        run_dir / "18_exported_clips.json",
        run_dir / "18_compiled_review_video.json",
        run_dir / "19_demo_report.html",
    ]
    st.dataframe(
        [
            {
                "file": path.name,
                "status": "exists" if path.exists() else "missing",
                "path": str(path),
            }
            for path in evidence_files
        ],
        width="stretch",
    )
    st.subheader("Latest Pipeline Logs")
    st.code(logs or "No logs captured in this session.", language="text")
    st.subheader("Open Report Path")
    st.write(str(run_dir / "19_demo_report.html"))


def _render_sidebar_run_summary(run_dir: Path | None) -> None:
    st.sidebar.header("Run Summary")
    if run_dir is None or not run_dir.exists():
        st.sidebar.caption("Current active run:")
        st.sidebar.write("No active run selected.")
        return

    summary = load_json(run_dir / "17_topk_final_summary.json", default={}) or {}
    video_info = load_json(run_dir / "01_video_info.json", default={}) or {}
    compiled_manifest = (
        load_json(run_dir / "18_compiled_review_video.json", default=None)
        or load_json(run_dir / "18_exported_clips" / "18_compiled_review_video.json", default={})
        or {}
    )
    st.sidebar.caption("Current active run:")
    st.sidebar.write(f"`{run_dir}`")
    st.sidebar.write(f"Video: `{video_info.get('video_name', run_dir.name)}`")
    st.sidebar.write(f"Duration: `{format_duration(video_info.get('duration_seconds'))}`")
    processing = summary.get("processing_summary", {}) if isinstance(summary, dict) else {}
    st.sidebar.write(f"Priority events: `{processing.get('priority_suspicious_events', 0)}`")
    st.sidebar.write(f"Possible review clips: `{processing.get('possible_review_clips', 0)}`")
    st.sidebar.write(f"Report available: `{'yes' if (run_dir / '19_demo_report.html').exists() else 'no'}`")
    verification = compiled_manifest.get("playback_recommended_verification") or compiled_manifest.get("video_verification", {})
    compiled_available = bool(verification.get("exists")) and bool(verification.get("readable_by_opencv"))
    st.sidebar.write(f"Compiled video available: `{'yes' if compiled_available else 'no'}`")


def _render_search_tab(run_dir: Path | None) -> None:
    st.subheader("Evidence Search")
    st.caption("Pipeline mode: Optimized Top-K + Safety Guardrails + Evidence Search")

    search_scope = st.radio("Search scope", ["Current selected run", "All completed runs"], horizontal=True)
    st.text_area(
        "Natural language search",
        key="search_nl_query",
        placeholder="Example: show me the person with a red shirt carrying a bag near the counter after 01:20",
        height=90,
    )
    interpret_clicked = st.button("Interpret query")
    if interpret_clicked:
        interpreted = interpret_nl_search_query(st.session_state.get("search_nl_query", ""))
        st.session_state["search_interpreted_filters"] = interpreted
        st.session_state["search_query"] = ""
        st.session_state["search_object_type"] = ""
        st.session_state["search_person_appearance"] = ""
        st.session_state["search_vehicle"] = ""
        st.session_state["search_event_type"] = "All"
        st.session_state["search_category"] = "All"
        st.session_state["search_risk_level"] = "All"
        st.session_state["search_time_start"] = 0.0
        st.session_state["search_time_end"] = 0.0
        if interpreted.get("search_terms"):
            st.session_state["search_query"] = " ".join(interpreted["search_terms"])
        if interpreted.get("object_type"):
            st.session_state["search_object_type"] = str(interpreted["object_type"])
        if interpreted.get("person_appearance"):
            st.session_state["search_person_appearance"] = str(interpreted["person_appearance"])
        if interpreted.get("vehicle"):
            st.session_state["search_vehicle"] = str(interpreted["vehicle"])
        if interpreted.get("event_type") and interpreted.get("event_type") != "All":
            st.session_state["search_event_type"] = str(interpreted["event_type"])
        if interpreted.get("time_start") is not None:
            st.session_state["search_time_start"] = float(interpreted["time_start"])
        if interpreted.get("time_end") is not None:
            st.session_state["search_time_end"] = float(interpreted["time_end"])

    interpreted_filters = st.session_state.get("search_interpreted_filters", {})
    if interpreted_filters:
        st.caption("Parsed filter JSON")
        st.json(interpreted_filters)

    query = st.text_input(
        "Free text search",
        placeholder="Search: red cap, bag, display case, robbery, fight, accident, person bending...",
        key="search_query",
    )
    st.caption("Examples: `red cap`, `display case`, `bag`, `person bending`, `suspicious reaching`, `robbery`, `collision`, `crowding`, `white shirt`")

    filter_cols = st.columns(3)
    with filter_cols[0]:
        category = st.selectbox("Category", ["All", "Priority suspicious event", "Possible review clip", "Normal activity", "Uncertain activity", "Object evidence"], key="search_category")
        risk_level = st.selectbox("Risk level", ["All", "low", "medium", "high", "unknown"], key="search_risk_level")
        event_type = st.selectbox(
            "Event/activity type",
            [
                "All",
                "robbery",
                "theft",
                "suspicious reaching",
                "fight",
                "fall",
                "collision",
                "accident",
                "intrusion",
                "crowding",
                "normal activity",
                "person object interaction",
                "bending",
                "reaching",
                "walking",
                "standing",
            ],
            key="search_event_type",
        )
    with filter_cols[1]:
        object_type = st.text_input("Object type", placeholder="bag, backpack, bottle, laptop, phone, display case, jewelry, vehicle, car, bike", key="search_object_type")
        person_appearance = st.text_input("Person appearance / clothing", placeholder="red cap, black shirt, white shirt, dark clothing", key="search_person_appearance")
        vehicle = st.text_input("Vehicle", placeholder="car, bike, truck, white car", key="search_vehicle")
    with filter_cols[2]:
        time_start = st.number_input("Start seconds", min_value=0.0, step=1.0, key="search_time_start")
        time_end = st.number_input("End seconds", min_value=0.0, step=1.0, key="search_time_end")
        st.caption("Vehicle speed search is not available yet because speed needs tracking across frames. This can be added in a future tracking step.")

    debug_runs_dir = project_root() / "tests" / "tender_demo_case" / "debug_runs"
    current_run_str = str(run_dir) if run_dir and run_dir.exists() else ""
    cache_buster = _compute_search_cache_buster(debug_runs_dir, search_scope, run_dir if run_dir and run_dir.exists() else None)
    records = load_all_search_records(str(debug_runs_dir), search_scope, current_run_str, cache_buster)
    if search_scope == "Current selected run" and run_dir is None:
        st.info("Select or run a debug session first, or change scope to All completed runs.")
        return

    video_options = ["All videos"]
    if search_scope == "All completed runs":
        seen = []
        for record in records:
            for candidate in [record.get("video_name"), record.get("run_name")]:
                if candidate and candidate not in seen:
                    seen.append(candidate)
        video_options.extend(seen)
    selected_videos = st.multiselect(
        "Video selection",
        video_options,
        default=["All videos"],
        help="Choose All videos, one video, or multiple videos to search across.",
    )
    if not selected_videos:
        selected_videos = ["All videos"]
    elif "All videos" in selected_videos and len(selected_videos) > 1:
        selected_videos = ["All videos"]

    search_clicked = st.button("Search")
    filters = {
        "query": query,
        "category": category,
        "risk_level": risk_level,
        "event_type": event_type,
        "object_type": object_type,
        "person_appearance": person_appearance,
        "vehicle": vehicle,
        "selected_videos": selected_videos,
        "time_start": time_start if time_start > 0 else None,
        "time_end": time_end if time_end > 0 else None,
    }
    results = search_records(records, filters) if search_clicked or query or category != "All" or risk_level != "All" else search_records(records, filters)

    st.write(f"Found {len(results)} result(s)")
    if not results:
        st.info("No matching analyzed event found. Try searching by visible object, clothing color, activity, or risk category.")
        return

    for record in results:
        with st.container(border=True):
            _render_search_result(record)


def _initialize_state() -> None:
    st.session_state.setdefault("active_run_dir", "")
    st.session_state.setdefault("run_dir_input", "")
    st.session_state.setdefault("last_detected_run_dir", "")
    st.session_state.setdefault("latest_logs", "")
    st.session_state.setdefault("last_pipeline_result", None)
    st.session_state.setdefault("uploaded_video_path", "")
    st.session_state.setdefault("pipeline_completed", False)
    st.session_state.setdefault("current_stage", "Waiting to start")
    st.session_state.setdefault("progress_percent", 0)
    st.session_state.setdefault("pipeline_running", False)
    st.session_state.setdefault("pipeline_process_pid", None)
    st.session_state.setdefault("search_nl_query", "")
    st.session_state.setdefault("search_interpreted_filters", {})
    st.session_state.setdefault("search_query", "")
    st.session_state.setdefault("search_category", "All")
    st.session_state.setdefault("search_risk_level", "All")
    st.session_state.setdefault("search_event_type", "All")
    st.session_state.setdefault("search_object_type", "")
    st.session_state.setdefault("search_person_appearance", "")
    st.session_state.setdefault("search_vehicle", "")
    st.session_state.setdefault("search_time_start", 0.0)
    st.session_state.setdefault("search_time_end", 0.0)


def main() -> None:
    st.set_page_config(page_title="Tender Demo Video Analysis UI", layout="wide")
    _initialize_state()

    st.title("Tender Demo Video Analysis UI")
    st.info("Pipeline mode: Optimized Top-K + Safety Guardrails + Evidence Search")

    with st.sidebar:
        st.header("Upload / Input")
        pipeline_engine = st.selectbox("Pipeline engine", PIPELINE_ENGINES, index=0)
        st.caption(PIPELINE_ENGINE_MAP[pipeline_engine]["description"])
        if pipeline_engine == "Standard demo pipeline":
            st.warning("Standard demo pipeline can take longer than the video length because it may run extra compatibility steps. Use Fast parallel Top-K for faster processing.")
        else:
            st.caption("Standard pipeline is slower and mainly kept for compatibility/debugging. For tender demo processing, use Fast parallel Top-K pipeline.")
        processing_preset = st.selectbox("Processing preset", list(PROCESSING_PRESETS.keys()), index=2)
        st.info("For large CCTV videos, use Existing video path or Import folder. Browser upload has Streamlit limits and can be slow. The analysis pipeline itself can process large files from disk.")
        input_mode = st.radio("Video input mode", INPUT_MODE_OPTIONS, index=0)
        uploaded_file = st.file_uploader(
            "Upload video",
            type=sorted(SUPPORTED_EXTENSIONS),
        )
        existing_video_path_input = st.text_input(
            "Existing video path",
            placeholder=r"C:\Videos\camera_01.mp4",
        )
        import_folder = import_folder_path()
        st.caption(f"Import folder path: `{import_folder}`")
        st.caption("Copy large videos into this folder, then click Refresh.")
        refresh_import_folder = st.button("Refresh import folder")
        imported_videos = list_import_folder_videos()
        imported_video_labels = [
            f"{path.name} | {format_file_size(path.stat().st_size)} | {datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}"
            for path in imported_videos
        ]
        selected_import_label = st.selectbox(
            "Select imported video",
            ["None"] + imported_video_labels,
            index=0,
        )
        show_debug_logs = st.checkbox("Show debug logs", value=False)

        st.header("Pipeline Settings")
        preset_values = PROCESSING_PRESETS[processing_preset]
        robbery_demo_mode = processing_preset == "Jewelry shop robbery demo"
        quick_result_mode = st.checkbox("Quick result mode", value=False)
        if robbery_demo_mode and quick_result_mode:
            st.warning("Quick result mode overrides this robbery-demo preset toward Quick scan behavior. Turn it OFF for jewelry robbery/theft analysis.")
        st.caption("Quick result mode scans the video, selects the most important few clips, and sends only those clips to Qwen. This is faster but less exhaustive.")
        st.subheader("Analysis sensitivity")
        st.caption("Fast modes review fewer frames and fewer clips. They are good for normal traffic/road videos. Sensitive modes sample more frames and send more clips to Qwen. Use Jewelry shop robbery demo, Sensitive Incident Review, or High Accuracy Review for robbery, theft, fight, fall, collision, or other subtle incidents.")
        is_custom_mode = processing_preset == "Custom"
        if is_custom_mode:
            sample_every_seconds = st.slider(
                "Frame sampling interval, seconds",
                min_value=0.25,
                max_value=10.0,
                value=float(preset_values["sample_every_seconds"]),
                step=0.25,
                help="Lower value means more frames are checked. Better for robbery, fight, theft, fall, and subtle incidents, but slower.",
            )
            sampled_fps = 1.0 / sample_every_seconds if sample_every_seconds > 0 else 0.0
            st.caption(f"Equivalent analysis sampled FPS: `{sampled_fps:.3f}`")
            st.caption("This is analysis sampling FPS, not the original video FPS.")
            top_k = st.slider(
                "Top-K clips sent to Qwen",
                min_value=3,
                max_value=25,
                value=int(preset_values["top_k"]),
                step=1,
                help="More clips means Qwen reviews more moments. Better for subtle incidents, but slower.",
            )
            motion_threshold = st.slider(
                "Motion threshold",
                min_value=0.05,
                max_value=0.50,
                value=float(preset_values["motion_threshold"]),
                step=0.01,
                help="Lower threshold catches smaller movement but may include more normal activity.",
            )
            qwen_max_new_tokens = st.slider("Qwen max tokens", min_value=128, max_value=768, value=int(preset_values["qwen_max_new_tokens"]), step=64)
        else:
            sample_every_seconds = float(preset_values["sample_every_seconds"])
            top_k = int(preset_values["top_k"])
            motion_threshold = float(preset_values["motion_threshold"])
            qwen_max_new_tokens = int(preset_values["qwen_max_new_tokens"])
            sampled_fps = 1.0 / sample_every_seconds if sample_every_seconds > 0 else 0.0
            st.write(f"Sample interval: `{sample_every_seconds}` sec")
            st.write(f"Equivalent analysis sampled FPS: `{sampled_fps:.3f}`")
            st.write(f"Top-K clips sent to Qwen: `{top_k}`")
            st.write(f"Motion threshold: `{motion_threshold}`")
        qwen_model_id = st.text_input("Qwen model id", value="qwen2.5vl:7b")
        qwen_batch_size = st.number_input("Qwen batch size", min_value=1, value=int(preset_values["qwen_batch_size"]), step=1)
        st.caption("Batch size 4 may be slower or unstable depending on GPU memory. Start with 1 or 2.")
        run_yolo = st.checkbox("Run YOLO", value=True)
        yolo_model = st.text_input("YOLO model", value="yolov8n.pt")
        if is_custom_mode:
            yolo_conf = st.slider("YOLO confidence", min_value=0.10, max_value=0.60, value=float(preset_values["yolo_conf"]), step=0.05)
            yolo_imgsz = st.selectbox("YOLO image size", [416, 512, 640], index=[416, 512, 640].index(int(preset_values["yolo_imgsz"])))
        else:
            yolo_conf = float(preset_values["yolo_conf"])
            yolo_imgsz = int(preset_values["yolo_imgsz"])
            st.write(f"YOLO image size: `{yolo_imgsz}`")
            st.write(f"YOLO confidence: `{yolo_conf}`")
        parallel_branches = st.checkbox("Enable fast parallel branches", value=bool(preset_values["parallel_branches"]))
        sensitive_mode = processing_preset in {"Sensitive Incident Review", "High accuracy review"}
        if is_custom_mode:
            incident_fallback_pass = st.checkbox(
                "Incident-sensitive fallback pass",
                value=bool(preset_values.get("incident_fallback_pass", False)),
                help="If no priority/review clips are found, the pipeline may retry with more Top-K clips. This is useful for subtle incidents but slower.",
            )
            enable_incident_recheck = st.checkbox(
                "Enable incident recheck",
                value=bool(preset_values.get("enable_incident_recheck", False)),
                help="Reserved for Step 16B incident reasoning. If Step 16B is not implemented yet, this setting will be recorded but skipped safely.",
            )
            incident_recheck_all_topk = st.checkbox(
                "Recheck all Top-K clips",
                value=bool(preset_values.get("incident_recheck_all_topk", False)),
                help="Reserved for high-accuracy incident reasoning.",
            )
            incident_focus_label = st.selectbox(
                "Incident focus",
                list(INCIDENT_FOCUS_OPTIONS.keys()),
                index=list(INCIDENT_FOCUS_OPTIONS.values()).index(str(preset_values.get("incident_focus", "general"))),
                help="Choose the main incident family the serious-incident recheck should prioritize.",
            )
            incident_focus = INCIDENT_FOCUS_OPTIONS[incident_focus_label]
        else:
            incident_fallback_pass = bool(preset_values.get("incident_fallback_pass", False))
            enable_incident_recheck = bool(preset_values.get("enable_incident_recheck", False))
            incident_recheck_all_topk = bool(preset_values.get("incident_recheck_all_topk", False))
            incident_focus = str(preset_values.get("incident_focus", "general"))
            st.write(f"Incident-sensitive fallback pass: `{incident_fallback_pass}`")
            st.write(f"Enable incident recheck: `{enable_incident_recheck}`")
            st.write(f"Recheck all Top-K clips: `{incident_recheck_all_topk}`")
            st.write(f"Incident focus: `{incident_focus}`")
        if is_custom_mode or sensitive_mode:
            adaptive_sampling_enabled = st.checkbox(
                "Enable adaptive sampling",
                value=bool(preset_values.get("adaptive_sampling_enabled", False)),
            )
            adaptive_base_interval_seconds = st.number_input(
                "Adaptive base interval seconds",
                min_value=0.25,
                value=float(preset_values.get("adaptive_base_interval_seconds", 1.0)),
                step=0.25,
            )
            adaptive_max_frame_gap_seconds = st.number_input(
                "Adaptive max frame gap seconds",
                min_value=1.0,
                value=float(preset_values.get("adaptive_max_frame_gap_seconds", 4.0)),
                step=0.5,
            )
            coverage_guardrails_enabled = st.checkbox(
                "Enable coverage guardrails",
                value=bool(preset_values.get("coverage_guardrails_enabled", False)),
            )
            critical_timestamps = st.text_input(
                "Critical timestamps",
                value=str(preset_values.get("critical_timestamps", "")),
                placeholder="00:21,01:00,01:12,02:18",
            )
            critical_window_seconds = st.number_input(
                "Critical window seconds",
                min_value=1.0,
                value=float(preset_values.get("critical_window_seconds", 8.0)),
                step=1.0,
            )
            vlm_input_strategy = st.selectbox(
                "VLM input strategy",
                ["center_only", "peak_motion", "adaptive_peak", "multi_focus"],
                index=["center_only", "peak_motion", "adaptive_peak", "multi_focus"].index(str(preset_values.get("vlm_input_strategy", "center_only"))),
            )
            max_vlm_inputs = st.number_input(
                "Max VLM inputs",
                min_value=5,
                max_value=80,
                value=int(preset_values.get("max_vlm_inputs", 25)),
                step=1,
            )
        else:
            adaptive_sampling_enabled = bool(preset_values.get("adaptive_sampling_enabled", False))
            adaptive_base_interval_seconds = float(preset_values.get("adaptive_base_interval_seconds", 1.0))
            adaptive_max_frame_gap_seconds = float(preset_values.get("adaptive_max_frame_gap_seconds", 4.0))
            coverage_guardrails_enabled = bool(preset_values.get("coverage_guardrails_enabled", False))
            critical_timestamps = str(preset_values.get("critical_timestamps", ""))
            critical_window_seconds = float(preset_values.get("critical_window_seconds", 8.0))
            vlm_input_strategy = str(preset_values.get("vlm_input_strategy", "center_only"))
            max_vlm_inputs = int(preset_values.get("max_vlm_inputs", 25))
            st.write(f"Adaptive sampling enabled: `{adaptive_sampling_enabled}`")
            st.write(f"Adaptive base interval seconds: `{adaptive_base_interval_seconds}`")
            st.write(f"Adaptive max frame gap seconds: `{adaptive_max_frame_gap_seconds}`")
            st.write(f"Coverage guardrails enabled: `{coverage_guardrails_enabled}`")
            st.write(f"Critical timestamps: `{critical_timestamps or 'none'}`")
            st.write(f"Critical window seconds: `{critical_window_seconds}`")
            st.write(f"VLM input strategy: `{vlm_input_strategy}`")
            st.write(f"Max VLM inputs: `{max_vlm_inputs}`")
        if top_k >= 20 or sample_every_seconds <= 1.0:
            st.warning("Sensitive settings may be slower because more frames are sampled and more clips are sent to Qwen.")
        if sample_every_seconds >= 3.0:
            st.warning("Sparse sampling can miss short actions. For robbery/theft/fight videos, use 1.0 sec or 0.5 sec sampling.")
        create_compiled_review_video = st.checkbox("Create compiled review video", value=True)
        compiled_video_fps = st.number_input("Compiled video FPS", min_value=1, value=5, step=1)
        seconds_per_frame = st.number_input("Seconds per frame", min_value=0.1, value=1.0, step=0.1)
        seconds_per_title_card = st.number_input("Seconds per title card", min_value=0.1, value=1.5, step=0.1)
        max_video_seconds = st.text_input("Process first N seconds only", value="", placeholder="Leave empty for full video")
        st.caption("For testing speed, process only the first N seconds. Leave empty to process full video.")
        st.caption("Large videos are processed by sampling and Top-K selection. The full video is not sent to Qwen.")
        st.caption("Speed recommendations:")
        st.caption("For 1-5 minute videos: Balanced or Fast demo")
        st.caption("For 30-60 minute videos: Fast demo, sample every 4-5 seconds, Top-K 5, Qwen tokens 192-256")
        st.caption("For multi-hour CCTV: Use existing file path/import folder, sample every 5-10 seconds, run quick result first")

        st.header("Existing Run Viewer")
        active_run_value = st.session_state.get("active_run_dir", "").strip()
        st.caption("Current active run:")
        if active_run_value:
            st.code(active_run_value)
        else:
            st.caption("No active run selected.")

        last_detected_run_dir = st.session_state.get("last_detected_run_dir", "").strip()
        if last_detected_run_dir:
            st.caption("Last detected run:")
            st.code(last_detected_run_dir)

        st.text_input(
            "Existing run folder",
            key="run_dir_input",
            placeholder=r"C:\...\tests\tender_demo_case\debug_runs\run_name",
        )
        if st.button("Load existing run"):
            run_dir_value = st.session_state.get("run_dir_input", "").strip()
            candidate = Path(run_dir_value) if run_dir_value else None
            if candidate and candidate.exists() and candidate.is_dir():
                st.session_state["active_run_dir"] = str(candidate)
                st.session_state["pipeline_completed"] = True
                st.success("Run loaded.")
            else:
                st.error("Invalid run folder.")
        if st.button("Use last detected run"):
            if last_detected_run_dir:
                candidate = Path(last_detected_run_dir)
                if candidate.exists() and candidate.is_dir():
                    st.session_state["active_run_dir"] = str(candidate)
                    st.session_state["pipeline_completed"] = True
                    st.success("Last detected run activated.")
                else:
                    st.error("Last detected run folder is no longer available.")
            else:
                st.warning("No detected run is available yet.")
        if st.button("Use latest completed run"):
            latest_run = find_latest_debug_run()
            if latest_run is not None:
                st.session_state["active_run_dir"] = str(latest_run)
                st.session_state["last_detected_run_dir"] = str(latest_run)
                st.session_state["pipeline_completed"] = True
                st.success("Latest run activated.")
            else:
                st.warning("No debug run directories found.")

    settings = {
        "pipeline_engine": pipeline_engine,
        "pipeline_engine_id": PIPELINE_ENGINE_MAP[pipeline_engine]["engine_id"],
        "analysis_sensitivity_mode": processing_preset,
        "processing_preset": processing_preset,
        "quick_result_mode": quick_result_mode,
        "sample_every_seconds": sample_every_seconds,
        "top_k": int(top_k),
        "top_k_max": 25,
        "motion_threshold": motion_threshold,
        "qwen_model_id": qwen_model_id,
        "qwen_max_new_tokens": int(qwen_max_new_tokens),
        "qwen_batch_size": int(qwen_batch_size),
        "run_yolo": run_yolo,
        "yolo_model": yolo_model,
        "yolo_conf": yolo_conf,
        "yolo_imgsz": int(yolo_imgsz),
        "parallel_branches": parallel_branches,
        "incident_fallback_pass": incident_fallback_pass,
        "enable_incident_recheck": enable_incident_recheck,
        "incident_recheck_all_topk": incident_recheck_all_topk,
        "incident_focus": incident_focus,
        "adaptive_sampling_enabled": adaptive_sampling_enabled,
        "adaptive_base_interval_seconds": adaptive_base_interval_seconds,
        "adaptive_max_frame_gap_seconds": adaptive_max_frame_gap_seconds,
        "coverage_guardrails_enabled": coverage_guardrails_enabled,
        "critical_timestamps": critical_timestamps,
        "critical_window_seconds": critical_window_seconds,
        "vlm_input_strategy": vlm_input_strategy,
        "max_vlm_inputs": int(max_vlm_inputs),
        "yolo_input_scope": "frame_candidate_pool" if adaptive_sampling_enabled else str(preset_values.get("yolo_input_scope", "motion_candidates")),
        "create_compiled_review_video": create_compiled_review_video,
        "compiled_video_fps": int(compiled_video_fps),
        "seconds_per_frame": seconds_per_frame,
        "seconds_per_title_card": seconds_per_title_card,
        "max_video_seconds": max_video_seconds,
    }
    if robbery_demo_mode:
        settings["quick_result_mode"] = False
    if quick_result_mode:
        settings["sample_every_seconds"] = QUICK_RESULT_SETTINGS["sample_every_seconds"]
        settings["top_k"] = QUICK_RESULT_SETTINGS["top_k"]
        settings["top_k_max"] = 25
        settings["motion_threshold"] = QUICK_RESULT_SETTINGS["motion_threshold"]
        settings["qwen_max_new_tokens"] = QUICK_RESULT_SETTINGS["qwen_max_new_tokens"]
        settings["yolo_imgsz"] = QUICK_RESULT_SETTINGS["yolo_imgsz"]
        settings["yolo_conf"] = QUICK_RESULT_SETTINGS["yolo_conf"]
        settings["analysis_sensitivity_mode"] = "Quick scan"
        settings["pipeline_engine"] = "Fast parallel Top-K pipeline"
        settings["pipeline_engine_id"] = PIPELINE_ENGINE_MAP["Fast parallel Top-K pipeline"]["engine_id"]
        settings["parallel_branches"] = True
        settings["incident_fallback_pass"] = False
        settings["enable_incident_recheck"] = False
        settings["incident_recheck_all_topk"] = False
        settings["incident_focus"] = "general"
        settings["adaptive_sampling_enabled"] = False
        settings["coverage_guardrails_enabled"] = False
        settings["critical_timestamps"] = ""
        settings["vlm_input_strategy"] = "center_only"
        settings["max_vlm_inputs"] = 25
        settings["yolo_input_scope"] = "motion_candidates"
    if robbery_demo_mode:
        settings["quick_result_mode"] = False
        settings["sample_every_seconds"] = float(preset_values["sample_every_seconds"])
        settings["top_k"] = int(preset_values["top_k"])
        settings["top_k_max"] = 25
        settings["motion_threshold"] = float(preset_values["motion_threshold"])
        settings["qwen_max_new_tokens"] = int(preset_values["qwen_max_new_tokens"])
        settings["yolo_imgsz"] = int(preset_values["yolo_imgsz"])
        settings["yolo_conf"] = float(preset_values["yolo_conf"])
        settings["analysis_sensitivity_mode"] = processing_preset
        settings["pipeline_engine"] = "Fast parallel Top-K pipeline"
        settings["pipeline_engine_id"] = PIPELINE_ENGINE_MAP["Fast parallel Top-K pipeline"]["engine_id"]
        settings["parallel_branches"] = bool(preset_values["parallel_branches"])
        settings["incident_fallback_pass"] = bool(preset_values["incident_fallback_pass"])
        settings["enable_incident_recheck"] = bool(preset_values["enable_incident_recheck"])
        settings["incident_recheck_all_topk"] = bool(preset_values["incident_recheck_all_topk"])
        settings["incident_focus"] = str(preset_values["incident_focus"])
        settings["adaptive_sampling_enabled"] = bool(preset_values["adaptive_sampling_enabled"])
        settings["adaptive_base_interval_seconds"] = float(preset_values["adaptive_base_interval_seconds"])
        settings["adaptive_max_frame_gap_seconds"] = float(preset_values["adaptive_max_frame_gap_seconds"])
        settings["coverage_guardrails_enabled"] = bool(preset_values["coverage_guardrails_enabled"])
        settings["critical_timestamps"] = str(preset_values["critical_timestamps"])
        settings["critical_window_seconds"] = float(preset_values["critical_window_seconds"])
        settings["vlm_input_strategy"] = str(preset_values["vlm_input_strategy"])
        settings["max_vlm_inputs"] = int(preset_values["max_vlm_inputs"])
        settings["yolo_input_scope"] = str(preset_values["yolo_input_scope"])
    effective_pipeline_engine = settings["pipeline_engine"]
    st.session_state["pipeline_engine"] = effective_pipeline_engine

    selected_video_path: Path | None = None
    input_mode_display = input_mode
    input_access_mode = "read_directly_from_disk"
    if input_mode == "Upload video file":
        st.caption("Browser upload is best for small/medium videos. For large CCTV files, use Existing video path or Import folder.")
        input_access_mode = "copied_from_browser_upload"
        if uploaded_file is not None:
            if Path(uploaded_file.name).suffix.lower().lstrip(".") not in SUPPORTED_EXTENSIONS:
                st.error("Unsupported file extension.")
                st.stop()
            if st.session_state.get("uploaded_video_path") and Path(st.session_state["uploaded_video_path"]).exists():
                selected_video_path = Path(st.session_state["uploaded_video_path"])
            else:
                selected_video_path = save_uploaded_video(uploaded_file)
                st.session_state["uploaded_video_path"] = str(selected_video_path)
    elif input_mode == "Use existing local/server video path":
        if existing_video_path_input.strip():
            candidate = Path(existing_video_path_input.strip()).expanduser()
            if candidate.exists() and candidate.is_file() and candidate.suffix.lower().lstrip(".") in SUPPORTED_EXTENSIONS:
                selected_video_path = candidate
                st.success("Using video directly from disk. No browser upload/copy required.")
            elif candidate.exists() and not candidate.is_file():
                st.error("Existing video path must point to a file.")
            elif candidate.suffix and candidate.suffix.lower().lstrip(".") not in SUPPORTED_EXTENSIONS:
                st.error("Unsupported video extension for Existing video path.")
    elif input_mode == "Select from import folder":
        if refresh_import_folder:
            st.success("Import folder refreshed.")
        if selected_import_label != "None":
            selected_index = imported_video_labels.index(selected_import_label)
            selected_video_path = imported_videos[selected_index]

    sidebar_run_dir = get_active_run_dir()
    _render_sidebar_run_summary(sidebar_run_dir if sidebar_run_dir and sidebar_run_dir.exists() else None)

    tabs = st.tabs(["Run Pipeline", "Results Summary", "Events", "Search", "Evidence Timeline", "Files"])

    with tabs[0]:
        st.info("Processing runs in this local Streamlit session. Keep this page open until the pipeline finishes.")
        if st.session_state.get("pipeline_running"):
            running_pid = st.session_state.get("pipeline_process_pid")
            st.warning(f"Pipeline is currently running. PID: {running_pid}")
            if st.button("Stop Processing"):
                if running_pid and stop_pipeline_process(int(running_pid)):
                    st.session_state["pipeline_running"] = False
                    st.session_state["pipeline_process_pid"] = None
                    st.warning("Processing stop signal sent.")
                else:
                    st.error("Failed to stop the running pipeline process.")
        st.write("Why this is faster than manual review")
        st.caption("The fast pipeline does not try to watch every frame with Qwen. It scans the video using motion and YOLO, selects Top-K important clips, and sends only those clips to Qwen. This creates a searchable evidence report so reviewers do not need to watch the full video.")
        st.caption("Large videos are handled by sampling and Top-K selection. Processing time depends mainly on video length, GPU speed, YOLO settings, and number of selected Qwen clips.")
        if selected_video_path is not None:
            st.write(f"Selected video path: `{selected_video_path}`")
            duration_seconds = get_video_duration_seconds(selected_video_path)
            file_size_bytes = selected_video_path.stat().st_size if selected_video_path.exists() else 0
            st.write(f"File size: `{format_file_size(file_size_bytes)}`")
            st.write(f"Rough video duration: `{format_duration(duration_seconds)}`")
            st.write(f"Input mode: `{input_mode_display}`")
            st.write(
                "File handling: `"
                + ("copied into ui_uploads" if input_access_mode == "copied_from_browser_upload" else "read directly from disk")
                + "`"
            )
            if input_mode == "Upload video file" and file_size_bytes > 200 * 1024 * 1024:
                st.warning("This file is large for browser upload. Existing path/import folder is recommended.")
            if file_size_bytes > 2 * 1024 * 1024 * 1024:
                st.warning("For multi-GB CCTV files, use Existing video path or Import folder. Browser upload may be unstable.")
        else:
            st.write("Selected video path: `None`")
            duration_seconds = None
        st.write(f"Expected pipeline mode: `{effective_pipeline_engine}`")
        if quick_result_mode:
            st.info("Quick Result Mode is enabled. The UI will prioritize a fast first result using sparse sampling, small Top-K selection, and lower Qwen token limits.")
        st.write("Analysis settings to be passed:")
        st.json(
            {
                "mode": settings["analysis_sensitivity_mode"],
                "sample_every_seconds": settings["sample_every_seconds"],
                "approx_sampled_fps": round(1.0 / float(settings["sample_every_seconds"]), 3) if float(settings["sample_every_seconds"]) > 0 else 0.0,
                "top_k_clips": settings["top_k"],
                "top_k_max_clips": settings["top_k_max"],
                "motion_threshold": settings["motion_threshold"],
                "qwen_max_new_tokens": settings["qwen_max_new_tokens"],
                "yolo_imgsz": settings["yolo_imgsz"],
                "yolo_conf": settings["yolo_conf"],
                "enable_incident_recheck": settings["enable_incident_recheck"],
                "incident_recheck_all_topk": settings["incident_recheck_all_topk"],
                "incident_fallback_pass": settings["incident_fallback_pass"],
                "incident_focus": settings["incident_focus"],
                "adaptive_sampling_enabled": settings["adaptive_sampling_enabled"],
                "adaptive_base_interval_seconds": settings["adaptive_base_interval_seconds"],
                "adaptive_max_frame_gap_seconds": settings["adaptive_max_frame_gap_seconds"],
                "coverage_guardrails_enabled": settings["coverage_guardrails_enabled"],
                "critical_timestamps": settings["critical_timestamps"],
                "critical_window_seconds": settings["critical_window_seconds"],
                "vlm_input_strategy": settings["vlm_input_strategy"],
                "max_vlm_inputs": settings["max_vlm_inputs"],
                "yolo_input_scope": settings["yolo_input_scope"],
            }
        )

        run_clicked = st.button("Run Tender Demo Pipeline")
        if run_clicked:
            if selected_video_path is None:
                st.error("Please upload a video or provide a valid existing video path before running the pipeline.")
                st.stop()

            estimated_seconds = 60 + ((duration_seconds or 0.0) * 0.5) + (settings["top_k"] * 15)
            estimated_seconds = max(120.0, estimated_seconds)

            placeholders = {
                "status_placeholder": st.empty(),
                "message_placeholder": st.empty(),
                "progress_bar": st.progress(0),
                "eta_placeholder": st.empty(),
                "output_placeholder": st.empty(),
                "estimated_seconds": estimated_seconds,
            }

            env = build_pipeline_env(settings, selected_video_path)
            command = [sys.executable, PIPELINE_ENGINE_MAP[effective_pipeline_engine]["script_path"]]
            stage_weights = STANDARD_STAGE_WEIGHTS if effective_pipeline_engine == "Standard demo pipeline" else FAST_STAGE_WEIGHTS
            result = run_pipeline_with_live_logs(
                command=command,
                env=env,
                cwd=project_root(),
                placeholders=placeholders,
                stage_weights=stage_weights,
            )
            st.session_state["latest_logs"] = result["logs"]
            st.session_state["last_pipeline_result"] = result
            if result.get("detected_run_dir"):
                detected_run_dir_str = str(result["detected_run_dir"])
                st.session_state["active_run_dir"] = detected_run_dir_str
                st.session_state["last_detected_run_dir"] = detected_run_dir_str
                st.session_state["pipeline_completed"] = True
                try:
                    detected_run = Path(detected_run_dir_str)
                    (detected_run / "20_ui_pipeline_run.log").write_text(result["logs"], encoding="utf-8")
                except Exception:
                    pass

            if result["return_code"] == 0:
                placeholders["status_placeholder"].success("Current stage: Pipeline completed")
                placeholders["message_placeholder"].success("Pipeline completed successfully.")
                placeholders["progress_bar"].progress(1.0)
                placeholders["eta_placeholder"].caption(
                    f"Elapsed: {format_duration(result.get('elapsed_seconds'))} | Estimated remaining: complete | Progress: 100%\n"
                    "Estimated time is approximate and depends mainly on video length, GPU speed, YOLO, and Qwen."
                )
                detected_run_path = result.get("detected_run_dir", "not detected")
                st.caption(
                    f"Elapsed time: {format_duration(result.get('elapsed_seconds'))} | "
                    f"Run folder: {detected_run_path}"
                )
                if result.get("detected_run_dir"):
                    detected_run = Path(result["detected_run_dir"])
                    _render_success_panel(detected_run)
            else:
                st.session_state["pipeline_completed"] = False
                st.error(f"Pipeline failed. Return code: {result['return_code']}")
                st.caption(f"Detected run folder: {result.get('detected_run_dir', 'not detected')}")
                with st.expander("Last 40 log lines"):
                    last_lines = result["logs"].splitlines()[-40:]
                    st.code("\n".join(last_lines), language="text")
                if result.get("detected_run_dir"):
                    st.write(f"Saved log path: `{Path(result['detected_run_dir']) / '20_ui_pipeline_run.log'}`")

            if show_debug_logs:
                with st.expander("Debug logs"):
                    st.code("\n".join(result["logs"].splitlines()[-80:]), language="text")
    with tabs[1]:
        active_run_dir = get_active_run_dir()
        results = _load_run_results(active_run_dir) if active_run_dir else None
        if results is None:
            st.warning("Load or run a debug session to view results.")
        else:
            _render_results_summary(active_run_dir, results)

    with tabs[2]:
        active_run_dir = get_active_run_dir()
        results = _load_run_results(active_run_dir) if active_run_dir else None
        if results is None:
            st.warning("Load or run a debug session to view events.")
        else:
            _render_events_tab(active_run_dir, results)

    with tabs[3]:
        _render_search_tab(get_active_run_dir())

    with tabs[4]:
        active_run_dir = get_active_run_dir()
        results = _load_run_results(active_run_dir) if active_run_dir else None
        if results is None:
            st.warning("Load or run a debug session to view the evidence timeline.")
        else:
            _render_timeline_tab(results)

    with tabs[5]:
        active_run_dir = get_active_run_dir()
        if active_run_dir is None or not active_run_dir.exists():
            st.warning("Load or run a debug session to view files and logs.")
        else:
            _render_files_tab(active_run_dir, st.session_state.get("latest_logs", ""))


if __name__ == "__main__":
    main()
