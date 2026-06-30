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
    3: 5,
    4: 3,
    5: 4,
    6: 3,
    7: 6,
    10: 15,
    11: 8,
    12: 4,
    13: 4,
    14: 3,
    15: 5,
    16: 25,
    17: 4,
    18: 6,
    19: 2,
}
STANDARD_STAGE_LABELS = {
    1: "Reading video information",
    2: "Sampling frames",
    3: "Scoring motion",
    4: "Selecting motion candidates",
    5: "Grouping motion into clips",
    6: "Expanding clips with context",
    7: "Creating VLM temporal strips",
    10: "Running YOLO detection",
    11: "Scoring YOLO object evidence",
    12: "Fusing motion + YOLO + VLM evidence",
    13: "Ranking candidate clips",
    14: "Selecting Top-K + guardrail clips",
    15: "Creating Top-K VLM inputs",
    16: "Running Qwen on Top-K clips",
    17: "Creating final summary",
    18: "Exporting/compiling review video",
    19: "Creating HTML demo report",
}
STANDARD_STAGE_PROGRESS_PERCENT = {
    1: 3,
    2: 8,
    3: 13,
    4: 18,
    5: 22,
    6: 26,
    7: 31,
    10: 45,
    11: 55,
    12: 60,
    13: 66,
    14: 70,
    15: 75,
    16: 85,
    17: 92,
    18: 97,
    19: 99,
}
FAST_STAGE_WEIGHTS = {
    1: 5,
    2: 5,
    3: 5,
    4: 5,
    56: 35,
    13: 7,
    14: 6,
    15: 5,
    16: 15,
    17: 5,
    18: 4,
    19: 3,
}
FAST_STAGE_PROGRESS_PERCENT = {
    1: 5,
    2: 10,
    3: 15,
    4: 20,
    56: 55,
    13: 62,
    14: 68,
    15: 73,
    16: 88,
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
    "Fast demo": {
        "sample_every_seconds": 3.0,
        "top_k": 5,
        "qwen_max_new_tokens": 256,
        "qwen_batch_size": 1,
        "yolo_imgsz": 416,
        "yolo_conf": 0.35,
        "parallel_branches": True,
    },
    "Balanced": {
        "sample_every_seconds": 2.0,
        "top_k": 8,
        "qwen_max_new_tokens": 384,
        "qwen_batch_size": 1,
        "yolo_imgsz": 512,
        "yolo_conf": 0.30,
        "parallel_branches": True,
    },
    "Higher accuracy": {
        "sample_every_seconds": 1.0,
        "top_k": 10,
        "qwen_max_new_tokens": 512,
        "qwen_batch_size": 1,
        "yolo_imgsz": 640,
        "yolo_conf": 0.25,
        "parallel_branches": True,
    },
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
        "TENDER_DEMO_SAMPLE_EVERY_SECONDS": str(settings["sample_every_seconds"]),
        "TENDER_DEMO_TOP_K_CLIPS": str(settings["top_k"]),
        "TENDER_DEMO_QWEN_MODEL_ID": str(settings["qwen_model_id"]),
        "TENDER_DEMO_QWEN_BATCH_SIZE": str(settings["qwen_batch_size"]),
        "TENDER_DEMO_QWEN_MAX_NEW_TOKENS": str(settings["qwen_max_new_tokens"]),
        "TENDER_DEMO_RUN_YOLO": "true" if settings["run_yolo"] else "false",
        "TENDER_DEMO_YOLO_MODEL": str(settings["yolo_model"]),
        "TENDER_DEMO_YOLO_CONF": str(settings["yolo_conf"]),
        "TENDER_DEMO_YOLO_IMGSZ": str(settings["yolo_imgsz"]),
        "TENDER_DEMO_FAST_PARALLEL_BRANCHES": "true" if settings["parallel_branches"] else "false",
        "TENDER_DEMO_QUICK_RESULT_MODE": "true" if settings.get("quick_result_mode") else "false",
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
        (["Starting Step 3", "03_motion_scores.json"], "Scoring motion", 15),
        (["Starting Step 4", "04_motion_candidates.json"], "Selecting motion candidates", 20),
        (["Starting Step 5", "05_candidate_clips.json"], "Grouping candidate clips", 25),
        (["Starting Step 6", "06_expanded_clips.json"], "Expanding clips", 30),
        (["Starting Step 10", "10_yolo_detections.json"], "Running YOLO object detection", 45),
        (["Starting Step 11", "11_yolo_object_scores.json"], "Scoring YOLO evidence", 55),
        (["Starting Step 13", "13_ranked_clips.json"], "Ranking candidate clips", 62),
        (["Starting Step 14", "14_selected_top_clips.json"], "Selecting Top-K + guardrail clips", 68),
        (["Starting Step 15", "15_topk_vlm_inputs.json"], "Creating Top-K VLM inputs", 73),
        (["Starting Step 16", "16_topk_vlm_outputs.json"], "Running Qwen on selected clips", 88),
        (["Starting Step 17", "17_topk_final_summary.json", "17_topk_final_summary.md"], "Creating final summary", 93),
        (["Starting Step 18", "18_compiled_review_video.json", "18_exported_clips.json"], "Creating compiled review video", 97),
        (["Starting Step 19", "19_demo_report.html"], "Creating HTML report", 99),
    ]
    fast_stage_map = [
        (["Starting Step 1", "01_video_info.json"], "Reading video information", 5),
        (["Starting Step 2", "02_sampled_frames.json"], "Sampling frames", 10),
        (["Starting Step 3", "03_motion_scores.json"], "Scoring motion", 15),
        (["Starting Step 4", "04_motion_candidates.json"], "Selecting motion candidates", 20),
        (["Starting Step 5", "05_candidate_clips.json"], "Grouping candidate clips", 25),
        (["Starting Step 6", "06_expanded_clips.json"], "Expanding clips", 30),
        (["Starting parallel section", "Starting clip branch", "Starting YOLO branch"], "Building clips and running YOLO evidence", 55),
        (["Starting Step 10", "10_yolo_detections.json"], "Running YOLO object detection", 45),
        (["Starting Step 11", "11_yolo_object_scores.json"], "Scoring YOLO evidence", 55),
        (["Starting Step 13", "13_ranked_clips.json"], "Ranking candidate clips", 62),
        (["Starting Step 14", "14_selected_top_clips.json"], "Selecting Top-K + guardrail clips", 68),
        (["Starting Step 15", "15_topk_vlm_inputs.json"], "Creating Top-K VLM inputs", 73),
        (["Starting Step 16", "16_topk_vlm_outputs.json"], "Running Qwen on selected clips", 88),
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
        "Starting Step 3": "Scoring motion...",
        "Starting Step 4": "Selecting motion candidates...",
        "Starting Step 5": "Grouping motion into clips...",
        "Starting Step 6": "Expanding clips with context...",
        "Starting Step 7": "Creating VLM temporal strips...",
        "Starting Step 10": "Running YOLO object detection...",
        "Starting Step 11": "Scoring YOLO object evidence...",
        "Starting Step 12": "Fusing motion + YOLO + VLM evidence...",
        "Starting Step 13": "Ranking candidate clips...",
        "Starting Step 14": "Selecting Top-K + guardrail clips...",
        "Starting Step 15": "Creating Top-K VLM input strips...",
        "Starting Step 16": "Running Qwen on selected Top-K clips...",
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
    compiled_path_value = (
        compiled_info.get("playback_recommended_file")
        or compiled_info.get("compiled_video_path")
        or export_info.get("playback_recommended_file")
        or export_info.get("output_path")
        or str(run_dir / "18_exported_clips" / "18_compiled_review_video.mp4")
    )
    return resolve_media_path(run_dir, compiled_path_value)


def _timeline_description(item: dict[str, Any]) -> str:
    for key in ["best_event_description", "caption"]:
        value = str(item.get(key, "")).strip()
        if value:
            return value
    activities = item.get("activity_descriptions", [])
    if isinstance(activities, list) and activities:
        return str(activities[0])
    return "Selected clip contains visually important activity."


@st.cache_data(show_spinner=False)
def load_all_search_records(debug_runs_dir_str: str, scope: str, current_run_dir_str: str) -> list[dict]:
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


def build_or_load_search_index_for_run(run_dir: Path, force_rebuild=False) -> list[dict]:
    cache_path = run_dir / "20_search_index.json"
    summary_path = run_dir / "17_topk_final_summary.json"
    vlm_path = run_dir / "16_topk_vlm_outputs.json"
    if not summary_path.exists():
        return []

    if not force_rebuild and cache_path.exists():
        cache_mtime = cache_path.stat().st_mtime
        summary_mtime = summary_path.stat().st_mtime
        vlm_mtime = vlm_path.stat().st_mtime if vlm_path.exists() else 0.0
        if cache_mtime >= max(summary_mtime, vlm_mtime):
            cached = load_json(cache_path, default=[])
            if isinstance(cached, list):
                return cached

    records = build_search_records_for_run(run_dir)
    cache_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
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
        compiled_video_path = (
            compiled_manifest.get("playback_recommended_file")
            or compiled_manifest.get("compiled_video_path")
            or export_manifest.get("compiled_review_video", {}).get("playback_recommended_file")
            or export_manifest.get("compiled_review_video", {}).get("output_path")
        ) if isinstance(compiled_manifest, dict) and isinstance(export_manifest, dict) else None

        records.append(
            {
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
    selected_video = filters.get("selected_video", "All videos")
    time_start = filters.get("time_start")
    time_end = filters.get("time_end")

    category_map = {
        "Priority suspicious event": "priority_suspicious_event",
        "Possible review clip": "possible_review_clip",
        "Normal activity": "normal_activity",
        "Uncertain activity": "uncertain_activity",
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
            object_text = " ".join(record.get("object_names", []) + record.get("yolo_object_classes", [])).lower()
            if object_filter not in object_text:
                continue
        if appearance_filter:
            appearance_text = " ".join(record.get("person_descriptions", []) + record.get("person_clothing_text", [])).lower()
            if appearance_filter not in appearance_text:
                continue
        if vehicle_filter:
            vehicle_text = " ".join(record.get("vehicle_terms", []) + record.get("object_names", [])).lower()
            if vehicle_filter not in vehicle_text:
                continue
        if selected_video != "All videos" and selected_video not in {record.get("video_name"), record.get("run_name")}:
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
                if term in " ".join(record.get("person_descriptions", []) + record.get("person_clothing_text", [])).lower():
                    score += 5
        if record.get("final_category") == "priority_suspicious_event":
            score += 2
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


def _render_search_result(record: dict) -> None:
    st.markdown(
        f"### {record.get('clip_id', 'unknown')} | {record.get('time_range', 'unknown')} | "
        f"{record.get('video_name', 'unknown video')}"
    )
    st.write(
        {
            "run_name": record.get("run_name"),
            "final_category": record.get("final_category"),
            "risk_level": record.get("risk_level"),
            "confidence": record.get("confidence"),
            "event_label": record.get("event_label"),
            "description": record.get("description"),
            "people_count": record.get("people_count"),
            "yolo_object_classes": record.get("yolo_object_classes"),
            "person_descriptions": record.get("person_descriptions"),
            "activity_types": record.get("activity_types"),
            "event_types": record.get("event_types"),
            "matched_terms": record.get("matched_terms", []),
        }
    )

    run_dir = Path(str(record.get("run_dir")))
    media_cols = st.columns(2)
    strip_path = resolve_media_path(run_dir, record.get("strip_path"))
    yolo_path = resolve_media_path(run_dir, record.get("top_annotated_frame_path"))
    with media_cols[0]:
        if media_exists(strip_path):
            st.image(str(strip_path), caption="Incident image / temporal strip", width="stretch")
        else:
            st.warning("Incident image not found.")
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


def render_event_card(event, qwen_by_clip_id, run_dir, show_raw_qwen=False):
    clip_id = str(event.get("clip_id", "unknown_clip"))
    qwen_item = qwen_by_clip_id.get(clip_id, {})

    st.markdown(f"**{clip_id}**")
    left, right = st.columns(2)
    with left:
        st.write(f"Time: `{event.get('time_range', 'unknown')}`")
        st.write(f"Event label: `{event.get('event_label', 'unknown')}`")
        st.write(f"Risk: `{event.get('risk_level', 'unknown')}`")
        st.write(f"Confidence: `{event.get('confidence', 'unknown')}`")
        st.write(f"Description: {event.get('best_event_description', 'n/a')}")
        st.write(f"Selection reasons: {', '.join(event.get('selection_reasons', [])) or 'n/a'}")
        st.write(f"Why selected: {event.get('why_selected', 'n/a')}")
        st.write(f"Review note: {event.get('review_note', 'n/a')}")
    with right:
        st.write(f"Ranked clip score: `{event.get('ranked_clip_score', 'n/a')}`")
        st.write(f"Motion score: `{event.get('motion_score', 'n/a')}`")
        st.write(f"YOLO person max: `{event.get('yolo_person_max', 'n/a')}`")
        st.write(f"YOLO classes: {', '.join(event.get('yolo_top_classes', [])) or 'n/a'}")

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
        "vlm_outputs": load_json(run_dir / "16_topk_vlm_outputs.json", default={}) or {},
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

    st.subheader("Final Summary")
    st.write(descriptive_summary)
    st.info("This summary is generated from selected Top-K clips. For richer summary, use the descriptive summary fields from Step 17.")

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
    compiled_path_value = (
        compiled_info.get("playback_recommended_file")
        or compiled_info.get("compiled_video_path")
        or export_info.get("playback_recommended_file")
        or export_info.get("output_path")
        or str(run_dir / "18_exported_clips" / "18_compiled_review_video.mp4")
    )
    compiled_path = resolve_media_path(run_dir, compiled_path_value)
    st.subheader("Compiled Review Video")
    if media_exists(compiled_path):
        st.video(str(compiled_path))
    else:
        st.warning("Compiled review video not found. Re-run Step 18. For normal-only videos, enable TENDER_DEMO_COMPILE_NORMAL_IF_NO_EVENTS=true.")
        st.caption("Run Step 18 again to generate compiled review video.")

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

    st.subheader("Priority Suspicious Events")
    for event in summary.get("priority_suspicious_events", []):
        with st.expander(
            f"{event.get('clip_id', 'unknown')} | {event.get('time_range', 'unknown')} | "
            f"{event.get('risk_level', 'unknown')} | {event.get('confidence', 'unknown')}",
            expanded=True,
        ):
            render_event_card(event, qwen_by_clip_id, run_dir, show_raw_qwen=show_raw_qwen)

    st.subheader("Possible Review Clips")
    for event in summary.get("possible_review_clips", []):
        with st.expander(
            f"{event.get('clip_id', 'unknown')} | {event.get('time_range', 'unknown')} | "
            f"{event.get('risk_level', 'unknown')} | {event.get('confidence', 'unknown')}",
            expanded=False,
        ):
            render_event_card(event, qwen_by_clip_id, run_dir, show_raw_qwen=show_raw_qwen)

    st.subheader("Normal Activity Clips")
    for event in summary.get("normal_activity_clips", []):
        with st.expander(f"{event.get('clip_id', 'unknown')} | {event.get('time_range', 'unknown')}", expanded=False):
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
    query = st.text_input(
        "Free text search",
        placeholder="Search: red cap, bag, display case, robbery, fight, accident, person bending...",
    )
    st.caption("Examples: `red cap`, `display case`, `bag`, `person bending`, `suspicious reaching`, `robbery`, `collision`, `crowding`, `white shirt`")

    filter_cols = st.columns(3)
    with filter_cols[0]:
        category = st.selectbox("Category", ["All", "Priority suspicious event", "Possible review clip", "Normal activity", "Uncertain activity"])
        risk_level = st.selectbox("Risk level", ["All", "low", "medium", "high", "unknown"])
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
        )
    with filter_cols[1]:
        object_type = st.text_input("Object type", placeholder="bag, backpack, bottle, laptop, phone, display case, jewelry, vehicle, car, bike")
        person_appearance = st.text_input("Person appearance / clothing", placeholder="red cap, black shirt, white shirt, dark clothing")
        vehicle = st.text_input("Vehicle", placeholder="car, bike, truck, white car")
    with filter_cols[2]:
        time_start = st.number_input("Start seconds", min_value=0.0, value=0.0, step=1.0)
        time_end = st.number_input("End seconds", min_value=0.0, value=0.0, step=1.0)
        st.caption("Vehicle speed search is not available yet because speed needs tracking across frames. This can be added in a future tracking step.")

    debug_runs_dir = project_root() / "tests" / "tender_demo_case" / "debug_runs"
    current_run_str = str(run_dir) if run_dir and run_dir.exists() else ""
    records = load_all_search_records(str(debug_runs_dir), search_scope, current_run_str)
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
    selected_video = st.selectbox("Video selection", video_options)

    search_clicked = st.button("Search")
    filters = {
        "query": query,
        "category": category,
        "risk_level": risk_level,
        "event_type": event_type,
        "object_type": object_type,
        "person_appearance": person_appearance,
        "vehicle": vehicle,
        "selected_video": selected_video,
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
        processing_preset = st.selectbox("Processing preset", ["Fast demo", "Balanced", "Higher accuracy"], index=0)
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
        quick_result_mode = st.checkbox("Quick result mode", value=True)
        st.caption("Quick result mode scans the video, selects the most important few clips, and sends only those clips to Qwen. This is faster but less exhaustive.")
        sample_every_seconds = st.number_input("Sample every seconds", min_value=0.1, value=float(preset_values["sample_every_seconds"]), step=0.1)
        top_k = st.number_input("Top-K clips", min_value=1, value=int(preset_values["top_k"]), step=1)
        qwen_model_id = st.text_input("Qwen model id", value="qwen2.5vl:7b")
        qwen_max_new_tokens = st.number_input("Qwen max new tokens", min_value=1, value=int(preset_values["qwen_max_new_tokens"]), step=1)
        qwen_batch_size = st.number_input("Qwen batch size", min_value=1, value=int(preset_values["qwen_batch_size"]), step=1)
        st.caption("Batch size 4 may be slower or unstable depending on GPU memory. Start with 1 or 2.")
        run_yolo = st.checkbox("Run YOLO", value=True)
        yolo_model = st.text_input("YOLO model", value="yolov8n.pt")
        yolo_conf = st.number_input("YOLO confidence", min_value=0.01, max_value=1.0, value=float(preset_values["yolo_conf"]), step=0.01)
        yolo_imgsz = st.number_input("YOLO image size", min_value=32, value=int(preset_values["yolo_imgsz"]), step=32)
        parallel_branches = st.checkbox("Enable fast parallel branches", value=bool(preset_values["parallel_branches"]))
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
        "processing_preset": processing_preset,
        "quick_result_mode": quick_result_mode,
        "sample_every_seconds": sample_every_seconds,
        "top_k": int(top_k),
        "qwen_model_id": qwen_model_id,
        "qwen_max_new_tokens": int(qwen_max_new_tokens),
        "qwen_batch_size": int(qwen_batch_size),
        "run_yolo": run_yolo,
        "yolo_model": yolo_model,
        "yolo_conf": yolo_conf,
        "yolo_imgsz": int(yolo_imgsz),
        "parallel_branches": parallel_branches,
        "create_compiled_review_video": create_compiled_review_video,
        "compiled_video_fps": int(compiled_video_fps),
        "seconds_per_frame": seconds_per_frame,
        "seconds_per_title_card": seconds_per_title_card,
        "max_video_seconds": max_video_seconds,
    }
    if quick_result_mode:
        settings["sample_every_seconds"] = QUICK_RESULT_SETTINGS["sample_every_seconds"]
        settings["top_k"] = QUICK_RESULT_SETTINGS["top_k"]
        settings["qwen_max_new_tokens"] = QUICK_RESULT_SETTINGS["qwen_max_new_tokens"]
        settings["yolo_imgsz"] = QUICK_RESULT_SETTINGS["yolo_imgsz"]
        settings["yolo_conf"] = QUICK_RESULT_SETTINGS["yolo_conf"]
        settings["pipeline_engine"] = "Fast parallel Top-K pipeline"
        settings["pipeline_engine_id"] = PIPELINE_ENGINE_MAP["Fast parallel Top-K pipeline"]["engine_id"]
        settings["parallel_branches"] = True
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
