from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

import td_case2_results_ui as results_ui
import td_case2_traffic_search_ui as traffic_ui
from config import (
    CASE_ENV_PATH,
    DEFAULT_ADAPTIVE_ENABLED,
    DEFAULT_ADAPTIVE_HEARTBEAT_SECONDS,
    DEFAULT_ADAPTIVE_HISTOGRAM_CHANGE_THRESHOLD,
    DEFAULT_ADAPTIVE_MAX_BRIGHTNESS,
    DEFAULT_ADAPTIVE_MIN_BLOB_AREA_RATIO,
    DEFAULT_ADAPTIVE_MIN_BLUR_SCORE,
    DEFAULT_ADAPTIVE_MIN_BRIGHTNESS,
    DEFAULT_ADAPTIVE_MIN_PIXEL_VARIANCE,
    DEFAULT_ADAPTIVE_MIN_SELECTED_GAP_SECONDS,
    DEFAULT_ADAPTIVE_MOTION_PIXELS_RATIO_THRESHOLD,
    DEFAULT_ADAPTIVE_MOTION_SCORE_THRESHOLD,
    DEFAULT_QWEN_API_MODEL,
    DEFAULT_QWEN_API_PROVIDER,
    DEFAULT_SAMPLE_EVERY_SECONDS,
    DEFAULT_STEP05_AVOID_NEAR_DUPLICATES,
    DEFAULT_STEP05_FALLBACK_TOP_K_PER_TRACK,
    DEFAULT_STEP05_MIN_TIME_GAP_BETWEEN_SELECTED_SECONDS,
    DEFAULT_STEP05_PRIMARY_TOP_K_PER_TRACK,
    DEFAULT_STEP06_MIN_PLATE_CONFIDENCE,
    DEFAULT_STEP06_NUM_BEAMS,
    DEFAULT_STEP06_REUSE_EXISTING_RAW_RESULTS,
    DEFAULT_STEP07_INCLUDE_FALLBACK,
    DEFAULT_STEP07_INCLUDE_POSSIBLE_OCR,
    DEFAULT_STEP08_TIME_TOLERANCE_SECONDS,
    DEFAULT_STEP08_TOP_K,
    DEFAULT_STEP09_TOP_K,
    DEFAULT_STEP11_5_MAX_CANDIDATES_TO_CHECK,
    DEFAULT_STEP11_5_MAX_NEW_TOKENS,
    DEFAULT_STEP11_5_MODEL_PATH,
    DEFAULT_STEP11_CONTEXT_AFTER_SECONDS,
    DEFAULT_STEP11_CONTEXT_BEFORE_SECONDS,
    DEFAULT_STEP11_MAX_EVENT_SECONDS,
    DEFAULT_STEP11_MERGE_GAP_SECONDS,
    DEFAULT_STEP11_MIN_CANDIDATE_SCORE,
    DEFAULT_STEP11_WINDOW_SECONDS,
    DEFAULT_STEP11_WINDOW_STRIDE_SECONDS,
    DEFAULT_STEP12_MAX_PER_EVENT_TYPE,
    DEFAULT_STEP12_MAX_PER_TIME_CLUSTER,
    DEFAULT_STEP12_MIN_RANKING_SCORE,
    DEFAULT_STEP12_MIN_TEMPORAL_GAP_SECONDS,
    DEFAULT_STEP12_TOP_K,
    DEFAULT_STEP13_CONTEXT_AFTER_SECONDS,
    DEFAULT_STEP13_CONTEXT_BEFORE_SECONDS,
    DEFAULT_STEP13_MAX_GROUP_DURATION_SECONDS,
    DEFAULT_STEP13_MAX_INPUTS,
    DEFAULT_STEP13_MERGE_GAP_SECONDS,
    DEFAULT_STEP13_MERGE_NEARBY_SELECTED,
    DEFAULT_STEP13_STRIP_MODE,
    DEFAULT_STEP13_STRIP_PANEL_HEIGHT,
    DEFAULT_STEP13_STRIP_WIDTH,
    DEFAULT_STEP14_MAX_INPUTS,
    DEFAULT_STEP14_MAX_NEW_TOKENS,
    DEFAULT_STEP14_MODEL_PATH,
    DEFAULT_STEP16_CLIP_FPS,
    DEFAULT_STEP16_HEADER_SECONDS,
    DEFAULT_STEP16_INCLUDE_NORMAL_CONTEXT_SCENE_EVENTS,
    DEFAULT_STEP16_INCLUDE_PERSON_EVENTS,
    DEFAULT_STEP16_MAX_OBJECT_EVENTS,
    DEFAULT_STEP16_OBJECT_CONTEXT_AFTER_SECONDS,
    DEFAULT_STEP16_OBJECT_CONTEXT_BEFORE_SECONDS,
    DEFAULT_STEP16_SUMMARY_SECONDS,
    DEFAULT_TRACKING_MAX_TIME_GAP_SECONDS,
    DEFAULT_TRACKING_MIN_CONFIDENCE,
    DEFAULT_TRACKING_MIN_IOU,
    DEFAULT_TRACKING_MIN_TRACK_LENGTH,
    DEFAULT_VLM_BACKEND,
    DEFAULT_YOLO_CONF_THRESHOLD,
    DEFAULT_YOLO_IOU_THRESHOLD,
    ENV_ADAPTIVE_ENABLED,
    ENV_ADAPTIVE_HEARTBEAT_SECONDS,
    ENV_ADAPTIVE_HISTOGRAM_CHANGE_THRESHOLD,
    ENV_ADAPTIVE_MAX_BRIGHTNESS,
    ENV_ADAPTIVE_MIN_BLOB_AREA_RATIO,
    ENV_ADAPTIVE_MIN_BLUR_SCORE,
    ENV_ADAPTIVE_MIN_BRIGHTNESS,
    ENV_ADAPTIVE_MIN_PIXEL_VARIANCE,
    ENV_ADAPTIVE_MIN_SELECTED_GAP_SECONDS,
    ENV_ADAPTIVE_MOTION_PIXELS_RATIO_THRESHOLD,
    ENV_ADAPTIVE_MOTION_SCORE_THRESHOLD,
    ENV_DEVICE,
    ENV_DEVICE_INDEX,
    ENV_FLORENCE_MODEL_PATH,
    ENV_INPUT_VIDEO,
    ENV_OUTPUT_ROOT,
    ENV_PLATE_DETECTOR_MODEL_PATH,
    ENV_QWEN_API_KEY,
    ENV_QWEN_API_MODEL,
    ENV_QWEN_API_PROVIDER,
    ENV_RUN_DIR,
    ENV_SAMPLE_EVERY_SECONDS,
    ENV_STEP05_AVOID_NEAR_DUPLICATES,
    ENV_STEP05_FALLBACK_TOP_K_PER_TRACK,
    ENV_STEP05_MIN_TIME_GAP_BETWEEN_SELECTED_SECONDS,
    ENV_STEP05_PRIMARY_TOP_K_PER_TRACK,
    ENV_STEP06_MIN_PLATE_CONFIDENCE,
    ENV_STEP06_NUM_BEAMS,
    ENV_STEP06_REUSE_EXISTING_RAW_RESULTS,
    ENV_STEP07_INCLUDE_FALLBACK,
    ENV_STEP07_INCLUDE_POSSIBLE_OCR,
    ENV_STEP08_TIME_TOLERANCE_SECONDS,
    ENV_STEP08_TOP_K,
    ENV_STEP09_TOP_K,
    ENV_STEP11_5_MAX_CANDIDATES_TO_CHECK,
    ENV_STEP11_5_MAX_NEW_TOKENS,
    ENV_STEP11_5_MODEL_PATH,
    ENV_STEP11_CONTEXT_AFTER_SECONDS,
    ENV_STEP11_CONTEXT_BEFORE_SECONDS,
    ENV_STEP11_MAX_EVENT_SECONDS,
    ENV_STEP11_MERGE_GAP_SECONDS,
    ENV_STEP11_MIN_CANDIDATE_SCORE,
    ENV_STEP11_WINDOW_SECONDS,
    ENV_STEP11_WINDOW_STRIDE_SECONDS,
    ENV_STEP12_MAX_PER_EVENT_TYPE,
    ENV_STEP12_MAX_PER_TIME_CLUSTER,
    ENV_STEP12_MIN_RANKING_SCORE,
    ENV_STEP12_MIN_TEMPORAL_GAP_SECONDS,
    ENV_STEP12_TOP_K,
    ENV_STEP13_CONTEXT_AFTER_SECONDS,
    ENV_STEP13_CONTEXT_BEFORE_SECONDS,
    ENV_STEP13_MAX_GROUP_DURATION_SECONDS,
    ENV_STEP13_MAX_INPUTS,
    ENV_STEP13_MERGE_GAP_SECONDS,
    ENV_STEP13_MERGE_NEARBY_SELECTED,
    ENV_STEP13_STRIP_MODE,
    ENV_STEP13_STRIP_PANEL_HEIGHT,
    ENV_STEP13_STRIP_WIDTH,
    ENV_STEP14_MAX_INPUTS,
    ENV_STEP14_MAX_NEW_TOKENS,
    ENV_STEP14_MODEL_PATH,
    ENV_STEP16_CLIP_FPS,
    ENV_STEP16_HEADER_SECONDS,
    ENV_STEP16_INCLUDE_NORMAL_CONTEXT_SCENE_EVENTS,
    ENV_STEP16_INCLUDE_PERSON_EVENTS,
    ENV_STEP16_MAX_OBJECT_EVENTS,
    ENV_STEP16_OBJECT_CONTEXT_AFTER_SECONDS,
    ENV_STEP16_OBJECT_CONTEXT_BEFORE_SECONDS,
    ENV_STEP16_SUMMARY_SECONDS,
    ENV_TRACKING_MAX_TIME_GAP_SECONDS,
    ENV_TRACKING_MIN_CONFIDENCE,
    ENV_TRACKING_MIN_IOU,
    ENV_TRACKING_MIN_TRACK_LENGTH,
    ENV_VLM_BACKEND,
    ENV_YOLO_CONF_THRESHOLD,
    ENV_YOLO_IOU_THRESHOLD,
    ENV_YOLO_MODEL_PATH,
    default_output_root,
    read_local_env_path,
)


UPLOADS_DIR = Path(__file__).resolve().parent / "uploads"
CASE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CASE_ROOT.parents[1]
SESSION_LOG_KEY = "td_case2_workbench_logs"
SESSION_UPLOAD_KEY = "td_case2_workbench_uploaded_video"


@dataclass(frozen=True)
class ParameterDef:
    env_name: str
    label: str
    kind: str
    default: Any
    help_text: str = ""
    options: tuple[str, ...] | None = None
    min_value: float | int | None = None
    max_value: float | int | None = None
    step: float | int | None = None
    password: bool = False


PARAMETER_GROUPS: dict[str, list[ParameterDef]] = {
    "General": [
        ParameterDef(ENV_SAMPLE_EVERY_SECONDS, "Sample every seconds", "float", DEFAULT_SAMPLE_EVERY_SECONDS, min_value=0.1, step=0.1),
        ParameterDef(ENV_OUTPUT_ROOT, "Output root", "text", str(default_output_root())),
        ParameterDef(ENV_DEVICE, "Global device", "select", "auto", options=("auto", "cuda", "cpu")),
        ParameterDef(ENV_DEVICE_INDEX, "CUDA device index", "int", 0, min_value=0, step=1),
    ],
    "Adaptive Sampling": [
        ParameterDef(ENV_ADAPTIVE_ENABLED, "Enable adaptive sampling", "bool", DEFAULT_ADAPTIVE_ENABLED),
        ParameterDef(ENV_ADAPTIVE_MOTION_SCORE_THRESHOLD, "Motion score threshold", "float", DEFAULT_ADAPTIVE_MOTION_SCORE_THRESHOLD, min_value=0.0, step=0.001),
        ParameterDef(ENV_ADAPTIVE_MOTION_PIXELS_RATIO_THRESHOLD, "Motion pixels ratio threshold", "float", DEFAULT_ADAPTIVE_MOTION_PIXELS_RATIO_THRESHOLD, min_value=0.0, step=0.001),
        ParameterDef(ENV_ADAPTIVE_HISTOGRAM_CHANGE_THRESHOLD, "Histogram change threshold", "float", DEFAULT_ADAPTIVE_HISTOGRAM_CHANGE_THRESHOLD, min_value=0.0, step=0.01),
        ParameterDef(ENV_ADAPTIVE_MIN_BLOB_AREA_RATIO, "Min blob area ratio", "float", DEFAULT_ADAPTIVE_MIN_BLOB_AREA_RATIO, min_value=0.0, step=0.001),
        ParameterDef(ENV_ADAPTIVE_HEARTBEAT_SECONDS, "Heartbeat seconds", "float", DEFAULT_ADAPTIVE_HEARTBEAT_SECONDS, min_value=0.1, step=0.1),
        ParameterDef(ENV_ADAPTIVE_MIN_SELECTED_GAP_SECONDS, "Min selected gap seconds", "float", DEFAULT_ADAPTIVE_MIN_SELECTED_GAP_SECONDS, min_value=0.0, step=0.1),
        ParameterDef(ENV_ADAPTIVE_MIN_BRIGHTNESS, "Min brightness", "float", DEFAULT_ADAPTIVE_MIN_BRIGHTNESS, min_value=0.0, step=1.0),
        ParameterDef(ENV_ADAPTIVE_MAX_BRIGHTNESS, "Max brightness", "float", DEFAULT_ADAPTIVE_MAX_BRIGHTNESS, min_value=0.0, step=1.0),
        ParameterDef(ENV_ADAPTIVE_MIN_PIXEL_VARIANCE, "Min pixel variance", "float", DEFAULT_ADAPTIVE_MIN_PIXEL_VARIANCE, min_value=0.0, step=1.0),
        ParameterDef(ENV_ADAPTIVE_MIN_BLUR_SCORE, "Min blur score", "float", DEFAULT_ADAPTIVE_MIN_BLUR_SCORE, min_value=0.0, step=1.0),
    ],
    "YOLO / Tracking": [
        ParameterDef(ENV_YOLO_MODEL_PATH, "YOLO model path", "text", "../../yolo11m.pt"),
        ParameterDef(ENV_YOLO_CONF_THRESHOLD, "YOLO confidence threshold", "float", DEFAULT_YOLO_CONF_THRESHOLD, min_value=0.0, max_value=1.0, step=0.01),
        ParameterDef(ENV_YOLO_IOU_THRESHOLD, "YOLO IoU threshold", "float", DEFAULT_YOLO_IOU_THRESHOLD, min_value=0.0, max_value=1.0, step=0.01),
        ParameterDef(ENV_TRACKING_MIN_CONFIDENCE, "Tracking min confidence", "float", DEFAULT_TRACKING_MIN_CONFIDENCE, min_value=0.0, max_value=1.0, step=0.01),
        ParameterDef(ENV_TRACKING_MIN_IOU, "Tracking min IoU", "float", DEFAULT_TRACKING_MIN_IOU, min_value=0.0, max_value=1.0, step=0.01),
        ParameterDef(ENV_TRACKING_MAX_TIME_GAP_SECONDS, "Tracking max time gap seconds", "float", DEFAULT_TRACKING_MAX_TIME_GAP_SECONDS, min_value=0.1, step=0.1),
        ParameterDef(ENV_TRACKING_MIN_TRACK_LENGTH, "Tracking min track length", "int", DEFAULT_TRACKING_MIN_TRACK_LENGTH, min_value=1, step=1),
    ],
    "Florence / OCR": [
        ParameterDef(ENV_FLORENCE_MODEL_PATH, "Florence model path", "text", str(read_local_env_path(ENV_FLORENCE_MODEL_PATH) or "")),
        ParameterDef(ENV_PLATE_DETECTOR_MODEL_PATH, "Plate detector model path", "text", "../../OCR_MUKUL/license_plate_weights.pt"),
        ParameterDef(ENV_STEP05_PRIMARY_TOP_K_PER_TRACK, "Step 05 primary top-K per track", "int", DEFAULT_STEP05_PRIMARY_TOP_K_PER_TRACK, min_value=1, step=1),
        ParameterDef(ENV_STEP05_FALLBACK_TOP_K_PER_TRACK, "Step 05 fallback top-K per track", "int", DEFAULT_STEP05_FALLBACK_TOP_K_PER_TRACK, min_value=0, step=1),
        ParameterDef(ENV_STEP05_AVOID_NEAR_DUPLICATES, "Avoid near duplicate crops", "bool", DEFAULT_STEP05_AVOID_NEAR_DUPLICATES),
        ParameterDef(ENV_STEP05_MIN_TIME_GAP_BETWEEN_SELECTED_SECONDS, "Min gap between selected crops", "float", DEFAULT_STEP05_MIN_TIME_GAP_BETWEEN_SELECTED_SECONDS, min_value=0.0, step=0.1),
        ParameterDef(ENV_STEP06_MIN_PLATE_CONFIDENCE, "Step 06 min plate confidence", "float", DEFAULT_STEP06_MIN_PLATE_CONFIDENCE, min_value=0.0, max_value=1.0, step=0.01),
        ParameterDef(ENV_STEP06_NUM_BEAMS, "Step 06 Florence beams", "int", DEFAULT_STEP06_NUM_BEAMS, min_value=1, step=1),
        ParameterDef(ENV_STEP06_REUSE_EXISTING_RAW_RESULTS, "Reuse Step 06 existing raw results", "bool", DEFAULT_STEP06_REUSE_EXISTING_RAW_RESULTS),
    ],
    "Search": [
        ParameterDef(ENV_STEP07_INCLUDE_FALLBACK, "Include fallback records", "bool", DEFAULT_STEP07_INCLUDE_FALLBACK),
        ParameterDef(ENV_STEP07_INCLUDE_POSSIBLE_OCR, "Include possible OCR", "bool", DEFAULT_STEP07_INCLUDE_POSSIBLE_OCR),
        ParameterDef(ENV_STEP08_TOP_K, "Step 08 top-K", "int", DEFAULT_STEP08_TOP_K, min_value=1, step=1),
        ParameterDef(ENV_STEP08_TIME_TOLERANCE_SECONDS, "Step 08 time tolerance seconds", "float", DEFAULT_STEP08_TIME_TOLERANCE_SECONDS, min_value=0.0, step=0.5),
        ParameterDef(ENV_STEP09_TOP_K, "Step 09 top-K", "int", DEFAULT_STEP09_TOP_K, min_value=1, step=1),
    ],
    "Events / VLM": [
        ParameterDef(ENV_VLM_BACKEND, "VLM backend", "select", DEFAULT_VLM_BACKEND, options=("local_qwen", "api_qwen", "disabled")),
        ParameterDef(ENV_STEP11_WINDOW_SECONDS, "Step 11 window seconds", "float", DEFAULT_STEP11_WINDOW_SECONDS, min_value=0.1, step=0.5),
        ParameterDef(ENV_STEP11_WINDOW_STRIDE_SECONDS, "Step 11 window stride seconds", "float", DEFAULT_STEP11_WINDOW_STRIDE_SECONDS, min_value=0.1, step=0.5),
        ParameterDef(ENV_STEP11_MERGE_GAP_SECONDS, "Step 11 merge gap seconds", "float", DEFAULT_STEP11_MERGE_GAP_SECONDS, min_value=0.0, step=0.5),
        ParameterDef(ENV_STEP11_MAX_EVENT_SECONDS, "Step 11 max event seconds", "float", DEFAULT_STEP11_MAX_EVENT_SECONDS, min_value=0.5, step=0.5),
        ParameterDef(ENV_STEP11_CONTEXT_BEFORE_SECONDS, "Step 11 context before seconds", "float", DEFAULT_STEP11_CONTEXT_BEFORE_SECONDS, min_value=0.0, step=0.5),
        ParameterDef(ENV_STEP11_CONTEXT_AFTER_SECONDS, "Step 11 context after seconds", "float", DEFAULT_STEP11_CONTEXT_AFTER_SECONDS, min_value=0.0, step=0.5),
        ParameterDef(ENV_STEP11_MIN_CANDIDATE_SCORE, "Step 11 min candidate score", "float", DEFAULT_STEP11_MIN_CANDIDATE_SCORE, min_value=0.0, max_value=1.0, step=0.01),
        ParameterDef(ENV_STEP11_5_MODEL_PATH, "Step 11.5 model path", "text", str(DEFAULT_STEP11_5_MODEL_PATH)),
        ParameterDef(ENV_STEP11_5_MAX_CANDIDATES_TO_CHECK, "Step 11.5 max candidates to check", "int", DEFAULT_STEP11_5_MAX_CANDIDATES_TO_CHECK, min_value=1, step=1),
        ParameterDef(ENV_STEP11_5_MAX_NEW_TOKENS, "Step 11.5 max new tokens", "int", DEFAULT_STEP11_5_MAX_NEW_TOKENS, min_value=10, step=10),
        ParameterDef(ENV_STEP12_TOP_K, "Step 12 top-K", "int", DEFAULT_STEP12_TOP_K, min_value=1, step=1),
        ParameterDef(ENV_STEP12_MIN_RANKING_SCORE, "Step 12 min ranking score", "float", DEFAULT_STEP12_MIN_RANKING_SCORE, min_value=0.0, max_value=1.0, step=0.01),
        ParameterDef(ENV_STEP12_MIN_TEMPORAL_GAP_SECONDS, "Step 12 min temporal gap seconds", "float", DEFAULT_STEP12_MIN_TEMPORAL_GAP_SECONDS, min_value=0.0, step=0.5),
        ParameterDef(ENV_STEP12_MAX_PER_EVENT_TYPE, "Step 12 max per event type", "int", DEFAULT_STEP12_MAX_PER_EVENT_TYPE, min_value=1, step=1),
        ParameterDef(ENV_STEP12_MAX_PER_TIME_CLUSTER, "Step 12 max per time cluster", "int", DEFAULT_STEP12_MAX_PER_TIME_CLUSTER, min_value=1, step=1),
        ParameterDef(ENV_STEP13_MERGE_NEARBY_SELECTED, "Step 13 merge nearby selected", "bool", DEFAULT_STEP13_MERGE_NEARBY_SELECTED),
        ParameterDef(ENV_STEP13_MERGE_GAP_SECONDS, "Step 13 merge gap seconds", "float", DEFAULT_STEP13_MERGE_GAP_SECONDS, min_value=0.0, step=0.5),
        ParameterDef(ENV_STEP13_MAX_GROUP_DURATION_SECONDS, "Step 13 max group duration seconds", "float", DEFAULT_STEP13_MAX_GROUP_DURATION_SECONDS, min_value=0.5, step=0.5),
        ParameterDef(ENV_STEP13_CONTEXT_BEFORE_SECONDS, "Step 13 context before seconds", "float", DEFAULT_STEP13_CONTEXT_BEFORE_SECONDS, min_value=0.0, step=0.5),
        ParameterDef(ENV_STEP13_CONTEXT_AFTER_SECONDS, "Step 13 context after seconds", "float", DEFAULT_STEP13_CONTEXT_AFTER_SECONDS, min_value=0.0, step=0.5),
        ParameterDef(ENV_STEP13_STRIP_MODE, "Step 13 strip mode", "select", DEFAULT_STEP13_STRIP_MODE, options=("three_panel", "five_panel")),
        ParameterDef(ENV_STEP13_STRIP_WIDTH, "Step 13 strip width", "int", DEFAULT_STEP13_STRIP_WIDTH, min_value=320, step=10),
        ParameterDef(ENV_STEP13_STRIP_PANEL_HEIGHT, "Step 13 strip panel height", "int", DEFAULT_STEP13_STRIP_PANEL_HEIGHT, min_value=100, step=10),
        ParameterDef(ENV_STEP13_MAX_INPUTS, "Step 13 max inputs", "int", DEFAULT_STEP13_MAX_INPUTS, min_value=1, step=1),
        ParameterDef(ENV_STEP14_MODEL_PATH, "Step 14 model path", "text", str(DEFAULT_STEP14_MODEL_PATH)),
        ParameterDef(ENV_STEP14_MAX_INPUTS, "Step 14 max inputs", "int", DEFAULT_STEP14_MAX_INPUTS, min_value=1, step=1),
        ParameterDef(ENV_STEP14_MAX_NEW_TOKENS, "Step 14 max new tokens", "int", DEFAULT_STEP14_MAX_NEW_TOKENS, min_value=10, step=10),
        ParameterDef(ENV_STEP16_CLIP_FPS, "Step 16 evidence clip fps", "int", DEFAULT_STEP16_CLIP_FPS, min_value=1, step=1),
        ParameterDef(ENV_STEP16_HEADER_SECONDS, "Step 16 header seconds", "float", DEFAULT_STEP16_HEADER_SECONDS, min_value=0.1, step=0.1),
        ParameterDef(ENV_STEP16_SUMMARY_SECONDS, "Step 16 summary seconds", "float", DEFAULT_STEP16_SUMMARY_SECONDS, min_value=0.1, step=0.1),
        ParameterDef(
            ENV_STEP16_OBJECT_CONTEXT_BEFORE_SECONDS,
            "Step 16 object context before seconds",
            "float",
            DEFAULT_STEP16_OBJECT_CONTEXT_BEFORE_SECONDS,
            min_value=0.0,
            step=0.1,
        ),
        ParameterDef(
            ENV_STEP16_OBJECT_CONTEXT_AFTER_SECONDS,
            "Step 16 object context after seconds",
            "float",
            DEFAULT_STEP16_OBJECT_CONTEXT_AFTER_SECONDS,
            min_value=0.0,
            step=0.1,
        ),
        ParameterDef(ENV_STEP16_MAX_OBJECT_EVENTS, "Step 16 max object events", "int", DEFAULT_STEP16_MAX_OBJECT_EVENTS, min_value=1, step=1),
        ParameterDef(ENV_STEP16_INCLUDE_PERSON_EVENTS, "Step 16 include person events", "bool", DEFAULT_STEP16_INCLUDE_PERSON_EVENTS),
        ParameterDef(
            ENV_STEP16_INCLUDE_NORMAL_CONTEXT_SCENE_EVENTS,
            "Step 16 include normal context scene events",
            "bool",
            DEFAULT_STEP16_INCLUDE_NORMAL_CONTEXT_SCENE_EVENTS,
        ),
        ParameterDef(ENV_QWEN_API_PROVIDER, "Qwen API provider", "text", DEFAULT_QWEN_API_PROVIDER),
        ParameterDef(ENV_QWEN_API_MODEL, "Qwen API model", "text", DEFAULT_QWEN_API_MODEL),
        ParameterDef(ENV_QWEN_API_KEY, "Qwen API key", "text", "", password=True),
    ],
}


def parse_created_run_dir(log_text: str) -> str | None:
    patterns = [
        r"TD_CASE2_RUN_DIR=(.+)",
        r"Created run directory:\s*(.+)",
        r"Run directory:\s*(.+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, log_text)
        if matches:
            value = str(matches[-1]).strip().strip('"').strip("'")
            if value:
                return value
    return None


def parse_extra_env_json(raw_text: str) -> dict[str, str]:
    normalized = str(raw_text or "").strip()
    if not normalized:
        return {}
    payload = json.loads(normalized)
    if not isinstance(payload, dict):
        raise ValueError("Extra environment overrides must be a JSON object.")
    output: dict[str, str] = {}
    for key, value in payload.items():
        if value is None:
            continue
        output[str(key)] = str(value)
    return output


def _resolve_python_executable() -> str:
    return sys.executable


def _save_uploaded_file(uploaded_file: Any) -> Path:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{timestamp}_{Path(uploaded_file.name).name}"
    output_path = UPLOADS_DIR / file_name
    output_path.write_bytes(uploaded_file.getbuffer())
    return output_path.resolve()


def _render_parameter_input(parameter: ParameterDef) -> Any:
    key = f"param_{parameter.env_name}"
    if parameter.kind == "bool":
        return st.checkbox(parameter.label, value=bool(parameter.default), help=parameter.help_text, key=key)
    if parameter.kind == "int":
        return st.number_input(
            parameter.label,
            value=int(parameter.default),
            min_value=int(parameter.min_value) if parameter.min_value is not None else None,
            max_value=int(parameter.max_value) if parameter.max_value is not None else None,
            step=int(parameter.step or 1),
            help=parameter.help_text,
            key=key,
        )
    if parameter.kind == "float":
        return st.number_input(
            parameter.label,
            value=float(parameter.default),
            min_value=float(parameter.min_value) if parameter.min_value is not None else None,
            max_value=float(parameter.max_value) if parameter.max_value is not None else None,
            step=float(parameter.step or 0.1),
            format="%.4f",
            help=parameter.help_text,
            key=key,
        )
    if parameter.kind == "select":
        options = list(parameter.options or [])
        default_index = options.index(parameter.default) if parameter.default in options else 0
        return st.selectbox(parameter.label, options, index=default_index, help=parameter.help_text, key=key)
    return st.text_input(
        parameter.label,
        value=str(parameter.default or ""),
        help=parameter.help_text,
        type="password" if parameter.password else "default",
        key=key,
    )


def _collect_parameter_values() -> dict[str, str]:
    env_values: dict[str, str] = {}
    for group_name, parameters in PARAMETER_GROUPS.items():
        with st.expander(group_name, expanded=group_name == "General"):
            for parameter in parameters:
                value = _render_parameter_input(parameter)
                if parameter.kind == "bool":
                    env_values[parameter.env_name] = "true" if bool(value) else "false"
                else:
                    env_values[parameter.env_name] = str(value).strip()
    return env_values


def _build_runner_env(
    *,
    ui_env_values: dict[str, str],
    input_video_path: str,
    selected_run_dir: str,
    extra_env_text: str,
) -> dict[str, str]:
    env = dict(os.environ)
    for key, value in ui_env_values.items():
        if value != "":
            env[key] = value
    if input_video_path.strip():
        env[ENV_INPUT_VIDEO] = input_video_path.strip()
    if selected_run_dir.strip():
        env[ENV_RUN_DIR] = selected_run_dir.strip()
    env.update(parse_extra_env_json(extra_env_text))
    return env


def _run_command(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _append_log(title: str, result: subprocess.CompletedProcess[str]) -> None:
    logs = list(st.session_state.get(SESSION_LOG_KEY, []))
    logs.append(
        {
            "title": title,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    )
    st.session_state[SESSION_LOG_KEY] = logs[-8:]


def _latest_debug_run_dir() -> Path | None:
    return traffic_ui._find_latest_valid_run_dir()


def _render_control_metrics(run_dir: Path | None) -> None:
    latest_run = run_dir if run_dir and run_dir.exists() else _latest_debug_run_dir()
    payloads = traffic_ui._load_payloads(latest_run) if latest_run and latest_run.exists() else {}
    video_info = payloads.get("01_video_info.json") or {}
    pipeline_status = payloads.get("pipeline_status.json") or {}
    search_report = payloads.get("07B_traffic_object_search_index_report.json") or {}
    vlm_report = payloads.get("14_vlm_event_review_report.json") or {}

    cols = st.columns(5)
    cols[0].metric("Active run", latest_run.name if latest_run else "None")
    cols[1].metric("Video", str(video_info.get("video_name", "-")))
    cols[2].metric("Search status", str(pipeline_status.get("object_search_status", "-")))
    cols[3].metric("VLM status", str(pipeline_status.get("vlm_status", vlm_report.get("status", "-"))))
    cols[4].metric("Searchable objects", int(search_report.get("total_object_records", 0) or 0))


def _render_stage_table(run_dir: Path | None) -> None:
    if run_dir is None or not run_dir.exists():
        st.info("No run directory is loaded yet.")
        return
    stage_gate = traffic_ui._read_json_if_exists(run_dir / "00_stage_gate_report.json")
    rows = results_ui._build_stage_rows(stage_gate if isinstance(stage_gate, dict) else None)
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("Stage gate report is missing.")


def _render_search_tab(run_dir: Path | None) -> None:
    if run_dir is None or not traffic_ui._is_search_ready_run_dir(run_dir):
        st.info("Load or run a search-ready `td_case2` run to use the integrated search page.")
        return

    payloads = traffic_ui._load_payloads(run_dir)
    video_info = payloads.get("01_video_info.json") or {}
    index_payload = payloads.get("07B_traffic_object_search_index.json") or {}
    records = list(index_payload.get("records", [])) if isinstance(index_payload, dict) else []
    duration_seconds = float(video_info.get("duration_seconds", 0.0) or 0.0)
    max_time = max([float(item.get("timestamp_seconds", 0.0) or 0.0) for item in records], default=0.0)

    st.caption(f"Searching run: `{run_dir}`")
    filter_cols = st.columns(4)
    query = filter_cols[0].text_input("Search query", value="")
    selected_class = filter_cols[1].selectbox(
        "Class",
        ["All"] + sorted({str(item.get("class_name", "") or "unknown") for item in records}),
    )
    selected_verified_color = filter_cols[2].selectbox(
        "Verified color",
        ["All"] + sorted({str(item.get("verified_vehicle_color", "") or "").strip() for item in records if item.get("verified_vehicle_color")}),
    )
    plate_filter = filter_cols[3].selectbox("Plate filter", ["All", "Verified Plate Only", "Possible Plate Allowed", "No Verified Plate"])

    filter_cols_2 = st.columns(4)
    selected_possible_color = filter_cols_2[0].selectbox(
        "Possible color",
        ["All"] + sorted({str(item.get("possible_vehicle_color", "") or "").strip() for item in records if item.get("possible_vehicle_color")}),
    )
    selected_object_type = filter_cols_2[1].selectbox(
        "Object type",
        ["All"] + sorted({str(item.get("object_type", "") or "unknown") for item in records}),
    )
    selected_quality = filter_cols_2[2].selectbox(
        "Quality",
        ["All"] + sorted({str(item.get("quality", "") or "unknown") for item in records}),
    )
    max_cards_to_show = filter_cols_2[3].selectbox("Max cards", [10, 25, 50, 100], index=1)

    include_possible_colors = st.checkbox("Include possible colors", value=True)
    include_possible_plates = st.checkbox("Include possible plates", value=False)
    time_range = st.slider("Time range (seconds)", 0.0, float(max_time), (0.0, float(max_time)))

    active_filters = traffic_ui._has_active_filters(
        query,
        selected_class,
        selected_verified_color,
        selected_possible_color,
        plate_filter,
        selected_object_type,
        selected_quality,
        time_range,
        max_time,
        include_possible_colors,
        include_possible_plates,
    )
    show_all_objects = st.button("Show all objects", key="workbench_show_all_objects")
    filtered_records = (
        traffic_ui._filter_records(
            records,
            run_dir,
            query,
            selected_class,
            selected_verified_color,
            selected_possible_color,
            plate_filter,
            selected_object_type,
            selected_quality,
            time_range,
            include_possible_colors,
            include_possible_plates,
        )
        if (active_filters or show_all_objects)
        else []
    )

    if not active_filters and not show_all_objects:
        st.info("Enter a query or use filters to view object/event search results.")
        return

    summary = traffic_ui._build_search_summary(filtered_records, query)
    overview_cols = st.columns(6)
    overview_cols[0].metric("Results", summary["total_matches"])
    overview_cols[1].metric("Unique tracks", summary["unique_tracks"])
    overview_cols[2].metric("First seen", summary["first_seen"])
    overview_cols[3].metric("Last seen", summary["last_seen"])
    overview_cols[4].metric("Verified plates found", summary["verified_plates_found"])
    overview_cols[5].metric("Video duration", traffic_ui._format_seconds(duration_seconds))

    clusters = traffic_ui._cluster_records(filtered_records, threshold_seconds=5.0)
    st.subheader("Timeline")
    traffic_ui._render_timeline(clusters, duration_seconds or max_time)

    st.subheader(f"Results ({min(len(filtered_records), max_cards_to_show)} shown)")
    for record in filtered_records[:max_cards_to_show]:
        traffic_ui._render_card(record, run_dir, "Medium")


def _render_vlm_tab(run_dir: Path | None) -> None:
    if run_dir is None or not run_dir.exists():
        st.info("Load a run directory to view VLM and event-review results.")
        return

    loaded: dict[str, dict[str, Any] | list[Any] | None] = {}
    for file_name in results_ui.FILE_NAMES:
        payload, _message = results_ui._read_json_if_exists(run_dir, file_name)
        loaded[file_name] = payload

    stage_gate = loaded["00_stage_gate_report.json"] if isinstance(loaded["00_stage_gate_report.json"], dict) else None
    video_info = loaded["01_video_info.json"] if isinstance(loaded["01_video_info.json"], dict) else None
    step11_5_report = loaded["11_5_vlm_filter_report.json"] if isinstance(loaded["11_5_vlm_filter_report.json"], dict) else None
    step13_report = loaded["13_vlm_event_input_report.json"] if isinstance(loaded["13_vlm_event_input_report.json"], dict) else None
    reviews_payload = loaded["14_vlm_event_reviews.json"] if isinstance(loaded["14_vlm_event_reviews.json"], dict) else None
    flat_reviews_payload = loaded["14_vlm_event_reviews_flat.json"] if isinstance(loaded["14_vlm_event_reviews_flat.json"], list) else None
    step14_report = loaded["14_vlm_event_review_report.json"] if isinstance(loaded["14_vlm_event_review_report.json"], dict) else None
    final_summary = loaded["14_final_video_summary.json"] if isinstance(loaded["14_final_video_summary.json"], dict) else None
    evidence_report = loaded["16_evidence_video_report.json"] if isinstance(loaded["16_evidence_video_report.json"], dict) else None
    evidence_index = loaded["evidence_video_index.json"] if isinstance(loaded["evidence_video_index.json"], dict) else None

    st.caption(f"Reviewing run: `{run_dir}`")
    summary_cols = st.columns(5)
    summary_cols[0].metric("Video", results_ui._safe_get(video_info, "video_name"))
    summary_cols[1].metric("Duration", results_ui._safe_get(video_info, "duration_text"))
    summary_cols[2].metric("Overall status", results_ui._safe_get(final_summary, "overall_status"))
    summary_cols[3].metric("Event count", results_ui._safe_get(final_summary, "event_count", 0))
    summary_cols[4].metric("Recommended action", results_ui._safe_get(final_summary, "recommended_action"))

    review_cols = st.columns(5)
    review_cols[0].metric("Step 11.5 inputs", results_ui._safe_get(step11_5_report, "input_candidate_count", 0))
    review_cols[1].metric("Step 13 inputs", results_ui._safe_get(step13_report, "vlm_inputs_created", 0))
    review_cols[2].metric("Reviewed", results_ui._safe_get(step14_report, "inputs_reviewed", 0))
    review_cols[3].metric("Event visible", results_ui._safe_get(step14_report, "event_visible_count", 0))
    review_cols[4].metric("Uncertain", results_ui._safe_get(step14_report, "uncertain_count", 0))

    stage_rows = results_ui._build_stage_rows(stage_gate if isinstance(stage_gate, dict) else None)
    if stage_rows:
        st.dataframe(stage_rows, use_container_width=True, hide_index=True)

    if final_summary:
        st.subheader("Final summary")
        st.info(results_ui._safe_get(final_summary, "summary"))
        st.json(final_summary)

    st.subheader("Evidence video")
    if evidence_report:
        evidence_cols = st.columns(6)
        evidence_cols[0].metric("Evidence clips", results_ui._safe_get(evidence_report, "event_count", 0))
        evidence_cols[1].metric("Evidence duration (s)", results_ui._safe_get(evidence_report, "evidence_duration_seconds", 0))
        evidence_cols[2].metric("Vehicles", results_ui._safe_get(evidence_report, "vehicles_detected", 0))
        evidence_cols[3].metric("Persons", results_ui._safe_get(evidence_report, "persons_detected", 0))
        evidence_cols[4].metric("Plates", results_ui._safe_get(evidence_report, "license_plates_detected", 0))
        evidence_cols[5].metric("Gen time (s)", results_ui._safe_get(evidence_report, "generation_time_seconds", 0))

        evidence_video_path = run_dir / str(evidence_report.get("video_file", "evidence_video.mp4"))
        if evidence_video_path.exists():
            st.video(str(evidence_video_path))
        else:
            st.info(f"Evidence video file is missing: {evidence_video_path.name}")

        if evidence_index and isinstance(evidence_index.get("clips"), list) and evidence_index["clips"]:
            st.dataframe(pd.DataFrame(evidence_index["clips"]), use_container_width=True, hide_index=True)
        with st.expander("Evidence report JSON"):
            st.json(evidence_report)
    else:
        st.info("Step 16 evidence outputs are not available for this run yet.")

    st.subheader("Reviewed moments")
    reviews = results_ui._review_list(reviews_payload, flat_reviews_payload)
    if not reviews:
        st.info("Step 14 VLM review files are not available for this run yet.")
        return
    for review in reviews:
        model_review = review.get("model_review", review)
        image_path, image_label = results_ui._moment_image_path(run_dir, review)
        with st.container(border=True):
            head = st.columns(6)
            head[0].markdown(f"**Timestamp**  \n{review.get('best_timestamp_text', '—')}")
            head[1].markdown(f"**Decision**  \n{model_review.get('review_decision', '—')}")
            head[2].markdown(f"**Event type**  \n{model_review.get('event_type', '—')}")
            head[3].markdown(f"**Risk**  \n{model_review.get('risk_level', '—')}")
            head[4].markdown(f"**Confidence**  \n{model_review.get('confidence', '—')}")
            head[5].markdown(f"**Human review**  \n{model_review.get('needs_human_review', '—')}")
            st.markdown(f"**Summary caption**: {model_review.get('summary_caption', '—')}")
            if image_path is not None:
                st.image(str(image_path), caption=f"{image_label}: {image_path.name}", use_container_width=True)
            with st.expander("Details"):
                st.markdown(f"**What is visible**: {model_review.get('what_is_visible', '—')}")
                st.markdown(f"**Why decision**: {model_review.get('why_decision', '—')}")
                st.json(review)


def _render_events_tab(run_dir: Path | None) -> None:
    if run_dir is None or not run_dir.exists():
        st.info("Load a run directory to inspect event artifacts.")
        return
    step11_path = run_dir / "11_full_scene_event_candidates.json"
    step12_path = run_dir / "12_selected_top_event_candidates.json"
    step13_path = run_dir / "13_vlm_event_inputs.json"
    candidates = traffic_ui._read_json_if_exists(step11_path) if step11_path.exists() else None
    selected = traffic_ui._read_json_if_exists(step12_path) if step12_path.exists() else None
    vlm_inputs = traffic_ui._read_json_if_exists(step13_path) if step13_path.exists() else None

    cols = st.columns(3)
    cols[0].metric("Step 11 candidates", len(list((candidates or {}).get("candidate_events", []))) if isinstance(candidates, dict) else 0)
    cols[1].metric("Step 12 selected", len(list((selected or {}).get("selected_candidates", []))) if isinstance(selected, dict) else 0)
    cols[2].metric("Step 13 VLM inputs", len(list((vlm_inputs or {}).get("vlm_inputs", []))) if isinstance(vlm_inputs, dict) else 0)

    if isinstance(selected, dict):
        st.subheader("Selected event candidates")
        rows = []
        for item in list(selected.get("selected_candidates", [])):
            if isinstance(item, dict):
                rows.append(
                    {
                        "candidate_id": item.get("candidate_id"),
                        "event_type": item.get("event_type"),
                        "time_range": f"{item.get('start_time_text', '-')}-{item.get('end_time_text', '-')}",
                        "score": item.get("ranking_score"),
                        "needs_vlm_review": item.get("needs_vlm_review"),
                    }
                )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    artifact_name = st.selectbox(
        "Artifact preview",
        [
            "11_full_scene_event_candidates.json",
            "12_selected_top_event_candidates.json",
            "13_vlm_event_inputs.json",
            "14_final_video_summary.json",
            "16_evidence_video_report.json",
            "evidence_video_index.json",
        ],
    )
    artifact_path = run_dir / artifact_name
    if artifact_path.exists():
        payload = traffic_ui._read_json_if_exists(artifact_path)
        st.json(payload)
    else:
        st.info(f"{artifact_name} is not available for this run yet.")


def _render_logs_tab() -> None:
    logs = list(st.session_state.get(SESSION_LOG_KEY, []))
    if not logs:
        st.info("No pipeline commands have been launched from this workbench yet.")
        return
    for item in reversed(logs):
        with st.expander(f"{item['timestamp']} | {item['title']} | rc={item['returncode']}", expanded=False):
            st.markdown("**stdout**")
            st.code(item.get("stdout") or "", language="text")
            if item.get("stderr"):
                st.markdown("**stderr**")
                st.code(item.get("stderr") or "", language="text")


SECTIONS = ["Pipeline Control", "Search & Clips", "VLM Summary", "Events", "Logs"]


def main(*, configure_page: bool = True, initial_section: str = "Pipeline Control") -> None:
    if configure_page:
        st.set_page_config(page_title="TD Case 2 Workbench", layout="wide")
    traffic_ui._inject_styles()
    st.title("TD Case 2 Workbench")
    st.caption("One integrated UI for upload, parameter control, search, event review, and VLM summary.")

    resolved_run_dir, run_dir_source = traffic_ui._resolve_initial_run_dir()
    initial_run_dir = str(resolved_run_dir) if resolved_run_dir is not None else ""

    with st.sidebar:
        st.header("Run Selection")
        run_dir_input = st.text_input("Run directory", value=initial_run_dir, key="workbench_run_dir_input")
        selected_run_dir = Path(run_dir_input).expanduser() if run_dir_input.strip() else None
        if selected_run_dir is not None and not selected_run_dir.is_absolute():
            selected_run_dir = selected_run_dir.resolve()
        button_cols = st.columns(2)
        if button_cols[0].button("Load latest run"):
            latest = _latest_debug_run_dir()
            if latest is not None:
                st.session_state[traffic_ui.SESSION_RUN_DIR_KEY] = str(latest)
                st.rerun()
        if button_cols[1].button("Use selected"):
            st.session_state[traffic_ui.SESSION_RUN_DIR_KEY] = run_dir_input.strip()
            st.rerun()
        st.caption(f"Run source: {run_dir_source}")
        st.caption(f"Local td_case2 env file: `{CASE_ENV_PATH}`")

    active_run_dir_text = str(st.session_state.get(traffic_ui.SESSION_RUN_DIR_KEY) or run_dir_input.strip() or initial_run_dir).strip()
    active_run_dir = Path(active_run_dir_text).expanduser() if active_run_dir_text else None
    if active_run_dir is not None and not active_run_dir.is_absolute():
        active_run_dir = active_run_dir.resolve()

    top_section = st.radio(
        "Section",
        SECTIONS,
        index=SECTIONS.index(initial_section) if initial_section in SECTIONS else 0,
        horizontal=True,
        key="td_case2_top_section",
    )
    selected_section = top_section

    if selected_section == "Pipeline Control":
        _render_control_metrics(active_run_dir)
        st.subheader("Video input")
        upload_cols = st.columns([1.2, 1.0])
        with upload_cols[0]:
            uploaded_file = st.file_uploader("Upload a local video for td_case2", type=["mp4", "avi", "mov", "mkv", "wmv", "ts", "3gp"])
            if uploaded_file is not None:
                if st.button("Save uploaded video", key="save_uploaded_video"):
                    saved_path = _save_uploaded_file(uploaded_file)
                    st.session_state[SESSION_UPLOAD_KEY] = str(saved_path)
                    st.success(f"Saved upload to {saved_path}")
        with upload_cols[1]:
            uploaded_path_value = str(st.session_state.get(SESSION_UPLOAD_KEY, "") or "")
            input_video_path = st.text_input("Input video path", value=uploaded_path_value)
            if input_video_path.strip() and Path(input_video_path).exists():
                st.video(input_video_path.strip())

        st.subheader("Pipeline parameters")
        ui_env_values = _collect_parameter_values()
        extra_env_text = st.text_area(
            "Extra environment overrides (JSON)",
            value="{}",
            help="Use this for any td_case2 environment variable that is not already surfaced above.",
            height=140,
        )

        env_preview = {
            key: value
            for key, value in _build_runner_env(
                ui_env_values=ui_env_values,
                input_video_path=input_video_path,
                selected_run_dir=str(active_run_dir or ""),
                extra_env_text=extra_env_text,
            ).items()
            if key.startswith("TD_CASE2_")
        }
        with st.expander("Effective TD_CASE2 environment preview"):
            st.json(env_preview)

        st.subheader("Run pipeline")
        run_cols = st.columns(4)
        resume_after_step03 = run_cols[0].checkbox("Resume search-ready from existing Step 03 run", value=False)
        run_search_ready = run_cols[1].button("Run search-ready (Step 01-10B)", use_container_width=True)
        run_vlm = run_cols[2].button("Run VLM pipeline (Step 11-16)", use_container_width=True)
        run_full = run_cols[3].button("Run full end-to-end", use_container_width=True)

        if run_search_ready or run_vlm or run_full:
            try:
                env = _build_runner_env(
                    ui_env_values=ui_env_values,
                    input_video_path=input_video_path,
                    selected_run_dir=str(active_run_dir or ""),
                    extra_env_text=extra_env_text,
                )
            except Exception as exc:
                st.error(f"Could not parse environment overrides: {exc}")
                st.stop()

            python_exe = _resolve_python_executable()

            if run_search_ready:
                args = [python_exe, str(CASE_ROOT / "run_td_case2_search_ready_pipeline.py")]
                if resume_after_step03 and active_run_dir is not None:
                    args.extend(["--resume-run-dir", str(active_run_dir)])
                with st.spinner("Running search-ready pipeline..."):
                    result = _run_command(args, env)
                _append_log("search_ready_pipeline", result)
                created_run_dir = parse_created_run_dir((result.stdout or "") + "\n" + (result.stderr or ""))
                if result.returncode == 0 and created_run_dir:
                    st.session_state[traffic_ui.SESSION_RUN_DIR_KEY] = created_run_dir
                    st.success(f"Search-ready pipeline finished. Active run set to {created_run_dir}")
                else:
                    st.error("Search-ready pipeline failed. Check the Logs tab.")

            if run_vlm:
                if active_run_dir is None:
                    st.error("Select or load a run directory before running the VLM pipeline.")
                else:
                    env[ENV_RUN_DIR] = str(active_run_dir)
                    with st.spinner("Running VLM pipeline..."):
                        result = _run_command([python_exe, str(CASE_ROOT / "run_td_case2_vlm_event_pipeline.py")], env)
                    _append_log("vlm_event_pipeline", result)
                    if result.returncode == 0:
                        st.success("VLM pipeline finished.")
                    else:
                        st.error("VLM pipeline failed. Check the Logs tab.")

            if run_full:
                args = [python_exe, str(CASE_ROOT / "run_td_case2_search_ready_pipeline.py")]
                with st.spinner("Running full search-ready pipeline..."):
                    search_result = _run_command(args, env)
                _append_log("full_pipeline_search_ready", search_result)
                created_run_dir = parse_created_run_dir((search_result.stdout or "") + "\n" + (search_result.stderr or ""))
                if search_result.returncode != 0 or not created_run_dir:
                    st.error("Search-ready part of the full pipeline failed. Check the Logs tab.")
                else:
                    st.session_state[traffic_ui.SESSION_RUN_DIR_KEY] = created_run_dir
                    env[ENV_RUN_DIR] = created_run_dir
                    with st.spinner("Running full VLM/event pipeline..."):
                        vlm_result = _run_command([python_exe, str(CASE_ROOT / "run_td_case2_vlm_event_pipeline.py")], env)
                    _append_log("full_pipeline_vlm", vlm_result)
                    if vlm_result.returncode == 0:
                        st.success(f"Full end-to-end pipeline finished. Active run set to {created_run_dir}")
                    else:
                        st.error("VLM part of the full pipeline failed. Check the Logs tab.")
                st.rerun()

        st.subheader("Pipeline stage status")
        _render_stage_table(active_run_dir)

    elif selected_section == "Search & Clips":
        _render_search_tab(active_run_dir)

    elif selected_section == "VLM Summary":
        _render_vlm_tab(active_run_dir)

    elif selected_section == "Events":
        _render_events_tab(active_run_dir)

    elif selected_section == "Logs":
        _render_logs_tab()


if __name__ == "__main__":
    main()
