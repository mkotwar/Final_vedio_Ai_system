from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import cv2


def format_seconds(seconds: float | int | None) -> str:
    if seconds is None:
        return "unknown time"
    try:
        total_seconds = float(seconds)
    except (TypeError, ValueError):
        return "unknown time"
    if total_seconds < 0:
        return "unknown time"
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    remaining_seconds = total_seconds - (hours * 3600) - (minutes * 60)
    if float(remaining_seconds).is_integer():
        return f"{hours:02d}:{minutes:02d}:{int(remaining_seconds):02d}"
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:04.1f}"


def parse_timestamp_to_seconds(value: str) -> float:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Timestamp is empty.")
    parts = text.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        minutes, seconds = parts
        return (int(minutes) * 60) + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return (int(hours) * 3600) + (int(minutes) * 60) + float(seconds)
    raise ValueError(f"Unsupported timestamp format: {value!r}")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_required_json(path: Path) -> list[dict[str, Any]] | dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Top-K VLM input file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_json(path: Path) -> list[dict[str, Any]] | dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _read_env_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _load_motion_state_hints(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "11b_object_motion_states.json"
    payload = _load_optional_json(path)
    if not isinstance(payload, dict):
        return {}
    return {
        str(item.get("clip_id", "")).strip(): item
        for item in payload.get("clip_motion_states", [])
        if isinstance(item, dict) and str(item.get("clip_id", "")).strip()
    }


def _strategy_defaults(mode: str) -> tuple[str, int]:
    normalized = str(mode or "").strip().lower()
    if normalized == "sensitive incident review":
        return "multi_focus", 40
    if normalized in {"high accuracy", "high accuracy review"}:
        return "multi_focus", 50
    return "center_only", 25


def _read_critical_timestamps() -> list[float]:
    raw_value = os.environ.get("TENDER_DEMO_CRITICAL_TIMESTAMPS", "").strip()
    if not raw_value:
        return []
    values: list[float] = []
    for part in raw_value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(parse_timestamp_to_seconds(part))
        except ValueError:
            print(f"[tender-demo] Warning: invalid critical timestamp ignored: {part!r}")
    return values


def _load_or_create_selected_clips(run_dir: Path) -> list[dict[str, Any]]:
    selected_clips_path = run_dir / "14b_coverage_selected_clips.json"
    if selected_clips_path.exists():
        payload = _load_required_json(selected_clips_path)
        if not isinstance(payload, list):
            raise ValueError("Expected a list in 14b_coverage_selected_clips.json")
        return payload

    selected_clips_path = run_dir / "14_selected_top_clips.json"
    if selected_clips_path.exists():
        payload = _load_required_json(selected_clips_path)
        if not isinstance(payload, list):
            raise ValueError("Expected a list in 14_selected_top_clips.json")
        return payload

    print("[tender-demo] Step 14 output missing. Attempting to generate selected clips.")
    try:
        from tests.tender_demo_case.step_14_select_topk_clips import select_topk_clips_for_qwen
    except ModuleNotFoundError:
        step_14_path = Path(__file__).resolve().parent / "step_14_select_topk_clips.py"
        spec = importlib.util.spec_from_file_location("step_14_select_topk_clips", step_14_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load Step 14 selection module from: {step_14_path}")
        step_14_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step_14_module)
        select_topk_clips_for_qwen = step_14_module.select_topk_clips_for_qwen

    selection_result = select_topk_clips_for_qwen(run_dir)
    selected_clips = selection_result.get("selected_clips", [])
    if not isinstance(selected_clips, list):
        raise ValueError("Step 14 selection did not return a valid selected_clips list.")
    return selected_clips


def _load_candidate_pool(run_dir: Path) -> list[dict[str, Any]]:
    payload = _load_optional_json(run_dir / "02c_frame_candidate_pool.json")
    if not isinstance(payload, dict):
        return []
    items = payload.get("items", [])
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def read_frame_at_time(video_capture: cv2.VideoCapture, timestamp_seconds: float, fps: float, frame_count: int):
    if fps <= 0:
        raise ValueError("FPS must be greater than 0 for frame extraction.")
    frame_idx = int(round(timestamp_seconds * fps))
    frame_idx = max(0, min(frame_count - 1, frame_idx))
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = video_capture.read()
    if not success or frame is None:
        raise RuntimeError(f"Failed to read frame at timestamp {timestamp_seconds} seconds")
    return frame_idx, frame


def create_temporal_strip(previous_frame, current_frame, next_frame, labels, metadata_text):
    panel_width = 640
    panel_height = 360

    def render_panel(frame, label_text):
        panel = cv2.resize(frame, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
        cv2.rectangle(panel, (0, 0), (panel_width, 42), (0, 0, 0), thickness=-1)
        cv2.putText(panel, label_text, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        return panel

    strip = cv2.hconcat(
        [
            render_panel(previous_frame, labels["previous"]),
            render_panel(current_frame, labels["current"]),
            render_panel(next_frame, labels["next"]),
        ]
    )
    strip_height, _ = strip.shape[:2]
    footer_height = 34
    canvas = cv2.copyMakeBorder(strip, 0, footer_height, 0, 0, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(canvas, metadata_text, (14, strip_height + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _clip_time_range(clip: dict[str, Any], duration_seconds: float) -> tuple[float, float, float, float]:
    start_time = safe_float(clip.get("start_time"), 0.0)
    end_time = safe_float(clip.get("end_time"), start_time)
    expanded_start_time = safe_float(clip.get("expanded_start_time"), start_time)
    expanded_end_time = safe_float(clip.get("expanded_end_time"), end_time)
    clip_start = max(0.0, min(expanded_start_time, duration_seconds))
    clip_end = max(0.0, min(expanded_end_time, duration_seconds))
    center_time = (start_time + end_time) / 2.0 if end_time >= start_time else start_time
    center_time = max(0.0, min(center_time, duration_seconds))
    return start_time, end_time, clip_start, clip_end if clip_end >= clip_start else clip_start


def _clip_candidate_pool_frames(clip: dict[str, Any], candidate_pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    start_time = safe_float(clip.get("expanded_start_time", clip.get("start_time")), 0.0)
    end_time = safe_float(clip.get("expanded_end_time", clip.get("end_time")), start_time)
    return [
        item
        for item in candidate_pool
        if start_time <= safe_float(item.get("timestamp_seconds"), -1.0) <= end_time
    ]


def _best_motion_time(clip: dict[str, Any], candidate_pool: list[dict[str, Any]], fallback: float) -> float:
    candidates = _clip_candidate_pool_frames(clip, candidate_pool)
    if not candidates:
        return fallback
    candidates.sort(key=lambda item: (-safe_float(item.get("adaptive_motion_score", item.get("motion_score_norm")), 0.0), safe_float(item.get("timestamp_seconds"), 0.0)))
    return safe_float(candidates[0].get("timestamp_seconds"), fallback)


def _best_adaptive_time(clip: dict[str, Any], candidate_pool: list[dict[str, Any]], fallback: float) -> tuple[float, list[str]]:
    candidates = [
        item for item in _clip_candidate_pool_frames(clip, candidate_pool)
        if bool(item.get("source_adaptive"))
    ]
    if not candidates:
        return fallback, []
    candidates.sort(
        key=lambda item: (
            -(
                safe_float(item.get("adaptive_motion_score"), 0.0)
                + safe_float(item.get("adaptive_histogram_diff"), 0.0)
                + (1.0 - safe_float(item.get("adaptive_similarity_score"), 1.0))
            ),
            safe_float(item.get("timestamp_seconds"), 0.0),
        )
    )
    return safe_float(candidates[0].get("timestamp_seconds"), fallback), list(candidates[0].get("adaptive_reasons", []) or [])


def _critical_time_for_clip(clip: dict[str, Any], critical_timestamps: list[float]) -> float | None:
    start_time = safe_float(clip.get("expanded_start_time", clip.get("start_time")), 0.0)
    end_time = safe_float(clip.get("expanded_end_time", clip.get("end_time")), start_time)
    for value in critical_timestamps:
        if start_time <= value <= end_time:
            return value
    return None


def _dedupe_centers(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    for item in candidates:
        center_time = safe_float(item.get("center_time"), 0.0)
        if any(abs(safe_float(existing.get("center_time"), 0.0) - center_time) <= 2.0 for existing in accepted):
            continue
        accepted.append(item)
    return accepted


def _build_vlm_input_candidates(
    clip: dict[str, Any],
    candidate_pool: list[dict[str, Any]],
    critical_timestamps: list[float],
    strategy: str,
    duration_seconds: float,
) -> list[dict[str, Any]]:
    start_time = safe_float(clip.get("start_time"), 0.0)
    end_time = safe_float(clip.get("end_time"), start_time)
    expanded_start_time = safe_float(clip.get("expanded_start_time"), start_time)
    expanded_end_time = safe_float(clip.get("expanded_end_time"), end_time)
    clip_start = max(0.0, min(expanded_start_time, duration_seconds))
    clip_end = max(0.0, min(expanded_end_time, duration_seconds))
    center_time = max(0.0, min((start_time + end_time) / 2.0 if end_time >= start_time else start_time, duration_seconds))
    peak_motion_time = _best_motion_time(clip, candidate_pool, center_time)
    adaptive_time, adaptive_reasons = _best_adaptive_time(clip, candidate_pool, center_time)
    critical_time = _critical_time_for_clip(clip, critical_timestamps)

    candidates = [
        {
            "center_time": center_time,
            "input_strategy": "center_only",
            "selection_reason": ["clip_center"],
        }
    ]
    if strategy in {"peak_motion", "multi_focus"}:
        candidates.append(
            {
                "center_time": peak_motion_time,
                "input_strategy": "peak_motion",
                "selection_reason": ["peak_motion_frame"],
            }
        )
    if strategy in {"adaptive_peak", "multi_focus"}:
        candidates.append(
            {
                "center_time": adaptive_time,
                "input_strategy": "adaptive_peak",
                "selection_reason": adaptive_reasons or ["adaptive_peak_frame"],
            }
        )
    if strategy == "multi_focus" and critical_time is not None:
        candidates.append(
            {
                "center_time": critical_time,
                "input_strategy": "manual_critical_timestamp",
                "selection_reason": ["manual_critical_timestamp"],
            }
        )
    if strategy == "center_only":
        return candidates[:1]
    if strategy == "peak_motion":
        return [candidates[1] if len(candidates) > 1 else candidates[0]]
    if strategy == "adaptive_peak":
        return [candidates[-1]]
    return _dedupe_centers(candidates)


def _build_coverage_row(target_seconds: float, items: list[dict[str, Any]], threshold_seconds: float) -> dict[str, Any]:
    nearest_previous = None
    nearest_current = None
    nearest_next = None
    for item in items:
        previous_distance = abs(safe_float(item.get("previous_time"), 0.0) - target_seconds)
        current_distance = abs(safe_float(item.get("current_time"), 0.0) - target_seconds)
        next_distance = abs(safe_float(item.get("next_time"), 0.0) - target_seconds)
        if nearest_previous is None or previous_distance < nearest_previous["distance_seconds"]:
            nearest_previous = {"distance_seconds": round(previous_distance, 3), "time": safe_float(item.get("previous_time"), 0.0)}
        if nearest_current is None or current_distance < nearest_current["distance_seconds"]:
            nearest_current = {"distance_seconds": round(current_distance, 3), "time": safe_float(item.get("current_time"), 0.0)}
        if nearest_next is None or next_distance < nearest_next["distance_seconds"]:
            nearest_next = {"distance_seconds": round(next_distance, 3), "time": safe_float(item.get("next_time"), 0.0)}
    current_covered = bool(nearest_current and nearest_current["distance_seconds"] <= threshold_seconds)
    context_covered = bool(
        not current_covered
        and (
            (nearest_previous and nearest_previous["distance_seconds"] <= threshold_seconds)
            or (nearest_next and nearest_next["distance_seconds"] <= threshold_seconds)
        )
    )
    coverage_type = "current_panel_covered" if current_covered else "context_panel_covered" if context_covered else "missing"
    return {
        "timestamp": format_seconds(target_seconds),
        "tender_coverage_type": coverage_type,
        "nearest_current_time": nearest_current["time"] if nearest_current else None,
        "nearest_current_distance": nearest_current["distance_seconds"] if nearest_current else None,
        "nearest_previous_time": nearest_previous["time"] if nearest_previous else None,
        "nearest_previous_distance": nearest_previous["distance_seconds"] if nearest_previous else None,
        "nearest_next_time": nearest_next["time"] if nearest_next else None,
        "nearest_next_distance": nearest_next["distance_seconds"] if nearest_next else None,
    }


def _write_coverage_audit(run_dir: Path, items: list[dict[str, Any]], critical_timestamps: list[float]) -> dict[str, Any]:
    threshold_seconds = _read_env_float("TENDER_DEMO_ADAPTIVE_TARGET_WINDOW_SECONDS", 3.0)
    rows = [_build_coverage_row(timestamp, items, threshold_seconds) for timestamp in critical_timestamps]
    current_covered = [row["timestamp"] for row in rows if row["tender_coverage_type"] == "current_panel_covered"]
    context_covered = [row["timestamp"] for row in rows if row["tender_coverage_type"] == "context_panel_covered"]
    missing = [row["timestamp"] for row in rows if row["tender_coverage_type"] == "missing"]
    audit = {
        "critical_timestamps_requested": [format_seconds(value) for value in critical_timestamps],
        "current_panel_covered": current_covered,
        "context_panel_covered": context_covered,
        "missing": missing,
        "total_vlm_inputs": len(items),
        "coverage_threshold_seconds": threshold_seconds,
        "coverage_rows": rows,
    }
    output_path = run_dir / "15_vlm_coverage_audit.json"
    output_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if missing:
        print(f"[tender-demo] Warning: critical timestamps missing as CURRENT panel: {missing}")
    print(f"[tender-demo] VLM coverage audit path: {output_path}")
    return audit


def create_topk_vlm_inputs(run_dir: Path) -> dict[str, Any]:
    print("[tender-demo] Starting Step 15: create Top-K VLM inputs")
    selected_clips = _load_or_create_selected_clips(run_dir)
    video_info = _load_required_json(run_dir / "01_video_info.json")
    if not isinstance(video_info, dict):
        raise ValueError("Expected an object in 01_video_info.json")

    mode = os.environ.get("TENDER_DEMO_ANALYSIS_SENSITIVITY_MODE", "Balanced")
    default_strategy, default_max_vlm_inputs = _strategy_defaults(mode)
    strategy = os.environ.get("TENDER_DEMO_VLM_INPUT_STRATEGY", default_strategy).strip().lower() or default_strategy
    max_vlm_inputs = max(1, _read_env_int("TENDER_DEMO_MAX_VLM_INPUTS", default_max_vlm_inputs))
    critical_timestamps = _read_critical_timestamps()

    video_path_value = video_info.get("video_path")
    if not video_path_value:
        raise FileNotFoundError("01_video_info.json is missing video_path.")
    video_path = Path(str(video_path_value))
    if not video_path.exists():
        raise FileNotFoundError(f"Video path from 01_video_info.json does not exist: {video_path}")

    fps = safe_float(video_info.get("fps"), 0.0)
    frame_count = int(safe_float(video_info.get("total_frames"), 0))
    duration_seconds = safe_float(video_info.get("duration_seconds"), 0.0)
    output_dir = run_dir / "15_topk_vlm_inputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "15_topk_vlm_inputs.json"
    candidate_pool = _load_candidate_pool(run_dir)
    motion_state_hints_by_clip_id = _load_motion_state_hints(run_dir)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not open video for Top-K VLM input generation: {video_path}")

    items: list[dict[str, Any]] = []
    strips_created = 0
    strips_failed = 0
    strategies_used: dict[str, int] = {}

    try:
        input_index = 0
        for clip in selected_clips:
            if input_index >= max_vlm_inputs:
                break
            clip_id = clip.get("clip_id")
            candidates = _build_vlm_input_candidates(clip, candidate_pool, critical_timestamps, strategy, duration_seconds)
            for candidate in candidates:
                if input_index >= max_vlm_inputs:
                    break
                input_index += 1
                current_time = max(0.0, min(safe_float(candidate.get("center_time"), 0.0), duration_seconds))
                previous_time = max(0.0, min(current_time - 1.0, duration_seconds))
                next_time = max(0.0, min(current_time + 1.0, duration_seconds))
                topk_vlm_input_id = f"topk_vlm_input_{input_index:06d}"
                strategies_used[candidate["input_strategy"]] = strategies_used.get(candidate["input_strategy"], 0) + 1
                item_record = {
                    "topk_vlm_input_id": topk_vlm_input_id,
                    "input_id": topk_vlm_input_id,
                    "source_clip_id": clip_id,
                    "selection_order": clip.get("selection_order"),
                    "rank": clip.get("rank"),
                    "start_time": safe_float(clip.get("start_time"), 0.0),
                    "end_time": safe_float(clip.get("end_time"), 0.0),
                    "expanded_start_time": safe_float(clip.get("expanded_start_time"), safe_float(clip.get("start_time"), 0.0)),
                    "expanded_end_time": safe_float(clip.get("expanded_end_time"), safe_float(clip.get("end_time"), 0.0)),
                    "center_time": current_time,
                    "previous_time": previous_time,
                    "current_time": current_time,
                    "next_time": next_time,
                    "input_strategy": candidate["input_strategy"],
                    "selection_reason": list(candidate.get("selection_reason", []) or []),
                    "selection_reasons": list(clip.get("selection_reasons", []) or []),
                    "ranked_clip_score": clip.get("ranked_clip_score"),
                    "ranking_reasons": list(clip.get("ranking_reasons", []) or []),
                    "motion": clip.get("motion", {}),
                    "adaptive": clip.get("adaptive", {}),
                    "yolo": clip.get("yolo", {}),
                    "motion_state_hints": motion_state_hints_by_clip_id.get(str(clip_id).strip(), {}),
                    "top_annotated_frame_path": clip.get("top_annotated_frame_path"),
                    "strip_path": None,
                    "source_frame_times": {"previous": previous_time, "current": current_time, "next": next_time},
                    "source_frame_indices": {"previous": None, "current": None, "next": None},
                    "creation_success": False,
                    "error": None,
                }
                try:
                    previous_idx, previous_frame = read_frame_at_time(capture, previous_time, fps, frame_count)
                    current_idx, current_frame = read_frame_at_time(capture, current_time, fps, frame_count)
                    next_idx, next_frame = read_frame_at_time(capture, next_time, fps, frame_count)
                    item_record["source_frame_indices"] = {
                        "previous": previous_idx,
                        "current": current_idx,
                        "next": next_idx,
                    }
                    labels = {
                        "previous": f"PREVIOUS {format_seconds(previous_time)}",
                        "current": f"CURRENT {format_seconds(current_time)}",
                        "next": f"NEXT {format_seconds(next_time)}",
                    }
                    metadata_text = (
                        f"{clip_id or 'unknown_clip'} | rank {clip.get('rank', 'n/a')} | "
                        f"strategy: {candidate['input_strategy']} | reasons: {', '.join(candidate.get('selection_reason', [])) or 'none'}"
                    )
                    strip_image = create_temporal_strip(previous_frame, current_frame, next_frame, labels, metadata_text)
                    output_path = output_dir / f"{topk_vlm_input_id}_{clip_id or 'clip'}.jpg"
                    if not cv2.imwrite(str(output_path), strip_image):
                        raise RuntimeError(f"Failed to write Top-K VLM strip image: {output_path}")
                    repo_root = Path(__file__).resolve().parents[2]
                    item_record["strip_path"] = output_path.resolve().relative_to(repo_root).as_posix()
                    item_record["creation_success"] = True
                    strips_created += 1
                except Exception as exc:
                    item_record["error"] = str(exc)
                    strips_failed += 1
                items.append(item_record)
    finally:
        capture.release()

    coverage_audit = _write_coverage_audit(run_dir, items, critical_timestamps)
    manifest = {
        "video_name": video_info.get("video_name"),
        "video_path": str(video_path),
        "selection_source": "14b_coverage_selected_clips.json" if (run_dir / "14b_coverage_selected_clips.json").exists() else "14_selected_top_clips.json",
        "vlm_input_strategy": strategy,
        "max_vlm_inputs": max_vlm_inputs,
        "strategies_used": strategies_used,
        "critical_timestamps": [format_seconds(value) for value in critical_timestamps],
        "total_selected_clips": len(selected_clips),
        "total_strips_created": strips_created,
        "total_failed": strips_failed,
        "vlm_input_folder": str(output_dir),
        "coverage_audit_path": str(run_dir / "15_vlm_coverage_audit.json"),
        "items": items,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[tender-demo] Selected clips received: {len(selected_clips)}")
    print(f"[tender-demo] VLM input strategy: {strategy}")
    print(f"[tender-demo] Strips created: {strips_created}")
    print(f"[tender-demo] Strips failed: {strips_failed}")
    print(f"[tender-demo] Top-K VLM input folder path: {output_dir}")
    print(f"[tender-demo] Top-K VLM input manifest path: {manifest_path}")
    return {
        **manifest,
        "coverage_audit": coverage_audit,
    }
