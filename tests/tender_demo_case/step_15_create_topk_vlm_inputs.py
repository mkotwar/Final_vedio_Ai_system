from __future__ import annotations

import importlib.util
import json
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


def _load_or_create_selected_clips(run_dir: Path) -> list[dict[str, Any]]:
    selected_clips_path = run_dir / "14_selected_top_clips.json"
    if selected_clips_path.exists():
        selected_clips = _load_required_json(selected_clips_path)
        if not isinstance(selected_clips, list):
            raise ValueError("Expected a list in 14_selected_top_clips.json")
        return selected_clips

    print(
        "[tender-demo] 14_selected_top_clips.json is missing. "
        "Attempting to generate it from Step 14 selection logic."
    )
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


def read_frame_at_time(
    video_capture: cv2.VideoCapture,
    timestamp_seconds: float,
    fps: float,
    frame_count: int,
):
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
        cv2.putText(
            panel,
            label_text,
            (14, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return panel

    previous_panel = render_panel(previous_frame, labels["previous"])
    current_panel = render_panel(current_frame, labels["current"])
    next_panel = render_panel(next_frame, labels["next"])
    strip = cv2.hconcat([previous_panel, current_panel, next_panel])

    strip_height, strip_width = strip.shape[:2]
    footer_height = 34
    canvas = cv2.copyMakeBorder(
        strip,
        0,
        footer_height,
        0,
        0,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    cv2.putText(
        canvas,
        metadata_text,
        (14, strip_height + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def create_topk_vlm_inputs(run_dir: Path) -> dict[str, Any]:
    print("[tender-demo] Starting Step 15: create Top-K VLM inputs")

    selected_clips = _load_or_create_selected_clips(run_dir)
    video_info = _load_required_json(run_dir / "01_video_info.json")

    if not isinstance(video_info, dict):
        raise ValueError("Expected an object in 01_video_info.json")

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

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not open video for Top-K VLM input generation: {video_path}")

    items: list[dict[str, Any]] = []
    strips_created = 0
    strips_failed = 0

    try:
        for index, clip in enumerate(selected_clips, start=1):
            topk_vlm_input_id = f"topk_vlm_input_{index:06d}"
            clip_id = clip.get("clip_id")
            start_time = safe_float(clip.get("start_time"), 0.0)
            end_time = safe_float(clip.get("end_time"), start_time)
            expanded_start_time = safe_float(
                clip.get("expanded_start_time"),
                start_time,
            )
            expanded_end_time = safe_float(
                clip.get("expanded_end_time"),
                end_time,
            )

            clip_start = max(0.0, min(expanded_start_time, duration_seconds))
            clip_end = max(0.0, min(expanded_end_time, duration_seconds))
            clip_current = (start_time + end_time) / 2.0 if end_time >= start_time else start_time
            clip_current = max(0.0, min(clip_current, duration_seconds))

            previous_time = clip_start
            current_time = clip_current
            next_time = clip_end

            item_record = {
                "topk_vlm_input_id": topk_vlm_input_id,
                "source_clip_id": clip_id,
                "selection_order": clip.get("selection_order"),
                "rank": clip.get("rank"),
                "start_time": start_time,
                "end_time": end_time,
                "expanded_start_time": expanded_start_time,
                "expanded_end_time": expanded_end_time,
                "previous_time": previous_time,
                "current_time": current_time,
                "next_time": next_time,
                "strip_path": None,
                "source_frame_times": {
                    "previous": previous_time,
                    "current": current_time,
                    "next": next_time,
                },
                "source_frame_indices": {
                    "previous": None,
                    "current": None,
                    "next": None,
                },
                "selection_reasons": clip.get("selection_reasons", []),
                "ranked_clip_score": clip.get("ranked_clip_score"),
                "ranking_reasons": clip.get("ranking_reasons", []),
                "motion": clip.get("motion", {}),
                "yolo": clip.get("yolo", {}),
                "top_annotated_frame_path": clip.get("top_annotated_frame_path"),
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
                reasons_text = ", ".join(item_record["selection_reasons"]) if item_record["selection_reasons"] else "none"
                metadata_text = (
                    f"{clip_id or 'unknown_clip'} | rank {clip.get('rank', 'n/a')} | "
                    f"reasons: {reasons_text}"
                )
                strip_image = create_temporal_strip(
                    previous_frame=previous_frame,
                    current_frame=current_frame,
                    next_frame=next_frame,
                    labels=labels,
                    metadata_text=metadata_text,
                )

                output_filename = f"{topk_vlm_input_id}_{clip_id or 'clip'}.jpg"
                output_path = output_dir / output_filename
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

    manifest = {
        "video_name": video_info.get("video_name"),
        "video_path": str(video_path),
        "total_selected_clips": len(selected_clips),
        "total_strips_created": strips_created,
        "total_failed": strips_failed,
        "vlm_input_folder": str(output_dir),
        "items": items,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[tender-demo] Selected clips received: {len(selected_clips)}")
    print(f"[tender-demo] Strips created: {strips_created}")
    print(f"[tender-demo] Strips failed: {strips_failed}")
    print(f"[tender-demo] Top-K VLM input folder path: {output_dir}")
    print(f"[tender-demo] Top-K VLM input manifest path: {manifest_path}")

    return manifest
