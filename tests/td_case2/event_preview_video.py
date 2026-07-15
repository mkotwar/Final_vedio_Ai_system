from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2

from stage_checks import read_json, write_json


TRACKS_FILE = "04B_tracks.json"
ADAPTIVE_FRAMES_FILE = "02A_adaptive_frames.json"
SAMPLED_FRAMES_FILE = "02_sampled_frames.json"
OUTPUT_DIR = "10C_search_event_clips"
MANIFEST_FILE = "10C_search_event_clips_manifest.json"
DEFAULT_CLIP_FPS = 2.0
DEFAULT_BEST_FRAME_HOLD_SECONDS = 1.5


def format_seconds_text(seconds: float | int | None) -> str:
    total_seconds = max(0, int(round(float(seconds or 0.0))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _relative_to_run(run_dir: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(run_dir.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve())


def resolve_run_path(run_dir: Path, path_value: str | None) -> Path | None:
    if not path_value:
        return None
    normalized = str(path_value).strip().replace("\\", "/")
    if not normalized:
        return None
    path = Path(normalized)
    if path.is_absolute():
        return path
    return (run_dir / path).resolve()


def _load_frame_image_map(run_dir: Path) -> dict[str, str]:
    payload_path = run_dir / ADAPTIVE_FRAMES_FILE
    frame_items: list[dict[str, Any]] = []
    if payload_path.exists():
        payload = read_json(payload_path)
        frame_items = [item for item in list(payload.get("selected_frames", [])) if isinstance(item, dict)]
    else:
        fallback_payload_path = run_dir / SAMPLED_FRAMES_FILE
        if fallback_payload_path.exists():
            payload = read_json(fallback_payload_path)
            frame_items = [item for item in list(payload.get("sampled_frames", [])) if isinstance(item, dict)]
    return {
        str(item.get("frame_id", "") or ""): str(item.get("image_path", "") or "")
        for item in frame_items
        if str(item.get("frame_id", "") or "") and str(item.get("image_path", "") or "")
    }


def _load_track_map(run_dir: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(run_dir / TRACKS_FILE)
    return {
        str(item.get("track_id", "") or ""): item
        for item in list(payload.get("tracks", []))
        if isinstance(item, dict) and str(item.get("track_id", "") or "")
    }


def _read_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / MANIFEST_FILE
    if not manifest_path.exists():
        return {"status": "success", "clips": {}}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "success", "clips": {}}
    if not isinstance(payload, dict):
        return {"status": "success", "clips": {}}
    clips = payload.get("clips")
    if not isinstance(clips, dict):
        payload["clips"] = {}
    return payload


def _write_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    write_json(run_dir / MANIFEST_FILE, manifest)


def _record_key(record: dict[str, Any]) -> str:
    return str(
        record.get("object_record_id")
        or record.get("search_record_id")
        or record.get("track_id")
        or record.get("detection_id")
        or "unknown_record"
    )


def _clip_name(record: dict[str, Any]) -> str:
    raw_key = _record_key(record)
    safe_key = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in raw_key)
    return f"{safe_key}_event_preview.mp4"


def _frame_entries_for_record(run_dir: Path, record: dict[str, Any]) -> list[dict[str, Any]]:
    full_frame_path = resolve_run_path(run_dir, str(record.get("full_frame_path", "") or ""))
    frame_image_map = _load_frame_image_map(run_dir)
    track_id = str(record.get("track_id", "") or "")
    if track_id:
        track_map = _load_track_map(run_dir)
        track = dict(track_map.get(track_id, {}))
        detections = [item for item in list(track.get("detections", [])) if isinstance(item, dict)]
        entries: list[dict[str, Any]] = []
        for detection in detections:
            frame_id = str(detection.get("frame_id", "") or "")
            frame_path = resolve_run_path(run_dir, frame_image_map.get(frame_id))
            if frame_path is None or not frame_path.exists():
                continue
            entries.append(
                {
                    "frame_id": frame_id,
                    "timestamp_seconds": float(detection.get("timestamp_seconds", 0.0) or 0.0),
                    "timestamp_text": format_seconds_text(float(detection.get("timestamp_seconds", 0.0) or 0.0)),
                    "bbox_xyxy": list(detection.get("bbox_xyxy", [])),
                    "frame_path": frame_path,
                    "is_best_frame": frame_path == full_frame_path,
                }
            )
        if entries:
            return entries

    if full_frame_path is not None and full_frame_path.exists():
        timestamp_seconds = float(record.get("timestamp_seconds", 0.0) or 0.0)
        return [
            {
                "frame_id": str(record.get("frame_id", "") or ""),
                "timestamp_seconds": timestamp_seconds,
                "timestamp_text": str(record.get("timestamp_text", "") or format_seconds_text(timestamp_seconds)),
                "bbox_xyxy": list(record.get("bbox_xyxy", [])),
                "frame_path": full_frame_path,
                "is_best_frame": True,
            }
        ]
    return []


def get_existing_event_preview(run_dir: Path, record: dict[str, Any]) -> dict[str, Any] | None:
    manifest = _read_manifest(run_dir)
    clip_payload = manifest.get("clips", {}).get(_record_key(record))
    if not isinstance(clip_payload, dict):
        return None
    clip_path = resolve_run_path(run_dir, str(clip_payload.get("event_clip_path", "") or ""))
    if clip_path is None or not clip_path.exists():
        return None
    return clip_payload


def _draw_overlay(
    frame: Any,
    *,
    video_time_text: str,
    detected_window_text: str,
    record_label: str,
    best_frame: bool,
    bbox_xyxy: list[Any],
) -> Any:
    output = frame.copy()
    height, width = output.shape[:2]
    top_panel_height = 74
    bottom_panel_height = 36
    cv2.rectangle(output, (0, 0), (width, top_panel_height), (0, 0, 0), thickness=-1)
    cv2.rectangle(output, (0, height - bottom_panel_height), (width, height), (0, 0, 0), thickness=-1)
    cv2.putText(output, f"Video time: {video_time_text}", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output, detected_window_text, (18, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    footer_text = record_label + (" | BEST FRAME" if best_frame else "")
    cv2.putText(output, footer_text, (18, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    if len(bbox_xyxy) == 4:
        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox_xyxy]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        if x2 > x1 and y2 > y1:
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 215, 255), 3)
    return output


def build_event_preview_clip(
    run_dir: Path,
    record: dict[str, Any],
    *,
    clip_fps: float = DEFAULT_CLIP_FPS,
    best_frame_hold_seconds: float = DEFAULT_BEST_FRAME_HOLD_SECONDS,
) -> dict[str, Any]:
    existing = get_existing_event_preview(run_dir, record)
    if existing is not None:
        return existing

    frame_entries = _frame_entries_for_record(run_dir, record)
    if not frame_entries:
        raise FileNotFoundError(f"No frame sequence is available for record {_record_key(record)}.")

    output_dir = run_dir / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / _clip_name(record)

    first_frame = cv2.imread(str(frame_entries[0]["frame_path"]))
    if first_frame is None:
        raise FileNotFoundError(f"Could not read frame image: {frame_entries[0]['frame_path']}")
    frame_height, frame_width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(clip_fps),
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    start_seconds = float(record.get("first_seen_seconds", record.get("timestamp_seconds", 0.0)) or 0.0)
    end_seconds = float(record.get("last_seen_seconds", record.get("timestamp_seconds", 0.0)) or 0.0)
    if end_seconds < start_seconds:
        end_seconds = start_seconds
    detected_window_text = (
        f"Detected: {format_seconds_text(start_seconds)} to {format_seconds_text(end_seconds)}"
        f" | Duration: {round(max(0.0, end_seconds - start_seconds), 2)}s"
    )
    record_label = str(record.get("class_name", "object") or "object").title()
    if record.get("track_id"):
        record_label += f" | {record.get('track_id')}"

    source_frame_count = 0
    for entry in frame_entries:
        frame = cv2.imread(str(entry["frame_path"]))
        if frame is None:
            continue
        if frame.shape[0] != frame_height or frame.shape[1] != frame_width:
            frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
        annotated = _draw_overlay(
            frame,
            video_time_text=str(entry["timestamp_text"]),
            detected_window_text=detected_window_text,
            record_label=record_label,
            best_frame=bool(entry["is_best_frame"]),
            bbox_xyxy=list(entry.get("bbox_xyxy", [])),
        )
        writer.write(annotated)
        source_frame_count += 1

    best_entry = next((item for item in frame_entries if item.get("is_best_frame")), frame_entries[-1])
    best_frame = cv2.imread(str(best_entry["frame_path"]))
    if best_frame is not None:
        if best_frame.shape[0] != frame_height or best_frame.shape[1] != frame_width:
            best_frame = cv2.resize(best_frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
        hold_frames = max(2, int(round(float(clip_fps) * float(best_frame_hold_seconds))))
        for _ in range(hold_frames):
            annotated = _draw_overlay(
                best_frame,
                video_time_text=str(best_entry["timestamp_text"]),
                detected_window_text=detected_window_text,
                record_label=record_label,
                best_frame=True,
                bbox_xyxy=list(best_entry.get("bbox_xyxy", [])),
            )
            writer.write(annotated)

    writer.release()

    clip_payload = {
        "record_key": _record_key(record),
        "event_clip_path": _relative_to_run(run_dir, output_path),
        "best_frame_path": str(record.get("full_frame_path", "") or ""),
        "start_timestamp_seconds": round(start_seconds, 6),
        "end_timestamp_seconds": round(end_seconds, 6),
        "start_timestamp_text": format_seconds_text(start_seconds),
        "end_timestamp_text": format_seconds_text(end_seconds),
        "detection_duration_seconds": round(max(0.0, end_seconds - start_seconds), 6),
        "source_frame_count": source_frame_count,
        "clip_frame_count": source_frame_count + max(2, int(round(float(clip_fps) * float(best_frame_hold_seconds)))),
        "clip_fps": float(clip_fps),
        "source_type": str(record.get("source_type", "unknown") or "unknown"),
        "track_id": record.get("track_id"),
        "class_name": record.get("class_name"),
    }
    manifest = _read_manifest(run_dir)
    manifest.setdefault("clips", {})
    manifest["clips"][_record_key(record)] = clip_payload
    _write_manifest(run_dir, manifest)
    return clip_payload
