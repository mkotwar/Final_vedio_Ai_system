from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_EXPORT_PRIORITY_CLIPS = True
DEFAULT_EXPORT_REVIEW_CLIPS = True
DEFAULT_EXPORT_NORMAL_CLIPS = False
DEFAULT_EXPORT_FPS = 5
DEFAULT_EXPORT_FORMAT = "mp4"
DEFAULT_CREATE_COMPILED_REVIEW_VIDEO = True
DEFAULT_COMPILE_NORMAL_IF_NO_EVENTS = True
DEFAULT_COMPILED_VIDEO_FPS = 5
DEFAULT_SECONDS_PER_FRAME = 1.0
DEFAULT_SECONDS_PER_TITLE_CARD = 1.5
DEFAULT_STRIP_PANEL_COUNT = 3
DEFAULT_COMPILED_FRAME_WIDTH = 1280
DEFAULT_COMPILED_FRAME_HEIGHT = 720


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


def safe_bool_env(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def safe_int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError:
        return default
    return value if value > 0 else default


def safe_float_env(name: str, default: float) -> float:
    raw_value = os.environ.get(name, str(default)).strip()
    try:
        value = float(raw_value)
    except ValueError:
        return default
    return value if value > 0 else default


def safe_filename(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    normalized = normalized.strip("._")
    return normalized or "clip"


def _load_required_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Step 18 input file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in JSON file: {path}")
    return payload


def _relative_to_repo(path_value: Any) -> str | None:
    if not path_value:
        return None
    repo_root = Path(__file__).resolve().parents[2]
    path = Path(str(path_value))
    try:
        if path.is_absolute():
            return path.resolve().relative_to(repo_root).as_posix()
        return path.as_posix()
    except Exception:
        return str(path_value)


def _absolute_from_repo(path_value: Any) -> Path | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / path


def _load_step17_summary(run_dir: Path) -> dict[str, Any]:
    return _load_required_json(run_dir / "17_topk_final_summary.json")


def _load_optional_topk_vlm_inputs(run_dir: Path) -> dict[str, dict[str, Any]]:
    manifest_path = run_dir / "15_topk_vlm_inputs.json"
    if not manifest_path.exists():
        return {}

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = payload.get("items", []) if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return {}

    indexed_items: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        clip_id = str(item.get("source_clip_id", "")).strip()
        if clip_id:
            indexed_items[clip_id] = item
    return indexed_items


def _split_strip_into_frames(image) -> list[Any]:
    if image is None or image.size == 0:
        return []

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return []

    segment_count = 3 if width >= (height * 2) else 1
    segment_width = width // segment_count
    if segment_width <= 0:
        return []

    frames: list[Any] = []
    for index in range(segment_count):
        start_x = index * segment_width
        end_x = width if index == segment_count - 1 else (index + 1) * segment_width
        frame = image[:, start_x:end_x]
        if frame.size == 0:
            continue
        frames.append(frame)
    return frames


def resize_frame(frame, width: int, height: int):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def draw_text_box(frame, lines: list[str], x: int, y: int):
    valid_lines = [str(line).strip() for line in lines if str(line or "").strip()]
    if not valid_lines:
        return frame

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2
    line_height = 30
    padding = 12

    max_width = 0
    for line in valid_lines:
        (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_width = max(max_width, text_width)
    box_height = (line_height * len(valid_lines)) + (padding * 2)
    box_width = max_width + (padding * 2)

    cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), (0, 0, 0), thickness=-1)
    baseline_y = y + padding + 20
    for line in valid_lines:
        cv2.putText(
            frame,
            line,
            (x + padding, baseline_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        baseline_y += line_height
    return frame


def create_title_card(width: int, height: int, lines: list[str]):
    card = np.zeros((height, width, 3), dtype=np.uint8)
    title_lines = [str(line).strip() for line in lines if str(line or "").strip()]
    title_lines = title_lines or ["Tender Demo Review Clip"]

    draw_text_box(card, title_lines, 60, 80)
    return card


def split_strip_into_panels(strip_image, panel_count: int) -> list[Any]:
    if strip_image is None or strip_image.size == 0:
        return []

    height, width = strip_image.shape[:2]
    if panel_count <= 0:
        panel_count = DEFAULT_STRIP_PANEL_COUNT

    panels: list[Any] = []
    for index in range(panel_count):
        start_x = int((width * index) / panel_count)
        end_x = int((width * (index + 1)) / panel_count)
        panel = strip_image[:, start_x:end_x]
        if panel.size == 0:
            continue
        panels.append(panel)
    return panels


def write_frame_for_duration(writer, frame, fps: int, seconds: float) -> int:
    frame_count = max(1, int(round(float(fps) * float(seconds))))
    for _ in range(frame_count):
        writer.write(frame)
    return frame_count


def find_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def ensure_even_dimensions(width: int, height: int) -> tuple[int, int]:
    safe_width = max(2, int(width))
    safe_height = max(2, int(height))
    if safe_width % 2 == 1:
        safe_width -= 1
    if safe_height % 2 == 1:
        safe_height -= 1
    return safe_width, safe_height


def save_compiled_frames_to_temp(frames, frames_dir: Path, width: int, height: int) -> int:
    frames_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    for existing_file in frames_dir.glob("frame_*.jpg"):
        try:
            existing_file.unlink()
        except OSError:
            pass

    even_width, even_height = ensure_even_dimensions(width, height)
    for index, frame in enumerate(frames, start=1):
        normalized_frame = resize_frame(frame, even_width, even_height)
        output_path = frames_dir / f"frame_{index:06d}.jpg"
        if cv2.imwrite(str(output_path), normalized_frame):
            saved_count += 1
    return saved_count


def run_ffmpeg_h264_export(frames_dir: Path, fps: int, output_path: Path) -> tuple[bool, str | None]:
    ffmpeg_path = find_ffmpeg()
    if ffmpeg_path is None:
        return False, "ffmpeg not found in PATH"

    command = [
        ffmpeg_path,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%06d.jpg"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        error_output = (completed.stderr or completed.stdout or "").strip()
        return False, error_output or f"ffmpeg exited with code {completed.returncode}"
    return True, None


def verify_video_readable(video_path: Path) -> dict[str, Any]:
    verification = {
        "exists": video_path.exists(),
        "file_size_bytes": int(video_path.stat().st_size) if video_path.exists() else 0,
        "readable_by_opencv": False,
        "frame_count": 0,
        "fps": 0.0,
        "width": 0,
        "height": 0,
    }
    if not video_path.exists():
        return verification

    capture = cv2.VideoCapture(str(video_path))
    try:
        if capture.isOpened():
            verification["readable_by_opencv"] = True
            verification["frame_count"] = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            verification["fps"] = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            verification["width"] = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            verification["height"] = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()
    return verification


def _open_video_writer(output_path: Path, frame_size: tuple[int, int], fps: int):
    codecs = ["mp4v", "XVID"]
    for codec_name in codecs:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec_name),
            float(fps),
            frame_size,
        )
        if writer.isOpened():
            return writer, codec_name
        writer.release()
    return None, None


def _open_mjpg_writer(output_path: Path, frame_size: tuple[int, int], fps: int):
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        float(fps),
        frame_size,
    )
    if writer.isOpened():
        return writer
    writer.release()
    return None


def _build_compiled_video(
    run_dir: Path,
    selected_items: list[dict[str, Any]],
    output_dir: Path,
    compiled_from: str,
    normal_fallback_used: bool,
) -> dict[str, Any]:
    print("[tender-demo] Starting Step 18 compiled review video")

    enabled = safe_bool_env("TENDER_DEMO_CREATE_COMPILED_REVIEW_VIDEO", DEFAULT_CREATE_COMPILED_REVIEW_VIDEO)
    compiled_fps = safe_int_env("TENDER_DEMO_COMPILED_VIDEO_FPS", DEFAULT_COMPILED_VIDEO_FPS)
    seconds_per_frame = safe_float_env("TENDER_DEMO_SECONDS_PER_FRAME", DEFAULT_SECONDS_PER_FRAME)
    seconds_per_title_card = safe_float_env("TENDER_DEMO_SECONDS_PER_TITLE_CARD", DEFAULT_SECONDS_PER_TITLE_CARD)
    panel_count = safe_int_env("TENDER_DEMO_STRIP_PANEL_COUNT", DEFAULT_STRIP_PANEL_COUNT)
    frame_width = safe_int_env("TENDER_DEMO_COMPILED_FRAME_WIDTH", DEFAULT_COMPILED_FRAME_WIDTH)
    frame_height = safe_int_env("TENDER_DEMO_COMPILED_FRAME_HEIGHT", DEFAULT_COMPILED_FRAME_HEIGHT)

    compiled_video_path = output_dir / "18_compiled_review_video.mp4"
    fallback_video_path = output_dir / "18_compiled_review_video_fallback.avi"
    compiled_manifest_path = run_dir / "18_compiled_review_video.json"
    compatibility_manifest_path = output_dir / "18_compiled_review_video.json"
    frames_dir = output_dir / "_compiled_frames"
    ffmpeg_path = find_ffmpeg()
    ffmpeg_available = ffmpeg_path is not None
    even_width, even_height = ensure_even_dimensions(frame_width, frame_height)

    manifest: dict[str, Any] = {
        "compiled_video_path": _relative_to_repo(compiled_video_path),
        "compiled_video_backend": None,
        "playback_recommended_file": None,
        "ffmpeg_available": ffmpeg_available,
        "ffmpeg_error": None,
        "compiled_from": compiled_from,
        "normal_fallback_used": normal_fallback_used,
        "fps": compiled_fps,
        "seconds_per_frame": seconds_per_frame,
        "seconds_per_title_card": seconds_per_title_card,
        "panel_count": panel_count,
        "total_events_used": 0,
        "total_frames_written": 0,
        "duration_seconds_estimated": 0.0,
        "video_verification": {},
        "included_clips": [],
    }

    if not enabled:
        compiled_manifest_text = json.dumps(manifest, indent=2)
        compiled_manifest_path.write_text(compiled_manifest_text, encoding="utf-8")
        compatibility_manifest_path.write_text(compiled_manifest_text, encoding="utf-8")
        return manifest

    for stale_path in [compiled_video_path, fallback_video_path]:
        if stale_path.exists():
            try:
                stale_path.unlink()
            except OSError:
                pass

    total_frames_written = 0
    total_events_used = 0
    compiled_frames: list[Any] = []

    for item in selected_items:
        strip_path_value = item.get("strip_path")
        included_record = {
            "clip_id": item.get("clip_id"),
            "source_category": item.get("final_category"),
            "time_range": item.get("time_range"),
            "event_label": item.get("event_label"),
            "strip_path": _relative_to_repo(strip_path_value),
            "panels_extracted": 0,
            "frames_written": 0,
            "included": False,
            "error": None,
        }

        strip_abs_path = _absolute_from_repo(strip_path_value)
        if strip_abs_path is None or not strip_abs_path.exists():
            included_record["error"] = f"Missing strip_path: {strip_path_value}"
            manifest["included_clips"].append(included_record)
            continue

        strip_image = cv2.imread(str(strip_abs_path))
        if strip_image is None:
            included_record["error"] = f"Failed to load strip image: {strip_abs_path}"
            manifest["included_clips"].append(included_record)
            continue

        panels = split_strip_into_panels(strip_image, panel_count)
        if not panels:
            included_record["error"] = f"No panels could be extracted from strip image: {strip_abs_path}"
            manifest["included_clips"].append(included_record)
            continue

        title_card = create_title_card(
            even_width,
            even_height,
            [
                "Tender Demo Review Clip",
                str(item.get("clip_id", "")),
                str(item.get("final_category", "")),
                str(item.get("time_range", "")),
                str(item.get("event_label", "")),
                str(item.get("risk_level", "")),
                str(item.get("confidence", "")),
                str(item.get("best_event_description", "")),
            ],
        )
        title_card_count = max(1, int(round(float(compiled_fps) * float(seconds_per_title_card))))
        frames_written = 0
        for _ in range(title_card_count):
            compiled_frames.append(title_card.copy())
            frames_written += 1

        panel_frame_count = max(1, int(round(float(compiled_fps) * float(seconds_per_frame))))
        for panel in panels:
            frame = resize_frame(panel, even_width, even_height)
            frame = draw_text_box(
                frame,
                [
                    f"{item.get('clip_id', '')} | {item.get('final_category', '')}",
                    str(item.get("time_range", "")),
                ],
                20,
                20,
            )
            frame = draw_text_box(
                frame,
                [
                    str(item.get("best_event_description", "")),
                    str(item.get("event_label", "")),
                ],
                20,
                max(20, even_height - 110),
            )
            for _ in range(panel_frame_count):
                compiled_frames.append(frame.copy())
                frames_written += 1

        included_record["panels_extracted"] = len(panels)
        included_record["frames_written"] = frames_written
        included_record["included"] = True
        manifest["included_clips"].append(included_record)
        total_frames_written += frames_written
        total_events_used += 1

    manifest["total_events_used"] = total_events_used
    manifest["total_frames_written"] = total_frames_written
    manifest["duration_seconds_estimated"] = round(
        total_frames_written / float(compiled_fps),
        3,
    ) if compiled_fps > 0 else 0.0

    if total_events_used == 0 or not compiled_frames:
        compiled_manifest_text = json.dumps(manifest, indent=2)
        compiled_manifest_path.write_text(compiled_manifest_text, encoding="utf-8")
        compatibility_manifest_path.write_text(compiled_manifest_text, encoding="utf-8")
        return manifest

    frames_saved = save_compiled_frames_to_temp(compiled_frames, frames_dir, even_width, even_height)
    ffmpeg_error: str | None = None
    backend_used = None
    playback_recommended_file: str | None = None
    primary_video_verification = verify_video_readable(compiled_video_path)
    recommended_video_verification: dict[str, Any] = {}
    playable = False

    if ffmpeg_available and frames_saved > 0:
        success, ffmpeg_error = run_ffmpeg_h264_export(frames_dir, compiled_fps, compiled_video_path)
        manifest["ffmpeg_error"] = ffmpeg_error
        if success:
            backend_used = "ffmpeg_h264"
            playback_recommended_file = _relative_to_repo(compiled_video_path)
            primary_video_verification = verify_video_readable(compiled_video_path)
            recommended_video_verification = dict(primary_video_verification)
            playable = bool(primary_video_verification.get("readable_by_opencv")) and int(primary_video_verification.get("frame_count", 0)) > 1

    if backend_used is None:
        avi_writer = _open_mjpg_writer(
            output_path=fallback_video_path,
            frame_size=(even_width, even_height),
            fps=compiled_fps,
        )
        if avi_writer is not None:
            try:
                for frame in compiled_frames:
                    avi_writer.write(frame)
            finally:
                avi_writer.release()
        backend_used = "opencv_mjpg_fallback"
        playback_recommended_file = _relative_to_repo(fallback_video_path)
        primary_video_verification = verify_video_readable(compiled_video_path)
        recommended_video_verification = verify_video_readable(fallback_video_path)
        playable = False

    manifest["compiled_video_backend"] = backend_used
    manifest["playback_recommended_file"] = playback_recommended_file
    manifest["video_verification"] = primary_video_verification
    manifest["playback_recommended_verification"] = recommended_video_verification
    manifest["playable"] = playable

    compiled_manifest_text = json.dumps(manifest, indent=2)
    compiled_manifest_path.write_text(compiled_manifest_text, encoding="utf-8")
    compatibility_manifest_path.write_text(compiled_manifest_text, encoding="utf-8")

    print(f"[tender-demo] Total selected events considered: {len(selected_items)}")
    print(f"[tender-demo] Events included: {total_events_used}")
    print(f"[tender-demo] Total frames written: {total_frames_written}")
    print(f"[tender-demo] Estimated duration: {manifest['duration_seconds_estimated']}")
    if ffmpeg_error:
        print(f"[tender-demo] FFmpeg error: {ffmpeg_error}")
    print(f"[tender-demo] Backend used: {backend_used}")
    print(f"[tender-demo] Output video path: {playback_recommended_file}")
    verification_for_print = recommended_video_verification if recommended_video_verification else primary_video_verification
    print(f"[tender-demo] File size: {verification_for_print.get('file_size_bytes', 0)}")
    print(f"[tender-demo] Frame count: {verification_for_print.get('frame_count', 0)}")
    print(f"[tender-demo] Readable: {'yes' if verification_for_print.get('readable_by_opencv') else 'no'}")
    print(f"[tender-demo] Compiled video path: {compiled_video_path}")
    print(f"[tender-demo] Compiled manifest path: {compiled_manifest_path}")
    return manifest


def _category_export_enabled(category: str, settings: dict[str, Any]) -> bool:
    if category == "priority_suspicious_event":
        return bool(settings["export_priority_clips"])
    if category == "possible_review_clip":
        return bool(settings["export_review_clips"])
    if category == "normal_activity":
        return bool(settings["export_normal_clips"])
    return False


def _output_prefix(category: str) -> str:
    if category == "priority_suspicious_event":
        return "priority"
    if category == "possible_review_clip":
        return "review"
    if category == "normal_activity":
        return "normal"
    return "clip"


def _selected_clips(summary_data: dict[str, Any], settings: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    ordered_groups = [
        ("priority_suspicious_event", summary_data.get("priority_suspicious_events", [])),
        ("possible_review_clip", summary_data.get("possible_review_clips", [])),
        ("normal_activity", summary_data.get("normal_activity_clips", [])),
    ]
    for category, items in ordered_groups:
        if not _category_export_enabled(category, settings):
            continue
        if not isinstance(items, list):
            continue
        sorted_items = sorted(items, key=lambda item: float(item.get("start_time", 0.0) or 0.0))
        selected.extend(sorted_items)
    return selected


def _compiled_video_source_items(summary_data: dict[str, Any], settings: dict[str, Any]) -> tuple[list[dict[str, Any]], str, bool]:
    priority_items = summary_data.get("priority_suspicious_events", [])
    review_items = summary_data.get("possible_review_clips", [])
    normal_items = summary_data.get("normal_activity_clips", [])
    event_timeline = summary_data.get("event_timeline", [])

    if not isinstance(priority_items, list):
        priority_items = []
    if not isinstance(review_items, list):
        review_items = []
    if not isinstance(normal_items, list):
        normal_items = []
    if not isinstance(event_timeline, list):
        event_timeline = []

    if priority_items or review_items:
        compiled_items: list[dict[str, Any]] = []
        compiled_items.extend(priority_items)
        compiled_items.extend(review_items)
        if bool(settings.get("export_normal_clips")):
            compiled_items.extend(normal_items)
        return sorted(compiled_items, key=lambda item: float(item.get("start_time", 0.0) or 0.0)), "priority_and_review", False

    if bool(settings.get("compile_normal_if_no_events", DEFAULT_COMPILE_NORMAL_IF_NO_EVENTS)):
        if normal_items:
            return sorted(normal_items, key=lambda item: float(item.get("start_time", 0.0) or 0.0)), "normal_activity_fallback", True
        if event_timeline:
            return sorted(event_timeline, key=lambda item: float(item.get("start_time", 0.0) or 0.0)), "normal_activity_fallback", True

    return [], "none", False


def export_event_clips(run_dir: Path) -> dict[str, Any]:
    print("[tender-demo] Starting Step 18: export event clips")

    summary_data = _load_step17_summary(run_dir)
    video_info = _load_required_json(run_dir / "01_video_info.json")
    topk_vlm_inputs_by_clip_id = _load_optional_topk_vlm_inputs(run_dir)

    export_settings = {
        "export_priority_clips": safe_bool_env("TENDER_DEMO_EXPORT_PRIORITY_CLIPS", DEFAULT_EXPORT_PRIORITY_CLIPS),
        "export_review_clips": safe_bool_env("TENDER_DEMO_EXPORT_REVIEW_CLIPS", DEFAULT_EXPORT_REVIEW_CLIPS),
        "export_normal_clips": safe_bool_env("TENDER_DEMO_EXPORT_NORMAL_CLIPS", DEFAULT_EXPORT_NORMAL_CLIPS),
        "compile_normal_if_no_events": safe_bool_env("TENDER_DEMO_COMPILE_NORMAL_IF_NO_EVENTS", DEFAULT_COMPILE_NORMAL_IF_NO_EVENTS),
        "fps": safe_int_env("TENDER_DEMO_EXPORT_FPS", DEFAULT_EXPORT_FPS),
        "format": str(os.environ.get("TENDER_DEMO_EXPORT_FORMAT", DEFAULT_EXPORT_FORMAT)).strip().lower() or DEFAULT_EXPORT_FORMAT,
    }

    output_dir = run_dir / "18_exported_clips"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "18_exported_clips.json"

    selected_items = _selected_clips(summary_data, export_settings)
    print(f"[tender-demo] Total clips selected: {len(selected_items)}")
    compiled_source_items, compiled_from, normal_fallback_used = _compiled_video_source_items(summary_data, export_settings)
    print(f"[tender-demo] Compiled review source clips selected: {len(compiled_source_items)}")

    exported_clips: list[dict[str, Any]] = []
    total_failed = 0
    category_counts = {
        "priority_suspicious_event": 0,
        "possible_review_clip": 0,
        "normal_activity": 0,
    }

    for index, item in enumerate(selected_items, start=1):
        category = str(item.get("final_category", ""))
        category_counts[category] = category_counts.get(category, 0) + 1
        prefix = _output_prefix(category)
        clip_id = safe_filename(str(item.get("clip_id", "clip")))
        output_filename = f"{prefix}_{category_counts[category]:03d}_{clip_id}.{export_settings['format']}"
        output_path = output_dir / output_filename
        topk_vlm_item = topk_vlm_inputs_by_clip_id.get(str(item.get("clip_id", "")).strip(), {})
        strip_path_value = topk_vlm_item.get("strip_path", item.get("strip_path"))
        source_frame_indices = topk_vlm_item.get("source_frame_indices", {})
        source_frame_times = topk_vlm_item.get("source_frame_times", {})

        export_record = {
            "export_id": f"export_{index:06d}",
            "clip_id": item.get("clip_id"),
            "source_category": category,
            "event_label": item.get("event_label"),
            "time_range": item.get("time_range", f"{format_seconds(item.get('start_time'))} - {format_seconds(item.get('end_time'))}"),
            "strip_path": _relative_to_repo(strip_path_value),
            "output_path": _relative_to_repo(output_path),
            "source_topk_vlm_input_id": topk_vlm_item.get("topk_vlm_input_id"),
            "source_frame_indices": source_frame_indices if isinstance(source_frame_indices, dict) else {},
            "source_frame_times": source_frame_times if isinstance(source_frame_times, dict) else {},
            "frames_used": 0,
            "fps": export_settings["fps"],
            "duration_seconds": 0.0,
            "export_success": False,
            "error": None,
        }

        strip_abs_path = _absolute_from_repo(strip_path_value)
        if strip_abs_path is None or not strip_abs_path.exists():
            export_record["error"] = f"Missing strip_path: {strip_path_value}"
            exported_clips.append(export_record)
            total_failed += 1
            continue

        strip_image = cv2.imread(str(strip_abs_path))
        if strip_image is None:
            export_record["error"] = f"Failed to load strip image: {strip_abs_path}"
            exported_clips.append(export_record)
            total_failed += 1
            continue

        frames = _split_strip_into_frames(strip_image)
        if not frames:
            export_record["error"] = f"No frames could be reconstructed from strip image: {strip_abs_path}"
            exported_clips.append(export_record)
            total_failed += 1
            continue

        target_height, target_width = frames[0].shape[:2]
        normalized_frames = []
        for frame in frames:
            if frame.shape[0] != target_height or frame.shape[1] != target_width:
                frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            normalized_frames.append(frame)

        writer, codec_used = _open_video_writer(
            output_path=output_path,
            frame_size=(target_width, target_height),
            fps=export_settings["fps"],
        )
        if writer is None:
            export_record["error"] = f"Failed to initialize video writer for: {output_path}"
            exported_clips.append(export_record)
            total_failed += 1
            continue

        try:
            for frame in normalized_frames:
                writer.write(frame)
        finally:
            writer.release()

        if not output_path.exists() or output_path.stat().st_size == 0:
            export_record["error"] = f"Video export failed or produced empty file: {output_path}"
            exported_clips.append(export_record)
            total_failed += 1
            continue

        export_record["frames_used"] = len(normalized_frames)
        export_record["duration_seconds"] = round(len(normalized_frames) / float(export_settings["fps"]), 3)
        export_record["export_success"] = True
        export_record["codec_used"] = codec_used
        exported_clips.append(export_record)

    manifest = {
        "video_name": video_info.get("video_name"),
        "video_path": _relative_to_repo(video_info.get("video_path")) or video_info.get("video_path"),
        "export_folder": _relative_to_repo(output_dir),
        "export_settings": export_settings,
        "total_events_considered": len(selected_items),
        "total_clips_exported": sum(1 for item in exported_clips if item.get("export_success") is True),
        "total_failed": total_failed,
        "exported_clips": exported_clips,
    }
    compiled_review_video = _build_compiled_video(
        run_dir=run_dir,
        selected_items=compiled_source_items,
        output_dir=output_dir,
        compiled_from=compiled_from,
        normal_fallback_used=normal_fallback_used,
    )
    manifest["compiled_review_video"] = {
        "enabled": safe_bool_env("TENDER_DEMO_CREATE_COMPILED_REVIEW_VIDEO", DEFAULT_CREATE_COMPILED_REVIEW_VIDEO),
        "output_path": compiled_review_video.get("compiled_video_path"),
        "manifest_path": _relative_to_repo(run_dir / "18_compiled_review_video.json"),
        "backend": compiled_review_video.get("compiled_video_backend"),
        "playable": compiled_review_video.get("playable", False),
        "compiled_from": compiled_review_video.get("compiled_from"),
        "normal_fallback_used": compiled_review_video.get("normal_fallback_used", False),
        "playback_recommended_file": compiled_review_video.get("playback_recommended_file"),
        "fallback_available": compiled_review_video.get("compiled_video_backend") == "opencv_mjpg_fallback",
        "total_events_used": compiled_review_video.get("total_events_used", 0),
        "total_frames_written": compiled_review_video.get("total_frames_written", 0),
        "video_verification": compiled_review_video.get("video_verification", {}),
        "playback_recommended_verification": compiled_review_video.get("playback_recommended_verification", {}),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[tender-demo] Total exported: {manifest['total_clips_exported']}")
    print(f"[tender-demo] Total failed: {manifest['total_failed']}")
    print(f"[tender-demo] Output folder path: {output_dir}")
    print(f"[tender-demo] Manifest path: {manifest_path}")
    return manifest
