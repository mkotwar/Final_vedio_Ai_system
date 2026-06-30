from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2


ENV_FINAL_DEMO_INPUT_VIDEO = "FINAL_DEMO_INPUT_VIDEO"


def get_final_demo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_final_demo_directories() -> dict[str, Path]:
    final_demo_root = get_final_demo_root()
    directories = {
        "debug_runs": final_demo_root / "debug_runs",
        "ui_uploads": final_demo_root / "ui_uploads",
        "video_imports": final_demo_root / "video_imports",
        "exports": final_demo_root / "exports",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def current_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def build_run_suffix() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_input_video_path() -> Path:
    raw_value = os.environ.get(ENV_FINAL_DEMO_INPUT_VIDEO)
    if not raw_value:
        raise ValueError(
            f"Environment variable {ENV_FINAL_DEMO_INPUT_VIDEO} is not set. "
            "Set it before running the final demo pipeline."
        )

    candidate_path = Path(raw_value).expanduser()
    resolved_path = candidate_path if candidate_path.is_absolute() else candidate_path.resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(f"Input video path does not exist: {resolved_path}")
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Input video path is not a file: {resolved_path}")

    return resolved_path


def create_run_directory(video_path: Path) -> Path:
    directories = ensure_final_demo_directories()
    run_dir = directories["debug_runs"] / f"{video_path.stem}_{build_run_suffix()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def read_video_metadata(video_path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not open input video: {video_path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        try:
            read_backend = capture.getBackendName()
        except Exception:
            read_backend = "opencv"
    finally:
        capture.release()

    duration_seconds = round(frame_count / fps, 3) if fps > 0 else 0.0
    created_at = current_timestamp()

    return {
        "video_path": str(video_path),
        "file_name": video_path.name,
        "stem": video_path.stem,
        "suffix": video_path.suffix,
        "duration_seconds": duration_seconds,
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "read_backend": read_backend,
        "created_at": created_at,
    }


def write_json(output_path: Path, payload: dict[str, Any]) -> Path:
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path
