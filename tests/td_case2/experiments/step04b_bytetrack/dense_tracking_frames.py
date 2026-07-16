from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2


def build_dense_tracking_frames(
    *,
    video_path: Path,
    output_dir: Path,
    tracking_fps: float,
) -> dict[str, Any]:
    """Decode a denser tracking-only frame branch from the original video."""

    frames_dir = output_dir / "experimental_dense_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video for dense tracking frames: {video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if source_fps <= 0.0:
        source_fps = tracking_fps

    selected_frames: list[dict[str, Any]] = []
    written_frame_indices: set[int] = set()
    next_sample_time = 0.0
    source_frame_idx = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        timestamp_seconds = source_frame_idx / source_fps if source_fps > 0 else 0.0
        should_keep = (
            not selected_frames
            or timestamp_seconds + 1e-9 >= next_sample_time
        )
        if should_keep and source_frame_idx not in written_frame_indices:
            frame_id = f"tracking_frame_{source_frame_idx:06d}"
            image_name = f"{frame_id}.jpg"
            image_path = frames_dir / image_name
            if not cv2.imwrite(str(image_path), frame):
                raise RuntimeError(f"Failed to write dense tracking frame: {image_path}")
            selected_frames.append(
                {
                    "frame_id": frame_id,
                    "frame_idx": source_frame_idx,
                    "timestamp_seconds": round(timestamp_seconds, 6),
                    "timestamp_text": f"{timestamp_seconds:0.3f}s",
                    "image_path": str(Path("experimental_dense_frames") / image_name).replace("\\", "/"),
                    "video_fps": round(source_fps, 6),
                    "tracking_sampling_fps": round(tracking_fps, 6),
                    "source": "experimental_dense_tracking_branch",
                }
            )
            written_frame_indices.add(source_frame_idx)
            next_sample_time = timestamp_seconds + (1.0 / tracking_fps)
        source_frame_idx += 1

    capture.release()
    return {
        "status": "success",
        "video_path": str(video_path),
        "video_fps": round(source_fps, 6),
        "tracking_sampling_fps": round(tracking_fps, 6),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "selected_frame_count": len(selected_frames),
        "selected_frames": selected_frames,
    }

