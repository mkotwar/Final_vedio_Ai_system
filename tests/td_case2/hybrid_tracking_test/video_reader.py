from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import cv2


@dataclass(frozen=True)
class VideoMetadata:
    video_path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float


def read_video_metadata(video_path: Path) -> VideoMetadata:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    if fps <= 0.0:
        raise ValueError(f"Video FPS is invalid: {fps}")
    return VideoMetadata(
        video_path=video_path,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_seconds=(frame_count / fps) if fps > 0 else 0.0,
    )


def iter_processed_frames(
    *,
    video_path: Path,
    processing_fps: float,
) -> Generator[dict[str, object], None, None]:
    if processing_fps <= 0.0:
        raise ValueError("processing_fps must be greater than 0.")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 0.0:
        capture.release()
        raise ValueError(f"Source FPS is invalid for video: {video_path}")

    next_processing_timestamp = 0.0
    processing_interval = 1.0 / processing_fps
    processed_frame_index = 0
    previous_processed_timestamp: float | None = None
    source_frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        source_timestamp_seconds = source_frame_index / source_fps
        if source_timestamp_seconds + 1e-9 < next_processing_timestamp:
            source_frame_index += 1
            continue
        yield {
            "frame": frame,
            "source_frame_index": source_frame_index,
            "processed_frame_index": processed_frame_index,
            "source_timestamp_seconds": source_timestamp_seconds,
            "time_delta_from_previous_processed_frame": (
                0.0 if previous_processed_timestamp is None else source_timestamp_seconds - previous_processed_timestamp
            ),
        }
        processed_frame_index += 1
        previous_processed_timestamp = source_timestamp_seconds
        next_processing_timestamp += processing_interval
        source_frame_index += 1

    capture.release()

