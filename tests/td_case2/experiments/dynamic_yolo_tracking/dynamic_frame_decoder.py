from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    duration_seconds = (frame_count / fps) if fps > 0 else 0.0
    return VideoMetadata(
        video_path=video_path,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_seconds=duration_seconds,
    )


def frame_interval_for_fps(video_fps: float, target_fps: float) -> int:
    if video_fps <= 0 or target_fps <= 0:
        return 1
    return max(1, int(round(video_fps / target_fps)))


def next_frame_index(*, current_frame_idx: int, video_fps: float, target_fps: float, total_frames: int) -> int:
    step = frame_interval_for_fps(video_fps, target_fps)
    return min(total_frames - 1, current_frame_idx + step)


def validate_chronological_frame_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    duplicate_indexes = len({int(item["frame_idx"]) for item in records}) != len(records)
    out_of_order = any(
        int(records[index]["frame_idx"]) <= int(records[index - 1]["frame_idx"])
        for index in range(1, len(records))
    )
    return {
        "chronological": not out_of_order,
        "duplicate_frame_indexes": duplicate_indexes,
    }

