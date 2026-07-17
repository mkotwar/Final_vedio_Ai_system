from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2


@dataclass(frozen=True)
class VideoInfo:
    video_path: Path
    source_fps: float
    source_frame_count: int
    width: int
    height: int
    duration_seconds: float


@dataclass(frozen=True)
class FrameStreamRecord:
    source_frame_index: int
    processed_frame_index: int
    timestamp_seconds: float
    source_fps: float
    processing_fps: float
    frame_time_delta: float
    frame_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_frame_index": self.source_frame_index,
            "processed_frame_index": self.processed_frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 6),
            "source_fps": round(self.source_fps, 6),
            "processing_fps": round(self.processing_fps, 6),
            "frame_time_delta": round(self.frame_time_delta, 6),
            "frame_path": self.frame_path,
        }


def read_video_info(video_path: Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    duration_seconds = (source_frame_count / source_fps) if source_fps > 0 else 0.0
    return VideoInfo(
        video_path=video_path,
        source_fps=source_fps,
        source_frame_count=source_frame_count,
        width=width,
        height=height,
        duration_seconds=duration_seconds,
    )


def _save_debug_frame(frame: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Failed to write debug frame: {output_path}")


def stream_processed_frames(
    *,
    video_path: Path,
    processing_fps: float,
    debug_frames_dir: Path | None = None,
) -> tuple[VideoInfo, list[dict[str, Any]], dict[str, Any], Iterator[tuple[FrameStreamRecord, Any]]]:
    video_info = read_video_info(video_path)

    def _iterator() -> Iterator[tuple[FrameStreamRecord, Any]]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        next_target_timestamp = 0.0
        processed_frame_index = 0
        last_processed_timestamp: float | None = None
        source_frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp_seconds = source_frame_index / video_info.source_fps if video_info.source_fps > 0 else 0.0
                if timestamp_seconds + 1e-9 < next_target_timestamp:
                    source_frame_index += 1
                    continue
                frame_path: str | None = None
                if debug_frames_dir is not None:
                    image_name = f"processed_{processed_frame_index:06d}.jpg"
                    output_path = debug_frames_dir / image_name
                    _save_debug_frame(frame, output_path)
                    frame_path = str(output_path)
                delta = 0.0 if last_processed_timestamp is None else max(0.0, timestamp_seconds - last_processed_timestamp)
                record = FrameStreamRecord(
                    source_frame_index=source_frame_index,
                    processed_frame_index=processed_frame_index,
                    timestamp_seconds=timestamp_seconds,
                    source_fps=video_info.source_fps,
                    processing_fps=processing_fps,
                    frame_time_delta=delta,
                    frame_path=frame_path,
                )
                yield record, frame.copy()
                processed_frame_index += 1
                last_processed_timestamp = timestamp_seconds
                next_target_timestamp = processed_frame_index / processing_fps
                source_frame_index += 1
        finally:
            capture.release()

    records: list[dict[str, Any]] = []
    metrics = {
        "status": "success",
        "source_frame_count": video_info.source_frame_count,
        "source_fps": round(video_info.source_fps, 6),
        "duration_seconds": round(video_info.duration_seconds, 6),
        "target_processing_fps": round(processing_fps, 6),
    }
    return video_info, records, metrics, _iterator()

