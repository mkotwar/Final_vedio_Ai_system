"""OpenCV-backed local video frame source for Step 3 sequential validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .schemas import FramePacket
from .sources import build_processing_frame_indices
from .validation import validate_non_empty_string, validate_positive_float, validate_positive_int


class OpenCvVideoSource:
    """Sequentially decode a local video and emit selected FramePacket objects."""

    def __init__(
        self,
        source_path: str | Path,
        source_id: str | None = None,
        target_processing_fps: float | None = None,
        use_source_fps: bool = False,
        max_processed_frames: int | None = None,
        start_sec: float | None = None,
        end_sec: float | None = None,
    ) -> None:
        self.source_path = Path(source_path).expanduser()
        if not self.source_path.exists():
            raise FileNotFoundError(f"Video source path does not exist: {self.source_path}")
        self._source_id = validate_non_empty_string(source_id or self.source_path.stem, "source_id")
        if target_processing_fps is not None:
            validate_positive_float(target_processing_fps, "target_processing_fps")
        if max_processed_frames is not None:
            validate_positive_int(max_processed_frames, "max_processed_frames")
        if start_sec is not None and start_sec < 0.0:
            raise ValueError("start_sec must be non-negative.")
        if end_sec is not None:
            validate_positive_float(end_sec, "end_sec")
        if start_sec is not None and end_sec is not None and end_sec <= start_sec:
            raise ValueError("end_sec must be greater than start_sec.")
        self.target_processing_fps = None if use_source_fps else target_processing_fps
        self.use_source_fps = bool(use_source_fps)
        self.max_processed_frames = max_processed_frames
        self.start_sec = start_sec
        self.end_sec = end_sec
        self._capture: Any | None = None
        self._opened = False
        self._closed = False
        self._current_source_index = 0
        self._selected_cursor = 0
        self._emitted_count = 0
        self._source_fps = 0.0
        self._frame_width = 0
        self._frame_height = 0
        self.total_frames: int | None = None
        self.duration_sec: float | None = None
        self.selected_frame_indices: tuple[int, ...] = ()

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def source_fps(self) -> float:
        return self._source_fps

    @property
    def frame_width(self) -> int:
        return self._frame_width

    @property
    def frame_height(self) -> int:
        return self._frame_height

    @property
    def opened(self) -> bool:
        return self._opened

    @property
    def closed(self) -> bool:
        return self._closed

    def open(self) -> None:
        if self._opened and not self._closed:
            return
        if self._closed:
            raise RuntimeError("Cannot reopen a closed OpenCvVideoSource; call reset() first.")
        import cv2

        capture = cv2.VideoCapture(str(self.source_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Failed to open video source: {self.source_path}")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count_raw = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        validate_positive_float(source_fps, "source_fps")
        validate_positive_int(frame_width, "frame_width")
        validate_positive_int(frame_height, "frame_height")
        self._capture = capture
        self._source_fps = source_fps
        self._frame_width = frame_width
        self._frame_height = frame_height
        self.total_frames = frame_count_raw if frame_count_raw > 0 else None
        self.duration_sec = (frame_count_raw / source_fps) if frame_count_raw > 0 else None
        if self.total_frames is None:
            self.selected_frame_indices = ()
        else:
            selected = build_processing_frame_indices(self.total_frames, self.source_fps, self.target_processing_fps)
            if self.start_sec is not None:
                start_frame = int(self.start_sec * self.source_fps)
                selected = tuple(index for index in selected if index >= start_frame)
            if self.end_sec is not None:
                end_frame = int(self.end_sec * self.source_fps)
                selected = tuple(index for index in selected if index < end_frame)
            if self.max_processed_frames is not None:
                selected = selected[: self.max_processed_frames]
            self.selected_frame_indices = selected
        self._opened = True
        self._closed = False

    def read(self) -> FramePacket | None:
        if not self._opened:
            raise RuntimeError("OpenCvVideoSource must be opened before read().")
        if self._closed:
            raise RuntimeError("Cannot read from a closed OpenCvVideoSource.")
        if self.max_processed_frames is not None and self._emitted_count >= self.max_processed_frames:
            return None
        if self.total_frames is not None and self._selected_cursor >= len(self.selected_frame_indices):
            return None
        if self._capture is None:
            raise RuntimeError("Video capture is not available.")

        selected_set = set(self.selected_frame_indices)
        while True:
            ok, frame = self._capture.read()
            if not ok:
                return None
            source_index = self._current_source_index
            self._current_source_index += 1
            should_emit = source_index in selected_set if self.total_frames is not None else True
            if not should_emit:
                continue
            while self._selected_cursor < len(self.selected_frame_indices) and self.selected_frame_indices[self._selected_cursor] < source_index:
                self._selected_cursor += 1
            if self._selected_cursor < len(self.selected_frame_indices) and self.selected_frame_indices[self._selected_cursor] == source_index:
                self._selected_cursor += 1
            self._emitted_count += 1
            return FramePacket(
                source_id=self.source_id,
                frame_index=source_index,
                timestamp_sec=source_index / self.source_fps,
                source_fps=self.source_fps,
                frame_width=self.frame_width,
                frame_height=self.frame_height,
                frame=frame,
            )

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._closed = True

    def reset(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._opened = False
        self._closed = False
        self._current_source_index = 0
        self._selected_cursor = 0
        self._emitted_count = 0

    def metadata_report(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "source_id": self.source_id,
            "source_fps": self.source_fps,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "total_source_frames": self.total_frames,
            "duration_sec": self.duration_sec,
            "target_processing_fps": self.target_processing_fps,
            "expected_selected_frames": len(self.selected_frame_indices) if self.total_frames is not None else None,
            "max_processed_frames": self.max_processed_frames,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "selected_frame_indices": list(self.selected_frame_indices),
        }
