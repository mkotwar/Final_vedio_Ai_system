from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .camera_config import CameraConfig
from .frame_packet import FramePacket

LOGGER = logging.getLogger(__name__)


class CameraSourceError(RuntimeError):
    """Raised when a camera source cannot be opened or read safely."""


@dataclass(frozen=True, slots=True)
class CameraSourceMetadata:
    camera_code: str
    camera_name: str
    source_path: Path
    source_fps: float
    source_frame_count: int | None
    frame_width: int
    frame_height: int
    duration_seconds: float | None


class CameraSource:
    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._capture: Any | None = None
        self._source_fps = 0.0
        self._source_frame_count: int | None = None
        self._frame_width = 0
        self._frame_height = 0
        self._opened = False
        self._end_of_stream = False
        self._last_frame_number = -1

    def __enter__(self) -> "CameraSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        if self._opened:
            return
        if not self.config.source_path.exists():
            raise FileNotFoundError(f"Camera '{self.config.camera_code}' source file does not exist: {self.config.source_path}")
        import cv2

        capture = cv2.VideoCapture(str(self.config.source_path))
        if not capture.isOpened():
            capture.release()
            raise CameraSourceError(f"Camera '{self.config.camera_code}' video could not be opened: {self.config.source_path}")

        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if source_fps <= 0.0:
            capture.release()
            raise CameraSourceError(f"Camera '{self.config.camera_code}' has invalid FPS: {source_fps}")
        if frame_count <= 0:
            capture.release()
            raise CameraSourceError(f"Camera '{self.config.camera_code}' appears to be empty: {self.config.source_path}")
        if frame_width <= 0 or frame_height <= 0:
            capture.release()
            raise CameraSourceError(f"Camera '{self.config.camera_code}' has invalid frame dimensions.")

        self._capture = capture
        self._source_fps = source_fps
        self._source_frame_count = frame_count
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._opened = True
        self._end_of_stream = False
        self._last_frame_number = -1
        LOGGER.info("Opened camera source camera_code=%s source_path=%s fps=%.3f", self.config.camera_code, self.config.source_path, self._source_fps)

    def read_next(self) -> FramePacket | None:
        if not self._opened or self._capture is None:
            raise CameraSourceError(f"Camera '{self.config.camera_code}' must be opened before reading.")
        if self._end_of_stream:
            return None

        ok, frame = self._capture.read()
        if not ok:
            self._end_of_stream = True
            LOGGER.info("Reached end of stream camera_code=%s source_path=%s last_frame_number=%s", self.config.camera_code, self.config.source_path, self._last_frame_number)
            return None

        frame_number = self._last_frame_number + 1
        self._last_frame_number = frame_number
        video_time_seconds = frame_number / self._source_fps
        camera_timestamp = None
        if self.config.start_time is not None:
            camera_timestamp = self.config.start_time + __import__("datetime").timedelta(seconds=video_time_seconds)
        LOGGER.debug("Read frame camera_code=%s source_path=%s frame_number=%s", self.config.camera_code, self.config.source_path, frame_number)
        return FramePacket(
            camera_code=self.config.camera_code,
            camera_name=self.config.camera_name,
            source_path=self.config.source_path,
            frame_number=frame_number,
            source_fps=self._source_fps,
            source_frame_count=self._source_frame_count,
            video_time_seconds=video_time_seconds,
            camera_timestamp=camera_timestamp,
            frame=frame,
        )

    def is_open(self) -> bool:
        return self._opened and self._capture is not None

    def metadata(self) -> CameraSourceMetadata:
        if not self._opened:
            raise CameraSourceError(f"Camera '{self.config.camera_code}' must be opened before metadata().")
        duration_seconds = None
        if self._source_frame_count is not None:
            duration_seconds = self._source_frame_count / self._source_fps
        return CameraSourceMetadata(
            camera_code=self.config.camera_code,
            camera_name=self.config.camera_name,
            source_path=self.config.source_path,
            source_fps=self._source_fps,
            source_frame_count=self._source_frame_count,
            frame_width=self._frame_width,
            frame_height=self._frame_height,
            duration_seconds=duration_seconds,
        )

    def reset(self) -> None:
        self.close()
        self.open()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        if self._opened:
            LOGGER.info("Closed camera source camera_code=%s source_path=%s", self.config.camera_code, self.config.source_path)
        self._opened = False
        self._end_of_stream = True
