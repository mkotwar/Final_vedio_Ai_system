from __future__ import annotations

import logging
from collections.abc import Iterator

from .camera_config import CameraConfig
from .camera_source import CameraSource
from .frame_packet import FramePacket

LOGGER = logging.getLogger(__name__)


class MultiCameraReaderError(RuntimeError):
    """Raised when the multi-camera reader configuration is invalid."""


class MultiCameraReader(Iterator[FramePacket]):
    def __init__(self, camera_configs: list[CameraConfig], *, mode: str = "round_robin", max_frames_per_camera: int | None = None) -> None:
        if mode not in {"sequential", "round_robin"}:
            raise MultiCameraReaderError(f"Unsupported reader mode: {mode}")
        if not camera_configs:
            raise MultiCameraReaderError("At least one enabled camera config is required.")
        self.camera_configs = list(camera_configs)
        self.mode = mode
        self.max_frames_per_camera = max_frames_per_camera
        self.sources = [CameraSource(config) for config in self.camera_configs]
        self._frames_read_by_camera: dict[str, int] = {config.camera_code: 0 for config in self.camera_configs}
        self._exhausted: set[str] = set()
        self._opened = False
        self._sequential_index = 0
        self._round_robin_index = 0

    def __iter__(self) -> "MultiCameraReader":
        if not self._opened:
            self.open()
        return self

    def __next__(self) -> FramePacket:
        packet = self._read_next_packet()
        if packet is None:
            self.close()
            raise StopIteration
        return packet

    def open(self) -> None:
        if self._opened:
            return
        for source in self.sources:
            source.open()
        self._opened = True

    def close(self) -> None:
        for source in self.sources:
            source.close()
        self._opened = False

    def _read_next_packet(self) -> FramePacket | None:
        if self.mode == "sequential":
            return self._read_next_sequential()
        return self._read_next_round_robin()

    def _can_read_more(self, camera_code: str) -> bool:
        if self.max_frames_per_camera is None:
            return True
        return self._frames_read_by_camera[camera_code] < self.max_frames_per_camera

    def _mark_exhausted(self, camera_code: str) -> None:
        self._exhausted.add(camera_code)

    def _read_from_source(self, source: CameraSource) -> FramePacket | None:
        if source.config.camera_code in self._exhausted:
            return None
        if not self._can_read_more(source.config.camera_code):
            self._mark_exhausted(source.config.camera_code)
            return None
        packet = source.read_next()
        if packet is None:
            self._mark_exhausted(source.config.camera_code)
            return None
        self._frames_read_by_camera[source.config.camera_code] += 1
        return packet

    def _read_next_sequential(self) -> FramePacket | None:
        while self._sequential_index < len(self.sources):
            source = self.sources[self._sequential_index]
            packet = self._read_from_source(source)
            if packet is not None:
                return packet
            self._sequential_index += 1
        return None

    def _read_next_round_robin(self) -> FramePacket | None:
        if len(self._exhausted) == len(self.sources):
            return None
        attempts = 0
        while attempts < len(self.sources):
            source = self.sources[self._round_robin_index]
            self._round_robin_index = (self._round_robin_index + 1) % len(self.sources)
            attempts += 1
            packet = self._read_from_source(source)
            if packet is not None:
                return packet
        if len(self._exhausted) == len(self.sources):
            LOGGER.info("All camera sources reached end-of-stream or max frame limit.")
            return None
        return self._read_next_round_robin()
