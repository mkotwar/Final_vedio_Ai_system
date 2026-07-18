"""Deterministic synthetic frame sources for Step 2 contract validation."""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Callable

from .schemas import FramePacket
from .validation import validate_non_empty_string, validate_non_negative_int, validate_positive_float, validate_positive_int


def _rate(value: float) -> Fraction:
    return Fraction(str(float(value)))


def build_processing_frame_indices(
    total_frames: int,
    source_fps: float,
    target_processing_fps: float | None,
) -> tuple[int, ...]:
    """Build monotonic selected source-frame indices using timestamp scheduling.

    The first frame is included for non-empty sources. The final source frame is
    not forced; selected frames follow the target cadence on the source timeline.
    A target FPS above source FPS is rejected rather than silently upsampling.
    """

    frame_count = validate_non_negative_int(total_frames, "total_frames")
    source_rate = _rate(validate_positive_float(source_fps, "source_fps"))
    if frame_count == 0:
        return ()
    if target_processing_fps is None:
        return tuple(range(frame_count))
    target_rate = _rate(validate_positive_float(target_processing_fps, "target_processing_fps"))
    if target_rate > source_rate:
        raise ValueError("target_processing_fps must not exceed source_fps; Step 2 does not upsample synthetic sources.")
    if target_rate == source_rate:
        return tuple(range(frame_count))

    selected: list[int] = []
    selected_count = 0
    for frame_index in range(frame_count):
        frame_time = Fraction(frame_index, 1) / source_rate
        next_time = Fraction(selected_count, 1) / target_rate
        if frame_time >= next_time:
            selected.append(frame_index)
            selected_count += 1
    return tuple(selected)


class SyntheticFrameSource:
    """Deterministic source that emits selected synthetic frames without OpenCV."""

    def __init__(
        self,
        *,
        source_id: str,
        total_frames: int,
        source_fps: float,
        frame_width: int,
        frame_height: int,
        target_processing_fps: float | None = None,
        use_source_fps: bool = False,
        frame_factory: Callable[[int], Any] | None = None,
    ) -> None:
        self._source_id = validate_non_empty_string(source_id, "source_id")
        self.total_frames = validate_non_negative_int(total_frames, "total_frames")
        self._source_fps = validate_positive_float(source_fps, "source_fps")
        self._frame_width = validate_positive_int(frame_width, "frame_width")
        self._frame_height = validate_positive_int(frame_height, "frame_height")
        if target_processing_fps is not None:
            validate_positive_float(target_processing_fps, "target_processing_fps")
        self.target_processing_fps = None if use_source_fps else target_processing_fps
        self.use_source_fps = bool(use_source_fps)
        self._frame_factory = frame_factory
        self.selected_frame_indices = build_processing_frame_indices(
            self.total_frames,
            self._source_fps,
            self.target_processing_fps,
        )
        self._opened = False
        self._closed = False
        self._cursor = 0

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
        if self._closed:
            raise RuntimeError("Cannot reopen a closed SyntheticFrameSource; call reset() first.")
        self._opened = True

    def read(self) -> FramePacket | None:
        if not self._opened:
            raise RuntimeError("SyntheticFrameSource must be opened before read().")
        if self._closed:
            raise RuntimeError("Cannot read from a closed SyntheticFrameSource.")
        if self._cursor >= len(self.selected_frame_indices):
            return None
        frame_index = self.selected_frame_indices[self._cursor]
        self._cursor += 1
        frame = (
            self._frame_factory(frame_index)
            if self._frame_factory is not None
            else {"synthetic": True, "source_id": self.source_id, "frame_index": frame_index}
        )
        return FramePacket(
            source_id=self.source_id,
            frame_index=frame_index,
            timestamp_sec=frame_index / self.source_fps,
            source_fps=self.source_fps,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            frame=frame,
        )

    def close(self) -> None:
        self._closed = True

    def reset(self) -> None:
        self._cursor = 0
        self._opened = False
        self._closed = False
