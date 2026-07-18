"""Contracts and metadata validation for sequential streaming pipeline stages."""

from __future__ import annotations

from typing import Any, Protocol

from .schemas import DetectionPacket, FramePacket, TrackedFramePacket


class FrameSource(Protocol):
    """Sequential frame source contract."""

    @property
    def source_id(self) -> str:
        ...

    @property
    def source_fps(self) -> float:
        ...

    @property
    def frame_width(self) -> int:
        ...

    @property
    def frame_height(self) -> int:
        ...

    def open(self) -> None:
        ...

    def read(self) -> FramePacket | None:
        ...

    def close(self) -> None:
        ...

    def reset(self) -> None:
        ...


class DetectionStage(Protocol):
    """One-frame-in, one-detection-packet-out detector contract."""

    def process(self, packet: FramePacket) -> DetectionPacket:
        ...


class TrackingStage(Protocol):
    """Sequential tracking contract for one source stream."""

    def process(self, packet: DetectionPacket) -> TrackedFramePacket:
        ...

    def reset(self) -> None:
        ...

    def flush(self) -> list[Any]:
        ...


class PacketSink(Protocol):
    """Optional packet artifact sink contract."""

    def write_frame(self, packet: FramePacket) -> None:
        ...

    def write_detection(self, packet: DetectionPacket) -> None:
        ...

    def write_tracked_frame(self, packet: TrackedFramePacket) -> None:
        ...

    def close(self) -> None:
        ...


def validate_detection_packet_matches_frame(frame_packet: FramePacket, detection_packet: DetectionPacket) -> None:
    """Raise when a detection stage changes frame metadata unexpectedly."""

    mismatches: list[str] = []
    if detection_packet.source_id != frame_packet.source_id:
        mismatches.append("source_id")
    if detection_packet.frame_index != frame_packet.frame_index:
        mismatches.append("frame_index")
    if detection_packet.timestamp_sec != frame_packet.timestamp_sec:
        mismatches.append("timestamp_sec")
    if detection_packet.frame_width != frame_packet.frame_width:
        mismatches.append("frame_width")
    if detection_packet.frame_height != frame_packet.frame_height:
        mismatches.append("frame_height")
    if detection_packet.frame is not frame_packet.frame:
        mismatches.append("frame")
    if mismatches:
        raise ValueError(f"DetectionPacket metadata mismatch: {', '.join(mismatches)}.")


def validate_tracked_packet_matches_detection(detection_packet: DetectionPacket, tracked_packet: TrackedFramePacket) -> None:
    """Raise when a tracking stage changes frame metadata unexpectedly."""

    mismatches: list[str] = []
    if tracked_packet.source_id != detection_packet.source_id:
        mismatches.append("source_id")
    if tracked_packet.frame_index != detection_packet.frame_index:
        mismatches.append("frame_index")
    if tracked_packet.timestamp_sec != detection_packet.timestamp_sec:
        mismatches.append("timestamp_sec")
    if tracked_packet.frame_width != detection_packet.frame_width:
        mismatches.append("frame_width")
    if tracked_packet.frame_height != detection_packet.frame_height:
        mismatches.append("frame_height")
    if tracked_packet.frame is not detection_packet.frame:
        mismatches.append("frame")
    if mismatches:
        raise ValueError(f"TrackedFramePacket metadata mismatch: {', '.join(mismatches)}.")
