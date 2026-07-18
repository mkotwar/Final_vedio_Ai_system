"""Deterministic model-free stages for Step 2 contract validation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from .contracts import validate_detection_packet_matches_frame, validate_tracked_packet_matches_detection
from .schemas import BoundingBox, DetectionPacket, DetectionRecord, FramePacket, TrackedFramePacket, TrackedObject


def _box_inside_frame(bbox: BoundingBox, frame_width: int, frame_height: int) -> bool:
    return bbox.x1 >= 0 and bbox.y1 >= 0 and bbox.x2 <= frame_width and bbox.y2 <= frame_height


class DeterministicMockDetectionStage:
    """Return configured detections sorted by class, position, then confidence."""

    def __init__(
        self,
        *,
        detections_by_frame: Mapping[int, Sequence[DetectionRecord]] | None = None,
        detection_factory: Callable[[FramePacket], Sequence[DetectionRecord]] | None = None,
    ) -> None:
        self._detections_by_frame = {int(key): tuple(value) for key, value in (detections_by_frame or {}).items()}
        self._detection_factory = detection_factory
        self.call_count = 0
        self.processed_frame_indices: list[int] = []

    def process(self, packet: FramePacket) -> DetectionPacket:
        self.call_count += 1
        self.processed_frame_indices.append(packet.frame_index)
        if self._detection_factory is not None:
            detections = list(self._detection_factory(packet))
        else:
            detections = list(self._detections_by_frame.get(packet.frame_index, ()))
        for detection in detections:
            if not _box_inside_frame(detection.bbox, packet.frame_width, packet.frame_height):
                raise ValueError(f"Detection bbox is outside frame bounds for frame {packet.frame_index}.")
        ordered = sorted(
            detections,
            key=lambda item: (item.class_id, item.bbox.x1, item.bbox.y1, -item.confidence),
        )
        output = DetectionPacket(
            source_id=packet.source_id,
            frame_index=packet.frame_index,
            timestamp_sec=packet.timestamp_sec,
            frame_width=packet.frame_width,
            frame_height=packet.frame_height,
            detections=list(ordered),
            frame=packet.frame,
        )
        validate_detection_packet_matches_frame(packet, output)
        return output


class DeterministicMockTrackingStage:
    """Configured tracker-output stage; this does not associate detections."""

    def __init__(
        self,
        *,
        tracks_by_frame: Mapping[int, Sequence[TrackedObject]] | None = None,
        track_factory: Callable[[DetectionPacket], Sequence[TrackedObject]] | None = None,
        reject_duplicate_frame_indices: bool = True,
    ) -> None:
        self._tracks_by_frame = {int(key): tuple(value) for key, value in (tracks_by_frame or {}).items()}
        self._track_factory = track_factory
        self.reject_duplicate_frame_indices = reject_duplicate_frame_indices
        self.processed_frame_indices: list[int] = []
        self.reset_count = 0
        self.flush_count = 0
        self._last_frame_index: int | None = None
        self._source_id: str | None = None

    def process(self, packet: DetectionPacket) -> TrackedFramePacket:
        if self._source_id is None:
            self._source_id = packet.source_id
        elif packet.source_id != self._source_id:
            raise ValueError("Tracking stage received a second source without reset().")
        if self._last_frame_index is not None:
            if packet.frame_index < self._last_frame_index:
                raise ValueError("Tracking stage rejected frame-order regression.")
            if self.reject_duplicate_frame_indices and packet.frame_index == self._last_frame_index:
                raise ValueError("Tracking stage rejected duplicate frame_index.")
        if self._track_factory is not None:
            tracks = list(self._track_factory(packet))
        else:
            tracks = list(self._tracks_by_frame.get(packet.frame_index, ()))
        for track in tracks:
            if track.frame_index != packet.frame_index:
                raise ValueError("Configured TrackedObject frame_index does not match packet.")
            if track.timestamp_sec != packet.timestamp_sec:
                raise ValueError("Configured TrackedObject timestamp_sec does not match packet.")
            if not _box_inside_frame(track.bbox, packet.frame_width, packet.frame_height):
                raise ValueError(f"Tracked bbox is outside frame bounds for frame {packet.frame_index}.")
        ordered = sorted(tracks, key=lambda item: (item.track_id, item.class_id, item.bbox.x1, item.bbox.y1))
        self._last_frame_index = packet.frame_index
        self.processed_frame_indices.append(packet.frame_index)
        output = TrackedFramePacket(
            source_id=packet.source_id,
            frame_index=packet.frame_index,
            timestamp_sec=packet.timestamp_sec,
            frame_width=packet.frame_width,
            frame_height=packet.frame_height,
            tracks=list(ordered),
            frame=packet.frame,
        )
        validate_tracked_packet_matches_detection(packet, output)
        return output

    def reset(self) -> None:
        self.reset_count += 1
        self.processed_frame_indices.clear()
        self._last_frame_index = None
        self._source_id = None

    def flush(self) -> list[dict[str, int]]:
        self.flush_count += 1
        return []
