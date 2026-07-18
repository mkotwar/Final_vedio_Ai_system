"""Sequential contract pipeline for Step 2 validation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .contracts import (
    DetectionStage,
    FrameSource,
    PacketSink,
    TrackingStage,
    validate_detection_packet_matches_frame,
    validate_tracked_packet_matches_detection,
)
from .serialization import dataclass_to_dict


@dataclass(frozen=True)
class SequentialPipelineReport:
    """JSON-safe report for one sequential contract run."""

    source_id: str
    source_fps: float
    target_processing_fps: float | None
    total_source_frames: int | None
    selected_frames_processed: int
    first_frame_index: int | None
    last_frame_index: int | None
    first_timestamp_sec: float | None
    last_timestamp_sec: float | None
    detection_packets_created: int
    tracked_packets_created: int
    total_detections: int
    total_tracked_objects: int
    frame_order_valid: bool
    end_of_stream_reached: bool
    tracker_flushed: bool
    source_closed: bool
    runtime_sec: float
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


class SequentialContractPipeline:
    """Run source -> detection -> tracking one selected frame at a time."""

    def __init__(
        self,
        *,
        source: FrameSource,
        detection_stage: DetectionStage,
        tracking_stage: TrackingStage,
        sink: PacketSink | None = None,
    ) -> None:
        self.source = source
        self.detection_stage = detection_stage
        self.tracking_stage = tracking_stage
        self.sink = sink
        self.last_report: SequentialPipelineReport | None = None

    def run(self) -> SequentialPipelineReport:
        """Run the contract pipeline; errors are raised after cleanup."""

        started_at = time.perf_counter()
        frame_indices: list[int] = []
        timestamps: list[float] = []
        detection_packets_created = 0
        tracked_packets_created = 0
        total_detections = 0
        total_tracked_objects = 0
        end_of_stream_reached = False
        tracker_flushed = False
        source_closed = False
        errors: list[str] = []

        try:
            self.source.open()
            while True:
                frame_packet = self.source.read()
                if frame_packet is None:
                    end_of_stream_reached = True
                    break
                if frame_indices and frame_packet.frame_index <= frame_indices[-1]:
                    raise ValueError("Source emitted non-increasing frame indices.")
                frame_indices.append(frame_packet.frame_index)
                timestamps.append(frame_packet.timestamp_sec)
                if self.sink is not None:
                    self.sink.write_frame(frame_packet)

                detection_packet = self.detection_stage.process(frame_packet)
                validate_detection_packet_matches_frame(frame_packet, detection_packet)
                detection_packets_created += 1
                total_detections += len(detection_packet.detections)
                if self.sink is not None:
                    self.sink.write_detection(detection_packet)

                tracked_packet = self.tracking_stage.process(detection_packet)
                validate_tracked_packet_matches_detection(detection_packet, tracked_packet)
                tracked_packets_created += 1
                total_tracked_objects += len(tracked_packet.tracks)
                if self.sink is not None:
                    self.sink.write_tracked_frame(tracked_packet)

            self.tracking_stage.flush()
            tracker_flushed = True
        except Exception as exc:
            errors.append(str(exc))
            self.last_report = self._build_report(
                started_at=started_at,
                frame_indices=frame_indices,
                timestamps=timestamps,
                detection_packets_created=detection_packets_created,
                tracked_packets_created=tracked_packets_created,
                total_detections=total_detections,
                total_tracked_objects=total_tracked_objects,
                end_of_stream_reached=end_of_stream_reached,
                tracker_flushed=tracker_flushed,
                source_closed=False,
                errors=errors,
            )
            raise
        finally:
            try:
                self.source.close()
            finally:
                source_closed = True
                if self.sink is not None:
                    self.sink.close()

        self.last_report = self._build_report(
            started_at=started_at,
            frame_indices=frame_indices,
            timestamps=timestamps,
            detection_packets_created=detection_packets_created,
            tracked_packets_created=tracked_packets_created,
            total_detections=total_detections,
            total_tracked_objects=total_tracked_objects,
            end_of_stream_reached=end_of_stream_reached,
            tracker_flushed=tracker_flushed,
            source_closed=source_closed,
            errors=errors,
        )
        return self.last_report

    def _build_report(
        self,
        *,
        started_at: float,
        frame_indices: list[int],
        timestamps: list[float],
        detection_packets_created: int,
        tracked_packets_created: int,
        total_detections: int,
        total_tracked_objects: int,
        end_of_stream_reached: bool,
        tracker_flushed: bool,
        source_closed: bool,
        errors: list[str],
    ) -> SequentialPipelineReport:
        frame_order_valid = all(left < right for left, right in zip(frame_indices, frame_indices[1:]))
        return SequentialPipelineReport(
            source_id=self.source.source_id,
            source_fps=self.source.source_fps,
            target_processing_fps=getattr(self.source, "target_processing_fps", None),
            total_source_frames=getattr(self.source, "total_frames", None),
            selected_frames_processed=len(frame_indices),
            first_frame_index=frame_indices[0] if frame_indices else None,
            last_frame_index=frame_indices[-1] if frame_indices else None,
            first_timestamp_sec=timestamps[0] if timestamps else None,
            last_timestamp_sec=timestamps[-1] if timestamps else None,
            detection_packets_created=detection_packets_created,
            tracked_packets_created=tracked_packets_created,
            total_detections=total_detections,
            total_tracked_objects=total_tracked_objects,
            frame_order_valid=frame_order_valid,
            end_of_stream_reached=end_of_stream_reached,
            tracker_flushed=tracker_flushed,
            source_closed=source_closed,
            runtime_sec=round(time.perf_counter() - started_at, 6),
            errors=list(errors),
        )
