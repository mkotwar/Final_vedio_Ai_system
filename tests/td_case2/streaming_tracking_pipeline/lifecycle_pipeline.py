"""Sequential Step 4 pipeline: tracking output followed by lifecycle management."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import DetectionStage, FrameSource, TrackingStage, validate_detection_packet_matches_frame, validate_tracked_packet_matches_detection
from .lifecycle import LifecycleUpdateResult, TrackLifecycleManager
from .lifecycle_metrics import LifecycleMetricsAccumulator
from .schemas import DetectionPacket, FramePacket, TrackCompletionReason, TrackRecord, TrackedFramePacket
from .serialization import dataclass_to_dict, read_jsonl, to_json_safe, write_json
from .tracking_metrics import TrackingMetricsAccumulator


@dataclass(frozen=True)
class LifecyclePipelineReport:
    """JSON-safe report for a Step 4 lifecycle validation run."""

    run_id: str
    source_path: str
    source_id: str
    tracking_backend: str
    detector_model_path: str | None
    source_fps: float
    target_processing_fps: float | None
    total_source_frames: int | None
    selected_frames_processed: int
    first_processed_frame: int | None
    last_processed_frame: int | None
    first_timestamp_sec: float | None
    last_timestamp_sec: float | None
    raw_tracking_summary: dict[str, Any]
    detection_metrics: dict[str, Any]
    tracker_metrics: dict[str, Any]
    lifecycle_metrics: dict[str, Any]
    lifecycle_summary: dict[str, Any]
    end_of_stream_reached: bool
    tracker_flushed: bool
    lifecycle_flushed: bool
    source_closed: bool
    sink_closed: bool
    runtime_sec: float
    realtime_factor: float | None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


class LifecycleArtifactSink:
    """Write Step 4 tracking and lifecycle JSONL artifacts."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.source_dir = self.run_dir / "01_source"
        self.detection_dir = self.run_dir / "02_detections"
        self.track_dir = self.run_dir / "03_tracks"
        self.lifecycle_dir = self.run_dir / "04_lifecycle"
        self.report_dir = self.run_dir / "reports"
        for directory in (self.source_dir, self.detection_dir, self.track_dir, self.lifecycle_dir, self.report_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self._handles = {
            "detections": (self.detection_dir / "detection_packets.jsonl").open("w", encoding="utf-8", newline="\n"),
            "tracked_frames": (self.track_dir / "tracked_frame_packets.jsonl").open("w", encoding="utf-8", newline="\n"),
            "lifecycle_events": (self.lifecycle_dir / "lifecycle_events.jsonl").open("w", encoding="utf-8", newline="\n"),
            "active_snapshots": (self.lifecycle_dir / "active_track_snapshots.jsonl").open("w", encoding="utf-8", newline="\n"),
            "completed_tracks": (self.lifecycle_dir / "completed_tracks.jsonl").open("w", encoding="utf-8", newline="\n"),
        }
        self.frames_seen = 0
        self.closed = False

    def write_frame(self, packet: FramePacket) -> None:
        self.frames_seen += 1

    def write_detection(self, packet: DetectionPacket) -> None:
        self._write("detections", packet)

    def write_tracked_frame(self, packet: TrackedFramePacket) -> None:
        self._write("tracked_frames", packet)

    def write_lifecycle_result(self, result: LifecycleUpdateResult) -> None:
        for event in result.events:
            self._write("lifecycle_events", event)
        self._write(
            "active_snapshots",
            {
                "frame_index": result.frame_index,
                "timestamp_sec": result.timestamp_sec,
                "active_tracks": [track.to_dict() for track in result.active_tracks],
            },
        )
        for track in result.newly_completed_tracks:
            self._write("completed_tracks", track)

    def close(self) -> None:
        if self.closed:
            return
        for handle in self._handles.values():
            handle.flush()
            handle.close()
        self.closed = True

    def _write(self, name: str, value: Any) -> None:
        if self.closed:
            raise RuntimeError("Cannot write to closed LifecycleArtifactSink.")
        self._handles[name].write(json.dumps(to_json_safe(value), ensure_ascii=False, sort_keys=True))
        self._handles[name].write("\n")
        self._handles[name].flush()


class SequentialLifecycleTrackingPipeline:
    """Strict sequential source -> detection -> tracking -> lifecycle pipeline."""

    def __init__(
        self,
        *,
        run_id: str,
        source: FrameSource,
        detection_stage: DetectionStage,
        tracking_stage: TrackingStage,
        lifecycle_manager: TrackLifecycleManager,
        sink: LifecycleArtifactSink | None = None,
        source_path: str = "",
        detector_model_path: str | None = None,
        tracking_backend: str = "unknown",
    ) -> None:
        self.run_id = run_id
        self.source = source
        self.detection_stage = detection_stage
        self.tracking_stage = tracking_stage
        self.lifecycle_manager = lifecycle_manager
        self.sink = sink
        self.source_path = source_path
        self.detector_model_path = detector_model_path
        self.tracking_backend = tracking_backend
        self.raw_tracking_metrics = TrackingMetricsAccumulator()
        self.lifecycle_metrics = LifecycleMetricsAccumulator()
        self.last_report: LifecyclePipelineReport | None = None

    def run(self) -> LifecyclePipelineReport:
        started_at = time.perf_counter()
        frame_indices: list[int] = []
        timestamps: list[float] = []
        end_of_stream_reached = False
        tracker_flushed = False
        lifecycle_flushed = False
        source_closed = False
        sink_closed = False
        errors: list[str] = []
        try:
            self.source.open()
            while True:
                frame_packet = self.source.read()
                if frame_packet is None:
                    end_of_stream_reached = True
                    break
                frame_indices.append(frame_packet.frame_index)
                timestamps.append(frame_packet.timestamp_sec)
                if self.sink is not None:
                    self.sink.write_frame(frame_packet)
                detection_packet = self.detection_stage.process(frame_packet)
                validate_detection_packet_matches_frame(frame_packet, detection_packet)
                tracked_packet = self.tracking_stage.process(detection_packet)
                validate_tracked_packet_matches_detection(detection_packet, tracked_packet)
                lifecycle_result = self.lifecycle_manager.update(tracked_packet)
                self.raw_tracking_metrics.update(detection_packet, tracked_packet)
                self.lifecycle_metrics.update(lifecycle_result)
                if self.sink is not None:
                    self.sink.write_detection(detection_packet)
                    self.sink.write_tracked_frame(tracked_packet)
                    self.sink.write_lifecycle_result(lifecycle_result)
            self.tracking_stage.flush()
            tracker_flushed = True
            flush_result = self.lifecycle_manager.flush(
                frame_index=frame_indices[-1] if frame_indices else None,
                timestamp_sec=timestamps[-1] if timestamps else None,
                reason=TrackCompletionReason.VIDEO_ENDED,
            )
            lifecycle_flushed = True
            self.lifecycle_metrics.update(flush_result)
            if self.sink is not None:
                self.sink.write_lifecycle_result(flush_result)
        except Exception as exc:
            errors.append(str(exc))
            self.last_report = self._build_report(
                started_at=started_at,
                frame_indices=frame_indices,
                timestamps=timestamps,
                end_of_stream_reached=end_of_stream_reached,
                tracker_flushed=tracker_flushed,
                lifecycle_flushed=lifecycle_flushed,
                source_closed=False,
                sink_closed=False,
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
                    sink_closed = bool(getattr(self.sink, "closed", False))
        self.last_report = self._build_report(
            started_at=started_at,
            frame_indices=frame_indices,
            timestamps=timestamps,
            end_of_stream_reached=end_of_stream_reached,
            tracker_flushed=tracker_flushed,
            lifecycle_flushed=lifecycle_flushed,
            source_closed=source_closed,
            sink_closed=sink_closed,
            errors=errors,
        )
        return self.last_report

    def _build_report(
        self,
        *,
        started_at: float,
        frame_indices: list[int],
        timestamps: list[float],
        end_of_stream_reached: bool,
        tracker_flushed: bool,
        lifecycle_flushed: bool,
        source_closed: bool,
        sink_closed: bool,
        errors: list[str],
    ) -> LifecyclePipelineReport:
        runtime_sec = round(time.perf_counter() - started_at, 6)
        processed_duration = (timestamps[-1] - timestamps[0]) if len(timestamps) >= 2 else 0.0
        lifecycle_summary = build_lifecycle_summary(
            active_tracks=list(self.lifecycle_manager.get_active_tracks()),
            completed_tracks=list(self.lifecycle_manager.get_completed_tracks()),
            metrics=self.lifecycle_metrics.to_dict(),
        )
        return LifecyclePipelineReport(
            run_id=self.run_id,
            source_path=self.source_path,
            source_id=self.source.source_id,
            tracking_backend=self.tracking_backend,
            detector_model_path=self.detector_model_path,
            source_fps=self.source.source_fps,
            target_processing_fps=getattr(self.source, "target_processing_fps", None),
            total_source_frames=getattr(self.source, "total_frames", None),
            selected_frames_processed=len(frame_indices),
            first_processed_frame=frame_indices[0] if frame_indices else None,
            last_processed_frame=frame_indices[-1] if frame_indices else None,
            first_timestamp_sec=timestamps[0] if timestamps else None,
            last_timestamp_sec=timestamps[-1] if timestamps else None,
            raw_tracking_summary=self.raw_tracking_metrics.to_dict(),
            detection_metrics=self.detection_stage.to_dict() if hasattr(self.detection_stage, "to_dict") else {},
            tracker_metrics=self.tracking_stage.to_dict() if hasattr(self.tracking_stage, "to_dict") else {},
            lifecycle_metrics=self.lifecycle_metrics.to_dict(),
            lifecycle_summary=lifecycle_summary,
            end_of_stream_reached=end_of_stream_reached,
            tracker_flushed=tracker_flushed,
            lifecycle_flushed=lifecycle_flushed,
            source_closed=source_closed,
            sink_closed=sink_closed,
            runtime_sec=runtime_sec,
            realtime_factor=round(runtime_sec / processed_duration, 6) if processed_duration > 0 else None,
            errors=list(errors),
        )


def build_lifecycle_summary(
    *,
    active_tracks: list[TrackRecord],
    completed_tracks: list[TrackRecord],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "active_track_count": len(active_tracks),
        "completed_track_count": len(completed_tracks),
        "active_tracks": [track.to_dict() for track in active_tracks],
        "completed_tracks": [track.to_dict() for track in completed_tracks],
        "metrics": metrics,
    }


def finalize_step4_artifacts(run_dir: str | Path, report: LifecyclePipelineReport, source_metadata: dict[str, Any]) -> None:
    base = Path(run_dir)
    write_json(base / "01_source" / "source_metadata.json", source_metadata)
    write_json(base / "04_lifecycle" / "lifecycle_summary.json", report.lifecycle_summary)
    write_json(base / "reports" / "step4_lifecycle_report.json", report)
    read_jsonl(base / "02_detections" / "detection_packets.jsonl")
    read_jsonl(base / "03_tracks" / "tracked_frame_packets.jsonl")
    read_jsonl(base / "04_lifecycle" / "lifecycle_events.jsonl")
    read_jsonl(base / "04_lifecycle" / "active_track_snapshots.jsonl")
    read_jsonl(base / "04_lifecycle" / "completed_tracks.jsonl")
