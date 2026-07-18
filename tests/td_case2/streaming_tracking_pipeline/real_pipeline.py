"""Real sequential video -> YOLO -> ByteTrack pipeline for Step 3."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import DetectionStage, FrameSource, TrackingStage, validate_detection_packet_matches_frame, validate_tracked_packet_matches_detection
from .schemas import DetectionPacket, FramePacket, TrackedFramePacket
from .serialization import dataclass_to_dict, read_jsonl, to_json_safe, write_json
from .tracking_metrics import TrackingMetricsAccumulator


@dataclass(frozen=True)
class RealTrackingPipelineReport:
    """JSON-safe report for a real sequential tracking validation run."""

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
    detection_metrics: dict[str, Any]
    tracker_metrics: dict[str, Any]
    tracking_summary: dict[str, Any]
    end_of_stream_reached: bool
    tracker_flushed: bool
    source_closed: bool
    sink_closed: bool
    runtime_sec: float
    realtime_factor: float | None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


class Step3ArtifactSink:
    """Write Step 3 packet JSONL files into the requested output layout."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        save_annotated_video: bool = False,
        annotated_video_fps: float | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.source_dir = self.run_dir / "01_source"
        self.detection_dir = self.run_dir / "02_detections"
        self.track_dir = self.run_dir / "03_tracks"
        self.report_dir = self.run_dir / "reports"
        for directory in (self.source_dir, self.detection_dir, self.track_dir, self.report_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self._handles = {
            "detections": (self.detection_dir / "detection_packets.jsonl").open("w", encoding="utf-8", newline="\n"),
            "tracked_frames": (self.track_dir / "tracked_frame_packets.jsonl").open("w", encoding="utf-8", newline="\n"),
            "track_observations": (self.track_dir / "track_observations.jsonl").open("w", encoding="utf-8", newline="\n"),
        }
        self.frames_seen = 0
        self.closed = False
        self.annotated_video_path: Path | None = (
            self.run_dir / "04_visualization" / "tracked_video.mp4" if save_annotated_video else None
        )
        self.annotated_video_fps = annotated_video_fps
        self._video_writer: Any | None = None

    def write_frame(self, packet: FramePacket) -> None:
        self.frames_seen += 1

    def write_detection(self, packet: DetectionPacket) -> None:
        self._write("detections", packet)

    def write_tracked_frame(self, packet: TrackedFramePacket) -> None:
        self._write("tracked_frames", packet)
        for track in packet.tracks:
            self._write(
                "track_observations",
                {
                    "source_id": packet.source_id,
                    "frame_index": packet.frame_index,
                    "timestamp_sec": packet.timestamp_sec,
                    **dataclass_to_dict(track),
                },
            )
        self._write_annotated_frame(packet)

    def close(self) -> None:
        if self.closed:
            return
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        for handle in self._handles.values():
            handle.flush()
            handle.close()
        self.closed = True

    def __enter__(self) -> "Step3ArtifactSink":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _write(self, name: str, value: Any) -> None:
        if self.closed:
            raise RuntimeError("Cannot write to closed Step3ArtifactSink.")
        import json

        self._handles[name].write(json.dumps(to_json_safe(value), ensure_ascii=False, sort_keys=True))
        self._handles[name].write("\n")
        self._handles[name].flush()

    def _write_annotated_frame(self, packet: TrackedFramePacket) -> None:
        if self.annotated_video_path is None:
            return
        if packet.frame is None:
            raise RuntimeError("Annotated video output requires TrackedFramePacket.frame.")
        import cv2

        self.annotated_video_path.parent.mkdir(parents=True, exist_ok=True)
        if self._video_writer is None:
            fps = self.annotated_video_fps or max(1.0, float(packet.frame_width and packet.frame_height and 5.0))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(self.annotated_video_path), fourcc, fps, (packet.frame_width, packet.frame_height))
            if not writer.isOpened():
                writer.release()
                raise RuntimeError(f"Failed to create annotated video: {self.annotated_video_path}")
            self._video_writer = writer
        frame = packet.frame.copy()
        for track in packet.tracks:
            x1, y1, x2, y2 = [int(round(value)) for value in track.bbox.to_xyxy()]
            color = (37, 189, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"id {track.track_id} {track.class_name} {track.confidence:.2f}"
            cv2.putText(frame, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        self._video_writer.write(frame)


class RealSequentialTrackingPipeline:
    """Strict one-frame-at-a-time real tracking reference pipeline."""

    def __init__(
        self,
        *,
        run_id: str,
        source: FrameSource,
        detection_stage: DetectionStage,
        tracking_stage: TrackingStage,
        sink: Step3ArtifactSink | None = None,
        source_path: str = "",
        detector_model_path: str | None = None,
        tracking_backend: str = "unknown",
        expected_physical_objects: int | None = None,
    ) -> None:
        self.run_id = run_id
        self.source = source
        self.detection_stage = detection_stage
        self.tracking_stage = tracking_stage
        self.sink = sink
        self.source_path = source_path
        self.detector_model_path = detector_model_path
        self.tracking_backend = tracking_backend
        self.metrics = TrackingMetricsAccumulator(expected_physical_objects=expected_physical_objects)
        self.last_report: RealTrackingPipelineReport | None = None

    def run(self) -> RealTrackingPipelineReport:
        started_at = time.perf_counter()
        frame_indices: list[int] = []
        timestamps: list[float] = []
        end_of_stream_reached = False
        tracker_flushed = False
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
                if frame_indices and frame_packet.frame_index <= frame_indices[-1]:
                    raise ValueError("Source emitted non-increasing frame indices.")
                frame_indices.append(frame_packet.frame_index)
                timestamps.append(frame_packet.timestamp_sec)
                if self.sink is not None:
                    self.sink.write_frame(frame_packet)
                detection_packet = self.detection_stage.process(frame_packet)
                validate_detection_packet_matches_frame(frame_packet, detection_packet)
                if self.sink is not None:
                    self.sink.write_detection(detection_packet)
                tracked_packet = self.tracking_stage.process(detection_packet)
                validate_tracked_packet_matches_detection(detection_packet, tracked_packet)
                if self.sink is not None:
                    self.sink.write_tracked_frame(tracked_packet)
                self.metrics.update(detection_packet, tracked_packet)
            self.tracking_stage.flush()
            tracker_flushed = True
        except Exception as exc:
            errors.append(str(exc))
            self.last_report = self._build_report(
                started_at=started_at,
                frame_indices=frame_indices,
                timestamps=timestamps,
                end_of_stream_reached=end_of_stream_reached,
                tracker_flushed=tracker_flushed,
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
        source_closed: bool,
        sink_closed: bool,
        errors: list[str],
    ) -> RealTrackingPipelineReport:
        runtime_sec = round(time.perf_counter() - started_at, 6)
        processed_duration = (timestamps[-1] - timestamps[0]) if len(timestamps) >= 2 else 0.0
        realtime_factor = round(runtime_sec / processed_duration, 6) if processed_duration > 0 else None
        detection_metrics = self.detection_stage.to_dict() if hasattr(self.detection_stage, "to_dict") else {}
        tracker_metrics = self.tracking_stage.to_dict() if hasattr(self.tracking_stage, "to_dict") else {}
        return RealTrackingPipelineReport(
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
            detection_metrics=detection_metrics,
            tracker_metrics=tracker_metrics,
            tracking_summary=self.metrics.to_dict(),
            end_of_stream_reached=end_of_stream_reached,
            tracker_flushed=tracker_flushed,
            source_closed=source_closed,
            sink_closed=sink_closed,
            runtime_sec=runtime_sec,
            realtime_factor=realtime_factor,
            errors=list(errors),
        )


def finalize_step3_artifacts(run_dir: str | Path, report: RealTrackingPipelineReport, source_metadata: dict[str, Any]) -> None:
    """Write source metadata, track summary, and final report."""

    base = Path(run_dir)
    write_json(base / "01_source" / "source_metadata.json", source_metadata)
    write_json(base / "03_tracks" / "track_summary.json", report.tracking_summary)
    write_json(base / "reports" / "step3_real_tracking_report.json", report)
    # Sanity-read output files so runner failures catch corrupt JSONL immediately.
    read_jsonl(base / "02_detections" / "detection_packets.jsonl")
    read_jsonl(base / "03_tracks" / "tracked_frame_packets.jsonl")
    read_jsonl(base / "03_tracks" / "track_observations.jsonl")
