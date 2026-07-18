from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import DetectionStage, FrameSource, TrackingStage, validate_detection_packet_matches_frame, validate_tracked_packet_matches_detection
from .crop_artifacts import CropArtifactSink, CompletedTrackCropBundle
from .crop_collector import CropCandidateCollector
from .lifecycle import TrackLifecycleManager
from .lifecycle_metrics import LifecycleMetricsAccumulator
from .lifecycle_pipeline import LifecycleArtifactSink, build_lifecycle_summary
from .schemas import TrackCompletionReason
from .serialization import dataclass_to_dict, read_jsonl, write_json
from .tracking_metrics import TrackingMetricsAccumulator


@dataclass(frozen=True)
class CropPipelineReport:
    """JSON-safe report for Step 5 sequential crop collection validation."""

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
    crop_collection_summary: dict[str, Any]
    end_of_stream_reached: bool
    tracker_flushed: bool
    lifecycle_flushed: bool
    source_closed: bool
    lifecycle_sink_closed: bool
    crop_sink_closed: bool
    runtime_sec: float
    realtime_factor: float | None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


class SequentialCropCollectionPipeline:
    """Strict sequential source -> detection -> tracking -> lifecycle -> crop collection pipeline."""

    def __init__(
        self,
        *,
        run_id: str,
        source: FrameSource,
        detection_stage: DetectionStage,
        tracking_stage: TrackingStage,
        lifecycle_manager: TrackLifecycleManager,
        crop_collector: CropCandidateCollector,
        lifecycle_sink: LifecycleArtifactSink | None = None,
        crop_sink: CropArtifactSink | None = None,
        source_path: str = "",
        detector_model_path: str | None = None,
        tracking_backend: str = "unknown",
    ) -> None:
        self.run_id = run_id
        self.source = source
        self.detection_stage = detection_stage
        self.tracking_stage = tracking_stage
        self.lifecycle_manager = lifecycle_manager
        self.crop_collector = crop_collector
        self.lifecycle_sink = lifecycle_sink
        self.crop_sink = crop_sink
        self.source_path = source_path
        self.detector_model_path = detector_model_path
        self.tracking_backend = tracking_backend
        self.raw_tracking_metrics = TrackingMetricsAccumulator()
        self.lifecycle_metrics = LifecycleMetricsAccumulator()
        self.completed_bundles: list[CompletedTrackCropBundle] = []
        self.last_report: CropPipelineReport | None = None

    def run(self) -> CropPipelineReport:
        started_at = time.perf_counter()
        frame_indices: list[int] = []
        timestamps: list[float] = []
        end_of_stream_reached = False
        tracker_flushed = False
        lifecycle_flushed = False
        source_closed = False
        lifecycle_sink_closed = False
        crop_sink_closed = False
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
                if self.lifecycle_sink is not None:
                    self.lifecycle_sink.write_frame(frame_packet)
                detection_packet = self.detection_stage.process(frame_packet)
                validate_detection_packet_matches_frame(frame_packet, detection_packet)
                tracked_packet = self.tracking_stage.process(detection_packet)
                validate_tracked_packet_matches_detection(detection_packet, tracked_packet)
                lifecycle_result = self.lifecycle_manager.update(tracked_packet)
                crop_result = self.crop_collector.update(tracked_packet, lifecycle_result)
                self.raw_tracking_metrics.update(detection_packet, tracked_packet)
                self.lifecycle_metrics.update(lifecycle_result)
                if self.lifecycle_sink is not None:
                    self.lifecycle_sink.write_detection(detection_packet)
                    self.lifecycle_sink.write_tracked_frame(tracked_packet)
                    self.lifecycle_sink.write_lifecycle_result(lifecycle_result)
                if self.crop_sink is not None:
                    for runtime_observation in crop_result.observation_result.observations:
                        self.crop_sink.write_observation(runtime_observation.observation)
                    for candidate in crop_result.candidates_created:
                        self.crop_sink.write_candidate(candidate)
                    for bundle in crop_result.completed_bundles:
                        self.crop_sink.write_completed_bundle(bundle)
                self.completed_bundles.extend(crop_result.completed_bundles)

            self.tracking_stage.flush()
            tracker_flushed = True
            flush_result = self.lifecycle_manager.flush(
                frame_index=frame_indices[-1] if frame_indices else None,
                timestamp_sec=timestamps[-1] if timestamps else None,
                reason=TrackCompletionReason.VIDEO_ENDED,
            )
            lifecycle_flushed = True
            self.lifecycle_metrics.update(flush_result)
            flush_bundles = self.crop_collector.complete_tracks(flush_result.newly_completed_tracks)
            self.completed_bundles.extend(flush_bundles)
            if self.lifecycle_sink is not None:
                self.lifecycle_sink.write_lifecycle_result(flush_result)
            if self.crop_sink is not None:
                for bundle in flush_bundles:
                    self.crop_sink.write_completed_bundle(bundle)
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
                lifecycle_sink_closed=False,
                crop_sink_closed=False,
                errors=errors,
            )
            raise
        finally:
            try:
                self.source.close()
            finally:
                source_closed = True
                if self.lifecycle_sink is not None:
                    self.lifecycle_sink.close()
                    lifecycle_sink_closed = bool(getattr(self.lifecycle_sink, "closed", False))
                if self.crop_sink is not None:
                    self.crop_sink.close()
                    crop_sink_closed = bool(getattr(self.crop_sink, "closed", False))
        self.last_report = self._build_report(
            started_at=started_at,
            frame_indices=frame_indices,
            timestamps=timestamps,
            end_of_stream_reached=end_of_stream_reached,
            tracker_flushed=tracker_flushed,
            lifecycle_flushed=lifecycle_flushed,
            source_closed=source_closed,
            lifecycle_sink_closed=lifecycle_sink_closed,
            crop_sink_closed=crop_sink_closed,
            errors=errors,
        )
        if self.crop_sink is not None:
            self.crop_sink.write_summary(self.last_report.crop_collection_summary)
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
        lifecycle_sink_closed: bool,
        crop_sink_closed: bool,
        errors: list[str],
    ) -> CropPipelineReport:
        runtime_sec = round(time.perf_counter() - started_at, 6)
        processed_duration = (timestamps[-1] - timestamps[0]) if len(timestamps) >= 2 else 0.0
        lifecycle_summary = build_lifecycle_summary(
            active_tracks=list(self.lifecycle_manager.get_active_tracks()),
            completed_tracks=list(self.lifecycle_manager.get_completed_tracks()),
            metrics=self.lifecycle_metrics.to_dict(),
        )
        crop_summary = self.crop_collector.to_dict()
        crop_summary.update(
            {
                "completed_bundle_count": len(self.completed_bundles),
                "bundles_with_candidates": sum(1 for item in self.completed_bundles if item.retained_candidate_count > 0),
                "empty_completed_bundles": sum(1 for item in self.completed_bundles if item.retained_candidate_count == 0),
            }
        )
        if self.crop_sink is not None:
            crop_summary["artifact_counts"] = dict(self.crop_sink.counts)
        return CropPipelineReport(
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
            crop_collection_summary=crop_summary,
            end_of_stream_reached=end_of_stream_reached,
            tracker_flushed=tracker_flushed,
            lifecycle_flushed=lifecycle_flushed,
            source_closed=source_closed,
            lifecycle_sink_closed=lifecycle_sink_closed,
            crop_sink_closed=crop_sink_closed,
            runtime_sec=runtime_sec,
            realtime_factor=round(runtime_sec / processed_duration, 6) if processed_duration > 0 else None,
            errors=list(errors),
        )


def finalize_step5_artifacts(run_dir: str | Path, report: CropPipelineReport, source_metadata: dict[str, Any]) -> None:
    base = Path(run_dir)
    write_json(base / "01_source" / "source_metadata.json", source_metadata)
    write_json(base / "04_lifecycle" / "lifecycle_summary.json", report.lifecycle_summary)
    write_json(base / "05_crops" / "crop_collection_summary.json", report.crop_collection_summary)
    write_json(base / "reports" / "step5_crop_collection_report.json", report)
    read_jsonl(base / "02_detections" / "detection_packets.jsonl")
    read_jsonl(base / "03_tracks" / "tracked_frame_packets.jsonl")
    read_jsonl(base / "04_lifecycle" / "lifecycle_events.jsonl")
    read_jsonl(base / "04_lifecycle" / "active_track_snapshots.jsonl")
    read_jsonl(base / "04_lifecycle" / "completed_tracks.jsonl")
    read_jsonl(base / "05_crops" / "track_observations.jsonl")
    read_jsonl(base / "05_crops" / "crop_candidates.jsonl")
    read_jsonl(base / "05_crops" / "completed_track_crop_bundles.jsonl")
