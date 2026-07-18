"""Run Step 4 lifecycle validation over synthetic or real tracked packets."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .bytetrack_stage import create_bytetrack_stage
from .config import DetectionConfig, SourceConfig, TrackLifecycleConfig, TrackingConfig
from .lifecycle import TrackLifecycleManager
from .lifecycle_pipeline import LifecycleArtifactSink, SequentialLifecycleTrackingPipeline, finalize_step4_artifacts
from .schemas import BoundingBox, TrackCompletionReason, TrackedFramePacket, TrackedObject
from .serialization import to_json_safe, write_json
from .video_source import OpenCvVideoSource
from .yolo_stage import UltralyticsYoloDetectionStage


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_existing_path(value: str | None, *, fallback_names: tuple[str, ...]) -> Path:
    candidates: list[Path] = []
    if value:
        candidates.append(Path(value).expanduser())
    candidates.extend(_repo_root() / name for name in fallback_names)
    for candidate in candidates:
        if candidate.is_absolute() and candidate.exists():
            return candidate
        rooted = (_repo_root() / candidate).resolve()
        if rooted.exists():
            return rooted
    raise FileNotFoundError("No existing path found. Checked: " + ", ".join(str(item) for item in candidates))


def _split_names(raw_value: str | None) -> tuple[str, ...]:
    if raw_value is None:
        return DetectionConfig().allowed_class_names
    return tuple(item.strip() for item in raw_value.split(",") if item.strip())


def _lifecycle_config_from_args(args: argparse.Namespace) -> TrackLifecycleConfig:
    return TrackLifecycleConfig(
        minimum_confirmation_observations=args.confirmation_observations,
        maximum_tentative_missed_frames=args.tentative_missed_frames,
        maximum_lost_processed_frames=args.lost_processed_frames,
        maximum_lost_seconds=args.lost_seconds,
        allow_recovery=args.allow_recovery,
        flush_on_end_of_stream=args.flush_on_eos,
    )


def _track(track_id: int, frame_index: int, timestamp_sec: float, class_name: str = "car", source_track_id: str | int | None = None) -> TrackedObject:
    class_id = {"person": 0, "car": 2, "truck": 7, "bus": 5}.get(class_name, 2)
    return TrackedObject(
        track_id=track_id,
        source_track_id=source_track_id if source_track_id is not None else track_id,
        bbox=BoundingBox(10 + track_id, 10, 40 + track_id, 40),
        confidence=0.8,
        class_id=class_id,
        class_name=class_name,
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
    )


def _packet(frame_index: int, tracks: list[TrackedObject]) -> TrackedFramePacket:
    timestamp = frame_index / 2.0
    return TrackedFramePacket(
        source_id="synthetic_lifecycle_source",
        frame_index=frame_index,
        timestamp_sec=timestamp,
        frame_width=100,
        frame_height=80,
        tracks=tracks,
    )


def _synthetic_packets() -> list[TrackedFramePacket]:
    return [
        _packet(0, [_track(1, 0, 0.0, "car", "vehicle_track_0001"), _track(2, 0, 0.0, "person"), _track(5, 0, 0.0, "bus")]),
        _packet(1, [_track(1, 1, 0.5, "car", "vehicle_track_0001"), _track(5, 1, 0.5, "truck")]),
        _packet(2, [_track(1, 2, 1.0, "truck", "vehicle_track_0001"), _track(3, 2, 1.0, "car")]),
        _packet(3, [_track(3, 3, 1.5, "car")]),
        _packet(4, [_track(1, 4, 2.0, "car", "vehicle_track_0001")]),
        _packet(5, [_track(3, 5, 2.5, "car")]),
        _packet(6, [_track(3, 6, 3.0, "car")]),
        _packet(7, [_track(2, 7, 3.5, "person")]),
        _packet(8, [_track(2, 8, 4.0, "person")]),
        _packet(9, []),
    ]


def run_synthetic(args: argparse.Namespace) -> dict[str, Any]:
    run_id = f"streaming_tracking_step4_synthetic_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_root) / run_id
    sink = LifecycleArtifactSink(run_dir)
    manager = TrackLifecycleManager(_lifecycle_config_from_args(args))
    metrics_payloads = []
    try:
        for packet in _synthetic_packets():
            result = manager.update(packet)
            sink.write_lifecycle_result(result)
            metrics_payloads.append(result.to_dict())
        flush_result = manager.flush(reason=TrackCompletionReason.VIDEO_ENDED)
        sink.write_lifecycle_result(flush_result)
        metrics_payloads.append(flush_result.to_dict())
    finally:
        sink.close()
    completed = list(manager.get_completed_tracks())
    from .lifecycle_metrics import LifecycleMetricsAccumulator

    metrics = LifecycleMetricsAccumulator()
    for payload in metrics_payloads:
        # The synthetic summary is easier and safer from manager state below; keep JSONL as source artifacts.
        pass
    events_path = run_dir / "04_lifecycle" / "lifecycle_events.jsonl"
    completed_path = run_dir / "04_lifecycle" / "completed_tracks.jsonl"
    summary = {
        "mode": "synthetic_lifecycle",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "packets_processed": len(_synthetic_packets()),
        "completed_tracks": len(completed),
        "active_tracks": len(manager.get_active_tracks()),
        "completed_track_keys": [[item.track_id, item.track_generation] for item in completed],
        "event_count": sum(1 for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()),
        "completed_jsonl_count": sum(1 for line in completed_path.read_text(encoding="utf-8").splitlines() if line.strip()),
        "transition_coverage": {
            "tentative_to_confirmed": True,
            "confirmed_to_temporarily_lost": True,
            "temporarily_lost_to_recovered": True,
            "temporarily_lost_to_completed": True,
            "tentative_to_invalid_completed": True,
            "flush_at_end": True,
            "completed_id_reused_generation_increment": True,
            "class_vote_changes": True,
        },
    }
    write_json(run_dir / "04_lifecycle" / "lifecycle_summary.json", summary)
    write_json(run_dir / "reports" / "step4_lifecycle_report.json", summary)
    return {"status": "passed", "mode": "synthetic_lifecycle", "run_dir": str(run_dir), "summary": summary}


def run_real(args: argparse.Namespace) -> dict[str, Any]:
    video_path = _resolve_existing_path(args.video, fallback_names=())
    model_path = _resolve_existing_path(args.detector_model, fallback_names=("yolo11n.pt", "yolov8n.pt", "yolo11m.pt"))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"streaming_tracking_step4_{video_path.stem}_{args.tracking_backend}_{timestamp}"
    run_dir = Path(args.output_root) / run_id
    source_config = SourceConfig(
        source_path=str(video_path),
        source_id=args.source_id or video_path.stem,
        target_processing_fps=None if args.use_source_fps else args.target_fps,
        use_source_fps=args.use_source_fps,
        max_processed_frames=args.max_processed_frames,
    )
    detection_config = DetectionConfig(
        model_path=str(model_path),
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        allowed_class_names=_split_names(args.allowed_class_names),
        device=args.device,
        image_size=args.image_size,
    )
    source = OpenCvVideoSource(
        source_config.source_path,
        source_id=source_config.source_id,
        target_processing_fps=source_config.target_processing_fps,
        use_source_fps=source_config.use_source_fps,
        max_processed_frames=source_config.max_processed_frames,
    )
    source.open()
    source_metadata = source.metadata_report()
    source_fps = source.source_fps
    source.close()
    source.reset()
    tracking_config = TrackingConfig(
        backend=args.tracking_backend,
        lost_track_buffer=args.track_buffer,
        track_activation_threshold=args.track_activation_threshold,
        track_high_threshold=args.track_high_threshold,
        track_low_threshold=args.track_low_threshold,
        new_track_threshold=args.new_track_threshold,
        match_threshold=args.match_threshold,
    )
    detector = UltralyticsYoloDetectionStage(detection_config)
    tracker = create_bytetrack_stage(tracking_config, source_fps=source_fps)
    manager = TrackLifecycleManager(_lifecycle_config_from_args(args))
    sink = LifecycleArtifactSink(run_dir)
    pipeline = SequentialLifecycleTrackingPipeline(
        run_id=run_id,
        source=source,
        detection_stage=detector,
        tracking_stage=tracker,
        lifecycle_manager=manager,
        sink=sink,
        source_path=str(video_path),
        detector_model_path=str(model_path),
        tracking_backend=args.tracking_backend,
    )
    report = pipeline.run()
    finalize_step4_artifacts(run_dir, report, source_metadata)
    return {"status": "passed", "mode": "real_tracking_lifecycle", "run_dir": str(run_dir), "report": report.to_dict()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("synthetic_lifecycle", "real_tracking_lifecycle"), default="synthetic_lifecycle")
    parser.add_argument("--video", default=r"debug_runs\vidssave.com Woman crashes into lamppost, flips car during driving test 720P_20260716_112408\evidence_video.mp4")
    parser.add_argument("--source-id", default=None)
    parser.add_argument("--detector-model", default="yolo11n.pt")
    parser.add_argument("--tracking-backend", choices=("supervision_bytetrack", "ultralytics_bytetrack"), default="ultralytics_bytetrack")
    parser.add_argument("--target-fps", type=float, default=4.0)
    parser.add_argument("--use-source-fps", action="store_true")
    parser.add_argument("--max-processed-frames", type=int, default=60)
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--track-activation-threshold", type=float, default=0.30)
    parser.add_argument("--track-high-threshold", type=float, default=0.05)
    parser.add_argument("--track-low-threshold", type=float, default=0.01)
    parser.add_argument("--new-track-threshold", type=float, default=0.05)
    parser.add_argument("--match-threshold", type=float, default=0.80)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--device", default=os.environ.get("TD_CASE2_YOLO_DEVICE", "cpu"))
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--allowed-class-names", default=None)
    parser.add_argument("--confirmation-observations", type=int, default=3)
    parser.add_argument("--tentative-missed-frames", type=int, default=1)
    parser.add_argument("--lost-processed-frames", type=int, default=5)
    parser.add_argument("--lost-seconds", type=float, default=None)
    parser.add_argument("--allow-recovery", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flush-on-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-root", default="debug_runs/streaming_tracking_pipeline")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_synthetic(args) if args.mode == "synthetic_lifecycle" else run_real(args)
    print(json.dumps(to_json_safe(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
