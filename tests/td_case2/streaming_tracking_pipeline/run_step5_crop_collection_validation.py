from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .bytetrack_stage import create_bytetrack_stage
from .config import CropCollectionConfig, DetectionConfig, SourceConfig, TrackLifecycleConfig, TrackingConfig
from .crop_artifacts import CropArtifactSink, CropImageWriter
from .crop_collector import CropCandidateCollector
from .crop_pipeline import SequentialCropCollectionPipeline, finalize_step5_artifacts
from .lifecycle import TrackLifecycleManager
from .lifecycle_pipeline import LifecycleArtifactSink
from .mock_stages import DeterministicMockDetectionStage, DeterministicMockTrackingStage
from .schemas import BoundingBox, DetectionRecord, TrackedObject
from .serialization import to_json_safe, write_json
from .sources import SyntheticFrameSource
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


def _crop_config_from_args(args: argparse.Namespace) -> CropCollectionConfig:
    return CropCollectionConfig(
        save_crop_images=args.save_crop_images,
        max_candidates_per_track=args.max_candidates_per_track,
        max_observations_per_track=args.max_observations_per_track,
        retention_policy=args.retention_policy,
        padding_ratio=args.crop_padding_ratio,
        minimum_crop_width=args.minimum_crop_width,
        minimum_crop_height=args.minimum_crop_height,
        minimum_bbox_area_ratio=args.minimum_bbox_area_ratio,
    )


def _synthetic_frame(frame_index: int) -> Any:
    import numpy as np

    frame = np.zeros((96, 128, 3), dtype=np.uint8)
    frame[:, :, 0] = 40 + (frame_index * 7) % 90
    frame[:, :, 1] = 80
    frame[:, :, 2] = 150
    frame[20:70, 20 + frame_index : 62 + frame_index, :] = 230
    frame[42:58, 70:112, :] = 30 + frame_index * 10
    return frame


def _synthetic_track(track_id: int, frame_index: int, timestamp_sec: float, bbox: BoundingBox, class_name: str = "car") -> TrackedObject:
    class_id = {"person": 0, "car": 2, "truck": 7, "bus": 5}.get(class_name, 2)
    return TrackedObject(
        track_id=track_id,
        source_track_id=f"synthetic_{track_id}",
        bbox=bbox,
        confidence=0.72 + min(frame_index, 6) * 0.03,
        class_id=class_id,
        class_name=class_name,
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
    )


def _synthetic_tracks(packet: Any) -> list[TrackedObject]:
    timestamp = packet.timestamp_sec
    index = packet.frame_index
    if index in {0, 1, 2, 3, 4, 5}:
        return [_synthetic_track(1, index, timestamp, BoundingBox(20 + index, 20, 62 + index, 70))]
    if index in {7, 8, 9, 10}:
        return [_synthetic_track(1, index, timestamp, BoundingBox(28 + index, 22, 70 + index, 72), "truck")]
    return []


def run_synthetic(args: argparse.Namespace) -> dict[str, Any]:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"streaming_tracking_step5_synthetic_{timestamp}"
    run_dir = Path(args.output_root) / run_id
    source = SyntheticFrameSource(
        source_id="synthetic_crop_source",
        total_frames=12,
        source_fps=2.0,
        frame_width=128,
        frame_height=96,
        use_source_fps=True,
        frame_factory=_synthetic_frame,
    )
    detector = DeterministicMockDetectionStage(detection_factory=lambda packet: [
        DetectionRecord(
            bbox=track.bbox,
            confidence=track.confidence,
            class_id=track.class_id,
            class_name=track.class_name,
        )
        for track in _synthetic_tracks(packet)
    ])
    tracker = DeterministicMockTrackingStage(track_factory=_synthetic_tracks)
    manager = TrackLifecycleManager(_lifecycle_config_from_args(args))
    lifecycle_sink = LifecycleArtifactSink(run_dir)
    crop_sink = CropArtifactSink(run_dir)
    crop_writer = CropImageWriter(run_dir, enabled=args.save_crop_images, overwrite=True)
    collector = CropCandidateCollector(_crop_config_from_args(args), image_writer=crop_writer)
    pipeline = SequentialCropCollectionPipeline(
        run_id=run_id,
        source=source,
        detection_stage=detector,
        tracking_stage=tracker,
        lifecycle_manager=manager,
        crop_collector=collector,
        lifecycle_sink=lifecycle_sink,
        crop_sink=crop_sink,
        source_path="synthetic",
        detector_model_path=None,
        tracking_backend="deterministic_mock",
    )
    report = pipeline.run()
    finalize_step5_artifacts(run_dir, report, {"source_type": "synthetic", "total_frames": source.total_frames, "source_fps": source.source_fps})
    return {"status": "passed", "mode": "synthetic_crop_collection", "run_dir": str(run_dir), "report": report.to_dict()}


def run_real(args: argparse.Namespace) -> dict[str, Any]:
    video_path = _resolve_existing_path(args.video, fallback_names=())
    model_path = _resolve_existing_path(args.detector_model, fallback_names=("yolo11n.pt", "yolov8n.pt", "yolo11m.pt"))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"streaming_tracking_step5_{video_path.stem}_{args.tracking_backend}_{timestamp}"
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
    lifecycle_sink = LifecycleArtifactSink(run_dir)
    crop_sink = CropArtifactSink(run_dir)
    crop_writer = CropImageWriter(run_dir, enabled=args.save_crop_images, overwrite=True)
    collector = CropCandidateCollector(_crop_config_from_args(args), image_writer=crop_writer)
    pipeline = SequentialCropCollectionPipeline(
        run_id=run_id,
        source=source,
        detection_stage=detector,
        tracking_stage=tracker,
        lifecycle_manager=manager,
        crop_collector=collector,
        lifecycle_sink=lifecycle_sink,
        crop_sink=crop_sink,
        source_path=str(video_path),
        detector_model_path=str(model_path),
        tracking_backend=args.tracking_backend,
    )
    report = pipeline.run()
    finalize_step5_artifacts(run_dir, report, source_metadata)
    return {"status": "passed", "mode": "real_tracking_crop_collection", "run_dir": str(run_dir), "report": report.to_dict()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("synthetic_crop_collection", "real_tracking_crop_collection"), default="synthetic_crop_collection")
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
    parser.add_argument("--save-crop-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-candidates-per-track", type=int, default=4)
    parser.add_argument("--max-observations-per-track", type=int, default=16)
    parser.add_argument("--retention-policy", choices=("highest_preliminary_score", "uniform_temporal", "hybrid_quality_temporal"), default="hybrid_quality_temporal")
    parser.add_argument("--crop-padding-ratio", type=float, default=0.08)
    parser.add_argument("--minimum-crop-width", type=int, default=8)
    parser.add_argument("--minimum-crop-height", type=int, default=8)
    parser.add_argument("--minimum-bbox-area-ratio", type=float, default=0.0005)
    parser.add_argument("--output-root", default="debug_runs/streaming_tracking_pipeline")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_synthetic(args) if args.mode == "synthetic_crop_collection" else run_real(args)
    write_json(Path(result["run_dir"]) / "reports" / "step5_validation_result.json", result)
    print(json.dumps(to_json_safe(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
