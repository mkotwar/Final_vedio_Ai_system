from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .bytetrack_stage import create_bytetrack_stage
from .config import (
    BestCropScoreConfig,
    BestCropSelectionConfig,
    CropCollectionConfig,
    DetectionConfig,
    SourceConfig,
    TrackLifecycleConfig,
    TrackingConfig,
)
from .crop_artifacts import CompletedTrackCropBundle, CropArtifactSink, CropImageWriter
from .crop_collector import CropCandidateCollector
from .crop_pipeline import SequentialCropCollectionPipeline
from .crop_selection import FinalBestCropSelector
from .crop_selection_artifacts import CropSelectionArtifactSink, copy_step5_artifact_context, read_completed_crop_bundles
from .crop_selection_pipeline import finalize_step6_artifacts, run_selection_for_existing_bundles
from .lifecycle import TrackLifecycleManager
from .lifecycle_pipeline import LifecycleArtifactSink
from .mock_stages import DeterministicMockDetectionStage, DeterministicMockTrackingStage
from .schemas import BoundingBox, CropCandidate, CropQualityMetrics, DetectionRecord, TrackedObject
from .serialization import to_json_safe, write_json
from .sources import SyntheticFrameSource
from .video_source import OpenCvVideoSource
from .yolo_stage import UltralyticsYoloDetectionStage


DEFAULT_STEP5_ARTIFACT_RUN = Path(
    r"debug_runs\streaming_tracking_pipeline\streaming_tracking_step5_evidence_video_ultralytics_bytetrack_20260718_142138"
)


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


def _selection_config_from_args(args: argparse.Namespace) -> BestCropSelectionConfig:
    return BestCropSelectionConfig(
        enabled=not args.disable_selection,
        primary_crop_count=args.primary_crop_count,
        keep_fallback_crop=args.keep_fallback_crop,
        minimum_primary_score=args.minimum_primary_score,
        minimum_fallback_score=args.minimum_fallback_score,
        minimum_track_observations_for_primary=args.minimum_track_observations_for_primary,
        minimum_candidates_for_primary=args.minimum_candidates_for_primary,
        minimum_temporal_separation_sec=args.minimum_temporal_separation_sec,
        minimum_frame_separation=args.minimum_frame_separation,
        maximum_bbox_overlap_similarity=args.maximum_bbox_overlap_similarity,
        primary_selection_policy=args.primary_selection_policy,
        fallback_selection_policy=args.fallback_selection_policy,
        create_previews=args.create_previews,
    )


def _candidate(
    *,
    source_id: str,
    track_id: int,
    generation: int,
    frame_index: int,
    score_hint: float = 0.8,
    path: str | None = "synthetic/crop.jpg",
    edge: bool = False,
    complete: bool = True,
    brightness: float | None = 0.5,
    sharpness: float | None = 3000.0,
    contrast: float | None = 40.0,
    width: int | None = 48,
    height: int | None = 36,
    bbox: BoundingBox | None = None,
) -> CropCandidate:
    bbox = bbox or BoundingBox(10 + frame_index, 10, 58 + frame_index, 46)
    return CropCandidate(
        source_id=source_id,
        track_id=track_id,
        track_generation=generation,
        source_track_id=f"raw_{track_id}",
        frame_index=frame_index,
        timestamp_sec=frame_index / 4.0,
        bbox=bbox,
        crop_bbox=bbox,
        full_frame_path=None,
        vehicle_crop_path=path,
        quality=CropQualityMetrics(
            detection_confidence=score_hint,
            bbox_area_ratio=0.02,
            sharpness=sharpness,
            brightness=brightness,
            contrast=contrast,
            crop_width=width,
            crop_height=height,
            crop_completeness=1.0 if complete else 0.72,
            edge_touching=edge,
            padding_clipped=not complete,
            preliminary_score=score_hint,
            combined_score=score_hint,
        ),
        class_name="car",
        detection_confidence=score_hint,
        preliminary_rank_score=score_hint,
        retention_reason="synthetic_step6",
    )


def _bundle(track_id: int, generation: int, observation_count: int, candidates: list[CropCandidate], *, reason: str = "video_ended") -> CompletedTrackCropBundle:
    return CompletedTrackCropBundle(
        source_id="synthetic_selection_source",
        track_id=track_id,
        track_generation=generation,
        source_track_id=f"raw_{track_id}",
        completion_reason=reason,
        lifecycle_status="completed",
        observation_count=observation_count,
        retained_candidate_count=len(candidates),
        candidates=list(candidates),
        metadata={"dominant_class": "car"},
    )


def _synthetic_bundles() -> list[CompletedTrackCropBundle]:
    bundles: list[CompletedTrackCropBundle] = []
    bundles.append(_bundle(1, 0, 8, [_candidate(source_id="synthetic_selection_source", track_id=1, generation=0, frame_index=i * 4, score_hint=0.95 - i * 0.03) for i in range(5)]))
    bundles.append(_bundle(2, 0, 7, [_candidate(source_id="synthetic_selection_source", track_id=2, generation=0, frame_index=i, score_hint=0.92 - i * 0.01) for i in range(5)]))
    bundles.append(_bundle(3, 0, 1, [_candidate(source_id="synthetic_selection_source", track_id=3, generation=0, frame_index=4, score_hint=0.68)]))
    bundles.append(_bundle(4, 0, 4, [_candidate(source_id="synthetic_selection_source", track_id=4, generation=0, frame_index=8, score_hint=0.88, edge=True)]))
    bundles.append(_bundle(5, 0, 4, [_candidate(source_id="synthetic_selection_source", track_id=5, generation=0, frame_index=8, score_hint=0.88, complete=False)]))
    bundles.append(_bundle(6, 0, 4, [_candidate(source_id="synthetic_selection_source", track_id=6, generation=0, frame_index=8, score_hint=0.8, brightness=0.04)]))
    bundles.append(_bundle(7, 0, 4, [_candidate(source_id="synthetic_selection_source", track_id=7, generation=0, frame_index=8, score_hint=0.8, brightness=0.97)]))
    bundles.append(_bundle(8, 0, 4, [_candidate(source_id="synthetic_selection_source", track_id=8, generation=0, frame_index=8, score_hint=0.8, sharpness=0.0)]))
    bundles.append(_bundle(9, 0, 4, [_candidate(source_id="synthetic_selection_source", track_id=9, generation=0, frame_index=8, score_hint=0.75, brightness=None, sharpness=None, contrast=None)]))
    bundles.append(_bundle(10, 0, 4, [_candidate(source_id="synthetic_selection_source", track_id=10, generation=0, frame_index=12, score_hint=0.8), _candidate(source_id="synthetic_selection_source", track_id=10, generation=0, frame_index=16, score_hint=0.8)]))
    dup = _candidate(source_id="synthetic_selection_source", track_id=11, generation=0, frame_index=12, score_hint=0.82)
    bundles.append(_bundle(11, 0, 4, [dup, dup]))
    bundles.append(_bundle(12, 0, 4, [_candidate(source_id="synthetic_selection_source", track_id=12, generation=0, frame_index=8, score_hint=0.1, path=None)]))
    bundles.append(_bundle(1, 1, 5, [_candidate(source_id="synthetic_selection_source", track_id=1, generation=1, frame_index=20, score_hint=0.9), _candidate(source_id="synthetic_selection_source", track_id=1, generation=1, frame_index=24, score_hint=0.86)]))
    bundles.append(_bundle(13, 0, 5, [_candidate(source_id="synthetic_selection_source", track_id=13, generation=0, frame_index=0, score_hint=0.9), _candidate(source_id="synthetic_selection_source", track_id=13, generation=0, frame_index=8, score_hint=0.88)]))
    bundles.append(_bundle(14, 0, 5, [_candidate(source_id="synthetic_selection_source", track_id=14, generation=0, frame_index=0, score_hint=0.9)]))
    bundles.append(_bundle(15, 0, 5, []))
    return bundles


def run_synthetic(args: argparse.Namespace) -> dict[str, Any]:
    run_id = f"streaming_tracking_step6_synthetic_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_root) / run_id
    selector = FinalBestCropSelector(_selection_config_from_args(args), BestCropScoreConfig())
    sink = CropSelectionArtifactSink(run_dir, create_previews=args.create_previews)
    report = run_selection_for_existing_bundles(
        run_id=run_id,
        mode="synthetic_best_crop_selection",
        bundles=_synthetic_bundles(),
        selector=selector,
        sink=sink,
        source_path="synthetic",
        source_id="synthetic_selection_source",
        tracking_backend="none",
    )
    disabled_selector = FinalBestCropSelector(BestCropSelectionConfig(enabled=False), BestCropScoreConfig())
    disabled_result = disabled_selector.select(_synthetic_bundles()[0])
    finalize_step6_artifacts(run_dir, report)
    payload = report.to_dict()
    payload["disabled_selector_check"] = disabled_result.selection_status
    write_json(run_dir / "reports" / "step6_validation_result.json", payload)
    return {"status": "passed", "mode": "synthetic_best_crop_selection", "run_dir": str(run_dir), "report": payload}


def run_existing_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    step5_run_dir = _resolve_existing_path(args.step5_run_dir, fallback_names=(str(DEFAULT_STEP5_ARTIFACT_RUN),))
    run_id = f"streaming_tracking_step6_existing_step5_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_root) / run_id
    copy_step5_artifact_context(step5_run_dir, run_dir)
    bundles = read_completed_crop_bundles(step5_run_dir)
    selector = FinalBestCropSelector(_selection_config_from_args(args), BestCropScoreConfig())
    sink = CropSelectionArtifactSink(run_dir, create_previews=args.create_previews)
    report = run_selection_for_existing_bundles(
        run_id=run_id,
        mode="existing_step5_artifacts",
        bundles=bundles,
        selector=selector,
        sink=sink,
        source_path=str(step5_run_dir),
        source_id=bundles[0].source_id if bundles else "",
        tracking_backend="artifact_replay",
    )
    finalize_step6_artifacts(run_dir, report)
    payload = report.to_dict()
    write_json(run_dir / "reports" / "step6_validation_result.json", payload)
    return {"status": "passed", "mode": "existing_step5_artifacts", "run_dir": str(run_dir), "report": payload}


def _lifecycle_config_from_args(args: argparse.Namespace) -> TrackLifecycleConfig:
    return TrackLifecycleConfig(
        minimum_confirmation_observations=args.confirmation_observations,
        maximum_tentative_missed_frames=args.tentative_missed_frames,
        maximum_lost_processed_frames=args.lost_processed_frames,
        maximum_lost_seconds=args.lost_seconds,
        allow_recovery=args.allow_recovery,
        flush_on_end_of_stream=args.flush_on_eos,
    )


def _crop_collection_config_from_args(args: argparse.Namespace) -> CropCollectionConfig:
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


def run_real(args: argparse.Namespace) -> dict[str, Any]:
    video_path = _resolve_existing_path(args.video, fallback_names=())
    model_path = _resolve_existing_path(args.detector_model, fallback_names=("yolo11n.pt", "yolov8n.pt", "yolo11m.pt"))
    run_id = f"streaming_tracking_step6_{video_path.stem}_{args.tracking_backend}_{time.strftime('%Y%m%d_%H%M%S')}"
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
    crop_pipeline = SequentialCropCollectionPipeline(
        run_id=run_id,
        source=source,
        detection_stage=UltralyticsYoloDetectionStage(detection_config),
        tracking_stage=create_bytetrack_stage(tracking_config, source_fps=source_fps),
        lifecycle_manager=TrackLifecycleManager(_lifecycle_config_from_args(args)),
        crop_collector=CropCandidateCollector(_crop_collection_config_from_args(args), image_writer=CropImageWriter(run_dir, enabled=args.save_crop_images, overwrite=True)),
        lifecycle_sink=LifecycleArtifactSink(run_dir),
        crop_sink=CropArtifactSink(run_dir),
        source_path=str(video_path),
        detector_model_path=str(model_path),
        tracking_backend=args.tracking_backend,
    )
    crop_report = crop_pipeline.run()
    selector = FinalBestCropSelector(_selection_config_from_args(args), BestCropScoreConfig())
    sink = CropSelectionArtifactSink(run_dir, create_previews=args.create_previews)
    report = run_selection_for_existing_bundles(
        run_id=run_id,
        mode="real_best_crop_selection",
        bundles=crop_pipeline.completed_bundles,
        selector=selector,
        sink=sink,
        source_path=str(video_path),
        source_id=source.source_id,
        tracking_backend=args.tracking_backend,
    )
    object.__setattr__(report, "crop_collection_report", crop_report.to_dict())
    finalize_step6_artifacts(run_dir, report)
    payload = report.to_dict()
    write_json(run_dir / "reports" / "step6_validation_result.json", payload)
    return {"status": "passed", "mode": "real_best_crop_selection", "run_dir": str(run_dir), "report": payload}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("synthetic_best_crop_selection", "existing_step5_artifacts", "real_best_crop_selection"), default="synthetic_best_crop_selection")
    parser.add_argument("--step5-run-dir", default=str(DEFAULT_STEP5_ARTIFACT_RUN))
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
    parser.add_argument("--disable-selection", action="store_true")
    parser.add_argument("--primary-crop-count", type=int, default=3)
    parser.add_argument("--keep-fallback-crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--minimum-primary-score", type=float, default=0.45)
    parser.add_argument("--minimum-fallback-score", type=float, default=0.20)
    parser.add_argument("--minimum-track-observations-for-primary", type=int, default=3)
    parser.add_argument("--minimum-candidates-for-primary", type=int, default=2)
    parser.add_argument("--minimum-temporal-separation-sec", type=float, default=0.50)
    parser.add_argument("--minimum-frame-separation", type=int, default=2)
    parser.add_argument("--maximum-bbox-overlap-similarity", type=float, default=0.92)
    parser.add_argument("--primary-selection-policy", choices=("quality_only", "quality_with_temporal_diversity", "quality_with_visual_diversity", "hybrid"), default="hybrid")
    parser.add_argument("--fallback-selection-policy", choices=("best_available", "earliest_valid", "largest_valid", "sharpest_valid"), default="best_available")
    parser.add_argument("--create-previews", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-root", default="debug_runs/streaming_tracking_pipeline")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "synthetic_best_crop_selection":
        result = run_synthetic(args)
    elif args.mode == "existing_step5_artifacts":
        result = run_existing_artifacts(args)
    else:
        result = run_real(args)
    print(json.dumps(to_json_safe(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
