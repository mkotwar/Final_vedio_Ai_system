"""Run Step 3 real sequential YOLO -> ByteTrack validation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from .bytetrack_stage import create_bytetrack_stage
from .config import DetectionConfig, OutputConfig, SourceConfig, TrackingConfig
from .real_pipeline import RealSequentialTrackingPipeline, Step3ArtifactSink, finalize_step3_artifacts
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
    joined = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"No existing path found. Checked: {joined}")


def _find_default_video() -> Path:
    video_dir = _repo_root() / "data" / "videos"
    extensions = {".mp4", ".avi", ".mov", ".mkv"}
    candidates = sorted(
        (path for path in video_dir.glob("*") if path.suffix.lower() in extensions),
        key=lambda path: (path.stat().st_size, path.name),
    )
    if not candidates:
        raise FileNotFoundError(f"No local test videos found under {video_dir}")
    return candidates[0]


def _split_names(raw_value: str | None) -> tuple[str, ...]:
    if raw_value is None:
        return DetectionConfig().allowed_class_names
    return tuple(item.strip() for item in raw_value.split(",") if item.strip())


def _build_run_id(video_path: Path, backend: str) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"step3_real_{video_path.stem}_{backend}_{timestamp}"


def run_backend_once(
    *,
    backend: str,
    args: argparse.Namespace,
    video_path: Path,
    model_path: Path,
    parent_run_dir: Path | None = None,
) -> dict[str, Any]:
    run_id = _build_run_id(video_path, backend)
    run_dir = (parent_run_dir / backend) if parent_run_dir is not None else Path(args.output_root) / run_id
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
        backend=backend,
        lost_track_buffer=args.track_buffer,
        track_activation_threshold=args.track_activation_threshold,
        track_high_threshold=args.track_high_threshold,
        track_low_threshold=args.track_low_threshold,
        new_track_threshold=args.new_track_threshold,
        match_threshold=args.match_threshold,
    )
    detector = UltralyticsYoloDetectionStage(detection_config)
    tracker = create_bytetrack_stage(tracking_config, source_fps=source_fps)
    output_config = OutputConfig(
        output_root=str(args.output_root),
        save_annotated_video=args.save_annotated_video,
        annotated_video_fps=args.annotated_video_fps,
    )
    sink = Step3ArtifactSink(
        run_dir,
        save_annotated_video=output_config.save_annotated_video,
        annotated_video_fps=output_config.annotated_video_fps or source_config.target_processing_fps,
    )
    pipeline = RealSequentialTrackingPipeline(
        run_id=run_id,
        source=source,
        detection_stage=detector,
        tracking_stage=tracker,
        sink=sink,
        source_path=str(video_path),
        detector_model_path=str(model_path),
        tracking_backend=backend,
        expected_physical_objects=args.expected_physical_objects,
    )
    report = pipeline.run()
    finalize_step3_artifacts(run_dir, report, source_metadata)
    return {
        "status": "passed",
        "backend": backend,
        "run_dir": str(run_dir),
        "report": report.to_dict(),
    }


def _run_backend_with_failure_capture(
    *,
    backend: str,
    args: argparse.Namespace,
    video_path: Path,
    model_path: Path,
    parent_run_dir: Path | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    try:
        return run_backend_once(
            backend=backend,
            args=args,
            video_path=video_path,
            model_path=model_path,
            parent_run_dir=parent_run_dir,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "backend": backend,
            "run_dir": str(parent_run_dir / backend) if parent_run_dir is not None else None,
            "runtime_sec": round(time.perf_counter() - started_at, 6),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=None, help="Local video path. Defaults to the smallest video in data/videos.")
    parser.add_argument("--source-id", default=None, help="Stable source ID for emitted packets.")
    parser.add_argument("--detector-model", default=None, help="YOLO model path. Defaults to local repo YOLO weights.")
    parser.add_argument(
        "--tracking-backend",
        choices=("supervision_bytetrack", "ultralytics_bytetrack", "both"),
        default="ultralytics_bytetrack",
        help="ByteTrack backend to validate.",
    )
    parser.add_argument("--target-fps", type=float, default=5.0, help="Target processing FPS.")
    parser.add_argument("--use-source-fps", action="store_true", help="Process every source frame.")
    parser.add_argument("--max-processed-frames", type=int, default=300, help="Limit emitted/processed frames.")
    parser.add_argument("--confidence", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO IoU threshold.")
    parser.add_argument("--track-activation-threshold", type=float, default=0.30, help="Supervision ByteTrack activation threshold.")
    parser.add_argument("--track-high-threshold", type=float, default=0.30, help="Ultralytics ByteTrack high threshold.")
    parser.add_argument("--track-low-threshold", type=float, default=0.10, help="Ultralytics ByteTrack low threshold.")
    parser.add_argument("--new-track-threshold", type=float, default=0.30, help="Ultralytics ByteTrack new-track threshold.")
    parser.add_argument("--match-threshold", type=float, default=0.80, help="ByteTrack matching threshold.")
    parser.add_argument("--track-buffer", type=int, default=30, help="ByteTrack lost-track buffer.")
    parser.add_argument("--device", default=os.environ.get("TD_CASE2_YOLO_DEVICE", "auto"), help="YOLO device, e.g. cpu/cuda/auto.")
    parser.add_argument("--image-size", type=int, default=None, help="Optional YOLO inference image size.")
    parser.add_argument(
        "--allowed-class-names",
        default=None,
        help="Comma-separated class-name allow list. Default is the Step 3 vehicle/person list.",
    )
    parser.add_argument("--expected-physical-objects", type=int, default=None, help="Optional ID stability baseline.")
    parser.add_argument("--output-root", default="debug_runs/streaming_tracking_pipeline", help="Artifact root.")
    parser.add_argument("--save-annotated-video", action="store_true", help="Write 04_visualization/tracked_video.mp4.")
    parser.add_argument("--annotated-video-fps", type=float, default=None, help="Annotated output FPS.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    video_path = _resolve_existing_path(args.video, fallback_names=()) if args.video else _find_default_video()
    model_path = _resolve_existing_path(args.detector_model, fallback_names=("yolo11n.pt", "yolov8n.pt", "yolo11m.pt"))
    backends = (
        ("supervision_bytetrack", "ultralytics_bytetrack")
        if args.tracking_backend == "both"
        else (args.tracking_backend,)
    )
    parent_run_dir: Path | None = None
    if args.tracking_backend == "both":
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        parent_run_dir = Path(args.output_root) / f"step3_real_backend_comparison_{video_path.stem}_{timestamp}"
        parent_run_dir.mkdir(parents=True, exist_ok=True)

    results = [
        _run_backend_with_failure_capture(
            backend=backend,
            args=args,
            video_path=video_path,
            model_path=model_path,
            parent_run_dir=parent_run_dir,
        )
        for backend in backends
    ]
    comparison = {
        "video_path": str(video_path),
        "detector_model_path": str(model_path),
        "target_fps": args.target_fps,
        "max_processed_frames": args.max_processed_frames,
        "results": results,
    }
    if parent_run_dir is not None:
        write_json(parent_run_dir / "reports" / "backend_comparison_report.json", comparison)
    print(json.dumps(to_json_safe(comparison), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if any(result["status"] == "passed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
