from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .anpr_artifacts import AnprArtifactSink, read_selected_crop_jobs
from .anpr_metrics import build_step7_metrics
from .anpr_pipeline import SequentialAnprColourPipeline, group_jobs_by_track
from .config import FlorenceConfig, PipelineConfig, PlateDetectionConfig, Step7InferenceConfig
from .crop_selection import SelectedCropJob
from .florence_inference import FlorenceInferenceEngine
from .plate_detection import UltralyticsPlateDetectionStage
from .serialization import write_json


DEFAULT_STEP6_ARTIFACT_RUN = Path(
    r"debug_runs\streaming_tracking_pipeline\streaming_tracking_step6_existing_step5_20260718_144055"
)


class _FakePlateModel:
    def __call__(self, image: Any, **_: Any) -> list[dict[str, Any]]:
        height, width = image.shape[:2]
        return [{"bbox": [width * 0.25, height * 0.55, width * 0.75, height * 0.75], "confidence": 0.91}]


class _FakeProcessor:
    def __call__(self, text: str, images: Any, return_tensors: str) -> dict[str, Any]:
        return {"input_ids": [1], "pixel_values": [2], "prompt": text}

    def batch_decode(self, generated_ids: Any, skip_special_tokens: bool = False) -> list[str]:
        return [str(generated_ids[0])]

    def post_process_generation(self, generated_text: str, task: str, image_size: tuple[int, int]) -> dict[str, str]:
        if "VQA" in task:
            return {task: "white"}
        return {task: "MH12AB1234"}


class _FakeModel:
    def generate(self, **kwargs: Any) -> list[str]:
        prompt = str(kwargs.get("prompt", ""))
        return ["white" if "VQA" in prompt else "MH12AB1234"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_existing_path(value: str | None, *, fallback: Path | None = None) -> Path:
    candidates: list[Path] = []
    if value:
        candidates.append(Path(value).expanduser())
    if fallback is not None:
        candidates.append(fallback)
    for candidate in candidates:
        if candidate.is_absolute() and candidate.exists():
            return candidate
        rooted = (_repo_root() / candidate).resolve()
        if rooted.exists():
            return rooted
    raise FileNotFoundError("No existing path found. Checked: " + ", ".join(str(item) for item in candidates))


def _resolve_optional_path(value: str | None) -> str | None:
    if value is None or not str(value).strip():
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((_repo_root() / path).resolve())


def _step7_config_from_args(args: argparse.Namespace) -> Step7InferenceConfig:
    return Step7InferenceConfig(
        process_primary_crops=not args.skip_primary,
        process_fallback_crops=not args.skip_fallback,
        maximum_vehicle_crops_per_track=args.maximum_vehicle_crops_per_track,
        maximum_plate_candidates_per_vehicle_crop=args.maximum_plate_candidates_per_vehicle_crop,
        stop_after_first_raw_plate_text=not args.disable_stop_after_first_raw_plate_text,
        run_colour_once_per_track=not args.disable_run_colour_once_per_track,
        prefer_primary_for_colour=not args.disable_prefer_primary_for_colour,
        save_inference_inputs=args.save_inference_inputs,
        save_inference_outputs=not args.disable_save_inference_outputs,
    )


def _plate_config_from_args(args: argparse.Namespace) -> PlateDetectionConfig:
    return PlateDetectionConfig(
        enabled=not args.disable_plate_detection,
        model_path=_resolve_optional_path(args.plate_detector_model_path),
        confidence_threshold=args.plate_confidence_threshold,
        iou_threshold=args.plate_iou_threshold,
        device=args.plate_device,
        image_size=args.plate_image_size,
        max_plate_detections_per_vehicle_crop=args.maximum_plate_candidates_per_vehicle_crop,
        minimum_plate_width=args.minimum_plate_width,
        minimum_plate_height=args.minimum_plate_height,
        crop_padding_ratio=args.plate_crop_padding_ratio,
        save_plate_crops=not args.disable_save_plate_crops,
    )


def _florence_config_from_args(args: argparse.Namespace) -> FlorenceConfig:
    return FlorenceConfig(
        enabled=not args.disable_florence,
        base_model_path=_resolve_optional_path(args.florence_model_path),
        adapter_path=_resolve_optional_path(args.florence_adapter_path),
        device=args.florence_device,
        dtype=args.florence_dtype,
        local_files_only=not args.disable_local_files_only,
        trust_remote_code=args.trust_remote_code,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        do_sample=args.do_sample,
        ocr_task_prompt=args.ocr_task_prompt,
        colour_task_prompt=args.colour_task_prompt,
    )


def _synthetic_jobs(run_dir: Path) -> list[SelectedCropJob]:
    from PIL import Image, ImageDraw

    image_dir = run_dir / "synthetic_inputs"
    image_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[SelectedCropJob] = []
    for index, role in enumerate(("primary", "fallback"), start=1):
        path = image_dir / f"vehicle_{role}.jpg"
        image = Image.new("RGB", (120, 80), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((28, 48, 92, 62), fill="black")
        image.save(path)
        jobs.append(
            SelectedCropJob(
                source_id="synthetic_step7_source",
                track_id=1,
                track_generation=0,
                source_track_id="raw_1",
                object_class="car",
                lifecycle_completion_reason="synthetic",
                crop_role=role,
                crop_rank=index,
                frame_index=index,
                timestamp_sec=float(index),
                vehicle_crop_path=str(path),
                full_frame_path=None,
                selection_score=0.95 - index * 0.05,
            )
        )
    return jobs


def _run_jobs(args: argparse.Namespace, jobs: list[SelectedCropJob], *, run_dir: Path, fake_models: bool = False) -> dict[str, Any]:
    plate_config = _plate_config_from_args(args)
    florence_config = _florence_config_from_args(args)
    step7_config = _step7_config_from_args(args)
    plate_stage = UltralyticsPlateDetectionStage(
        plate_config,
        output_dir=run_dir,
        run_dir=run_dir,
        model=_FakePlateModel() if fake_models else None,
    )
    if fake_models:
        from .florence_inference import FlorenceModelBundle

        florence_engine = FlorenceInferenceEngine(
            florence_config,
            bundle=FlorenceModelBundle(model=_FakeModel(), processor=_FakeProcessor(), device="cpu"),
            run_dir=run_dir,
        )
    else:
        florence_engine = FlorenceInferenceEngine(florence_config, run_dir=run_dir)
    pipeline = SequentialAnprColourPipeline(step7_config, plate_detector=plate_stage, florence_engine=florence_engine)
    results = pipeline.process_job_groups(group_jobs_by_track(jobs))
    sink = AnprArtifactSink(run_dir)
    try:
        for result in results:
            sink.write_result(result)
        summary = build_step7_metrics(
            results,
            extra_metrics={"plate_detection": plate_stage.metrics, "florence": florence_engine.metrics},
        )
        summary.update(
            {
                "mode": args.mode,
                "run_dir": str(run_dir),
                "input_selected_crop_jobs": len(jobs),
                "config": {
                    "plate_detection": plate_config.to_dict() if hasattr(plate_config, "to_dict") else plate_config.__dict__,
                    "florence": florence_config.to_dict() if hasattr(florence_config, "to_dict") else florence_config.__dict__,
                    "step7_inference": step7_config.__dict__,
                },
            }
        )
        sink.write_summary(summary)
    finally:
        sink.close()
    write_json(run_dir / "reports" / "step7_validation_result.json", summary)
    return {"status": "passed", "mode": args.mode, "run_dir": str(run_dir), "report": summary}


def run_synthetic(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.output_root) / f"streaming_tracking_step7_synthetic_{time.strftime('%Y%m%d_%H%M%S')}"
    return _run_jobs(args, _synthetic_jobs(run_dir), run_dir=run_dir, fake_models=True)


def run_existing_step6_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    step6_run_dir = _resolve_existing_path(args.step6_run_dir, fallback=DEFAULT_STEP6_ARTIFACT_RUN)
    run_dir = Path(args.output_root) / f"streaming_tracking_step7_existing_step6_{time.strftime('%Y%m%d_%H%M%S')}"
    jobs = read_selected_crop_jobs(step6_run_dir)
    if args.max_tracks:
        allowed = {key for key in sorted({(job.source_id, job.track_id, job.track_generation) for job in jobs})[: args.max_tracks]}
        jobs = [job for job in jobs if (job.source_id, job.track_id, job.track_generation) in allowed]
    return _run_jobs(args, jobs, run_dir=run_dir, fake_models=False)


def run_real_step1_to_step7(args: argparse.Namespace) -> dict[str, Any]:
    from .run_step6_best_crop_validation import run_real as run_step6_real

    step6_args = argparse.Namespace(
        video=args.video,
        source_id=args.source_id,
        detector_model=args.detector_model,
        tracking_backend=args.tracking_backend,
        target_fps=args.target_fps,
        use_source_fps=args.use_source_fps,
        max_processed_frames=args.max_processed_frames,
        confidence=args.confidence,
        iou=args.iou,
        track_activation_threshold=args.track_activation_threshold,
        track_high_threshold=args.track_high_threshold,
        track_low_threshold=args.track_low_threshold,
        new_track_threshold=args.new_track_threshold,
        match_threshold=args.match_threshold,
        track_buffer=args.track_buffer,
        device=args.device,
        image_size=args.image_size,
        allowed_class_names=args.allowed_class_names,
        confirmation_observations=args.confirmation_observations,
        tentative_missed_frames=args.tentative_missed_frames,
        lost_processed_frames=args.lost_processed_frames,
        lost_seconds=args.lost_seconds,
        allow_recovery=args.allow_recovery,
        flush_on_eos=args.flush_on_eos,
        save_crop_images=True,
        max_candidates_per_track=args.max_candidates_per_track,
        max_observations_per_track=args.max_observations_per_track,
        retention_policy=args.retention_policy,
        crop_padding_ratio=args.vehicle_crop_padding_ratio,
        minimum_crop_width=args.minimum_vehicle_crop_width,
        minimum_crop_height=args.minimum_vehicle_crop_height,
        minimum_bbox_area_ratio=args.minimum_bbox_area_ratio,
        disable_selection=False,
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
        output_root=args.output_root,
    )
    step6_result = run_step6_real(step6_args)
    step6_run_dir = Path(step6_result["run_dir"])
    jobs = read_selected_crop_jobs(step6_run_dir)
    if args.max_tracks:
        allowed = {key for key in sorted({(job.source_id, job.track_id, job.track_generation) for job in jobs})[: args.max_tracks]}
        jobs = [job for job in jobs if (job.source_id, job.track_id, job.track_generation) in allowed]
    step7_result = _run_jobs(args, jobs, run_dir=step6_run_dir, fake_models=False)
    step7_result["upstream_step6"] = step6_result
    return step7_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate isolated streaming Step 7 ANPR and vehicle colour enrichment.")
    parser.add_argument("--mode", choices=("synthetic_step7", "existing_step6_artifacts", "real_step1_to_step7"), default="synthetic_step7")
    parser.add_argument("--output-root", default=PipelineConfig().output.output_root)
    parser.add_argument("--step6-run-dir", default=os.environ.get("TD_CASE2_STREAM_STEP7_STEP6_RUN_DIR"))
    parser.add_argument("--max-tracks", type=int, default=0)
    parser.add_argument("--plate-detector-model-path", default=os.environ.get("TD_CASE2_STREAM_PLATE_MODEL_PATH", "OCR_MUKUL/license_plate_weights.pt"))
    parser.add_argument("--plate-confidence-threshold", type=float, default=0.20)
    parser.add_argument("--plate-iou-threshold", type=float, default=0.45)
    parser.add_argument("--plate-device", default=os.environ.get("TD_CASE2_STREAM_PLATE_DEVICE", "auto"))
    parser.add_argument("--plate-image-size", type=int, default=640)
    parser.add_argument("--minimum-plate-width", type=int, default=6)
    parser.add_argument("--minimum-plate-height", type=int, default=4)
    parser.add_argument("--plate-crop-padding-ratio", type=float, default=0.08)
    parser.add_argument("--disable-plate-detection", action="store_true")
    parser.add_argument("--disable-save-plate-crops", action="store_true")
    parser.add_argument("--florence-model-path", default=os.environ.get("TD_CASE2_STREAM_FLORENCE_MODEL_PATH") or os.environ.get("TD_CASE2_FLORENCE_MODEL_PATH"))
    parser.add_argument("--florence-adapter-path", default=os.environ.get("TD_CASE2_STREAM_FLORENCE_ADAPTER_PATH") or os.environ.get("TD_CASE2_FLORENCE_ADAPTER_PATH") or "OCR_MUKUL/adaptor_florance_baseFT")
    parser.add_argument("--florence-device", default=os.environ.get("TD_CASE2_STREAM_FLORENCE_DEVICE", "auto"))
    parser.add_argument("--florence-dtype", default=os.environ.get("TD_CASE2_STREAM_FLORENCE_DTYPE", "float16"))
    parser.add_argument("--disable-local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--ocr-task-prompt", default="<OCR>")
    parser.add_argument("--colour-task-prompt", default="<VQA>What is the primary colour of the vehicle?")
    parser.add_argument("--disable-florence", action="store_true")
    parser.add_argument("--skip-primary", action="store_true")
    parser.add_argument("--skip-fallback", action="store_true")
    parser.add_argument("--maximum-vehicle-crops-per-track", type=int, default=4)
    parser.add_argument("--maximum-plate-candidates-per-vehicle-crop", type=int, default=3)
    parser.add_argument("--disable-stop-after-first-raw-plate-text", action="store_true")
    parser.add_argument("--disable-run-colour-once-per-track", action="store_true")
    parser.add_argument("--disable-prefer-primary-for-colour", action="store_true")
    parser.add_argument("--save-inference-inputs", action="store_true")
    parser.add_argument("--disable-save-inference-outputs", action="store_true")
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
    parser.add_argument("--max-candidates-per-track", type=int, default=4)
    parser.add_argument("--max-observations-per-track", type=int, default=16)
    parser.add_argument("--retention-policy", choices=("highest_preliminary_score", "uniform_temporal", "hybrid_quality_temporal"), default="hybrid_quality_temporal")
    parser.add_argument("--vehicle-crop-padding-ratio", type=float, default=0.08)
    parser.add_argument("--minimum-vehicle-crop-width", type=int, default=8)
    parser.add_argument("--minimum-vehicle-crop-height", type=int, default=8)
    parser.add_argument("--minimum-bbox-area-ratio", type=float, default=0.0005)
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.mode == "synthetic_step7":
            result = run_synthetic(args)
        elif args.mode == "existing_step6_artifacts":
            result = run_existing_step6_artifacts(args)
        else:
            result = run_real_step1_to_step7(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "mode": args.mode, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
