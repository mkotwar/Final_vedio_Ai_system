from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .anpr_job_eligibility import AnprJobEligibilityConfig, filter_anpr_eligible_jobs
from .bytetrack_stage import create_bytetrack_stage
from .class_normalization import normalize_model_names
from .config import (
    BestCropScoreConfig,
    BestCropSelectionConfig,
    CropCollectionConfig,
    DetectionConfig,
    FlorenceConfig,
    PipelineConfig,
    PlateDetectionConfig,
    PlateDiagnosticConfig,
    TrackLifecycleConfig,
    TrackingConfig,
)
from .crop_artifacts import CropArtifactSink, CropImageWriter
from .crop_collector import CropCandidateCollector
from .crop_pipeline import SequentialCropCollectionPipeline, finalize_step5_artifacts
from .crop_selection import FinalBestCropSelector, SelectedCropJob
from .crop_selection_artifacts import CropSelectionArtifactSink
from .crop_selection_pipeline import finalize_step6_artifacts, run_selection_for_existing_bundles
from .lifecycle import TrackLifecycleManager
from .lifecycle_pipeline import LifecycleArtifactSink
from .multi_model_detection import CombinedSequentialDetectionStage
from .person_clothing_colour import analyse_person_clothing_for_selected_sets, build_person_clothing_colour_summary
from .plate_detection import UltralyticsPlateDetectionStage
from .plate_diagnostic_artifacts import PlateDiagnosticArtifactSink
from .plate_diagnostic_metrics import build_plate_diagnostic_metrics
from .plate_diagnostics import TrackPlateDiagnosticResult, model_class_names
from .plate_retry import BoundedPlateRetryController, group_selected_jobs_by_track
from .run_anpr_video_10fps_validation import (
    DEFAULT_ALLOWED_ANPR_CLASSES,
    DEFAULT_FLORENCE_ADAPTER,
    DEFAULT_FLORENCE_MODEL,
    DEFAULT_PLATE_MODEL,
    DEFAULT_VEHICLE_MODEL,
    _build_device_plan,
    _clear_cuda_cache,
    _colour_job,
    _cuda_available,
    _gpu_name,
    _jobs_from_selected_sets,
    _peak_vram_mb,
    _preload_florence_engine,
    _preload_plate_stage,
    _preload_vehicle_stage,
    _release_model,
    _reset_cuda_peak_memory,
    _resolve_optional_existing_path,
    _resolve_required_path,
    _resolve_vehicle_model,
    _split_csv,
    _write_anpr_jsonl,
)
from .run_step8_plate_validation import run as run_step8
from .run_step9_searchable_object_records import run as run_step9
from .run_step10_search_validation import main as run_step10_main
from .serialization import read_json, read_jsonl, to_json_safe, write_json, write_jsonl
from .video_source import OpenCvVideoSource
from .yolo_stage import UltralyticsYoloDetectionStage


DEFAULT_VIDEO = r"C:\Mukul K\mk\test_video\anpr_test_5min.mp4"
DEFAULT_PERSON_MODEL = r"object\Person_detection.pt"
SUPPORTED_CLASS_LABELS = ("person", "3Wheeler", "bus", "car", "motorcycle", "truck")


def _run_dir(output_root: str, video_path: Path) -> Path:
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in video_path.stem)
    return Path(output_root) / f"streaming_tracking_combined_{safe_stem}_{time.strftime('%Y%m%d_%H%M%S')}"


def _job_key(job: SelectedCropJob) -> tuple[str, int, int]:
    return (job.source_id, job.track_id, job.track_generation)


def _object_group(class_name: str | None) -> str:
    return "person" if str(class_name or "").lower() == "person" else "vehicle"


def _vehicle_jobs(jobs: list[SelectedCropJob]) -> list[SelectedCropJob]:
    return [job for job in jobs if _object_group(job.object_class) == "vehicle"]


def _copy_alias(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one combined vehicle + person streaming tracking validation pipeline.")
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--source-id", default=None)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--full-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-processed-frames", type=int, default=600)
    parser.add_argument("--start-sec", type=float, default=None)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument("--tracking-backend", default="ultralytics_bytetrack", choices=["ultralytics_bytetrack", "supervision_bytetrack", "passthrough"])
    parser.add_argument("--vehicle-detector-model", default=DEFAULT_VEHICLE_MODEL)
    parser.add_argument("--vehicle-detector-fallback-model", default=DEFAULT_VEHICLE_MODEL)
    parser.add_argument("--person-detector-model", default=DEFAULT_PERSON_MODEL)
    parser.add_argument("--plate-detector-model", default=DEFAULT_PLATE_MODEL)
    parser.add_argument("--florence-model", default=DEFAULT_FLORENCE_MODEL)
    parser.add_argument("--florence-adapter", default=DEFAULT_FLORENCE_ADAPTER)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--vehicle-device", default=None)
    parser.add_argument("--person-device", default=None)
    parser.add_argument("--plate-device", default=None)
    parser.add_argument("--florence-device", default=None)
    parser.add_argument("--florence-dtype", default="auto", choices=["auto", "float16", "float32", "bfloat16"])
    parser.add_argument("--florence-load-in-4bit", action="store_true", default=False)
    parser.add_argument("--florence-load-in-8bit", action="store_true", default=False)
    parser.add_argument("--mode", default="both", choices=["both", "vehicles_only", "persons_only"])
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--person-confidence", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--track-activation-threshold", type=float, default=0.15)
    parser.add_argument("--track-high-threshold", type=float, default=0.15)
    parser.add_argument("--track-low-threshold", type=float, default=0.05)
    parser.add_argument("--new-track-threshold", type=float, default=0.15)
    parser.add_argument("--match-threshold", type=float, default=0.80)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--confirmation-observations", type=int, default=3)
    parser.add_argument("--tentative-missed-frames", type=int, default=1)
    parser.add_argument("--lost-processed-frames", type=int, default=5)
    parser.add_argument("--save-crop-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-candidates-per-track", type=int, default=8)
    parser.add_argument("--max-observations-per-track", type=int, default=64)
    parser.add_argument("--crop-padding-ratio", type=float, default=0.08)
    parser.add_argument("--minimum-crop-width", type=int, default=8)
    parser.add_argument("--minimum-crop-height", type=int, default=8)
    parser.add_argument("--minimum-bbox-area-ratio", type=float, default=0.0005)
    parser.add_argument("--primary-crop-count", type=int, default=3)
    parser.add_argument("--keep-fallback-crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--minimum-primary-score", type=float, default=0.35)
    parser.add_argument("--minimum-fallback-score", type=float, default=0.10)
    parser.add_argument("--minimum-track-observations-for-primary", type=int, default=2)
    parser.add_argument("--minimum-candidates-for-primary", type=int, default=1)
    parser.add_argument("--minimum-temporal-separation-sec", type=float, default=0.50)
    parser.add_argument("--minimum-frame-separation", type=int, default=2)
    parser.add_argument("--normal-plate-confidence", type=float, default=0.25)
    parser.add_argument("--diagnostic-thresholds", default="0.25,0.15,0.10,0.05")
    parser.add_argument("--minimum-plate-width", type=int, default=6)
    parser.add_argument("--minimum-plate-height", type=int, default=4)
    parser.add_argument("--max-attempts-per-track", type=int, default=4)
    parser.add_argument("--minimum-anpr-vehicle-crop-width", type=int, default=64)
    parser.add_argument("--minimum-anpr-vehicle-crop-height", type=int, default=32)
    parser.add_argument("--minimum-anpr-vehicle-crop-area", type=int, default=2048)
    parser.add_argument("--allowed-anpr-classes", default=DEFAULT_ALLOWED_ANPR_CLASSES + ",3wheeler")
    parser.add_argument("--save-annotations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-rejected-plate-crops", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-after-first-non-empty-ocr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-root", default="debug_runs")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_total = time.perf_counter()
    env_config = PipelineConfig.from_env()
    if args.florence_load_in_4bit or args.florence_load_in_8bit:
        raise ValueError("Quantization is disabled; do not pass 4-bit or 8-bit flags.")
    args.vehicle_device = args.vehicle_device or args.device
    args.person_device = args.person_device or args.device
    args.plate_device = args.plate_device or args.device
    args.florence_device = args.florence_device or args.device

    video_path = _resolve_required_path(args.video, "video")
    person_model = _resolve_required_path(args.person_detector_model, "person detector model")
    plate_model = _resolve_required_path(args.plate_detector_model, "plate detector model")
    florence_model = _resolve_optional_existing_path(env_config.florence.base_model_path or args.florence_model)
    florence_adapter = _resolve_optional_existing_path(env_config.florence.adapter_path or args.florence_adapter)
    vehicle_model, vehicle_fallback_reason = _resolve_vehicle_model(args.vehicle_detector_model, args.vehicle_detector_fallback_model)
    cuda = _cuda_available()
    gpu_name = _gpu_name()
    _reset_cuda_peak_memory()
    device_plan = _build_device_plan(args, cuda_available=cuda, gpu_name=gpu_name)
    run_dir = _run_dir(args.output_root, video_path)
    max_processed_frames = None if args.full_video else args.max_processed_frames

    source = OpenCvVideoSource(
        video_path,
        source_id=args.source_id or video_path.stem,
        target_processing_fps=args.target_fps,
        max_processed_frames=max_processed_frames,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
    )
    source.open()
    source_metadata = source.metadata_report()
    source_fps = source.source_fps
    source.close()
    source.reset()

    vehicle_stage = None
    person_stage = None
    if args.mode in {"both", "vehicles_only"}:
        vehicle_config = DetectionConfig(
            model_path=str(vehicle_model),
            confidence_threshold=args.confidence,
            iou_threshold=args.iou,
            allowed_class_names=("3Wheeler", "bus", "car", "motorcycle", "truck"),
            device=device_plan.actual_vehicle_device,
            image_size=args.image_size,
        )
        vehicle_stage, device_plan = _preload_vehicle_stage(vehicle_config, plan=device_plan)
    if args.mode in {"both", "persons_only"}:
        person_config = DetectionConfig(
            model_path=str(person_model),
            confidence_threshold=args.person_confidence,
            iou_threshold=args.iou,
            allowed_class_names=("person",),
            device=args.person_device if args.person_device != "auto" else device_plan.actual_vehicle_device,
            image_size=args.image_size,
        )
        person_stage = UltralyticsYoloDetectionStage(person_config)
        _ = person_stage.model

    detection_stage = CombinedSequentialDetectionStage(vehicle_stage, person_stage)
    tracking_config = TrackingConfig(
        backend=args.tracking_backend,
        track_activation_threshold=args.track_activation_threshold,
        lost_track_buffer=args.track_buffer,
        track_high_threshold=args.track_high_threshold,
        track_low_threshold=args.track_low_threshold,
        new_track_threshold=args.new_track_threshold,
        match_threshold=args.match_threshold,
    )
    crop_pipeline = SequentialCropCollectionPipeline(
        run_id=run_dir.name,
        source=source,
        detection_stage=detection_stage,
        tracking_stage=create_bytetrack_stage(tracking_config, source_fps=source_fps),
        lifecycle_manager=TrackLifecycleManager(
            TrackLifecycleConfig(
                minimum_confirmation_observations=args.confirmation_observations,
                maximum_tentative_missed_frames=args.tentative_missed_frames,
                maximum_lost_processed_frames=args.lost_processed_frames,
            )
        ),
        crop_collector=CropCandidateCollector(
            CropCollectionConfig(
                save_crop_images=args.save_crop_images,
                max_candidates_per_track=args.max_candidates_per_track,
                max_observations_per_track=args.max_observations_per_track,
                padding_ratio=args.crop_padding_ratio,
                minimum_crop_width=args.minimum_crop_width,
                minimum_crop_height=args.minimum_crop_height,
                minimum_bbox_area_ratio=args.minimum_bbox_area_ratio,
            ),
            image_writer=CropImageWriter(run_dir, enabled=args.save_crop_images, overwrite=True),
        ),
        lifecycle_sink=LifecycleArtifactSink(run_dir),
        crop_sink=CropArtifactSink(run_dir),
        source_path=str(video_path),
        detector_model_path=f"{vehicle_model};{person_model}",
        tracking_backend=args.tracking_backend,
    )
    crop_report = crop_pipeline.run()
    finalize_step5_artifacts(run_dir, crop_report, source_metadata)

    selector = FinalBestCropSelector(
        BestCropSelectionConfig(
            primary_crop_count=args.primary_crop_count,
            keep_fallback_crop=args.keep_fallback_crop,
            minimum_primary_score=args.minimum_primary_score,
            minimum_fallback_score=args.minimum_fallback_score,
            minimum_track_observations_for_primary=args.minimum_track_observations_for_primary,
            minimum_candidates_for_primary=args.minimum_candidates_for_primary,
            minimum_temporal_separation_sec=args.minimum_temporal_separation_sec,
            minimum_frame_separation=args.minimum_frame_separation,
        ),
        BestCropScoreConfig(),
    )
    selection_report = run_selection_for_existing_bundles(
        run_id=run_dir.name,
        mode="combined_vehicle_person",
        bundles=crop_pipeline.completed_bundles,
        selector=selector,
        sink=CropSelectionArtifactSink(run_dir, create_previews=True),
        source_path=str(video_path),
        source_id=source.source_id,
        tracking_backend=args.tracking_backend,
    )
    object.__setattr__(selection_report, "crop_collection_report", crop_report.to_dict())
    finalize_step6_artifacts(run_dir, selection_report)

    all_jobs = _jobs_from_selected_sets(selection_report.selected_track_crop_sets)
    person_clothing_results = analyse_person_clothing_for_selected_sets(selection_report.selected_track_crop_sets, output_dir=run_dir)
    vehicle_jobs = _vehicle_jobs(all_jobs)
    eligibility_config = AnprJobEligibilityConfig(
        allowed_anpr_classes=_split_csv(args.allowed_anpr_classes),
        minimum_anpr_vehicle_crop_width=args.minimum_anpr_vehicle_crop_width,
        minimum_anpr_vehicle_crop_height=args.minimum_anpr_vehicle_crop_height,
        minimum_anpr_vehicle_crop_area=args.minimum_anpr_vehicle_crop_area,
    )
    eligible_jobs, eligibility_records = filter_anpr_eligible_jobs(vehicle_jobs, eligibility_config, run_dir=run_dir)
    write_jsonl(run_dir / "07_anpr" / "anpr_job_eligibility.jsonl", eligibility_records)

    _release_model(vehicle_stage)
    _release_model(person_stage)
    _clear_cuda_cache()

    florence_config = FlorenceConfig(
        enabled=env_config.vision.backend_mode != "disabled",
        base_model_path=str(florence_model) if florence_model is not None else None,
        adapter_path=str(florence_adapter) if florence_adapter is not None else None,
        device=device_plan.actual_florence_device,
        dtype=device_plan.actual_florence_dtype,
        load_in_4bit=False,
        load_in_8bit=False,
    )
    florence_engine, device_plan = _preload_florence_engine(
        florence_config,
        vision_config=env_config.vision,
        gemini_config=env_config.gemini,
        run_dir=run_dir,
        plan=device_plan,
    )
    plate_config = PlateDetectionConfig(
        model_path=str(plate_model),
        confidence_threshold=args.normal_plate_confidence,
        device=device_plan.actual_plate_device,
        minimum_plate_width=args.minimum_plate_width,
        minimum_plate_height=args.minimum_plate_height,
    )
    diagnostic_config = PlateDiagnosticConfig(
        maximum_vehicle_crop_attempts_per_track=args.max_attempts_per_track,
        diagnostic_confidence_thresholds=tuple(float(item.strip()) for item in args.diagnostic_thresholds.split(",") if item.strip()),
        save_annotated_vehicle_crops=args.save_annotations,
        save_rejected_plate_crops=args.save_rejected_plate_crops,
        run_ocr_on_valid_plate_candidates=True,
        stop_after_first_valid_plate_candidate=False,
        stop_after_first_non_empty_ocr_text=args.stop_after_first_non_empty_ocr,
    )
    plate_stage, device_plan = _preload_plate_stage(plate_config, run_dir=run_dir, plan=device_plan)
    retry_controller = BoundedPlateRetryController(plate_stage, florence_engine, diagnostic_config)
    diagnostic_results = retry_controller.process_job_groups(group_selected_jobs_by_track(eligible_jobs))
    diagnostic_sink = PlateDiagnosticArtifactSink(run_dir)
    for result in diagnostic_results:
        diagnostic_sink.write_result(result)
    plate_model_metadata = {"class_names": model_class_names(plate_stage._ensure_model()) if eligible_jobs else {}}
    plate_metrics = build_plate_diagnostic_metrics(
        diagnostic_results,
        processor_metrics=retry_controller.processor.metrics,
        model_metadata=plate_model_metadata,
    )
    diagnostic_sink.write_summary(plate_metrics)
    diagnostic_sink.close()

    colour_results = []
    jobs_by_track: dict[tuple[str, int, int], list[SelectedCropJob]] = defaultdict(list)
    for job in eligible_jobs:
        jobs_by_track[_job_key(job)].append(job)
    colour_started = time.perf_counter()
    for jobs in [jobs_by_track[key] for key in sorted(jobs_by_track)]:
        job = _colour_job(jobs)
        if job is not None:
            colour_results.append(florence_engine.run_colour(job))
    colour_runtime = round(time.perf_counter() - colour_started, 6)
    _write_anpr_jsonl(run_dir, diagnostic_results, colour_results)

    lifecycle_rows = crop_report.lifecycle_summary.get("completed_tracks", [])
    completed_by_class = Counter(str(row.get("dominant_class") or row.get("last_class_name") or "unknown").lower() for row in lifecycle_rows)
    selected_by_class = Counter(str(item.lifecycle_record.dominant_class or "unknown").lower() for item in selection_report.selected_track_crop_sets)
    clothing_summary = build_person_clothing_colour_summary(person_clothing_results)
    preliminary_summary = {
        "run_id": run_dir.name,
        "command": " ".join(sys.argv),
        "mode": args.mode,
        "video_path": str(video_path),
        "source_fps": crop_report.source_fps,
        "target_fps": crop_report.target_processing_fps,
        "source_frames": crop_report.total_source_frames,
        "selected_processed_frames": crop_report.selected_frames_processed,
        "first_processed_index": crop_report.first_processed_frame,
        "last_processed_index": crop_report.last_processed_frame,
        "first_timestamp_sec": crop_report.first_timestamp_sec,
        "last_timestamp_sec": crop_report.last_timestamp_sec,
        "vehicle_model_actual": str(vehicle_model),
        "person_model_actual": str(person_model),
        "plate_model": str(plate_model),
        "florence_model": str(florence_model) if florence_model is not None else None,
        "florence_adapter": str(florence_adapter) if florence_adapter is not None else None,
        "vision_backend_mode": env_config.vision.backend_mode,
        "vehicle_detector_device": device_plan.actual_vehicle_device,
        "person_detector_device": getattr(person_stage.config, "device", None) if person_stage is not None else None,
        "plate_detector_device": device_plan.actual_plate_device,
        "florence_device": device_plan.actual_florence_device,
        "florence_dtype": device_plan.actual_florence_dtype,
        "cuda_available": device_plan.cuda_available,
        "gpu_name": device_plan.gpu_name,
        "vehicle_detector_class_mapping": getattr(vehicle_stage, "_class_names", {}) if vehicle_stage is not None else {},
        "person_detector_class_mapping": getattr(person_stage, "_class_names", {}) if person_stage is not None else {},
        "resolved_class_mapping": {
            "vehicle": normalize_model_names(getattr(vehicle_stage, "_class_names", {}) if vehicle_stage is not None else {}),
            "person": normalize_model_names(getattr(person_stage, "_class_names", {}) if person_stage is not None else {}),
        },
        "supported_class_labels": list(SUPPORTED_CLASS_LABELS),
        "detection_metrics": crop_report.detection_metrics,
        "tracker_metrics": crop_report.tracker_metrics,
        "raw_tracking_summary": crop_report.raw_tracking_summary,
        "lifecycle_metrics": crop_report.lifecycle_metrics,
        "completed_generations_by_class": dict(sorted(completed_by_class.items())),
        "selected_crop_sets_by_class": dict(sorted(selected_by_class.items())),
        "person_clothing_colour_summary": clothing_summary,
        "person_anpr_bypass_count": len([job for job in all_jobs if _object_group(job.object_class) == "person"]),
        "vehicle_crop_jobs": len(vehicle_jobs),
        "anpr_eligible_crop_jobs": len(eligible_jobs),
        "plate_metrics": plate_metrics,
        "colour_calls": len(colour_results),
        "vision_backend_metrics": florence_engine.metrics,
        "runtime_device_policy": {
            "vehicle_yolo": "cuda if available else cpu",
            "person_yolo": "cuda if available else cpu",
            "plate_yolo": "cuda if available else cpu",
            "florence": "cuda if available else cpu",
            "bytetrack": "cpu",
            "opencv_video_processing": "cpu",
        },
        "inference_scheduling": "vehicle detector then person detector for each frame, then merged detections enter one ordered tracker",
        "tracking_strategy": "single ordered ByteTrack stage over merged normalized detections",
        "vehicle_model_fallback_reason": vehicle_fallback_reason,
        "vehicle_detector_device_fallback_reason": device_plan.vehicle_fallback_reason,
        "plate_detector_device_fallback_reason": device_plan.plate_fallback_reason,
        "florence_device_fallback_reason": device_plan.florence_fallback_reason,
        "full_video_executed": bool(args.full_video),
        "output_directory": str(run_dir),
    }
    write_json(run_dir / "reports" / "full_video_anpr_summary.json", preliminary_summary)
    write_json(run_dir / "reports" / "combined_pipeline_pre_step8_summary.json", preliminary_summary)

    step8_result = run_step8(run_dir=run_dir)
    step9_result = run_step9(run_dir=run_dir)
    query_file = run_dir / "10_structured_search" / "combined_validation_queries.json"
    query_file.parent.mkdir(parents=True, exist_ok=True)
    write_json(query_file, ["person", "white shirt person", "person in blue", "red car", "motorcycle"])
    run_step10_main(["--run-dir", str(run_dir), "--queries-file", str(query_file), "--top-k", "20"])
    step10_summary = read_json(run_dir / "10_structured_search" / "reports" / "step10_search_summary.json")
    records = [record.to_dict() for record in step9_result["records"]]
    record_counts = Counter(str(record.get("normalized_class_name") or record.get("object_class") or "unknown") for record in records)
    group_counts = Counter(str(record.get("object_group") or "unknown") for record in records)
    person_full_frames = sum(1 for record in records if record.get("object_group") == "person" and record.get("full_frame_path"))
    vehicle_final_rows = [row for row in read_jsonl(run_dir / "08_plate_validation" / "final_track_anpr_results.jsonl") if str(row.get("object_class") or "").lower() != "person"]
    final_summary = {
        **preliminary_summary,
        "peak_vram_mb": _peak_vram_mb(),
        "step8_summary": step8_result["summary"],
        "step9_summary": step9_result["summary"],
        "step10_summary": step10_summary,
        "records_written_by_class": dict(sorted(record_counts.items())),
        "records_written_by_group": dict(sorted(group_counts.items())),
        "person_records_with_full_frames": person_full_frames,
        "vehicle_anpr_result_counts": dict(sorted(Counter(str(row.get("plate_status") or "unknown") for row in vehicle_final_rows).items())),
        "total_runtime_sec": round(time.perf_counter() - started_total, 6),
        "realtime_factor": crop_report.realtime_factor,
        "remaining_limitations": [
            "Single ByteTrack instance is used over merged detections; class labels are preserved but no ReID/fragment merging is performed.",
            "Person clothing colour uses deterministic crop-region colour analysis, not identity or sensitive-attribute inference.",
            "Full-frame evidence is available only for newly generated combined runs.",
        ],
    }
    write_json(run_dir / "pipeline_report.json", final_summary)
    write_json(run_dir / "reports" / "combined_pipeline_report.json", final_summary)
    _write_alias_outputs(run_dir)
    return {"status": "passed", "run_dir": str(run_dir), "summary": final_summary}


def _write_alias_outputs(run_dir: Path) -> None:
    _copy_alias(run_dir / "09_searchable_objects" / "searchable_vehicle_records.jsonl", run_dir / "08_validated_objects" / "searchable_object_records.jsonl")
    _copy_alias(run_dir / "09_searchable_objects" / "searchable_vehicle_records_flat.json", run_dir / "08_validated_objects" / "validated_object_records.json")
    _copy_alias(run_dir / "10_structured_search" / "search_index_summary.json", run_dir / "09_search_index" / "search_index_summary.json")
    _copy_alias(run_dir / "10_structured_search" / "validation_search_results.jsonl", run_dir / "10_search_validation" / "validation_search_results.jsonl")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(json.dumps(to_json_safe(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
