from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .anpr_job_eligibility import AnprJobEligibilityConfig, filter_anpr_eligible_jobs
from .bytetrack_stage import create_bytetrack_stage
from .config import (
    BestCropScoreConfig,
    BestCropSelectionConfig,
    CropCollectionConfig,
    DetectionConfig,
    FlorenceConfig,
    GeminiConfig,
    PipelineConfig,
    PlateDetectionConfig,
    PlateDiagnosticConfig,
    TrackLifecycleConfig,
    TrackingConfig,
    VisionBackendConfig,
)
from .crop_artifacts import CropArtifactSink, CropImageWriter
from .crop_collector import CropCandidateCollector
from .crop_pipeline import SequentialCropCollectionPipeline, finalize_step5_artifacts
from .crop_selection import FinalBestCropSelector, SelectedCropJob
from .crop_selection_artifacts import CropSelectionArtifactSink
from .crop_selection_pipeline import finalize_step6_artifacts, run_selection_for_existing_bundles
from .lifecycle import TrackLifecycleManager
from .lifecycle_pipeline import LifecycleArtifactSink
from .plate_detection import UltralyticsPlateDetectionStage
from .plate_diagnostic_artifacts import PlateDiagnosticArtifactSink
from .plate_diagnostic_metrics import build_plate_diagnostic_metrics
from .plate_diagnostics import TrackPlateDiagnosticResult, model_class_names
from .plate_retry import BoundedPlateRetryController, group_selected_jobs_by_track
from .serialization import to_json_safe, write_json, write_jsonl
from .video_source import OpenCvVideoSource
from .vision_backends.base import VisionInferenceBackend
from .vision_backends.factory import create_vision_backend
from .yolo_stage import UltralyticsYoloDetectionStage


DEFAULT_VIDEO = r"C:\Mukul K\mk\test_video\anpr_test_5min.mp4"
DEFAULT_VEHICLE_MODEL = r"object\vehical_detection\best_old.pt"
DEFAULT_PLATE_MODEL = r"OCR_MUKUL\license_plate_weights.pt"
DEFAULT_FLORENCE_MODEL = r"C:\Mukul K\mk\models\Florence-2-base-ft"
DEFAULT_FLORENCE_ADAPTER = r"OCR_MUKUL\adaptor_florance_baseFT"
DEFAULT_ALLOWED_ANPR_CLASSES = "car,motorcycle,bus,truck,bicycle,auto,van,vehicle"
PERSON_CLASSES = {"person", "pedestrian"}
PREVIOUS_CPU_FLORENCE_BOUNDED_RUNTIME = {
    "run_dir": r"debug_runs\streaming_tracking_anpr_10fps_anpr_test_5min_20260718_155607",
    "total_runtime_sec": 45.403463,
    "florence_runtime_sec": 32.13337,
    "florence_device": "cpu",
    "florence_dtype": "float32",
}


@dataclass(frozen=True)
class RuntimeDevicePlan:
    requested_vehicle_device: str
    actual_vehicle_device: str
    vehicle_fallback_reason: str | None
    requested_plate_device: str
    actual_plate_device: str
    plate_fallback_reason: str | None
    requested_florence_device: str
    actual_florence_device: str
    florence_fallback_reason: str | None
    requested_florence_dtype: str
    actual_florence_dtype: str
    cuda_available: bool
    gpu_name: str | None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_required_path(path_value: str, label: str) -> Path:
    path = Path(path_value).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(_repo_root() / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"{label} does not exist: {path_value}")


def _resolve_optional_existing_path(path_value: str | None) -> Path | None:
    if path_value is None or not str(path_value).strip():
        return None
    path = Path(path_value).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(_repo_root() / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _resolve_vehicle_model(requested: str, fallback: str) -> tuple[Path, str | None]:
    requested_path = Path(requested).expanduser()
    requested_candidates = [requested_path]
    if not requested_path.is_absolute():
        requested_candidates.append(_repo_root() / requested_path)
    for candidate in requested_candidates:
        if candidate.exists():
            return candidate.resolve(), None
    fallback_path = _resolve_required_path(fallback, "fallback vehicle model")
    return fallback_path, f"requested vehicle model missing: {requested}"


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _cuda_available() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _gpu_name() -> str | None:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:
        return None
    return None


def _reset_cuda_peak_memory() -> None:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def _peak_vram_mb() -> float | None:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return round(float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0), 3)
    except Exception:
        return None
    return None


def _clear_cuda_cache() -> None:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        return


def _is_cuda_memory_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text or "cuda memory" in text or "cublas_status_alloc_failed" in text


def _resolve_auto_device(requested: str, *, cuda_available: bool) -> str:
    normalized = requested.strip().lower()
    if normalized == "auto":
        return "cuda" if cuda_available else "cpu"
    return normalized


def _resolve_auto_dtype(requested: str, *, actual_device: str) -> str:
    normalized = requested.strip().lower()
    if normalized == "auto":
        return "float16" if actual_device.startswith("cuda") else "float32"
    return normalized


def _build_device_plan(args: argparse.Namespace, *, cuda_available: bool, gpu_name: str | None) -> RuntimeDevicePlan:
    vehicle_device = _resolve_auto_device(args.vehicle_device, cuda_available=cuda_available)
    plate_device = _resolve_auto_device(args.plate_device, cuda_available=cuda_available)
    florence_device = _resolve_auto_device(args.florence_device, cuda_available=cuda_available)
    florence_dtype = _resolve_auto_dtype(args.florence_dtype, actual_device=florence_device)
    return RuntimeDevicePlan(
        requested_vehicle_device=args.vehicle_device,
        actual_vehicle_device=vehicle_device,
        vehicle_fallback_reason=None,
        requested_plate_device=args.plate_device,
        actual_plate_device=plate_device,
        plate_fallback_reason=None,
        requested_florence_device=args.florence_device,
        actual_florence_device=florence_device,
        florence_fallback_reason=None,
        requested_florence_dtype=args.florence_dtype,
        actual_florence_dtype=florence_dtype,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
    )


def _with_device_fallback(plan: RuntimeDevicePlan, *, component: str, reason: str) -> RuntimeDevicePlan:
    values = plan.__dict__.copy()
    if component == "vehicle":
        values["actual_vehicle_device"] = "cpu"
        values["vehicle_fallback_reason"] = reason
    elif component == "plate":
        values["actual_plate_device"] = "cpu"
        values["plate_fallback_reason"] = reason
    elif component == "florence":
        values["actual_florence_device"] = "cpu"
        values["actual_florence_dtype"] = "float32"
        values["florence_fallback_reason"] = reason
    else:
        raise ValueError(f"Unsupported device fallback component: {component}")
    return RuntimeDevicePlan(**values)


def _run_dir(output_root: str, video_path: Path) -> Path:
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in video_path.stem)
    return Path(output_root) / f"streaming_tracking_anpr_10fps_{safe_stem}_{time.strftime('%Y%m%d_%H%M%S')}"


def _job_key(job: SelectedCropJob) -> tuple[str, int, int]:
    return (job.source_id, job.track_id, job.track_generation)


def _jobs_from_selected_sets(selected_sets: Iterable[Any]) -> list[SelectedCropJob]:
    jobs: list[SelectedCropJob] = []
    for selected_set in selected_sets:
        jobs.extend(selected_set.to_crop_jobs())
    return jobs


def _class_counts_from_jobs(jobs: Iterable[SelectedCropJob]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for job in jobs:
        counts[str(job.object_class or "unknown").lower()] += 1
    return counts


def _colour_job(jobs: list[SelectedCropJob]) -> SelectedCropJob | None:
    ordered = sorted(jobs, key=lambda job: (0 if job.crop_role == "primary" and job.crop_rank == 1 else 1 if job.crop_role == "primary" else 2, job.crop_rank, job.frame_index))
    return ordered[0] if ordered else None


def _write_anpr_jsonl(run_dir: Path, diagnostic_results: list[TrackPlateDiagnosticResult], colour_results: list[Any]) -> None:
    candidates = []
    ocr_results = []
    track_rows = []
    for result in diagnostic_results:
        for attempt in result.attempts:
            candidates.extend(attempt.accepted_candidates)
            ocr_results.extend(attempt.ocr_results)
        track_rows.append(
            {
                "source_id": result.source_id,
                "track_id": result.track_id,
                "track_generation": result.track_generation,
                "source_track_id": result.source_track_id,
                "object_class": result.object_class,
                "final_status": result.final_status,
                "selected_attempt_number": result.selected_attempt_number,
                "selected_plate_candidate": result.selected_plate_candidate,
                "selected_ocr_result": result.selected_ocr_result,
                "attempt_count": len(result.attempts),
                "final_failure_reasons": result.final_failure_reasons,
            }
        )
    write_jsonl(run_dir / "07_anpr" / "plate_detection_candidates.jsonl", candidates)
    write_jsonl(run_dir / "07_anpr" / "florence_ocr_results.jsonl", ocr_results)
    write_jsonl(run_dir / "07_anpr" / "florence_colour_results.jsonl", colour_results)
    write_jsonl(run_dir / "07_anpr" / "track_anpr_colour_results.jsonl", track_rows)


def _preload_vehicle_stage(
    config: DetectionConfig,
    *,
    plan: RuntimeDevicePlan,
) -> tuple[UltralyticsYoloDetectionStage, RuntimeDevicePlan]:
    stage = UltralyticsYoloDetectionStage(config)
    if config.device != "cuda":
        _ = stage.model
        return stage, plan
    try:
        _ = stage.model
        return stage, plan
    except Exception as exc:
        if not _is_cuda_memory_error(exc):
            raise
        _clear_cuda_cache()
        try:
            _ = stage.model
            return stage, plan
        except Exception as retry_exc:
            if not _is_cuda_memory_error(retry_exc):
                raise
            reason = f"vehicle YOLO CUDA load failed after retry: {retry_exc}"
            cpu_config = DetectionConfig(
                model_path=config.model_path,
                confidence_threshold=config.confidence_threshold,
                iou_threshold=config.iou_threshold,
                allowed_class_ids=config.allowed_class_ids,
                allowed_class_names=config.allowed_class_names,
                device="cpu",
                image_size=config.image_size,
            )
            cpu_stage = UltralyticsYoloDetectionStage(cpu_config)
            _ = cpu_stage.model
            return cpu_stage, _with_device_fallback(plan, component="vehicle", reason=reason)


def _preload_plate_stage(
    config: PlateDetectionConfig,
    *,
    run_dir: Path,
    plan: RuntimeDevicePlan,
) -> tuple[UltralyticsPlateDetectionStage, RuntimeDevicePlan]:
    stage = UltralyticsPlateDetectionStage(config, output_dir=run_dir, run_dir=run_dir)
    if config.device != "cuda":
        _ = stage._ensure_model()
        return stage, plan
    try:
        _ = stage._ensure_model()
        return stage, plan
    except Exception as exc:
        if not _is_cuda_memory_error(exc):
            raise
        _clear_cuda_cache()
        try:
            _ = stage._ensure_model()
            return stage, plan
        except Exception as retry_exc:
            if not _is_cuda_memory_error(retry_exc):
                raise
            reason = f"plate YOLO CUDA load failed after retry: {retry_exc}"
            cpu_config = PlateDetectionConfig(
                enabled=config.enabled,
                model_path=config.model_path,
                confidence_threshold=config.confidence_threshold,
                iou_threshold=config.iou_threshold,
                device="cpu",
                image_size=config.image_size,
                max_plate_detections_per_vehicle_crop=config.max_plate_detections_per_vehicle_crop,
                minimum_plate_width=config.minimum_plate_width,
                minimum_plate_height=config.minimum_plate_height,
                crop_padding_ratio=config.crop_padding_ratio,
                save_plate_crops=config.save_plate_crops,
            )
            cpu_stage = UltralyticsPlateDetectionStage(cpu_config, output_dir=run_dir, run_dir=run_dir)
            _ = cpu_stage._ensure_model()
            return cpu_stage, _with_device_fallback(plan, component="plate", reason=reason)


def _preload_florence_engine(
    config: FlorenceConfig,
    *,
    vision_config: VisionBackendConfig,
    gemini_config: GeminiConfig,
    run_dir: Path,
    plan: RuntimeDevicePlan,
) -> tuple[VisionInferenceBackend, RuntimeDevicePlan]:
    mode = vision_config.backend_mode
    if mode == "disabled":
        return create_vision_backend(
            vision_config=vision_config,
            florence_config=config,
            gemini_config=gemini_config,
            run_dir=run_dir,
        ), plan
    if mode == "gemini":
        return create_vision_backend(
            vision_config=vision_config,
            florence_config=config,
            gemini_config=gemini_config,
            run_dir=run_dir,
        ), plan
    engine = create_vision_backend(
        vision_config=VisionBackendConfig(backend_mode="florence"),
        florence_config=config,
        gemini_config=gemini_config,
        run_dir=run_dir,
    )
    if config.device != "cuda":
        try:
            engine.load()
        except Exception:
            if mode == "florence":
                raise
        if mode == "auto":
            return create_vision_backend(
                vision_config=vision_config,
                florence_config=config,
                gemini_config=gemini_config,
                run_dir=run_dir,
            ), plan
        return engine, plan
    try:
        engine.load()
        if mode == "auto":
            return create_vision_backend(
                vision_config=vision_config,
                florence_config=config,
                gemini_config=gemini_config,
                run_dir=run_dir,
            ), plan
        return engine, plan
    except Exception as exc:
        if not _is_cuda_memory_error(exc):
            if mode == "florence":
                raise
            return create_vision_backend(
                vision_config=vision_config,
                florence_config=config,
                gemini_config=gemini_config,
                run_dir=run_dir,
            ), plan
        if hasattr(getattr(engine, "engine", None), "bundle"):
            engine.engine.bundle = None
        _clear_cuda_cache()
        try:
            engine.load()
            if mode == "auto":
                return create_vision_backend(
                    vision_config=vision_config,
                    florence_config=config,
                    gemini_config=gemini_config,
                    run_dir=run_dir,
                ), plan
            return engine, plan
        except Exception as retry_exc:
            if not _is_cuda_memory_error(retry_exc):
                if mode == "florence":
                    raise
                return create_vision_backend(
                    vision_config=vision_config,
                    florence_config=config,
                    gemini_config=gemini_config,
                    run_dir=run_dir,
                ), plan
            reason = f"Florence CUDA load failed after retry: {retry_exc}"
            if hasattr(getattr(engine, "engine", None), "bundle"):
                engine.engine.bundle = None
            _clear_cuda_cache()
            cpu_config = FlorenceConfig(
                enabled=config.enabled,
                base_model_path=config.base_model_path,
                adapter_path=config.adapter_path,
                device="cpu",
                dtype="float32",
                local_files_only=config.local_files_only,
                trust_remote_code=config.trust_remote_code,
                load_in_4bit=False,
                load_in_8bit=False,
                max_new_tokens=config.max_new_tokens,
                num_beams=config.num_beams,
                do_sample=config.do_sample,
                ocr_task_prompt=config.ocr_task_prompt,
                colour_task_prompt=config.colour_task_prompt,
            )
            cpu_engine = create_vision_backend(
                vision_config=VisionBackendConfig(backend_mode="florence"),
                florence_config=cpu_config,
                gemini_config=gemini_config,
                run_dir=run_dir,
            )
            cpu_engine.load()
            next_plan = _with_device_fallback(plan, component="florence", reason=reason)
            if mode == "auto":
                return create_vision_backend(
                    vision_config=vision_config,
                    florence_config=cpu_config,
                    gemini_config=gemini_config,
                    run_dir=run_dir,
                ), next_plan
            return cpu_engine, next_plan


def _release_model(value: Any) -> None:
    try:
        if hasattr(value, "_model"):
            value._model = None
        if hasattr(value, "model"):
            value.model = None
        if hasattr(value, "bundle"):
            value.bundle = None
    except Exception:
        return


def _summarize_selected(selected_sets: Iterable[Any]) -> dict[str, Any]:
    selected_rows = list(selected_sets)
    vehicle_classes = {"car", "motorcycle", "bus", "truck", "bicycle", "auto", "van", "vehicle"}
    vehicle_sets = [item for item in selected_rows if str(item.lifecycle_record.dominant_class or "").lower() in vehicle_classes]
    all_primary = [crop for item in selected_rows for crop in item.primary_crops]
    all_fallback = [item.fallback_crop for item in selected_rows if item.fallback_crop is not None]
    vehicle_primary = [crop for item in vehicle_sets for crop in item.primary_crops]
    vehicle_fallback = [item.fallback_crop for item in vehicle_sets if item.fallback_crop is not None]
    dimensions = []
    for crop in vehicle_primary + vehicle_fallback:
        dimensions.append(
            {
                "track_id": crop.track_id,
                "track_generation": crop.track_generation,
                "role": crop.role,
                "rank": crop.rank,
                "frame_index": crop.frame_index,
                "timestamp_sec": crop.timestamp_sec,
                "selection_score": crop.final_score,
                "crop_path": crop.vehicle_crop_path,
                "source_bbox": crop.metadata.get("source_bbox"),
            }
        )
    return {
        "selected_track_sets": len(selected_rows),
        "vehicle_track_bundles": len(vehicle_sets),
        "vehicle_tracks_with_primary_crops": sum(1 for item in vehicle_sets if item.primary_crops),
        "vehicle_fallback_only_tracks": sum(1 for item in vehicle_sets if not item.primary_crops and item.fallback_crop is not None),
        "vehicle_no_valid_crop_tracks": sum(1 for item in vehicle_sets if item.selection_status == "no_valid_crop"),
        "primary_crops_all_classes": len(all_primary),
        "fallback_crops_all_classes": len(all_fallback),
        "primary_vehicle_crops": len(vehicle_primary),
        "fallback_vehicle_crops": len(vehicle_fallback),
        "vehicle_crop_details": dimensions,
    }


def _direct_answer_summary(summary: dict[str, Any]) -> dict[str, str]:
    eligible = summary["anpr_eligible_crop_jobs"]
    raw_boxes = summary["plate_metrics"]["raw_detector_boxes"]
    accepted = summary["plate_metrics"]["accepted_plate_candidates"]
    non_empty = summary["plate_metrics"]["ocr_non_empty_outputs"]
    excluded_small = summary["eligibility_exclusion_counts"].get("vehicle_crop_too_small", 0)
    primary_plate = summary["plate_metrics"].get("by_crop_role", {}).get("primary", {})
    fallback_plate = summary["plate_metrics"].get("by_crop_role", {}).get("fallback", {})
    return {
        "did_10fps_video_pipeline_produce_vehicle_crops_large_enough_for_plate_detection": "yes" if eligible else "no",
        "were_any_plate_boxes_truly_detected": "yes" if raw_boxes else "no",
        "did_any_accepted_plate_crop_produce_non_empty_ocr": "yes" if non_empty else "no",
        "likely_failure_area": _likely_failure_area(eligible, raw_boxes, accepted, non_empty, excluded_small),
        "primary_or_fallback_better_for_anpr": _better_role(primary_plate, fallback_plate),
        "are_current_vehicle_crop_size_gates_appropriate": "yes for this bounded diagnostic" if eligible else "too restrictive or selected crops are too small",
        "ready_for_step8_verification": "yes, bounded run has non-empty OCR to verify" if non_empty else "no, improve crop selection before Step 8",
    }


def _likely_failure_area(eligible: int, raw_boxes: int, accepted: int, non_empty: int, excluded_small: int) -> str:
    if not eligible:
        return "crop_selection_or_crop_size" if excluded_small else "tracking_or_vehicle_detection"
    if not raw_boxes:
        return "plate_visibility_or_plate_detection_on_selected_vehicle_crops"
    if not accepted:
        return "plate_box_filtering_or_plate_crop_geometry"
    if not non_empty:
        return "ocr"
    return "none_observed_in_bounded_run"


def _better_role(primary_counts: dict[str, int], fallback_counts: dict[str, int]) -> str:
    primary_success = sum(value for key, value in primary_counts.items() if "ocr_success_non_empty" in key or "plate_candidate" in key)
    fallback_success = sum(value for key, value in fallback_counts.items() if "ocr_success_non_empty" in key or "plate_candidate" in key)
    if primary_success > fallback_success:
        return "primary"
    if fallback_success > primary_success:
        return "fallback"
    return "inconclusive"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run bounded sequential 10 FPS ANPR video validation.")
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--source-id", default=None)
    parser.add_argument("--vehicle-detector-model", default=DEFAULT_VEHICLE_MODEL)
    parser.add_argument("--vehicle-detector-fallback-model", default="yolo11n.pt")
    parser.add_argument("--plate-detector-model", default=DEFAULT_PLATE_MODEL)
    parser.add_argument("--florence-model", default=DEFAULT_FLORENCE_MODEL)
    parser.add_argument("--florence-adapter", default=DEFAULT_FLORENCE_ADAPTER)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--max-processed-frames", type=int, default=600)
    parser.add_argument("--full-video", action="store_true")
    parser.add_argument("--start-sec", type=float, default=None)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument("--tracking-backend", choices=("ultralytics_bytetrack", "supervision_bytetrack"), default="ultralytics_bytetrack")
    parser.add_argument("--vehicle-device", default="auto")
    parser.add_argument("--plate-device", default="auto")
    parser.add_argument("--florence-device", default="auto")
    parser.add_argument("--florence-dtype", default="auto")
    parser.add_argument("--florence-load-in-4bit", action="store_true")
    parser.add_argument("--florence-load-in-8bit", action="store_true")
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--allowed-class-names", default="person,car,motorcycle,bus,truck,bicycle,auto,van,vehicle")
    parser.add_argument("--track-activation-threshold", type=float, default=0.30)
    parser.add_argument("--track-high-threshold", type=float, default=0.05)
    parser.add_argument("--track-low-threshold", type=float, default=0.01)
    parser.add_argument("--new-track-threshold", type=float, default=0.05)
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
    parser.add_argument("--minimum-primary-score", type=float, default=0.45)
    parser.add_argument("--minimum-fallback-score", type=float, default=0.20)
    parser.add_argument("--minimum-track-observations-for-primary", type=int, default=3)
    parser.add_argument("--minimum-candidates-for-primary", type=int, default=2)
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
    parser.add_argument("--allowed-anpr-classes", default=DEFAULT_ALLOWED_ANPR_CLASSES)
    parser.add_argument("--save-annotations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-rejected-plate-crops", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-after-first-non-empty-ocr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-root", default="debug_runs")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_total = time.perf_counter()
    env_config = PipelineConfig.from_env()
    if args.florence_load_in_4bit or args.florence_load_in_8bit:
        raise ValueError("Quantization is disabled for ANPR video validation; do not pass 4-bit or 8-bit flags.")
    video_path = _resolve_required_path(args.video, "video")
    plate_model = _resolve_required_path(args.plate_detector_model, "plate detector model")
    florence_model = _resolve_optional_existing_path(env_config.florence.base_model_path or args.florence_model)
    florence_adapter = _resolve_optional_existing_path(env_config.florence.adapter_path or args.florence_adapter)
    vehicle_model, fallback_reason = _resolve_vehicle_model(args.vehicle_detector_model, args.vehicle_detector_fallback_model)
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

    detection_config = DetectionConfig(
        model_path=str(vehicle_model),
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        allowed_class_names=_split_csv(args.allowed_class_names),
        device=device_plan.actual_vehicle_device,
        image_size=args.image_size,
    )
    tracking_config = TrackingConfig(
        backend=args.tracking_backend,
        track_activation_threshold=args.track_activation_threshold,
        lost_track_buffer=args.track_buffer,
        track_high_threshold=args.track_high_threshold,
        track_low_threshold=args.track_low_threshold,
        new_track_threshold=args.new_track_threshold,
        match_threshold=args.match_threshold,
    )
    detection_stage, device_plan = _preload_vehicle_stage(detection_config, plan=device_plan)
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
        detector_model_path=str(vehicle_model),
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
        mode="anpr_video_10fps_validation",
        bundles=crop_pipeline.completed_bundles,
        selector=selector,
        sink=CropSelectionArtifactSink(run_dir, create_previews=True),
        source_path=str(video_path),
        source_id=source.source_id,
        tracking_backend=args.tracking_backend,
    )
    object.__setattr__(selection_report, "crop_collection_report", crop_report.to_dict())
    finalize_step6_artifacts(run_dir, selection_report)
    _release_model(detection_stage)
    _clear_cuda_cache()

    all_jobs = _jobs_from_selected_sets(selection_report.selected_track_crop_sets)
    eligibility_config = AnprJobEligibilityConfig(
        allowed_anpr_classes=_split_csv(args.allowed_anpr_classes),
        minimum_anpr_vehicle_crop_width=args.minimum_anpr_vehicle_crop_width,
        minimum_anpr_vehicle_crop_height=args.minimum_anpr_vehicle_crop_height,
        minimum_anpr_vehicle_crop_area=args.minimum_anpr_vehicle_crop_area,
    )
    eligible_jobs, eligibility_records = filter_anpr_eligible_jobs(all_jobs, eligibility_config, run_dir=run_dir)
    write_jsonl(run_dir / "07_anpr" / "anpr_job_eligibility.jsonl", eligibility_records)

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

    eligibility_counts = Counter(record.exclusion_reason for record in eligibility_records if not record.eligible)
    person_excluded = sum(count for name, count in _class_counts_from_jobs(all_jobs).items() if name in PERSON_CLASSES)
    selection_summary = _summarize_selected(selection_report.selected_track_crop_sets)
    detection_metrics = crop_report.detection_metrics
    tracker_metrics = crop_report.tracker_metrics
    lifecycle_metrics = crop_report.lifecycle_metrics
    total_runtime = round(time.perf_counter() - started_total, 6)
    summary = {
        "run_id": run_dir.name,
        "mode": "full" if args.full_video else "bounded",
        "command": " ".join(sys.argv),
        "video_path": str(video_path),
        "source_fps": crop_report.source_fps,
        "target_fps": crop_report.target_processing_fps,
        "source_frames": crop_report.total_source_frames,
        "selected_processed_frames": crop_report.selected_frames_processed,
        "first_processed_index": crop_report.first_processed_frame,
        "last_processed_index": crop_report.last_processed_frame,
        "processed_timeline_sec": None
        if crop_report.first_timestamp_sec is None or crop_report.last_timestamp_sec is None
        else round(crop_report.last_timestamp_sec - crop_report.first_timestamp_sec, 6),
        "vehicle_model_requested": args.vehicle_detector_model,
        "vehicle_model_actual": str(vehicle_model),
        "vehicle_model_fallback_reason": fallback_reason,
        "vehicle_detector_requested_device": device_plan.requested_vehicle_device,
        "vehicle_detector_device": device_plan.actual_vehicle_device,
        "vehicle_detector_device_fallback_reason": device_plan.vehicle_fallback_reason,
        "plate_model": str(plate_model),
        "plate_detector_requested_device": device_plan.requested_plate_device,
        "plate_detector_device": device_plan.actual_plate_device,
        "plate_detector_device_fallback_reason": device_plan.plate_fallback_reason,
        "florence_model": str(florence_model) if florence_model is not None else None,
        "florence_adapter": str(florence_adapter) if florence_adapter is not None else None,
        "vision_backend_mode": env_config.vision.backend_mode,
        "florence_requested_device": device_plan.requested_florence_device,
        "florence_device": device_plan.actual_florence_device,
        "florence_device_fallback_reason": device_plan.florence_fallback_reason,
        "florence_requested_dtype": device_plan.requested_florence_dtype,
        "florence_dtype": device_plan.actual_florence_dtype,
        "florence_quantization": "4bit" if args.florence_load_in_4bit else "8bit" if args.florence_load_in_8bit else "none",
        "runtime_device_policy": {
            "vehicle_yolo": "cuda if available else cpu",
            "plate_yolo": "cuda if available else cpu",
            "florence": "cuda if available else cpu",
            "bytetrack": "cpu",
            "opencv_video_processing": "cpu",
        },
        "cuda_available": device_plan.cuda_available,
        "gpu_name": device_plan.gpu_name,
        "peak_vram_mb": _peak_vram_mb(),
        "previous_cpu_florence_bounded_runtime": PREVIOUS_CPU_FLORENCE_BOUNDED_RUNTIME,
        "vehicle_detector_class_names": getattr(detection_stage, "_class_names", {}),
        "detections_by_class": detection_metrics.get("class_counts", {}),
        "detection_metrics": detection_metrics,
        "raw_tracker_ids": sorted(crop_report.raw_tracking_summary.get("track_observation_counts", {}).keys(), key=str),
        "raw_tracking_summary": crop_report.raw_tracking_summary,
        "tracker_metrics": tracker_metrics,
        "lifecycle_metrics": lifecycle_metrics,
        "lifecycle_generations": crop_report.lifecycle_summary.get("completed_tracks", []),
        "confirmed_vehicle_tracks": sum(1 for item in crop_report.lifecycle_summary.get("completed_tracks", []) if str(item.get("dominant_class", "")).lower() not in PERSON_CLASSES),
        "completed_vehicle_tracks": selection_summary["vehicle_track_bundles"],
        "crop_collection_summary": crop_report.crop_collection_summary,
        "selection_summary": selection_summary,
        "all_selected_crop_jobs": len(all_jobs),
        "person_crops_excluded_from_anpr": person_excluded,
        "eligibility_exclusion_counts": {str(key): value for key, value in sorted(eligibility_counts.items())},
        "crops_excluded_by_minimum_size": eligibility_counts.get("vehicle_crop_too_small", 0),
        "anpr_eligible_crop_jobs": len(eligible_jobs),
        "plate_metrics": plate_metrics,
        "plate_crop_paths": [
            candidate.plate_crop_path
            for result in diagnostic_results
            for attempt in result.attempts
            for candidate in attempt.accepted_candidates
            if candidate.plate_crop_path
        ],
        "ocr_by_track_generation": [
            {
                "source_id": result.source_id,
                "track_id": result.track_id,
                "track_generation": result.track_generation,
                "raw_text": result.selected_ocr_result.raw_text if result.selected_ocr_result else None,
                "normalized_text": result.selected_ocr_result.normalized_text if result.selected_ocr_result else None,
                "status": result.selected_ocr_result.status if result.selected_ocr_result else None,
            }
            for result in diagnostic_results
        ],
        "colour_calls": len(colour_results),
        "normalized_colours_by_track_generation": [
            {
                "source_id": result.source_id,
                "track_id": result.track_id,
                "track_generation": result.track_generation,
                "raw_text": result.raw_text,
                "normalized_colour": result.normalized_colour,
                "status": result.status,
            }
            for result in colour_results
        ],
        "vision_backend_metrics": florence_engine.metrics,
        "detection_runtime_sec": detection_metrics.get("runtime_sec"),
        "tracking_runtime_sec": tracker_metrics.get("runtime_sec"),
        "crop_selection_runtime_sec": selection_report.runtime_sec,
        "florence_runtime_sec": colour_runtime + sum(attempt.runtime_sec for result in diagnostic_results for attempt in result.attempts),
        "total_runtime_sec": total_runtime,
        "output_directory": str(run_dir),
        "important_evidence_paths": {
            "source_metadata": str(run_dir / "01_source" / "source_metadata.json"),
            "selected_crop_jobs": str(run_dir / "06_selected_crops" / "selected_crop_jobs.jsonl"),
            "anpr_job_eligibility": str(run_dir / "07_anpr" / "anpr_job_eligibility.jsonl"),
            "plate_diagnostic_attempts": str(run_dir / "07_5_plate_diagnostics" / "plate_diagnostic_attempts.jsonl"),
            "raw_plate_box_diagnostics": str(run_dir / "07_5_plate_diagnostics" / "raw_plate_box_diagnostics.jsonl"),
            "accepted_plate_crops": str(run_dir / "07_5_plate_diagnostics" / "accepted_plate_crops"),
            "annotated_vehicle_crops": str(run_dir / "07_5_plate_diagnostics" / "annotated_vehicle_crops"),
        },
        "full_video_executed": bool(args.full_video),
        "limitations": [
            "raw OCR is not verified",
            "no Step 8 verification was run",
            "no ReID, fragment merging, queues, threads, multiprocessing, search indexing, or event detection was added",
        ],
    }
    summary["direct_answers"] = _direct_answer_summary(summary)
    summary["recommended_next_change"] = (
        "proceed to Step 8 verification on bounded OCR outputs"
        if plate_metrics["ocr_non_empty_outputs"]
        else "improve vehicle crop selection for larger, plate-visible vehicle crops before Step 8"
    )
    write_json(run_dir / "reports" / "full_video_anpr_summary.json", summary)
    write_json(run_dir / "reports" / "full_video_anpr_report.json", summary)
    return {"status": "passed", "run_dir": str(run_dir), "summary": summary}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(json.dumps(to_json_safe(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
