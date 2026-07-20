from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .anpr_artifacts import read_selected_crop_jobs
from .config import FlorenceConfig, PipelineConfig, PlateDetectionConfig, PlateDiagnosticConfig
from .crop_selection import SelectedCropJob
from .florence_inference import FlorenceInferenceEngine, FlorenceModelBundle
from .plate_detection import UltralyticsPlateDetectionStage
from .plate_diagnostic_artifacts import PlateDiagnosticArtifactSink
from .plate_diagnostic_metrics import build_plate_diagnostic_metrics
from .plate_diagnostics import PlateDiagnosticProcessor, model_class_names
from .plate_retry import BoundedPlateRetryController, group_selected_jobs_by_track
from .serialization import write_json
from .vision_backends.factory import create_vision_backend


DEFAULT_STEP6_ARTIFACT_RUN = Path(
    r"debug_runs\streaming_tracking_pipeline\streaming_tracking_step6_existing_step5_20260718_144055"
)
DEFAULT_STEP7_ARTIFACT_RUN = Path(
    r"debug_runs\streaming_tracking_pipeline\streaming_tracking_step7_existing_step6_20260718_150352"
)


class FakeDiagnosticPlateModel:
    names = {0: "license_plate", 1: "vehicle"}

    def predict(self, source: str, **_: Any) -> dict[str, Any]:
        name = Path(source).stem
        if "no_boxes" in name:
            return {"boxes": []}
        if "below" in name:
            return {"boxes": [{"bbox": [10, 10, 45, 24], "confidence": 0.08, "class_id": 0}]}
        if "diagnostic_accept" in name:
            return {"boxes": [{"bbox": [10, 10, 45, 24], "confidence": 0.12, "class_id": 0}]}
        if "wrong_class" in name:
            return {"boxes": [{"bbox": [10, 10, 45, 24], "confidence": 0.9, "class_id": 1}]}
        if "invalid" in name:
            return {"boxes": [{"bbox": [45, 10, 10, 24], "confidence": 0.9, "class_id": 0}]}
        if "empty_clip" in name:
            return {"boxes": [{"bbox": [-30, -30, -10, -5], "confidence": 0.9, "class_id": 0}]}
        if "width_small" in name:
            return {"boxes": [{"bbox": [10, 10, 13, 24], "confidence": 0.9, "class_id": 0}]}
        if "height_small" in name:
            return {"boxes": [{"bbox": [10, 10, 45, 12], "confidence": 0.9, "class_id": 0}]}
        if "multi" in name:
            return {
                "boxes": [
                    {"bbox": [10, 10, 45, 24], "confidence": 0.9, "class_id": 0},
                    {"bbox": [50, 12, 82, 25], "confidence": 0.7, "class_id": 0},
                ]
            }
        return {"boxes": [{"bbox": [10, 10, 45, 24], "confidence": 0.9, "class_id": 0}]}


class FakeOcrEngine:
    def run_ocr(self, candidate: Any) -> Any:
        from .anpr_schemas import FlorenceOcrResult

        text = "" if "empty_ocr" in str(candidate.vehicle_crop_path) else "MH12AB1234"
        status = "success" if text else "empty_output"
        return FlorenceOcrResult(
            source_id=candidate.source_id,
            track_id=candidate.track_id,
            track_generation=candidate.track_generation,
            crop_role=candidate.crop_role,
            crop_rank=candidate.crop_rank,
            frame_index=candidate.frame_index,
            plate_rank=candidate.plate_rank,
            plate_crop_path=candidate.plate_crop_path,
            raw_text=text,
            normalized_text=text,
            status=status,
            prompt="<OCR>",
        )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_existing_path(value: str | None, fallback: Path | None = None) -> Path:
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
    return str(path if path.is_absolute() else (_repo_root() / path).resolve())


def _parse_thresholds(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("--diagnostic-thresholds must include at least one threshold.")
    return values


def _plate_config(args: argparse.Namespace) -> PlateDetectionConfig:
    return PlateDetectionConfig(
        enabled=True,
        model_path=_resolve_optional_path(args.plate_detector_model),
        confidence_threshold=args.normal_plate_confidence,
        iou_threshold=args.plate_iou,
        device=args.device,
        image_size=args.image_size,
        max_plate_detections_per_vehicle_crop=args.max_plate_candidates,
        minimum_plate_width=args.minimum_plate_width,
        minimum_plate_height=args.minimum_plate_height,
        crop_padding_ratio=args.plate_crop_padding_ratio,
        save_plate_crops=True,
    )


def _diagnostic_config(args: argparse.Namespace) -> PlateDiagnosticConfig:
    return PlateDiagnosticConfig(
        enabled=True,
        maximum_vehicle_crop_attempts_per_track=args.max_attempts_per_track,
        run_multiple_confidence_thresholds=True,
        diagnostic_confidence_thresholds=_parse_thresholds(args.diagnostic_thresholds),
        minimum_box_confidence_to_record=args.minimum_box_confidence_to_record,
        save_annotated_vehicle_crops=args.save_annotations,
        save_rejected_plate_crops=args.save_rejected_plate_crops,
        save_valid_plate_crops=True,
        run_ocr_on_valid_plate_candidates=args.run_ocr,
        stop_after_first_valid_plate_candidate=args.stop_after_first_plate,
        stop_after_first_non_empty_ocr_text=args.stop_after_first_non_empty_ocr,
        maximum_raw_boxes_per_attempt=args.maximum_raw_boxes_per_attempt,
    )


def _florence_config(args: argparse.Namespace) -> FlorenceConfig:
    return FlorenceConfig(
        enabled=args.run_ocr,
        base_model_path=_resolve_optional_path(args.florence_model),
        adapter_path=_resolve_optional_path(args.florence_adapter),
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )


def _synthetic_jobs(run_dir: Path) -> list[SelectedCropJob]:
    image_dir = run_dir / "synthetic_plate_diag_inputs"
    image_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        ("no_boxes", 1, 0, "primary", 1),
        ("diagnostic_accept", 2, 0, "primary", 1),
        ("wrong_class", 3, 0, "primary", 1),
        ("invalid", 4, 0, "primary", 1),
        ("empty_clip", 5, 0, "primary", 1),
        ("width_small", 6, 0, "primary", 1),
        ("height_small", 7, 0, "primary", 1),
        ("accepted", 8, 0, "primary", 1),
        ("multi", 9, 0, "primary", 1),
        ("no_boxes", 10, 0, "primary", 1),
        ("accepted", 10, 0, "primary", 2),
        ("no_boxes", 11, 0, "primary", 1),
        ("accepted", 11, 0, "fallback", 1),
        ("no_boxes", 12, 0, "primary", 1),
        ("accepted", 13, 0, "primary", 1),
        ("empty_ocr", 14, 0, "primary", 1),
        ("accepted", 14, 0, "primary", 2),
        ("accepted", 1, 1, "primary", 1),
    ]
    jobs: list[SelectedCropJob] = []
    for case, track_id, generation, role, rank in cases:
        path = image_dir / f"{case}_track_{track_id}_gen_{generation}_{role}_{rank}.jpg"
        image = Image.new("RGB", (100, 60), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 10, 45, 24), fill="gray")
        image.save(path)
        jobs.append(
            SelectedCropJob(
                source_id="synthetic_plate_diag",
                track_id=track_id,
                track_generation=generation,
                source_track_id=f"raw_{track_id}",
                object_class="car",
                lifecycle_completion_reason="synthetic",
                crop_role=role,
                crop_rank=rank,
                frame_index=rank,
                timestamp_sec=float(rank),
                vehicle_crop_path=str(path),
                full_frame_path=None,
                selection_score=0.9 - rank * 0.05,
                metadata={"source_bbox": [10, 10, 90, 50], "synthetic_case": case},
            )
        )
    return jobs


def _make_run_dir(args: argparse.Namespace, mode_label: str) -> Path:
    return Path(args.output_root) / f"streaming_tracking_step75_{mode_label}_{time.strftime('%Y%m%d_%H%M%S')}"


def _copy_step6_context(step6_run_dir: Path, run_dir: Path) -> None:
    source = step6_run_dir / "06_selected_crops"
    target = run_dir / "06_selected_crops"
    if source.exists() and not target.exists():
        shutil.copytree(source, target)


def _run_jobs(args: argparse.Namespace, jobs: list[SelectedCropJob], run_dir: Path, *, fake: bool, source_path: str | None = None) -> dict[str, Any]:
    env_config = PipelineConfig.from_env()
    plate_config = _plate_config(args)
    diagnostic_config = _diagnostic_config(args)
    detector = UltralyticsPlateDetectionStage(
        plate_config,
        output_dir=run_dir,
        model=FakeDiagnosticPlateModel() if fake else None,
        run_dir=run_dir,
    )
    florence_engine = FakeOcrEngine() if fake and args.run_ocr else None
    if not fake and args.run_ocr:
        florence_engine = create_vision_backend(
            vision_config=env_config.vision,
            florence_config=_florence_config(args),
            gemini_config=env_config.gemini,
            run_dir=run_dir,
        )
    controller = BoundedPlateRetryController(detector, florence_engine, diagnostic_config)
    results = controller.process_job_groups(group_selected_jobs_by_track(jobs))
    processor_metrics = dict(controller.processor.metrics)
    model_metadata = {}
    try:
        model = detector._ensure_model()
        model_metadata["plate_model_classes"] = model_class_names(model)
    except Exception as exc:
        model_metadata["plate_model_class_error"] = str(exc)
    summary = build_plate_diagnostic_metrics(results, processor_metrics=processor_metrics, model_metadata=model_metadata)
    summary.update(
        {
            "mode": args.mode,
            "run_dir": str(run_dir),
            "source_path": source_path,
            "selected_crop_jobs_inspected": len(jobs),
            "normal_plate_confidence": args.normal_plate_confidence,
            "diagnostic_thresholds": list(_parse_thresholds(args.diagnostic_thresholds)),
            "plate_model_path": plate_config.model_path,
            "device": args.device,
            "crop_attempt_table": _attempt_table(results),
            "known_good_plate_model_check": _known_good_check(args, run_dir, detector, diagnostic_config) if not fake else {"status": "not_run_synthetic"},
        }
    )
    sink = PlateDiagnosticArtifactSink(run_dir)
    try:
        for result in results:
            sink.write_result(result)
        sink.write_summary(summary)
    finally:
        sink.close()
    write_json(run_dir / "reports" / "step75_validation_result.json", summary)
    return {"status": "passed", "mode": args.mode, "run_dir": str(run_dir), "report": summary}


def _attempt_table(results: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        for attempt in result.attempts:
            vehicle = attempt.metadata.get("vehicle_crop", {})
            best_conf = max((box.raw_confidence for box in attempt.raw_boxes), default=None)
            rows.append(
                {
                    "track_id": result.track_id,
                    "track_generation": result.track_generation,
                    "role": attempt.vehicle_crop_role,
                    "rank": attempt.vehicle_crop_rank,
                    "frame_index": attempt.source_frame_index,
                    "crop_dimensions": [vehicle.get("crop_width"), vehicle.get("crop_height")],
                    "selection_score": vehicle.get("selection_score"),
                    "sharpness": vehicle.get("sharpness"),
                    "brightness": vehicle.get("brightness"),
                    "edge_touching": vehicle.get("edge_touching"),
                    "raw_box_count": attempt.raw_box_count,
                    "best_raw_confidence": best_conf,
                    "accepted_candidate_count": attempt.accepted_plate_count,
                    "attempt_status": attempt.attempt_status.value,
                }
            )
    return rows


def _find_known_good_input() -> Path | None:
    patterns = [
        "OCR_MUKUL/detection/cropped_plates/*.jpg",
        "debug_runs/**/cropped_plates/*.jpg",
        "debug_runs/**/06_plate_crops/*.jpg",
        "debug_runs/**/plate_crops/*.jpg",
    ]
    for pattern in patterns:
        for path in _repo_root().glob(pattern):
            if path.is_file():
                return path
    return None


def _known_good_check(args: argparse.Namespace, run_dir: Path, detector: UltralyticsPlateDetectionStage, diagnostic_config: PlateDiagnosticConfig) -> dict[str, Any]:
    known_good = Path(args.known_good_input).expanduser() if args.known_good_input else _find_known_good_input()
    if known_good is None or not known_good.exists():
        return {"status": "not_found", "known_good_input_path": None}
    job = SelectedCropJob(
        source_id="known_good_plate_check",
        track_id=999999,
        track_generation=0,
        source_track_id=None,
        object_class="vehicle",
        lifecycle_completion_reason="known_good_check",
        crop_role="primary",
        crop_rank=1,
        frame_index=0,
        timestamp_sec=0.0,
        vehicle_crop_path=str(known_good),
        full_frame_path=None,
        selection_score=1.0,
    )
    processor = PlateDiagnosticProcessor(
        detector_stage=detector,
        plate_config=detector.config,
        diagnostic_config=diagnostic_config,
        output_dir=run_dir,
    )
    attempt = processor.process_job(job, attempt_number=999)
    return {
        "status": "ran",
        "known_good_input_path": str(known_good),
        "raw_boxes": attempt.raw_box_count,
        "accepted_boxes": attempt.accepted_plate_count,
        "best_confidence": max((box.raw_confidence for box in attempt.raw_boxes), default=None),
        "classes": sorted({str(box.raw_class_name or box.raw_class_id) for box in attempt.raw_boxes}),
    }


def run_synthetic(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = _make_run_dir(args, "synthetic")
    return _run_jobs(args, _synthetic_jobs(run_dir), run_dir, fake=True, source_path="synthetic")


def run_existing_step6(args: argparse.Namespace) -> dict[str, Any]:
    step6_run_dir = _resolve_existing_path(args.step6_run_dir, DEFAULT_STEP6_ARTIFACT_RUN)
    run_dir = _make_run_dir(args, "existing_step6")
    _copy_step6_context(step6_run_dir, run_dir)
    jobs = read_selected_crop_jobs(step6_run_dir)
    if args.max_tracks:
        allowed = {key for key in sorted({(job.source_id, job.track_id, job.track_generation) for job in jobs})[: args.max_tracks]}
        jobs = [job for job in jobs if (job.source_id, job.track_id, job.track_generation) in allowed]
    return _run_jobs(args, jobs, run_dir, fake=False, source_path=str(step6_run_dir))


def run_existing_step7(args: argparse.Namespace) -> dict[str, Any]:
    step7_run_dir = _resolve_existing_path(args.step7_run_dir, DEFAULT_STEP7_ARTIFACT_RUN)
    run_dir = _make_run_dir(args, "existing_step7")
    jobs = read_selected_crop_jobs(step7_run_dir / "07_anpr" / "step7_selected_crop_jobs.jsonl")
    if args.max_tracks:
        allowed = {key for key in sorted({(job.source_id, job.track_id, job.track_generation) for job in jobs})[: args.max_tracks]}
        jobs = [job for job in jobs if (job.source_id, job.track_id, job.track_generation) in allowed]
    return _run_jobs(args, jobs, run_dir, fake=False, source_path=str(step7_run_dir))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Step 7.5 plate detector diagnostics on Step 6 selected crops.")
    parser.add_argument("--mode", choices=("synthetic_plate_diagnostics", "existing_step6_artifacts", "existing_step7_artifacts"), default="synthetic_plate_diagnostics")
    parser.add_argument("--step6-run-dir", default=str(DEFAULT_STEP6_ARTIFACT_RUN))
    parser.add_argument("--step7-run-dir", default=str(DEFAULT_STEP7_ARTIFACT_RUN))
    parser.add_argument("--plate-detector-model", default=os.environ.get("TD_CASE2_STREAM_PLATE_MODEL_PATH", "OCR_MUKUL/license_plate_weights.pt"))
    parser.add_argument("--florence-model", default=os.environ.get("TD_CASE2_STREAM_FLORENCE_MODEL_PATH") or os.environ.get("TD_CASE2_FLORENCE_MODEL_PATH"))
    parser.add_argument("--florence-adapter", default=os.environ.get("TD_CASE2_STREAM_FLORENCE_ADAPTER_PATH") or "OCR_MUKUL/adaptor_florance_baseFT")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--normal-plate-confidence", type=float, default=0.20)
    parser.add_argument("--diagnostic-thresholds", default="0.25,0.15,0.10,0.05")
    parser.add_argument("--minimum-box-confidence-to-record", type=float, default=0.0)
    parser.add_argument("--minimum-plate-width", type=int, default=6)
    parser.add_argument("--minimum-plate-height", type=int, default=4)
    parser.add_argument("--max-attempts-per-track", type=int, default=4)
    parser.add_argument("--run-ocr", action="store_true")
    parser.add_argument("--stop-after-first-plate", action="store_true")
    parser.add_argument("--stop-after-first-non-empty-ocr", action="store_true")
    parser.add_argument("--save-annotations", action="store_true")
    parser.add_argument("--save-rejected-plate-crops", action="store_true")
    parser.add_argument("--output-root", default="debug_runs/streaming_tracking_pipeline")
    parser.add_argument("--plate-iou", type=float, default=0.45)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--max-plate-candidates", type=int, default=3)
    parser.add_argument("--maximum-raw-boxes-per-attempt", type=int, default=50)
    parser.add_argument("--plate-crop-padding-ratio", type=float, default=0.08)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--max-tracks", type=int, default=0)
    parser.add_argument("--known-good-input", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.mode == "synthetic_plate_diagnostics":
            result = run_synthetic(args)
        elif args.mode == "existing_step6_artifacts":
            result = run_existing_step6(args)
        else:
            result = run_existing_step7(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "mode": args.mode, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
