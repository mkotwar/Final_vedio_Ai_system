from __future__ import annotations

import json
import math
import re
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

from .anpr_schemas import FlorenceOcrResult, PlateDetectionCandidate
from .config import FlorenceConfig, PlateDetectionConfig, PlateDiagnosticConfig
from .crop_selection import SelectedCropJob
from .florence_inference import FlorenceInferenceEngine
from .plate_detection import UltralyticsPlateDetectionStage
from .plate_diagnostics import PlateDiagnosticProcessor, RawPlateBoxDiagnostic
from .serialization import dataclass_to_dict, to_json_safe, write_json
from .validation import validate_positive_int, validate_probability


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_FINAL_STATUSES = {
    "plate_found_ocr_non_empty",
    "plate_found_ocr_empty",
    "plate_found_ocr_not_run",
    "no_raw_detector_boxes",
    "all_boxes_rejected",
    "image_unreadable",
    "detector_error",
    "florence_error",
}


@dataclass(frozen=True)
class ImageInputRecord:
    input_path: str
    relative_path: str
    filename: str
    width: int | None
    height: int | None
    readable: bool
    error_message: str | None

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass(frozen=True)
class ImageAnprValidationConfig:
    input_dir: str
    recursive: bool = False
    normal_plate_confidence: float = 0.25
    diagnostic_thresholds: tuple[float, ...] = (0.25, 0.15, 0.10, 0.05)
    minimum_plate_width: int = 6
    minimum_plate_height: int = 4
    maximum_plate_candidates_per_image: int = 3
    save_annotations: bool = True
    save_accepted_plate_crops: bool = True
    save_rejected_plate_crops: bool = True
    run_florence_ocr: bool = False
    direct_ocr_on_input: bool = False
    maximum_ocr_candidates_per_image: int = 3
    stop_after_first_non_empty_ocr: bool = True
    max_images: int = 0

    def __post_init__(self) -> None:
        validate_probability(self.normal_plate_confidence, "image_anpr.normal_plate_confidence")
        if not self.diagnostic_thresholds:
            raise ValueError("image_anpr.diagnostic_thresholds must not be empty.")
        for threshold in self.diagnostic_thresholds:
            validate_probability(threshold, "image_anpr.diagnostic_thresholds")
        if len(set(self.diagnostic_thresholds)) != len(self.diagnostic_thresholds):
            raise ValueError("image_anpr.diagnostic_thresholds must be unique.")
        validate_positive_int(self.minimum_plate_width, "image_anpr.minimum_plate_width")
        validate_positive_int(self.minimum_plate_height, "image_anpr.minimum_plate_height")
        validate_positive_int(self.maximum_plate_candidates_per_image, "image_anpr.maximum_plate_candidates_per_image")
        validate_positive_int(self.maximum_ocr_candidates_per_image, "image_anpr.maximum_ocr_candidates_per_image")
        if self.max_images < 0:
            raise ValueError("image_anpr.max_images must be non-negative.")


@dataclass(frozen=True)
class ImageAnprDiagnosticResult:
    input_path: str
    relative_path: str
    filename: str
    image_width: int | None
    image_height: int | None
    image_read_status: str
    raw_detector_box_count: int
    accepted_plate_count: int
    raw_boxes: list[RawPlateBoxDiagnostic]
    accepted_candidates: list[PlateDetectionCandidate]
    ocr_results: list[FlorenceOcrResult]
    best_raw_ocr_text: str | None
    final_status: str
    failure_reasons: list[str]
    annotated_image_path: str | None
    runtime_sec: float
    metadata: dict[str, Any] = field(default_factory=dict)
    direct_input_ocr_result: FlorenceOcrResult | None = None

    def __post_init__(self) -> None:
        if self.final_status not in IMAGE_FINAL_STATUSES:
            raise ValueError(f"final_status must be one of: {sorted(IMAGE_FINAL_STATUSES)}")

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


def sanitize_output_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return sanitized.strip("._") or "image"


def discover_image_inputs(input_dir: str | Path, *, recursive: bool = False, max_images: int = 0) -> list[ImageInputRecord]:
    root = Path(input_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Input image directory not found: {input_dir}")
    candidates = root.rglob("*") if recursive else root.glob("*")
    paths = sorted(path for path in candidates if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS)
    if max_images:
        paths = paths[:max_images]
    records: list[ImageInputRecord] = []
    for path in paths:
        width = height = None
        readable = False
        error = None
        try:
            from PIL import Image

            with Image.open(path) as image:
                width, height = image.size
                image.verify()
            readable = True
        except Exception as exc:
            error = str(exc)
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = path.name
        records.append(
            ImageInputRecord(
                input_path=str(path),
                relative_path=relative,
                filename=path.name,
                width=width,
                height=height,
                readable=readable,
                error_message=error,
            )
        )
    return records


class ImageAnprValidator:
    def __init__(
        self,
        *,
        config: ImageAnprValidationConfig,
        plate_detector: UltralyticsPlateDetectionStage,
        output_dir: str | Path,
        florence_engine: FlorenceInferenceEngine | None = None,
    ) -> None:
        self.config = config
        self.plate_detector = plate_detector
        self.output_dir = Path(output_dir)
        self.florence_engine = florence_engine
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("annotated_images", "accepted_plate_crops", "rejected_plate_crops", "reports"):
            (self.output_dir / name).mkdir(parents=True, exist_ok=True)

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        records = discover_image_inputs(self.config.input_dir, recursive=self.config.recursive, max_images=self.config.max_images)
        results = [self.process_record(index, record) for index, record in enumerate(records, start=1)]
        summary = build_image_anpr_summary(records, results, total_runtime_sec=time.perf_counter() - started)
        self.write_artifacts(records, results, summary)
        return {"records": records, "results": results, "summary": summary}

    def process_record(self, image_index: int, record: ImageInputRecord) -> ImageAnprDiagnosticResult:
        started = time.perf_counter()
        if not record.readable:
            return ImageAnprDiagnosticResult(
                input_path=record.input_path,
                relative_path=record.relative_path,
                filename=record.filename,
                image_width=record.width,
                image_height=record.height,
                image_read_status="unreadable",
                raw_detector_box_count=0,
                accepted_plate_count=0,
                raw_boxes=[],
                accepted_candidates=[],
                ocr_results=[],
                best_raw_ocr_text=None,
                final_status="image_unreadable",
                failure_reasons=[record.error_message or "image_unreadable"],
                annotated_image_path=None,
                runtime_sec=round(time.perf_counter() - started, 6),
            )
        job = SelectedCropJob(
            source_id="image_anpr_validation",
            track_id=image_index,
            track_generation=0,
            source_track_id=record.relative_path,
            object_class=_infer_input_type(record),
            lifecycle_completion_reason="image_validation",
            crop_role="input",
            crop_rank=1,
            frame_index=image_index,
            timestamp_sec=0.0,
            vehicle_crop_path=record.input_path,
            full_frame_path=record.input_path,
            selection_score=1.0,
            metadata={"filename": record.filename, "relative_path": record.relative_path},
        )
        processor = PlateDiagnosticProcessor(
            detector_stage=self.plate_detector,
            plate_config=self._plate_config(),
            diagnostic_config=self._diagnostic_config(),
            output_dir=self.output_dir,
            florence_engine=None,
        )
        attempt = processor.process_job(job, attempt_number=1)
        raw_boxes = self._copy_raw_box_paths(record, attempt.raw_boxes)
        candidates = self._copy_candidate_paths(record, attempt.accepted_candidates)
        annotated = self._copy_annotation(record, attempt.metadata.get("annotated_vehicle_crop_path"))
        ocr_results = self._run_crop_ocr(candidates)
        direct_ocr = self._run_direct_ocr(record, image_index)
        final_status = _final_status(attempt, ocr_results, direct_ocr, self.config)
        failures = _failure_reasons(attempt)
        best_text = next((result.raw_text for result in ocr_results if result.normalized_text), None)
        return ImageAnprDiagnosticResult(
            input_path=record.input_path,
            relative_path=record.relative_path,
            filename=record.filename,
            image_width=record.width,
            image_height=record.height,
            image_read_status="readable",
            raw_detector_box_count=attempt.raw_box_count,
            accepted_plate_count=len(candidates),
            raw_boxes=raw_boxes,
            accepted_candidates=candidates,
            ocr_results=ocr_results,
            direct_input_ocr_result=direct_ocr,
            best_raw_ocr_text=best_text,
            final_status=final_status,
            failure_reasons=failures,
            annotated_image_path=annotated,
            runtime_sec=round(time.perf_counter() - started, 6),
            metadata={
                "input_type": _infer_input_type(record),
                "plate_model_calls": 1,
                "vehicle_crop": attempt.metadata.get("vehicle_crop", {}),
                "best_raw_confidence": max((box.raw_confidence for box in raw_boxes), default=None),
                "classification": _classify_image(record, attempt, ocr_results, direct_ocr),
            },
        )

    def _plate_config(self) -> PlateDetectionConfig:
        cfg = self.plate_detector.config
        return PlateDetectionConfig(
            enabled=cfg.enabled,
            model_path=cfg.model_path,
            confidence_threshold=self.config.normal_plate_confidence,
            iou_threshold=cfg.iou_threshold,
            device=cfg.device,
            image_size=cfg.image_size,
            max_plate_detections_per_vehicle_crop=self.config.maximum_plate_candidates_per_image,
            minimum_plate_width=self.config.minimum_plate_width,
            minimum_plate_height=self.config.minimum_plate_height,
            crop_padding_ratio=cfg.crop_padding_ratio,
            save_plate_crops=cfg.save_plate_crops,
        )

    def _diagnostic_config(self) -> PlateDiagnosticConfig:
        return PlateDiagnosticConfig(
            diagnostic_confidence_thresholds=self.config.diagnostic_thresholds,
            save_annotated_vehicle_crops=self.config.save_annotations,
            save_valid_plate_crops=self.config.save_accepted_plate_crops,
            save_rejected_plate_crops=self.config.save_rejected_plate_crops,
            run_ocr_on_valid_plate_candidates=False,
            stop_after_first_non_empty_ocr_text=self.config.stop_after_first_non_empty_ocr,
            maximum_raw_boxes_per_attempt=100,
        )

    def _run_crop_ocr(self, candidates: list[PlateDetectionCandidate]) -> list[FlorenceOcrResult]:
        if not self.config.run_florence_ocr or self.florence_engine is None:
            return []
        results: list[FlorenceOcrResult] = []
        for candidate in candidates[: self.config.maximum_ocr_candidates_per_image]:
            result = self.florence_engine.run_ocr(candidate)
            results.append(result)
            if self.config.stop_after_first_non_empty_ocr and result.normalized_text:
                break
        return results

    def _run_direct_ocr(self, record: ImageInputRecord, image_index: int) -> FlorenceOcrResult | None:
        if not self.config.direct_ocr_on_input or self.florence_engine is None:
            return None
        candidate = PlateDetectionCandidate(
            source_id="image_anpr_validation",
            track_id=image_index,
            track_generation=0,
            crop_role="direct_input",
            crop_rank=0,
            frame_index=image_index,
            vehicle_crop_path=record.input_path,
            plate_rank=0,
            confidence=1.0,
            bbox_xyxy=(0.0, 0.0, float(record.width or 1), float(record.height or 1)),
            padded_bbox_xyxy=(0, 0, int(record.width or 1), int(record.height or 1)),
            plate_crop_path=record.input_path,
            metadata={"direct_input_ocr": True, "relative_path": record.relative_path},
        )
        return self.florence_engine.run_ocr(candidate)

    def _copy_annotation(self, record: ImageInputRecord, source_path: Any) -> str | None:
        if not source_path:
            return None
        source = Path(str(source_path))
        if not source.exists():
            return None
        target = self.output_dir / "annotated_images" / f"{_unique_name(record)}.jpg"
        shutil.copyfile(source, target)
        return str(target)

    def _copy_candidate_paths(self, record: ImageInputRecord, candidates: list[PlateDetectionCandidate]) -> list[PlateDetectionCandidate]:
        copied: list[PlateDetectionCandidate] = []
        for index, candidate in enumerate(candidates, start=1):
            crop_path = candidate.plate_crop_path
            if crop_path and Path(crop_path).exists() and self.config.save_accepted_plate_crops:
                target = self.output_dir / "accepted_plate_crops" / f"{_unique_name(record)}_plate_{index:02d}.jpg"
                shutil.copyfile(crop_path, target)
                crop_path = str(target)
            copied.append(replace(candidate, plate_crop_path=crop_path))
        return copied

    def _copy_raw_box_paths(self, record: ImageInputRecord, raw_boxes: list[RawPlateBoxDiagnostic]) -> list[RawPlateBoxDiagnostic]:
        copied: list[RawPlateBoxDiagnostic] = []
        rejected_index = 0
        for raw_box in raw_boxes:
            crop_path = raw_box.plate_crop_path
            if crop_path and Path(crop_path).exists() and "rejected" in str(crop_path).lower():
                rejected_index += 1
                target = self.output_dir / "rejected_plate_crops" / f"{_unique_name(record)}_{raw_box.disposition.value}_{rejected_index:02d}.jpg"
                shutil.copyfile(crop_path, target)
                crop_path = str(target)
            copied.append(replace(raw_box, plate_crop_path=crop_path))
        return copied

    def write_artifacts(self, records: list[ImageInputRecord], results: list[ImageAnprDiagnosticResult], summary: dict[str, Any]) -> None:
        write_json(self.output_dir / "input_manifest.json", [record.to_dict() for record in records])
        _write_jsonl(self.output_dir / "image_results.jsonl", results)
        _write_jsonl(self.output_dir / "raw_plate_box_diagnostics.jsonl", [box for result in results for box in result.raw_boxes])
        _write_jsonl(self.output_dir / "florence_ocr_results.jsonl", [ocr for result in results for ocr in result.ocr_results + ([result.direct_input_ocr_result] if result.direct_input_ocr_result else [])])
        write_json(self.output_dir / "reports" / "image_anpr_summary.json", summary)
        write_json(self.output_dir / "reports" / "image_anpr_report.json", summary)


def build_image_anpr_summary(records: list[ImageInputRecord], results: list[ImageAnprDiagnosticResult], *, total_runtime_sec: float) -> dict[str, Any]:
    final_status_counts = Counter(result.final_status for result in results)
    by_extension: dict[str, Counter[str]] = defaultdict(Counter)
    by_size: dict[str, Counter[str]] = defaultdict(Counter)
    by_confidence: dict[str, Counter[str]] = defaultdict(Counter)
    rejected = Counter()
    ocr_results = [ocr for result in results for ocr in result.ocr_results]
    direct_results = [result.direct_input_ocr_result for result in results if result.direct_input_ocr_result is not None]
    for result in results:
        ext = Path(result.filename).suffix.lower()
        by_extension[ext][result.final_status] += 1
        by_size[_size_bucket(result.image_width, result.image_height)][result.final_status] += 1
        best_conf = max((box.raw_confidence for box in result.raw_boxes), default=None)
        by_confidence[_confidence_bucket(best_conf)][result.final_status] += 1
        for box in result.raw_boxes:
            if box.disposition.value != "accepted":
                rejected[box.rejection_reason or box.disposition.value] += 1
    raw_boxes = sum(result.raw_detector_box_count for result in results)
    accepted = sum(result.accepted_plate_count for result in results)
    return {
        "images_discovered": len(records),
        "images_read_successfully": sum(1 for record in records if record.readable),
        "images_unreadable": sum(1 for record in records if not record.readable),
        "plate_model_calls": sum(1 for result in results if result.image_read_status == "readable"),
        "images_with_raw_boxes": sum(1 for result in results if result.raw_detector_box_count > 0),
        "images_with_accepted_plates": sum(1 for result in results if result.accepted_plate_count > 0),
        "images_without_raw_boxes": sum(1 for result in results if result.image_read_status == "readable" and result.raw_detector_box_count == 0),
        "images_with_all_boxes_rejected": sum(1 for result in results if result.raw_detector_box_count > 0 and result.accepted_plate_count == 0),
        "raw_detector_boxes": raw_boxes,
        "boxes_below_normal_threshold": sum(1 for result in results for box in result.raw_boxes if not box.metadata.get("passes_normal_threshold", False)),
        "boxes_wrong_class": sum(1 for result in results for box in result.raw_boxes if box.disposition.value == "wrong_class"),
        "boxes_invalid_geometry": sum(1 for result in results for box in result.raw_boxes if box.disposition.value == "invalid_geometry"),
        "boxes_empty_after_clipping": sum(1 for result in results for box in result.raw_boxes if box.disposition.value == "empty_after_clipping"),
        "boxes_too_small": sum(1 for result in results for box in result.raw_boxes if box.disposition.value == "too_small"),
        "accepted_plate_candidates": accepted,
        "accepted_plate_crops_saved": sum(1 for result in results for candidate in result.accepted_candidates if candidate.plate_crop_path),
        "rejected_plate_crops_saved": sum(1 for result in results for box in result.raw_boxes if box.plate_crop_path and box.disposition.value != "accepted"),
        "ocr_calls": len(ocr_results),
        "ocr_non_empty_outputs": sum(1 for result in ocr_results if result.normalized_text),
        "ocr_empty_outputs": sum(1 for result in ocr_results if not result.normalized_text and result.status != "inference_error"),
        "ocr_failures": sum(1 for result in ocr_results if result.status == "inference_error"),
        "direct_input_ocr_calls": len(direct_results),
        "direct_input_ocr_non_empty_outputs": sum(1 for result in direct_results if result.normalized_text),
        "images_with_non_empty_ocr": sum(1 for result in results if any(ocr.normalized_text for ocr in result.ocr_results) or (result.direct_input_ocr_result and result.direct_input_ocr_result.normalized_text)),
        "images_without_non_empty_ocr": sum(1 for result in results if not any(ocr.normalized_text for ocr in result.ocr_results) and not (result.direct_input_ocr_result and result.direct_input_ocr_result.normalized_text)),
        "average_detector_runtime_sec": _average([result.runtime_sec for result in results if result.image_read_status == "readable"]),
        "average_ocr_runtime_sec": None,
        "total_runtime_sec": round(total_runtime_sec, 6),
        "final_status_counts": dict(final_status_counts),
        "rejection_reason_counts": dict(rejected),
        "by_file_extension": {key: dict(value) for key, value in sorted(by_extension.items())},
        "by_image_size_bucket": {key: dict(value) for key, value in sorted(by_size.items())},
        "by_best_raw_confidence_bucket": {key: dict(value) for key, value in sorted(by_confidence.items())},
        "per_image_table": [_table_row(result) for result in results],
    }


def _write_jsonl(path: Path, values: Iterable[Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(to_json_safe(value), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _unique_name(record: ImageInputRecord) -> str:
    return sanitize_output_name(record.relative_path.replace("/", "__"))


def _infer_input_type(record: ImageInputRecord) -> str:
    name = record.filename.lower()
    if "plate" in name:
        return "direct_plate_crop"
    if record.width and record.height and record.width / max(record.height, 1) > 3.0:
        return "wide_vehicle_or_plate_crop"
    return "unknown_input_type"


def _final_status(attempt: Any, ocr_results: list[FlorenceOcrResult], direct_ocr: FlorenceOcrResult | None, config: ImageAnprValidationConfig) -> str:
    if attempt.error_message and "detector" in attempt.attempt_status.value:
        return "detector_error"
    if attempt.accepted_plate_count > 0:
        if config.run_florence_ocr:
            if any(result.status == "inference_error" for result in ocr_results):
                return "florence_error"
            if any(result.normalized_text for result in ocr_results):
                return "plate_found_ocr_non_empty"
            return "plate_found_ocr_empty"
        return "plate_found_ocr_not_run"
    if attempt.raw_box_count == 0:
        return "no_raw_detector_boxes"
    return "all_boxes_rejected"


def _failure_reasons(attempt: Any) -> list[str]:
    reasons = []
    if attempt.error_message:
        reasons.append(attempt.error_message)
    for raw_box in attempt.raw_boxes:
        if raw_box.rejection_reason and raw_box.rejection_reason not in reasons:
            reasons.append(raw_box.rejection_reason)
    if not reasons and attempt.raw_box_count == 0:
        reasons.append("no_raw_detector_boxes")
    return reasons


def _classify_image(record: ImageInputRecord, attempt: Any, ocr_results: list[FlorenceOcrResult], direct_ocr: FlorenceOcrResult | None) -> str:
    if not record.readable:
        return "Unreadable image"
    if attempt.accepted_plate_count > 0 and any(result.normalized_text for result in ocr_results):
        return "Good ANPR input"
    if attempt.accepted_plate_count > 0:
        return "Detector found plate but OCR weak"
    if direct_ocr is not None and direct_ocr.normalized_text:
        return "Direct plate crop only"
    if record.width and record.height and min(record.width, record.height) < 80:
        return "Image too small"
    if attempt.raw_box_count == 0:
        return "Plate visible but detector missed" if "plate" in record.filename.lower() else "Plate not visible"
    return "Plate visible but detector missed"


def _size_bucket(width: int | None, height: int | None) -> str:
    if not width or not height:
        return "unknown"
    area = width * height
    if area < 10000:
        return "small"
    if area < 100000:
        return "medium"
    return "large"


def _confidence_bucket(value: float | None) -> str:
    if value is None:
        return "none"
    if value >= 0.75:
        return "0.75-1.00"
    if value >= 0.50:
        return "0.50-0.74"
    if value >= 0.25:
        return "0.25-0.49"
    return "0.00-0.24"


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _table_row(result: ImageAnprDiagnosticResult) -> dict[str, Any]:
    return {
        "filename": result.filename,
        "dimensions": [result.image_width, result.image_height],
        "raw_box_count": result.raw_detector_box_count,
        "best_raw_confidence": max((box.raw_confidence for box in result.raw_boxes), default=None),
        "accepted_plate_count": result.accepted_plate_count,
        "rejection_reasons": result.failure_reasons,
        "accepted_plate_crop_paths": [candidate.plate_crop_path for candidate in result.accepted_candidates if candidate.plate_crop_path],
        "detector_crop_ocr_text": [ocr.raw_text for ocr in result.ocr_results],
        "direct_input_ocr_text": result.direct_input_ocr_result.raw_text if result.direct_input_ocr_result else None,
        "final_status": result.final_status,
        "classification": result.metadata.get("classification"),
    }
