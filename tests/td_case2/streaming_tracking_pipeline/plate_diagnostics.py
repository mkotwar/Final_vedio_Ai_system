from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from .anpr_schemas import FlorenceOcrResult, PlateDetectionCandidate
from .config import PlateDetectionConfig, PlateDiagnosticConfig
from .crop_selection import SelectedCropJob
from .florence_inference import FlorenceInferenceEngine
from .plate_detection import _cv2, resolve_image_path
from .schemas import BoundingBox
from .serialization import dataclass_to_dict
from .validation import validate_allowed_value, validate_non_empty_string, validate_non_negative_int, validate_probability


class PlateAttemptStatus(str, Enum):
    PLATE_CANDIDATE_ACCEPTED = "plate_candidate_accepted"
    NO_RAW_DETECTOR_BOXES = "no_raw_detector_boxes"
    ALL_BOXES_BELOW_THRESHOLD = "all_boxes_below_threshold"
    ALL_BOXES_WRONG_CLASS = "all_boxes_wrong_class"
    ALL_BOXES_INVALID_GEOMETRY = "all_boxes_invalid_geometry"
    ALL_BOXES_EMPTY_AFTER_CLIPPING = "all_boxes_empty_after_clipping"
    ALL_BOXES_TOO_SMALL = "all_boxes_too_small"
    PLATE_CROP_WRITE_FAILED = "plate_crop_write_failed"
    VEHICLE_CROP_MISSING = "vehicle_crop_missing"
    VEHICLE_CROP_UNREADABLE = "vehicle_crop_unreadable"
    DETECTOR_DISABLED = "detector_disabled"
    DETECTOR_LOAD_ERROR = "detector_load_error"
    DETECTOR_INFERENCE_ERROR = "detector_inference_error"
    OCR_NOT_REQUESTED = "ocr_not_requested"
    OCR_NOT_ATTEMPTED_NO_PLATE = "ocr_not_attempted_no_plate"
    OCR_SUCCESS_NON_EMPTY = "ocr_success_non_empty"
    OCR_SUCCESS_EMPTY = "ocr_success_empty"
    OCR_INFERENCE_ERROR = "ocr_inference_error"


class PlateBoxDisposition(str, Enum):
    ACCEPTED = "accepted"
    BELOW_THRESHOLD = "below_threshold"
    WRONG_CLASS = "wrong_class"
    INVALID_GEOMETRY = "invalid_geometry"
    EMPTY_AFTER_CLIPPING = "empty_after_clipping"
    TOO_SMALL = "too_small"
    CANDIDATE_LIMIT_EXCEEDED = "candidate_limit_exceeded"
    CROP_WRITE_FAILED = "crop_write_failed"


TRACK_FINAL_STATUSES = {
    "plate_found_ocr_non_empty",
    "plate_found_ocr_empty",
    "plate_found_ocr_not_run",
    "no_plate_candidate",
    "input_failure",
    "detector_failure",
    "disabled",
}


@dataclass(frozen=True)
class RawModelPlateBox:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int | None = None
    class_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RawPlateBoxDiagnostic:
    source_id: str
    track_id: int
    track_generation: int
    source_track_id: str | int | None
    vehicle_crop_role: str
    vehicle_crop_rank: int
    source_frame_index: int
    timestamp_sec: float
    attempt_number: int
    diagnostic_threshold: float
    raw_box_index: int
    raw_bbox_xyxy: list[float]
    clipped_bbox: BoundingBox | None
    raw_confidence: float
    raw_class_id: int | None
    raw_class_name: str | None
    width: float | None
    height: float | None
    area: float | None
    disposition: PlateBoxDisposition
    rejection_reason: str | None
    plate_crop_path: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_non_empty_string(self.source_id, "source_id")
        validate_non_negative_int(self.track_id, "track_id")
        validate_non_negative_int(self.track_generation, "track_generation")
        validate_probability(self.diagnostic_threshold, "diagnostic_threshold")

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass(frozen=True)
class PlateDiagnosticAttempt:
    source_id: str
    track_id: int
    track_generation: int
    source_track_id: str | int | None
    attempt_number: int
    vehicle_crop_role: str
    vehicle_crop_rank: int
    vehicle_crop_path: str
    source_frame_index: int
    timestamp_sec: float
    configured_detection_threshold: float
    diagnostic_thresholds_used: list[float]
    raw_box_count: int
    below_threshold_box_count: int
    invalid_geometry_count: int
    empty_after_clipping_count: int
    too_small_count: int
    accepted_plate_count: int
    raw_boxes: list[RawPlateBoxDiagnostic]
    accepted_candidates: list[PlateDetectionCandidate]
    ocr_results: list[FlorenceOcrResult]
    attempt_status: PlateAttemptStatus
    stop_reason: str | None
    runtime_sec: float
    error_message: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass(frozen=True)
class TrackPlateDiagnosticResult:
    source_id: str
    track_id: int
    track_generation: int
    source_track_id: str | int | None
    object_class: str | None
    attempts: list[PlateDiagnosticAttempt]
    selected_attempt_number: int | None
    selected_plate_candidate: PlateDetectionCandidate | None
    selected_ocr_result: FlorenceOcrResult | None
    final_status: str
    final_failure_reasons: list[str]
    exhausted_selected_crops: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_allowed_value(self.final_status, TRACK_FINAL_STATUSES, "final_status")

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


def model_class_names(model: Any) -> dict[int, str]:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def extract_raw_plate_boxes(results: Any, *, model_names: dict[int, str] | None = None) -> list[RawModelPlateBox]:
    names = model_names or {}
    if results is None:
        return []
    if isinstance(results, dict):
        raw_items = results.get("boxes") or results.get("detections") or []
    elif isinstance(results, list) and results and isinstance(results[0], dict):
        raw_items = results
    else:
        first = results[0] if isinstance(results, (list, tuple)) and results else results
        names = names or getattr(first, "names", {}) or {}
        if isinstance(names, list):
            names = {index: value for index, value in enumerate(names)}
        boxes = getattr(first, "boxes", first)
        if getattr(boxes, "xyxy", None) is not None:
            xyxy_values = boxes.xyxy.tolist()
            conf_values = boxes.conf.tolist() if getattr(boxes, "conf", None) is not None else [0.0] * len(xyxy_values)
            cls_values = boxes.cls.tolist() if getattr(boxes, "cls", None) is not None else [None] * len(xyxy_values)
            return [
                RawModelPlateBox(
                    bbox_xyxy=tuple(float(value) for value in bbox[:4]),
                    confidence=float(confidence),
                    class_id=None if class_id is None else int(class_id),
                    class_name=None if class_id is None else str(names.get(int(class_id), "")) or None,
                )
                for bbox, confidence, class_id in zip(xyxy_values, conf_values, cls_values)
            ]
        raw_items = boxes if isinstance(boxes, list) else list(boxes)

    extracted: list[RawModelPlateBox] = []
    for item in raw_items:
        if isinstance(item, dict):
            bbox = item.get("bbox") or item.get("xyxy")
            confidence = item.get("confidence", item.get("conf", 0.0))
            class_id = item.get("class_id", item.get("cls"))
            class_name = item.get("class_name")
        else:
            bbox_value = getattr(item, "xyxy", None)
            if bbox_value is None:
                continue
            bbox = bbox_value[0].tolist() if hasattr(bbox_value[0], "tolist") else list(bbox_value[0])
            conf_value = getattr(item, "conf", [0.0])
            confidence = float(conf_value[0]) if isinstance(conf_value, (list, tuple)) else float(conf_value)
            cls_value = getattr(item, "cls", [None])
            class_id = cls_value[0] if isinstance(cls_value, (list, tuple)) else cls_value
            class_name = None
        if bbox is None:
            continue
        parsed_class_id = None if class_id is None else int(class_id)
        extracted.append(
            RawModelPlateBox(
                bbox_xyxy=tuple(float(value) for value in list(bbox)[:4]),
                confidence=float(confidence),
                class_id=parsed_class_id,
                class_name=str(class_name) if class_name else names.get(parsed_class_id) if parsed_class_id is not None else None,
            )
        )
    return extracted


def _is_plate_class(raw_box: RawModelPlateBox, names: dict[int, str]) -> bool:
    if not names or len(names) <= 1:
        return True
    name = (raw_box.class_name or "").lower()
    return any(token in name for token in ("plate", "license", "licence", "number"))


def _thresholds(config: PlateDiagnosticConfig, plate_config: PlateDetectionConfig) -> list[float]:
    if not config.run_multiple_confidence_thresholds:
        return [plate_config.confidence_threshold]
    values = list(config.diagnostic_confidence_thresholds)
    return values[: config.maximum_raw_boxes_per_attempt]


def _vehicle_metadata(job: SelectedCropJob, image: Any) -> dict[str, Any]:
    cv2 = _cv2()
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    source_bbox = job.metadata.get("source_bbox")
    edge_touching = False
    if isinstance(source_bbox, list) and len(source_bbox) >= 4:
        edge_touching = min(float(source_bbox[0]), float(source_bbox[1])) <= 1.0
    return {
        "crop_width": width,
        "crop_height": height,
        "crop_area": width * height,
        "brightness": round(float(gray.mean()), 6),
        "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 6),
        "edge_touching": edge_touching,
        "crop_completeness": job.metadata.get("crop_completeness"),
        "selection_score": job.selection_score,
        "quality_warnings": list(job.quality_warnings),
        "step6_metadata": dict(job.metadata),
    }


def size_bucket(width: int, height: int) -> str:
    area = width * height
    if area < 2048:
        return "tiny"
    if area < 8192:
        return "small"
    if area < 32768:
        return "medium"
    return "large"


class PlateDiagnosticProcessor:
    def __init__(
        self,
        *,
        detector_stage: Any,
        plate_config: PlateDetectionConfig,
        diagnostic_config: PlateDiagnosticConfig,
        output_dir: str | Path,
        florence_engine: FlorenceInferenceEngine | None = None,
    ) -> None:
        self.detector_stage = detector_stage
        self.plate_config = plate_config
        self.config = diagnostic_config
        self.output_dir = Path(output_dir)
        self.florence_engine = florence_engine
        self.metrics = {
            "plate_model_calls": 0,
            "threshold_probe_calls": 0,
            "annotated_images_written": 0,
            "annotation_write_failures": 0,
            "rejected_plate_crops_written": 0,
            "accepted_plate_crops_written": 0,
        }

    def process_job(self, job: SelectedCropJob, *, attempt_number: int) -> PlateDiagnosticAttempt:
        started = time.perf_counter()
        thresholds = _thresholds(self.config, self.plate_config)
        acceptance_threshold = min(thresholds) if thresholds else self.plate_config.confidence_threshold
        if not self.config.enabled:
            return self._empty_attempt(job, attempt_number, thresholds, PlateAttemptStatus.DETECTOR_DISABLED, started)
        if not self.plate_config.enabled:
            return self._empty_attempt(job, attempt_number, thresholds, PlateAttemptStatus.DETECTOR_DISABLED, started)

        crop_path = resolve_image_path(job.vehicle_crop_path, run_dir=getattr(self.detector_stage, "run_dir", None))
        if not crop_path.exists():
            return self._empty_attempt(job, attempt_number, thresholds, PlateAttemptStatus.VEHICLE_CROP_MISSING, started, "vehicle crop missing")
        image = _cv2().imread(str(crop_path))
        if image is None:
            return self._empty_attempt(job, attempt_number, thresholds, PlateAttemptStatus.VEHICLE_CROP_UNREADABLE, started, "vehicle crop unreadable")

        try:
            model = self.detector_stage._ensure_model()
        except Exception as exc:
            return self._empty_attempt(job, attempt_number, thresholds, PlateAttemptStatus.DETECTOR_LOAD_ERROR, started, str(exc), {"vehicle_crop": _vehicle_metadata(job, image)})

        names = model_class_names(model)
        try:
            raw_boxes = self._run_detector(model, crop_path, image, min(thresholds))
        except Exception as exc:
            return self._empty_attempt(job, attempt_number, thresholds, PlateAttemptStatus.DETECTOR_INFERENCE_ERROR, started, str(exc), {"model_names": names, "vehicle_crop": _vehicle_metadata(job, image)})

        raw_boxes = [
            box for box in raw_boxes if box.confidence >= self.config.minimum_box_confidence_to_record
        ][: self.config.maximum_raw_boxes_per_attempt]
        raw_diagnostics: list[RawPlateBoxDiagnostic] = []
        accepted_candidates: list[PlateDetectionCandidate] = []
        height, width = image.shape[:2]
        for raw_index, raw_box in enumerate(raw_boxes, start=1):
            diagnostic, candidate = self._classify_box(
                job=job,
                image=image,
                image_width=width,
                image_height=height,
                raw_box=raw_box,
                raw_box_index=raw_index,
                attempt_number=attempt_number,
                acceptance_threshold=acceptance_threshold,
                model_names=names,
                candidate_rank=len(accepted_candidates) + 1,
            )
            raw_diagnostics.append(diagnostic)
            if candidate is not None:
                if len(accepted_candidates) >= self.plate_config.max_plate_detections_per_vehicle_crop:
                    raw_diagnostics[-1] = self._replace_disposition(diagnostic, PlateBoxDisposition.CANDIDATE_LIMIT_EXCEEDED, "candidate_limit_exceeded")
                else:
                    accepted_candidates.append(candidate)

        ocr_results: list[FlorenceOcrResult] = []
        if self.config.run_ocr_on_valid_plate_candidates and self.florence_engine is not None:
            for candidate in accepted_candidates:
                ocr = self.florence_engine.run_ocr(candidate)
                ocr_results.append(ocr)
                if self.config.stop_after_first_non_empty_ocr_text and ocr.normalized_text:
                    break

        annotated_path = self._write_annotation(job, image, raw_diagnostics, attempt_number) if self.config.save_annotated_vehicle_crops else None
        status = self._attempt_status(raw_diagnostics, accepted_candidates, ocr_results)
        stop_reason = None
        if accepted_candidates and self.config.stop_after_first_valid_plate_candidate:
            stop_reason = "stop_after_first_valid_plate_candidate"
        if any(result.normalized_text for result in ocr_results) and self.config.stop_after_first_non_empty_ocr_text:
            stop_reason = "stop_after_first_non_empty_ocr_text"
        return PlateDiagnosticAttempt(
            source_id=job.source_id,
            track_id=job.track_id,
            track_generation=job.track_generation,
            source_track_id=job.source_track_id,
            attempt_number=attempt_number,
            vehicle_crop_role=job.crop_role,
            vehicle_crop_rank=job.crop_rank,
            vehicle_crop_path=job.vehicle_crop_path,
            source_frame_index=job.frame_index,
            timestamp_sec=job.timestamp_sec,
            configured_detection_threshold=self.plate_config.confidence_threshold,
            diagnostic_thresholds_used=list(thresholds),
            raw_box_count=len(raw_boxes),
            below_threshold_box_count=sum(1 for item in raw_diagnostics if item.raw_confidence < self.plate_config.confidence_threshold),
            invalid_geometry_count=sum(1 for item in raw_diagnostics if item.disposition == PlateBoxDisposition.INVALID_GEOMETRY),
            empty_after_clipping_count=sum(1 for item in raw_diagnostics if item.disposition == PlateBoxDisposition.EMPTY_AFTER_CLIPPING),
            too_small_count=sum(1 for item in raw_diagnostics if item.disposition == PlateBoxDisposition.TOO_SMALL),
            accepted_plate_count=len(accepted_candidates),
            raw_boxes=raw_diagnostics,
            accepted_candidates=accepted_candidates,
            ocr_results=ocr_results,
            attempt_status=status,
            stop_reason=stop_reason,
            runtime_sec=round(time.perf_counter() - started, 6),
            error_message=None,
            metadata={
                "model_names": names,
                "normal_threshold": self.plate_config.confidence_threshold,
                "diagnostic_acceptance_threshold": acceptance_threshold,
                "annotated_vehicle_crop_path": annotated_path,
                "vehicle_crop": _vehicle_metadata(job, image),
                "vehicle_crop_size_bucket": size_bucket(width, height),
            },
        )

    def _run_detector(self, model: Any, crop_path: Path, image: Any, threshold: float) -> list[RawModelPlateBox]:
        self.metrics["plate_model_calls"] += 1
        self.metrics["threshold_probe_calls"] += 1
        kwargs: dict[str, Any] = {
            "conf": threshold,
            "iou": self.plate_config.iou_threshold,
            "imgsz": self.plate_config.image_size,
            "verbose": False,
        }
        if self.plate_config.device != "auto":
            kwargs["device"] = self.plate_config.device
        if hasattr(model, "predict"):
            try:
                results = model.predict(source=str(crop_path), **kwargs)
            except TypeError:
                results = model.predict(str(crop_path), **kwargs)
        else:
            try:
                results = model(image, **kwargs)
            except TypeError:
                results = model(image)
        return extract_raw_plate_boxes(results, model_names=model_class_names(model))

    def _classify_box(
        self,
        *,
        job: SelectedCropJob,
        image: Any,
        image_width: int,
        image_height: int,
        raw_box: RawModelPlateBox,
        raw_box_index: int,
        attempt_number: int,
        acceptance_threshold: float,
        model_names: dict[int, str],
        candidate_rank: int,
    ) -> tuple[RawPlateBoxDiagnostic, PlateDetectionCandidate | None]:
        x1, y1, x2, y2 = raw_box.bbox_xyxy
        clipped_bbox = None
        width = height = area = None
        disposition = PlateBoxDisposition.ACCEPTED
        reason = None
        plate_crop_path = None
        if raw_box.confidence < acceptance_threshold:
            disposition = PlateBoxDisposition.BELOW_THRESHOLD
            reason = "below_diagnostic_acceptance_threshold"
        elif not _is_plate_class(raw_box, model_names):
            disposition = PlateBoxDisposition.WRONG_CLASS
            reason = "class_name_not_plate_like"
        elif x2 <= x1 or y2 <= y1:
            disposition = PlateBoxDisposition.INVALID_GEOMETRY
            reason = "raw_x2_y2_not_greater_than_x1_y1"
        else:
            cx1, cy1 = max(0.0, min(float(image_width), x1)), max(0.0, min(float(image_height), y1))
            cx2, cy2 = max(0.0, min(float(image_width), x2)), max(0.0, min(float(image_height), y2))
            if cx2 <= cx1 or cy2 <= cy1:
                disposition = PlateBoxDisposition.EMPTY_AFTER_CLIPPING
                reason = "empty_after_clipping_to_vehicle_crop"
            else:
                clipped_bbox = BoundingBox(cx1, cy1, cx2, cy2)
                width = clipped_bbox.width
                height = clipped_bbox.height
                area = clipped_bbox.area
                if width < self.plate_config.minimum_plate_width or height < self.plate_config.minimum_plate_height:
                    disposition = PlateBoxDisposition.TOO_SMALL
                    if width < self.plate_config.minimum_plate_width and height < self.plate_config.minimum_plate_height:
                        reason = "width_and_height_below_minimum"
                    elif width < self.plate_config.minimum_plate_width:
                        reason = "width_below_minimum"
                    else:
                        reason = "height_below_minimum"

        if clipped_bbox is not None and disposition == PlateBoxDisposition.ACCEPTED:
            try:
                plate_crop_path = self._save_plate_crop(job, image, clipped_bbox, attempt_number, candidate_rank, "accepted")
                if plate_crop_path is not None:
                    self.metrics["accepted_plate_crops_written"] += 1
            except Exception as exc:
                disposition = PlateBoxDisposition.CROP_WRITE_FAILED
                reason = str(exc)
        elif clipped_bbox is not None and disposition in {PlateBoxDisposition.BELOW_THRESHOLD, PlateBoxDisposition.TOO_SMALL} and self.config.save_rejected_plate_crops:
            try:
                plate_crop_path = self._save_plate_crop(job, image, clipped_bbox, attempt_number, raw_box_index, f"rejected_{disposition.value}")
                if plate_crop_path is not None:
                    self.metrics["rejected_plate_crops_written"] += 1
            except Exception as exc:
                reason = f"{reason}; rejected_crop_write_failed:{exc}" if reason else f"rejected_crop_write_failed:{exc}"

        diagnostic = RawPlateBoxDiagnostic(
            source_id=job.source_id,
            track_id=job.track_id,
            track_generation=job.track_generation,
            source_track_id=job.source_track_id,
            vehicle_crop_role=job.crop_role,
            vehicle_crop_rank=job.crop_rank,
            source_frame_index=job.frame_index,
            timestamp_sec=job.timestamp_sec,
            attempt_number=attempt_number,
            diagnostic_threshold=acceptance_threshold,
            raw_box_index=raw_box_index,
            raw_bbox_xyxy=[round(float(value), 6) for value in raw_box.bbox_xyxy],
            clipped_bbox=clipped_bbox,
            raw_confidence=round(raw_box.confidence, 6),
            raw_class_id=raw_box.class_id,
            raw_class_name=raw_box.class_name,
            width=None if width is None else round(width, 6),
            height=None if height is None else round(height, 6),
            area=None if area is None else round(area, 6),
            disposition=disposition,
            rejection_reason=reason,
            plate_crop_path=plate_crop_path,
            metadata={
                "normal_threshold": self.plate_config.confidence_threshold,
                "passes_normal_threshold": raw_box.confidence >= self.plate_config.confidence_threshold,
                "thresholds_passed": [threshold for threshold in self.config.diagnostic_confidence_thresholds if raw_box.confidence >= threshold],
                "aspect_ratio": None if not width or not height else round(width / height, 6),
                "edge_distances": None
                if clipped_bbox is None
                else {
                    "left": round(clipped_bbox.x1, 6),
                    "top": round(clipped_bbox.y1, 6),
                    "right": round(float(image_width) - clipped_bbox.x2, 6),
                    "bottom": round(float(image_height) - clipped_bbox.y2, 6),
                },
            },
        )
        if disposition != PlateBoxDisposition.ACCEPTED or clipped_bbox is None:
            return diagnostic, None
        candidate = PlateDetectionCandidate(
            source_id=job.source_id,
            track_id=job.track_id,
            track_generation=job.track_generation,
            crop_role=job.crop_role,
            crop_rank=job.crop_rank,
            frame_index=job.frame_index,
            vehicle_crop_path=job.vehicle_crop_path,
            plate_rank=candidate_rank,
            confidence=round(raw_box.confidence, 6),
            bbox_xyxy=tuple(round(value, 3) for value in clipped_bbox.to_xyxy()),
            padded_bbox_xyxy=tuple(int(round(value)) for value in clipped_bbox.to_xyxy()),
            plate_crop_path=plate_crop_path,
            metadata={
                "source_track_id": job.source_track_id,
                "normal_threshold": self.plate_config.confidence_threshold,
                "diagnostic_acceptance_threshold": acceptance_threshold,
                "accepted_by": "normal_threshold" if raw_box.confidence >= self.plate_config.confidence_threshold else "diagnostic_threshold",
                "raw_class_id": raw_box.class_id,
                "raw_class_name": raw_box.class_name,
            },
        )
        return diagnostic, candidate

    def _save_plate_crop(self, job: SelectedCropJob, image: Any, bbox: BoundingBox, attempt_number: int, rank: int, disposition: str) -> str | None:
        if disposition == "accepted" and not self.config.save_valid_plate_crops:
            return None
        x1, y1, x2, y2 = [int(round(value)) for value in bbox.to_xyxy()]
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        root = "accepted_plate_crops" if disposition == "accepted" else "rejected_plate_crops"
        directory = self.output_dir / "07_5_plate_diagnostics" / root / job.source_id / f"track_{job.track_id:06d}_gen_{job.track_generation:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"attempt_{attempt_number:02d}_{job.crop_role}_rank_{job.crop_rank:02d}_{disposition}_box_{rank:02d}.jpg"
        _cv2().imwrite(str(path), crop)
        return str(path)

    def _write_annotation(self, job: SelectedCropJob, image: Any, boxes: list[RawPlateBoxDiagnostic], attempt_number: int) -> str | None:
        cv2 = _cv2()
        annotated = image.copy()
        for box in boxes:
            if box.clipped_bbox is None:
                continue
            x1, y1, x2, y2 = [int(round(value)) for value in box.clipped_bbox.to_xyxy()]
            accepted = box.disposition == PlateBoxDisposition.ACCEPTED
            color = (0, 180, 0) if accepted else (0, 0, 220)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2 if accepted else 1)
            label = f"{box.raw_confidence:.2f} {box.raw_class_name or box.raw_class_id} {box.disposition.value}"
            cv2.putText(annotated, label, (max(0, x1), max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        cv2.putText(
            annotated,
            f"attempt {attempt_number} {job.crop_role} rank {job.crop_rank}",
            (5, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        directory = self.output_dir / "07_5_plate_diagnostics" / "annotated_vehicle_crops" / job.source_id / f"track_{job.track_id:06d}_gen_{job.track_generation:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"attempt_{attempt_number:02d}_{job.crop_role}_rank_{job.crop_rank:02d}.jpg"
        try:
            cv2.imwrite(str(path), annotated)
            self.metrics["annotated_images_written"] += 1
            return str(path)
        except Exception:
            self.metrics["annotation_write_failures"] += 1
            return None

    def _attempt_status(
        self,
        raw_boxes: list[RawPlateBoxDiagnostic],
        accepted: list[PlateDetectionCandidate],
        ocr_results: list[FlorenceOcrResult],
    ) -> PlateAttemptStatus:
        if accepted:
            if self.config.run_ocr_on_valid_plate_candidates:
                if any(result.normalized_text for result in ocr_results):
                    return PlateAttemptStatus.OCR_SUCCESS_NON_EMPTY
                if ocr_results:
                    if any(result.status == "inference_error" for result in ocr_results):
                        return PlateAttemptStatus.OCR_INFERENCE_ERROR
                    return PlateAttemptStatus.OCR_SUCCESS_EMPTY
            return PlateAttemptStatus.PLATE_CANDIDATE_ACCEPTED
        if not raw_boxes:
            return PlateAttemptStatus.NO_RAW_DETECTOR_BOXES
        dispositions = [item.disposition for item in raw_boxes]
        if all(item == PlateBoxDisposition.BELOW_THRESHOLD for item in dispositions):
            return PlateAttemptStatus.ALL_BOXES_BELOW_THRESHOLD
        if all(item == PlateBoxDisposition.WRONG_CLASS for item in dispositions):
            return PlateAttemptStatus.ALL_BOXES_WRONG_CLASS
        if all(item == PlateBoxDisposition.INVALID_GEOMETRY for item in dispositions):
            return PlateAttemptStatus.ALL_BOXES_INVALID_GEOMETRY
        if all(item == PlateBoxDisposition.EMPTY_AFTER_CLIPPING for item in dispositions):
            return PlateAttemptStatus.ALL_BOXES_EMPTY_AFTER_CLIPPING
        if all(item == PlateBoxDisposition.TOO_SMALL for item in dispositions):
            return PlateAttemptStatus.ALL_BOXES_TOO_SMALL
        if any(item == PlateBoxDisposition.CROP_WRITE_FAILED for item in dispositions):
            return PlateAttemptStatus.PLATE_CROP_WRITE_FAILED
        return PlateAttemptStatus.OCR_NOT_ATTEMPTED_NO_PLATE

    def _empty_attempt(
        self,
        job: SelectedCropJob,
        attempt_number: int,
        thresholds: list[float],
        status: PlateAttemptStatus,
        started: float,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PlateDiagnosticAttempt:
        return PlateDiagnosticAttempt(
            source_id=job.source_id,
            track_id=job.track_id,
            track_generation=job.track_generation,
            source_track_id=job.source_track_id,
            attempt_number=attempt_number,
            vehicle_crop_role=job.crop_role,
            vehicle_crop_rank=job.crop_rank,
            vehicle_crop_path=job.vehicle_crop_path,
            source_frame_index=job.frame_index,
            timestamp_sec=job.timestamp_sec,
            configured_detection_threshold=self.plate_config.confidence_threshold,
            diagnostic_thresholds_used=list(thresholds),
            raw_box_count=0,
            below_threshold_box_count=0,
            invalid_geometry_count=0,
            empty_after_clipping_count=0,
            too_small_count=0,
            accepted_plate_count=0,
            raw_boxes=[],
            accepted_candidates=[],
            ocr_results=[],
            attempt_status=status,
            stop_reason=None,
            runtime_sec=round(time.perf_counter() - started, 6),
            error_message=error,
            metadata=metadata or {},
        )

    def _replace_disposition(
        self,
        diagnostic: RawPlateBoxDiagnostic,
        disposition: PlateBoxDisposition,
        reason: str,
    ) -> RawPlateBoxDiagnostic:
        payload = diagnostic.to_dict()
        payload["disposition"] = disposition
        payload["rejection_reason"] = reason
        return RawPlateBoxDiagnostic(**payload)
