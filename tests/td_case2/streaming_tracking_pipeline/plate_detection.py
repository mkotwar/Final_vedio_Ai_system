from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .anpr_schemas import PlateDetectionCandidate
from .config import PlateDetectionConfig
from .crop_selection import SelectedCropJob


def _cv2() -> Any:
    try:
        import cv2  # type: ignore

        return cv2
    except Exception as exc:  # pragma: no cover - exercised by environments without cv2
        raise RuntimeError("OpenCV is required for Step 7 plate crop extraction.") from exc


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_image_path(path_value: str, *, run_dir: str | Path | None = None) -> Path:
    path = Path(path_value)
    candidates = [path]
    if run_dir is not None:
        candidates.append(Path(run_dir) / path)
    candidates.append(_repo_root() / path)
    for candidate in candidates:
        if candidate.is_absolute() and candidate.exists():
            return candidate
        rooted = candidate.resolve()
        if rooted.exists():
            return rooted
    return path


def _extract_prediction_items(results: Any) -> Iterable[tuple[tuple[float, float, float, float], float]]:
    if results is None:
        return []
    if isinstance(results, dict):
        raw_items = results.get("boxes") or results.get("detections") or []
    elif isinstance(results, list) and results and isinstance(results[0], dict):
        raw_items = results
    else:
        first = results[0] if isinstance(results, (list, tuple)) and results else results
        boxes = getattr(first, "boxes", first)
        raw_items = boxes if isinstance(boxes, list) else list(boxes)

    extracted: list[tuple[tuple[float, float, float, float], float]] = []
    for item in raw_items:
        if isinstance(item, dict):
            bbox = item.get("bbox") or item.get("xyxy")
            confidence = item.get("confidence", item.get("conf", 0.0))
        else:
            bbox_value = getattr(item, "xyxy", None)
            if bbox_value is None:
                continue
            try:
                bbox = bbox_value[0].tolist()
            except AttributeError:
                bbox = list(bbox_value[0])
            conf_value = getattr(item, "conf", [0.0])
            try:
                confidence = float(conf_value[0])
            except TypeError:
                confidence = float(conf_value)
        if bbox is None:
            continue
        coords = tuple(float(value) for value in list(bbox)[:4])
        extracted.append((coords, float(confidence)))
    return extracted


class UltralyticsPlateDetectionStage:
    """Sequential, lazy YOLO wrapper for vehicle-crop-local plate detection."""

    def __init__(
        self,
        config: PlateDetectionConfig,
        *,
        output_dir: str | Path,
        model: Any | None = None,
        run_dir: str | Path | None = None,
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.model = model
        self.loaded = model is not None
        self.metrics = {
            "vehicle_crops_seen": 0,
            "vehicle_crops_missing": 0,
            "plate_model_load_errors": 0,
            "plate_candidates_raw": 0,
            "plate_candidates_kept": 0,
            "plate_candidates_rejected": 0,
            "plate_crops_saved": 0,
        }

    def detect(self, job: SelectedCropJob) -> list[PlateDetectionCandidate]:
        self.metrics["vehicle_crops_seen"] += 1
        if not self.config.enabled:
            return []
        crop_path = resolve_image_path(job.vehicle_crop_path, run_dir=self.run_dir)
        if not crop_path.exists():
            self.metrics["vehicle_crops_missing"] += 1
            return []
        image = _cv2().imread(str(crop_path))
        if image is None:
            self.metrics["vehicle_crops_missing"] += 1
            return []
        model = self._ensure_model()
        call_kwargs: dict[str, Any] = {
            "conf": self.config.confidence_threshold,
            "iou": self.config.iou_threshold,
            "imgsz": self.config.image_size,
            "verbose": False,
        }
        if self.config.device != "auto":
            call_kwargs["device"] = self.config.device
        try:
            results = model(image, **call_kwargs)
        except TypeError:
            results = model(image)
        height, width = image.shape[:2]
        candidates = self._build_candidates(job, image, width, height, _extract_prediction_items(results))
        return candidates[: self.config.max_plate_detections_per_vehicle_crop]

    def process_with_diagnostics(self, job: SelectedCropJob, diagnostic_config: Any, *, florence_engine: Any | None = None) -> Any:
        from .plate_diagnostics import PlateDiagnosticProcessor

        processor = PlateDiagnosticProcessor(
            detector_stage=self,
            plate_config=self.config,
            diagnostic_config=diagnostic_config,
            output_dir=self.output_dir,
            florence_engine=florence_engine,
        )
        return processor.process_job(job, attempt_number=1)

    def _ensure_model(self) -> Any:
        if self.model is not None:
            return self.model
        if not self.config.model_path:
            self.metrics["plate_model_load_errors"] += 1
            raise FileNotFoundError("plate_detection.model_path is required when plate detection is enabled.")
        model_path = resolve_image_path(self.config.model_path, run_dir=self.run_dir)
        if not model_path.exists():
            self.metrics["plate_model_load_errors"] += 1
            raise FileNotFoundError(f"Plate detector model not found: {self.config.model_path}")
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency-specific
            self.metrics["plate_model_load_errors"] += 1
            raise RuntimeError("ultralytics is required for real Step 7 plate detection.") from exc
        self.model = YOLO(str(model_path))
        self.loaded = True
        return self.model

    def _build_candidates(
        self,
        job: SelectedCropJob,
        image: Any,
        width: int,
        height: int,
        predictions: Iterable[tuple[tuple[float, float, float, float], float]],
    ) -> list[PlateDetectionCandidate]:
        rows: list[tuple[float, float, float, float, float, tuple[int, int, int, int], str | None]] = []
        for bbox, confidence in predictions:
            self.metrics["plate_candidates_raw"] += 1
            if confidence < self.config.confidence_threshold:
                self.metrics["plate_candidates_rejected"] += 1
                continue
            x1, y1, x2, y2 = bbox
            x1 = max(0.0, min(float(width), x1))
            x2 = max(0.0, min(float(width), x2))
            y1 = max(0.0, min(float(height), y1))
            y2 = max(0.0, min(float(height), y2))
            if x2 <= x1 or y2 <= y1:
                self.metrics["plate_candidates_rejected"] += 1
                continue
            box_width = x2 - x1
            box_height = y2 - y1
            if box_width < self.config.minimum_plate_width or box_height < self.config.minimum_plate_height:
                self.metrics["plate_candidates_rejected"] += 1
                continue
            pad_x = int(round(box_width * self.config.crop_padding_ratio))
            pad_y = int(round(box_height * self.config.crop_padding_ratio))
            padded = (
                max(0, int(round(x1)) - pad_x),
                max(0, int(round(y1)) - pad_y),
                min(width, int(round(x2)) + pad_x),
                min(height, int(round(y2)) + pad_y),
            )
            rows.append((confidence, box_width * box_height, x1, y1, x2, y2, padded, None))
        rows.sort(key=lambda item: (-item[0], -item[1], item[2], item[3], item[4], item[5]))

        candidates: list[PlateDetectionCandidate] = []
        for index, row in enumerate(rows[: self.config.max_plate_detections_per_vehicle_crop], start=1):
            confidence, _area, x1, y1, x2, y2, padded, _ = row
            crop_output = self._write_plate_crop(job, image, padded, index) if self.config.save_plate_crops else None
            if crop_output is not None:
                self.metrics["plate_crops_saved"] += 1
            self.metrics["plate_candidates_kept"] += 1
            candidates.append(
                PlateDetectionCandidate(
                    source_id=job.source_id,
                    track_id=job.track_id,
                    track_generation=job.track_generation,
                    crop_role=job.crop_role,
                    crop_rank=job.crop_rank,
                    frame_index=job.frame_index,
                    vehicle_crop_path=job.vehicle_crop_path,
                    plate_rank=index,
                    confidence=round(float(confidence), 6),
                    bbox_xyxy=(round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)),
                    padded_bbox_xyxy=padded,
                    plate_crop_path=crop_output,
                    metadata={"source_track_id": job.source_track_id},
                )
            )
        return candidates

    def _write_plate_crop(self, job: SelectedCropJob, image: Any, padded: tuple[int, int, int, int], plate_rank: int) -> str | None:
        x1, y1, x2, y2 = padded
        plate_image = image[y1:y2, x1:x2]
        if plate_image.size == 0:
            return None
        directory = self.output_dir / "07_anpr" / "plate_crops" / job.source_id / f"track_{job.track_id:06d}_gen_{job.track_generation:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (
            f"frame_{job.frame_index:08d}_{job.crop_role}_{job.crop_rank:02d}_"
            f"plate_{plate_rank:02d}.jpg"
        )
        if path.exists():
            existing = _cv2().imread(str(path))
            if existing is not None and existing.shape == plate_image.shape and bool((existing == plate_image).all()):
                return str(path)
            raise FileExistsError(f"Refusing to overwrite different plate crop: {path}")
        _cv2().imwrite(str(path), plate_image)
        return str(path)
