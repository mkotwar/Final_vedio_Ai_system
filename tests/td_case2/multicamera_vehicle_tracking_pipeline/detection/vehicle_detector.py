from __future__ import annotations

import logging
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from ..ingestion.frame_packet import FramePacket
from ..persistence.vehicle_class_mapping import normalize_runtime_vehicle_class
from .detection_config import DetectionConfig
from .detection_models import DetectionPacket, VehicleDetection

LOGGER = logging.getLogger(__name__)

NORMALIZED_CLASS_TO_ID = {"3wheeler": 0, "bus": 1, "car": 2, "motorcycle": 3, "truck": 4}


class VehicleDetectorError(RuntimeError):
    """Raised when model loading or inference fails."""


def normalize_vehicle_class(raw_class_name: str | None) -> str | None:
    return normalize_runtime_vehicle_class(raw_class_name)


def resolve_detector_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    try:
        import torch  # type: ignore

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_yolo_class() -> Any:
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise VehicleDetectorError(f"Failed to import ultralytics.YOLO: {exc}") from exc
    return YOLO


def _extract_class_names(model: Any) -> dict[int, str]:
    names = getattr(model, "names", {})
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def _validate_and_clamp_bbox(raw_box: Any, *, frame_width: int, frame_height: int) -> tuple[float, float, float, float] | None:
    try:
        x1, y1, x2, y2 = [float(value) for value in raw_box[:4]]
    except Exception:
        return None
    values = (x1, y1, x2, y2)
    if any(not math.isfinite(value) for value in values):
        return None
    clamped = (
        max(0.0, min(float(frame_width), x1)),
        max(0.0, min(float(frame_height), y1)),
        max(0.0, min(float(frame_width), x2)),
        max(0.0, min(float(frame_height), y2)),
    )
    if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
        return None
    return clamped


class SharedVehicleDetector:
    def __init__(
        self,
        config: DetectionConfig,
        *,
        model: Any | None = None,
        yolo_loader: Callable[[str], Any] | None = None,
    ) -> None:
        self.config = config
        self._model = model
        self._yolo_loader = yolo_loader
        self._class_names = _extract_class_names(model) if model is not None else {}
        self.loaded_model_name: str | None = None
        self.resolved_model_path: str | None = None
        self.device = resolve_detector_device(config.device)
        self.allowed_classes = tuple(config.allowed_classes)
        self.unsupported_class_counts: Counter[str] = Counter()
        self.invalid_box_count = 0
        if model is not None:
            self.loaded_model_name = getattr(model, "ckpt_path", None) or config.model_path
            self.resolved_model_path = self.loaded_model_name

    @property
    def configuration(self) -> dict[str, object]:
        return {
            "model_path": self.config.model_path,
            "fallback_model_path": self.config.fallback_model_path,
            "allow_fallback": self.config.allow_fallback,
            "device": self.device,
            "confidence_threshold": self.config.confidence_threshold,
            "iou_threshold": self.config.iou_threshold,
            "image_size": self.config.image_size,
            "allowed_classes": list(self.allowed_classes),
        }

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = self._load_model()
            self._class_names = _extract_class_names(self._model)
        return self._model

    def _load_model_once(self, model_path: str) -> Any:
        loader = self._yolo_loader
        if loader is None:
            YOLO = _load_yolo_class()
            loader = YOLO
        return loader(model_path)

    def _load_model(self) -> Any:
        primary_model = self.config.model_path
        try:
            model = self._load_model_once(primary_model)
            self.loaded_model_name = primary_model
            self.resolved_model_path = str(Path(primary_model)) if Path(primary_model).exists() else primary_model
            LOGGER.info("Loaded vehicle detector model=%s device=%s", self.loaded_model_name, self.device)
            return model
        except Exception as primary_exc:
            if not self.config.allow_fallback or not self.config.fallback_model_path:
                LOGGER.error("Primary detector model failed model=%s error=%s", primary_model, primary_exc)
                raise VehicleDetectorError(f"Failed to load primary detector model '{primary_model}': {primary_exc}") from primary_exc
            fallback_model = self.config.fallback_model_path
            LOGGER.warning("Primary detector model failed, trying fallback primary=%s fallback=%s error=%s", primary_model, fallback_model, primary_exc)
            try:
                model = self._load_model_once(fallback_model)
                self.loaded_model_name = fallback_model
                self.resolved_model_path = str(Path(fallback_model)) if Path(fallback_model).exists() else fallback_model
                LOGGER.info("Loaded fallback vehicle detector model=%s device=%s", self.loaded_model_name, self.device)
                return model
            except Exception as fallback_exc:
                LOGGER.error("Fallback detector model failed fallback=%s error=%s", fallback_model, fallback_exc)
                raise VehicleDetectorError(
                    f"Failed to load primary model '{primary_model}' and fallback model '{fallback_model}'. "
                    f"Primary error: {primary_exc}; fallback error: {fallback_exc}"
                ) from fallback_exc

    def detect(self, frame_packet: FramePacket) -> DetectionPacket:
        started_at = time.perf_counter()
        try:
            predictions = self.model.predict(
                source=frame_packet.frame,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                imgsz=self.config.image_size,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            LOGGER.error(
                "Vehicle detection inference failed camera_code=%s source_path=%s frame_number=%s error=%s",
                frame_packet.camera_code,
                frame_packet.source_path,
                frame_packet.frame_number,
                exc,
            )
            raise VehicleDetectorError(
                f"Vehicle detector inference failed for camera '{frame_packet.camera_code}' "
                f"frame {frame_packet.frame_number}: {exc}"
            ) from exc

        detections = self._convert_predictions(predictions, frame_packet)
        inference_time_ms = (time.perf_counter() - started_at) * 1000.0
        LOGGER.debug(
            "Vehicle detection complete camera_code=%s frame_number=%s detections=%s inference_time_ms=%.3f",
            frame_packet.camera_code,
            frame_packet.frame_number,
            len(detections),
            inference_time_ms,
        )
        frame_height, frame_width = frame_packet.frame.shape[:2]
        return DetectionPacket(
            camera_code=frame_packet.camera_code,
            camera_name=frame_packet.camera_name,
            source_path=frame_packet.source_path,
            frame_number=frame_packet.frame_number,
            source_fps=frame_packet.source_fps,
            video_time_seconds=frame_packet.video_time_seconds,
            camera_timestamp=frame_packet.camera_timestamp,
            frame_width=int(frame_width),
            frame_height=int(frame_height),
            detections=detections,
            inference_time_ms=inference_time_ms,
            detector_model=self.loaded_model_name or self.config.model_path,
            detector_device=self.device,
            frame=frame_packet.frame,
        )

    def _convert_predictions(self, predictions: Any, frame_packet: FramePacket) -> list[VehicleDetection]:
        if not predictions:
            return []
        result = predictions[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or getattr(boxes, "xyxy", None) is None:
            return []
        xyxy_values = boxes.xyxy.tolist()
        cls_values = boxes.cls.tolist() if getattr(boxes, "cls", None) is not None else []
        conf_values = boxes.conf.tolist() if getattr(boxes, "conf", None) is not None else []
        frame_height, frame_width = frame_packet.frame.shape[:2]
        detections: list[VehicleDetection] = []
        for index, raw_box in enumerate(xyxy_values):
            class_id = int(cls_values[index]) if index < len(cls_values) else 0
            confidence = float(conf_values[index]) if index < len(conf_values) else 0.0
            raw_class_name = self._class_names.get(class_id, str(class_id))
            normalized_class = normalize_vehicle_class(raw_class_name)
            if normalized_class is None or normalized_class not in self.allowed_classes:
                self.unsupported_class_counts[str(raw_class_name).lower()] += 1
                continue
            bbox = _validate_and_clamp_bbox(raw_box, frame_width=frame_width, frame_height=frame_height)
            if bbox is None:
                self.invalid_box_count += 1
                continue
            detections.append(
                VehicleDetection(
                    class_id=NORMALIZED_CLASS_TO_ID[normalized_class],
                    class_name=normalized_class,
                    confidence=confidence,
                    bbox_xyxy=bbox,
                )
            )
        return detections
