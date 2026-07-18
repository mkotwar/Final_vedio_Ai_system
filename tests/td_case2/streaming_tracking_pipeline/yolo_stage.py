"""Ultralytics YOLO detection stage for Step 3 sequential validation."""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any

from .class_normalization import normalize_class_name
from .config import DetectionConfig
from .contracts import validate_detection_packet_matches_frame
from .schemas import BoundingBox, DetectionPacket, DetectionRecord, FramePacket


def _load_yolo_class() -> Any:
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to import ultralytics.YOLO: {exc}") from exc
    return YOLO


class UltralyticsYoloDetectionStage:
    """Run YOLO on one FramePacket at a time and emit DetectionPacket."""

    def __init__(self, config: DetectionConfig, model: Any | None = None) -> None:
        self.config = config
        self._model = model
        self._class_names: dict[int, str] = {}
        self.frames_processed = 0
        self.model_calls = 0
        self.raw_detections = 0
        self.filtered_detections = 0
        self.class_counts: Counter[str] = Counter()
        self.rejected_invalid_boxes = 0
        self.empty_detection_frames = 0
        self._runtime_sec = 0.0
        if model is not None:
            self._class_names = self._extract_class_names(model)

    @property
    def model(self) -> Any:
        if self._model is None:
            if self.config.model_path is None:
                raise ValueError("DetectionConfig.model_path is required when no YOLO model is injected.")
            model_path = Path(self.config.model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"YOLO model path does not exist: {model_path}")
            YOLO = _load_yolo_class()
            self._model = YOLO(str(model_path))
            self._class_names = self._extract_class_names(self._model)
        return self._model

    def process(self, packet: FramePacket) -> DetectionPacket:
        started_at = time.perf_counter()
        self.frames_processed += 1
        self.model_calls += 1
        kwargs: dict[str, Any] = {
            "source": packet.frame,
            "conf": self.config.confidence_threshold,
            "iou": self.config.iou_threshold,
            "verbose": False,
        }
        if self.config.device:
            kwargs["device"] = self.config.device
        if self.config.image_size is not None:
            kwargs["imgsz"] = self.config.image_size
        result_list = self.model.predict(**kwargs)
        detections = self._convert_results(result_list, packet)
        output = DetectionPacket(
            source_id=packet.source_id,
            frame_index=packet.frame_index,
            timestamp_sec=packet.timestamp_sec,
            frame_width=packet.frame_width,
            frame_height=packet.frame_height,
            detections=detections,
            frame=packet.frame,
        )
        if not detections:
            self.empty_detection_frames += 1
        self._runtime_sec += time.perf_counter() - started_at
        validate_detection_packet_matches_frame(packet, output)
        return output

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames_processed": self.frames_processed,
            "model_calls": self.model_calls,
            "raw_detections": self.raw_detections,
            "filtered_detections": self.filtered_detections,
            "class_counts": dict(sorted(self.class_counts.items())),
            "rejected_invalid_boxes": self.rejected_invalid_boxes,
            "empty_detection_frames": self.empty_detection_frames,
            "runtime_sec": round(self._runtime_sec, 6),
            "model_path": self.config.model_path,
            "device": self.config.device,
            "confidence_threshold": self.config.confidence_threshold,
            "iou_threshold": self.config.iou_threshold,
            "allowed_class_ids": list(self.config.allowed_class_ids),
            "allowed_class_names": list(self.config.allowed_class_names),
            "class_filter_policy": "union when either class IDs or names are configured; no filters keeps all classes",
        }

    def _convert_results(self, result_list: Any, packet: FramePacket) -> list[DetectionRecord]:
        if not result_list:
            return []
        result = result_list[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or getattr(boxes, "xyxy", None) is None:
            return []
        xyxy_list = boxes.xyxy.tolist()
        cls_list = boxes.cls.tolist() if getattr(boxes, "cls", None) is not None else []
        conf_list = boxes.conf.tolist() if getattr(boxes, "conf", None) is not None else []
        detections: list[DetectionRecord] = []
        for index, raw_box in enumerate(xyxy_list):
            self.raw_detections += 1
            confidence = float(conf_list[index]) if index < len(conf_list) else 0.0
            class_id = int(cls_list[index]) if index < len(cls_list) else 0
            raw_class_name = self._class_names.get(class_id, str(class_id)).lower()
            normalized = normalize_class_name(raw_class_name, class_id)
            class_name = normalized.normalized_class_name
            if not self._class_allowed(class_id, raw_class_name) and not self._class_allowed(class_id, class_name):
                continue
            try:
                bbox = BoundingBox(float(raw_box[0]), float(raw_box[1]), float(raw_box[2]), float(raw_box[3])).clip(
                    packet.frame_width,
                    packet.frame_height,
                )
            except (IndexError, TypeError, ValueError):
                self.rejected_invalid_boxes += 1
                continue
            detections.append(
                DetectionRecord(
                    bbox=bbox,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                    raw_class_id=class_id,
                    raw_class_name=raw_class_name,
                    normalized_class_name=class_name,
                    object_group=normalized.object_group,
                    detector_source=str(self.config.model_path or "injected_yolo_model"),
                )
            )
            self.filtered_detections += 1
            self.class_counts[class_name] += 1
        return sorted(detections, key=lambda item: (item.class_id, item.bbox.x1, item.bbox.y1, -item.confidence))

    def _class_allowed(self, class_id: int, class_name: str) -> bool:
        allowed_ids = set(self.config.allowed_class_ids)
        allowed_names = {item.lower() for item in self.config.allowed_class_names}
        if not allowed_ids and not allowed_names:
            return True
        return class_id in allowed_ids or class_name.lower() in allowed_names

    @staticmethod
    def _extract_class_names(model: Any) -> dict[int, str]:
        names = getattr(model, "names", {})
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        if isinstance(names, (list, tuple)):
            return {index: str(value) for index, value in enumerate(names)}
        return {}
