from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..enrichment.plate_models import PlateDetection


try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore[assignment]


class PlateDetectorRuntimeError(RuntimeError):
    """Raised when plate detector loading or inference fails."""


@dataclass(frozen=True, slots=True)
class PlateDetectorDependencies:
    yolo_cls: Any


class PlateDetectorRuntime:
    def __init__(
        self,
        *,
        model_path: Path,
        device: str,
        confidence_threshold: float,
        iou_threshold: float,
        maximum_detections_per_vehicle_crop: int,
        dependencies: PlateDetectorDependencies | None = None,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.maximum_detections_per_vehicle_crop = maximum_detections_per_vehicle_crop
        self.dependencies = dependencies or PlateDetectorDependencies(yolo_cls=YOLO)
        self.model = None
        self.loaded = False

    def load(self) -> None:
        if self.loaded:
            return
        if self.dependencies.yolo_cls is None:
            raise PlateDetectorRuntimeError("Ultralytics is not installed.")
        try:
            self.model = self.dependencies.yolo_cls(str(self.model_path))
        except Exception as exc:
            raise PlateDetectorRuntimeError(f"Failed to load plate detector: {exc}") from exc
        self.loaded = True

    def detect(self, image_path: Path) -> list[PlateDetection]:
        self.load()
        if self.model is None:
            raise PlateDetectorRuntimeError("Plate detector model is not loaded.")
        try:
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise PlateDetectorRuntimeError("OpenCV is required for plate detection.") from exc
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Unable to read plate-detector input image: {image_path}")
        height, width = image.shape[:2]
        try:
            results = self.model.predict(
                source=image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=None if self.device == "auto" else self.device,
                verbose=False,
                max_det=self.maximum_detections_per_vehicle_crop,
            )
        except Exception as exc:
            raise PlateDetectorRuntimeError(f"Plate detector inference failed: {exc}") from exc
        detections: list[PlateDetection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            names = getattr(result, "names", {}) or {}
            xyxy = getattr(boxes, "xyxy", [])
            confidences = getattr(boxes, "conf", [])
            classes = getattr(boxes, "cls", [])
            for index, raw_bbox in enumerate(xyxy):
                bbox = raw_bbox.tolist() if hasattr(raw_bbox, "tolist") else list(raw_bbox)
                x1, y1, x2, y2 = _clip_bbox(tuple(float(v) for v in bbox), width=width, height=height)
                if x2 <= x1 or y2 <= y1:
                    continue
                class_id = None
                if len(classes) > index:
                    value = classes[index]
                    class_id = int(value.item() if hasattr(value, "item") else value)
                confidence_value = confidences[index]
                confidence = float(confidence_value.item() if hasattr(confidence_value, "item") else confidence_value)
                class_name = names.get(class_id) if class_id is not None and isinstance(names, dict) else None
                detections.append(
                    PlateDetection(
                        bbox_xyxy=(x1, y1, x2, y2),
                        confidence=confidence,
                        class_id=class_id,
                        class_name=str(class_name) if class_name is not None else None,
                    )
                )
        return detections

    def close(self) -> None:
        self.model = None
        self.loaded = False


def _clip_bbox(bbox_xyxy: tuple[float, float, float, float], *, width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    return (
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    )
