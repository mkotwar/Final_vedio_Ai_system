from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import (
    DEFAULT_YOLO_CONF_THRESHOLD,
    DEFAULT_YOLO_IOU_THRESHOLD,
    ENV_OBJECT_YOLO_MODEL_PATH,
    ENV_PERSON_YOLO_MODEL_PATH,
    ENV_YOLO_DEVICE,
    ENV_YOLO_MODEL_PATH,
    resolve_case_path,
)
from device_manager import resolve_device
from step_03b_yolo_detection import _load_yolo_class


SUPPORTED_CLASS_NAMES = {"person", "car", "motorcycle", "bus", "truck", "bicycle", "auto", "van", "vehicle"}


@dataclass(frozen=True)
class DetectorConfig:
    minimum_detection_confidence: float
    iou_threshold: float
    device: str
    model_specs: list[dict[str, str]]
    class_confidence_thresholds: dict[str, float]


def _resolve_model_path(raw_value: str | None) -> Path | None:
    if not raw_value or not raw_value.strip():
        return None
    candidate = Path(raw_value.strip()).expanduser()
    if not candidate.is_absolute():
        candidate = resolve_case_path(str(candidate))
    return candidate


def build_detector_config(
    *,
    minimum_detection_confidence: float,
    class_confidence_thresholds: dict[str, float] | None = None,
    iou_threshold: float = DEFAULT_YOLO_IOU_THRESHOLD,
    device_override: str | None = None,
) -> DetectorConfig:
    person_model_path = _resolve_model_path(os.environ.get(ENV_PERSON_YOLO_MODEL_PATH))
    object_model_path = _resolve_model_path(os.environ.get(ENV_OBJECT_YOLO_MODEL_PATH))
    combined_model_path = _resolve_model_path(os.environ.get(ENV_YOLO_MODEL_PATH))

    model_specs: list[dict[str, str]] = []
    if person_model_path is not None:
        model_specs.append({"model_role": "person", "model_path": str(person_model_path)})
    if object_model_path is not None:
        model_specs.append({"model_role": "object_vehicle", "model_path": str(object_model_path)})
    if not model_specs and combined_model_path is not None:
        model_specs.append({"model_role": "combined", "model_path": str(combined_model_path)})
    if not model_specs:
        raise FileNotFoundError(
            f"At least one YOLO model must be configured through {ENV_PERSON_YOLO_MODEL_PATH}, "
            f"{ENV_OBJECT_YOLO_MODEL_PATH}, or {ENV_YOLO_MODEL_PATH}."
        )
    missing = [item["model_path"] for item in model_specs if not Path(item["model_path"]).exists()]
    if missing:
        raise FileNotFoundError(f"Configured YOLO model path(s) do not exist: {missing}")

    if device_override:
        device = device_override
    else:
        device = resolve_device(component_name="Hybrid Tracking YOLO", override_env_names=(ENV_YOLO_DEVICE,)).ultralytics_device
    return DetectorConfig(
        minimum_detection_confidence=float(minimum_detection_confidence),
        iou_threshold=float(iou_threshold or DEFAULT_YOLO_IOU_THRESHOLD),
        device=device,
        model_specs=model_specs,
        class_confidence_thresholds={str(key).lower(): float(value) for key, value in (class_confidence_thresholds or {}).items()},
    )


class DetectorAdapter:
    def __init__(self, config: DetectorConfig):
        self.config = config
        YOLO = _load_yolo_class()
        self.models = {item["model_role"]: YOLO(item["model_path"]) for item in self.config.model_specs}
        self.class_lookups: dict[str, dict[str, str]] = {}
        for role, model in self.models.items():
            names = getattr(model, "names", {})
            if isinstance(names, dict):
                self.class_lookups[role] = {str(key): str(value).lower() for key, value in names.items()}
            else:
                self.class_lookups[role] = {str(index): str(value).lower() for index, value in enumerate(names)}

    def detect(self, frame) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        for role, model in self.models.items():
            results = model.predict(
                source=frame,
                conf=self.config.minimum_detection_confidence,
                iou=self.config.iou_threshold,
                device=self.config.device,
                verbose=False,
            )
            if not results:
                continue
            boxes = getattr(results[0], "boxes", None)
            if boxes is None:
                continue
            xyxy_list = boxes.xyxy.tolist()
            cls_list = boxes.cls.tolist() if getattr(boxes, "cls", None) is not None else []
            conf_list = boxes.conf.tolist() if getattr(boxes, "conf", None) is not None else []
            for index, bbox_xyxy in enumerate(xyxy_list):
                class_id = int(cls_list[index]) if index < len(cls_list) else -1
                class_name = self.class_lookups[role].get(str(class_id), str(class_id)).lower()
                if class_name not in SUPPORTED_CLASS_NAMES:
                    continue
                confidence = float(conf_list[index]) if index < len(conf_list) else 0.0
                minimum_class_confidence = float(self.config.class_confidence_thresholds.get(class_name, self.config.minimum_detection_confidence))
                if confidence < minimum_class_confidence:
                    continue
                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox_xyxy": [float(value) for value in bbox_xyxy],
                        "model_source": role,
                    }
                )
        detections.sort(key=lambda item: (str(item["class_name"]), -float(item["confidence"])))
        return detections


def detector_runtime_metadata(config: DetectorConfig) -> dict[str, Any]:
    return {
        "device": config.device,
        "minimum_detection_confidence": float(config.minimum_detection_confidence),
        "iou_threshold": float(config.iou_threshold),
        "model_specs": list(config.model_specs),
        "class_confidence_thresholds": dict(config.class_confidence_thresholds),
        "fallback_default_conf_threshold": DEFAULT_YOLO_CONF_THRESHOLD,
    }
