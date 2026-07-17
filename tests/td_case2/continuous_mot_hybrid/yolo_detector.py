from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tests.td_case2.config import ENV_YOLO_DEVICE
from tests.td_case2.device_manager import resolve_device
from tests.td_case2.step_03b_yolo_detection import _load_yolo_class

from .track_state import object_family_for_class


SUPPORTED_CLASSES = {"person", "car", "motorcycle", "bus", "truck"}


@dataclass(frozen=True)
class YoloModelSpec:
    model_role: str
    model_path: Path


def resolve_model_specs(*, person_model_path: Path | None, object_model_path: Path | None, combined_model_path: Path | None) -> list[YoloModelSpec]:
    specs: list[YoloModelSpec] = []
    if person_model_path is not None and person_model_path.exists():
        specs.append(YoloModelSpec(model_role="person", model_path=person_model_path))
    if object_model_path is not None and object_model_path.exists():
        specs.append(YoloModelSpec(model_role="object_vehicle", model_path=object_model_path))
    if not specs and combined_model_path is not None and combined_model_path.exists():
        specs.append(YoloModelSpec(model_role="combined", model_path=combined_model_path))
    if not specs:
        raise FileNotFoundError("No usable YOLO model path was resolved for continuous_mot_hybrid.")
    return specs


class YoloDetector:
    def __init__(self, *, model_specs: list[YoloModelSpec], confidence: float, iou: float, device: str):
        YOLO = _load_yolo_class()
        self.models = {item.model_role: YOLO(str(item.model_path)) for item in model_specs}
        self.model_specs = model_specs
        self.confidence = confidence
        self.iou = iou
        if device == "auto":
            decision = resolve_device(component_name="continuous_mot_hybrid_yolo", override_env_names=(ENV_YOLO_DEVICE,))
            self.device = decision.ultralytics_device
            self.device_reason = decision.reason
        else:
            self.device = device
            self.device_reason = "explicit"
        self.class_names: dict[str, dict[str, str]] = {}
        for role, model in self.models.items():
            names = getattr(model, "names", {})
            if isinstance(names, dict):
                self.class_names[role] = {str(key): str(value).lower() for key, value in names.items()}
            else:
                self.class_names[role] = {str(index): str(value).lower() for index, value in enumerate(names)}

    def detect(self, *, frame: Any, frame_record: dict[str, Any], scheduler_state: str, detector_reason: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for spec in self.model_specs:
            result_list = self.models[spec.model_role].predict(
                source=frame,
                conf=self.confidence,
                iou=self.iou,
                device=self.device,
                verbose=False,
            )
            if not result_list:
                continue
            result = result_list[0]
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy_list = boxes.xyxy.tolist()
            cls_list = boxes.cls.tolist() if getattr(boxes, "cls", None) is not None else []
            conf_list = boxes.conf.tolist() if getattr(boxes, "conf", None) is not None else []
            for index, bbox_xyxy in enumerate(xyxy_list):
                class_id = int(cls_list[index]) if index < len(cls_list) else -1
                class_name = self.class_names[spec.model_role].get(str(class_id), str(class_id)).lower()
                if class_name not in SUPPORTED_CLASSES:
                    continue
                rows.append(
                    {
                        "detection_id": f"det_{frame_record['processed_frame_index']:06d}_{len(rows) + 1:03d}",
                        "class_id": class_id,
                        "class_name": class_name,
                        "family": object_family_for_class(class_name),
                        "confidence": round(float(conf_list[index]) if index < len(conf_list) else 0.0, 6),
                        "bbox_xyxy": [round(float(value), 3) for value in bbox_xyxy],
                        "source_frame_index": int(frame_record["source_frame_index"]),
                        "processed_frame_index": int(frame_record["processed_frame_index"]),
                        "timestamp_seconds": round(float(frame_record["timestamp_seconds"]), 6),
                        "scheduler_state": scheduler_state,
                        "detector_call_reason": detector_reason,
                        "model_role": spec.model_role,
                    }
                )
        return rows

