from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..detection.detection_models import DetectionPacket


class SupervisionDetectionConversionError(ValueError):
    """Raised when a detection packet cannot be converted into valid Supervision detections."""


@dataclass(frozen=True, slots=True)
class SupervisionDetectionDebugSnapshot:
    input_detection_count: int
    input_boxes: list[list[float]]
    input_confidences: list[float]
    input_class_ids: list[int]


def _empty_xyxy() -> np.ndarray:
    return np.empty((0, 4), dtype=np.float32)


def _empty_confidence() -> np.ndarray:
    return np.empty((0,), dtype=np.float32)


def _empty_class_id() -> np.ndarray:
    return np.empty((0,), dtype=np.int32)


def to_supervision_detections(packet: DetectionPacket):
    import supervision as sv  # type: ignore

    if not packet.detections:
        return sv.Detections(xyxy=_empty_xyxy(), confidence=_empty_confidence(), class_id=_empty_class_id())

    boxes: list[list[float]] = []
    confidences: list[float] = []
    class_ids: list[int] = []
    for index, detection in enumerate(packet.detections):
        x1, y1, x2, y2 = [float(value) for value in detection.bbox_xyxy]
        if not np.isfinite([x1, y1, x2, y2]).all():
            raise SupervisionDetectionConversionError(f"Detection {index} has non-finite bbox values.")
        if x2 <= x1:
            raise SupervisionDetectionConversionError(f"Detection {index} must satisfy x2 > x1.")
        if y2 <= y1:
            raise SupervisionDetectionConversionError(f"Detection {index} must satisfy y2 > y1.")
        confidence = float(detection.confidence)
        if not np.isfinite(confidence):
            raise SupervisionDetectionConversionError(f"Detection {index} confidence must be finite.")
        if not 0.0 <= confidence <= 1.0:
            raise SupervisionDetectionConversionError(f"Detection {index} confidence must be between 0 and 1.")
        boxes.append([x1, y1, x2, y2])
        confidences.append(confidence)
        class_ids.append(int(detection.class_id))

    xyxy = np.asarray(boxes, dtype=np.float32)
    confidence_array = np.asarray(confidences, dtype=np.float32)
    class_id_array = np.asarray(class_ids, dtype=np.int32)
    if xyxy.shape != (len(packet.detections), 4):
        raise SupervisionDetectionConversionError(f"xyxy must have shape (N, 4); got {xyxy.shape}.")
    if confidence_array.shape != (len(packet.detections),):
        raise SupervisionDetectionConversionError("confidence must align with the detection count.")
    if class_id_array.shape != (len(packet.detections),):
        raise SupervisionDetectionConversionError("class_id must align with the detection count.")
    return sv.Detections(xyxy=xyxy, confidence=confidence_array, class_id=class_id_array)


def build_supervision_debug_snapshot(packet: DetectionPacket) -> SupervisionDetectionDebugSnapshot:
    return SupervisionDetectionDebugSnapshot(
        input_detection_count=len(packet.detections),
        input_boxes=[[float(value) for value in detection.bbox_xyxy] for detection in packet.detections],
        input_confidences=[float(detection.confidence) for detection in packet.detections],
        input_class_ids=[int(detection.class_id) for detection in packet.detections],
    )
