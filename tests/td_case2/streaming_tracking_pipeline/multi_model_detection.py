from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from .class_normalization import normalize_class_name
from .contracts import DetectionStage
from .schemas import BoundingBox, DetectionPacket, DetectionRecord, FramePacket


@dataclass(frozen=True)
class DetectionModelSpec:
    detector_source: str
    model_path: str
    class_names: dict[int, str]
    model_role: str


class CombinedSequentialDetectionStage:
    """Run vehicle and person detectors on the same frame, then merge detections."""

    def __init__(
        self,
        vehicle_stage: DetectionStage | None,
        person_stage: DetectionStage | None,
        *,
        duplicate_iou_threshold: float = 0.70,
    ) -> None:
        self.vehicle_stage = vehicle_stage
        self.person_stage = person_stage
        self.duplicate_iou_threshold = duplicate_iou_threshold
        self.frames_processed = 0
        self.vehicle_detector_calls = 0
        self.person_detector_calls = 0
        self.merged_detections = 0
        self.class_counts: Counter[str] = Counter()
        self.source_counts: Counter[str] = Counter()

    def process(self, packet: FramePacket) -> DetectionPacket:
        packets: list[DetectionPacket] = []
        if self.vehicle_stage is not None:
            packets.append(self.vehicle_stage.process(packet))
            self.vehicle_detector_calls += 1
        if self.person_stage is not None:
            packets.append(self.person_stage.process(packet))
            self.person_detector_calls += 1
        combined = combine_detection_packets(packet, packets, duplicate_iou_threshold=self.duplicate_iou_threshold)
        self.frames_processed += 1
        self.merged_detections += len(combined.detections)
        self.class_counts.update(item.normalized_class_name for item in combined.detections)
        self.source_counts.update(item.detector_source or "unknown" for item in combined.detections)
        return combined

    def to_dict(self) -> dict[str, Any]:
        vehicle_metrics = self.vehicle_stage.to_dict() if hasattr(self.vehicle_stage, "to_dict") else {}
        person_metrics = self.person_stage.to_dict() if hasattr(self.person_stage, "to_dict") else {}
        return {
            "frames_processed": self.frames_processed,
            "vehicle_detector_calls": self.vehicle_detector_calls,
            "person_detector_calls": self.person_detector_calls,
            "merged_detections": self.merged_detections,
            "class_counts": dict(sorted(self.class_counts.items())),
            "by_detector_source": dict(sorted(self.source_counts.items())),
            "vehicle_detector_metrics": vehicle_metrics,
            "person_detector_metrics": person_metrics,
            "inference_scheduling": "vehicle_yolo_then_person_yolo_per_frame_then_merge",
            "tracking_strategy": "single_ordered_bytetrack_stage_over_merged_detections",
        }


def combine_detection_packets(
    frame: FramePacket,
    packets: list[DetectionPacket],
    *,
    duplicate_iou_threshold: float = 0.70,
) -> DetectionPacket:
    detections: list[DetectionRecord] = []
    for packet in packets:
        if packet.source_id != frame.source_id or packet.frame_index != frame.frame_index:
            raise ValueError("Cannot combine detections from different frames.")
        detections.extend(packet.detections)
    retained = class_aware_duplicate_suppression(detections, iou_threshold=duplicate_iou_threshold)
    return DetectionPacket(
        source_id=frame.source_id,
        frame_index=frame.frame_index,
        timestamp_sec=frame.timestamp_sec,
        frame_width=frame.frame_width,
        frame_height=frame.frame_height,
        detections=retained,
        frame=frame.frame,
    )


def class_aware_duplicate_suppression(detections: list[DetectionRecord], *, iou_threshold: float = 0.70) -> list[DetectionRecord]:
    retained: list[DetectionRecord] = []
    for detection in sorted(detections, key=lambda item: (-item.confidence, item.normalized_class_name, item.bbox.x1)):
        duplicate = False
        for existing in retained:
            if detection.normalized_class_name != existing.normalized_class_name:
                continue
            if _iou(detection.bbox, existing.bbox) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            retained.append(detection)
    return sorted(retained, key=lambda item: (item.normalized_class_name, item.bbox.x1, item.bbox.y1, -item.confidence))


def summarize_detection_sources(detections: list[DetectionRecord]) -> dict[str, Any]:
    return {
        "total_detections": len(detections),
        "by_detector_source": dict(sorted(Counter(item.detector_source or "unknown" for item in detections).items())),
        "by_normalized_class": dict(sorted(Counter(item.normalized_class_name for item in detections).items())),
    }


def normalize_detection_record(record: DetectionRecord, *, detector_source: str | None = None) -> DetectionRecord:
    normalized = normalize_class_name(record.raw_class_name or record.class_name, record.raw_class_id if record.raw_class_id is not None else record.class_id)
    return DetectionRecord(
        bbox=record.bbox,
        confidence=record.confidence,
        class_id=record.class_id,
        class_name=normalized.normalized_class_name,
        raw_class_id=record.raw_class_id if record.raw_class_id is not None else record.class_id,
        raw_class_name=record.raw_class_name or record.class_name,
        normalized_class_name=normalized.normalized_class_name,
        object_group=normalized.object_group,
        detector_source=detector_source or record.detector_source,
    )


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    union = a.area + b.area - intersection
    return 0.0 if union <= 0 else intersection / union
