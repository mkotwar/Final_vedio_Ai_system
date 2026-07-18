from __future__ import annotations

import unittest

from tests.td_case2.streaming_tracking_pipeline.multi_model_detection import (
    CombinedSequentialDetectionStage,
    class_aware_duplicate_suppression,
    combine_detection_packets,
)
from tests.td_case2.streaming_tracking_pipeline.schemas import BoundingBox, DetectionPacket, DetectionRecord, FramePacket


class MultiModelDetectionTests(unittest.TestCase):
    def test_class_aware_suppression_keeps_different_classes(self) -> None:
        detections = [
            _det("car", 0.9, 0, 0, 50, 50),
            _det("car", 0.7, 2, 2, 52, 52),
            _det("person", 0.8, 2, 2, 52, 52),
        ]
        retained = class_aware_duplicate_suppression(detections, iou_threshold=0.5)
        self.assertEqual([item.normalized_class_name for item in retained], ["car", "person"])
        self.assertEqual([item.confidence for item in retained if item.normalized_class_name == "car"], [0.9])

    def test_combines_packets_for_same_frame(self) -> None:
        frame = FramePacket("s", 1, 0.1, 10.0, 100, 100)
        packet_a = DetectionPacket("s", 1, 0.1, 100, 100, [_det("car", 0.9, 0, 0, 50, 50)])
        packet_b = DetectionPacket("s", 1, 0.1, 100, 100, [_det("person", 0.8, 60, 0, 90, 50)])
        combined = combine_detection_packets(frame, [packet_a, packet_b])
        self.assertEqual(len(combined.detections), 2)

    def test_combined_stage_runs_vehicle_then_person_on_same_frame(self) -> None:
        frame = FramePacket("s", 1, 0.1, 10.0, 100, 100)
        vehicle = _StaticStage(_det("car", 0.9, 0, 0, 50, 50, detector_source="vehicle_model"))
        person = _StaticStage(_det("person", 0.8, 60, 0, 90, 50, detector_source="person_model"))
        stage = CombinedSequentialDetectionStage(vehicle, person)

        combined = stage.process(frame)

        self.assertEqual([item.normalized_class_name for item in combined.detections], ["car", "person"])
        self.assertEqual(stage.to_dict()["vehicle_detector_calls"], 1)
        self.assertEqual(stage.to_dict()["person_detector_calls"], 1)
        self.assertEqual(stage.to_dict()["class_counts"], {"car": 1, "person": 1})


class _StaticStage:
    def __init__(self, detection: DetectionRecord) -> None:
        self.detection = detection

    def process(self, packet: FramePacket) -> DetectionPacket:
        return DetectionPacket(packet.source_id, packet.frame_index, packet.timestamp_sec, packet.frame_width, packet.frame_height, [self.detection])

    def to_dict(self) -> dict[str, int]:
        return {"fake": 1}


def _det(class_name: str, confidence: float, x1: float, y1: float, x2: float, y2: float, detector_source: str = "test") -> DetectionRecord:
    return DetectionRecord(
        bbox=BoundingBox(x1, y1, x2, y2),
        confidence=confidence,
        class_id=1,
        class_name=class_name,
        raw_class_id=1,
        raw_class_name=class_name,
        normalized_class_name=class_name,
        detector_source=detector_source,
    )


if __name__ == "__main__":
    unittest.main()
