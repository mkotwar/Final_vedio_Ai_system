from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path

from tests.td_case2.multicamera_vehicle_tracking_pipeline.detection.detection_models import DetectionPacket, VehicleDetection
from tests.td_case2.multicamera_vehicle_tracking_pipeline.detection.vehicle_detector import _validate_and_clamp_bbox, normalize_vehicle_class


class DetectionModelTests(unittest.TestCase):
    def test_valid_detection_packet_creation(self) -> None:
        packet = DetectionPacket(
            camera_code="CAM_001",
            camera_name="North Gate",
            source_path=Path("camera.mp4"),
            frame_number=1,
            video_time_seconds=0.2,
            camera_timestamp=datetime(2026, 7, 22, 10, 0, 0),
            frame_width=640,
            frame_height=480,
            detections=[VehicleDetection(class_id=0, class_name="car", confidence=0.8, bbox_xyxy=(1.0, 2.0, 10.0, 12.0))],
            inference_time_ms=12.5,
            detector_model="model.pt",
            detector_device="cpu",
        )
        self.assertEqual(packet.detections[0].class_name, "car")

    def test_supported_class_normalization(self) -> None:
        self.assertEqual(normalize_vehicle_class("motorbike"), "motorcycle")
        self.assertEqual(normalize_vehicle_class("automobile"), "car")
        self.assertEqual(normalize_vehicle_class("lorry"), "truck")

    def test_unsupported_class_rejection(self) -> None:
        self.assertIsNone(normalize_vehicle_class("person"))
        self.assertIsNone(normalize_vehicle_class("3wheeler"))

    def test_invalid_bounding_box_rejection(self) -> None:
        self.assertIsNone(_validate_and_clamp_bbox((10, 10, 5, 5), frame_width=100, frame_height=100))
        self.assertIsNone(_validate_and_clamp_bbox((float("nan"), 0, 10, 10), frame_width=100, frame_height=100))

    def test_bounding_box_clamping(self) -> None:
        bbox = _validate_and_clamp_bbox((-5, -4, 110, 120), frame_width=100, frame_height=90)
        self.assertEqual(bbox, (0.0, 0.0, 100.0, 90.0))

    def test_empty_detection_output(self) -> None:
        packet = DetectionPacket(
            camera_code="CAM_001",
            camera_name="North Gate",
            source_path=Path("camera.mp4"),
            frame_number=1,
            video_time_seconds=0.0,
            camera_timestamp=None,
            frame_width=640,
            frame_height=480,
            detections=[],
            inference_time_ms=0.0,
            detector_model="model.pt",
            detector_device="cpu",
        )
        self.assertEqual(packet.detections, [])
