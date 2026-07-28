from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

from tests.td_case2.multicamera_vehicle_tracking_pipeline.detection.detection_config import (
    ClassConfidenceThresholdsConfig,
    DetectionConfig,
)
from tests.td_case2.multicamera_vehicle_tracking_pipeline.detection.vehicle_detector import SharedVehicleDetector, VehicleDetectorError
from tests.td_case2.multicamera_vehicle_tracking_pipeline.ingestion.frame_packet import FramePacket


class _FakeTensor:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


class _FakeBoxes:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = _FakeTensor(xyxy)
        self.cls = _FakeTensor(cls)
        self.conf = _FakeTensor(conf)


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    def __init__(self, names, results):
        self.names = names
        self._results = results
        self.predict_calls = 0

    def predict(self, **kwargs):
        self.predict_calls += 1
        return self._results


class SharedVehicleDetectorTests(unittest.TestCase):
    def _config(
        self,
        *,
        allowed_classes: tuple[str, ...] = ("3wheeler", "bus", "car", "motorcycle", "truck"),
        class_thresholds: dict[str, float] | None = None,
    ) -> DetectionConfig:
        thresholds = class_thresholds or {class_name: 0.25 for class_name in allowed_classes}
        return DetectionConfig(
            model_path="yolov8n.pt",
            allowed_classes=allowed_classes,
            class_confidence_thresholds=ClassConfidenceThresholdsConfig(enabled=True, classes=thresholds),
        )

    def _frame_packet(self) -> FramePacket:
        return FramePacket(
            camera_code="CAM_001",
            camera_name="North Gate",
            source_path=Path("camera.mp4"),
            frame_number=0,
            source_fps=5.0,
            source_frame_count=5,
            video_time_seconds=0.0,
            camera_timestamp=datetime(2026, 7, 22, 10, 0, 0),
            frame=np.zeros((64, 64, 3), dtype=np.uint8),
        )

    def test_model_is_loaded_only_once(self) -> None:
        load_calls: list[str] = []

        def loader(path: str):
            load_calls.append(path)
            return _FakeModel({0: "car"}, [_FakeResult(_FakeBoxes([[1, 2, 10, 12]], [0], [0.9]))])

        detector = SharedVehicleDetector(self._config(), yolo_loader=loader)
        detector.detect(self._frame_packet())
        detector.detect(self._frame_packet())
        self.assertEqual(load_calls, ["yolov8n.pt"])

    def test_metadata_is_preserved(self) -> None:
        detector = SharedVehicleDetector(
            self._config(),
            model=_FakeModel({0: "car"}, [_FakeResult(_FakeBoxes([[1, 2, 10, 12]], [0], [0.9]))]),
        )
        packet = detector.detect(self._frame_packet())
        self.assertEqual(packet.camera_code, "CAM_001")
        self.assertEqual(packet.source_path, Path("camera.mp4"))

    def test_class_filtering_and_multiple_detections_work(self) -> None:
        detector = SharedVehicleDetector(
            self._config(),
            model=_FakeModel(
                {0: "car", 1: "person", 2: "motorbike", 3: "3Wheeler"},
                [
                    _FakeResult(
                        _FakeBoxes(
                            [[1, 2, 10, 12], [3, 4, 14, 18], [5, 6, 15, 20], [8, 9, 18, 22]],
                            [0, 1, 2, 3],
                            [0.9, 0.95, 0.8, 0.77],
                        )
                    )
                ],
            ),
        )
        packet = detector.detect(self._frame_packet())
        self.assertEqual([item.class_name for item in packet.detections], ["car", "motorcycle", "3wheeler"])

    def test_no_detection_frame_works(self) -> None:
        detector = SharedVehicleDetector(self._config(), model=_FakeModel({0: "car"}, []))
        packet = detector.detect(self._frame_packet())
        self.assertEqual(packet.detections, [])

    def test_inference_error_is_reported_clearly(self) -> None:
        class BrokenModel:
            names = {0: "car"}

            def predict(self, **kwargs):
                raise RuntimeError("boom")

        detector = SharedVehicleDetector(self._config(), model=BrokenModel())
        with self.assertRaises(VehicleDetectorError):
            detector.detect(self._frame_packet())

    def test_explicit_class_thresholds_accept_matching_detection(self) -> None:
        detector = SharedVehicleDetector(
            self._config(class_thresholds={"3wheeler": 0.38, "bus": 0.38, "car": 0.38, "motorcycle": 0.38, "truck": 0.38}),
            model=_FakeModel({0: "car"}, [_FakeResult(_FakeBoxes([[1, 2, 10, 12]], [0], [0.39]))]),
        )
        packet = detector.detect(self._frame_packet())
        self.assertEqual(detector.inference_confidence_floor, 0.38)
        self.assertEqual(len(packet.detections), 1)

    def test_inference_floor_uses_minimum_class_threshold(self) -> None:
        detector = SharedVehicleDetector(
            self._config(class_thresholds={"3wheeler": 0.38, "bus": 0.50, "car": 0.38, "motorcycle": 0.32, "truck": 0.50}),
            model=_FakeModel({0: "car"}, []),
        )
        self.assertEqual(detector.inference_confidence_floor, 0.32)

    def test_motorcycle_threshold_accepts_lower_confidence_than_bus(self) -> None:
        detector = SharedVehicleDetector(
            self._config(class_thresholds={"3wheeler": 0.38, "bus": 0.50, "car": 0.38, "motorcycle": 0.32, "truck": 0.38}),
            model=_FakeModel(
                {0: "motorbike", 1: "bus"},
                [_FakeResult(_FakeBoxes([[1, 2, 10, 12], [2, 3, 11, 13]], [0, 1], [0.34, 0.34]))],
            ),
        )
        packet = detector.detect(self._frame_packet())
        self.assertEqual([item.class_name for item in packet.detections], ["motorcycle"])
        self.assertEqual(detector.rejected_by_class_threshold_counts["bus"], 1)

    def test_missing_threshold_is_rejected_at_config_load(self) -> None:
        with self.assertRaisesRegex(ValueError, "must define every allowed class"):
            DetectionConfig(
                model_path="yolov8n.pt",
                allowed_classes=("car", "bus"),
                class_confidence_thresholds=ClassConfidenceThresholdsConfig(enabled=True, classes={"bus": 0.50}),
            )
