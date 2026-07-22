from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tests.td_case2.multicamera_vehicle_tracking_pipeline.detection.detection_config import DetectionConfig
from tests.td_case2.multicamera_vehicle_tracking_pipeline.detection.detection_models import DetectionPacket, VehicleDetection
from tests.td_case2.multicamera_vehicle_tracking_pipeline.ingestion.frame_packet import FramePacket
from tests.td_case2.multicamera_vehicle_tracking_pipeline.orchestration.multicamera_detection_orchestrator import MultiCameraDetectionOrchestrator


def _write_test_video(path: Path, *, frame_count: int) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (48, 32))
    if not writer.isOpened():
        raise RuntimeError("Failed to create temporary test video.")
    for index in range(frame_count):
        frame = np.full((32, 48, 3), index * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class _FakeDetector:
    def __init__(self) -> None:
        self.loaded_model_name = "fake_detector.pt"
        self.device = "cpu"
        self.calls: list[tuple[str, int]] = []

    def detect(self, frame_packet: FramePacket) -> DetectionPacket:
        self.calls.append((frame_packet.camera_code, frame_packet.frame_number))
        detections = []
        if frame_packet.frame_number % 2 == 0:
            detections.append(VehicleDetection(class_id=0, class_name="car", confidence=0.9, bbox_xyxy=(1.0, 2.0, 10.0, 12.0)))
        return DetectionPacket(
            camera_code=frame_packet.camera_code,
            camera_name=frame_packet.camera_name,
            source_path=frame_packet.source_path,
            frame_number=frame_packet.frame_number,
            video_time_seconds=frame_packet.video_time_seconds,
            camera_timestamp=frame_packet.camera_timestamp,
            frame_width=48,
            frame_height=32,
            detections=detections,
            inference_time_ms=5.0,
            detector_model=self.loaded_model_name,
            detector_device=self.device,
        )


class DetectionOrchestratorTests(unittest.TestCase):
    def test_round_robin_frames_reach_detector_and_metrics_stay_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir()
            (root / "data").mkdir()
            _write_test_video(root / "data" / "camera_1.avi", frame_count=2)
            _write_test_video(root / "data" / "camera_2.avi", frame_count=3)
            (root / "config" / "cameras.yaml").write_text(
                'cameras:\n'
                '  - camera_code: CAM_001\n'
                '    camera_name: North Gate\n'
                '    source_path: data/camera_1.avi\n'
                '    enabled: true\n'
                '  - camera_code: CAM_002\n'
                '    camera_name: Parking Entry\n'
                '    source_path: data/camera_2.avi\n'
                '    enabled: true\n',
                encoding="utf-8",
            )
            (root / "config" / "detection.yaml").write_text(
                'vehicle_detector:\n'
                '  model_path: yolov8n.pt\n'
                '  fallback_model_path: yolov8n.pt\n'
                '  allow_fallback: true\n'
                '  device: cpu\n'
                '  confidence_threshold: 0.25\n'
                '  iou_threshold: 0.45\n'
                '  image_size: 640\n'
                '  allowed_classes:\n'
                '    - car\n'
                '    - bus\n'
                '    - truck\n'
                '    - motorcycle\n',
                encoding="utf-8",
            )
            detector = _FakeDetector()
            orchestrator = MultiCameraDetectionOrchestrator(
                root / "config" / "cameras.yaml",
                root / "config" / "detection.yaml",
                mode="round_robin",
                detector=detector,
            )
            result = orchestrator.run(save_sample_frames=True, sample_frame_limit_per_camera=1, output_report=root / "report.json")
            self.assertEqual(detector.calls, [("CAM_001", 0), ("CAM_002", 0), ("CAM_001", 1), ("CAM_002", 1), ("CAM_002", 2)])
            self.assertEqual(result.report["total_frames_processed"], 5)
            self.assertEqual(result.report["cameras"]["CAM_001"]["frames_processed"], 2)
            self.assertEqual(result.report["cameras"]["CAM_002"]["frames_processed"], 3)
            self.assertTrue((root / "CAM_001" / "sample_000001.jpg").exists())
            self.assertTrue((root / "CAM_002" / "sample_000001.jpg").exists())
            loaded = json.loads((root / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded["detector"]["actual_model"], "fake_detector.pt")
