from __future__ import annotations

import threading
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

from tests.td_case2.multicamera_vehicle_tracking_pipeline.detection.detection_models import DetectionPacket, VehicleDetection
from tests.td_case2.multicamera_vehicle_tracking_pipeline.ingestion.frame_packet import FramePacket
from tests.td_case2.multicamera_vehicle_tracking_pipeline.workers.detection_worker import DetectionWorker
from tests.td_case2.multicamera_vehicle_tracking_pipeline.workers.worker_config import WorkerConfig
from tests.td_case2.multicamera_vehicle_tracking_pipeline.workers.worker_messages import EndOfCameraMessage, EndOfInputMessage
from tests.td_case2.multicamera_vehicle_tracking_pipeline.workers.worker_metrics import TrackedQueue


class _FakeDetector:
    def __init__(self) -> None:
        self.calls = 0
        self.loaded_model_name = "fake.pt"
        self.device = "cpu"

    def detect(self, frame_packet: FramePacket) -> DetectionPacket:
        self.calls += 1
        return DetectionPacket(
            camera_code=frame_packet.camera_code,
            camera_name=frame_packet.camera_name,
            source_path=frame_packet.source_path,
            frame_number=frame_packet.frame_number,
            video_time_seconds=frame_packet.video_time_seconds,
            camera_timestamp=frame_packet.camera_timestamp,
            frame_width=32,
            frame_height=24,
            detections=[VehicleDetection(0, "car", 0.9, (1.0, 2.0, 5.0, 6.0))],
            inference_time_ms=3.0,
            detector_model="fake.pt",
            detector_device="cpu",
            frame=frame_packet.frame,
        )


class DetectionWorkerTests(unittest.TestCase):
    def test_reuses_detector_and_forwards_end(self) -> None:
        frame_queue = TrackedQueue(10)
        detection_queue = TrackedQueue(10)
        error_queue = TrackedQueue(10)
        detector = _FakeDetector()
        worker = DetectionWorker(
            detector=detector,
            camera_count=1,
            frame_queue=frame_queue,
            detection_queue=detection_queue,
            error_queue=error_queue,
            shutdown_event=threading.Event(),
            worker_config=WorkerConfig(queue_put_timeout_seconds=0.1, queue_get_timeout_seconds=0.1),
        )
        frame_queue.put(
            FramePacket("CAM_001", "Cam", Path("cam.mp4"), 0, 5.0, 1, 0.0, datetime(2026, 7, 22, 10, 0, 0), np.zeros((24, 32, 3), dtype=np.uint8)),
            timeout=0.1,
        )
        frame_queue.put(EndOfCameraMessage("CAM_001"), timeout=0.1)
        worker.start()
        worker.join(5)
        first = detection_queue.get_nowait()
        second = detection_queue.get_nowait()
        third = detection_queue.get_nowait()
        self.assertEqual(detector.calls, 1)
        self.assertEqual(first.camera_code, "CAM_001")
        self.assertIsInstance(second, EndOfCameraMessage)
        self.assertIsInstance(third, EndOfInputMessage)


if __name__ == "__main__":
    unittest.main()
