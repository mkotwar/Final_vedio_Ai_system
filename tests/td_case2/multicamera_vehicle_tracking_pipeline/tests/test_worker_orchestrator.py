from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tests.td_case2.multicamera_vehicle_tracking_pipeline.detection.detection_models import DetectionPacket, VehicleDetection
from tests.td_case2.multicamera_vehicle_tracking_pipeline.ingestion.camera_config import CameraConfig, CameraConfigError
from tests.td_case2.multicamera_vehicle_tracking_pipeline.ingestion.frame_packet import FramePacket
from tests.td_case2.multicamera_vehicle_tracking_pipeline.orchestration.worker_multicamera_tracking_orchestrator import (
    WorkerMultiCameraTrackingOrchestrator,
    select_worker_cameras,
)


def _write_video(path: Path, frame_count: int) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (32, 24))
    for index in range(frame_count):
        writer.write(np.full((24, 32, 3), index * 5, dtype=np.uint8))
    writer.release()


class _FakeDetector:
    loaded_model_name = "fake.pt"
    device = "cpu"

    def detect(self, frame_packet: FramePacket) -> DetectionPacket:
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
            inference_time_ms=1.0,
            detector_model="fake.pt",
            detector_device="cpu",
            frame=frame_packet.frame,
        )


class WorkerOrchestratorTests(unittest.TestCase):
    def test_select_worker_cameras_filters_enabled_codes_and_limit(self) -> None:
        configs = [
            CameraConfig("CAM_001", "Cam1", Path("cam1.mp4"), True),
            CameraConfig("CAM_002", "Cam2", Path("cam2.mp4"), False),
            CameraConfig("CAM_003", "Cam3", Path("cam3.mp4"), True),
            CameraConfig("CAM_004", "Cam4", Path("cam4.mp4"), True),
        ]
        selected = select_worker_cameras(configs, camera_codes=["CAM_003", "CAM_001"], camera_limit=1)
        self.assertEqual([camera.camera_code for camera in selected], ["CAM_003"])

    def test_select_worker_cameras_rejects_unknown_or_zero_enabled(self) -> None:
        with self.assertRaises(CameraConfigError):
            select_worker_cameras([CameraConfig("CAM_001", "Cam1", Path("cam1.mp4"), False)])
        with self.assertRaises(CameraConfigError):
            select_worker_cameras([CameraConfig("CAM_001", "Cam1", Path("cam1.mp4"), True)], camera_codes=["CAM_999"])

    def test_worker_report_contains_dynamic_camera_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir()
            (root / "data").mkdir()
            _write_video(root / "data" / "camera_1.avi", 2)
            _write_video(root / "data" / "camera_2.avi", 2)
            _write_video(root / "data" / "camera_3.avi", 2)
            (root / "config" / "cameras.yaml").write_text(
                'cameras:\n'
                '  - camera_code: CAM_001\n'
                '    camera_name: North Gate\n'
                '    source_path: data/camera_1.avi\n'
                '    enabled: true\n'
                '    start_time: "2026-07-22T10:00:00+05:30"\n'
                '  - camera_code: CAM_002\n'
                '    camera_name: Parking Entry\n'
                '    source_path: data/camera_2.avi\n'
                '    enabled: true\n'
                '    start_time: "2026-07-22T10:00:00+05:30"\n'
                '  - camera_code: CAM_003\n'
                '    camera_name: South Gate\n'
                '    source_path: data/camera_3.avi\n'
                '    enabled: false\n'
                '    start_time: "2026-07-22T10:00:00+05:30"\n'
                '  - camera_code: CAM_004\n'
                '    camera_name: Warehouse\n'
                '    source_path: data/camera_1.avi\n'
                '    enabled: true\n'
                '    start_time: "2026-07-22T10:00:00+05:30"\n'
                '  - camera_code: CAM_005\n'
                '    camera_name: Loading Area\n'
                '    source_path: data/camera_2.avi\n'
                '    enabled: true\n'
                '    start_time: "2026-07-22T10:00:00+05:30"\n',
                encoding="utf-8",
            )
            (root / "config" / "detection.yaml").write_text(
                'vehicle_detector:\n  model_path: yolov8n.pt\n  fallback_model_path: yolov8n.pt\n  allow_fallback: true\n  device: cpu\n  confidence_threshold: 0.25\n  iou_threshold: 0.45\n  image_size: 640\n  allowed_classes:\n    - car\n    - bus\n    - truck\n    - motorcycle\n',
                encoding="utf-8",
            )
            (root / "config" / "tracking.yaml").write_text(
                'tracking:\n  backend: ultralytics_bytetrack\n  track_high_thresh: 0.30\n  track_low_thresh: 0.10\n  new_track_thresh: 0.30\n  match_thresh: 0.80\n  track_buffer: 30\n  min_confirmed_observations: 1\n  max_lost_frames: 0\n  preserve_state_per_camera: true\n',
                encoding="utf-8",
            )
            (root / "config" / "workers.yaml").write_text(
                'workers:\n  enabled: true\n  frame_queue_size: 10\n  detection_queue_size: 10\n  completed_track_queue_size: 10\n  error_queue_size: 10\n  queue_put_timeout_seconds: 0.1\n  queue_get_timeout_seconds: 0.1\n  shutdown_timeout_seconds: 5.0\n  stop_on_camera_error: false\n  stop_on_detector_error: true\n  stop_on_tracking_error: true\n  stop_on_persistence_error: false\n  enable_persistence_worker: false\n',
                encoding="utf-8",
            )
            orchestrator = WorkerMultiCameraTrackingOrchestrator(
                root / "config" / "cameras.yaml",
                root / "config" / "detection.yaml",
                root / "config" / "tracking.yaml",
                root / "config" / "workers.yaml",
                max_frames_per_camera=2,
                detector=_FakeDetector(),
                run_id="RUN_TEST",
                camera_limit=3,
            )
            result = orchestrator.run(output_report=root / "report.json")
            self.assertEqual(result.report["execution_mode"], "workers")
            self.assertEqual(result.report["configured_camera_count"], 5)
            self.assertEqual(result.report["enabled_camera_count"], 4)
            self.assertEqual(result.report["disabled_camera_count"], 1)
            self.assertEqual(result.report["selected_camera_count"], 3)
            self.assertEqual(result.report["workers"]["camera_reader_count"], 3)
            self.assertEqual(result.report["total_frames_processed"], 6)
            self.assertIn("thread_shutdown", result.report["workers"])
            loaded = json.loads((root / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded["detector"]["actual_model"], "fake.pt")


if __name__ == "__main__":
    unittest.main()
