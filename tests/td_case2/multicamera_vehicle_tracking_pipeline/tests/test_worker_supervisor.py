from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from tests.td_case2.multicamera_vehicle_tracking_pipeline.detection.detection_models import DetectionPacket, VehicleDetection
from tests.td_case2.multicamera_vehicle_tracking_pipeline.ingestion.camera_config import CameraConfig
from tests.td_case2.multicamera_vehicle_tracking_pipeline.ingestion.frame_packet import FramePacket
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.tracking_config import TrackingConfig
from tests.td_case2.multicamera_vehicle_tracking_pipeline.workers.worker_config import WorkerConfig
from tests.td_case2.multicamera_vehicle_tracking_pipeline.workers.worker_supervisor import WorkerSupervisor


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


class WorkerSupervisorTests(unittest.TestCase):
    def test_clean_shutdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = Path(tmpdir) / "cam1.avi"
            path2 = Path(tmpdir) / "cam2.avi"
            _write_video(path1, 2)
            _write_video(path2, 2)
            configs = [
                CameraConfig("CAM_001", "Cam1", path1, True, datetime(2026, 7, 22, 10, 0, 0)),
                CameraConfig("CAM_002", "Cam2", path2, True, datetime(2026, 7, 22, 10, 0, 0)),
            ]
            supervisor = WorkerSupervisor(
                camera_configs=configs,
                detector=_FakeDetector(),
                tracking_config=TrackingConfig(min_confirmed_observations=1),
                worker_config=WorkerConfig(enabled=True, queue_put_timeout_seconds=0.1, queue_get_timeout_seconds=0.1),
                max_frames_per_camera=2,
                run_id="RUN_TEST",
            )
            result = supervisor.run()
            self.assertTrue(result.shutdown_clean)
            self.assertEqual(sum(item["frames_read"] for item in result.camera_reader_metrics.values()), 4)
            self.assertIn("shared_detection_worker", result.thread_metrics)
            self.assertTrue(result.thread_metrics["shared_detection_worker"]["joined_successfully"])

    def test_dynamic_reader_count_ignores_disabled_cameras(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = Path(tmpdir) / "cam1.avi"
            path2 = Path(tmpdir) / "cam2.avi"
            path3 = Path(tmpdir) / "cam3.avi"
            _write_video(path1, 1)
            _write_video(path2, 1)
            _write_video(path3, 1)
            configs = [
                CameraConfig("CAM_001", "Cam1", path1, True, datetime(2026, 7, 22, 10, 0, 0)),
                CameraConfig("CAM_002", "Cam2", path2, False, datetime(2026, 7, 22, 10, 0, 0)),
                CameraConfig("CAM_003", "Cam3", path3, True, datetime(2026, 7, 22, 10, 0, 0)),
            ]
            supervisor = WorkerSupervisor(
                camera_configs=[config for config in configs if config.enabled],
                detector=_FakeDetector(),
                tracking_config=TrackingConfig(min_confirmed_observations=1),
                worker_config=WorkerConfig(enabled=True, queue_put_timeout_seconds=0.1, queue_get_timeout_seconds=0.1),
                max_frames_per_camera=1,
                run_id="RUN_TEST",
            )
            result = supervisor.run()
            self.assertEqual(result.runtime_worker_counts["camera_reader_count"], 2)
            self.assertEqual(result.runtime_worker_counts["joined_reader_count"], 2)


if __name__ == "__main__":
    unittest.main()
