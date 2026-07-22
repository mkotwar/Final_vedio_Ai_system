from __future__ import annotations

import tempfile
import threading
import unittest
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from tests.td_case2.multicamera_vehicle_tracking_pipeline.ingestion.camera_config import CameraConfig
from tests.td_case2.multicamera_vehicle_tracking_pipeline.workers.camera_reader_worker import CameraReaderWorker
from tests.td_case2.multicamera_vehicle_tracking_pipeline.workers.worker_config import WorkerConfig
from tests.td_case2.multicamera_vehicle_tracking_pipeline.workers.worker_messages import EndOfCameraMessage
from tests.td_case2.multicamera_vehicle_tracking_pipeline.workers.worker_metrics import TrackedQueue


def _write_video(path: Path, frame_count: int) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (32, 24))
    for index in range(frame_count):
        writer.write(np.full((24, 32, 3), index * 5, dtype=np.uint8))
    writer.release()


class CameraReaderWorkerTests(unittest.TestCase):
    def test_reads_frames_in_order_and_emits_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cam.avi"
            _write_video(path, 3)
            config = CameraConfig("CAM_001", "Cam", path, True, datetime(2026, 7, 22, 10, 0, 0))
            frame_queue = TrackedQueue(10)
            error_queue = TrackedQueue(10)
            worker = CameraReaderWorker(
                camera_config=config,
                frame_queue=frame_queue,
                error_queue=error_queue,
                shutdown_event=threading.Event(),
                worker_config=WorkerConfig(queue_put_timeout_seconds=0.1, queue_get_timeout_seconds=0.1),
            )
            worker.start()
            worker.join(5)
            items = [frame_queue.get_nowait() for _ in range(4)]
            self.assertEqual([items[0].frame_number, items[1].frame_number, items[2].frame_number], [0, 1, 2])
            self.assertIsInstance(items[3], EndOfCameraMessage)


if __name__ == "__main__":
    unittest.main()
