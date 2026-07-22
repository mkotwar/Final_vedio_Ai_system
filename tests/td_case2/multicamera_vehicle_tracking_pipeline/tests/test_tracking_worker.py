from __future__ import annotations

import threading
import unittest
from datetime import datetime
from pathlib import Path

from tests.td_case2.multicamera_vehicle_tracking_pipeline.detection.detection_models import DetectionPacket, VehicleDetection
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.tracker_factory import TrackerFactory
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.tracking_config import TrackingConfig
from tests.td_case2.multicamera_vehicle_tracking_pipeline.workers.tracking_worker import TrackingWorker
from tests.td_case2.multicamera_vehicle_tracking_pipeline.workers.worker_config import WorkerConfig
from tests.td_case2.multicamera_vehicle_tracking_pipeline.workers.worker_messages import CompletedTrackMessage, EndOfCameraMessage, EndOfInputMessage
from tests.td_case2.multicamera_vehicle_tracking_pipeline.workers.worker_metrics import TrackedQueue


class _SharedIdTracker:
    def update(self, results, img=None):
        if len(results) == 0:
            return []
        return [[1, 2, 10, 12, 1, 0.9, 0, 0]]


def _packet(camera_code: str, frame_number: int) -> DetectionPacket:
    return DetectionPacket(
        camera_code=camera_code,
        camera_name=f"Camera {camera_code}",
        source_path=Path(f"{camera_code}.mp4"),
        frame_number=frame_number,
        video_time_seconds=float(frame_number),
        camera_timestamp=datetime(2026, 7, 22, 10, 0, min(frame_number, 59)),
        frame_width=32,
        frame_height=24,
        detections=[VehicleDetection(0, "car", 0.9, (1.0, 2.0, 5.0, 6.0))],
        inference_time_ms=1.0,
        detector_model="fake.pt",
        detector_device="cpu",
        frame=None,
    )


class TrackingWorkerTests(unittest.TestCase):
    def test_routes_per_camera_and_flushes(self) -> None:
        detection_queue = TrackedQueue(20)
        completed_queue = TrackedQueue(20)
        error_queue = TrackedQueue(20)
        router_config = TrackingConfig(min_confirmed_observations=1)
        from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.camera_detection_router import CameraDetectionRouter

        router = CameraDetectionRouter(router_config, tracker_factory=TrackerFactory(router_config, tracker_creator=lambda config: _SharedIdTracker()), run_id="RUN_TEST")
        worker = TrackingWorker(
            router=router,
            detection_queue=detection_queue,
            completed_track_queue=completed_queue,
            error_queue=error_queue,
            shutdown_event=threading.Event(),
            worker_config=WorkerConfig(queue_put_timeout_seconds=0.1, queue_get_timeout_seconds=0.1),
        )
        detection_queue.put(_packet("CAM_001", 0), timeout=0.1)
        detection_queue.put(_packet("CAM_002", 0), timeout=0.1)
        detection_queue.put(EndOfCameraMessage("CAM_001"), timeout=0.1)
        detection_queue.put(EndOfInputMessage(), timeout=0.1)
        worker.start()
        worker.join(5)
        items = []
        while not completed_queue.empty():
            items.append(completed_queue.get_nowait())
        completed = [item for item in items if isinstance(item, CompletedTrackMessage)]
        self.assertTrue(any(item.track.track_uuid == "RUN_TEST:CAM_001:TRACK_1" for item in completed))
        self.assertTrue(any(item.track.track_uuid == "RUN_TEST:CAM_002:TRACK_1" for item in completed))


if __name__ == "__main__":
    unittest.main()
