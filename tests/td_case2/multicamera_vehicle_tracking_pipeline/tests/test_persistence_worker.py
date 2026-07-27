from __future__ import annotations

import threading
import unittest
from datetime import datetime
from pathlib import Path
import queue
import time

from tests.td_case2.multicamera_vehicle_tracking_pipeline.database.repository import SimpleVehicleRepository
from tests.td_case2.multicamera_vehicle_tracking_pipeline.ingestion.camera_config import CameraConfig
from tests.td_case2.multicamera_vehicle_tracking_pipeline.persistence.persistence_config import PersistenceConfig
from tests.td_case2.multicamera_vehicle_tracking_pipeline.persistence.persistence_models import TrackPersistenceResult
from tests.td_case2.multicamera_vehicle_tracking_pipeline.persistence.tracking_persistence_service import TrackingPersistenceService
from tests.td_case2.multicamera_vehicle_tracking_pipeline.persistence.persistence_service_protocol import PersistenceServiceProtocol
from tests.td_case2.multicamera_vehicle_tracking_pipeline.tracking.tracking_models import LocalVehicleTrack, TrackObservation
from tests.td_case2.multicamera_vehicle_tracking_pipeline.workers.persistence_worker import PersistenceWorker
from tests.td_case2.multicamera_vehicle_tracking_pipeline.workers.worker_config import WorkerConfig
from tests.td_case2.multicamera_vehicle_tracking_pipeline.workers.worker_messages import CompletedTrackMessage, EndOfInputMessage
from tests.td_case2.multicamera_vehicle_tracking_pipeline.workers.worker_metrics import TrackedQueue


def _track() -> LocalVehicleTrack:
    observation = TrackObservation("CAM_001", 1, 0, 0.0, datetime(2026, 7, 22, 10, 0, 0), "car", 0.9, (1.0, 2.0, 5.0, 6.0), "CAM_001:TRACK_1", "active")
    return LocalVehicleTrack(
        track_uuid="CAM_001:TRACK_1",
        camera_code="CAM_001",
        local_track_id=1,
        class_name="car",
        first_frame_number=0,
        last_frame_number=0,
        first_seen_at=observation.camera_timestamp,
        last_seen_at=observation.camera_timestamp,
        first_video_time_seconds=0.0,
        last_video_time_seconds=0.0,
        observation_count=1,
        best_confidence=0.9,
        state="completed",
        observations=[observation],
        camera_name="Cam",
        source_path=Path("cam.mp4"),
    )


class PersistenceWorkerTests(unittest.TestCase):
    def test_persists_completed_track(self) -> None:
        repo = SimpleVehicleRepository()
        service = TrackingPersistenceService(repo, PersistenceConfig())
        service.sync_cameras([CameraConfig("CAM_001", "Cam", Path("cam.mp4"), True, datetime(2026, 7, 22, 10, 0, 0))])
        completed_queue = TrackedQueue(10)
        error_queue = TrackedQueue(10)
        worker = PersistenceWorker(
            persistence_service=service,
            completed_track_queue=completed_queue,
            error_queue=error_queue,
            shutdown_event=threading.Event(),
            worker_config=WorkerConfig(queue_put_timeout_seconds=0.1, queue_get_timeout_seconds=0.1),
        )
        completed_queue.put(CompletedTrackMessage("CAM_001", _track()), timeout=0.1)
        completed_queue.put(EndOfInputMessage(), timeout=0.1)
        worker.start()
        worker.join(5)
        self.assertEqual(worker.metrics.tracks_inserted, 1)

    def test_temporary_full_follow_up_queue_does_not_fail_worker(self) -> None:
        class _FakePersistenceService(PersistenceServiceProtocol):
            def save_completed_track(self, track: LocalVehicleTrack) -> TrackPersistenceResult:
                return TrackPersistenceResult(
                    track_uuid=track.track_uuid,
                    status="inserted",
                    database_track_id="db-track-1",
                    observations_written=1,
                )

            def save_completed_tracks(self, tracks: list[LocalVehicleTrack]) -> list[TrackPersistenceResult]:
                return [self.save_completed_track(track) for track in tracks]

            def get_metrics(self):
                raise NotImplementedError

        completed_queue = TrackedQueue(10)
        error_queue = TrackedQueue(10)
        vehicle_colour_queue = TrackedQueue(1)
        shutdown_event = threading.Event()
        worker = PersistenceWorker(
            persistence_service=_FakePersistenceService(),
            completed_track_queue=completed_queue,
            error_queue=error_queue,
            shutdown_event=shutdown_event,
            worker_config=WorkerConfig(queue_put_timeout_seconds=0.05, queue_get_timeout_seconds=0.05),
            vehicle_colour_queue=vehicle_colour_queue,
        )
        vehicle_colour_queue.put(object(), timeout=0.05)

        released: list[object] = []

        def _drain_later() -> None:
            time.sleep(0.15)
            released.append(vehicle_colour_queue.get(timeout=0.1)[0])

        drain_thread = threading.Thread(target=_drain_later)
        drain_thread.start()
        completed_queue.put(CompletedTrackMessage("CAM_001", _track()), timeout=0.05)
        completed_queue.put(EndOfInputMessage(), timeout=0.05)
        worker.start()
        worker.join(5)
        drain_thread.join(5)

        self.assertFalse(worker.is_alive())
        self.assertEqual(worker.metrics.errors, 0)
        self.assertTrue(error_queue.empty())
        self.assertEqual(worker.metrics.tracks_inserted, 1)
        self.assertEqual(len(released), 1)


if __name__ == "__main__":
    unittest.main()
