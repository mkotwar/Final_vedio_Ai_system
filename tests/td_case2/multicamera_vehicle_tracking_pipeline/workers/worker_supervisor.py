from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from ..detection.vehicle_detector import SharedVehicleDetector
from ..ingestion.camera_config import CameraConfig
from ..persistence.tracking_persistence_service import TrackingPersistenceService
from ..tracking.camera_detection_router import CameraDetectionRouter
from ..tracking.tracking_config import TrackingConfig
from .camera_reader_worker import CameraReaderWorker
from .detection_worker import DetectionWorker
from .persistence_worker import PersistenceWorker
from .tracking_worker import TrackingWorker
from .worker_config import WorkerConfig
from .worker_messages import CompletedTrackMessage, EndOfInputMessage, WorkerErrorMessage
from .worker_metrics import ThreadLifecycleMetrics, TrackedQueue


@dataclass(slots=True)
class WorkerSupervisorResult:
    finalized_tracks: list[CompletedTrackMessage]
    persistence_results_by_track_uuid: dict[str, object]
    errors: list[WorkerErrorMessage]
    camera_reader_metrics: dict[str, dict[str, object]]
    detection_worker_metrics: dict[str, object]
    tracking_worker_metrics: dict[str, object]
    persistence_worker_metrics: dict[str, object] | None
    queue_metrics: dict[str, dict[str, int]]
    thread_metrics: dict[str, dict[str, bool]]
    shutdown_clean: bool


class WorkerSupervisor:
    def __init__(
        self,
        *,
        camera_configs: list[CameraConfig],
        detector: SharedVehicleDetector,
        tracking_config: TrackingConfig,
        worker_config: WorkerConfig,
        max_frames_per_camera: int | None = None,
        persistence_service: TrackingPersistenceService | None = None,
        run_id: str | None = None,
        save_sample_frames: bool = False,
        sample_frame_limit_per_camera: int = 1,
        sample_output_dir: str | Path | None = None,
    ) -> None:
        self.camera_configs = camera_configs
        self.detector = detector
        self.tracking_config = tracking_config
        self.worker_config = worker_config
        self.max_frames_per_camera = max_frames_per_camera
        self.persistence_service = persistence_service
        self.run_id = run_id
        self.save_sample_frames = save_sample_frames
        self.sample_frame_limit_per_camera = sample_frame_limit_per_camera
        self.sample_output_dir = sample_output_dir
        self.shutdown_event = threading.Event()
        self.frame_queue = TrackedQueue(worker_config.frame_queue_size)
        self.detection_queue = TrackedQueue(worker_config.detection_queue_size)
        self.completed_track_queue = TrackedQueue(worker_config.completed_track_queue_size)
        self.error_queue = TrackedQueue(worker_config.error_queue_size)
        self.router = CameraDetectionRouter(tracking_config, run_id=run_id)
        self.camera_workers = [
            CameraReaderWorker(
                camera_config=config,
                frame_queue=self.frame_queue,
                error_queue=self.error_queue,
                shutdown_event=self.shutdown_event,
                worker_config=worker_config,
                max_frames_per_camera=max_frames_per_camera,
            )
            for config in camera_configs
        ]
        self.detection_worker = DetectionWorker(
            detector=detector,
            camera_count=len(camera_configs),
            frame_queue=self.frame_queue,
            detection_queue=self.detection_queue,
            error_queue=self.error_queue,
            shutdown_event=self.shutdown_event,
            worker_config=worker_config,
        )
        self.tracking_worker = TrackingWorker(
            router=self.router,
            detection_queue=self.detection_queue,
            completed_track_queue=self.completed_track_queue,
            error_queue=self.error_queue,
            shutdown_event=self.shutdown_event,
            worker_config=worker_config,
            save_sample_frames=save_sample_frames,
            sample_frame_limit_per_camera=sample_frame_limit_per_camera,
            sample_output_dir=sample_output_dir,
        )
        self.persistence_worker = (
            PersistenceWorker(
                persistence_service=persistence_service,
                completed_track_queue=self.completed_track_queue,
                error_queue=self.error_queue,
                shutdown_event=self.shutdown_event,
                worker_config=worker_config,
            )
            if persistence_service is not None and worker_config.persist_completed_tracks
            else None
        )
        self.collected_tracks: list[CompletedTrackMessage] = []
        self.errors: list[WorkerErrorMessage] = []
        self.thread_metrics: dict[str, ThreadLifecycleMetrics] = {}

    def run(self) -> WorkerSupervisorResult:
        workers: list[threading.Thread] = []
        if self.persistence_worker is not None:
            workers.append(self.persistence_worker)
        workers.extend([self.tracking_worker, self.detection_worker, *self.camera_workers])
        for worker in workers:
            self.thread_metrics[worker.name] = ThreadLifecycleMetrics(started=True)
            worker.start()
        shutdown_clean = self._monitor()
        for worker in self.camera_workers:
            self._join_worker(worker)
        self._join_worker(self.detection_worker)
        self._join_worker(self.tracking_worker)
        if self.persistence_worker is not None:
            self._join_worker(self.persistence_worker)
        shutdown_clean = shutdown_clean and all(not worker.is_alive() for worker in workers)
        finalized_tracks = self.persistence_worker.finalized_tracks if self.persistence_worker is not None else list(self.collected_tracks)
        persistence_results = self.persistence_worker.results_by_track_uuid if self.persistence_worker is not None else {}
        return WorkerSupervisorResult(
            finalized_tracks=finalized_tracks,
            persistence_results_by_track_uuid=persistence_results,
            errors=list(self.errors),
            camera_reader_metrics={worker.camera_config.camera_code: worker.metrics.to_dict() for worker in self.camera_workers},
            detection_worker_metrics=self.detection_worker.metrics.to_dict(),
            tracking_worker_metrics=self.tracking_worker.metrics.to_dict(),
            persistence_worker_metrics=self.persistence_worker.metrics.to_dict() if self.persistence_worker is not None else None,
            queue_metrics={
                "frame_queue": self.frame_queue.metrics.to_dict(),
                "detection_queue": self.detection_queue.metrics.to_dict(),
                "completed_track_queue": self.completed_track_queue.metrics.to_dict(),
                "error_queue": self.error_queue.metrics.to_dict(),
            },
            thread_metrics={name: metrics.to_dict() for name, metrics in sorted(self.thread_metrics.items())},
            shutdown_clean=shutdown_clean,
        )

    def _monitor(self) -> bool:
        workers = list(self.camera_workers) + [self.detection_worker, self.tracking_worker] + ([self.persistence_worker] if self.persistence_worker is not None else [])
        while any(worker is not None and worker.is_alive() for worker in workers):
            self._drain_error_queue()
            if self.persistence_worker is None:
                self._drain_completed_queue()
            time.sleep(0.05)
        self._drain_error_queue()
        if self.persistence_worker is None:
            self._drain_completed_queue()
        return True

    def _join_worker(self, worker: threading.Thread) -> None:
        worker.join(timeout=self.worker_config.shutdown_timeout_seconds)
        metrics = self.thread_metrics.setdefault(worker.name, ThreadLifecycleMetrics())
        metrics.stopped = not worker.is_alive()
        metrics.joined_successfully = not worker.is_alive()

    def _drain_error_queue(self) -> None:
        while True:
            try:
                item = self.error_queue.get_nowait()
            except queue.Empty:
                return
            if isinstance(item, WorkerErrorMessage):
                self.errors.append(item)
                if item.fatal:
                    self.shutdown_event.set()

    def _drain_completed_queue(self) -> None:
        while True:
            try:
                item = self.completed_track_queue.get_nowait()
            except queue.Empty:
                return
            if isinstance(item, CompletedTrackMessage):
                self.collected_tracks.append(item)
            elif isinstance(item, EndOfInputMessage):
                return
