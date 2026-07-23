from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from ..detection.vehicle_detector import SharedVehicleDetector
from ..enrichment.anpr_enrichment_service import AnprEnrichmentService
from ..enrichment.vehicle_colour_enrichment_service import VehicleColourEnrichmentService
from ..evidence.track_evidence_collector import TrackEvidenceCollector
from ..ingestion.camera_config import CameraConfig
from ..persistence.persistence_service_protocol import PersistenceServiceProtocol
from ..tracking.camera_detection_router import CameraDetectionRouter
from ..tracking.tracking_config import TrackingConfig
from .camera_reader_worker import CameraReaderWorker
from .detection_worker import DetectionWorker
from .anpr_worker import AnprWorker
from .persistence_worker import PersistenceWorker
from .tracking_worker import TrackingWorker
from .vehicle_colour_worker import VehicleColourWorker
from .worker_config import WorkerConfig
from .worker_messages import CompletedTrackMessage, EndOfInputMessage, WorkerErrorMessage
from .worker_metrics import ThreadLifecycleMetrics, TrackedQueue


@dataclass(slots=True)
class WorkerSupervisorResult:
    finalized_tracks: list[CompletedTrackMessage]
    persistence_results_by_track_uuid: dict[str, object]
    vehicle_colour_results_by_track_uuid: dict[str, object]
    anpr_results_by_track_uuid: dict[str, object]
    errors: list[WorkerErrorMessage]
    camera_reader_metrics: dict[str, dict[str, object]]
    detection_worker_metrics: dict[str, object]
    tracking_worker_metrics: dict[str, object]
    persistence_worker_metrics: dict[str, object] | None
    vehicle_colour_worker_metrics: dict[str, object] | None
    anpr_worker_metrics: dict[str, object] | None
    queue_metrics: dict[str, dict[str, int]]
    thread_metrics: dict[str, dict[str, bool]]
    runtime_worker_counts: dict[str, object]
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
        persistence_service: PersistenceServiceProtocol | None = None,
        vehicle_colour_service: VehicleColourEnrichmentService | None = None,
        anpr_service: AnprEnrichmentService | None = None,
        evidence_collector: TrackEvidenceCollector | None = None,
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
        self.evidence_collector = evidence_collector
        self.run_id = run_id
        self.save_sample_frames = save_sample_frames
        self.sample_frame_limit_per_camera = sample_frame_limit_per_camera
        self.sample_output_dir = sample_output_dir
        self.expected_camera_codes = tuple(config.camera_code for config in camera_configs)
        self.expected_camera_code_set = set(self.expected_camera_codes)
        self.shutdown_event = threading.Event()
        self.frame_queue = TrackedQueue(worker_config.frame_queue_size)
        self.detection_queue = TrackedQueue(worker_config.detection_queue_size)
        self.completed_track_queue = TrackedQueue(worker_config.completed_track_queue_size)
        self.error_queue = TrackedQueue(worker_config.error_queue_size)
        self.vehicle_colour_queue = TrackedQueue(worker_config.vehicle_colour_queue_size)
        self.anpr_queue = TrackedQueue(worker_config.anpr_queue_size)
        self.router = CameraDetectionRouter(tracking_config, run_id=run_id, allowed_camera_codes=self.expected_camera_codes)
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
            expected_camera_codes=self.expected_camera_codes,
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
            expected_camera_codes=self.expected_camera_codes,
            evidence_collector=evidence_collector,
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
                vehicle_colour_queue=self.vehicle_colour_queue if vehicle_colour_service is not None and worker_config.enable_vehicle_colour_worker else None,
                anpr_queue=self.anpr_queue if anpr_service is not None and worker_config.enable_anpr_worker else None,
            )
            if persistence_service is not None and worker_config.enable_persistence_worker
            else None
        )
        self.vehicle_colour_worker = (
            VehicleColourWorker(
                enrichment_service=vehicle_colour_service,
                vehicle_colour_queue=self.vehicle_colour_queue,
                error_queue=self.error_queue,
                shutdown_event=self.shutdown_event,
                worker_config=worker_config,
            )
            if vehicle_colour_service is not None and worker_config.enable_vehicle_colour_worker
            else None
        )
        self.anpr_worker = (
            AnprWorker(
                enrichment_service=anpr_service,
                anpr_queue=self.anpr_queue,
                error_queue=self.error_queue,
                shutdown_event=self.shutdown_event,
                worker_config=worker_config,
            )
            if anpr_service is not None and worker_config.enable_anpr_worker
            else None
        )
        self.collected_tracks: list[CompletedTrackMessage] = []
        self.errors: list[WorkerErrorMessage] = []
        self.thread_metrics: dict[str, ThreadLifecycleMetrics] = {}

    def run(self) -> WorkerSupervisorResult:
        workers: list[threading.Thread] = []
        if self.anpr_worker is not None:
            workers.append(self.anpr_worker)
        if self.vehicle_colour_worker is not None:
            workers.append(self.vehicle_colour_worker)
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
        if self.vehicle_colour_worker is not None:
            self._join_worker(self.vehicle_colour_worker)
        if self.anpr_worker is not None:
            self._join_worker(self.anpr_worker)
        shutdown_clean = shutdown_clean and all(not worker.is_alive() for worker in workers)
        finalized_tracks = self.persistence_worker.finalized_tracks if self.persistence_worker is not None else list(self.collected_tracks)
        persistence_results = self.persistence_worker.results_by_track_uuid if self.persistence_worker is not None else {}
        vehicle_colour_results = self.vehicle_colour_worker.results_by_track_uuid if self.vehicle_colour_worker is not None else {}
        anpr_results = self.anpr_worker.results_by_track_uuid if self.anpr_worker is not None else {}
        return WorkerSupervisorResult(
            finalized_tracks=finalized_tracks,
            persistence_results_by_track_uuid=persistence_results,
            vehicle_colour_results_by_track_uuid=vehicle_colour_results,
            anpr_results_by_track_uuid=anpr_results,
            errors=list(self.errors),
            camera_reader_metrics={worker.camera_config.camera_code: worker.metrics.to_dict() for worker in self.camera_workers},
            detection_worker_metrics=self.detection_worker.metrics.to_dict(),
            tracking_worker_metrics=self.tracking_worker.metrics.to_dict(),
            persistence_worker_metrics=self.persistence_worker.metrics.to_dict() if self.persistence_worker is not None else None,
            vehicle_colour_worker_metrics=self.vehicle_colour_worker.metrics.to_dict() if self.vehicle_colour_worker is not None else None,
            anpr_worker_metrics=self.anpr_worker.metrics.to_dict() if self.anpr_worker is not None else None,
            queue_metrics={
                "frame_queue": self.frame_queue.metrics.to_dict(),
                "detection_queue": self.detection_queue.metrics.to_dict(),
                "completed_track_queue": self.completed_track_queue.metrics.to_dict(),
                "vehicle_colour_queue": self.vehicle_colour_queue.metrics.to_dict(),
                "anpr_queue": self.anpr_queue.metrics.to_dict(),
                "error_queue": self.error_queue.metrics.to_dict(),
            },
            thread_metrics={name: metrics.to_dict() for name, metrics in sorted(self.thread_metrics.items())},
            runtime_worker_counts={
                "expected_reader_count": len(self.camera_workers),
                "started_reader_count": sum(1 for worker in self.camera_workers if self.thread_metrics.get(worker.name, ThreadLifecycleMetrics()).started),
                "stopped_reader_count": sum(1 for worker in self.camera_workers if self.thread_metrics.get(worker.name, ThreadLifecycleMetrics()).stopped),
                "joined_reader_count": sum(1 for worker in self.camera_workers if self.thread_metrics.get(worker.name, ThreadLifecycleMetrics()).joined_successfully),
                "reader_thread_names": [worker.name for worker in self.camera_workers],
                "unfinished_reader_threads": [worker.name for worker in self.camera_workers if worker.is_alive()],
                "camera_reader_count": len(self.camera_workers),
                "detection_worker_count": 1,
                "tracking_worker_count": 1,
                "persistence_worker_count": 1 if self.persistence_worker is not None else 0,
                "vehicle_colour_worker_count": 1 if self.vehicle_colour_worker is not None else 0,
                "anpr_worker_count": 1 if self.anpr_worker is not None else 0,
            },
            shutdown_clean=shutdown_clean,
        )

    def _monitor(self) -> bool:
        workers = list(self.camera_workers) + [self.detection_worker, self.tracking_worker] + ([self.persistence_worker] if self.persistence_worker is not None else []) + ([self.vehicle_colour_worker] if self.vehicle_colour_worker is not None else []) + ([self.anpr_worker] if self.anpr_worker is not None else [])
        colour_end_sent = False
        anpr_end_sent = False
        while any(worker is not None and worker.is_alive() for worker in workers):
            self._drain_error_queue()
            if self.persistence_worker is None:
                self._drain_completed_queue()
            if (
                self.vehicle_colour_worker is not None
                and self.persistence_worker is not None
                and not self.persistence_worker.is_alive()
                and not colour_end_sent
            ):
                try:
                    self.vehicle_colour_queue.put(EndOfInputMessage(reason="persistence_complete"), timeout=self.worker_config.queue_put_timeout_seconds)
                    colour_end_sent = True
                except queue.Full:
                    pass
            if (
                self.anpr_worker is not None
                and self.persistence_worker is not None
                and not self.persistence_worker.is_alive()
                and not anpr_end_sent
            ):
                try:
                    self.anpr_queue.put(EndOfInputMessage(reason="persistence_complete"), timeout=self.worker_config.queue_put_timeout_seconds)
                    anpr_end_sent = True
                except queue.Full:
                    pass
            time.sleep(0.05)
        self._drain_error_queue()
        if self.persistence_worker is None:
            self._drain_completed_queue()
        if self.vehicle_colour_worker is not None and not colour_end_sent:
            try:
                self.vehicle_colour_queue.put(EndOfInputMessage(reason="persistence_complete"), timeout=self.worker_config.queue_put_timeout_seconds)
            except queue.Full:
                pass
        if self.anpr_worker is not None and not anpr_end_sent:
            try:
                self.anpr_queue.put(EndOfInputMessage(reason="persistence_complete"), timeout=self.worker_config.queue_put_timeout_seconds)
            except queue.Full:
                pass
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
