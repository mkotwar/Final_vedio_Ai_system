from __future__ import annotations

import queue
import threading
import time
import traceback

from ..persistence.tracking_persistence_service import TrackingPersistenceService
from .worker_config import WorkerConfig
from .worker_messages import CompletedTrackMessage, EndOfInputMessage, WorkerErrorMessage
from .worker_metrics import PersistenceWorkerMetrics, TrackedQueue


class PersistenceWorker(threading.Thread):
    def __init__(
        self,
        *,
        persistence_service: TrackingPersistenceService,
        completed_track_queue: TrackedQueue,
        error_queue: TrackedQueue,
        shutdown_event: threading.Event,
        worker_config: WorkerConfig,
    ) -> None:
        super().__init__(name="persistence_worker", daemon=worker_config.persistence_worker_daemon)
        self.persistence_service = persistence_service
        self.completed_track_queue = completed_track_queue
        self.error_queue = error_queue
        self.shutdown_event = shutdown_event
        self.worker_config = worker_config
        self.metrics = PersistenceWorkerMetrics()
        self.finalized_tracks: list[CompletedTrackMessage] = []
        self.results_by_track_uuid: dict[str, object] = {}

    def run(self) -> None:
        try:
            while True:
                try:
                    item, _ = self.completed_track_queue.get(timeout=self.worker_config.queue_get_timeout_seconds)
                except queue.Empty:
                    if self.shutdown_event.is_set():
                        break
                    continue
                if isinstance(item, EndOfInputMessage):
                    break
                if not isinstance(item, CompletedTrackMessage):
                    continue
                self.finalized_tracks.append(item)
                self.metrics.tracks_received += 1
                started = time.perf_counter()
                result = self.persistence_service.save_completed_track(item.track)
                self.metrics.database_time_seconds += time.perf_counter() - started
                self.results_by_track_uuid[item.track.track_uuid] = result
                if result.status == "inserted":
                    self.metrics.tracks_inserted += 1
                    self.metrics.observations_written += result.observations_written
                elif result.status == "already_exists":
                    self.metrics.tracks_already_existing += 1
                elif result.status in {"skipped_discarded", "skipped_invalid_state", "dry_run"}:
                    self.metrics.tracks_skipped += 1
                    if result.status == "dry_run":
                        self.metrics.observations_written += result.observations_written
                elif result.status == "failed":
                    self.metrics.tracks_failed += 1
        except Exception as exc:
            self.metrics.errors += 1
            fatal = self.worker_config.stop_on_persistence_error
            self._emit_error(exc, fatal=fatal)
            if fatal:
                self.shutdown_event.set()

    def _emit_error(self, exc: Exception, *, fatal: bool) -> None:
        message = WorkerErrorMessage(
            worker_name=self.name,
            worker_type="persistence_worker",
            camera_code=None,
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback_text=traceback.format_exc(),
            fatal=fatal,
        )
        try:
            self.error_queue.put(message, timeout=self.worker_config.queue_put_timeout_seconds)
        except queue.Full:
            pass
