from __future__ import annotations

import queue
import threading
import time
import traceback

from ..enrichment.anpr_enrichment_service import AnprEnrichmentService
from .worker_config import WorkerConfig
from .worker_messages import AnprJobMessage, EndOfInputMessage, WorkerErrorMessage
from .worker_metrics import AnprWorkerMetrics, TrackedQueue


class AnprWorker(threading.Thread):
    def __init__(
        self,
        *,
        enrichment_service: AnprEnrichmentService,
        anpr_queue: TrackedQueue,
        error_queue: TrackedQueue,
        shutdown_event: threading.Event,
        worker_config: WorkerConfig,
    ) -> None:
        super().__init__(name="anpr_worker", daemon=worker_config.anpr_worker_daemon)
        self.enrichment_service = enrichment_service
        self.anpr_queue = anpr_queue
        self.error_queue = error_queue
        self.shutdown_event = shutdown_event
        self.worker_config = worker_config
        self.metrics = AnprWorkerMetrics()
        self.results_by_track_uuid: dict[str, object] = {}

    def run(self) -> None:
        try:
            while True:
                try:
                    item, waited = self.anpr_queue.get(timeout=self.worker_config.queue_get_timeout_seconds)
                except queue.Empty:
                    if self.shutdown_event.is_set():
                        break
                    continue
                self.metrics.queue_wait_seconds += waited
                if isinstance(item, EndOfInputMessage):
                    break
                if not isinstance(item, AnprJobMessage):
                    continue
                self.metrics.jobs_received += 1
                started = time.perf_counter()
                result = self.enrichment_service.enrich_track(
                    completed_track=item.track,
                    persisted_vehicle_track_id=str(item.persistence_result.database_track_id),
                )
                self.metrics.inference_time_seconds += time.perf_counter() - started
                self.results_by_track_uuid[item.track.track_uuid] = result
                if result.persisted:
                    self.metrics.results_persisted += 1
                elif result.status in {"DISABLED", "NO_VEHICLE_EVIDENCE", "NO_PLATE_DETECTED", "OCR_EMPTY"}:
                    self.metrics.results_skipped += 1
                else:
                    self.metrics.results_failed += 1
        except Exception as exc:
            self.metrics.errors += 1
            message = WorkerErrorMessage(
                worker_name=self.name,
                worker_type="anpr_worker",
                camera_code=None,
                error_type=type(exc).__name__,
                error_message=str(exc),
                traceback_text=traceback.format_exc(),
                fatal=False,
            )
            try:
                self.error_queue.put(message, timeout=self.worker_config.queue_put_timeout_seconds)
            except queue.Full:
                pass
