from __future__ import annotations

import queue
import threading
import time
import traceback

from ..enrichment.vehicle_colour_enrichment_service import VehicleColourEnrichmentService
from .worker_config import WorkerConfig
from .worker_messages import EndOfInputMessage, VehicleColourJobMessage, WorkerErrorMessage
from .worker_metrics import TrackedQueue, VehicleColourWorkerMetrics


class VehicleColourWorker(threading.Thread):
    def __init__(
        self,
        *,
        enrichment_service: VehicleColourEnrichmentService,
        vehicle_colour_queue: TrackedQueue,
        error_queue: TrackedQueue,
        shutdown_event: threading.Event,
        worker_config: WorkerConfig,
    ) -> None:
        super().__init__(name="vehicle_colour_worker", daemon=worker_config.vehicle_colour_worker_daemon)
        self.enrichment_service = enrichment_service
        self.vehicle_colour_queue = vehicle_colour_queue
        self.error_queue = error_queue
        self.shutdown_event = shutdown_event
        self.worker_config = worker_config
        self.metrics = VehicleColourWorkerMetrics()
        self.results_by_track_uuid: dict[str, object] = {}

    def run(self) -> None:
        try:
            while True:
                try:
                    item, waited = self.vehicle_colour_queue.get(timeout=self.worker_config.queue_get_timeout_seconds)
                except queue.Empty:
                    if self.shutdown_event.is_set():
                        break
                    continue
                self.metrics.queue_wait_seconds += waited
                if isinstance(item, EndOfInputMessage):
                    break
                if not isinstance(item, VehicleColourJobMessage):
                    continue
                self.metrics.jobs_received += 1
                started = time.perf_counter()
                result = self.enrichment_service.enrich_track(
                    completed_track=item.track,
                    persisted_vehicle_track_id=str(item.persistence_result.database_track_id),
                )
                self.metrics.inference_time_seconds += time.perf_counter() - started
                self.results_by_track_uuid[item.track.track_uuid] = result
                if result.result.status == "SUCCESS":
                    self.metrics.results_persisted += 1 if result.persisted else 0
                elif result.result.status == "DISABLED":
                    self.metrics.results_skipped += 1
                elif result.result.status in {"LOW_CONFIDENCE", "UNKNOWN_RESULT"}:
                    self.metrics.results_skipped += 1
                else:
                    self.metrics.results_failed += 1
        except Exception as exc:
            self.metrics.errors += 1
            message = WorkerErrorMessage(
                worker_name=self.name,
                worker_type="vehicle_colour_worker",
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
