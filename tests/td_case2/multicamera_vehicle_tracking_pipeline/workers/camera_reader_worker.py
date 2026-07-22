from __future__ import annotations

import queue
import threading
import time
import traceback

from ..ingestion.camera_config import CameraConfig
from ..ingestion.camera_source import CameraSource
from .worker_config import WorkerConfig
from .worker_messages import EndOfCameraMessage, WorkerErrorMessage
from .worker_metrics import CameraReaderMetrics, TrackedQueue


class CameraReaderWorker(threading.Thread):
    def __init__(
        self,
        *,
        camera_config: CameraConfig,
        frame_queue: TrackedQueue,
        error_queue: TrackedQueue,
        shutdown_event: threading.Event,
        worker_config: WorkerConfig,
        max_frames_per_camera: int | None = None,
    ) -> None:
        super().__init__(name=f"camera_reader_{camera_config.camera_code}", daemon=worker_config.camera_reader_daemon)
        self.camera_config = camera_config
        self.frame_queue = frame_queue
        self.error_queue = error_queue
        self.shutdown_event = shutdown_event
        self.worker_config = worker_config
        self.max_frames_per_camera = max_frames_per_camera
        self.metrics = CameraReaderMetrics(camera_code=camera_config.camera_code)

    def run(self) -> None:
        self.metrics.start_time = time.perf_counter()
        sent_end = False
        source = CameraSource(self.camera_config)
        try:
            source.open()
            while not self.shutdown_event.is_set():
                if self.max_frames_per_camera is not None and self.metrics.frames_read >= self.max_frames_per_camera:
                    break
                packet = source.read_next()
                if packet is None:
                    break
                while not self.shutdown_event.is_set():
                    try:
                        duration = self.frame_queue.put(packet, timeout=self.worker_config.queue_put_timeout_seconds)
                        self.metrics.frames_read += 1
                        self.metrics.queue_put_count += 1
                        if duration > 0.001:
                            self.metrics.queue_block_count += 1
                            self.metrics.queue_block_time_seconds += duration
                        break
                    except queue.Full:
                        self.metrics.queue_block_count += 1
                        self.metrics.queue_block_time_seconds += self.worker_config.queue_put_timeout_seconds
            self._emit_end_of_camera()
            sent_end = True
        except Exception as exc:
            self.metrics.read_errors += 1
            self._emit_error(exc, fatal=self.worker_config.stop_on_camera_error)
            if self.worker_config.stop_on_camera_error:
                self.shutdown_event.set()
            if not sent_end:
                self._emit_end_of_camera()
                sent_end = True
        finally:
            source.close()
            if not sent_end:
                self._emit_end_of_camera()
            self.metrics.end_time = time.perf_counter()

    def _emit_end_of_camera(self) -> None:
        message = EndOfCameraMessage(camera_code=self.camera_config.camera_code)
        while not self.shutdown_event.is_set():
            try:
                self.frame_queue.put(message, timeout=self.worker_config.queue_put_timeout_seconds)
                return
            except queue.Full:
                continue

    def _emit_error(self, exc: Exception, *, fatal: bool) -> None:
        message = WorkerErrorMessage(
            worker_name=self.name,
            worker_type="camera_reader",
            camera_code=self.camera_config.camera_code,
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback_text=traceback.format_exc(),
            fatal=fatal,
        )
        try:
            self.error_queue.put(message, timeout=self.worker_config.queue_put_timeout_seconds)
        except queue.Full:
            pass
