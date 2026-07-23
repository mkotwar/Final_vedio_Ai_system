from __future__ import annotations

from collections.abc import Iterable
import queue
import threading
import time
import traceback

from ..detection.detection_models import DetectionPacket
from ..detection.vehicle_detector import SharedVehicleDetector
from ..ingestion.frame_packet import FramePacket
from .worker_config import WorkerConfig
from .worker_messages import EndOfCameraMessage, EndOfInputMessage, WorkerErrorMessage
from .worker_metrics import DetectionWorkerMetrics, TrackedQueue


class DetectionWorker(threading.Thread):
    def __init__(
        self,
        *,
        detector: SharedVehicleDetector,
        expected_camera_codes: Iterable[str],
        frame_queue: TrackedQueue,
        detection_queue: TrackedQueue,
        error_queue: TrackedQueue,
        shutdown_event: threading.Event,
        worker_config: WorkerConfig,
    ) -> None:
        super().__init__(name="shared_detection_worker", daemon=worker_config.detection_worker_daemon)
        self.detector = detector
        self.expected_camera_codes = tuple(expected_camera_codes)
        self._expected_camera_code_set = set(self.expected_camera_codes)
        self.frame_queue = frame_queue
        self.detection_queue = detection_queue
        self.error_queue = error_queue
        self.shutdown_event = shutdown_event
        self.worker_config = worker_config
        self.metrics = DetectionWorkerMetrics()

    def run(self) -> None:
        ended_cameras: set[str] = set()
        forwarded_end = False
        try:
            while True:
                try:
                    item, wait_time = self.frame_queue.get(timeout=self.worker_config.queue_get_timeout_seconds)
                except queue.Empty:
                    if self.shutdown_event.is_set():
                        break
                    continue
                self.metrics.frame_queue_wait_time_seconds += wait_time
                if isinstance(item, EndOfCameraMessage):
                    if item.camera_code not in self._expected_camera_code_set:
                        raise ValueError(f"Received end-of-camera for unexpected camera: {item.camera_code}")
                    if item.camera_code in ended_cameras:
                        continue
                    ended_cameras.add(item.camera_code)
                    self._forward(item)
                    if ended_cameras == self._expected_camera_code_set:
                        self._forward(EndOfInputMessage())
                        forwarded_end = True
                        break
                    continue
                if not isinstance(item, FramePacket):
                    continue
                if item.camera_code not in self._expected_camera_code_set:
                    raise ValueError(f"Received frame for unexpected camera: {item.camera_code}")
                detection_packet = self.detector.detect(item)
                self.metrics.frames_received += 1
                self.metrics.frames_processed += 1
                self.metrics.detections_produced += len(detection_packet.detections)
                if not detection_packet.detections:
                    self.metrics.empty_detection_frames += 1
                self.metrics.maximum_inference_time_ms = max(self.metrics.maximum_inference_time_ms, detection_packet.inference_time_ms)
                previous_total = self.metrics.average_inference_time_ms * max(self.metrics.frames_received - 1, 0)
                self.metrics.average_inference_time_ms = (previous_total + detection_packet.inference_time_ms) / self.metrics.frames_received
                duration = self._forward(detection_packet)
                self.metrics.detection_queue_block_time_seconds += duration
        except Exception as exc:
            self.metrics.errors += 1
            fatal = self.worker_config.stop_on_detector_error
            self._emit_error(exc, fatal=fatal)
            if fatal:
                self.shutdown_event.set()
        finally:
            if not forwarded_end:
                try:
                    self._forward(EndOfInputMessage(reason="detector_shutdown"))
                except queue.Full:
                    pass

    def _forward(self, item: DetectionPacket | EndOfCameraMessage | EndOfInputMessage) -> float:
        while not self.shutdown_event.is_set():
            try:
                return self.detection_queue.put(item, timeout=self.worker_config.queue_put_timeout_seconds)
            except queue.Full:
                continue
        return 0.0

    def _emit_error(self, exc: Exception, *, fatal: bool) -> None:
        message = WorkerErrorMessage(
            worker_name=self.name,
            worker_type="detection_worker",
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
