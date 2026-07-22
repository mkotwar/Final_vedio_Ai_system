from __future__ import annotations

import queue
import threading
import traceback
from pathlib import Path

import cv2

from ..detection.detection_models import DetectionPacket
from ..tracking.camera_detection_router import CameraDetectionRouter
from ..tracking.tracking_models import LocalVehicleTrack, TrackObservation
from .worker_config import WorkerConfig
from .worker_messages import CompletedTrackMessage, EndOfCameraMessage, EndOfInputMessage, WorkerErrorMessage
from .worker_metrics import TrackingWorkerMetrics, TrackedQueue


class TrackingWorker(threading.Thread):
    def __init__(
        self,
        *,
        router: CameraDetectionRouter,
        detection_queue: TrackedQueue,
        completed_track_queue: TrackedQueue,
        error_queue: TrackedQueue,
        shutdown_event: threading.Event,
        worker_config: WorkerConfig,
        save_sample_frames: bool = False,
        sample_frame_limit_per_camera: int = 1,
        sample_output_dir: str | Path | None = None,
    ) -> None:
        super().__init__(name="tracking_router_worker", daemon=worker_config.tracking_worker_daemon)
        self.router = router
        self.detection_queue = detection_queue
        self.completed_track_queue = completed_track_queue
        self.error_queue = error_queue
        self.shutdown_event = shutdown_event
        self.worker_config = worker_config
        self.save_sample_frames = save_sample_frames
        self.sample_frame_limit_per_camera = sample_frame_limit_per_camera
        self.sample_output_dir = Path(sample_output_dir) if sample_output_dir is not None else None
        self.metrics = TrackingWorkerMetrics()
        self._last_frame_number_by_camera: dict[str, int] = {}
        self._emitted_track_uuids: set[str] = set()
        self._sample_counts: dict[str, int] = {}

    def run(self) -> None:
        try:
            while True:
                try:
                    item, _ = self.detection_queue.get(timeout=self.worker_config.queue_get_timeout_seconds)
                except queue.Empty:
                    if self.shutdown_event.is_set():
                        break
                    continue
                if isinstance(item, EndOfCameraMessage):
                    self._flush_camera(item.camera_code)
                    continue
                if isinstance(item, EndOfInputMessage):
                    self._flush_all()
                    self._emit_completed_message(item)
                    break
                if not isinstance(item, DetectionPacket):
                    continue
                self._validate_order(item)
                result = self.router.route(item)
                self.metrics.detection_packets_received += 1
                self.metrics.track_observations_created += len(result.observations)
                camera_code = item.camera_code
                self.metrics.per_camera_frames[camera_code] = self.metrics.per_camera_frames.get(camera_code, 0) + 1
                self.metrics.per_camera_detections[camera_code] = self.metrics.per_camera_detections.get(camera_code, 0) + len(item.detections)
                self.metrics.per_camera_track_observations[camera_code] = self.metrics.per_camera_track_observations.get(camera_code, 0) + len(result.observations)
                self.metrics.per_camera_first_frame.setdefault(camera_code, item.frame_number)
                self.metrics.per_camera_last_frame[camera_code] = item.frame_number
                if self.save_sample_frames:
                    self._save_sample_frame(item, result.observations, len(result.active_tracks))
                self._emit_tracks(camera_code, result.completed_tracks)
        except Exception as exc:
            self.metrics.tracking_errors += 1
            fatal = self.worker_config.stop_on_tracking_error
            self._emit_error(exc, fatal=fatal)
            if fatal:
                self.shutdown_event.set()
            try:
                self._emit_completed_message(EndOfInputMessage(reason="tracking_error"))
            except queue.Full:
                pass

    def _validate_order(self, packet: DetectionPacket) -> None:
        previous = self._last_frame_number_by_camera.get(packet.camera_code)
        if previous is not None and packet.frame_number <= previous:
            self.metrics.out_of_order_packets += 1
            raise ValueError(f"Out-of-order detection packet for {packet.camera_code}: {packet.frame_number} after {previous}")
        self._last_frame_number_by_camera[packet.camera_code] = packet.frame_number

    def _flush_camera(self, camera_code: str) -> None:
        result = self.router.flush_camera(camera_code)
        self.metrics.camera_flush_count += 1
        self._emit_tracks(camera_code, result.completed_tracks)

    def _flush_all(self) -> None:
        result = self.router.flush_all()
        self.metrics.camera_flush_count += 1
        for track in result.completed_tracks:
            self._emit_tracks(track.camera_code, [track])

    def _emit_tracks(self, camera_code: str, tracks: list[LocalVehicleTrack]) -> None:
        for track in tracks:
            if track.track_uuid in self._emitted_track_uuids:
                continue
            self._emitted_track_uuids.add(track.track_uuid)
            if track.state == "completed":
                self.metrics.completed_tracks += 1
                self.metrics.per_camera_completed_tracks[camera_code] = self.metrics.per_camera_completed_tracks.get(camera_code, 0) + 1
            elif track.state == "discarded":
                self.metrics.discarded_tracks += 1
                self.metrics.per_camera_discarded_tracks[camera_code] = self.metrics.per_camera_discarded_tracks.get(camera_code, 0) + 1
            message = CompletedTrackMessage(camera_code=camera_code, track=track)
            while not self.shutdown_event.is_set():
                try:
                    self.completed_track_queue.put(message, timeout=self.worker_config.queue_put_timeout_seconds)
                    break
                except queue.Full:
                    continue

    def _emit_completed_message(self, message: EndOfInputMessage) -> None:
        while not self.shutdown_event.is_set():
            try:
                self.completed_track_queue.put(message, timeout=self.worker_config.queue_put_timeout_seconds)
                return
            except queue.Full:
                continue

    def _save_sample_frame(self, packet: DetectionPacket, observations: list[TrackObservation], active_track_count: int) -> None:
        if self.sample_output_dir is None:
            return
        camera_code = packet.camera_code
        count = self._sample_counts.get(camera_code, 0)
        if count >= self.sample_frame_limit_per_camera or packet.frame is None:
            return
        frame = packet.frame.copy()
        for observation in observations:
            x1, y1, x2, y2 = (int(round(value)) for value in observation.bbox_xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
            label = f"{observation.camera_code} | {observation.class_name} | ID {observation.local_track_id} | {observation.confidence:.2f}"
            cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2, cv2.LINE_AA)
        overlay_lines = [
            f"camera={packet.camera_code}",
            f"frame={packet.frame_number}",
            f"time={packet.video_time_seconds:.2f}s",
            f"active_tracks={active_track_count}",
        ]
        y = 24
        for line in overlay_lines:
            cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
            y += 24
        self._sample_counts[camera_code] = count + 1
        camera_dir = self.sample_output_dir / camera_code
        camera_dir.mkdir(parents=True, exist_ok=True)
        sample_path = camera_dir / f"sample_{self._sample_counts[camera_code]:06d}.jpg"
        cv2.imwrite(str(sample_path), frame)

    def _emit_error(self, exc: Exception, *, fatal: bool) -> None:
        message = WorkerErrorMessage(
            worker_name=self.name,
            worker_type="tracking_worker",
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
