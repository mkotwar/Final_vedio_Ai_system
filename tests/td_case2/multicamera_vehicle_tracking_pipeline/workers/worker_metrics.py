from __future__ import annotations

import queue
import time
from dataclasses import asdict
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class QueueMetrics:
    max_size: int
    maximum_observed_size: int = 0
    put_timeouts: int = 0
    get_timeouts: int = 0

    def observe_size(self, size: int) -> None:
        if size > self.maximum_observed_size:
            self.maximum_observed_size = size

    def to_dict(self) -> dict[str, int]:
        return {
            "configured_max_size": self.max_size,
            "maximum_observed_size": self.maximum_observed_size,
            "put_timeouts": self.put_timeouts,
            "get_timeouts": self.get_timeouts,
        }


class TrackedQueue:
    def __init__(self, max_size: int) -> None:
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max_size)
        self.metrics = QueueMetrics(max_size=max_size)

    def put(self, item: Any, timeout: float) -> float:
        started = time.perf_counter()
        try:
            self._queue.put(item, timeout=timeout)
        except queue.Full:
            self.metrics.put_timeouts += 1
            raise
        self.metrics.observe_size(self._queue.qsize())
        return time.perf_counter() - started

    def get(self, timeout: float) -> tuple[Any, float]:
        started = time.perf_counter()
        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty:
            self.metrics.get_timeouts += 1
            raise
        self.metrics.observe_size(self._queue.qsize())
        return item, time.perf_counter() - started

    def get_nowait(self) -> Any:
        item = self._queue.get_nowait()
        self.metrics.observe_size(self._queue.qsize())
        return item

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()


@dataclass(slots=True)
class CameraReaderMetrics:
    camera_code: str
    frames_read: int = 0
    first_frame_number: int | None = None
    last_frame_number: int | None = None
    queue_put_count: int = 0
    queue_wait_seconds: float = 0.0
    errors: int = 0
    start_time: float | None = None
    end_time: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class DetectionWorkerMetrics:
    frames_received: int = 0
    frames_processed: int = 0
    detections_produced: int = 0
    empty_detection_frames: int = 0
    average_inference_time_ms: float = 0.0
    maximum_inference_time_ms: float = 0.0
    frame_queue_wait_time_seconds: float = 0.0
    detection_queue_block_time_seconds: float = 0.0
    errors: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class TrackingWorkerMetrics:
    packets_received: int = 0
    track_observations: int = 0
    completed_tracks: int = 0
    discarded_tracks: int = 0
    out_of_order_packets: int = 0
    camera_flushes: int = 0
    tracking_errors: int = 0
    per_camera_frames: dict[str, int] = field(default_factory=dict)
    per_camera_detections: dict[str, int] = field(default_factory=dict)
    per_camera_track_observations: dict[str, int] = field(default_factory=dict)
    per_camera_completed_tracks: dict[str, int] = field(default_factory=dict)
    per_camera_discarded_tracks: dict[str, int] = field(default_factory=dict)
    per_camera_first_frame: dict[str, int] = field(default_factory=dict)
    per_camera_last_frame: dict[str, int] = field(default_factory=dict)
    unique_track_ids_by_camera: dict[str, list[int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class PersistenceWorkerMetrics:
    tracks_received: int = 0
    tracks_inserted: int = 0
    tracks_skipped: int = 0
    tracks_already_existing: int = 0
    tracks_failed: int = 0
    observations_written: int = 0
    database_time_seconds: float = 0.0
    errors: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class VehicleColourWorkerMetrics:
    jobs_received: int = 0
    results_persisted: int = 0
    results_skipped: int = 0
    results_failed: int = 0
    queue_wait_seconds: float = 0.0
    inference_time_seconds: float = 0.0
    errors: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class AnprWorkerMetrics:
    jobs_received: int = 0
    results_persisted: int = 0
    results_skipped: int = 0
    results_failed: int = 0
    queue_wait_seconds: float = 0.0
    inference_time_seconds: float = 0.0
    errors: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class ThreadLifecycleMetrics:
    started: bool = False
    stopped: bool = False
    joined_successfully: bool = False

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)
