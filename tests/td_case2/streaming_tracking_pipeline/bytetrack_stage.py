"""ByteTrack tracking stages for Step 3 sequential validation."""

from __future__ import annotations

import sys
import time
import types
from types import SimpleNamespace
from typing import Any

from .adapters import TrackIdNormalizer
from .config import SUPPORTED_TRACKING_BACKENDS, TrackingConfig
from .contracts import validate_tracked_packet_matches_detection
from .schemas import BoundingBox, DetectionPacket, TrackedFramePacket, TrackedObject
from .validation import validate_allowed_value, validate_positive_float


def _install_lap_shim() -> None:
    if "lap" in sys.modules:
        return
    import numpy as np

    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:  # pragma: no cover
        from ultralytics.utils.ops import linear_sum_assignment  # type: ignore

    lap_module = types.ModuleType("lap")
    lap_module.__version__ = "0.5.12"

    def lapjv(cost_matrix: Any, extend_cost: bool = True, cost_limit: float = np.inf):
        rows, cols = linear_sum_assignment(cost_matrix)
        x = np.full(cost_matrix.shape[0], -1, dtype=int)
        y = np.full(cost_matrix.shape[1], -1, dtype=int)
        total_cost = 0.0
        for row_index, col_index in zip(rows, cols):
            value = float(cost_matrix[row_index, col_index])
            if value <= float(cost_limit):
                x[row_index] = col_index
                y[col_index] = row_index
                total_cost += value
        return total_cost, x, y

    lap_module.lapjv = lapjv
    sys.modules["lap"] = lap_module


class _ResultsLike:
    def __init__(self, detections: list[Any]):
        import numpy as np

        self.xyxy = np.asarray([item.bbox.to_xyxy() for item in detections], dtype=np.float32) if detections else np.zeros((0, 4), dtype=np.float32)
        self.conf = np.asarray([item.confidence for item in detections], dtype=np.float32) if detections else np.zeros((0,), dtype=np.float32)
        self.cls = np.asarray([item.class_id for item in detections], dtype=np.float32) if detections else np.zeros((0,), dtype=np.float32)

    @property
    def xywh(self):
        xywh = self.xyxy.copy()
        xywh[:, 2] = xywh[:, 2] - xywh[:, 0]
        xywh[:, 3] = xywh[:, 3] - xywh[:, 1]
        xywh[:, 0] = xywh[:, 0] + (xywh[:, 2] / 2.0)
        xywh[:, 1] = xywh[:, 1] + (xywh[:, 3] / 2.0)
        return xywh

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])

    def __getitem__(self, mask: Any) -> "_ResultsLike":
        import numpy as np

        clone = object.__new__(_ResultsLike)
        clone.xyxy = np.asarray(self.xyxy[mask], dtype=np.float32)
        clone.conf = np.asarray(self.conf[mask], dtype=np.float32)
        clone.cls = np.asarray(self.cls[mask], dtype=np.float32)
        return clone


class BaseByteTrackStage:
    """Shared sequential frame-order and metric handling."""

    backend_name = "base"

    def __init__(self, *, config: TrackingConfig, source_fps: float, normalizer: TrackIdNormalizer | None = None) -> None:
        self.config = config
        self.source_fps = validate_positive_float(source_fps, "source_fps")
        self.normalizer = normalizer or TrackIdNormalizer()
        self.packets_processed = 0
        self.detections_received = 0
        self.tracked_objects_emitted = 0
        self.frames_with_no_tracks = 0
        self.reset_count = 0
        self.flush_count = 0
        self._runtime_sec = 0.0
        self._last_frame_index: int | None = None
        self._source_id: str | None = None
        self._source_track_ids: set[str | int] = set()
        self._normalized_track_ids: set[int] = set()

    def process(self, packet: DetectionPacket) -> TrackedFramePacket:
        started_at = time.perf_counter()
        self._validate_order(packet)
        self.packets_processed += 1
        self.detections_received += len(packet.detections)
        tracks = self._process_native(packet)
        if not tracks:
            self.frames_with_no_tracks += 1
        self.tracked_objects_emitted += len(tracks)
        for track in tracks:
            self._normalized_track_ids.add(track.track_id)
            if track.source_track_id is not None:
                self._source_track_ids.add(track.source_track_id)
        output = TrackedFramePacket(
            source_id=packet.source_id,
            frame_index=packet.frame_index,
            timestamp_sec=packet.timestamp_sec,
            frame_width=packet.frame_width,
            frame_height=packet.frame_height,
            tracks=tracks,
            frame=packet.frame,
        )
        validate_tracked_packet_matches_detection(packet, output)
        self._runtime_sec += time.perf_counter() - started_at
        return output

    def reset(self) -> None:
        self.reset_count += 1
        self._last_frame_index = None
        self._source_id = None
        self.normalizer.reset()
        self._reset_native()

    def flush(self) -> list[Any]:
        self.flush_count += 1
        return []

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "packets_processed": self.packets_processed,
            "detections_received": self.detections_received,
            "tracked_objects_emitted": self.tracked_objects_emitted,
            "unique_source_track_ids": len(self._source_track_ids),
            "unique_normalized_track_ids": len(self._normalized_track_ids),
            "frames_with_no_tracks": self.frames_with_no_tracks,
            "reset_count": self.reset_count,
            "flush_count": self.flush_count,
            "runtime_sec": round(self._runtime_sec, 6),
        }

    def _validate_order(self, packet: DetectionPacket) -> None:
        if self._source_id is None:
            self._source_id = packet.source_id
        elif packet.source_id != self._source_id:
            raise ValueError("ByteTrack stage received a second source without reset().")
        if self._last_frame_index is not None:
            if packet.frame_index < self._last_frame_index:
                raise ValueError("ByteTrack stage rejected frame-order regression.")
            if packet.frame_index == self._last_frame_index:
                raise ValueError("ByteTrack stage rejected duplicate frame_index.")
        self._last_frame_index = packet.frame_index

    def _process_native(self, packet: DetectionPacket) -> list[TrackedObject]:
        raise NotImplementedError

    def _reset_native(self) -> None:
        raise NotImplementedError


class UltralyticsByteTrackStage(BaseByteTrackStage):
    """Ultralytics BYTETracker adapter using one native tracker per source."""

    backend_name = "ultralytics_bytetrack"

    def __init__(self, *, config: TrackingConfig, source_fps: float, tracker: Any | None = None, normalizer: TrackIdNormalizer | None = None) -> None:
        super().__init__(config=config, source_fps=source_fps, normalizer=normalizer)
        self._injected_tracker = tracker
        self._tracker: Any | None = tracker
        self._labels_by_source_id: dict[int, tuple[int, str, float, int | None, str | None, str | None, str | None]] = {}

    @property
    def tracker(self) -> Any:
        if self._tracker is None:
            _install_lap_shim()
            from ultralytics.trackers.byte_tracker import BYTETracker  # type: ignore

            track_buffer = max(1, int(self.config.lost_track_buffer))
            args = SimpleNamespace(
                track_high_thresh=float(self.config.track_high_threshold if self.config.track_high_threshold is not None else self.config.track_activation_threshold),
                track_low_thresh=float(self.config.track_low_threshold if self.config.track_low_threshold is not None else 0.10),
                new_track_thresh=float(self.config.new_track_threshold if self.config.new_track_threshold is not None else self.config.track_activation_threshold),
                match_thresh=float(self.config.match_threshold if self.config.match_threshold is not None else self.config.minimum_matching_threshold),
                track_buffer=track_buffer,
                fuse_score=bool(self.config.fuse_score),
                model="manual",
            )
            self._tracker = BYTETracker(args)
        return self._tracker

    def _process_native(self, packet: DetectionPacket) -> list[TrackedObject]:
        results = _ResultsLike(packet.detections)
        native_output = self.tracker.update(results, img=packet.frame)
        rows = native_output.tolist() if hasattr(native_output, "tolist") else list(native_output or [])
        tracks: list[TrackedObject] = []
        for row in rows:
            if len(row) < 5:
                continue
            x1, y1, x2, y2 = [float(value) for value in row[:4]]
            native_id = int(row[4])
            score = float(row[5]) if len(row) > 5 else None
            detection_index = int(row[7]) if len(row) > 7 else -1
            source_detection = packet.detections[detection_index] if 0 <= detection_index < len(packet.detections) else None
            if source_detection is not None:
                self._labels_by_source_id[native_id] = (
                    source_detection.class_id,
                    source_detection.class_name,
                    source_detection.confidence,
                    source_detection.raw_class_id,
                    source_detection.raw_class_name,
                    source_detection.object_group,
                    source_detection.detector_source,
                )
            class_id, class_name, confidence, raw_class_id, raw_class_name, object_group, detector_source = self._labels_by_source_id.get(native_id, (0, "object", 0.0, None, None, None, None))
            if source_detection is not None:
                confidence = source_detection.confidence
            elif score is not None:
                confidence = max(0.0, min(1.0, score))
            try:
                bbox = BoundingBox(x1, y1, x2, y2).clip(packet.frame_width, packet.frame_height)
            except ValueError:
                continue
            tracks.append(
                TrackedObject(
                    track_id=self.normalizer.normalize(native_id),
                    source_track_id=native_id,
                    bbox=bbox,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                    frame_index=packet.frame_index,
                    timestamp_sec=packet.timestamp_sec,
                    raw_class_id=raw_class_id,
                    raw_class_name=raw_class_name,
                    normalized_class_name=class_name,
                    object_group=object_group,
                    detector_source=detector_source,
                )
            )
        return sorted(tracks, key=lambda item: (item.track_id, item.class_id, item.bbox.x1))

    def _reset_native(self) -> None:
        self._tracker = self._injected_tracker
        self._labels_by_source_id.clear()


class SupervisionByteTrackStage(BaseByteTrackStage):
    """Supervision ByteTrack adapter. Requires supervision to be installed."""

    backend_name = "supervision_bytetrack"

    def __init__(self, *, config: TrackingConfig, source_fps: float, tracker: Any | None = None, normalizer: TrackIdNormalizer | None = None) -> None:
        super().__init__(config=config, source_fps=source_fps, normalizer=normalizer)
        self._injected_tracker = tracker
        self._tracker: Any | None = tracker

    @property
    def tracker(self) -> Any:
        if self._tracker is None:
            try:
                import supervision as sv  # type: ignore
            except Exception as exc:
                raise RuntimeError("supervision is not installed; cannot use supervision_bytetrack backend.") from exc
            self._tracker = sv.ByteTrack(
                track_activation_threshold=self.config.track_activation_threshold,
                lost_track_buffer=self.config.lost_track_buffer,
                minimum_matching_threshold=self.config.minimum_matching_threshold,
                minimum_consecutive_frames=self.config.minimum_consecutive_frames,
                frame_rate=self.source_fps,
            )
        return self._tracker

    def _process_native(self, packet: DetectionPacket) -> list[TrackedObject]:
        try:
            import numpy as np
            import supervision as sv  # type: ignore
        except Exception as exc:
            if self._injected_tracker is None:
                raise RuntimeError("supervision is not installed; cannot use supervision_bytetrack backend.") from exc
            sv = None
            np = None
        if self._injected_tracker is not None and not hasattr(self._injected_tracker, "update_with_detections"):
            native_tracks = self._injected_tracker.update(packet.detections)
            return self._from_fake_tracks(native_tracks, packet)
        xyxy = np.asarray([item.bbox.to_xyxy() for item in packet.detections], dtype=float) if packet.detections else np.empty((0, 4), dtype=float)
        confidence = np.asarray([item.confidence for item in packet.detections], dtype=float) if packet.detections else np.empty((0,), dtype=float)
        class_id = np.asarray([item.class_id for item in packet.detections], dtype=int) if packet.detections else np.empty((0,), dtype=int)
        detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        tracked = self.tracker.update_with_detections(detections)
        tracks: list[TrackedObject] = []
        tracker_ids = list(tracked.tracker_id) if tracked.tracker_id is not None else []
        for index, native_id in enumerate(tracker_ids):
            if native_id is None:
                continue
            source_detection = packet.detections[index] if index < len(packet.detections) else None
            if source_detection is None:
                continue
            tracks.append(
                TrackedObject(
                    track_id=self.normalizer.normalize(int(native_id)),
                    source_track_id=int(native_id),
                    bbox=source_detection.bbox,
                    confidence=source_detection.confidence,
                    class_id=source_detection.class_id,
                    class_name=source_detection.class_name,
                    frame_index=packet.frame_index,
                    timestamp_sec=packet.timestamp_sec,
                    raw_class_id=source_detection.raw_class_id,
                    raw_class_name=source_detection.raw_class_name,
                    normalized_class_name=source_detection.normalized_class_name,
                    object_group=source_detection.object_group,
                    detector_source=source_detection.detector_source,
                )
            )
        return sorted(tracks, key=lambda item: (item.track_id, item.class_id, item.bbox.x1))

    def _from_fake_tracks(self, native_tracks: Any, packet: DetectionPacket) -> list[TrackedObject]:
        tracks: list[TrackedObject] = []
        for row in list(native_tracks or []):
            source_id = row["track_id"]
            bbox = _bbox_from_any(row.get("bbox", row.get("bbox_xyxy")))
            tracks.append(
                TrackedObject(
                    track_id=self.normalizer.normalize(source_id),
                    source_track_id=source_id,
                    bbox=bbox.clip(packet.frame_width, packet.frame_height),
                    confidence=float(row.get("confidence", 0.0)),
                    class_id=int(row.get("class_id", 0)),
                    class_name=str(row.get("class_name", "object")),
                    frame_index=packet.frame_index,
                    timestamp_sec=packet.timestamp_sec,
                    raw_class_id=int(row.get("raw_class_id", row.get("class_id", 0))),
                    raw_class_name=str(row.get("raw_class_name", row.get("class_name", "object"))),
                    normalized_class_name=str(row.get("normalized_class_name", row.get("class_name", "object"))),
                    object_group=str(row.get("object_group")) if row.get("object_group") else None,
                    detector_source=str(row.get("detector_source")) if row.get("detector_source") else None,
                )
            )
        return tracks

    def _reset_native(self) -> None:
        self._tracker = self._injected_tracker


def _bbox_from_any(value: Any) -> BoundingBox:
    values = list(value)
    return BoundingBox(float(values[0]), float(values[1]), float(values[2]), float(values[3]))


def create_bytetrack_stage(config: TrackingConfig, source_fps: float, tracker: Any | None = None) -> BaseByteTrackStage:
    backend = validate_allowed_value(config.backend, SUPPORTED_TRACKING_BACKENDS, "tracking.backend")
    if backend == "ultralytics_bytetrack":
        return UltralyticsByteTrackStage(config=config, source_fps=source_fps, tracker=tracker)
    if backend == "supervision_bytetrack":
        return SupervisionByteTrackStage(config=config, source_fps=source_fps, tracker=tracker)
    raise ValueError(f"Unsupported tracking backend: {backend}")
