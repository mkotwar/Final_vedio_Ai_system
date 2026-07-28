from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from ..detection.detection_models import DetectionPacket
from .track_lifecycle import LocalTrackLifecycle
from .supervision_conversion import build_supervision_debug_snapshot, to_supervision_detections
from .tracker_factory import TrackerFactory
from .tracking_config import TrackingConfig
from .tracking_models import LocalVehicleTrack
from .tracking_models import TrackObservation

LOGGER = logging.getLogger(__name__)


class _ResultsLike:
    def __init__(self, packet: DetectionPacket):
        import numpy as np

        detections = packet.detections
        self.xyxy = (
            np.asarray([item.bbox_xyxy for item in detections], dtype=np.float32)
            if detections
            else np.zeros((0, 4), dtype=np.float32)
        )
        self.conf = np.asarray([item.confidence for item in detections], dtype=np.float32) if detections else np.zeros((0,), dtype=np.float32)
        self.cls = np.asarray([item.class_id for item in detections], dtype=np.float32) if detections else np.zeros((0,), dtype=np.float32)

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])

    @property
    def xywh(self):
        xywh = self.xyxy.copy()
        xywh[:, 2] = xywh[:, 2] - xywh[:, 0]
        xywh[:, 3] = xywh[:, 3] - xywh[:, 1]
        xywh[:, 0] = xywh[:, 0] + (xywh[:, 2] / 2.0)
        xywh[:, 1] = xywh[:, 1] + (xywh[:, 3] / 2.0)
        return xywh

    def __getitem__(self, mask: Any) -> "_ResultsLike":
        import numpy as np

        clone = object.__new__(_ResultsLike)
        clone.xyxy = np.asarray(self.xyxy[mask], dtype=np.float32)
        clone.conf = np.asarray(self.conf[mask], dtype=np.float32)
        clone.cls = np.asarray(self.cls[mask], dtype=np.float32)
        return clone


@dataclass(slots=True)
class CameraTrackingResult:
    observations: list[TrackObservation]
    completed_tracks: list[LocalVehicleTrack]
    active_tracks: list[LocalVehicleTrack]
    native_observations: list[TrackObservation] | None = None


class CameraTracker:
    def __init__(
        self,
        camera_code: str,
        tracking_config: TrackingConfig,
        tracker_factory: TrackerFactory,
        lifecycle: LocalTrackLifecycle,
    ) -> None:
        self.camera_code = camera_code
        self.tracking_config = tracking_config
        self.tracker_factory = tracker_factory
        self.lifecycle = lifecycle
        self._labels_by_track_id: dict[int, tuple[str, float]] = {}
        self._supervision_debug_frames_remaining = 5
        self._metrics = {
            "frames_with_input_detections": 0,
            "frames_with_tracker_output": 0,
            "detections_without_tracker_id": 0,
            "tracker_output_rows": 0,
            "resolved_frame_rate": None,
        }

    def update(self, packet: DetectionPacket) -> CameraTrackingResult:
        if packet.camera_code != self.camera_code:
            raise ValueError(f"CameraTracker for {self.camera_code} received packet for {packet.camera_code}")
        tracker = self.tracker_factory.get_or_create(self.camera_code, frame_rate=packet.source_fps or self.tracking_config.frame_rate)
        if self._metrics["resolved_frame_rate"] is None:
            self._metrics["resolved_frame_rate"] = float(packet.source_fps or self.tracking_config.frame_rate or 0.0)
        if hasattr(tracker, "update_with_detections"):
            raw_observations = self._update_with_supervision_tracker(tracker, packet)
        else:
            raw_observations = self._update_with_row_tracker(tracker, packet)
        lifecycle_result = self.lifecycle.update(packet, raw_observations)
        return CameraTrackingResult(
            observations=lifecycle_result.observations,
            completed_tracks=lifecycle_result.completed_tracks,
            active_tracks=lifecycle_result.active_tracks,
            native_observations=raw_observations,
        )

    def flush(self) -> CameraTrackingResult:
        result = self.lifecycle.flush_camera(self.camera_code)
        return CameraTrackingResult(observations=[], completed_tracks=result.completed_tracks, active_tracks=result.active_tracks)

    def diagnostics(self) -> dict[str, int | float | None]:
        return dict(self._metrics)

    def _update_with_supervision_tracker(self, tracker: Any, packet: DetectionPacket) -> list[TrackObservation]:
        detections = to_supervision_detections(packet)
        debug_snapshot = build_supervision_debug_snapshot(packet)
        if debug_snapshot.input_detection_count > 0:
            self._metrics["frames_with_input_detections"] += 1
        tracked = tracker.update_with_detections(detections)
        raw_observations: list[TrackObservation] = []
        tracker_ids = list(tracked.tracker_id) if getattr(tracked, "tracker_id", None) is not None else []
        xyxy_rows = list(tracked.xyxy) if getattr(tracked, "xyxy", None) is not None else []
        confidence_rows = list(tracked.confidence) if getattr(tracked, "confidence", None) is not None else []
        class_rows = list(tracked.class_id) if getattr(tracked, "class_id", None) is not None else []
        output_detection_count = len(xyxy_rows)
        self._metrics["tracker_output_rows"] += output_detection_count
        if output_detection_count > 0:
            self._metrics["frames_with_tracker_output"] += 1
        self._metrics["detections_without_tracker_id"] += max(0, debug_snapshot.input_detection_count - len(tracker_ids))
        if self._supervision_debug_frames_remaining > 0:
            LOGGER.debug(
                "Supervision ByteTrack update camera_code=%s frame_number=%s input_detection_count=%s input_boxes=%s input_confidences=%s input_class_ids=%s output_detection_count=%s output_tracker_ids=%s",
                packet.camera_code,
                packet.frame_number,
                debug_snapshot.input_detection_count,
                debug_snapshot.input_boxes,
                debug_snapshot.input_confidences,
                debug_snapshot.input_class_ids,
                output_detection_count,
                tracker_ids,
            )
            self._supervision_debug_frames_remaining -= 1
        for index, tracker_id in enumerate(tracker_ids):
            if tracker_id is None:
                continue
            x1, y1, x2, y2 = [float(value) for value in xyxy_rows[index]]
            source_detection = packet.detections[index] if index < len(packet.detections) else None
            if source_detection is not None:
                self._labels_by_track_id[int(tracker_id)] = (source_detection.class_name, source_detection.confidence)
                class_name = source_detection.class_name
                confidence = source_detection.confidence
            else:
                class_id = int(class_rows[index]) if index < len(class_rows) else 0
                class_name = self._labels_by_track_id.get(int(tracker_id), ("object", 0.0))[0]
                confidence = float(confidence_rows[index]) if index < len(confidence_rows) else 0.0
                if class_name == "object" and 0 <= class_id < len(packet.detections):
                    class_name = packet.detections[class_id].class_name
            raw_observations.append(
                TrackObservation(
                    camera_code=packet.camera_code,
                    local_track_id=int(tracker_id),
                    native_tracker_id=int(tracker_id),
                    frame_number=packet.frame_number,
                    video_time_seconds=packet.video_time_seconds,
                    camera_timestamp=packet.camera_timestamp,
                    class_name=class_name,
                    confidence=confidence,
                    bbox_xyxy=(x1, y1, x2, y2),
                )
            )
        return raw_observations

    def _update_with_row_tracker(self, tracker: Any, packet: DetectionPacket) -> list[TrackObservation]:
        native_output = tracker.update(_ResultsLike(packet), img=packet.frame)
        rows = native_output.tolist() if hasattr(native_output, "tolist") else list(native_output or [])
        raw_observations: list[TrackObservation] = []
        for row in rows:
            if len(row) < 5:
                continue
            x1, y1, x2, y2 = [float(value) for value in row[:4]]
            local_track_id = int(row[4])
            score = float(row[5]) if len(row) > 5 else 0.0
            detection_index = int(row[7]) if len(row) > 7 else -1
            if 0 <= detection_index < len(packet.detections):
                source_detection = packet.detections[detection_index]
                self._labels_by_track_id[local_track_id] = (source_detection.class_name, source_detection.confidence)
                class_name = source_detection.class_name
                confidence = source_detection.confidence
            else:
                class_name, remembered_confidence = self._labels_by_track_id.get(local_track_id, ("object", 0.0))
                confidence = max(remembered_confidence, score)
            raw_observations.append(
                TrackObservation(
                    camera_code=packet.camera_code,
                    local_track_id=local_track_id,
                    native_tracker_id=local_track_id,
                    frame_number=packet.frame_number,
                    video_time_seconds=packet.video_time_seconds,
                    camera_timestamp=packet.camera_timestamp,
                    class_name=class_name,
                    confidence=confidence,
                    bbox_xyxy=(x1, y1, x2, y2),
                )
            )
        return raw_observations
