from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


def _install_lap_shim() -> None:
    """Provide the minimal lap API ByteTrack imports expect without installing packages."""

    if "lap" in sys.modules:
        return

    lap_module = types.ModuleType("lap")
    lap_module.__version__ = "0.5.12"

    def lapjv(cost_matrix: np.ndarray, extend_cost: bool = True, cost_limit: float = np.inf):
        from ultralytics.utils.ops import linear_sum_assignment

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


def load_bytetrack_types() -> tuple[type[Any], type[Any]]:
    """Load Ultralytics BYTETracker safely with a local lap shim."""

    _install_lap_shim()
    from ultralytics.trackers.byte_tracker import BYTETracker, TrackState  # type: ignore

    return BYTETracker, TrackState


class DetectionSet:
    """Tiny Results-like wrapper for feeding detections into Ultralytics BYTETracker."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = xyxy.astype(np.float32)
        self.conf = conf.astype(np.float32)
        self.cls = cls.astype(np.float32)

    @property
    def xywh(self) -> np.ndarray:
        xywh = self.xyxy.copy()
        xywh[:, 2] = xywh[:, 2] - xywh[:, 0]
        xywh[:, 3] = xywh[:, 3] - xywh[:, 1]
        xywh[:, 0] = xywh[:, 0] + (xywh[:, 2] / 2.0)
        xywh[:, 1] = xywh[:, 1] + (xywh[:, 3] / 2.0)
        return xywh

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])

    def __getitem__(self, mask: Any) -> "DetectionSet":
        return DetectionSet(self.xyxy[mask], self.conf[mask], self.cls[mask])


def _class_group(class_name: str) -> str | None:
    normalized = class_name.lower()
    if normalized == "person":
        return "person"
    if normalized in {"car", "motorcycle", "bus", "truck", "bicycle", "auto", "van", "vehicle"}:
        return "vehicle"
    return None


def run_bytetrack_tracking(
    *,
    frame_items: list[dict[str, Any]],
    detections_by_frame_id: dict[str, list[dict[str, Any]]],
    image_width: int,
    image_height: int,
    tracking_fps: float,
    track_buffer_seconds: float,
    high_confidence: float,
    low_confidence: float,
    match_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run isolated ByteTrack over dense detections while keeping person/vehicle groups separate."""

    BYTETracker, TrackState = load_bytetrack_types()

    class InstrumentedBYTETracker(BYTETracker):
        def __init__(self, args: Any):
            super().__init__(args)
            self.refind_count = 0

        def _apply_match(self, track: Any, det: Any, activated: list[Any], refind: list[Any]) -> None:
            if track.state != TrackState.Tracked:
                self.refind_count += 1
            super()._apply_match(track, det, activated, refind)

    tracker_args = SimpleNamespace(
        track_high_thresh=float(high_confidence),
        track_low_thresh=float(low_confidence),
        new_track_thresh=float(high_confidence),
        match_thresh=float(match_threshold),
        track_buffer=max(1, int(round(track_buffer_seconds * tracking_fps))),
        fuse_score=False,
        model="manual",
    )

    trackers = {
        "vehicle": InstrumentedBYTETracker(tracker_args),
        "person": InstrumentedBYTETracker(tracker_args),
    }
    track_histories: dict[str, dict[str, Any]] = {}

    for frame_item in frame_items:
        frame_id = str(frame_item["frame_id"])
        frame_detections = list(detections_by_frame_id.get(frame_id, []))
        grouped_detections = {"vehicle": [], "person": []}
        for detection in frame_detections:
            group = _class_group(str(detection.get("class_name", "")))
            if group in grouped_detections:
                grouped_detections[group].append(detection)

        for group_name, group_detections in grouped_detections.items():
            if group_detections:
                xyxy = np.asarray([item["bbox_xyxy"] for item in group_detections], dtype=np.float32)
                conf = np.asarray([item["confidence"] for item in group_detections], dtype=np.float32)
                cls = np.asarray([item["class_id"] for item in group_detections], dtype=np.float32)
            else:
                xyxy = np.zeros((0, 4), dtype=np.float32)
                conf = np.zeros((0,), dtype=np.float32)
                cls = np.zeros((0,), dtype=np.float32)

            results = DetectionSet(xyxy, conf, cls)
            tracked = trackers[group_name].update(results)
            for row in tracked.tolist():
                x1, y1, x2, y2, numeric_track_id, score, _cls_value, detection_index = row
                source_detection = group_detections[int(detection_index)]
                track_id = f"{group_name}_track_{int(numeric_track_id):04d}"
                history = track_histories.setdefault(
                    track_id,
                    {
                        "track_id": track_id,
                        "track_type": group_name,
                        "source": "bytetrack_raw",
                        "detections": [],
                    },
                )
                bbox_xyxy = [
                    max(0.0, min(float(x1), float(image_width))),
                    max(0.0, min(float(y1), float(image_height))),
                    max(0.0, min(float(x2), float(image_width))),
                    max(0.0, min(float(y2), float(image_height))),
                ]
                history["detections"].append(
                    {
                        "frame_id": source_detection["frame_id"],
                        "frame_idx": source_detection["frame_idx"],
                        "timestamp_seconds": source_detection["timestamp_seconds"],
                        "timestamp_text": source_detection["timestamp_text"],
                        "image_path": source_detection["image_path"],
                        "detection_id": source_detection["detection_id"],
                        "class_id": source_detection["class_id"],
                        "class_name": source_detection["class_name"],
                        "confidence": round(float(source_detection["confidence"]), 6),
                        "bbox_xyxy": [round(value, 3) for value in bbox_xyxy],
                        "bbox_area_ratio": round(float(source_detection["bbox_area_ratio"]), 6),
                        "crop_path": source_detection["crop_path"],
                        "crop_exists": bool(source_detection["crop_exists"]),
                        "match_score": round(float(score), 6),
                    }
                )

    raw_tracks = list(track_histories.values())
    raw_tracks.sort(key=lambda item: item["detections"][0]["timestamp_seconds"] if item["detections"] else 0.0)
    return raw_tracks, {
        "track_buffer_frames": tracker_args.track_buffer,
        "track_high_thresh": tracker_args.track_high_thresh,
        "track_low_thresh": tracker_args.track_low_thresh,
        "match_thresh": tracker_args.match_thresh,
        "vehicle_refind_count": trackers["vehicle"].refind_count,
        "person_refind_count": trackers["person"].refind_count,
    }

