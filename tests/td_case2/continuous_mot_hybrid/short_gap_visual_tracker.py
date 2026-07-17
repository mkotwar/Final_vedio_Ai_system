from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2


def _create_tracker(name: str):
    normalized = name.lower()
    if normalized == "kcf" and hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    raise RuntimeError("OpenCV visual tracker backend is not available.")


def _bbox_center(bbox_xyxy: list[float]) -> tuple[float, float]:
    return ((float(bbox_xyxy[0]) + float(bbox_xyxy[2])) / 2.0, (float(bbox_xyxy[1]) + float(bbox_xyxy[3])) / 2.0)


def _bbox_area(bbox_xyxy: list[float]) -> float:
    return max(0.0, float(bbox_xyxy[2]) - float(bbox_xyxy[0])) * max(0.0, float(bbox_xyxy[3]) - float(bbox_xyxy[1]))


@dataclass
class VisualBridgeSession:
    track_id: str
    tracker_name: str
    started_at_seconds: float
    last_valid_bbox_xyxy: list[float]
    last_timestamp_seconds: float
    tracker: Any
    updates: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)


class ShortGapVisualTrackerManager:
    def __init__(self, *, tracker_name: str, maximum_bridge_seconds: float, frame_width: int, frame_height: int):
        self.tracker_name = tracker_name
        self.maximum_bridge_seconds = maximum_bridge_seconds
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.sessions: dict[str, VisualBridgeSession] = {}
        self.events: list[dict[str, Any]] = []
        self.failures: list[dict[str, Any]] = []

    def start_or_refresh(self, *, track_id: str, frame: Any, bbox_xyxy: list[float], timestamp_seconds: float) -> None:
        x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
        tracker = _create_tracker(self.tracker_name)
        bounding_box = (
            int(round(x1)),
            int(round(y1)),
            int(round(max(1.0, x2 - x1))),
            int(round(max(1.0, y2 - y1))),
        )
        try:
            tracker.init(frame, bounding_box)
        except Exception:
            self.failures.append(
                {
                    "track_id": track_id,
                    "timestamp_seconds": round(timestamp_seconds, 6),
                    "reason": "tracker_init_failure",
                    "bbox_xyxy": [round(float(value), 3) for value in bbox_xyxy],
                }
            )
            self.sessions.pop(track_id, None)
            return
        self.sessions[track_id] = VisualBridgeSession(
            track_id=track_id,
            tracker_name=self.tracker_name,
            started_at_seconds=timestamp_seconds,
            last_valid_bbox_xyxy=[x1, y1, x2, y2],
            last_timestamp_seconds=timestamp_seconds,
            tracker=tracker,
        )

    def update(self, *, track_id: str, frame: Any, timestamp_seconds: float) -> dict[str, Any] | None:
        session = self.sessions.get(track_id)
        if session is None:
            return None
        bridge_duration = max(0.0, timestamp_seconds - session.started_at_seconds)
        if bridge_duration > self.maximum_bridge_seconds:
            return self._fail(session, timestamp_seconds, "bridge_duration_exceeded")
        ok, bbox = session.tracker.update(frame)
        if not ok:
            return self._fail(session, timestamp_seconds, "tracker_failure")
        x, y, w, h = [float(value) for value in bbox]
        bbox_xyxy = [x, y, x + w, y + h]
        flags = self._validate_bridge(session=session, bbox_xyxy=bbox_xyxy)
        validity = "visual_bridge_supported" if not flags else "visual_bridge_invalid"
        event = {
            "track_id": track_id,
            "timestamp_seconds": round(timestamp_seconds, 6),
            "bbox_xyxy": [round(float(value), 3) for value in bbox_xyxy],
            "bridge_duration_seconds": round(bridge_duration, 6),
            "bbox_source": validity,
            "flags": flags,
        }
        if flags:
            self.failures.append(event)
            session.failures.append(event)
            del self.sessions[track_id]
            return event
        session.last_valid_bbox_xyxy = bbox_xyxy
        session.last_timestamp_seconds = timestamp_seconds
        session.updates += 1
        self.events.append(event)
        return event

    def reconcile_with_detector(self, *, track_id: str, detector_bbox_xyxy: list[float], timestamp_seconds: float) -> dict[str, Any] | None:
        session = self.sessions.pop(track_id, None)
        if session is None:
            return None
        bridge_bbox = session.last_valid_bbox_xyxy
        center_left = _bbox_center(bridge_bbox)
        center_right = _bbox_center(detector_bbox_xyxy)
        bridge_area = max(_bbox_area(bridge_bbox), 1.0)
        detector_area = max(_bbox_area(detector_bbox_xyxy), 1.0)
        center_distance = (((center_left[0] - center_right[0]) ** 2 + (center_left[1] - center_right[1]) ** 2) ** 0.5) / max(bridge_area ** 0.5, 1.0)
        area_ratio = detector_area / bridge_area
        if center_distance > 1.25 or area_ratio < 0.4 or area_ratio > 2.5:
            event = {
                "track_id": track_id,
                "timestamp_seconds": round(timestamp_seconds, 6),
                "bridge_bbox_xyxy": [round(float(value), 3) for value in bridge_bbox],
                "detector_bbox_xyxy": [round(float(value), 3) for value in detector_bbox_xyxy],
                "reason": "detector_disagreement",
                "center_distance_ratio": round(center_distance, 6),
                "area_ratio": round(area_ratio, 6),
            }
            self.failures.append(event)
            return event
        return None

    def _validate_bridge(self, *, session: VisualBridgeSession, bbox_xyxy: list[float]) -> list[str]:
        flags: list[str] = []
        previous = session.last_valid_bbox_xyxy
        if all(abs(float(left) - float(right)) <= 1.0 for left, right in zip(previous, bbox_xyxy)):
            flags.append("frozen_bbox")
        x1, y1, x2, y2 = bbox_xyxy
        if x1 <= 1.0 or y1 <= 1.0 or x2 >= self.frame_width - 1.0 or y2 >= self.frame_height - 1.0:
            flags.append("boundary_stuck")
        previous_area = max(_bbox_area(previous), 1.0)
        current_area = max(_bbox_area(bbox_xyxy), 1.0)
        area_ratio = current_area / previous_area
        if area_ratio < 0.35 or area_ratio > 2.8:
            flags.append("abnormal_scale_change")
        center_left = _bbox_center(previous)
        center_right = _bbox_center(bbox_xyxy)
        jump_pixels = ((center_left[0] - center_right[0]) ** 2 + (center_left[1] - center_right[1]) ** 2) ** 0.5
        if jump_pixels > max(self.frame_width, self.frame_height) * 0.25:
            flags.append("impossible_center_jump")
        return flags

    def _fail(self, session: VisualBridgeSession, timestamp_seconds: float, reason: str) -> dict[str, Any]:
        event = {
            "track_id": session.track_id,
            "timestamp_seconds": round(timestamp_seconds, 6),
            "reason": reason,
            "bbox_xyxy": [round(float(value), 3) for value in session.last_valid_bbox_xyxy],
        }
        self.failures.append(event)
        session.failures.append(event)
        self.sessions.pop(session.track_id, None)
        return event
