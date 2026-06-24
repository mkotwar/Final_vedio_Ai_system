from typing import Any, Dict, List, Tuple

from app.core.config import settings


class EventCandidateSelector:
    """Selects representative frames for VLM based on motion, detections, and tracks."""

    DYNAMIC_CLASSES = {
        "person",
        "car",
        "truck",
        "bus",
        "motorcycle",
        "bicycle",
        "backpack",
        "handbag",
        "suitcase",
        "cell phone",
        "book",
        "bottle",
        "cup",
        "sports ball",
    }

    @staticmethod
    def _timestamp_in_windows(
        timestamp_seconds: float,
        windows: List[Tuple[float, float]],
        context_seconds: float = 0.0,
    ) -> bool:
        return any(
            (start - context_seconds) <= timestamp_seconds <= (end + context_seconds)
            for start, end in windows
        )

    @classmethod
    def _dynamic_detections(cls, detection: Any) -> List[Any]:
        if not detection:
            return []
        return [
            det for det in detection.detections
            if str(det.class_name).lower() in cls.DYNAMIC_CLASSES
        ]

    @classmethod
    def _window_edge_flags(
        cls,
        timestamp_seconds: float,
        windows: List[Tuple[float, float]],
        context_seconds: float,
    ) -> Tuple[bool, bool, bool]:
        in_motion_window = False
        near_window_edge = False
        in_context_only = False

        for start, end in windows:
            if start <= timestamp_seconds <= end:
                in_motion_window = True
                if timestamp_seconds <= (start + context_seconds) or timestamp_seconds >= (end - context_seconds):
                    near_window_edge = True
                break
            if (start - context_seconds) <= timestamp_seconds <= (end + context_seconds):
                in_context_only = True

        return in_motion_window, near_window_edge, in_context_only

    @classmethod
    def select(
        cls,
        extracted_tuples: List[Tuple[str, str, float, Any]],
        frame_detections: List[Any],
        tracking_map: Dict[str, Dict[str, Any]],
        motion_windows: List[Tuple[float, float]],
    ) -> Dict[str, Dict[str, Any]]:
        selection: Dict[str, Dict[str, Any]] = {}
        detection_map = {item.frame_id: item for item in frame_detections}
        previous_dynamic_counts: Dict[str, int] = {}
        previous_dynamic_track_ids: List[int] = []
        context_seconds = max(0.0, float(settings.VLM_EVENT_CONTEXT_SECONDS))

        for frame_id, _video_id, timestamp_seconds, _path in extracted_tuples:
            reasons: List[str] = []
            signal_reasons: List[str] = []
            detection = detection_map.get(frame_id)
            tracking = tracking_map.get(frame_id, {})
            dynamic_detections = cls._dynamic_detections(detection)
            dynamic_counts: Dict[str, int] = {}
            dynamic_track_ids: List[int] = []

            for entity in tracking.get("tracked_entities", []):
                class_name = str(entity.get("class_name", "")).lower()
                if class_name in cls.DYNAMIC_CLASSES:
                    dynamic_track_ids.append(entity.get("track_id"))
                    dynamic_counts[class_name] = dynamic_counts.get(class_name, 0) + 1

            in_motion_window, near_window_edge, in_context_only = cls._window_edge_flags(
                timestamp_seconds,
                motion_windows,
                context_seconds,
            )

            if in_motion_window:
                reasons.append("motion_window")
            if in_context_only:
                reasons.append("event_context")
            if near_window_edge:
                reasons.append("window_edge")

            detection_count = len(dynamic_detections)
            if detection_count > 0:
                reasons.append("dynamic_object_detected")
                signal_reasons.append("dynamic_object_detected")

            if tracking.get("new_track_count", 0) > 0 and dynamic_track_ids:
                reasons.append("new_track")
                signal_reasons.append("new_track")
            if tracking.get("ended_track_count", 0) > 0 and previous_dynamic_track_ids:
                reasons.append("track_ended")
                signal_reasons.append("track_ended")

            if dynamic_counts != previous_dynamic_counts and dynamic_counts:
                reasons.append("dynamic_count_changed")
                signal_reasons.append("dynamic_count_changed")
            if dynamic_track_ids != previous_dynamic_track_ids and dynamic_track_ids:
                reasons.append("dynamic_track_set_changed")
                signal_reasons.append("dynamic_track_set_changed")

            selected = bool(signal_reasons or near_window_edge or in_context_only)

            selection[frame_id] = {
                "selected": selected,
                "candidate_reasons": list(dict.fromkeys(reasons)),
                "detection_count": detection_count,
                "detected_objects": [
                    {
                        "class_name": det.class_name,
                        "confidence": det.confidence,
                        "bbox": det.bbox,
                    }
                    for det in dynamic_detections
                ],
                "tracked_entities": [
                    entity for entity in tracking.get("tracked_entities", [])
                    if str(entity.get("class_name", "")).lower() in cls.DYNAMIC_CLASSES
                ],
                "track_ids": dynamic_track_ids,
                "object_counts": dynamic_counts,
            }

            previous_dynamic_counts = dynamic_counts
            previous_dynamic_track_ids = dynamic_track_ids

        return selection
