from __future__ import annotations

from typing import Any

from .mot_backend import BackendDetection, BackendTrack
from .track_state import TrackObservation


def detections_from_rows(rows: list[dict[str, Any]]) -> list[BackendDetection]:
    return [
        BackendDetection(
            detection_id=str(item["detection_id"]),
            class_id=int(item["class_id"]),
            class_name=str(item["class_name"]),
            family=str(item.get("family", item.get("object_family", "other"))),
            confidence=float(item["confidence"]),
            bbox_xyxy=[float(value) for value in item["bbox_xyxy"]],
        )
        for item in rows
    ]


def observation_from_backend_track(
    *,
    backend_track: BackendTrack,
    frame_record: dict[str, Any],
    delta_seconds: float,
    bbox_source: str,
    lifecycle_state: str,
    observation_validity: str = "valid",
) -> TrackObservation:
    return TrackObservation(
        track_id=backend_track.track_id,
        object_family=backend_track.family,
        class_name=backend_track.class_name,
        timestamp_seconds=float(frame_record["timestamp_seconds"]),
        source_frame_index=int(frame_record["source_frame_index"]),
        processed_frame_index=int(frame_record["processed_frame_index"]),
        bbox_xyxy=[float(value) for value in backend_track.bbox_xyxy],
        bbox_source=bbox_source,
        detector_confidence=backend_track.matched_detection_confidence,
        detector_detection_id=backend_track.matched_detection_id,
        lifecycle_state=lifecycle_state,
        confirmed=bool(backend_track.confirmed),
        age_frames=int(backend_track.age_frames),
        hits=int(backend_track.hits),
        time_since_update_seconds=round(float(backend_track.time_since_update_frames) * max(delta_seconds, 0.0), 6),
        track_backend_state=str(backend_track.backend_state),
        association_cost=backend_track.association_cost,
        observation_validity=observation_validity,
    )
