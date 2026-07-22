from __future__ import annotations

import cv2

from ..ingestion.frame_packet import FramePacket
from .tracking_models import TrackObservation


def annotate_tracking_frame(
    frame_packet: FramePacket,
    observations: list[TrackObservation],
    *,
    active_track_count: int,
) -> object:
    frame = frame_packet.frame.copy()
    for observation in observations:
        x1, y1, x2, y2 = (int(round(value)) for value in observation.bbox_xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        label = f"{observation.camera_code} | {observation.class_name} | ID {observation.local_track_id} | {observation.confidence:.2f}"
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2, cv2.LINE_AA)
    overlay_lines = [
        f"camera={frame_packet.camera_code}",
        f"frame={frame_packet.frame_number}",
        f"time={frame_packet.video_time_seconds:.2f}s",
        f"active_tracks={active_track_count}",
    ]
    if frame_packet.camera_timestamp is not None:
        overlay_lines.append(frame_packet.camera_timestamp.isoformat())
    y = 24
    for line in overlay_lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        y += 24
    return frame
