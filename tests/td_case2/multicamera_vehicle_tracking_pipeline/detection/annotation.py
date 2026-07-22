from __future__ import annotations

from datetime import datetime

import cv2

from ..ingestion.frame_packet import FramePacket
from .detection_models import DetectionPacket


def annotate_detection_frame(frame_packet: FramePacket, detection_packet: DetectionPacket) -> object:
    frame = frame_packet.frame.copy()
    for detection in detection_packet.detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox_xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    overlay_lines = [
        f"camera={frame_packet.camera_code}",
        f"frame={frame_packet.frame_number}",
        f"time={frame_packet.video_time_seconds:.2f}s",
        f"detections={len(detection_packet.detections)}",
    ]
    if frame_packet.camera_timestamp is not None:
        overlay_lines.append(frame_packet.camera_timestamp.isoformat())
    y = 24
    for line in overlay_lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        y += 24
    return frame
