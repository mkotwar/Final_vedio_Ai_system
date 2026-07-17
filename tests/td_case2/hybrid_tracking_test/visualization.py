from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2


STATUS_COLORS = {
    "tentative": (60, 180, 255),
    "confirmed": (50, 220, 50),
    "propagated": (255, 200, 60),
    "propagated_unconfirmed": (255, 150, 60),
    "reactivated": (80, 255, 220),
    "temporarily_lost": (60, 60, 255),
    "lost": (60, 60, 255),
    "completed": (120, 120, 120),
    "removed": (120, 120, 120),
}


def draw_frame_overlay(
    *,
    frame,
    frame_payload: dict[str, Any],
) -> Any:
    rendered = frame.copy()
    for track in list(frame_payload.get("tracks", [])):
        x1, y1, x2, y2 = [int(round(float(value))) for value in track["bbox_xyxy"]]
        color = STATUS_COLORS.get(str(track.get("status", "")), (255, 255, 255))
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
        status = str(track.get("status", ""))
        if status == "reactivated":
            label = f"REACTIVATED ID {track['track_id']} | {track['class_name']}"
        else:
            label = f"ID {track['track_id']} | {track['class_name']} | {track['bbox_source'].upper()} | {status}"
        cv2.putText(rendered, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)
    overlay_lines = [
        f"source_frame={frame_payload['source_frame_index']}",
        f"timestamp={float(frame_payload['timestamp_seconds']):.3f}s",
        f"processing_fps={float(frame_payload['processing_fps']):.2f}",
        f"active_tracks={int(frame_payload.get('active_track_count', 0))}",
        f"yolo_executed={'yes' if frame_payload.get('yolo_executed') else 'no'}",
        f"trigger_reasons={','.join(frame_payload.get('yolo_trigger_reasons', [])) or '-'}",
        f"cumulative_yolo_calls={int(frame_payload.get('cumulative_yolo_calls', 0))}",
        f"cumulative_kcf_updates={int(frame_payload.get('cumulative_kcf_updates', 0))}",
    ]
    for index, line in enumerate(overlay_lines):
        cv2.putText(rendered, line, (16, 28 + (index * 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return rendered


def build_annotated_video(
    *,
    video_path: Path,
    frame_payloads_by_source_index: dict[int, dict[str, Any]],
    output_path: Path,
    source_fps: float,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video for annotation: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, source_fps), (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Failed to create annotated video writer: {output_path}")
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        payload = frame_payloads_by_source_index.get(frame_index)
        if payload is not None:
            writer.write(draw_frame_overlay(frame=frame, frame_payload=payload))
        frame_index += 1
    writer.release()
    capture.release()
