from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2


def _read_frame(video_path: Path, timestamp_seconds: float) -> tuple[Any, float]:
    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_index = max(0, int(round(timestamp_seconds * fps))) if fps > 0 else 0
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    actual_timestamp = frame_index / fps if fps > 0 else timestamp_seconds
    capture.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame at {timestamp_seconds:.3f}s from {video_path}")
    return frame, actual_timestamp


def generate_visual_review_timestamps(source_duration_seconds: float, count: int = 20) -> list[float]:
    if source_duration_seconds <= 0:
        return [0.0] * count
    if count <= 1:
        return [round(min(source_duration_seconds, source_duration_seconds / 2.0), 3)]
    step = source_duration_seconds / float(count + 1)
    timestamps = []
    for index in range(count):
        timestamps.append(round(min(source_duration_seconds, step * float(index + 1)), 3))
    return timestamps


def _draw_boxes(frame, boxes: list[dict[str, Any]], color: tuple[int, int, int]) -> Any:
    output = frame.copy()
    for item in boxes:
        bbox = item.get("bbox_xyxy") or item.get("crop_bbox_xyxy")
        if not bbox or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = str(item.get("class_name") or item.get("final_class") or item.get("track_id") or "object")
        cv2.putText(output, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return output


def _nearest_td_case2_frame(td_case2_metrics: dict[str, Any], timestamp_seconds: float) -> dict[str, Any]:
    frames = list(td_case2_metrics["best_frames"].get("tracks", []))
    nearest: dict[str, Any] = {"timestamp_seconds": timestamp_seconds, "selected_detections": []}
    best_delta = None
    for track in frames:
        for detection in track.get("selected_detections", []):
            detection_timestamp = float(detection.get("timestamp_seconds", 0.0) or 0.0)
            delta = abs(detection_timestamp - timestamp_seconds)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                nearest = {
                    "timestamp_seconds": detection_timestamp,
                    "selected_detections": [detection],
                    "track_id": track.get("track_id"),
                }
    return nearest


def _nearest_hybrid_frame(hybrid_metrics: dict[str, Any], timestamp_seconds: float) -> dict[str, Any]:
    frames = list(hybrid_metrics["frame_metrics"].get("frames", []))
    if not frames:
        return {"timestamp_seconds": timestamp_seconds, "tracks": []}
    best = min(frames, key=lambda item: abs(float(item.get("timestamp_seconds", 0.0) or 0.0) - timestamp_seconds))
    return best


def build_visual_review_manifest(
    *,
    comparison_dir: Path,
    video_path: Path,
    td_case2_metrics: dict[str, Any],
    hybrid_metrics: dict[str, Any],
    count: int = 20,
) -> dict[str, Any]:
    visual_root = comparison_dir / "visual_review"
    visual_root.mkdir(parents=True, exist_ok=True)
    source_duration = float(td_case2_metrics.get("source_duration_seconds") or hybrid_metrics["tracking_report"]["video_metadata"]["duration_seconds"] or 0.0)
    timestamps = generate_visual_review_timestamps(source_duration, count=count)
    cases: list[dict[str, Any]] = []
    for index, timestamp in enumerate(timestamps, start=1):
        case_dir = visual_root / f"case_{index:03d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        td_case = _nearest_td_case2_frame(td_case2_metrics, timestamp)
        hy_case = _nearest_hybrid_frame(hybrid_metrics, timestamp)
        td_frame, td_actual = _read_frame(video_path, float(td_case.get("timestamp_seconds", timestamp)))
        hy_frame, hy_actual = _read_frame(video_path, float(hy_case.get("timestamp_seconds", timestamp)))
        td_annotated = _draw_boxes(td_frame, list(td_case.get("selected_detections", [])), (0, 255, 0))
        hy_annotated = _draw_boxes(hy_frame, list(hy_case.get("tracks", [])), (0, 165, 255))
        td_full_frame_path = case_dir / "td_case2_full_frame.jpg"
        hy_full_frame_path = case_dir / "hybrid_full_frame.jpg"
        td_annotated_path = case_dir / "td_case2_annotated.jpg"
        hy_annotated_path = case_dir / "hybrid_annotated.jpg"
        cv2.imwrite(str(td_full_frame_path), td_frame)
        cv2.imwrite(str(hy_full_frame_path), hy_frame)
        cv2.imwrite(str(td_annotated_path), td_annotated)
        cv2.imwrite(str(hy_annotated_path), hy_annotated)
        td_metadata = {
            "requested_timestamp_seconds": timestamp,
            "actual_timestamp_seconds": td_actual,
            "time_delta_seconds": round(td_actual - timestamp, 6),
            "track_id": td_case.get("track_id"),
            "selected_detections": td_case.get("selected_detections", []),
        }
        hy_metadata = {
            "requested_timestamp_seconds": timestamp,
            "actual_timestamp_seconds": hy_actual,
            "time_delta_seconds": round(hy_actual - timestamp, 6),
            "frame_payload": hy_case,
        }
        (case_dir / "td_case2_metadata.json").write_text(json.dumps(td_metadata, indent=2), encoding="utf-8")
        (case_dir / "hybrid_metadata.json").write_text(json.dumps(hy_metadata, indent=2), encoding="utf-8")
        (case_dir / "comparison_notes_template.json").write_text(
            json.dumps(
                {
                    "case_id": f"case_{index:03d}",
                    "review_notes": "",
                    "observed_difference": "",
                    "follow_up_required": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        cases.append(
            {
                "case_id": f"case_{index:03d}",
                "requested_timestamp_seconds": timestamp,
                "case_dir": str(case_dir),
            }
        )
    manifest = {
        "status": "success",
        "visual_review_root": str(visual_root),
        "case_count": len(cases),
        "cases": cases,
    }
    (comparison_dir / "09_visual_review_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
