from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from .report_writer import write_json
from .video_frame_stream import stream_processed_frames


def select_visual_timestamps(*, bytetrack_events: list[dict[str, Any]], fallback_duration_seconds: float) -> list[float]:
    interesting = [
        float(item["timestamp_seconds"])
        for item in bytetrack_events
        if item.get("new_track_ids") or item.get("reactivated_track_ids") or item.get("lost_track_ids")
    ]
    if len(interesting) >= 20:
        values = sorted(dict.fromkeys(round(item, 3) for item in interesting))
        stride = max(1, len(values) // 20)
        return values[::stride][:20]
    evenly_spaced = [round((fallback_duration_seconds * index) / 19.0, 3) for index in range(20)] if fallback_duration_seconds > 0 else [0.0]
    merged = sorted(dict.fromkeys([round(item, 3) for item in interesting] + evenly_spaced))
    if len(merged) < 20:
        merged.extend([merged[-1]] * (20 - len(merged)))
    return merged[:20]


def save_visual_review_cases(
    *,
    video_path: Path,
    processing_fps: float,
    output_dir: Path,
    timestamps: list[float],
    frames_by_backend: dict[str, dict[int, Any]],
) -> dict[str, Any]:
    review_root = output_dir / "visual_review"
    review_root.mkdir(parents=True, exist_ok=True)
    timestamp_to_case = {index + 1: timestamp for index, timestamp in enumerate(timestamps)}
    _, _, _, iterator = stream_processed_frames(video_path=video_path, processing_fps=processing_fps, debug_frames_dir=None)
    manifest_cases: list[dict[str, Any]] = []
    for frame_record, frame in iterator:
        timestamp_seconds = float(frame_record.timestamp_seconds)
        processed_frame_index = int(frame_record.processed_frame_index)
        for case_index, target_timestamp in list(timestamp_to_case.items()):
            if abs(timestamp_seconds - target_timestamp) > 0.051:
                continue
            case_dir = review_root / f"case_{case_index:03d}"
            case_dir.mkdir(parents=True, exist_ok=True)
            source_path = case_dir / "source_frame.jpg"
            cv2.imwrite(str(source_path), frame)
            metadata = {
                "case_id": f"case_{case_index:03d}",
                "processed_frame_index": processed_frame_index,
                "timestamp_seconds": timestamp_seconds,
                "review_required": True,
            }
            for backend_name, images_by_index in frames_by_backend.items():
                annotated = images_by_index.get(processed_frame_index)
                if annotated is None:
                    continue
                cv2.imwrite(str(case_dir / f"{backend_name}_annotated.jpg"), annotated)
            write_json(case_dir / "metadata.json", metadata)
            write_json(
                case_dir / "review_template.json",
                {
                    "status": "review_required",
                    "questions": [
                        "Did any backend keep the same object identity through overlap or occlusion more cleanly?",
                        "Did any backend create a likely fragment or spurious new ID here?",
                    ],
                },
            )
            manifest_cases.append(metadata)
            del timestamp_to_case[case_index]
        if not timestamp_to_case:
            break
    payload = {"status": "success", "case_count": len(manifest_cases), "cases": manifest_cases}
    write_json(output_dir / "07_visual_review_manifest.json", payload)
    return payload
