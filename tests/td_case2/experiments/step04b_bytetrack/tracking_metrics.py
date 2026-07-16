from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any


def _safe_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "avg": 0.0}
    return {
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "avg": round(sum(values) / len(values), 6),
    }


def _bbox_center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _bbox_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _dominant_class(track: dict[str, Any]) -> str:
    counter = Counter(str(item.get("class_name", "")).lower() for item in track["detections"])
    return counter.most_common(1)[0][0] if counter else track["track_type"]


def _border_metrics(box: list[float], image_width: int, image_height: int) -> tuple[bool, float]:
    x1, y1, x2, y2 = box
    min_dimension = max(1.0, float(min(image_width, image_height)))
    nearest_border = min(x1, y1, max(0.0, image_width - x2), max(0.0, image_height - y2))
    normalized_margin = nearest_border / (0.05 * min_dimension)
    border_touch_ratio = round(max(0.0, min(1.0, 1.0 - normalized_margin)), 6)
    return border_touch_ratio > 0.0, border_touch_ratio


def _track_quality(track: dict[str, Any], min_track_length: int, image_diagonal: float) -> tuple[str, float]:
    detections = list(track["detections"])
    count = len(detections)
    if count == 1:
        return "single_frame", 0.1
    durations = float(detections[-1]["timestamp_seconds"]) - float(detections[0]["timestamp_seconds"])
    gaps = [
        float(detections[index]["timestamp_seconds"]) - float(detections[index - 1]["timestamp_seconds"])
        for index in range(1, len(detections))
    ]
    jumps = []
    for index in range(1, len(detections)):
        ax, ay = _bbox_center(detections[index - 1]["bbox_xyxy"])
        bx, by = _bbox_center(detections[index]["bbox_xyxy"])
        jumps.append((((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5) / image_diagonal if image_diagonal > 0 else 0.0)
    avg_conf = sum(float(item["confidence"]) for item in detections) / count
    if count < min_track_length:
        return "short", 0.2
    if max(gaps) > 2.0 if gaps else False:
        return "fragmented", 0.45
    if max(jumps) > 0.25 if jumps else False:
        return "fragmented", 0.45
    if count >= 4 and durations >= 1.0 and avg_conf >= 0.4:
        return "good", 0.9
    return "fragmented", 0.5


def build_step05_compatible_tracks(
    *,
    run_dir: Path,
    tracks: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    min_track_length: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert experimental tracks into the important schema Step 05 expects."""

    image_diagonal = (image_width**2 + image_height**2) ** 0.5
    payload_tracks: list[dict[str, Any]] = []
    quality_counts: Counter[str] = Counter()
    track_type_counts: Counter[str] = Counter()
    track_lengths: list[float] = []
    track_durations: list[float] = []

    for track in tracks:
        detections = sorted(track["detections"], key=lambda item: (float(item["timestamp_seconds"]), int(item["frame_idx"])))
        dominant_class_name = _dominant_class({"track_type": track["track_type"], "detections": detections})
        class_counts = Counter(str(item.get("class_name", "")).lower() for item in detections)
        class_consistency_ratio = round(class_counts[dominant_class_name] / len(detections), 6) if detections else 0.0
        quality_label, quality_score = _track_quality(track, min_track_length, image_diagonal)
        for detection in detections:
            border_touching, border_touch_ratio = _border_metrics(detection["bbox_xyxy"], image_width, image_height)
            detection["border_touching"] = border_touching
            detection["border_touch_ratio"] = border_touch_ratio
            bbox_area_score = min(1.0, float(detection["bbox_area_ratio"]) / 0.20)
            class_bonus = 1.0 if str(detection["class_name"]).lower() == dominant_class_name else 0.0
            not_border_bonus = 0.0 if border_touching else 1.0
            detection["best_frame_score"] = round(
                0.45 * float(detection["confidence"])
                + 0.35 * bbox_area_score
                + 0.10 * class_bonus
                + 0.10 * not_border_bonus,
                6,
            )
        best_detection = max(detections, key=lambda item: float(item.get("best_frame_score", 0.0)))
        duration_seconds = round(float(detections[-1]["timestamp_seconds"]) - float(detections[0]["timestamp_seconds"]), 6)
        avg_confidence = round(sum(float(item["confidence"]) for item in detections) / len(detections), 6)
        avg_area_ratio = round(sum(float(item["bbox_area_ratio"]) for item in detections) / len(detections), 6)
        usable_for_next_step = bool(
            track["track_type"] == "vehicle"
            and quality_label == "good"
            and len(detections) >= 4
            and avg_confidence >= 0.4
            and class_consistency_ratio >= 0.85
            and bool(best_detection.get("crop_exists"))
        )
        payload_tracks.append(
            {
                "track_id": track["track_id"],
                "track_type": track["track_type"],
                "dominant_class_name": dominant_class_name,
                "class_group": track["track_type"],
                "class_counts": dict(sorted(class_counts.items())),
                "class_consistency_ratio": class_consistency_ratio,
                "class_switch_count": 0,
                "class_switches": [],
                "start_timestamp_seconds": float(detections[0]["timestamp_seconds"]),
                "end_timestamp_seconds": float(detections[-1]["timestamp_seconds"]),
                "duration_seconds": duration_seconds,
                "first_timestamp_seconds": float(detections[0]["timestamp_seconds"]),
                "last_timestamp_seconds": float(detections[-1]["timestamp_seconds"]),
                "first_frame_id": detections[0]["frame_id"],
                "last_frame_id": detections[-1]["frame_id"],
                "detection_count": len(detections),
                "track_length": len(detections),
                "avg_confidence": avg_confidence,
                "max_confidence": round(max(float(item["confidence"]) for item in detections), 6),
                "avg_bbox_area_ratio": avg_area_ratio,
                "max_bbox_area_ratio": round(max(float(item["bbox_area_ratio"]) for item in detections), 6),
                "track_quality": quality_label,
                "quality_label": quality_label,
                "track_quality_score": quality_score,
                "best_detection_id": best_detection["detection_id"],
                "best_crop_path": best_detection["crop_path"],
                "usable_for_next_step": usable_for_next_step,
                "source_information": {
                    "source": track.get("source", "experimental"),
                    "merged_from_track_ids": track.get("merged_from_track_ids", [track["track_id"]]),
                },
                "detections": detections,
            }
        )
        quality_counts[quality_label] += 1
        track_type_counts[track["track_type"]] += 1
        track_lengths.append(float(len(detections)))
        track_durations.append(duration_seconds)

    payload_tracks.sort(key=lambda item: item["start_timestamp_seconds"])
    usable_tracks = [item["track_id"] for item in payload_tracks if item["usable_for_next_step"]]
    tracks_payload = {
        "status": "success",
        "input_yolo_detections_file": "experimental_dense_yolo_detections.json",
        "frames_processed": 0,
        "detections_considered": sum(len(track["detections"]) for track in payload_tracks),
        "detections_tracked": sum(len(track["detections"]) for track in payload_tracks),
        "tracks_created": len(payload_tracks),
        "same_frame_multi_assignment_prevented_count": 0,
        "usable_tracks_for_next_step": usable_tracks,
        "tracks": payload_tracks,
    }
    report_payload = {
        "status": "success",
        "frames_processed": 0,
        "tracks_created": len(payload_tracks),
        "track_type_counts": dict(sorted(track_type_counts.items())),
        "track_quality_counts": dict(sorted(quality_counts.items())),
        "track_length_stats": _safe_stats(track_lengths),
        "track_duration_stats": _safe_stats(track_durations),
        "usable_vehicle_tracks_for_ocr_color": sum(
            1 for item in payload_tracks if item["track_type"] == "vehicle" and item["usable_for_next_step"]
        ),
    }
    return tracks_payload, report_payload


def validate_step05_compatibility(tracks_payload: dict[str, Any]) -> dict[str, Any]:
    required_track_fields = {
        "track_id",
        "track_type",
        "dominant_class_name",
        "detection_count",
        "duration_seconds",
        "track_quality",
        "best_detection_id",
        "best_crop_path",
        "usable_for_next_step",
        "detections",
    }
    required_detection_fields = {
        "frame_id",
        "frame_idx",
        "timestamp_seconds",
        "detection_id",
        "class_name",
        "confidence",
        "bbox_xyxy",
        "bbox_area_ratio",
        "crop_path",
        "crop_exists",
    }
    missing: list[str] = []
    for track in list(tracks_payload.get("tracks", [])):
        missing_fields = sorted(required_track_fields - set(track.keys()))
        if missing_fields:
            missing.append(f"{track.get('track_id', 'unknown')}: missing track fields {missing_fields}")
        for detection in list(track.get("detections", [])):
            missing_detection_fields = sorted(required_detection_fields - set(detection.keys()))
            if missing_detection_fields:
                missing.append(
                    f"{track.get('track_id', 'unknown')}:{detection.get('detection_id', 'unknown')}: "
                    f"missing detection fields {missing_detection_fields}"
                )
    return {"compatible": not missing, "errors": missing}

