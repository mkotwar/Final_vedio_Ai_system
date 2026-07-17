from __future__ import annotations

from collections import Counter
from typing import Any


def _bbox_center(bbox_xyxy: list[float]) -> tuple[float, float]:
    return ((float(bbox_xyxy[0]) + float(bbox_xyxy[2])) / 2.0, (float(bbox_xyxy[1]) + float(bbox_xyxy[3])) / 2.0)


def _bbox_area(bbox_xyxy: list[float]) -> float:
    return max(0.0, float(bbox_xyxy[2]) - float(bbox_xyxy[0])) * max(0.0, float(bbox_xyxy[3]) - float(bbox_xyxy[1]))


def sanitize_tracks(
    track_rows: list[dict[str, Any]],
    *,
    frame_width: int,
    frame_height: int,
    maximum_active_detector_gap_seconds: float,
    maximum_visual_bridge_seconds: float,
    frozen_window_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    sanitized: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    flag_counts: Counter[str] = Counter()
    for track in track_rows:
        valid_timeline: list[dict[str, Any]] = []
        track_flags: set[str] = set()
        previous: dict[str, Any] | None = None
        repeated_seconds = 0.0
        for observation in list(track.get("trajectory", [])):
            flags = set(list(observation.get("integrity_flags", [])))
            if previous is not None:
                delta_t = max(0.0, float(observation["timestamp_seconds"]) - float(previous["timestamp_seconds"]))
                center_left = _bbox_center(list(previous["bbox_xyxy"]))
                center_right = _bbox_center(list(observation["bbox_xyxy"]))
                center_jump = ((center_left[0] - center_right[0]) ** 2 + (center_left[1] - center_right[1]) ** 2) ** 0.5
                if center_jump > max(frame_width, frame_height) * 0.35:
                    flags.add("impossible_center_jump")
                area_left = max(_bbox_area(list(previous["bbox_xyxy"])), 1.0)
                area_right = max(_bbox_area(list(observation["bbox_xyxy"])), 1.0)
                area_ratio = area_right / area_left
                if area_ratio < 0.25 or area_ratio > 4.0:
                    flags.add("abnormal_area_change")
                if all(abs(float(left) - float(right)) <= 1.0 for left, right in zip(previous["bbox_xyxy"], observation["bbox_xyxy"])):
                    repeated_seconds += delta_t
                    if repeated_seconds >= frozen_window_seconds:
                        flags.add("frozen_bbox")
                else:
                    repeated_seconds = 0.0
            if str(observation["bbox_source"]).startswith("visual_bridge") and float(observation["time_since_update_seconds"]) > maximum_visual_bridge_seconds:
                flags.add("visual_bridge_too_long")
            if float(observation["time_since_update_seconds"]) > maximum_active_detector_gap_seconds and str(observation["bbox_source"]) != "yolo":
                flags.add("long_detector_gap")
            x1, y1, x2, y2 = [float(value) for value in observation["bbox_xyxy"]]
            if x1 <= 1.0 or y1 <= 1.0 or x2 >= frame_width - 1.0 or y2 >= frame_height - 1.0:
                flags.add("boundary_stuck")
            event_flags = sorted(flags)
            if event_flags:
                events.append(
                    {
                        "track_id": track["track_id"],
                        "timestamp_seconds": round(float(observation["timestamp_seconds"]), 6),
                        "flags": event_flags,
                    }
                )
            flag_counts.update(event_flags)
            validity = "invalid" if event_flags else str(observation.get("observation_validity", "valid"))
            valid_copy = {**observation, "integrity_flags": event_flags, "observation_validity": validity}
            if validity != "invalid":
                valid_timeline.append(valid_copy)
            previous = observation
            track_flags.update(event_flags)
        integrity_status = "usable"
        if not valid_timeline:
            integrity_status = "invalid"
        elif {"frozen_bbox", "boundary_stuck"} & track_flags:
            integrity_status = "manual_review"
        sanitized.append(
            {
                **track,
                "sanitized_valid_timeline": valid_timeline,
                "valid_observation_count": len(valid_timeline),
                "invalid_observation_count": max(0, len(list(track.get("trajectory", []))) - len(valid_timeline)),
                "track_integrity_status": integrity_status,
                "integrity_flags": sorted(track_flags),
                "sanitized_start_timestamp_seconds": round(float(valid_timeline[0]["timestamp_seconds"]) if valid_timeline else float(track["start_timestamp_seconds"]), 6),
                "sanitized_end_timestamp_seconds": round(float(valid_timeline[-1]["timestamp_seconds"]) if valid_timeline else float(track["end_timestamp_seconds"]), 6),
                "sanitized_duration_seconds": round((float(valid_timeline[-1]["timestamp_seconds"]) - float(valid_timeline[0]["timestamp_seconds"])) if len(valid_timeline) >= 2 else 0.0, 6),
            }
        )
    report = {
        "status": "success",
        "track_count": len(track_rows),
        "integrity_flag_counts": dict(sorted(flag_counts.items())),
        "frozen_tracks": len([item for item in sanitized if "frozen_bbox" in list(item.get("integrity_flags", []))]),
        "boundary_stuck_tracks": len([item for item in sanitized if "boundary_stuck" in list(item.get("integrity_flags", []))]),
        "invalid_tracks": len([item for item in sanitized if str(item["track_integrity_status"]) == "invalid"]),
        "manual_review_tracks": len([item for item in sanitized if str(item["track_integrity_status"]) == "manual_review"]),
    }
    return sanitized, report, events

