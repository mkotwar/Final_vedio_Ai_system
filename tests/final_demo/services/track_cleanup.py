from __future__ import annotations

import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from tests.final_demo.services.chunk_planner import read_json
from tests.final_demo.services.tracker_adapter import (
    compute_hist_similarity,
    crop_bbox_from_image,
    to_absolute_repo_path,
)
from tests.final_demo.services.video_io import current_timestamp, write_json


ENV_FINAL_DEMO_CLEANUP_ENABLED = "FINAL_DEMO_CLEANUP_ENABLED"
ENV_FINAL_DEMO_CLEANUP_MAX_GAP_SECONDS = "FINAL_DEMO_CLEANUP_MAX_GAP_SECONDS"
ENV_FINAL_DEMO_CLEANUP_CENTER_DISTANCE_RATIO = "FINAL_DEMO_CLEANUP_CENTER_DISTANCE_RATIO"
ENV_FINAL_DEMO_CLEANUP_MIN_APPEARANCE_SIM = "FINAL_DEMO_CLEANUP_MIN_APPEARANCE_SIM"
ENV_FINAL_DEMO_CLEANUP_MIN_DURATION_SECONDS = "FINAL_DEMO_CLEANUP_MIN_DURATION_SECONDS"
ENV_FINAL_DEMO_CLEANUP_MIN_DETECTIONS = "FINAL_DEMO_CLEANUP_MIN_DETECTIONS"
ENV_FINAL_DEMO_CLEANUP_USE_APPEARANCE = "FINAL_DEMO_CLEANUP_USE_APPEARANCE"
ENV_FINAL_DEMO_CLEANUP_PROTECT_OVERLAP = "FINAL_DEMO_CLEANUP_PROTECT_OVERLAP"
ENV_FINAL_DEMO_CLEANUP_MERGE_SCORE_THRESHOLD = "FINAL_DEMO_CLEANUP_MERGE_SCORE_THRESHOLD"
ENV_FINAL_DEMO_CLEANUP_MERGE_DUPLICATES = "FINAL_DEMO_CLEANUP_MERGE_DUPLICATES"
ENV_FINAL_DEMO_CLEANUP_VEHICLE_NOISE_MAX_CONF = "FINAL_DEMO_CLEANUP_VEHICLE_NOISE_MAX_CONF"
ENV_FINAL_DEMO_CLEANUP_ALLOW_CROSS_CLASS_MERGE = "FINAL_DEMO_CLEANUP_ALLOW_CROSS_CLASS_MERGE"

DEFAULT_CLEANUP_ENABLED = True
DEFAULT_CLEANUP_MAX_GAP_SECONDS = 8.0
DEFAULT_CLEANUP_CENTER_DISTANCE_RATIO = 0.45
DEFAULT_CLEANUP_MIN_APPEARANCE_SIM = 0.45
DEFAULT_CLEANUP_MIN_DURATION_SECONDS = 1.0
DEFAULT_CLEANUP_MIN_DETECTIONS = 3
DEFAULT_CLEANUP_USE_APPEARANCE = True
DEFAULT_CLEANUP_PROTECT_OVERLAP = True
DEFAULT_CLEANUP_MERGE_SCORE_THRESHOLD = 0.55
DEFAULT_CLEANUP_MERGE_DUPLICATES = False
DEFAULT_CLEANUP_VEHICLE_NOISE_MAX_CONF = 0.50
DEFAULT_CLEANUP_ALLOW_CROSS_CLASS_MERGE = False
VEHICLE_CLEANUP_MAX_GAP_SECONDS = 6.0
VEHICLE_CLEANUP_CENTER_DISTANCE_RATIO = 0.40
VEHICLE_CLEANUP_MIN_APPEARANCE_SIM = 0.40
VEHICLE_CLEANUP_MIN_DURATION_SECONDS = 0.5
VEHICLE_CLEANUP_MIN_DETECTIONS = 2
VEHICLE_CLEANUP_MERGE_SCORE_THRESHOLD = 0.50
VEHICLE_LENIENT_FOCUS_PROFILES = {"vehicle_only", "traffic_road", "parking"}
VEHICLE_CLASSES = {"car", "bicycle", "motorcycle", "bus", "truck"}


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def read_float_env(env_name: str, default_value: float) -> float:
    raw_value = os.environ.get(env_name, str(default_value))
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {env_name} must be a valid number. Received: {raw_value!r}"
        ) from exc


def read_non_negative_float_env(env_name: str, default_value: float) -> float:
    value = read_float_env(env_name, default_value)
    if value < 0:
        raise ValueError(
            f"Environment variable {env_name} must be greater than or equal to 0. Received: {value}"
        )
    return value


def read_positive_int_env(env_name: str, default_value: int) -> int:
    raw_value = os.environ.get(env_name, str(default_value))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {env_name} must be a valid integer. Received: {raw_value!r}"
        ) from exc
    if value <= 0:
        raise ValueError(
            f"Environment variable {env_name} must be greater than 0. Received: {value}"
        )
    return value


def read_bool_env(env_name: str, default_value: bool) -> bool:
    raw_value = os.environ.get(env_name)
    if raw_value is None or raw_value == "":
        return default_value
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(
        f"Environment variable {env_name} must be boolean-like. Received: {raw_value!r}"
    )


def has_env_override(env_name: str) -> bool:
    raw_value = os.environ.get(env_name)
    return raw_value is not None and raw_value.strip() != ""


def build_cleanup_defaults(focus_payload: dict[str, Any] | None) -> dict[str, Any]:
    selected_focus_profile = (
        str(focus_payload.get("selected_focus_profile"))
        if isinstance(focus_payload, dict) and focus_payload.get("selected_focus_profile")
        else ""
    )
    explicit_env_override_used = any(
        has_env_override(env_name)
        for env_name in (
            ENV_FINAL_DEMO_CLEANUP_MAX_GAP_SECONDS,
            ENV_FINAL_DEMO_CLEANUP_CENTER_DISTANCE_RATIO,
            ENV_FINAL_DEMO_CLEANUP_MIN_APPEARANCE_SIM,
            ENV_FINAL_DEMO_CLEANUP_MIN_DURATION_SECONDS,
            ENV_FINAL_DEMO_CLEANUP_MIN_DETECTIONS,
            ENV_FINAL_DEMO_CLEANUP_MERGE_SCORE_THRESHOLD,
            ENV_FINAL_DEMO_CLEANUP_VEHICLE_NOISE_MAX_CONF,
        )
    )
    if explicit_env_override_used:
        profile = "custom_env_override"
    elif selected_focus_profile in VEHICLE_LENIENT_FOCUS_PROFILES:
        profile = "vehicle_lenient"
    else:
        profile = "person_security_default"

    if profile == "vehicle_lenient":
        return {
            "max_gap_seconds": VEHICLE_CLEANUP_MAX_GAP_SECONDS,
            "center_distance_ratio": VEHICLE_CLEANUP_CENTER_DISTANCE_RATIO,
            "min_appearance_sim": VEHICLE_CLEANUP_MIN_APPEARANCE_SIM,
            "min_duration_seconds": VEHICLE_CLEANUP_MIN_DURATION_SECONDS,
            "min_detections": VEHICLE_CLEANUP_MIN_DETECTIONS,
            "merge_score_threshold": VEHICLE_CLEANUP_MERGE_SCORE_THRESHOLD,
            "vehicle_noise_max_conf": DEFAULT_CLEANUP_VEHICLE_NOISE_MAX_CONF,
            "profile": profile,
        }
    return {
        "max_gap_seconds": DEFAULT_CLEANUP_MAX_GAP_SECONDS,
        "center_distance_ratio": DEFAULT_CLEANUP_CENTER_DISTANCE_RATIO,
        "min_appearance_sim": DEFAULT_CLEANUP_MIN_APPEARANCE_SIM,
        "min_duration_seconds": DEFAULT_CLEANUP_MIN_DURATION_SECONDS,
        "min_detections": DEFAULT_CLEANUP_MIN_DETECTIONS,
        "merge_score_threshold": DEFAULT_CLEANUP_MERGE_SCORE_THRESHOLD,
        "vehicle_noise_max_conf": DEFAULT_CLEANUP_VEHICLE_NOISE_MAX_CONF,
        "profile": profile,
    }


def read_cleanup_settings(defaults: dict[str, Any]) -> dict[str, Any]:
    return {
        "cleanup_enabled": read_bool_env(
            ENV_FINAL_DEMO_CLEANUP_ENABLED,
            DEFAULT_CLEANUP_ENABLED,
        ),
        "max_gap_seconds": round(
            read_non_negative_float_env(
                ENV_FINAL_DEMO_CLEANUP_MAX_GAP_SECONDS,
                float(defaults["max_gap_seconds"]),
            ),
            3,
        ),
        "center_distance_ratio": round(
            read_non_negative_float_env(
                ENV_FINAL_DEMO_CLEANUP_CENTER_DISTANCE_RATIO,
                float(defaults["center_distance_ratio"]),
            ),
            3,
        ),
        "min_appearance_sim": round(
            read_non_negative_float_env(
                ENV_FINAL_DEMO_CLEANUP_MIN_APPEARANCE_SIM,
                float(defaults["min_appearance_sim"]),
            ),
            3,
        ),
        "min_duration_seconds": round(
            read_non_negative_float_env(
                ENV_FINAL_DEMO_CLEANUP_MIN_DURATION_SECONDS,
                float(defaults["min_duration_seconds"]),
            ),
            3,
        ),
        "min_detections": read_positive_int_env(
            ENV_FINAL_DEMO_CLEANUP_MIN_DETECTIONS,
            int(defaults["min_detections"]),
        ),
        "use_appearance": read_bool_env(
            ENV_FINAL_DEMO_CLEANUP_USE_APPEARANCE,
            DEFAULT_CLEANUP_USE_APPEARANCE,
        ),
        "protect_overlap": read_bool_env(
            ENV_FINAL_DEMO_CLEANUP_PROTECT_OVERLAP,
            DEFAULT_CLEANUP_PROTECT_OVERLAP,
        ),
        "merge_score_threshold": round(
            read_non_negative_float_env(
                ENV_FINAL_DEMO_CLEANUP_MERGE_SCORE_THRESHOLD,
                float(defaults["merge_score_threshold"]),
            ),
            3,
        ),
        "merge_duplicates": read_bool_env(
            ENV_FINAL_DEMO_CLEANUP_MERGE_DUPLICATES,
            DEFAULT_CLEANUP_MERGE_DUPLICATES,
        ),
        "vehicle_noise_max_conf": round(
            read_non_negative_float_env(
                ENV_FINAL_DEMO_CLEANUP_VEHICLE_NOISE_MAX_CONF,
                float(defaults["vehicle_noise_max_conf"]),
            ),
            3,
        ),
        "allow_cross_class_merge": read_bool_env(
            ENV_FINAL_DEMO_CLEANUP_ALLOW_CROSS_CLASS_MERGE,
            DEFAULT_CLEANUP_ALLOW_CROSS_CLASS_MERGE,
        ),
        "cleanup_parameter_profile": str(defaults["profile"]),
    }


def choose_number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number):
        return default
    return number


def choose_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_bbox_area(bbox_xyxy: list[float]) -> float:
    return max(0.0, float(bbox_xyxy[2]) - float(bbox_xyxy[0])) * max(
        0.0, float(bbox_xyxy[3]) - float(bbox_xyxy[1])
    )


def get_bbox_center(bbox_xyxy: list[float]) -> list[float]:
    return [
        round((float(bbox_xyxy[0]) + float(bbox_xyxy[2])) / 2.0, 3),
        round((float(bbox_xyxy[1]) + float(bbox_xyxy[3])) / 2.0, 3),
    ]


def compute_center_distance(center_a: list[float], center_b: list[float]) -> float:
    return math.dist(center_a, center_b)


def normalize_bbox_sequence(raw_bbox_sequence: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_bbox_sequence, list):
        return []
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(raw_bbox_sequence):
        if isinstance(item, dict):
            bbox_xyxy = item.get("bbox_xyxy")
            if not isinstance(bbox_xyxy, list) or len(bbox_xyxy) < 4:
                continue
            normalized.append(
                {
                    "timestamp": round(choose_number(item.get("timestamp"), float(index)), 3),
                    "frame_id": str(item.get("frame_id") or ""),
                    "bbox_xyxy": [round(float(value), 3) for value in bbox_xyxy[:4]],
                    "confidence": round(choose_number(item.get("confidence"), 0.0), 4),
                }
            )
        elif isinstance(item, list) and len(item) >= 4:
            normalized.append(
                {
                    "timestamp": round(float(index), 3),
                    "frame_id": "",
                    "bbox_xyxy": [round(float(value), 3) for value in item[:4]],
                    "confidence": 0.0,
                }
            )
    normalized.sort(key=lambda entry: (float(entry["timestamp"]), str(entry["frame_id"])))
    return normalized


def normalize_center_sequence(
    raw_center_sequence: Any,
    bbox_sequence: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    reconstructed = False
    normalized: list[dict[str, Any]] = []
    if isinstance(raw_center_sequence, list):
        for index, item in enumerate(raw_center_sequence):
            if isinstance(item, dict):
                center = item.get("center")
                if isinstance(center, list) and len(center) >= 2:
                    normalized.append(
                        {
                            "timestamp": round(
                                choose_number(
                                    item.get("timestamp"),
                                    float(index),
                                ),
                                3,
                            ),
                            "center": [round(float(center[0]), 3), round(float(center[1]), 3)],
                        }
                    )
            elif isinstance(item, list) and len(item) >= 2:
                normalized.append(
                    {
                        "timestamp": round(float(index), 3),
                        "center": [round(float(item[0]), 3), round(float(item[1]), 3)],
                    }
                )

    if normalized:
        normalized.sort(key=lambda entry: float(entry["timestamp"]))
        return normalized, reconstructed

    reconstructed = True
    for bbox_item in bbox_sequence:
        normalized.append(
            {
                "timestamp": round(float(bbox_item["timestamp"]), 3),
                "center": get_bbox_center(list(bbox_item["bbox_xyxy"])),
            }
        )
    return normalized, reconstructed


def build_frame_lookup(
    frames_index_payload: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    if not isinstance(frames_index_payload, dict):
        return lookup
    for item in list(frames_index_payload.get("frames") or []):
        if not isinstance(item, dict):
            continue
        frame_id = str(item.get("frame_id") or "")
        if frame_id:
            lookup[frame_id] = item
    return lookup


def build_chunk_lookup(chunk_manifest_payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    if not isinstance(chunk_manifest_payload, dict):
        return lookup
    for item in list(chunk_manifest_payload.get("chunks") or []):
        if isinstance(item, dict) and item.get("chunk_id"):
            lookup[str(item["chunk_id"])] = item
    return lookup


def infer_track_source(
    tracks_payload: dict[str, Any],
    quality_payload: dict[str, Any],
    diagnostics_payload: dict[str, Any] | None,
    report_payload: dict[str, Any] | None,
) -> str:
    tracks = list(tracks_payload.get("tracks") or [])
    if any(
        isinstance(track, dict)
        and (
            len(list(track.get("source_fragment_track_ids") or [])) > 1
            or track.get("source_raw_tracker_ids") is not None
        )
        for track in tracks
    ):
        return "step5_final_merged_tracks"

    if isinstance(diagnostics_payload, dict):
        raw_count = len(list(diagnostics_payload.get("raw_tracks_before_merge") or []))
        final_count = len(list(diagnostics_payload.get("final_tracks_after_merge") or []))
        if final_count and len(tracks) == final_count:
            return "step5_final_merged_tracks"
        if raw_count and len(tracks) == raw_count:
            return "step5_raw_tracks"

    if isinstance(report_payload, dict):
        total_kept = choose_int(report_payload.get("total_tracks_kept"), -1)
        raw_count = choose_int(report_payload.get("raw_tracks_created_before_merge"), -1)
        if total_kept >= 0 and len(tracks) == total_kept and raw_count != total_kept:
            return "step5_final_merged_tracks"
        if raw_count >= 0 and len(tracks) == raw_count:
            return "step5_raw_tracks"

    if quality_payload.get("main_problem"):
        return "unknown_assumed_05_tracks"
    return "unknown_assumed_05_tracks"


def normalize_track(
    track: dict[str, Any],
    *,
    frame_lookup: dict[str, dict[str, Any]],
    chunk_lookup: dict[str, dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    bbox_sequence = normalize_bbox_sequence(track.get("bbox_sequence"))
    center_sequence, used_reconstruction = normalize_center_sequence(
        track.get("center_sequence"),
        bbox_sequence,
    )
    if used_reconstruction and bbox_sequence:
        warnings.append(
            "Center data was reconstructed from bbox_sequence for some tracks."
        )

    start_time = round(
        choose_number(
            track.get("start_time"),
            bbox_sequence[0]["timestamp"] if bbox_sequence else 0.0,
        ),
        3,
    )
    end_time = round(
        choose_number(
            track.get("end_time"),
            bbox_sequence[-1]["timestamp"] if bbox_sequence else start_time,
        ),
        3,
    )
    duration_seconds = round(
        choose_number(track.get("duration_seconds"), end_time - start_time),
        3,
    )
    detection_ids = [str(item) for item in list(track.get("detection_ids") or [])]
    detection_count = choose_int(track.get("detection_count"), 0)
    if detection_count <= 0:
        detection_count = len(detection_ids) or len(bbox_sequence) or len(center_sequence)

    chunk_id = str(track.get("chunk_id") or "unknown_chunk")
    best_image_path = track.get("best_image_path")
    if not best_image_path:
        candidate_frame_id = str(track.get("best_frame_id") or track.get("first_frame_id") or "")
        if candidate_frame_id and candidate_frame_id in frame_lookup:
            best_image_path = frame_lookup[candidate_frame_id].get("image_path")

    raw_track_id = track.get("raw_tracker_id")
    source_raw_tracker_ids = [
        str(item)
        for item in list(track.get("source_raw_tracker_ids") or ([] if raw_track_id is None else [raw_track_id]))
        if str(item)
    ]
    source_fragment_track_ids = [str(item) for item in list(track.get("source_fragment_track_ids") or []) if str(item)]
    chunk_info = chunk_lookup.get(chunk_id, {})

    return {
        "source_track_id": str(track.get("local_track_id") or track.get("track_id") or ""),
        "chunk_id": chunk_id,
        "chunk_index": choose_int(track.get("chunk_index"), choose_int(chunk_info.get("chunk_index"), 0)),
        "class_name": str(track.get("class_name") or "unknown").lower(),
        "dominant_class_name": str(track.get("dominant_class_name") or track.get("class_name") or "unknown").lower(),
        "dominant_class_confidence": round(choose_number(track.get("dominant_class_confidence"), 0.0), 4),
        "class_votes": {
            str(key).lower(): choose_int(value, 0)
            for key, value in dict(track.get("class_votes") or {}).items()
            if str(key).strip()
        },
        "class_confidence_sum": {
            str(key).lower(): round(choose_number(value, 0.0), 4)
            for key, value in dict(track.get("class_confidence_sum") or {}).items()
            if str(key).strip()
        },
        "class_history": [
            {
                "frame_id": str(item.get("frame_id") or ""),
                "timestamp": round(choose_number(item.get("timestamp"), 0.0), 3),
                "detection_id": str(item.get("detection_id") or ""),
                "class_name": str(item.get("class_name") or "").lower(),
                "class_id": choose_int(item.get("class_id"), -1),
                "confidence": round(choose_number(item.get("confidence"), 0.0), 4),
            }
            for item in list(track.get("class_history") or [])
            if isinstance(item, dict)
        ],
        "class_consistency_score": round(choose_number(track.get("class_consistency_score"), 1.0), 4),
        "class_conflict": bool(track.get("class_conflict", False)),
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": max(0.0, duration_seconds),
        "detection_count": max(0, detection_count),
        "average_confidence": round(choose_number(track.get("average_confidence"), 0.0), 4),
        "max_confidence": round(choose_number(track.get("max_confidence"), 0.0), 4),
        "first_frame_id": str(track.get("first_frame_id") or (bbox_sequence[0]["frame_id"] if bbox_sequence else "")),
        "last_frame_id": str(track.get("last_frame_id") or (bbox_sequence[-1]["frame_id"] if bbox_sequence else "")),
        "best_frame_id": str(track.get("best_frame_id") or track.get("first_frame_id") or ""),
        "best_image_path": str(best_image_path) if best_image_path else None,
        "thumbnail_path": track.get("thumbnail_path"),
        "bbox_sequence": bbox_sequence,
        "center_sequence": center_sequence,
        "detection_ids": detection_ids,
        "status": str(track.get("status") or "closed"),
        "is_cross_chunk_candidate": bool(track.get("is_cross_chunk_candidate", False)),
        "source_raw_tracker_ids": sorted(set(source_raw_tracker_ids)),
        "source_fragment_track_ids": sorted(set(source_fragment_track_ids)),
    }


def compute_frame_diagonal(track_a: dict[str, Any], track_b: dict[str, Any]) -> float | None:
    max_x = 0.0
    max_y = 0.0
    for track in (track_a, track_b):
        for item in list(track.get("bbox_sequence") or []):
            bbox_xyxy = item.get("bbox_xyxy")
            if isinstance(bbox_xyxy, list) and len(bbox_xyxy) >= 4:
                max_x = max(max_x, float(bbox_xyxy[2]))
                max_y = max(max_y, float(bbox_xyxy[3]))
    if max_x > 0 and max_y > 0:
        return math.hypot(max_x, max_y)
    return None


def get_last_bbox(track: dict[str, Any]) -> dict[str, Any] | None:
    sequence = list(track.get("bbox_sequence") or [])
    return sequence[-1] if sequence else None


def get_first_bbox(track: dict[str, Any]) -> dict[str, Any] | None:
    sequence = list(track.get("bbox_sequence") or [])
    return sequence[0] if sequence else None


def get_last_center(track: dict[str, Any]) -> dict[str, Any] | None:
    sequence = list(track.get("center_sequence") or [])
    return sequence[-1] if sequence else None


def get_first_center(track: dict[str, Any]) -> dict[str, Any] | None:
    sequence = list(track.get("center_sequence") or [])
    return sequence[0] if sequence else None


def compute_time_gap_score(time_gap: float, max_gap_seconds: float) -> float:
    if time_gap <= 1.0:
        return 1.0
    if max_gap_seconds <= 1.0:
        return 0.0
    remaining = max(0.0, max_gap_seconds - time_gap)
    return round(remaining / (max_gap_seconds - 1.0), 4)


def compute_center_distance_score(
    track_a: dict[str, Any],
    track_b: dict[str, Any],
    *,
    center_distance_ratio: float,
) -> tuple[float, float | None]:
    last_center = get_last_center(track_a)
    first_center = get_first_center(track_b)
    if not last_center or not first_center:
        return 0.5, None

    distance = compute_center_distance(
        list(last_center["center"]),
        list(first_center["center"]),
    )
    frame_diagonal = compute_frame_diagonal(track_a, track_b)
    if frame_diagonal and frame_diagonal > 0:
        max_allowed = frame_diagonal * center_distance_ratio
        score = 1.0 - min(distance / max_allowed, 1.0) if max_allowed > 0 else 0.0
        return round(max(0.0, score), 4), round(distance, 3)

    if all(0.0 <= float(value) <= 1.0 for value in list(last_center["center"]) + list(first_center["center"])):
        score = 1.0 - min(distance / center_distance_ratio, 1.0) if center_distance_ratio > 0 else 0.0
        return round(max(0.0, score), 4), round(distance, 3)
    return 0.5, round(distance, 3)


def compute_trajectory_score(
    track_a: dict[str, Any],
    track_b: dict[str, Any],
    *,
    time_gap: float,
    center_distance_score: float,
    center_distance_ratio: float,
) -> float:
    centers = list(track_a.get("center_sequence") or [])
    first_center = get_first_center(track_b)
    if len(centers) < 2 or not first_center:
        return center_distance_score

    prev_center = centers[-2]
    last_center = centers[-1]
    delta_time = float(last_center["timestamp"]) - float(prev_center["timestamp"])
    if delta_time <= 0:
        return center_distance_score

    velocity_x = (float(last_center["center"][0]) - float(prev_center["center"][0])) / delta_time
    velocity_y = (float(last_center["center"][1]) - float(prev_center["center"][1])) / delta_time
    predicted_center = [
        float(last_center["center"][0]) + velocity_x * time_gap,
        float(last_center["center"][1]) + velocity_y * time_gap,
    ]
    distance = compute_center_distance(predicted_center, list(first_center["center"]))
    frame_diagonal = compute_frame_diagonal(track_a, track_b)
    if frame_diagonal and frame_diagonal > 0:
        max_allowed = frame_diagonal * center_distance_ratio
        score = 1.0 - min(distance / max_allowed, 1.0) if max_allowed > 0 else 0.0
        return round(max(0.0, score), 4)
    return center_distance_score


def compute_bbox_size_score(track_a: dict[str, Any], track_b: dict[str, Any]) -> float:
    bbox_a = get_last_bbox(track_a)
    bbox_b = get_first_bbox(track_b)
    if not bbox_a or not bbox_b:
        return 0.5
    area_a = get_bbox_area(list(bbox_a["bbox_xyxy"]))
    area_b = get_bbox_area(list(bbox_b["bbox_xyxy"]))
    if area_a <= 0 or area_b <= 0:
        return 0.5
    return round(min(area_a, area_b) / max(area_a, area_b), 4)


def get_track_crop(
    track: dict[str, Any],
    *,
    use_last: bool,
    warnings: list[str],
) -> Any | None:
    bbox_item = get_last_bbox(track) if use_last else get_first_bbox(track)
    if bbox_item is None:
        return None

    image_path = track.get("best_image_path")
    if not image_path:
        warnings.append(
            "Appearance crop unavailable for some tracks; neutral appearance score was used."
        )
        return None

    try:
        absolute_path = to_absolute_repo_path(str(image_path))
    except Exception:
        warnings.append(
            "Appearance crop unavailable for some tracks; neutral appearance score was used."
        )
        return None
    return crop_bbox_from_image(absolute_path, list(bbox_item["bbox_xyxy"]))


def compute_appearance_score(
    track_a: dict[str, Any],
    track_b: dict[str, Any],
    *,
    settings: dict[str, Any],
    warnings: list[str],
) -> float:
    if not settings["use_appearance"]:
        return 0.5

    crop_a = get_track_crop(track_a, use_last=True, warnings=warnings)
    crop_b = get_track_crop(track_b, use_last=False, warnings=warnings)
    if crop_a is None or crop_b is None:
        return 0.5
    return round(compute_hist_similarity(crop_a, crop_b), 4)


def compute_overlap_seconds(track_a: dict[str, Any], track_b: dict[str, Any]) -> float:
    return round(
        min(float(track_a["end_time"]), float(track_b["end_time"]))
        - max(float(track_a["start_time"]), float(track_b["start_time"])),
        3,
    )


def evaluate_merge_pair(
    track_a: dict[str, Any],
    track_b: dict[str, Any],
    *,
    settings: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    overlap_seconds = compute_overlap_seconds(track_a, track_b)
    cross_class_merge = track_a["class_name"] != track_b["class_name"]
    if cross_class_merge and not settings["allow_cross_class_merge"]:
        return {
            "accepted": False,
            "reason": "class_mismatch",
            "source_track_id": track_a["source_track_id"],
            "target_track_id": track_b["source_track_id"],
            "source_class_name": track_a["class_name"],
            "target_class_name": track_b["class_name"],
            "chunk_id": track_a["chunk_id"],
        }
    if track_a["chunk_id"] != track_b["chunk_id"]:
        return {"accepted": False, "reason": "chunk_mismatch"}

    if overlap_seconds > 0:
        duplicate_candidate = {
            "source_track_id": track_a["source_track_id"],
            "target_track_id": track_b["source_track_id"],
            "class_name": track_a["class_name"],
            "chunk_id": track_a["chunk_id"],
            "overlap_seconds": overlap_seconds,
        }
        if settings["protect_overlap"] and not settings["merge_duplicates"]:
            return {
                "accepted": False,
                "reason": "overlap_protected",
                "possible_duplicate_track": duplicate_candidate,
            }

    if float(track_a["end_time"]) > float(track_b["start_time"]):
        return {"accepted": False, "reason": "invalid_temporal_order"}

    time_gap = round(float(track_b["start_time"]) - float(track_a["end_time"]), 3)
    if time_gap > float(settings["max_gap_seconds"]):
        return {"accepted": False, "reason": "gap_too_large", "time_gap_seconds": time_gap}

    time_gap_score = compute_time_gap_score(time_gap, float(settings["max_gap_seconds"]))
    center_score, center_distance = compute_center_distance_score(
        track_a,
        track_b,
        center_distance_ratio=float(settings["center_distance_ratio"]),
    )
    trajectory_score = compute_trajectory_score(
        track_a,
        track_b,
        time_gap=time_gap,
        center_distance_score=center_score,
        center_distance_ratio=float(settings["center_distance_ratio"]),
    )
    bbox_size_score = compute_bbox_size_score(track_a, track_b)
    appearance_score = compute_appearance_score(
        track_a,
        track_b,
        settings=settings,
        warnings=warnings,
    )

    if appearance_score < float(settings["min_appearance_sim"]):
        return {
            "accepted": False,
            "reason": "appearance_too_different",
            "appearance_score": appearance_score,
            "time_gap_seconds": time_gap,
        }

    impossible_jump = center_score <= 0.0 and trajectory_score <= 0.0
    if impossible_jump:
        return {
            "accepted": False,
            "reason": "impossible_jump",
            "time_gap_seconds": time_gap,
            "center_distance_score": center_score,
            "trajectory_score": trajectory_score,
        }

    overlap_penalty = 0.3 if overlap_seconds > 0 and settings["merge_duplicates"] else 0.0
    merge_score = round(
        (time_gap_score * 0.25)
        + (center_score * 0.25)
        + (trajectory_score * 0.20)
        + (bbox_size_score * 0.15)
        + (appearance_score * 0.15)
        - overlap_penalty,
        4,
    )

    return {
        "accepted": merge_score >= float(settings["merge_score_threshold"]),
        "reason": "score_passed" if merge_score >= float(settings["merge_score_threshold"]) else "score_below_threshold",
        "source_track_id": track_a["source_track_id"],
        "target_track_id": track_b["source_track_id"],
        "class_name": track_a["class_name"],
        "source_class_name": track_a["class_name"],
        "target_class_name": track_b["class_name"],
        "cross_class_merge": cross_class_merge,
        "chunk_id": track_a["chunk_id"],
        "time_gap_seconds": time_gap,
        "time_gap_score": time_gap_score,
        "center_distance_score": center_score,
        "center_distance_pixels": center_distance,
        "trajectory_score": trajectory_score,
        "bbox_size_score": bbox_size_score,
        "appearance_score": appearance_score,
        "overlap_penalty": overlap_penalty,
        "merge_score": merge_score,
        "possible_duplicate_track": (
            {
                "source_track_id": track_a["source_track_id"],
                "target_track_id": track_b["source_track_id"],
                "class_name": track_a["class_name"],
                "chunk_id": track_a["chunk_id"],
                "overlap_seconds": overlap_seconds,
            }
            if overlap_seconds > 0
            else None
        ),
    }


def dedupe_sequence_items(
    items: list[dict[str, Any]],
    *,
    key_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        marker = tuple(item.get(field) for field in key_fields)
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(item)
    deduped.sort(key=lambda entry: (float(entry.get("timestamp", 0.0)), str(entry.get("frame_id", ""))))
    return deduped


def build_merge_chains(
    tracks: list[dict[str, Any]],
    accepted_edges: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    track_lookup = {track["source_track_id"]: track for track in tracks}
    next_map = {edge["source_track_id"]: edge["target_track_id"] for edge in accepted_edges}
    prev_map = {edge["target_track_id"]: edge["source_track_id"] for edge in accepted_edges}

    chains: list[list[dict[str, Any]]] = []
    visited: set[str] = set()
    for track in tracks:
        source_track_id = track["source_track_id"]
        if source_track_id in visited or source_track_id in prev_map:
            continue
        chain: list[dict[str, Any]] = []
        current = source_track_id
        while current and current not in visited:
            visited.add(current)
            chain.append(track_lookup[current])
            current = next_map.get(current)
        chains.append(chain)

    for track in tracks:
        source_track_id = track["source_track_id"]
        if source_track_id not in visited:
            chains.append([track])
            visited.add(source_track_id)
    return chains


def merge_track_chain(
    chain: list[dict[str, Any]],
    *,
    clean_track_id: str,
    merge_reason: str,
    merge_score: float | None,
    chunk_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    merged_bbox_sequence = dedupe_sequence_items(
        [item for track in chain for item in list(track.get("bbox_sequence") or [])],
        key_fields=("timestamp", "frame_id"),
    )
    merged_center_sequence = dedupe_sequence_items(
        [item for track in chain for item in list(track.get("center_sequence") or [])],
        key_fields=("timestamp",),
    )
    detection_ids: list[str] = []
    for track in chain:
        for detection_id in list(track.get("detection_ids") or []):
            detection_id_str = str(detection_id)
            if detection_id_str not in detection_ids:
                detection_ids.append(detection_id_str)

    ordered_tracks = sorted(chain, key=lambda item: (float(item["start_time"]), float(item["end_time"])))
    best_track = max(
        ordered_tracks,
        key=lambda item: (
            float(item.get("max_confidence", 0.0)),
            float(item.get("average_confidence", 0.0)),
            float(item.get("duration_seconds", 0.0)),
        ),
    )
    start_time = round(min(float(track["start_time"]) for track in ordered_tracks), 3)
    end_time = round(max(float(track["end_time"]) for track in ordered_tracks), 3)
    duration_seconds = round(max(0.0, end_time - start_time), 3)
    total_detection_count = sum(int(track.get("detection_count", 0)) for track in ordered_tracks)
    count_weight = max(1, len(ordered_tracks))
    average_confidence = round(
        sum(float(track.get("average_confidence", 0.0)) * max(1, int(track.get("detection_count", 1))) for track in ordered_tracks)
        / max(1, total_detection_count),
        4,
    )
    max_confidence = round(max(float(track.get("max_confidence", 0.0)) for track in ordered_tracks), 4)
    source_raw_tracker_ids = sorted(
        {
            str(raw_id)
            for track in ordered_tracks
            for raw_id in list(track.get("source_raw_tracker_ids") or [])
            if str(raw_id)
        }
    )
    source_fragment_track_ids = sorted(
        {
            str(fragment_id)
            for track in ordered_tracks
            for fragment_id in list(track.get("source_fragment_track_ids") or [])
            if str(fragment_id)
        }
    )
    class_votes: dict[str, int] = defaultdict(int)
    class_confidence_sum: dict[str, float] = defaultdict(float)
    class_history: list[dict[str, Any]] = []
    for track in ordered_tracks:
        for class_name, vote_count in dict(track.get("class_votes") or {}).items():
            class_votes[str(class_name)] += int(vote_count)
        for class_name, confidence_sum in dict(track.get("class_confidence_sum") or {}).items():
            class_confidence_sum[str(class_name)] += float(confidence_sum)
        for item in list(track.get("class_history") or []):
            if isinstance(item, dict):
                class_history.append(item)
    dominant_class_name = str(ordered_tracks[0]["class_name"])
    dominant_vote_count = -1
    dominant_conf_sum = -1.0
    for class_name in sorted(class_votes.keys()):
        vote_count = int(class_votes[class_name])
        conf_sum = float(class_confidence_sum.get(class_name, 0.0))
        if vote_count > dominant_vote_count or (vote_count == dominant_vote_count and conf_sum > dominant_conf_sum):
            dominant_class_name = str(class_name)
            dominant_vote_count = vote_count
            dominant_conf_sum = conf_sum
    total_votes = sum(class_votes.values())
    class_conflict = len(class_votes) > 1
    class_consistency_score = round(dominant_vote_count / max(1, total_votes), 4)
    dominant_class_confidence = round(dominant_conf_sum / max(1, dominant_vote_count), 4) if dominant_vote_count > 0 else 0.0
    chunk_info = chunk_lookup.get(str(ordered_tracks[0]["chunk_id"]), {})
    boundary_seconds = 10.0
    chunk_start = choose_number(chunk_info.get("start_time"), start_time)
    chunk_end = choose_number(chunk_info.get("end_time"), end_time)
    is_cross_chunk_candidate = (
        any(bool(track.get("is_cross_chunk_candidate")) for track in ordered_tracks)
        or start_time <= chunk_start + boundary_seconds
        or end_time >= chunk_end - boundary_seconds
    )

    return {
        "clean_track_id": clean_track_id,
        "class_name": dominant_class_name,
        "dominant_class_name": dominant_class_name,
        "dominant_class_confidence": dominant_class_confidence,
        "class_votes": dict(sorted(class_votes.items())),
        "class_confidence_sum": {key: round(value, 4) for key, value in sorted(class_confidence_sum.items())},
        "class_history": sorted(
            class_history,
            key=lambda item: (float(item.get("timestamp", 0.0)), str(item.get("frame_id", "")), str(item.get("detection_id", ""))),
        ),
        "class_consistency_score": class_consistency_score,
        "class_conflict": class_conflict,
        "chunk_id": str(ordered_tracks[0]["chunk_id"]),
        "chunk_index": int(ordered_tracks[0]["chunk_index"]),
        "source_track_ids": [str(track["source_track_id"]) for track in ordered_tracks],
        "source_raw_tracker_ids": source_raw_tracker_ids or None,
        "source_fragment_track_ids": source_fragment_track_ids or None,
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": duration_seconds,
        "detection_count": max(total_detection_count, len(detection_ids), len(merged_bbox_sequence)),
        "average_confidence": average_confidence,
        "max_confidence": max_confidence,
        "first_frame_id": str(ordered_tracks[0]["first_frame_id"]),
        "last_frame_id": str(ordered_tracks[-1]["last_frame_id"]),
        "best_frame_id": str(best_track.get("best_frame_id") or best_track.get("first_frame_id") or ""),
        "best_image_path": best_track.get("best_image_path"),
        "thumbnail_path": best_track.get("thumbnail_path"),
        "bbox_sequence": merged_bbox_sequence,
        "center_sequence": merged_center_sequence,
        "detection_ids": detection_ids,
        "cleanup_status": "merged_track" if len(ordered_tracks) > 1 else "unchanged_track",
        "count_for_summary": True,
        "needs_review": class_conflict,
        "is_cross_chunk_candidate": is_cross_chunk_candidate,
        "merge_reason": merge_reason,
        "merge_score": merge_score,
        "status": "closed",
    }


def mark_noise_tracks(
    clean_tracks: list[dict[str, Any]],
    *,
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    removed_count = 0
    for track in clean_tracks:
        class_name = str(track.get("class_name") or "")
        duration_seconds = float(track["duration_seconds"])
        detection_count = int(track["detection_count"])
        max_confidence = float(track.get("max_confidence", 0.0))
        if class_name in VEHICLE_CLASSES:
            if detection_count == 1 and max_confidence >= 0.60:
                track["cleanup_status"] = "short_vehicle_candidate"
                track["count_for_summary"] = "review"
                track["needs_review"] = True
                continue
            if detection_count >= 2 and max_confidence >= float(settings["vehicle_noise_max_conf"]):
                track["needs_review"] = False
                continue
            if (
                duration_seconds < float(settings["min_duration_seconds"])
                and detection_count < int(settings["min_detections"])
                and max_confidence < float(settings["vehicle_noise_max_conf"])
            ):
                track["cleanup_status"] = "noise_short_track"
                track["count_for_summary"] = False
                track["needs_review"] = False
                removed_count += 1
            elif (
                duration_seconds < float(settings["min_duration_seconds"])
                or detection_count < int(settings["min_detections"])
            ):
                track["cleanup_status"] = "short_vehicle_candidate"
                track["count_for_summary"] = "review"
                track["needs_review"] = True
            else:
                track["needs_review"] = False
            continue

        if (
            duration_seconds < float(settings["min_duration_seconds"])
            and detection_count < int(settings["min_detections"])
        ):
            track["cleanup_status"] = "noise_short_track"
            track["count_for_summary"] = False
            track["needs_review"] = False
            removed_count += 1
        else:
            track["needs_review"] = False
    return clean_tracks, removed_count


def is_counted_track(track: dict[str, Any]) -> bool:
    return track.get("count_for_summary") in {True, "review"}


def is_strong_track(track: dict[str, Any]) -> bool:
    return track.get("count_for_summary") is True


def is_review_track(track: dict[str, Any]) -> bool:
    return track.get("count_for_summary") == "review"


def compute_max_visible(tracks: list[dict[str, Any]]) -> int:
    events: list[tuple[float, int]] = []
    for track in tracks:
        if not is_counted_track(track):
            continue
        events.append((float(track["start_time"]), 1))
        events.append((float(track["end_time"]), -1))
    events.sort(key=lambda item: (item[0], -item[1]))
    active = 0
    maximum = 0
    for _, delta in events:
        active += delta
        maximum = max(maximum, active)
    return maximum


def cleanup_status_for_counts(clean_track_count: int, max_visible: int) -> str:
    if clean_track_count <= max_visible + 2:
        return "good"
    if clean_track_count <= max_visible + 5:
        return "acceptable_needs_review"
    return "needs_cleanup"


def build_outputs(
    *,
    run_dir: Path,
    normalized_tracks: list[dict[str, Any]],
    selected_input_track_source: str,
    settings: dict[str, Any],
    quality_payload: dict[str, Any],
    chunk_lookup: dict[str, dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    merge_candidates_considered: list[dict[str, Any]] = []
    accepted_edges: list[dict[str, Any]] = []
    rejected_edges: list[dict[str, Any]] = []
    rejected_competing_edges: list[dict[str, Any]] = []
    possible_duplicate_tracks: list[dict[str, Any]] = []
    cross_class_merge_attempts = 0
    cross_class_merges_blocked = 0
    cross_class_merges_accepted = 0

    if settings["cleanup_enabled"]:
        ordered_tracks = sorted(
            normalized_tracks,
            key=lambda item: (
                str(item["chunk_id"]),
                str(item["class_name"]),
                float(item["start_time"]),
                str(item["source_track_id"]),
            ),
        )
        for index, track_a in enumerate(ordered_tracks):
            for track_b in ordered_tracks[index + 1 :]:
                if track_a["chunk_id"] != track_b["chunk_id"]:
                    continue
                if track_a["class_name"] != track_b["class_name"]:
                    continue
                evaluation = evaluate_merge_pair(
                    track_a,
                    track_b,
                    settings=settings,
                    warnings=warnings,
                )
                if evaluation.get("possible_duplicate_track"):
                    possible_duplicate_tracks.append(evaluation["possible_duplicate_track"])
                if bool(evaluation.get("cross_class_merge")) or evaluation.get("reason") == "class_mismatch":
                    cross_class_merge_attempts += 1
                if evaluation.get("source_track_id") and evaluation.get("target_track_id"):
                    merge_candidates_considered.append(evaluation)
                if evaluation.get("accepted"):
                    accepted_edges.append(evaluation)
                    if bool(evaluation.get("cross_class_merge")):
                        cross_class_merges_accepted += 1
                else:
                    rejected_edges.append(evaluation)
                    if evaluation.get("reason") == "class_mismatch":
                        cross_class_merges_blocked += 1

        accepted_edges.sort(key=lambda item: float(item.get("merge_score", 0.0)), reverse=True)
        final_accepted: list[dict[str, Any]] = []
        used_sources: set[str] = set()
        used_targets: set[str] = set()
        for edge in accepted_edges:
            source_track_id = str(edge["source_track_id"])
            target_track_id = str(edge["target_track_id"])
            if source_track_id in used_sources or target_track_id in used_targets:
                rejected_edge = {**edge, "reason": "rejected_competing_lower_score_edge"}
                rejected_competing_edges.append(rejected_edge)
                continue
            used_sources.add(source_track_id)
            used_targets.add(target_track_id)
            final_accepted.append(edge)
        accepted_edges = final_accepted
    else:
        warnings.append("Track cleanup is disabled; Step 5B returned unchanged tracks.")

    chains = build_merge_chains(normalized_tracks, accepted_edges)
    clean_tracks: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    class_counters: dict[tuple[str, str], int] = defaultdict(int)
    edge_lookup = {
        (str(edge["source_track_id"]), str(edge["target_track_id"])): edge
        for edge in accepted_edges
    }

    for chain in sorted(
        chains,
        key=lambda items: (
            str(items[0]["chunk_id"]),
            str(items[0]["class_name"]),
            float(items[0]["start_time"]),
        ),
    ):
        chunk_id = str(chain[0]["chunk_id"])
        class_name = str(chain[0]["class_name"])
        class_counters[(chunk_id, class_name)] += 1
        clean_track_id = (
            f"clean_{chunk_id}_{class_name}_{class_counters[(chunk_id, class_name)]:06d}"
        )
        chain_edges = [
            edge_lookup[(chain[index]["source_track_id"], chain[index + 1]["source_track_id"])]
            for index in range(len(chain) - 1)
            if (chain[index]["source_track_id"], chain[index + 1]["source_track_id"]) in edge_lookup
        ]
        merge_reason = (
            "small_time_gap_close_center_similar_appearance"
            if len(chain) > 1
            else "unchanged_single_track"
        )
        merge_score = (
            round(sum(float(edge["merge_score"]) for edge in chain_edges) / len(chain_edges), 4)
            if chain_edges
            else None
        )
        clean_track = merge_track_chain(
            chain,
            clean_track_id=clean_track_id,
            merge_reason=merge_reason,
            merge_score=merge_score,
            chunk_lookup=chunk_lookup,
        )
        clean_tracks.append(clean_track)
        mapping_rows.append(
            {
                "clean_track_id": clean_track_id,
                "class_name": str(clean_track["class_name"]),
                "chunk_id": chunk_id,
                "source_track_ids": list(clean_track["source_track_ids"]),
                "merge_reason": merge_reason,
                "merge_score": merge_score,
            }
        )

    clean_tracks, tracks_removed_as_noise = mark_noise_tracks(
        clean_tracks,
        settings=settings,
    )

    cleanup_by_class: dict[str, dict[str, Any]] = {}
    max_objects_visible_at_once_by_class: dict[str, int] = {}
    estimated_unique_objects_by_class: dict[str, dict[str, int]] = {}

    grouped_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    class_changed_tracks: list[dict[str, Any]] = []
    for track in clean_tracks:
        grouped_by_class[str(track["class_name"])].append(track)
        if str(track.get("class_name") or "") != str(track.get("dominant_class_name") or track.get("class_name") or ""):
            class_changed_tracks.append(
                {
                    "clean_track_id": str(track.get("clean_track_id") or ""),
                    "class_name": str(track.get("class_name") or ""),
                    "dominant_class_name": str(track.get("dominant_class_name") or ""),
                }
            )

    for class_name, class_tracks in sorted(grouped_by_class.items()):
        counted_tracks = [track for track in class_tracks if is_counted_track(track)]
        strong_tracks = [track for track in class_tracks if is_strong_track(track)]
        review_tracks = [track for track in class_tracks if is_review_track(track)]
        noise_tracks = [
            track for track in class_tracks if str(track["cleanup_status"]) == "noise_short_track"
        ]
        max_visible = compute_max_visible(class_tracks)
        clean_track_count = len(counted_tracks)
        status = cleanup_status_for_counts(clean_track_count, max_visible)
        cleanup_by_class[class_name] = {
            "input_track_count": sum(
                1 for track in normalized_tracks if str(track["class_name"]) == class_name
            ),
            "clean_track_count": clean_track_count,
            "strong_clean_track_count": len(strong_tracks),
            "review_track_count": len(review_tracks),
            "noise_track_count": len(noise_tracks),
            "tracks_removed_as_noise": len(noise_tracks),
            "tracks_merged_count": sum(
                max(0, len(list(track["source_track_ids"])) - 1) for track in counted_tracks
            ),
            "cleanup_status": status,
        }
        max_objects_visible_at_once_by_class[class_name] = max_visible
        estimated_unique_objects_by_class[class_name] = {
            "lower_bound": max_visible,
            "upper_bound": clean_track_count if clean_track_count >= max_visible else max_visible,
        }

    person_input_count = sum(
        1 for track in normalized_tracks if str(track["class_name"]) == "person"
    )
    person_clean_count = len(
        [track for track in clean_tracks if str(track["class_name"]) == "person" and is_counted_track(track)]
    )
    person_max_visible = max_objects_visible_at_once_by_class.get("person", 0)
    person_cleanup_status = cleanup_status_for_counts(person_clean_count, person_max_visible)
    vehicle_short_candidates_count = sum(1 for track in clean_tracks if is_review_track(track))
    strong_clean_track_count = sum(1 for track in clean_tracks if is_strong_track(track))
    review_track_count = sum(1 for track in clean_tracks if is_review_track(track))
    noise_track_count = sum(
        1 for track in clean_tracks if str(track["cleanup_status"]) == "noise_short_track"
    )

    report_payload = {
        "created_at": current_timestamp(),
        "selected_input_track_source": selected_input_track_source,
        "input_track_count": len(normalized_tracks),
        "clean_track_count": sum(1 for track in clean_tracks if is_counted_track(track)),
        "strong_clean_track_count": strong_clean_track_count,
        "review_track_count": review_track_count,
        "noise_track_count": noise_track_count,
        "tracks_removed_as_noise": tracks_removed_as_noise,
        "tracks_merged_count": sum(max(0, len(list(track["source_track_ids"])) - 1) for track in clean_tracks),
        "cleanup_parameter_profile": settings["cleanup_parameter_profile"],
        "cross_class_merge_attempts": cross_class_merge_attempts,
        "cross_class_merges_blocked": cross_class_merges_blocked,
        "cross_class_merges_accepted": cross_class_merges_accepted,
        "class_changed_tracks": class_changed_tracks,
        "vehicle_short_candidates_count": vehicle_short_candidates_count,
        "tracks_marked_review_count": review_track_count,
        "cleanup_by_class": cleanup_by_class,
        "max_objects_visible_at_once_by_class": max_objects_visible_at_once_by_class,
        "estimated_unique_objects_by_class": estimated_unique_objects_by_class,
        "possible_duplicate_tracks": possible_duplicate_tracks,
        "merge_candidates_considered": merge_candidates_considered,
        "merge_edges_accepted": accepted_edges,
        "merge_edges_rejected": rejected_edges,
        "rejected_competing_edges_count": len(rejected_competing_edges),
        "rejected_competing_edges": rejected_competing_edges,
        "accepted_merge_edges": accepted_edges,
        "raw_person_tracks_input": person_input_count,
        "clean_person_tracks": person_clean_count,
        "max_people_visible_at_once": person_max_visible,
        "estimated_unique_people_range": {
            "lower_bound": person_max_visible,
            "upper_bound": person_clean_count if person_clean_count >= person_max_visible else person_max_visible,
        },
        "cleanup_status": person_cleanup_status,
        "warnings": list(dict.fromkeys(warnings)),
        "recommendations": build_recommendations(
            quality_payload=quality_payload,
            report_data={
                "person_clean_count": person_clean_count,
                "person_input_count": person_input_count,
                "possible_duplicate_count": len(possible_duplicate_tracks),
                "rejected_competing_edges_count": len(rejected_competing_edges),
                "untracked_detection_ratio": choose_number(
                    quality_payload.get("total_untracked_detections"), 0.0
                ) / max(1.0, choose_number(quality_payload.get("total_input_detections"), 0.0)),
            },
        ),
    }

    clean_tracks_payload = {
        "created_at": current_timestamp(),
        "source_tracks_file": str(run_dir / "05_tracks.json"),
        "selected_input_track_source": selected_input_track_source,
        "cleanup_enabled": bool(settings["cleanup_enabled"]),
        "cleanup_settings": settings,
        "raw_input_track_count": len(normalized_tracks),
        "clean_track_count": sum(1 for track in clean_tracks if is_counted_track(track)),
        "tracks_removed_as_noise": tracks_removed_as_noise,
        "tracks_merged_count": sum(max(0, len(list(track["source_track_ids"])) - 1) for track in clean_tracks),
        "clean_tracks": clean_tracks,
    }
    mapping_payload = {
        "created_at": current_timestamp(),
        "selected_input_track_source": selected_input_track_source,
        "mappings": mapping_rows,
    }
    return {
        "clean_tracks_payload": clean_tracks_payload,
        "report_payload": report_payload,
        "mapping_payload": mapping_payload,
    }


def build_recommendations(
    *,
    quality_payload: dict[str, Any],
    report_data: dict[str, Any],
) -> list[str]:
    recommendations: list[str] = []
    main_problem = list(quality_payload.get("main_problem") or [])
    if "dropped_tracker_ids" in main_problem:
        recommendations.append("Lower FINAL_DEMO_YOLO_CONF to 0.15 if tracker is dropping detections.")
        recommendations.append("Increase FINAL_DEMO_SAMPLE_FPS to 5 or 8.")
    if report_data["untracked_detection_ratio"] > 0.35:
        recommendations.append("Lower FINAL_DEMO_YOLO_CONF to 0.10.")
        recommendations.append("Increase FINAL_DEMO_SAMPLE_FPS to 8 or 10.")
        recommendations.append("Lower FINAL_DEMO_BYTETRACK_TRACK_THRESH to 0.05.")
        recommendations.append("Lower FINAL_DEMO_BYTETRACK_MATCH_THRESH to 0.60.")
        recommendations.append("Use BoT-SORT/OC-SORT/DeepOCSORT for production vehicle tracking.")
    if report_data["rejected_competing_edges_count"] > 0:
        recommendations.append("Review rejected competing edges before loosening merge thresholds.")
    if report_data["possible_duplicate_count"] > 0:
        recommendations.append("Inspect possible duplicate tracks before enabling FINAL_DEMO_CLEANUP_MERGE_DUPLICATES.")
    recommendations.append("If cleanup still leaves heavy fragmentation, use BoT-SORT/ReID or global identity linking.")
    recommendations.append("For production unique identity, add cross-chunk/global ReID after local tracking.")
    return list(dict.fromkeys(recommendations))


def build_track_cleanup_outputs(run_dir: Path) -> dict[str, Any]:
    tracks_path = run_dir / "05_tracks.json"
    quality_path = run_dir / "05A_tracking_quality_report.json"
    if not tracks_path.exists():
        raise FileNotFoundError(
            f"Missing required Step 5B input: {tracks_path}. Step 5 must run before Step 5B."
        )
    if not quality_path.exists():
        raise FileNotFoundError(
            f"Missing required Step 5B input: {quality_path}. Step 5A must run before Step 5B."
        )

    diagnostics_payload = read_optional_json(run_dir / "05_tracking_diagnostics.json")
    report_payload = read_optional_json(run_dir / "05_tracking_report.json")
    frames_index_payload = read_optional_json(run_dir / "03_sampled_frames_index.json")
    chunk_manifest_payload = read_optional_json(run_dir / "02_chunk_manifest.json")
    focus_payload = read_optional_json(run_dir / "05_tracking_focus.json")
    tracks_payload = read_json(tracks_path)
    quality_payload = read_json(quality_path)

    warnings: list[str] = []
    if diagnostics_payload is None:
        warnings.append("Optional file missing: 05_tracking_diagnostics.json")
    if report_payload is None:
        warnings.append("Optional file missing: 05_tracking_report.json")
    if frames_index_payload is None:
        warnings.append("Optional file missing: 03_sampled_frames_index.json")
    if chunk_manifest_payload is None:
        warnings.append("Optional file missing: 02_chunk_manifest.json")
    if focus_payload is None:
        warnings.append("Optional file missing: 05_tracking_focus.json")

    cleanup_defaults = build_cleanup_defaults(focus_payload)
    settings = read_cleanup_settings(cleanup_defaults)
    selected_input_track_source = infer_track_source(
        tracks_payload,
        quality_payload,
        diagnostics_payload,
        report_payload,
    )
    frame_lookup = build_frame_lookup(frames_index_payload)
    chunk_lookup = build_chunk_lookup(chunk_manifest_payload)

    normalized_tracks: list[dict[str, Any]] = []
    for item in list(tracks_payload.get("tracks") or []):
        if isinstance(item, dict):
            normalized_tracks.append(
                normalize_track(
                    item,
                    frame_lookup=frame_lookup,
                    chunk_lookup=chunk_lookup,
                    warnings=warnings,
                )
            )

    outputs = build_outputs(
        run_dir=run_dir,
        normalized_tracks=normalized_tracks,
        selected_input_track_source=selected_input_track_source,
        settings=settings,
        quality_payload=quality_payload,
        chunk_lookup=chunk_lookup,
        warnings=warnings,
    )
    tracking_focus = {
        "focus_mode": focus_payload.get("focus_mode") if isinstance(focus_payload, dict) else None,
        "selected_focus_profile": (
            focus_payload.get("selected_focus_profile") if isinstance(focus_payload, dict) else None
        ),
        "selected_track_classes": (
            list(focus_payload.get("selected_track_classes") or [])
            if isinstance(focus_payload, dict)
            else []
        ),
        "focus_confidence": (
            focus_payload.get("focus_confidence") if isinstance(focus_payload, dict) else None
        ),
    }
    outputs["clean_tracks_payload"]["tracking_focus"] = tracking_focus
    outputs["report_payload"]["tracking_focus"] = tracking_focus
    outputs["mapping_payload"]["tracking_focus"] = tracking_focus
    return outputs


def update_run_manifest_for_track_cleanup(run_manifest_path: Path) -> dict[str, Any]:
    run_manifest = read_json(run_manifest_path)
    completed_steps = list(run_manifest.get("completed_steps", []))
    if "05B_track_cleanup" not in completed_steps:
        completed_steps.append("05B_track_cleanup")
    run_manifest["completed_steps"] = completed_steps
    run_manifest["next_step"] = "06_attribute_extraction"
    write_json(run_manifest_path, run_manifest)
    return run_manifest
