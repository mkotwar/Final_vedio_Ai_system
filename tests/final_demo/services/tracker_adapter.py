from __future__ import annotations

import inspect
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tests.final_demo.services.chunk_planner import read_json
from tests.final_demo.services.video_io import current_timestamp, write_json


ENV_FINAL_DEMO_BYTETRACK_TRACK_THRESH = "FINAL_DEMO_BYTETRACK_TRACK_THRESH"
ENV_FINAL_DEMO_BYTETRACK_MATCH_THRESH = "FINAL_DEMO_BYTETRACK_MATCH_THRESH"
ENV_FINAL_DEMO_BYTETRACK_TRACK_BUFFER = "FINAL_DEMO_BYTETRACK_TRACK_BUFFER"
ENV_FINAL_DEMO_TRACK_MIN_DETECTIONS = "FINAL_DEMO_TRACK_MIN_DETECTIONS"
ENV_FINAL_DEMO_TRACK_BOUNDARY_SECONDS = "FINAL_DEMO_TRACK_BOUNDARY_SECONDS"
ENV_FINAL_DEMO_TRACK_CLASSES = "FINAL_DEMO_TRACK_CLASSES"
ENV_FINAL_DEMO_MERGE_TRACK_FRAGMENTS = "FINAL_DEMO_MERGE_TRACK_FRAGMENTS"
ENV_FINAL_DEMO_FRAGMENT_MAX_GAP_SECONDS = "FINAL_DEMO_FRAGMENT_MAX_GAP_SECONDS"
ENV_FINAL_DEMO_FRAGMENT_MAX_CENTER_DISTANCE_RATIO = "FINAL_DEMO_FRAGMENT_MAX_CENTER_DISTANCE_RATIO"
ENV_FINAL_DEMO_FRAGMENT_MIN_APPEARANCE_SIM = "FINAL_DEMO_FRAGMENT_MIN_APPEARANCE_SIM"
ENV_FINAL_DEMO_SAMPLE_FPS = "FINAL_DEMO_SAMPLE_FPS"
ENV_FINAL_DEMO_SOURCE_FPS = "FINAL_DEMO_SOURCE_FPS"
ENV_FINAL_DEMO_FRAME_SKIP_RATIO = "FINAL_DEMO_FRAME_SKIP_RATIO"

DEFAULT_BYTETRACK_TRACK_THRESH = 0.15
DEFAULT_BYTETRACK_MATCH_THRESH = 0.75
DEFAULT_BYTETRACK_TRACK_BUFFER = 60
DEFAULT_TRACK_MIN_DETECTIONS = 2
DEFAULT_TRACK_BOUNDARY_SECONDS = 10.0
DEFAULT_MERGE_TRACK_FRAGMENTS = True
DEFAULT_FRAGMENT_MAX_GAP_SECONDS = 3.0
DEFAULT_FRAGMENT_MAX_CENTER_DISTANCE_RATIO = 0.30
DEFAULT_FRAGMENT_MIN_APPEARANCE_SIM = 0.55
DEFAULT_TRACK_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "backpack",
    "handbag",
    "suitcase",
    "dog",
    "cat",
    "horse",
    "sheep",
    "cow",
]

TRACKER_NAME = "supervision_bytetrack"
TRACKER_TYPE = "bytetrack"


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def to_absolute_repo_path(relative_repo_path: str) -> Path:
    return get_repo_root() / Path(relative_repo_path)


def to_repo_relative_path(path: Path) -> str:
    return path.resolve().relative_to(get_repo_root()).as_posix()


def read_float_env(env_name: str, default_value: float) -> float:
    raw_value = os.environ.get(env_name, str(default_value))
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {env_name} must be a valid number. Received: {raw_value!r}"
        ) from exc
    return value


def read_positive_float_env(env_name: str, default_value: float) -> float:
    value = read_float_env(env_name, default_value)
    if value <= 0:
        raise ValueError(
            f"Environment variable {env_name} must be greater than 0. Received: {value}"
        )
    return value


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
        f"Environment variable {env_name} must be a boolean-like value. Received: {raw_value!r}"
    )


def read_track_classes() -> list[str]:
    raw_value = os.environ.get(ENV_FINAL_DEMO_TRACK_CLASSES, "")
    if not raw_value.strip():
        return list(DEFAULT_TRACK_CLASSES)

    classes = [item.strip().lower() for item in raw_value.split(",") if item.strip()]
    if not classes:
        raise ValueError(
            f"Environment variable {ENV_FINAL_DEMO_TRACK_CLASSES} did not contain any valid class names."
        )
    return classes


def read_tracker_settings() -> dict[str, Any]:
    return {
        "track_thresh": round(
            read_positive_float_env(
                ENV_FINAL_DEMO_BYTETRACK_TRACK_THRESH,
                DEFAULT_BYTETRACK_TRACK_THRESH,
            ),
            3,
        ),
        "match_thresh": round(
            read_positive_float_env(
                ENV_FINAL_DEMO_BYTETRACK_MATCH_THRESH,
                DEFAULT_BYTETRACK_MATCH_THRESH,
            ),
            3,
        ),
        "track_buffer": read_positive_int_env(
            ENV_FINAL_DEMO_BYTETRACK_TRACK_BUFFER,
            DEFAULT_BYTETRACK_TRACK_BUFFER,
        ),
        "min_detections": read_positive_int_env(
            ENV_FINAL_DEMO_TRACK_MIN_DETECTIONS,
            DEFAULT_TRACK_MIN_DETECTIONS,
        ),
        "boundary_seconds": round(
            read_non_negative_float_env(
                ENV_FINAL_DEMO_TRACK_BOUNDARY_SECONDS,
                DEFAULT_TRACK_BOUNDARY_SECONDS,
            ),
            3,
        ),
        "merge_track_fragments": read_bool_env(
            ENV_FINAL_DEMO_MERGE_TRACK_FRAGMENTS,
            DEFAULT_MERGE_TRACK_FRAGMENTS,
        ),
        "fragment_max_gap_seconds": round(
            read_non_negative_float_env(
                ENV_FINAL_DEMO_FRAGMENT_MAX_GAP_SECONDS,
                DEFAULT_FRAGMENT_MAX_GAP_SECONDS,
            ),
            3,
        ),
        "fragment_max_center_distance_ratio": round(
            read_non_negative_float_env(
                ENV_FINAL_DEMO_FRAGMENT_MAX_CENTER_DISTANCE_RATIO,
                DEFAULT_FRAGMENT_MAX_CENTER_DISTANCE_RATIO,
            ),
            3,
        ),
        "fragment_min_appearance_sim": round(
            read_non_negative_float_env(
                ENV_FINAL_DEMO_FRAGMENT_MIN_APPEARANCE_SIM,
                DEFAULT_FRAGMENT_MIN_APPEARANCE_SIM,
            ),
            3,
        ),
        "track_classes": read_track_classes(),
    }


def load_supervision_module() -> Any:
    try:
        import supervision as sv
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Supervision is not installed. Install it with: pip install supervision"
        ) from exc

    if not hasattr(sv, "ByteTrack") or not hasattr(sv, "Detections"):
        raise RuntimeError(
            "Installed supervision package does not expose ByteTrack/Detections as expected."
        )
    return sv


def build_chunk_lookup(chunk_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(chunk["chunk_id"]): chunk
        for chunk in list(chunk_manifest.get("chunks", []))
        if isinstance(chunk, dict) and "chunk_id" in chunk
    }


def compute_bbox_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def compute_center_distance(center_a: list[float], center_b: list[float]) -> float:
    return math.dist(center_a, center_b)


def compute_frame_diagonal(frame_width: int, frame_height: int) -> float:
    return math.hypot(frame_width, frame_height)


def compute_bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    union = compute_bbox_area(box_a) + compute_bbox_area(box_b) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def crop_bbox_from_image(image_path: Path, bbox_xyxy: list[float]) -> Any | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    x1, y1, x2, y2 = [int(round(value)) for value in bbox_xyxy]
    height, width = image.shape[:2]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def compute_hist_similarity(crop_a: Any | None, crop_b: Any | None) -> float:
    if crop_a is None or crop_b is None:
        return 0.5

    hsv_a = cv2.cvtColor(crop_a, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(crop_b, cv2.COLOR_BGR2HSV)
    hist_a = cv2.calcHist([hsv_a], [0, 1], None, [16, 16], [0, 180, 0, 256])
    hist_b = cv2.calcHist([hsv_b], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    similarity = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
    return max(0.0, min(1.0, round((float(similarity) + 1.0) / 2.0, 4)))


def group_detections_by_chunk(
    detections: list[dict[str, Any]],
    *,
    allowed_classes: set[str],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detection in detections:
        class_name = str(detection["class_name"]).lower()
        if class_name not in allowed_classes:
            continue
        grouped[str(detection["chunk_id"])].append(detection)

    for items in grouped.values():
        items.sort(
            key=lambda item: (
                float(item["global_timestamp_seconds"]),
                str(item["frame_id"]),
                str(item["detection_id"]),
            )
        )
    return dict(grouped)


def group_detections_by_frame(chunk_detections: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    detections_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detection in chunk_detections:
        detections_by_frame[str(detection["frame_id"])].append(detection)

    ordered_frames = sorted(
        detections_by_frame.values(),
        key=lambda items: (
            float(items[0]["global_timestamp_seconds"]),
            str(items[0]["frame_id"]),
        ),
    )
    for items in ordered_frames:
        items.sort(key=lambda item: str(item["detection_id"]))
    return ordered_frames


def extract_sample_timing_info(
    frames_index_payload: dict[str, Any],
) -> dict[str, float | None]:
    frames = list(frames_index_payload.get("frames", []))
    sample_fps_env = os.environ.get(ENV_FINAL_DEMO_SAMPLE_FPS)
    source_fps_env = os.environ.get(ENV_FINAL_DEMO_SOURCE_FPS)
    frame_skip_ratio_env = os.environ.get(ENV_FINAL_DEMO_FRAME_SKIP_RATIO)

    sample_fps: float | None = None
    source_fps: float | None = None
    frame_skip_ratio: float | None = None
    average_time_gap_between_frames = 0.0

    if sample_fps_env:
        sample_fps = float(sample_fps_env)
    elif frames:
        sample_fps = float(frames[0].get("sample_fps", 0.0) or 0.0)
    if sample_fps is not None and sample_fps <= 0:
        sample_fps = None

    if source_fps_env:
        source_fps = float(source_fps_env)
    elif frames:
        frame_index_estimates = [
            int(item.get("frame_index_estimate", 0) or 0)
            for item in frames[: min(len(frames), 8)]
        ]
        timestamp_values = [
            float(item.get("global_timestamp_seconds", 0.0) or 0.0)
            for item in frames[: min(len(frames), 8)]
        ]
        ratios = []
        for (frame_a, time_a), (frame_b, time_b) in zip(
            zip(frame_index_estimates, timestamp_values),
            zip(frame_index_estimates[1:], timestamp_values[1:]),
        ):
            delta_time = time_b - time_a
            delta_frames = frame_b - frame_a
            if delta_time > 0 and delta_frames > 0:
                ratios.append(delta_frames / delta_time)
        if ratios:
            source_fps = round(sum(ratios) / len(ratios), 3)

    timestamps = []
    seen_frames: set[str] = set()
    for frame in frames:
        frame_id = str(frame.get("frame_id", ""))
        if frame_id in seen_frames:
            continue
        seen_frames.add(frame_id)
        timestamps.append(float(frame.get("global_timestamp_seconds", 0.0) or 0.0))

    if len(timestamps) > 1:
        deltas = [
            round(current - previous, 3)
            for previous, current in zip(timestamps, timestamps[1:])
            if current > previous
        ]
        if deltas:
            average_time_gap_between_frames = round(sum(deltas) / len(deltas), 3)

    if frame_skip_ratio_env:
        frame_skip_ratio = float(frame_skip_ratio_env)
    elif source_fps and sample_fps and sample_fps > 0:
        frame_skip_ratio = round(source_fps / sample_fps, 3)
    elif sample_fps and average_time_gap_between_frames > 0:
        frame_skip_ratio = round(average_time_gap_between_frames * sample_fps, 3)

    return {
        "sample_fps": round(sample_fps, 3) if sample_fps is not None else None,
        "source_fps": round(source_fps, 3) if source_fps is not None else None,
        "frame_skip_ratio": round(frame_skip_ratio, 3) if frame_skip_ratio is not None else None,
        "average_time_gap_between_frames": average_time_gap_between_frames,
    }


def infer_frame_rate_from_sample_fps(sample_fps: float | None, chunk_detections: list[dict[str, Any]]) -> int:
    if sample_fps and sample_fps > 0:
        return max(1, int(round(sample_fps)))

    timestamps = sorted(
        {
            round(float(item["global_timestamp_seconds"]), 3)
            for item in chunk_detections
        }
    )
    if len(timestamps) < 2:
        return 30
    deltas = [current - previous for previous, current in zip(timestamps, timestamps[1:]) if current > previous]
    if not deltas:
        return 30
    average_delta = sum(deltas) / len(deltas)
    if average_delta <= 0:
        return 30
    return max(1, min(120, int(round(1.0 / average_delta))))


def create_bytetrack_instance(
    sv: Any,
    *,
    settings: dict[str, Any],
    frame_rate: int,
    effective_track_buffer: int,
) -> Any:
    signature = inspect.signature(sv.ByteTrack)
    kwargs: dict[str, Any] = {}
    parameter_aliases = {
        "track_thresh": ["track_activation_threshold", "track_thresh", "track_threshold"],
        "match_thresh": ["minimum_matching_threshold", "match_thresh", "match_threshold"],
        "track_buffer": ["lost_track_buffer", "track_buffer", "buffer_size"],
        "frame_rate": ["frame_rate", "fps"],
    }
    configured_values = {
        "track_thresh": settings["track_thresh"],
        "match_thresh": settings["match_thresh"],
        "track_buffer": effective_track_buffer,
        "frame_rate": frame_rate,
    }
    for logical_name, aliases in parameter_aliases.items():
        for alias in aliases:
            if alias in signature.parameters:
                kwargs[alias] = configured_values[logical_name]
                break
    return sv.ByteTrack(**kwargs)


def build_supervision_detections(sv: Any, frame_detections: list[dict[str, Any]]) -> Any:
    xyxy = np.array(
        [list(map(float, item["bbox_xyxy"])) for item in frame_detections],
        dtype=np.float32,
    )
    confidence = np.array(
        [float(item["confidence"]) for item in frame_detections],
        dtype=np.float32,
    )
    class_id = np.array(
        [int(item["class_id"]) for item in frame_detections],
        dtype=np.int32,
    )
    data = {
        "detection_id": np.array([str(item["detection_id"]) for item in frame_detections], dtype=object),
        "frame_id": np.array([str(item["frame_id"]) for item in frame_detections], dtype=object),
        "image_path": np.array([str(item["image_path"]) for item in frame_detections], dtype=object),
        "class_name": np.array([str(item["class_name"]) for item in frame_detections], dtype=object),
        "timestamp": np.array(
            [float(item["global_timestamp_seconds"]) for item in frame_detections],
            dtype=np.float32,
        ),
        "confidence_value": np.array(
            [float(item["confidence"]) for item in frame_detections],
            dtype=np.float32,
        ),
    }
    return sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
        data=data,
    )


def read_data_array(detections_obj: Any, key: str) -> list[Any]:
    data = getattr(detections_obj, "data", None)
    if data is None:
        return []
    values = data.get(key)
    if values is None:
        return []
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


def normalize_tracker_id(raw_tracker_id: Any) -> str | None:
    if raw_tracker_id is None:
        return None
    if isinstance(raw_tracker_id, float) and math.isnan(raw_tracker_id):
        return None
    if isinstance(raw_tracker_id, np.generic):
        raw_tracker_id = raw_tracker_id.item()
    if isinstance(raw_tracker_id, float) and raw_tracker_id.is_integer():
        raw_tracker_id = int(raw_tracker_id)
    return str(raw_tracker_id)


def build_track_output(
    track: dict[str, Any],
    *,
    local_track_id: str,
    class_name: str,
    thumbnail_root: Path,
    chunk_start_time: float,
    chunk_end_time: float,
    boundary_seconds: float,
    warnings: list[str],
) -> dict[str, Any]:
    detections = list(track["detections"])
    best_detection = max(
        detections,
        key=lambda item: (
            float(item["confidence"]),
            float(item["bbox_area_pixels"]),
            -float(item["global_timestamp_seconds"]),
        ),
    )
    thumbnail_path: str | None = None
    best_image_path = to_absolute_repo_path(str(best_detection["image_path"]))
    crop = crop_bbox_from_image(best_image_path, list(best_detection["bbox_xyxy"]))
    if crop is None:
        warnings.append(
            f"Failed to crop thumbnail for track {local_track_id} from frame {best_detection['frame_id']}."
        )
    else:
        chunk_thumbnail_dir = thumbnail_root / str(track["chunk_id"])
        chunk_thumbnail_dir.mkdir(parents=True, exist_ok=True)
        thumbnail_output_path = chunk_thumbnail_dir / f"{local_track_id}.jpg"
        write_success = cv2.imwrite(str(thumbnail_output_path), crop)
        if not write_success:
            warnings.append(f"Failed to write thumbnail image: {thumbnail_output_path}")
        else:
            thumbnail_path = to_repo_relative_path(thumbnail_output_path)

    start_time = float(detections[0]["global_timestamp_seconds"])
    end_time = float(detections[-1]["global_timestamp_seconds"])
    is_cross_chunk_candidate = (
        start_time <= chunk_start_time + boundary_seconds
        or end_time >= chunk_end_time - boundary_seconds
    )
    raw_tracker_ids = sorted({str(item["raw_tracker_id"]) for item in detections})
    source_fragment_track_ids = list(track.get("source_fragment_track_ids", [local_track_id]))

    raw_tracker_id_value: str | None
    if len(raw_tracker_ids) == 1:
        raw_tracker_id_value = raw_tracker_ids[0]
        source_raw_tracker_ids: list[str] | None = None
    else:
        raw_tracker_id_value = None
        source_raw_tracker_ids = raw_tracker_ids

    return {
        "local_track_id": local_track_id,
        "raw_tracker_id": raw_tracker_id_value,
        "source_raw_tracker_ids": source_raw_tracker_ids,
        "source_fragment_track_ids": source_fragment_track_ids,
        "chunk_id": track["chunk_id"],
        "chunk_index": track["chunk_index"],
        "class_name": class_name,
        "start_time": round(start_time, 3),
        "end_time": round(end_time, 3),
        "duration_seconds": round(end_time - start_time, 3),
        "detection_count": len(detections),
        "average_confidence": round(
            sum(float(item["confidence"]) for item in detections) / len(detections),
            4,
        ),
        "max_confidence": round(max(float(item["confidence"]) for item in detections), 4),
        "first_frame_id": str(detections[0]["frame_id"]),
        "last_frame_id": str(detections[-1]["frame_id"]),
        "best_frame_id": str(best_detection["frame_id"]),
        "best_image_path": str(best_detection["image_path"]),
        "thumbnail_path": thumbnail_path,
        "bbox_sequence": [
            {
                "timestamp": round(float(item["global_timestamp_seconds"]), 3),
                "frame_id": str(item["frame_id"]),
                "bbox_xyxy": [round(float(value), 3) for value in item["bbox_xyxy"]],
                "confidence": round(float(item["confidence"]), 4),
            }
            for item in detections
        ],
        "center_sequence": [
            {
                "timestamp": round(float(item["global_timestamp_seconds"]), 3),
                "center": [round(float(value), 3) for value in item["bbox_center"]],
            }
            for item in detections
        ],
        "detection_ids": [str(item["detection_id"]) for item in detections],
        "status": "closed",
        "is_cross_chunk_candidate": is_cross_chunk_candidate,
    }


def build_raw_track_fragments(
    sv: Any,
    *,
    chunk_id: str,
    chunk_detections: list[dict[str, Any]],
    settings: dict[str, Any],
    timing_info: dict[str, float | None],
    warnings: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    raw_tracks: list[dict[str, Any]] = []
    frame_diagnostics: list[dict[str, Any]] = []
    total_untracked_detections = 0

    detections_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detection in chunk_detections:
        detections_by_class[str(detection["class_name"]).lower()].append(detection)

    sample_fps = timing_info.get("sample_fps")
    frame_skip_ratio = timing_info.get("frame_skip_ratio") or 1.0
    effective_track_buffer = max(
        1,
        int(round(int(settings["track_buffer"]) * float(frame_skip_ratio))),
    )

    for class_name, class_detections in sorted(detections_by_class.items()):
        frame_rate = infer_frame_rate_from_sample_fps(
            float(sample_fps) if sample_fps is not None else None,
            class_detections,
        )
        tracker = create_bytetrack_instance(
            sv,
            settings=settings,
            frame_rate=frame_rate,
            effective_track_buffer=effective_track_buffer,
        )
        track_map: dict[str, dict[str, Any]] = {}

        for frame_detections in group_detections_by_frame(class_detections):
            supervision_detections = build_supervision_detections(sv, frame_detections)
            tracked_detections = tracker.update_with_detections(supervision_detections)

            tracker_ids = getattr(tracked_detections, "tracker_id", None)
            tracker_ids_list = tracker_ids.tolist() if hasattr(tracker_ids, "tolist") else list(tracker_ids or [])
            returned_detection_ids = [str(item) for item in read_data_array(tracked_detections, "detection_id")]
            detection_lookup = {
                str(item["detection_id"]): item
                for item in frame_detections
            }

            tracked_detection_ids: list[str] = []
            assigned_tracker_ids: list[str] = []
            for detection_id, raw_tracker_id in zip(returned_detection_ids, tracker_ids_list):
                tracker_id = normalize_tracker_id(raw_tracker_id)
                if tracker_id is None:
                    continue
                detection = detection_lookup.get(detection_id)
                if detection is None:
                    continue
                tracked_detection_ids.append(detection_id)
                assigned_tracker_ids.append(tracker_id)
                track_entry = track_map.get(tracker_id)
                if track_entry is None:
                    track_entry = {
                        "raw_tracker_id": tracker_id,
                        "chunk_id": chunk_id,
                        "chunk_index": int(detection["chunk_index"]),
                        "class_name": class_name,
                        "detections": [],
                    }
                    track_map[tracker_id] = track_entry
                detection_with_tracker = {**detection, "raw_tracker_id": tracker_id}
                track_entry["detections"].append(detection_with_tracker)

            untracked_detection_ids = [
                str(item["detection_id"])
                for item in frame_detections
                if str(item["detection_id"]) not in set(tracked_detection_ids)
            ]
            total_untracked_detections += len(untracked_detection_ids)
            frame_diagnostics.append(
                {
                    "chunk_id": chunk_id,
                    "class_name": class_name,
                    "frame_id": str(frame_detections[0]["frame_id"]),
                    "global_timestamp_seconds": round(
                        float(frame_detections[0]["global_timestamp_seconds"]),
                        3,
                    ),
                    "input_detection_count": len(frame_detections),
                    "tracked_detection_count": len(tracked_detection_ids),
                    "untracked_detection_count": len(untracked_detection_ids),
                    "assigned_tracker_ids": assigned_tracker_ids,
                    "untracked_detection_ids": untracked_detection_ids,
                }
            )

        raw_tracks.extend(track_map.values())

    if total_untracked_detections > 0:
        warnings.append(
            f"ByteTrack did not return tracker_id for {total_untracked_detections} detections in {chunk_id}; "
            "those detections were skipped instead of creating synthetic tracks."
        )

    return raw_tracks, frame_diagnostics, total_untracked_detections


def can_merge_track_fragments(
    track_a: dict[str, Any],
    track_b: dict[str, Any],
    *,
    settings: dict[str, Any],
    warnings: list[str],
) -> tuple[bool, dict[str, Any]]:
    detections_a = list(track_a["detections"])
    detections_b = list(track_b["detections"])
    end_detection = detections_a[-1]
    start_detection = detections_b[0]
    time_gap = float(start_detection["global_timestamp_seconds"]) - float(end_detection["global_timestamp_seconds"])
    if time_gap < 0 or time_gap > float(settings["fragment_max_gap_seconds"]):
        return False, {"time_gap_seconds": round(time_gap, 3), "reason": "gap_too_large"}

    end_center = [float(value) for value in end_detection["bbox_center"]]
    start_center = [float(value) for value in start_detection["bbox_center"]]
    frame_diagonal = compute_frame_diagonal(
        int(end_detection["frame_width"]),
        int(end_detection["frame_height"]),
    )
    max_center_distance = frame_diagonal * float(settings["fragment_max_center_distance_ratio"])
    center_distance = compute_center_distance(end_center, start_center)
    if center_distance > max_center_distance:
        return False, {
            "time_gap_seconds": round(time_gap, 3),
            "center_distance": round(center_distance, 3),
            "reason": "center_distance_too_large",
        }

    if len(detections_a) >= 2:
        prev_detection = detections_a[-2]
        prev_center = [float(value) for value in prev_detection["bbox_center"]]
        prev_time = float(prev_detection["global_timestamp_seconds"])
        end_time = float(end_detection["global_timestamp_seconds"])
        delta_time = end_time - prev_time
        if delta_time > 0:
            velocity_x = (end_center[0] - prev_center[0]) / delta_time
            velocity_y = (end_center[1] - prev_center[1]) / delta_time
            predicted_center = [
                end_center[0] + velocity_x * time_gap,
                end_center[1] + velocity_y * time_gap,
            ]
            predicted_distance = compute_center_distance(predicted_center, start_center)
            if predicted_distance > max_center_distance:
                return False, {
                    "time_gap_seconds": round(time_gap, 3),
                    "predicted_distance": round(predicted_distance, 3),
                    "reason": "motion_not_plausible",
                }

    crop_a = crop_bbox_from_image(
        to_absolute_repo_path(str(end_detection["image_path"])),
        list(end_detection["bbox_xyxy"]),
    )
    crop_b = crop_bbox_from_image(
        to_absolute_repo_path(str(start_detection["image_path"])),
        list(start_detection["bbox_xyxy"]),
    )
    appearance_similarity = compute_hist_similarity(crop_a, crop_b)
    if appearance_similarity < float(settings["fragment_min_appearance_sim"]):
        return False, {
            "time_gap_seconds": round(time_gap, 3),
            "appearance_similarity": round(appearance_similarity, 4),
            "reason": "appearance_too_different",
        }

    return True, {
        "time_gap_seconds": round(time_gap, 3),
        "center_distance": round(center_distance, 3),
        "appearance_similarity": round(appearance_similarity, 4),
        "reason": "merge_candidate",
    }


def merge_track_fragments_within_chunk(
    raw_tracks: list[dict[str, Any]],
    *,
    settings: dict[str, Any],
    warnings: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not settings["merge_track_fragments"]:
        return raw_tracks, []

    tracks = sorted(
        raw_tracks,
        key=lambda item: (
            str(item["chunk_id"]),
            str(item["class_name"]),
            float(item["detections"][0]["global_timestamp_seconds"]),
        ),
    )
    merged_flags = [False] * len(tracks)
    final_tracks: list[dict[str, Any]] = []
    merge_pairs: list[dict[str, Any]] = []

    for index, track in enumerate(tracks):
        if merged_flags[index]:
            continue

        current_track = {
            **track,
            "detections": list(track["detections"]),
            "source_fragment_track_ids": [f"{track['chunk_id']}_{track['class_name']}_{int(index + 1):06d}"],
        }

        for candidate_index in range(index + 1, len(tracks)):
            if merged_flags[candidate_index]:
                continue
            candidate_track = tracks[candidate_index]
            if current_track["chunk_id"] != candidate_track["chunk_id"]:
                continue
            if str(current_track["class_name"]) != str(candidate_track["class_name"]):
                continue

            should_merge, merge_reason = can_merge_track_fragments(
                current_track,
                candidate_track,
                settings=settings,
                warnings=warnings,
            )
            if not should_merge:
                continue

            current_track["detections"].extend(candidate_track["detections"])
            current_track["detections"].sort(
                key=lambda item: (
                    float(item["global_timestamp_seconds"]),
                    str(item["frame_id"]),
                    str(item["detection_id"]),
                )
            )
            current_track["source_fragment_track_ids"].append(
                f"{candidate_track['chunk_id']}_{candidate_track['class_name']}_{int(candidate_index + 1):06d}"
            )
            merged_flags[candidate_index] = True
            merge_pairs.append(
                {
                    "chunk_id": str(current_track["chunk_id"]),
                    "class_name": str(current_track["class_name"]),
                    "from_fragment": current_track["source_fragment_track_ids"][-2],
                    "to_fragment": current_track["source_fragment_track_ids"][-1],
                    "reason": merge_reason,
                }
            )

        final_tracks.append(current_track)

    return final_tracks, merge_pairs


def build_final_tracks(
    merged_tracks: list[dict[str, Any]],
    *,
    thumbnail_root: Path,
    chunk_lookup: dict[str, dict[str, Any]],
    settings: dict[str, Any],
    warnings: list[str],
) -> list[dict[str, Any]]:
    class_counters_by_chunk: dict[tuple[str, str], int] = defaultdict(int)
    final_tracks: list[dict[str, Any]] = []

    for track in sorted(
        merged_tracks,
        key=lambda item: (
            str(item["chunk_id"]),
            str(item["class_name"]),
            float(item["detections"][0]["global_timestamp_seconds"]),
        ),
    ):
        chunk_id = str(track["chunk_id"])
        class_name = str(track["class_name"])
        class_counters_by_chunk[(chunk_id, class_name)] += 1
        local_track_id = f"{chunk_id}_{class_name}_{class_counters_by_chunk[(chunk_id, class_name)]:06d}"
        chunk_info = chunk_lookup[chunk_id]
        final_tracks.append(
            build_track_output(
                track,
                local_track_id=local_track_id,
                class_name=class_name,
                thumbnail_root=thumbnail_root,
                chunk_start_time=float(chunk_info["start_time"]),
                chunk_end_time=float(chunk_info["end_time"]),
                boundary_seconds=float(settings["boundary_seconds"]),
                warnings=warnings,
            )
        )

    return final_tracks


def update_chunk_manifest_for_tracking(
    chunk_manifest: dict[str, Any],
    processed_chunk_ids: set[str],
    final_tracks: list[dict[str, Any]],
    raw_track_counts: dict[str, int],
    fragments_merged_counts: dict[str, int],
) -> dict[str, Any]:
    tracks_by_chunk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for track in final_tracks:
        tracks_by_chunk[str(track["chunk_id"])].append(track)

    for chunk in list(chunk_manifest.get("chunks", [])):
        chunk_id = str(chunk.get("chunk_id", ""))
        if chunk_id not in processed_chunk_ids:
            continue

        chunk_tracks = tracks_by_chunk.get(chunk_id, [])
        steps_completed = list(chunk.get("steps_completed", []))
        if "05_object_tracking" not in steps_completed:
            steps_completed.append("05_object_tracking")
        chunk["steps_completed"] = steps_completed
        chunk["raw_track_count_before_merge"] = int(raw_track_counts.get(chunk_id, 0))
        chunk["track_count"] = len(chunk_tracks)
        chunk["fragments_merged_count"] = int(fragments_merged_counts.get(chunk_id, 0))
        chunk["cross_chunk_candidate_count"] = sum(
            1 for item in chunk_tracks if bool(item["is_cross_chunk_candidate"])
        )
        if str(chunk.get("status", "")) == "detected":
            chunk["status"] = "tracked"

    return chunk_manifest


def update_run_manifest_for_tracking(run_manifest_path: Path) -> dict[str, Any]:
    run_manifest = read_json(run_manifest_path)
    completed_steps = list(run_manifest.get("completed_steps", []))
    if "05_object_tracking" not in completed_steps:
        completed_steps.append("05_object_tracking")
    run_manifest["completed_steps"] = completed_steps
    run_manifest["next_step"] = "06_attribute_extraction"
    write_json(run_manifest_path, run_manifest)
    return run_manifest


def build_tracking_outputs(
    *,
    run_dir: Path,
    detections_payload: dict[str, Any],
    chunk_manifest: dict[str, Any],
) -> dict[str, Any]:
    sv = load_supervision_module()
    settings = read_tracker_settings()
    frames_index_path = run_dir / "03_sampled_frames_index.json"
    frames_index_payload = read_json(frames_index_path) if frames_index_path.exists() else {}
    timing_info = extract_sample_timing_info(frames_index_payload)
    detections = list(detections_payload.get("detections", []))
    chunk_lookup = build_chunk_lookup(chunk_manifest)
    grouped_detections = group_detections_by_chunk(
        detections,
        allowed_classes=set(settings["track_classes"]),
    )
    thumbnail_root = run_dir / "05_track_thumbnails"
    thumbnail_root.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    warnings.append(
        "Tracking is performed on sampled frames. Frame skipping may increase identity fragmentation and reduce temporal continuity."
    )
    if timing_info["average_time_gap_between_frames"] > 0.5:
        warnings.append(
            f"Average time gap between sampled frames is {timing_info['average_time_gap_between_frames']} seconds; "
            "large frame gaps can reduce tracking continuity."
        )

    processed_chunk_ids: set[str] = set()
    raw_tracks_before_merge: list[dict[str, Any]] = []
    diagnostics_frames: list[dict[str, Any]] = []
    total_untracked_detections = 0
    raw_track_counts: dict[str, int] = defaultdict(int)

    if not detections:
        warnings.append("YOLO detections payload did not contain any detections.")

    for chunk_id, chunk_detections in grouped_detections.items():
        chunk_info = chunk_lookup.get(chunk_id)
        if chunk_info is None:
            warnings.append(f"Missing chunk manifest entry for detections in chunk: {chunk_id}")
            continue

        processed_chunk_ids.add(chunk_id)
        chunk_raw_tracks, chunk_frame_diagnostics, chunk_untracked = build_raw_track_fragments(
            sv,
            chunk_id=chunk_id,
            chunk_detections=chunk_detections,
            settings=settings,
            timing_info=timing_info,
            warnings=warnings,
        )
        raw_tracks_before_merge.extend(chunk_raw_tracks)
        diagnostics_frames.extend(chunk_frame_diagnostics)
        total_untracked_detections += chunk_untracked
        raw_track_counts[chunk_id] = len(chunk_raw_tracks)

    merged_tracks, merge_pairs = merge_track_fragments_within_chunk(
        raw_tracks_before_merge,
        settings=settings,
        warnings=warnings,
    )
    fragments_merged_count = len(merge_pairs)
    fragments_merged_counts: dict[str, int] = defaultdict(int)
    for pair in merge_pairs:
        fragments_merged_counts[str(pair["chunk_id"])] += 1

    final_tracks = build_final_tracks(
        merged_tracks,
        thumbnail_root=thumbnail_root,
        chunk_lookup=chunk_lookup,
        settings=settings,
        warnings=warnings,
    )

    tracks_by_class: dict[str, int] = defaultdict(int)
    tracks_by_chunk: dict[str, int] = defaultdict(int)
    cross_chunk_candidates_count = 0
    total_track_duration_seconds = 0.0
    total_track_detection_count = 0

    for track in final_tracks:
        class_name = str(track["class_name"])
        tracks_by_class[class_name] += 1
        tracks_by_chunk[str(track["chunk_id"])] += 1
        total_track_duration_seconds += float(track["duration_seconds"])
        total_track_detection_count += int(track["detection_count"])
        if bool(track["is_cross_chunk_candidate"]):
            cross_chunk_candidates_count += 1

    average_track_duration_seconds = round(
        total_track_duration_seconds / len(final_tracks),
        3,
    ) if final_tracks else 0.0
    average_detections_per_track = round(
        total_track_detection_count / len(final_tracks),
        3,
    ) if final_tracks else 0.0

    if tracks_by_class.get("person", 0) >= 8:
        warnings.append(
            "Person track count may still include fragments. Unique person count requires fragment merging and later cross-chunk/global ReID."
        )

    updated_chunk_manifest = update_chunk_manifest_for_tracking(
        chunk_manifest,
        processed_chunk_ids,
        final_tracks,
        raw_track_counts,
        fragments_merged_counts,
    )

    final_tracks.sort(
        key=lambda item: (
            int(item["chunk_index"]),
            str(item["class_name"]),
            str(item["local_track_id"]),
        )
    )

    tracks_payload = {
        "tracker_name": TRACKER_NAME,
        "tracker_type": TRACKER_TYPE,
        "total_input_detections": len(detections),
        "total_tracks_created": len(raw_tracks_before_merge),
        "total_tracks_kept": len(final_tracks),
        "total_tracks_filtered": len(raw_tracks_before_merge) - len(final_tracks),
        "created_at": current_timestamp(),
        "tracks": final_tracks,
    }

    report_payload = {
        "tracker_name": TRACKER_NAME,
        "tracker_type": TRACKER_TYPE,
        "association_mapping": "detection_id_data_mapping_not_order_based",
        "total_input_detections": len(detections),
        "total_tracked_detections": len(detections) - total_untracked_detections,
        "total_untracked_detections": total_untracked_detections,
        "raw_tracks_created_before_merge": len(raw_tracks_before_merge),
        "total_tracks_kept": len(final_tracks),
        "fragments_merged_count": fragments_merged_count,
        "tracks_by_class": dict(sorted(tracks_by_class.items())),
        "tracks_by_chunk": dict(sorted(tracks_by_chunk.items())),
        "cross_chunk_candidates_count": cross_chunk_candidates_count,
        "average_track_duration_seconds": average_track_duration_seconds,
        "average_detections_per_track": average_detections_per_track,
        "sample_fps": timing_info["sample_fps"],
        "frame_skip_ratio": timing_info["frame_skip_ratio"],
        "warnings": warnings,
        "created_at": current_timestamp(),
    }

    diagnostics_payload = {
        "tracker_name": TRACKER_NAME,
        "tracker_type": TRACKER_TYPE,
        "class_filter_used": settings["track_classes"],
        "bytetrack_settings": {
            "track_thresh": settings["track_thresh"],
            "match_thresh": settings["match_thresh"],
            "track_buffer": settings["track_buffer"],
            "min_detections": settings["min_detections"],
            "effective_track_buffer_multiplier": timing_info["frame_skip_ratio"],
        },
        "sample_fps": timing_info["sample_fps"],
        "source_fps": timing_info["source_fps"],
        "frame_skip_ratio": timing_info["frame_skip_ratio"],
        "average_time_gap_between_frames": timing_info["average_time_gap_between_frames"],
        "raw_tracks_before_merge": [
            {
                "chunk_id": str(track["chunk_id"]),
                "class_name": str(track["class_name"]),
                "raw_tracker_id": str(track["raw_tracker_id"]),
                "detection_count": len(track["detections"]),
                "start_time": round(float(track["detections"][0]["global_timestamp_seconds"]), 3),
                "end_time": round(float(track["detections"][-1]["global_timestamp_seconds"]), 3),
            }
            for track in raw_tracks_before_merge
        ],
        "final_tracks_after_merge": [
            {
                "local_track_id": str(track["local_track_id"]),
                "chunk_id": str(track["chunk_id"]),
                "class_name": str(track["class_name"]),
                "source_fragment_track_ids": list(track.get("source_fragment_track_ids", [])),
                "detection_count": int(track["detection_count"]),
                "start_time": float(track["start_time"]),
                "end_time": float(track["end_time"]),
            }
            for track in final_tracks
        ],
        "fragments_merged_count": fragments_merged_count,
        "merge_pairs": merge_pairs,
        "total_input_detections": len(detections),
        "total_frames_with_detections": len(diagnostics_frames),
        "diagnostics": diagnostics_frames,
        "created_at": current_timestamp(),
    }

    return {
        "tracks_payload": tracks_payload,
        "report_payload": report_payload,
        "diagnostics_payload": diagnostics_payload,
        "updated_chunk_manifest": updated_chunk_manifest,
    }
