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

DEFAULT_BYTETRACK_TRACK_THRESH = 0.25
DEFAULT_BYTETRACK_MATCH_THRESH = 0.80
DEFAULT_BYTETRACK_TRACK_BUFFER = 30
DEFAULT_TRACK_MIN_DETECTIONS = 2
DEFAULT_TRACK_BOUNDARY_SECONDS = 10.0

# This ByteTrack wrapper is still a demo tracker adapter. For production,
# consider validating configuration and long-term ReID behavior explicitly.
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


def read_non_negative_float_env(env_name: str, default_value: float) -> float:
    value = read_float_env(env_name, default_value)
    if value < 0:
        raise ValueError(
            f"Environment variable {env_name} must be greater than or equal to 0. Received: {value}"
        )
    return value


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


def infer_frame_rate(chunk_detections: list[dict[str, Any]]) -> int:
    timestamps = sorted(
        {
            round(float(item["global_timestamp_seconds"]), 3)
            for item in chunk_detections
        }
    )
    if len(timestamps) < 2:
        return 30

    deltas = [
        round(current - previous, 3)
        for previous, current in zip(timestamps, timestamps[1:])
        if current > previous
    ]
    if not deltas:
        return 30

    median_delta = sorted(deltas)[len(deltas) // 2]
    if median_delta <= 0:
        return 30
    return max(1, min(120, int(round(1.0 / median_delta))))


def create_bytetrack_instance(
    sv: Any,
    *,
    settings: dict[str, Any],
    frame_rate: int,
) -> Any:
    signature = inspect.signature(sv.ByteTrack)
    kwargs: dict[str, Any] = {}

    parameter_aliases = {
        "track_thresh": ["track_activation_threshold", "track_thresh", "track_threshold"],
        "match_thresh": [
            "minimum_matching_threshold",
            "match_thresh",
            "match_threshold",
        ],
        "track_buffer": ["lost_track_buffer", "track_buffer", "buffer_size"],
        "frame_rate": ["frame_rate", "fps"],
    }

    configured_values = {
        "track_thresh": settings["track_thresh"],
        "match_thresh": settings["match_thresh"],
        "track_buffer": settings["track_buffer"],
        "frame_rate": frame_rate,
    }

    for logical_name, aliases in parameter_aliases.items():
        for alias in aliases:
            if alias in signature.parameters:
                kwargs[alias] = configured_values[logical_name]
                break

    return sv.ByteTrack(**kwargs)


def compute_bbox_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def compute_bbox_iou(box_a: list[float], box_b: list[float] | None) -> float:
    if box_b is None:
        return 0.0

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


def compute_bbox_l1_distance(box_a: list[float], box_b: list[float] | None) -> float:
    if box_b is None:
        return float("inf")
    return sum(abs(float(a) - float(b)) for a, b in zip(box_a, box_b))


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


def build_chunk_lookup(chunk_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(chunk["chunk_id"]): chunk
        for chunk in list(chunk_manifest.get("chunks", []))
        if isinstance(chunk, dict) and "chunk_id" in chunk
    }


def group_detections_by_chunk(detections: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detection in detections:
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

    return sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
    )


def read_tracker_ids_from_detections(tracked_detections: Any) -> list[Any]:
    tracker_ids = getattr(tracked_detections, "tracker_id", None)
    if tracker_ids is None:
        return []
    if hasattr(tracker_ids, "tolist"):
        return tracker_ids.tolist()
    return list(tracker_ids)


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


def read_array_like(values: Any) -> list[Any]:
    if values is None:
        return []
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


def score_tracked_item_match(
    detection: dict[str, Any],
    tracked_item: dict[str, Any],
) -> tuple[float, float, float]:
    detection_bbox = [float(value) for value in detection["bbox_xyxy"]]
    tracked_bbox = tracked_item["bbox_xyxy"]
    iou_score = compute_bbox_iou(detection_bbox, tracked_bbox)
    confidence_gap = abs((tracked_item["confidence"] or 0.0) - float(detection["confidence"]))
    bbox_l1_distance = compute_bbox_l1_distance(detection_bbox, tracked_bbox)
    return (iou_score, -bbox_l1_distance, -confidence_gap)


def align_tracker_ids_to_input_detections(
    frame_detections: list[dict[str, Any]],
    tracked_detections: Any,
) -> list[str | None]:
    tracker_ids = [normalize_tracker_id(item) for item in read_tracker_ids_from_detections(tracked_detections)]
    tracked_xyxy = read_array_like(getattr(tracked_detections, "xyxy", None))
    tracked_confidence = read_array_like(getattr(tracked_detections, "confidence", None))
    tracked_class_id = read_array_like(getattr(tracked_detections, "class_id", None))

    if (
        len(tracker_ids) == len(frame_detections)
        and len(tracked_xyxy) != len(frame_detections)
    ):
        return tracker_ids

    if len(tracker_ids) == len(frame_detections) and len(tracked_xyxy) == len(frame_detections):
        ordered_match = True
        for detection, tracked_bbox, tracked_class_id_value in zip(
            frame_detections,
            tracked_xyxy,
            tracked_class_id,
        ):
            if tracked_bbox is None:
                ordered_match = False
                break
            detection_bbox = [float(value) for value in detection["bbox_xyxy"]]
            tracked_bbox_values = [float(value) for value in tracked_bbox]
            if int(detection["class_id"]) != int(tracked_class_id_value):
                ordered_match = False
                break
            if compute_bbox_iou(detection_bbox, tracked_bbox_values) < 0.98:
                ordered_match = False
                break
        if ordered_match:
            return tracker_ids

    tracked_items: list[dict[str, Any]] = []
    for index, tracker_id in enumerate(tracker_ids):
        bbox_xyxy = tracked_xyxy[index] if index < len(tracked_xyxy) else None
        confidence = tracked_confidence[index] if index < len(tracked_confidence) else None
        class_id = tracked_class_id[index] if index < len(tracked_class_id) else None
        tracked_items.append(
            {
                "tracker_id": tracker_id,
                "bbox_xyxy": [float(value) for value in bbox_xyxy] if bbox_xyxy is not None else None,
                "confidence": float(confidence) if confidence is not None else None,
                "class_id": int(class_id) if class_id is not None else None,
            }
        )

    aligned_tracker_ids: list[str | None] = [None] * len(frame_detections)
    used_tracked_indices: set[int] = set()

    for input_index, detection in enumerate(frame_detections):
        detection_class_id = int(detection["class_id"])

        best_index: int | None = None
        best_score: tuple[float, float] | None = None
        second_best_score: tuple[float, float] | None = None
        for tracked_index, tracked_item in enumerate(tracked_items):
            if tracked_index in used_tracked_indices:
                continue
            if tracked_item["tracker_id"] is None:
                continue
            if tracked_item["class_id"] is not None and tracked_item["class_id"] != detection_class_id:
                continue

            score = score_tracked_item_match(detection, tracked_item)
            if best_score is None or score > best_score:
                second_best_score = best_score
                best_score = score
                best_index = tracked_index
            elif second_best_score is None or score > second_best_score:
                second_best_score = score

        if best_index is None:
            continue

        best_iou = best_score[0] if best_score is not None else 0.0
        second_best_iou = second_best_score[0] if second_best_score is not None else -1.0
        # Be conservative here: it is better to leave a detection unmatched than
        # to attach the wrong tracker_id and fragment one real person into many tracks.
        if best_iou < 0.85:
            continue
        if second_best_score is not None and abs(best_iou - second_best_iou) < 0.05:
            continue

        used_tracked_indices.add(best_index)
        aligned_tracker_ids[input_index] = tracked_items[best_index]["tracker_id"]

    if all(item is None for item in aligned_tracker_ids) and len(tracker_ids) == len(frame_detections):
        return tracker_ids
    return aligned_tracker_ids


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

    return {
        "local_track_id": local_track_id,
        "raw_tracker_id": str(track["raw_tracker_id"]),
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


def track_chunk_with_bytetrack(
    sv: Any,
    *,
    chunk_id: str,
    chunk_detections: list[dict[str, Any]],
    chunk_info: dict[str, Any],
    settings: dict[str, Any],
    thumbnail_root: Path,
    warnings: list[str],
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    track_map: dict[str, dict[str, Any]] = {}
    detections_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    frame_diagnostics: list[dict[str, Any]] = []
    for detection in chunk_detections:
        detections_by_class[str(detection["class_name"])].append(detection)

    total_untracked_detections = 0

    for class_name, class_detections in sorted(detections_by_class.items()):
        frame_rate = infer_frame_rate(class_detections)
        tracker = create_bytetrack_instance(
            sv,
            settings=settings,
            frame_rate=frame_rate,
        )

        for frame_detections in group_detections_by_frame(class_detections):
            supervision_detections = build_supervision_detections(sv, frame_detections)
            tracked_detections = tracker.update_with_detections(supervision_detections)
            tracker_ids = align_tracker_ids_to_input_detections(frame_detections, tracked_detections)
            assigned_tracker_ids: list[str] = []
            untracked_detection_ids: list[str] = []

            for index, detection in enumerate(frame_detections):
                raw_tracker_id = None
                if index < len(tracker_ids):
                    raw_tracker_id = normalize_tracker_id(tracker_ids[index])

                if raw_tracker_id is None:
                    total_untracked_detections += 1
                    untracked_detection_ids.append(str(detection["detection_id"]))
                    continue

                assigned_tracker_ids.append(str(raw_tracker_id))
                track_key = f"{class_name}:{raw_tracker_id}"
                track_entry = track_map.get(track_key)
                if track_entry is None:
                    track_entry = {
                        "raw_tracker_id": raw_tracker_id,
                        "chunk_id": chunk_id,
                        "chunk_index": int(detection["chunk_index"]),
                        "detections": [],
                    }
                    track_map[track_key] = track_entry

                track_entry["detections"].append(detection)

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
                    "tracked_detection_count": len(assigned_tracker_ids),
                    "untracked_detection_count": len(untracked_detection_ids),
                    "assigned_tracker_ids": assigned_tracker_ids,
                    "untracked_detection_ids": untracked_detection_ids,
                }
            )

    if total_untracked_detections > 0:
        warnings.append(
            f"ByteTrack did not return tracker_id for {total_untracked_detections} detections in {chunk_id}; "
            "those detections were skipped instead of creating synthetic tracks."
        )

    total_tracks_created = len(track_map)
    kept_tracks: list[dict[str, Any]] = []
    class_counters: dict[str, int] = defaultdict(int)

    for raw_tracker_id, track in sorted(track_map.items(), key=lambda item: str(item[0])):
        detections = list(track["detections"])
        if len(detections) < int(settings["min_detections"]):
            continue

        class_counts = Counter(str(item["class_name"]) for item in detections)
        majority_class_name, _majority_count = class_counts.most_common(1)[0]
        if len(class_counts) > 1:
            warnings.append(
                f"Chunk {chunk_id} tracker_id {raw_tracker_id} had mixed classes {dict(class_counts)}; "
                f"using majority class {majority_class_name!r}."
            )

        class_counters[majority_class_name] += 1
        local_track_id = f"{chunk_id}_{majority_class_name}_{class_counters[majority_class_name]:06d}"
        kept_tracks.append(
            build_track_output(
                track,
                local_track_id=local_track_id,
                class_name=majority_class_name,
                thumbnail_root=thumbnail_root,
                chunk_start_time=float(chunk_info["start_time"]),
                chunk_end_time=float(chunk_info["end_time"]),
                boundary_seconds=float(settings["boundary_seconds"]),
                warnings=warnings,
            )
        )

    return kept_tracks, total_tracks_created, frame_diagnostics


def update_chunk_manifest_for_tracking(
    chunk_manifest: dict[str, Any],
    processed_chunk_ids: set[str],
    tracks: list[dict[str, Any]],
) -> dict[str, Any]:
    tracks_by_chunk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for track in tracks:
        tracks_by_chunk[str(track["chunk_id"])].append(track)

    for chunk in list(chunk_manifest.get("chunks", [])):
        chunk_id = str(chunk.get("chunk_id", ""))
        if chunk_id not in processed_chunk_ids:
            continue

        chunk_tracks = tracks_by_chunk[chunk_id]
        steps_completed = list(chunk.get("steps_completed", []))
        if "05_object_tracking" not in steps_completed:
            steps_completed.append("05_object_tracking")

        chunk["steps_completed"] = steps_completed
        chunk["track_count"] = len(chunk_tracks)
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
    detections = list(detections_payload.get("detections", []))
    chunk_lookup = build_chunk_lookup(chunk_manifest)
    grouped_detections = group_detections_by_chunk(detections)
    thumbnail_root = run_dir / "05_track_thumbnails"
    thumbnail_root.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    kept_tracks: list[dict[str, Any]] = []
    total_tracks_created = 0
    processed_chunk_ids: set[str] = set()
    tracking_diagnostics: list[dict[str, Any]] = []

    if not detections:
        warnings.append("YOLO detections payload did not contain any detections.")

    for chunk_id, chunk_detections in grouped_detections.items():
        chunk_info = chunk_lookup.get(chunk_id)
        if chunk_info is None:
            warnings.append(f"Missing chunk manifest entry for detections in chunk: {chunk_id}")
            continue

        processed_chunk_ids.add(chunk_id)
        chunk_tracks, chunk_created_count, chunk_frame_diagnostics = track_chunk_with_bytetrack(
            sv,
            chunk_id=chunk_id,
            chunk_detections=chunk_detections,
            chunk_info=chunk_info,
            settings=settings,
            thumbnail_root=thumbnail_root,
            warnings=warnings,
        )
        kept_tracks.extend(chunk_tracks)
        total_tracks_created += chunk_created_count
        tracking_diagnostics.extend(chunk_frame_diagnostics)

    kept_tracks.sort(
        key=lambda item: (
            int(item["chunk_index"]),
            str(item["class_name"]),
            str(item["local_track_id"]),
        )
    )

    total_tracks_kept = len(kept_tracks)
    total_tracks_filtered = total_tracks_created - total_tracks_kept
    tracks_by_class: dict[str, int] = defaultdict(int)
    tracks_by_chunk: dict[str, int] = defaultdict(int)
    cross_chunk_candidates_count = 0
    total_track_duration_seconds = 0.0
    total_track_detection_count = 0

    for track in kept_tracks:
        tracks_by_class[str(track["class_name"])] += 1
        tracks_by_chunk[str(track["chunk_id"])] += 1
        total_track_duration_seconds += float(track["duration_seconds"])
        total_track_detection_count += int(track["detection_count"])
        if bool(track["is_cross_chunk_candidate"]):
            cross_chunk_candidates_count += 1

    average_track_duration_seconds = round(
        total_track_duration_seconds / total_tracks_kept,
        3,
    ) if total_tracks_kept > 0 else 0.0
    average_detections_per_track = round(
        total_track_detection_count / total_tracks_kept,
        3,
    ) if total_tracks_kept > 0 else 0.0

    updated_chunk_manifest = update_chunk_manifest_for_tracking(
        chunk_manifest,
        processed_chunk_ids,
        kept_tracks,
    )

    tracks_payload = {
        "tracker_name": TRACKER_NAME,
        "tracker_type": TRACKER_TYPE,
        "total_input_detections": len(detections),
        "total_tracks_created": total_tracks_created,
        "total_tracks_kept": total_tracks_kept,
        "total_tracks_filtered": total_tracks_filtered,
        "created_at": current_timestamp(),
        "tracks": kept_tracks,
    }

    report_payload = {
        "tracker_name": TRACKER_NAME,
        "total_input_detections": len(detections),
        "total_tracks_created": total_tracks_created,
        "total_tracks_kept": total_tracks_kept,
        "total_tracks_filtered": total_tracks_filtered,
        "tracks_by_class": dict(sorted(tracks_by_class.items())),
        "tracks_by_chunk": dict(sorted(tracks_by_chunk.items())),
        "cross_chunk_candidates_count": cross_chunk_candidates_count,
        "average_track_duration_seconds": average_track_duration_seconds,
        "average_detections_per_track": average_detections_per_track,
        "warnings": warnings,
        "created_at": current_timestamp(),
    }

    return {
        "tracks_payload": tracks_payload,
        "report_payload": report_payload,
        "diagnostics_payload": {
            "tracker_name": TRACKER_NAME,
            "total_input_detections": len(detections),
            "total_frames_with_detections": len(tracking_diagnostics),
            "diagnostics": tracking_diagnostics,
            "created_at": current_timestamp(),
        },
        "updated_chunk_manifest": updated_chunk_manifest,
    }
