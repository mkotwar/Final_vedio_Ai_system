from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stage_checks import read_json, write_json


FALLBACK_TRACK_QUALITIES = {"single_frame", "fragmented", "weak", "short", "class_mixed"}


def _clip_score(value: float) -> float:
    """Clamp a score into the 0..1 range."""

    return max(0.0, min(1.0, float(value)))


def _resolve_run_relative(run_dir: Path, path_value: str) -> Path | None:
    """Resolve a run-relative path safely."""

    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (run_dir / path).resolve()


def _relative_to_run(run_dir: Path, path: Path | None) -> str | None:
    """Convert an absolute path back to a run-relative POSIX path when possible."""

    if path is None:
        return None
    try:
        return path.relative_to(run_dir).as_posix()
    except ValueError:
        return str(path)


def _load_frame_lookup(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Build a lookup from frame_id to frame metadata for full-frame copying."""

    frame_lookup: dict[str, dict[str, Any]] = {}

    yolo_path = run_dir / "03_yolo_detections.json"
    if yolo_path.exists():
        yolo_payload = read_json(yolo_path)
        for frame_item in list(yolo_payload.get("detections", [])):
            frame_id = str(frame_item.get("frame_id", "")).strip()
            if frame_id:
                frame_lookup[frame_id] = {
                    "frame_id": frame_id,
                    "frame_idx": int(frame_item.get("frame_idx", 0) or 0),
                    "timestamp_seconds": float(frame_item.get("timestamp_seconds", 0.0) or 0.0),
                    "timestamp_text": str(frame_item.get("timestamp_text", "")),
                    "image_path": str(frame_item.get("image_path", "")),
                }

    adaptive_path = run_dir / "02A_adaptive_frames.json"
    if adaptive_path.exists():
        adaptive_payload = read_json(adaptive_path)
        for frame_item in list(adaptive_payload.get("selected_frames", [])):
            frame_id = str(frame_item.get("frame_id", "")).strip()
            if frame_id and frame_id not in frame_lookup:
                frame_lookup[frame_id] = {
                    "frame_id": frame_id,
                    "frame_idx": int(frame_item.get("frame_idx", 0) or 0),
                    "timestamp_seconds": float(frame_item.get("timestamp_seconds", 0.0) or 0.0),
                    "timestamp_text": str(frame_item.get("timestamp_text", "")),
                    "image_path": str(frame_item.get("image_path", "")),
                }

    return frame_lookup


def _temporal_position_score(index: int, total_count: int) -> float:
    """Prefer detections near the temporal middle of the track."""

    if total_count <= 1:
        return 1.0
    mid_index = (total_count - 1) / 2.0
    max_distance = max(1.0, mid_index)
    distance = abs(index - mid_index)
    return round(max(0.0, 1.0 - (distance / max_distance)), 6)


def _score_detection(
    detection: dict[str, Any],
    *,
    dominant_class_name: str,
    class_consistency_ratio: float,
    bbox_area_min: float,
    bbox_area_max: float,
    temporal_index: int,
    temporal_total: int,
) -> dict[str, Any]:
    """Compute the final Step 05 selection score for one detection."""

    confidence_score = _clip_score(float(detection.get("confidence", 0.0) or 0.0))
    bbox_area_ratio = float(detection.get("bbox_area_ratio", 0.0) or 0.0)
    if bbox_area_max > bbox_area_min:
        bbox_area_score = _clip_score((bbox_area_ratio - bbox_area_min) / (bbox_area_max - bbox_area_min))
    else:
        bbox_area_score = 1.0 if bbox_area_ratio > 0 else 0.0

    border_touching = bool(detection.get("border_touching", False))
    border_touch_ratio = _clip_score(float(detection.get("border_touch_ratio", 0.0) or 0.0))
    not_border_touching_score = 1.0 if not border_touching else max(0.0, 1.0 - border_touch_ratio)
    class_consistency_score = 1.0 if str(detection.get("class_name", "")).lower() == dominant_class_name.lower() else 0.5
    temporal_position_score = _temporal_position_score(temporal_index, temporal_total)
    crop_exists_score = 1.0 if bool(detection.get("crop_exists")) else 0.0

    final_selection_score = round(
        0.35 * confidence_score
        + 0.25 * bbox_area_score
        + 0.15 * not_border_touching_score
        + 0.10 * class_consistency_score
        + 0.10 * temporal_position_score
        + 0.05 * crop_exists_score,
        6,
    )

    scored_detection = dict(detection)
    scored_detection["source_best_frame_score"] = detection.get("best_frame_score")
    scored_detection["final_selection_score"] = final_selection_score
    scored_detection["score_parts"] = {
        "confidence_score": round(confidence_score, 6),
        "bbox_area_score": round(bbox_area_score, 6),
        "not_border_touching_score": round(not_border_touching_score, 6),
        "class_consistency_score": round(class_consistency_score, 6),
        "temporal_position_score": round(temporal_position_score, 6),
        "crop_exists_score": round(crop_exists_score, 6),
        "track_class_consistency_ratio": round(class_consistency_ratio, 6),
    }
    return scored_detection


def _copy_image_file(source_path: Path | None, destination_path: Path) -> bool:
    """Copy an image file if it exists."""

    if source_path is None or not source_path.exists() or not source_path.is_file():
        return False
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    return True


def _build_contact_sheet(
    *,
    run_dir: Path,
    track_id: str,
    selection_group: str,
    track_quality: str,
    selected_detections: list[dict[str, Any]],
    contact_sheet_output_path: Path,
) -> bool:
    """Create a simple labeled contact sheet for selected crop images."""

    if not selected_detections:
        return False

    tiles: list[np.ndarray] = []
    tile_width = 320
    tile_height = 220
    footer_height = 90

    for detection in selected_detections:
        source_crop_path = _resolve_run_relative(run_dir, str(detection.get("selected_crop_path", "") or ""))
        if source_crop_path is None or not source_crop_path.exists():
            continue
        image = cv2.imread(str(source_crop_path))
        if image is None:
            continue

        resized = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        canvas = np.full((tile_height + footer_height, tile_width, 3), 245, dtype=np.uint8)
        canvas[:tile_height, :, :] = resized

        text_lines = [
            f"{track_id} | {selection_group} | rank {detection['rank']}",
            f"t={float(detection.get('timestamp_seconds', 0.0)):.2f}s | {detection.get('class_name', '')}",
            f"conf={float(detection.get('confidence', 0.0)):.2f} | score={float(detection.get('final_selection_score', 0.0)):.2f}",
            f"quality={track_quality}",
        ]
        for line_index, text in enumerate(text_lines):
            cv2.putText(
                canvas,
                text,
                (10, tile_height + 22 + (line_index * 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
        tiles.append(canvas)

    if not tiles:
        return False

    contact_sheet = np.hstack(tiles)
    contact_sheet_output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(contact_sheet_output_path), contact_sheet))


def _selection_reason(selection_group: str, detection: dict[str, Any]) -> str:
    """Describe why a detection was kept."""

    if selection_group == "primary":
        return "primary track: high confidence, large crop, not border touching"
    if bool(detection.get("crop_exists")):
        return "fallback track: only available crop kept despite low quality"
    return "fallback track: selected as backup evidence"


def _is_primary_track(track: dict[str, Any], usable_tracks_for_next_step: set[str]) -> bool:
    """Decide whether a vehicle track belongs to the primary group."""

    if bool(track.get("usable_for_next_step")):
        return True
    if str(track.get("track_id", "")) in usable_tracks_for_next_step:
        return True
    return str(track.get("track_quality", "")) == "good"


def _prepare_track_detections(
    run_dir: Path,
    track: dict[str, Any],
    frame_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize and enrich track detections for selection."""

    detections = list(track.get("detections", []))
    bbox_area_values = [float(item.get("bbox_area_ratio", 0.0) or 0.0) for item in detections]
    bbox_area_min = min(bbox_area_values) if bbox_area_values else 0.0
    bbox_area_max = max(bbox_area_values) if bbox_area_values else 0.0
    dominant_class_name = str(track.get("dominant_class_name", "vehicle"))
    class_consistency_ratio = float(track.get("class_consistency_ratio", 0.0) or 0.0)

    prepared: list[dict[str, Any]] = []
    for index, detection in enumerate(detections):
        frame_id = str(detection.get("frame_id", ""))
        frame_item = frame_lookup.get(frame_id, {})
        detection_copy = dict(detection)
        detection_copy["frame_idx"] = int(detection_copy.get("frame_idx", frame_item.get("frame_idx", 0)) or 0)
        detection_copy["timestamp_seconds"] = float(
            detection_copy.get("timestamp_seconds", frame_item.get("timestamp_seconds", 0.0)) or 0.0
        )
        detection_copy["timestamp_text"] = str(
            detection_copy.get("timestamp_text", frame_item.get("timestamp_text", ""))
        )
        detection_copy["image_path"] = str(detection_copy.get("image_path", frame_item.get("image_path", "")))
        detection_copy["crop_path"] = str(detection_copy.get("crop_path", ""))
        crop_path = _resolve_run_relative(run_dir, detection_copy["crop_path"])
        detection_copy["crop_exists"] = bool(detection_copy.get("crop_exists")) and bool(crop_path and crop_path.exists())
        detection_copy["border_touching"] = bool(detection_copy.get("border_touching", False))
        detection_copy["border_touch_ratio"] = float(detection_copy.get("border_touch_ratio", 0.0) or 0.0)
        detection_copy["bbox_xyxy"] = [float(value) for value in list(detection_copy.get("bbox_xyxy", []))]
        prepared.append(
            _score_detection(
                detection_copy,
                dominant_class_name=dominant_class_name,
                class_consistency_ratio=class_consistency_ratio,
                bbox_area_min=bbox_area_min,
                bbox_area_max=bbox_area_max,
                temporal_index=index,
                temporal_total=len(detections),
            )
        )

    prepared.sort(
        key=lambda item: (
            float(item.get("final_selection_score", 0.0)),
            float(item.get("confidence", 0.0)),
            float(item.get("bbox_area_ratio", 0.0)),
            -float(item.get("border_touch_ratio", 0.0)),
        ),
        reverse=True,
    )
    return prepared


def _select_track_detections(
    *,
    track: dict[str, Any],
    scored_detections: list[dict[str, Any]],
    selection_group: str,
    selection_config: dict[str, Any],
    counters: Counter[str],
) -> tuple[list[dict[str, Any]], bool]:
    """Select the best detections from one vehicle track."""

    selected: list[dict[str, Any]] = []
    require_crop_exists = bool(selection_config["require_crop_exists"])
    avoid_near_duplicates = bool(selection_config["avoid_near_duplicates"])
    min_gap = float(selection_config["min_time_gap_between_selected_seconds"])
    score_floor = float(selection_config["min_primary_score"] if selection_group == "primary" else selection_config["min_fallback_score"])
    target_count = int(
        selection_config["primary_top_k_per_track"] if selection_group == "primary" else selection_config["fallback_top_k_per_track"]
    )

    for detection in scored_detections:
        if require_crop_exists and not bool(detection.get("crop_exists")):
            counters["missing_crop_count"] += 1
            continue
        if float(detection.get("final_selection_score", 0.0)) < score_floor and selection_group == "primary":
            continue
        if selection_group == "fallback" and not bool(detection.get("crop_exists")):
            counters["missing_crop_count"] += 1
            continue

        if avoid_near_duplicates and selection_group == "primary":
            too_close = any(
                abs(float(detection.get("timestamp_seconds", 0.0)) - float(existing.get("timestamp_seconds", 0.0))) < min_gap
                for existing in selected
            )
            if too_close:
                continue

        selected.append(detection)
        if len(selected) >= target_count:
            break

    if selection_group == "fallback" and not selected:
        for detection in scored_detections:
            if require_crop_exists and not bool(detection.get("crop_exists")):
                counters["missing_crop_count"] += 1
                continue
            if not bool(detection.get("crop_exists")):
                counters["missing_crop_count"] += 1
                continue
            selected.append(detection)
            break

    has_any_valid_crop = any(bool(item.get("crop_exists")) for item in scored_detections)
    return selected, has_any_valid_crop


def run_best_track_frame_selector(
    *,
    run_dir: Path,
    selection_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select primary and fallback vehicle crops for downstream OCR/color."""

    tracks_payload = read_json(run_dir / "04B_tracks.json")
    report_payload = read_json(run_dir / "04B_tracking_report.json")
    quality_report_path = run_dir / "04B_tracking_quality_report.json"
    quality_report_payload = read_json(quality_report_path) if quality_report_path.exists() else {}
    frame_lookup = _load_frame_lookup(run_dir)

    selected_crops_dir = run_dir / "05_selected_track_crops"
    selected_frames_dir = run_dir / "05_selected_track_full_frames"
    contact_sheets_dir = run_dir / "05_track_contact_sheets"
    selected_crops_dir.mkdir(parents=True, exist_ok=True)
    selected_frames_dir.mkdir(parents=True, exist_ok=True)
    contact_sheets_dir.mkdir(parents=True, exist_ok=True)

    track_types = {item.strip().lower() for item in selection_config["track_types"] if item.strip()}
    usable_tracks_for_next_step = {str(item) for item in list(tracks_payload.get("usable_tracks_for_next_step", []))}
    all_tracks = list(tracks_payload.get("tracks", []))
    filtered_tracks = [track for track in all_tracks if str(track.get("track_type", "")).lower() in track_types]
    vehicle_tracks = [track for track in filtered_tracks if str(track.get("track_type", "")).lower() == "vehicle"]

    max_tracks = int(selection_config["max_tracks"])
    if max_tracks > 0:
        vehicle_tracks = vehicle_tracks[:max_tracks]

    if not vehicle_tracks:
        no_tracks_payload = {
            "status": "no_vehicle_tracks",
            "input_tracks_file": "04B_tracks.json",
            "selection_config": selection_config,
            "vehicle_track_count_total": 0,
            "primary_vehicle_track_count": 0,
            "fallback_vehicle_track_count": 0,
            "selected_track_count": 0,
            "skipped_track_count": 0,
            "total_selected_detections": 0,
            "primary_selected_detections": 0,
            "fallback_selected_detections": 0,
            "tracks": [],
            "skipped_tracks": [],
        }
        no_tracks_report = {
            "status": "no_vehicle_tracks",
            "vehicle_track_count_total": 0,
            "primary_vehicle_track_count": 0,
            "fallback_vehicle_track_count": 0,
            "selected_track_count": 0,
            "primary_selected_track_count": 0,
            "fallback_selected_track_count": 0,
            "skipped_track_count": 0,
            "total_selected_detections": 0,
            "primary_selected_detections": 0,
            "fallback_selected_detections": 0,
            "selected_crops_saved": 0,
            "selected_full_frames_saved": 0,
            "missing_crop_count": 0,
            "missing_full_frame_count": 0,
            "contact_sheets_saved": 0,
            "skipped_no_valid_crop_count": 0,
            "skipped_track_ids_no_valid_crop": [],
            "selection_group_counts": {"primary": 0, "fallback": 0},
            "track_quality_counts_selected": {},
            "class_counts_selected": {},
            "avg_primary_selected_score": 0.0,
            "avg_fallback_selected_score": 0.0,
            "top_primary_selected_tracks": [],
            "top_fallback_selected_tracks": [],
            "recommendation": "No vehicle tracks were available for Step 05 selection.",
        }
        write_json(run_dir / "05_best_track_frames.json", no_tracks_payload)
        write_json(run_dir / "05_best_track_frames_report.json", no_tracks_report)
        return no_tracks_payload, no_tracks_report

    counters: Counter[str] = Counter()
    selected_tracks_payload: list[dict[str, Any]] = []
    skipped_tracks_payload: list[dict[str, Any]] = []
    primary_scores: list[float] = []
    fallback_scores: list[float] = []
    track_quality_counts_selected: Counter[str] = Counter()
    class_counts_selected: Counter[str] = Counter()
    primary_track_summaries: list[dict[str, Any]] = []
    fallback_track_summaries: list[dict[str, Any]] = []

    include_all_vehicle_tracks = bool(selection_config["include_all_vehicle_tracks"])
    primary_vehicle_tracks = [track for track in vehicle_tracks if _is_primary_track(track, usable_tracks_for_next_step)]
    fallback_vehicle_tracks = [
        track
        for track in vehicle_tracks
        if not _is_primary_track(track, usable_tracks_for_next_step)
        and str(track.get("track_quality", "")) in FALLBACK_TRACK_QUALITIES
    ]
    if not include_all_vehicle_tracks:
        fallback_vehicle_tracks = []

    ordered_tracks: list[tuple[dict[str, Any], str]] = [
        *( (track, "primary") for track in primary_vehicle_tracks ),
        *( (track, "fallback") for track in fallback_vehicle_tracks ),
    ]

    for track, selection_group in ordered_tracks:
        scored_detections = _prepare_track_detections(run_dir, track, frame_lookup)
        selected_detections, has_any_valid_crop = _select_track_detections(
            track=track,
            scored_detections=scored_detections,
            selection_group=selection_group,
            selection_config=selection_config,
            counters=counters,
        )
        if not has_any_valid_crop or not selected_detections:
            skipped_tracks_payload.append({"track_id": track["track_id"], "reason": "no_valid_crop"})
            counters["skipped_no_valid_crop_count"] += 1
            continue

        selection_ranked_payloads: list[dict[str, Any]] = []
        for rank, detection in enumerate(selected_detections, start=1):
            class_name = str(detection.get("class_name", "vehicle")).lower()
            crop_name = (
                f"{selection_group}_{track['track_id']}_rank{rank:02d}_{detection['frame_id']}_"
                f"{detection['detection_id']}_{class_name}.jpg"
            )
            full_frame_name = f"{selection_group}_{track['track_id']}_rank{rank:02d}_{detection['frame_id']}_full.jpg"

            source_crop_path = _resolve_run_relative(run_dir, str(detection.get("crop_path", "")))
            selected_crop_output_path = selected_crops_dir / crop_name
            crop_copied = False
            if bool(selection_config["save_selected_crops"]):
                crop_copied = _copy_image_file(source_crop_path, selected_crop_output_path)
                if crop_copied:
                    counters["selected_crops_saved"] += 1
            if not crop_copied and not bool(detection.get("crop_exists")):
                counters["missing_crop_count"] += 1

            source_full_frame_path = _resolve_run_relative(run_dir, str(detection.get("image_path", "")))
            selected_full_frame_output_path = selected_frames_dir / full_frame_name
            full_frame_copied = False
            if bool(selection_config["save_selected_full_frames"]):
                full_frame_copied = _copy_image_file(source_full_frame_path, selected_full_frame_output_path)
                if full_frame_copied:
                    counters["selected_full_frames_saved"] += 1
                else:
                    counters["missing_full_frame_count"] += 1

            final_detection_payload = {
                "rank": rank,
                "detection_id": detection["detection_id"],
                "frame_id": detection["frame_id"],
                "frame_idx": int(detection.get("frame_idx", 0) or 0),
                "timestamp_seconds": float(detection.get("timestamp_seconds", 0.0) or 0.0),
                "timestamp_text": str(detection.get("timestamp_text", "")),
                "class_name": class_name,
                "confidence": round(float(detection.get("confidence", 0.0) or 0.0), 6),
                "bbox_xyxy": [round(float(value), 3) for value in list(detection.get("bbox_xyxy", []))],
                "bbox_area_ratio": round(float(detection.get("bbox_area_ratio", 0.0) or 0.0), 6),
                "border_touching": bool(detection.get("border_touching", False)),
                "border_touch_ratio": round(float(detection.get("border_touch_ratio", 0.0) or 0.0), 6),
                "source_crop_path": str(detection.get("crop_path", "")),
                "selected_crop_path": _relative_to_run(run_dir, selected_crop_output_path if crop_copied else None),
                "source_full_frame_path": str(detection.get("image_path", "")),
                "selected_full_frame_path": _relative_to_run(run_dir, selected_full_frame_output_path if full_frame_copied else None),
                "source_best_frame_score": detection.get("source_best_frame_score"),
                "final_selection_score": round(float(detection.get("final_selection_score", 0.0) or 0.0), 6),
                "selection_group": selection_group,
                "quality_label": "good" if selection_group == "primary" else "low_quality",
                "selection_reason": _selection_reason(selection_group, detection),
            }
            selection_ranked_payloads.append(final_detection_payload)
            class_counts_selected[class_name] += 1
            if selection_group == "primary":
                primary_scores.append(final_detection_payload["final_selection_score"])
            else:
                fallback_scores.append(final_detection_payload["final_selection_score"])

        contact_sheet_path = contact_sheets_dir / f"{track['track_id']}_contact_sheet.jpg"
        contact_sheet_written = False
        if bool(selection_config["save_contact_sheets"]):
            contact_sheet_written = _build_contact_sheet(
                run_dir=run_dir,
                track_id=str(track["track_id"]),
                selection_group=selection_group,
                track_quality=str(track.get("track_quality", "")),
                selected_detections=selection_ranked_payloads,
                contact_sheet_output_path=contact_sheet_path,
            )
            if contact_sheet_written:
                counters["contact_sheets_saved"] += 1

        track_payload = {
            "track_id": track["track_id"],
            "selection_group": selection_group,
            "quality_label": "good" if selection_group == "primary" else "low_quality",
            "track_type": track["track_type"],
            "dominant_class_name": track["dominant_class_name"],
            "class_consistency_ratio": round(float(track.get("class_consistency_ratio", 0.0) or 0.0), 6),
            "track_quality": track["track_quality"],
            "detection_count": int(track.get("detection_count", 0) or 0),
            "duration_seconds": round(float(track.get("duration_seconds", 0.0) or 0.0), 6),
            "selected_count": len(selection_ranked_payloads),
            "best_selected_detection_id": selection_ranked_payloads[0]["detection_id"],
            "best_selected_crop_path": selection_ranked_payloads[0]["selected_crop_path"],
            "contact_sheet_path": _relative_to_run(run_dir, contact_sheet_path if contact_sheet_written else None),
            "selected_detections": selection_ranked_payloads,
        }
        selected_tracks_payload.append(track_payload)
        track_quality_counts_selected[str(track.get("track_quality", ""))] += 1

        summary_item = {
            "track_id": track["track_id"],
            "dominant_class_name": track["dominant_class_name"],
            "best_score": selection_ranked_payloads[0]["final_selection_score"],
            "selected_count": len(selection_ranked_payloads),
        }
        if selection_group == "primary":
            primary_track_summaries.append(summary_item)
        else:
            fallback_track_summaries.append(
                {**summary_item, "track_quality": track["track_quality"]}
            )

    selected_tracks_payload.sort(key=lambda item: (item["selection_group"] != "primary", item["track_id"]))
    selected_track_count = len(selected_tracks_payload)
    primary_selected_track_count = sum(1 for item in selected_tracks_payload if item["selection_group"] == "primary")
    fallback_selected_track_count = sum(1 for item in selected_tracks_payload if item["selection_group"] == "fallback")
    primary_selected_detections = sum(item["selected_count"] for item in selected_tracks_payload if item["selection_group"] == "primary")
    fallback_selected_detections = sum(item["selected_count"] for item in selected_tracks_payload if item["selection_group"] == "fallback")
    total_selected_detections = primary_selected_detections + fallback_selected_detections

    selection_payload = {
        "status": "success",
        "input_tracks_file": "04B_tracks.json",
        "selection_config": {
            "track_types": sorted(track_types),
            "include_all_vehicle_tracks": include_all_vehicle_tracks,
            "primary_top_k_per_track": int(selection_config["primary_top_k_per_track"]),
            "fallback_top_k_per_track": int(selection_config["fallback_top_k_per_track"]),
            "min_primary_score": float(selection_config["min_primary_score"]),
            "min_fallback_score": float(selection_config["min_fallback_score"]),
            "require_crop_exists": bool(selection_config["require_crop_exists"]),
            "save_selected_crops": bool(selection_config["save_selected_crops"]),
            "save_selected_full_frames": bool(selection_config["save_selected_full_frames"]),
            "save_contact_sheets": bool(selection_config["save_contact_sheets"]),
            "max_tracks": int(selection_config["max_tracks"]),
            "avoid_near_duplicates": bool(selection_config["avoid_near_duplicates"]),
            "min_time_gap_between_selected_seconds": float(selection_config["min_time_gap_between_selected_seconds"]),
        },
        "vehicle_track_count_total": len(vehicle_tracks),
        "primary_vehicle_track_count": len(primary_vehicle_tracks),
        "fallback_vehicle_track_count": len(fallback_vehicle_tracks),
        "selected_track_count": selected_track_count,
        "skipped_track_count": len(skipped_tracks_payload),
        "total_selected_detections": total_selected_detections,
        "primary_selected_detections": primary_selected_detections,
        "fallback_selected_detections": fallback_selected_detections,
        "tracks": selected_tracks_payload,
        "skipped_tracks": skipped_tracks_payload,
    }

    selection_report = {
        "status": "success",
        "source_tracking_status": str(report_payload.get("status", "unknown")),
        "source_tracking_quality_status": str(quality_report_payload.get("status", "unknown")) if quality_report_payload else "missing",
        "vehicle_track_count_total": len(vehicle_tracks),
        "primary_vehicle_track_count": len(primary_vehicle_tracks),
        "fallback_vehicle_track_count": len(fallback_vehicle_tracks),
        "selected_track_count": selected_track_count,
        "primary_selected_track_count": primary_selected_track_count,
        "fallback_selected_track_count": fallback_selected_track_count,
        "skipped_track_count": len(skipped_tracks_payload),
        "total_selected_detections": total_selected_detections,
        "primary_selected_detections": primary_selected_detections,
        "fallback_selected_detections": fallback_selected_detections,
        "selected_crops_saved": counters["selected_crops_saved"],
        "selected_full_frames_saved": counters["selected_full_frames_saved"],
        "missing_crop_count": counters["missing_crop_count"],
        "missing_full_frame_count": counters["missing_full_frame_count"],
        "contact_sheets_saved": counters["contact_sheets_saved"],
        "skipped_no_valid_crop_count": counters["skipped_no_valid_crop_count"],
        "skipped_track_ids_no_valid_crop": [item["track_id"] for item in skipped_tracks_payload if item["reason"] == "no_valid_crop"],
        "selection_group_counts": {
            "primary": primary_selected_track_count,
            "fallback": fallback_selected_track_count,
        },
        "track_quality_counts_selected": dict(sorted(track_quality_counts_selected.items())),
        "class_counts_selected": dict(sorted(class_counts_selected.items())),
        "avg_primary_selected_score": round(sum(primary_scores) / len(primary_scores), 6) if primary_scores else 0.0,
        "avg_fallback_selected_score": round(sum(fallback_scores) / len(fallback_scores), 6) if fallback_scores else 0.0,
        "top_primary_selected_tracks": sorted(primary_track_summaries, key=lambda item: item["best_score"], reverse=True)[:10],
        "top_fallback_selected_tracks": sorted(fallback_track_summaries, key=lambda item: item["best_score"], reverse=True)[:10],
        "recommendation": (
            "Proceed to Step 06 OCR/color. Run OCR/color on primary tracks first, then fallback tracks if time allows."
            if total_selected_detections > 0
            else "No usable crops were copied in Step 05. Review crop availability in Step 03 and track outputs in Step 04B."
        ),
    }

    write_json(run_dir / "05_best_track_frames.json", selection_payload)
    write_json(run_dir / "05_best_track_frames_report.json", selection_report)
    return selection_payload, selection_report
