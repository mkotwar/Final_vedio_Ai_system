from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from stage_checks import format_seconds_text, read_json, write_json
from step_09_search_result_packaging import write_json_any
from vehicle_color import CANONICAL_COLORS


TRAFFIC_CLASS_WHITELIST = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "traffic_light",
    "auto",
    "van",
    "vehicle",
}
IGNORED_NOISY_CLASSES = {
    "backpack",
    "bed",
    "bird",
    "boat",
    "clock",
    "suitcase",
    "tennis racket",
    "sports ball",
    "remote",
    "handbag",
    "toilet",
}
INVALID_OCR_TERMS = {"UNANSWERABLE", "STOP", "AMBULANCE", "CITYDL1FT"}
COLOR_VOCAB = set(CANONICAL_COLORS)
COLOR_NORMALIZATION = {"grey": "gray"}
CLASS_ALIASES = {
    "car": "car",
    "motorcycle": "motorcycle",
    "bike": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "person": "person",
    "pedestrian": "person",
    "bicycle": "bicycle",
    "auto": "auto",
    "van": "van",
    "vehicle": "vehicle",
    "traffic light": "traffic light",
    "traffic_light": "traffic light",
}
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "bicycle", "auto", "van", "vehicle"}
GENERIC_TRAFFIC_QUERIES = ["car", "motorcycle", "bus", "truck", "person"]
PLATE_REGEX = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$")


def normalize_class_name(value: str | None) -> str:
    normalized = str(value or "").strip().lower().replace("_", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def normalize_token(value: str | None) -> str:
    normalized = normalize_class_name(value)
    normalized = re.sub(r"[^a-z0-9:]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def normalize_color(value: str | None) -> str | None:
    token = normalize_class_name(value)
    if not token:
        return None
    token = COLOR_NORMALIZATION.get(token, token)
    return token if token in COLOR_VOCAB else None


def normalize_plate(value: str | None) -> str | None:
    text = re.sub(r"[^A-Za-z0-9]", "", str(value or "").upper())
    return text or None


def is_valid_indian_plate(value: str | None) -> bool:
    normalized = normalize_plate(value)
    if not normalized or normalized in INVALID_OCR_TERMS:
        return False
    return bool(PLATE_REGEX.fullmatch(normalized))


def path_exists(run_dir: Path, path_value: str | None) -> bool:
    resolved = resolve_run_path(run_dir, path_value)
    return resolved is not None and resolved.exists()


def resolve_run_path(run_dir: Path, path_value: str | None) -> Path | None:
    if not path_value:
        return None
    normalized = str(path_value).strip().replace("\\", "/")
    if not normalized:
        return None
    path = Path(normalized)
    if path.is_absolute():
        return path
    return (run_dir / path).resolve()


def object_type_for_class(class_name: str) -> str:
    normalized = normalize_class_name(class_name)
    if normalized == "person":
        return "person"
    if normalized in {"traffic light", "traffic_light"}:
        return "traffic_signal"
    if normalized in VEHICLE_CLASSES:
        return "vehicle"
    return "other_road_object"


def search_class_for_class(class_name: str) -> str:
    normalized = normalize_class_name(class_name)
    if normalized in {"traffic light", "traffic_light"}:
        return "traffic light"
    return normalized


def is_useful_traffic_class(class_name: str) -> bool:
    return normalize_class_name(class_name) in TRAFFIC_CLASS_WHITELIST


def is_ignored_noisy_class(class_name: str) -> bool:
    return normalize_class_name(class_name) in IGNORED_NOISY_CLASSES


def _normalize_possible_plate_text(candidate: Any) -> str | None:
    if isinstance(candidate, dict):
        value = candidate.get("text")
    else:
        value = candidate
    normalized = normalize_plate(str(value or ""))
    return normalized


def _choose_plate_fields(ocr: dict[str, Any], quality: str, object_type: str) -> dict[str, Any]:
    if object_type != "vehicle":
        return {
            "verified_license_plate": None,
            "verified_plate_status": "none",
            "plate_confidence": None,
            "plate_source": "unknown",
            "plate_warning": None,
            "rejected_plate_reason": None,
            "possible_plate_text": None,
            "weak_ocr_text": [],
        }

    crop_results = [item for item in list(ocr.get("crop_results", [])) if isinstance(item, dict)]
    possible_candidates = list(ocr.get("possible_license_plate_candidates", []))
    weak_ocr_text = [normalize_plate(item) for item in list(ocr.get("weak_ocr_text", [])) if normalize_plate(item)]

    verified_candidates: list[dict[str, Any]] = []
    rejected_count = 0
    rejected_reason = None
    for crop_result in crop_results:
        plate_text = normalize_plate(crop_result.get("verified_license_plate"))
        source = "plate_crop_ocr" if crop_result.get("plate_crop_found") else "vehicle_crop_ocr"
        plate_confidence = crop_result.get("plate_confidence")
        if (
            crop_result.get("verified_license_plate_valid") is True
            and crop_result.get("plate_crop_found") is True
            and source == "plate_crop_ocr"
            and is_valid_indian_plate(plate_text)
            and isinstance(plate_confidence, (int, float))
            and float(plate_confidence) >= 0.75
        ):
            verified_candidates.append(
                {
                    "plate": plate_text,
                    "plate_confidence": float(plate_confidence),
                    "plate_source": source,
                }
            )
        else:
            rejected_count += 1
            rejected_reason = str(crop_result.get("verified_license_plate_reason") or crop_result.get("license_plate_reject_reason") or "rejected")

    if verified_candidates:
        best = max(verified_candidates, key=lambda item: item["plate_confidence"])
        return {
            "verified_license_plate": best["plate"],
            "verified_plate_status": "verified",
            "plate_confidence": round(best["plate_confidence"], 6),
            "plate_source": best["plate_source"],
            "plate_warning": None,
            "rejected_plate_reason": None,
            "possible_plate_text": None,
            "weak_ocr_text": weak_ocr_text,
        }

    possible_plate = None
    for candidate in possible_candidates:
        normalized = _normalize_possible_plate_text(candidate)
        if is_valid_indian_plate(normalized):
            possible_plate = normalized
            break
    if possible_plate is None:
        for candidate in weak_ocr_text:
            if is_valid_indian_plate(candidate):
                possible_plate = candidate
                break

    if possible_plate:
        return {
            "verified_license_plate": None,
            "verified_plate_status": "possible",
            "plate_confidence": 0.45 if quality == "primary" else 0.25,
            "plate_source": "vehicle_crop_ocr",
            "plate_warning": "possible_plate_only_not_trusted",
            "rejected_plate_reason": rejected_reason,
            "possible_plate_text": possible_plate,
            "weak_ocr_text": weak_ocr_text,
        }

    return {
        "verified_license_plate": None,
        "verified_plate_status": "rejected" if rejected_count > 0 else "none",
        "plate_confidence": None,
        "plate_source": "unknown",
        "plate_warning": None,
        "rejected_plate_reason": rejected_reason,
        "possible_plate_text": None,
        "weak_ocr_text": weak_ocr_text,
    }


def _choose_color_fields(ocr: dict[str, Any], quality: str, object_type: str) -> dict[str, Any]:
    if object_type != "vehicle":
        return {
            "vehicle_color": None,
            "verified_vehicle_color": None,
            "possible_vehicle_color": None,
            "color_confidence": "low",
            "color_status": "unknown",
            "color_source": "unknown",
            "color_warning": None,
        }

    raw_candidates = [item for item in list(ocr.get("all_candidate_colors", [])) if isinstance(item, dict)]
    colors = []
    for candidate in raw_candidates:
        color = normalize_color(candidate.get("color"))
        if color:
            colors.append({"color": color, "source": str(candidate.get("source", "") or "unknown")})

    best_color = normalize_color(ocr.get("best_vehicle_color"))
    if best_color:
        colors.append({"color": best_color, "source": str(ocr.get("best_color_source", "") or "best_color")})

    if not colors:
        return {
            "vehicle_color": None,
            "verified_vehicle_color": None,
            "possible_vehicle_color": None,
            "color_confidence": "low",
            "color_status": "unknown",
            "color_source": "unknown",
            "color_warning": None,
        }

    color_counter = Counter(item["color"] for item in colors)
    dominant_color, dominant_count = color_counter.most_common(1)[0]
    distinct_count = len(color_counter)
    dominant_sources = sorted({item["source"] for item in colors if item["color"] == dominant_color})

    if distinct_count > 1:
        return {
            "vehicle_color": None,
            "verified_vehicle_color": None,
            "possible_vehicle_color": dominant_color,
            "color_confidence": "low",
            "color_status": "conflict",
            "color_source": ",".join(dominant_sources) or "unknown",
            "color_warning": "multiple_color_candidates_conflict",
        }

    if dominant_color == "red":
        return {
            "vehicle_color": None,
            "verified_vehicle_color": None,
            "possible_vehicle_color": "red",
            "color_confidence": "low",
            "color_status": "possible",
            "color_source": ",".join(dominant_sources) or "unknown",
            "color_warning": "possible_tail_light_color_confusion",
        }

    if quality != "primary" or dominant_count < 2:
        return {
            "vehicle_color": None,
            "verified_vehicle_color": None,
            "possible_vehicle_color": dominant_color,
            "color_confidence": "low",
            "color_status": "possible",
            "color_source": ",".join(dominant_sources) or "unknown",
            "color_warning": "single_or_fallback_color_evidence",
        }

    return {
        "vehicle_color": dominant_color,
        "verified_vehicle_color": dominant_color,
        "possible_vehicle_color": None,
        "color_confidence": "high",
        "color_status": "verified",
        "color_source": ",".join(dominant_sources) or "unknown",
        "color_warning": None,
    }


def searchable_tokens_for_record(record: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    for raw_value in [
        record.get("object_type"),
        record.get("class_name"),
        record.get("normalized_class_name"),
        record.get("search_class"),
        record.get("verified_vehicle_color"),
        record.get("possible_vehicle_color"),
        record.get("verified_license_plate"),
        record.get("possible_plate_text"),
        record.get("timestamp_text"),
        record.get("track_id"),
        *[
            value
            for key, value in dict(record.get("vehicle_attributes", {})).items()
            if key not in {"source", "confidence"} and isinstance(value, str)
        ],
        *[
            value
            for key, value in dict(record.get("scene_attributes", {})).items()
            if key not in {"source", "confidence"} and isinstance(value, str)
        ],
    ]:
        if isinstance(raw_value, list):
            candidates = raw_value
        else:
            candidates = [raw_value]
        for candidate in candidates:
            token = normalize_token(str(candidate or ""))
            if token and token not in {"not visible", "unknown", "none", "null"}:
                for part in token.split():
                    if part not in tokens:
                        tokens.append(part)
                if token not in tokens:
                    tokens.append(token)
    if record.get("verified_license_plate"):
        plate_token = normalize_plate(record["verified_license_plate"])
        if plate_token and plate_token.lower() not in tokens:
            tokens.append(plate_token.lower())
    timestamp_seconds = record.get("timestamp_seconds")
    if isinstance(timestamp_seconds, (int, float)):
        exact_seconds = str(int(round(float(timestamp_seconds))))
        if exact_seconds not in tokens:
            tokens.append(exact_seconds)
    return tokens


def build_search_text(record: dict[str, Any]) -> str:
    components: list[str] = []
    for key in [
        "class_name",
        "object_type",
        "verified_vehicle_color",
        "possible_vehicle_color",
        "verified_license_plate",
        "possible_plate_text",
        "timestamp_text",
        "vehicle_make",
        "vehicle_model",
        "vehicle_body_type",
        "vehicle_category",
    ]:
        value = record.get(key)
        if isinstance(value, list):
            text = " ".join(str(item) for item in value if str(item).strip())
        else:
            text = str(value or "").strip()
        if text and text.lower() not in {"unknown", "not_visible", "not visible", "none"}:
            components.append(text)
    return " | ".join(components)


def detection_frame_items(detections_payload: dict[str, Any]) -> list[dict[str, Any]]:
    frames = list(detections_payload.get("detections", []))
    return [item for item in frames if isinstance(item, dict)]


def flatten_detection_items(detections_payload: dict[str, Any]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for frame in detection_frame_items(detections_payload):
        base = {
            "frame_id": frame.get("frame_id"),
            "frame_idx": frame.get("frame_idx"),
            "timestamp_seconds": frame.get("timestamp_seconds"),
            "timestamp_text": frame.get("timestamp_text"),
            "image_path": frame.get("image_path"),
        }
        for detection in list(frame.get("detections", [])):
            if not isinstance(detection, dict):
                continue
            item = dict(base)
            item.update(detection)
            flattened.append(item)
    return flattened


def build_track_helpers(
    tracks_payload: dict[str, Any],
    best_frames_payload: dict[str, Any],
    ocr_payload: dict[str, Any] | None,
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    best_frame_map = {
        str(item.get("track_id", "") or ""): item
        for item in list(best_frames_payload.get("tracks", []))
        if isinstance(item, dict) and str(item.get("track_id", "") or "")
    }
    ocr_map = {
        str(item.get("track_id", "") or ""): item
        for item in list((ocr_payload or {}).get("track_results", []))
        if isinstance(item, dict) and str(item.get("track_id", "") or "")
    }
    helpers: dict[str, dict[str, Any]] = {}
    tracked_detection_ids: set[str] = set()
    for track in list(tracks_payload.get("tracks", [])):
        if not isinstance(track, dict):
            continue
        track_id = str(track.get("track_id", "") or "")
        if not track_id:
            continue
        for detection in list(track.get("detections", [])):
            if isinstance(detection, dict):
                detection_id = str(detection.get("detection_id", "") or "")
                if detection_id:
                    tracked_detection_ids.add(detection_id)
        helpers[track_id] = {
            "track": track,
            "best_frames": best_frame_map.get(track_id, {}),
            "ocr": ocr_map.get(track_id, {}),
        }
    return helpers, tracked_detection_ids


def build_traffic_index_payload(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    detections_payload = read_json(run_dir / "03_yolo_detections.json")
    tracks_payload = read_json(run_dir / "04B_tracks.json")
    best_frames_payload = read_json(run_dir / "05_best_track_frames.json")
    video_info = read_json(run_dir / "01_video_info.json")

    verified_path = run_dir / "06_ocr_color_results_verified.json"
    fallback_path = run_dir / "06_ocr_color_results.json"
    if verified_path.exists():
        ocr_payload = read_json(verified_path)
        ocr_source_file = "06_ocr_color_results_verified.json"
    elif fallback_path.exists():
        ocr_payload = read_json(fallback_path)
        ocr_source_file = "06_ocr_color_results.json"
    else:
        ocr_payload = None
        ocr_source_file = None

    track_helpers, tracked_detection_ids = build_track_helpers(tracks_payload, best_frames_payload, ocr_payload)
    flattened_detections = flatten_detection_items(detections_payload)
    frame_image_map = {
        str(item.get("frame_id", "") or ""): str(item.get("image_path", "") or "")
        for item in detection_frame_items(detections_payload)
    }

    records: list[dict[str, Any]] = []
    ignored_class_counts: Counter[str] = Counter()
    excluded_detection_ids: list[str] = []

    def append_record(record: dict[str, Any]) -> None:
        record["search_text"] = build_search_text(record)
        record["searchable_tokens"] = searchable_tokens_for_record(record)
        if not record.get("vehicle_color"):
            record["vehicle_color"] = record.get("verified_vehicle_color")
        records.append(record)

    for track_id, helper in track_helpers.items():
        track = dict(helper["track"])
        class_name = normalize_class_name(track.get("dominant_class_name"))
        if not is_useful_traffic_class(class_name):
            if is_ignored_noisy_class(class_name):
                ignored_class_counts[class_name] += 1
            continue

        best_frames = dict(helper["best_frames"])
        ocr = dict(helper["ocr"])
        selected_detections = [item for item in list(best_frames.get("selected_detections", [])) if isinstance(item, dict)]
        best_selected = selected_detections[0] if selected_detections else {}
        best_track_detection = next(
            (
                item
                for item in list(track.get("detections", []))
                if str(item.get("detection_id", "") or "") == str(track.get("best_detection_id", "") or "")
            ),
            {},
        )
        image_path = (
            best_selected.get("selected_full_frame_path")
            or best_selected.get("source_full_frame_path")
            or frame_image_map.get(str(best_track_detection.get("frame_id", "") or ""))
        )
        crop_path = best_selected.get("selected_crop_path") or best_selected.get("source_crop_path") or track.get("best_crop_path")
        timestamp_seconds = best_selected.get("timestamp_seconds")
        if timestamp_seconds is None:
            timestamp_seconds = track.get("start_timestamp_seconds")
        duration_seconds = float(track.get("duration_seconds", 0.0) or 0.0)
        confidence = float(best_selected.get("confidence", track.get("avg_confidence", 0.0)) or 0.0)
        quality = str(best_frames.get("selection_group", "") or "").strip().lower()
        if quality not in {"primary", "fallback"}:
            quality = "low_quality" if track.get("track_quality") in {"fragmented", "weak", "unknown"} else "primary"

        warnings: list[str] = []
        if not image_path:
            warnings.append("missing_full_frame_path")
        elif not path_exists(run_dir, str(image_path)):
            warnings.append("full_frame_path_missing_on_disk")
        if crop_path and not path_exists(run_dir, str(crop_path)):
            warnings.append("crop_path_missing_on_disk")
        contact_sheet_path = str(best_frames.get("contact_sheet_path", "") or "") or None
        if contact_sheet_path and not path_exists(run_dir, contact_sheet_path):
            warnings.append("contact_sheet_missing_on_disk")

        object_type = object_type_for_class(class_name)
        color_fields = _choose_color_fields(ocr, quality, object_type)
        plate_fields = _choose_plate_fields(ocr, quality, object_type)
        if color_fields.get("color_warning"):
            warnings.append(str(color_fields["color_warning"]))
        if plate_fields.get("plate_warning"):
            warnings.append(str(plate_fields["plate_warning"]))
        if plate_fields.get("rejected_plate_reason"):
            warnings.append(f"plate_rejected:{plate_fields['rejected_plate_reason']}")

        append_record(
            {
                "object_record_id": f"obj_track_{track_id}",
                "source_type": "track",
                "track_id": track_id,
                "detection_id": best_selected.get("detection_id") or track.get("best_detection_id"),
                "frame_id": best_selected.get("frame_id") or track.get("first_frame_id"),
                "object_type": object_type,
                "class_name": class_name,
                "normalized_class_name": class_name,
                "search_class": search_class_for_class(class_name),
                "timestamp_seconds": round(float(timestamp_seconds or 0.0), 6),
                "timestamp_text": str(best_selected.get("timestamp_text") or format_seconds_text(float(timestamp_seconds or 0.0))),
                "first_seen_seconds": round(float(track.get("start_timestamp_seconds", 0.0) or 0.0), 6),
                "last_seen_seconds": round(float(track.get("end_timestamp_seconds", 0.0) or 0.0), 6),
                "duration_seconds": round(duration_seconds, 6),
                "confidence": round(confidence, 6),
                "bbox_xyxy": best_selected.get("bbox_xyxy") or best_track_detection.get("bbox_xyxy"),
                "full_frame_path": image_path,
                "crop_path": crop_path,
                "contact_sheet_path": contact_sheet_path,
                **color_fields,
                **plate_fields,
                "vehicle_attributes": dict(ocr.get("vehicle_attributes", {})),
                "license_plate_attributes": dict(ocr.get("license_plate_attributes", {})),
                "scene_attributes": dict(ocr.get("scene_attributes", {})),
                "vehicle_make": dict(ocr.get("vehicle_attributes", {})).get("make"),
                "vehicle_model": dict(ocr.get("vehicle_attributes", {})).get("model"),
                "vehicle_body_type": dict(ocr.get("vehicle_attributes", {})).get("body_type"),
                "vehicle_category": dict(ocr.get("vehicle_attributes", {})).get("vehicle_category"),
                "quality": quality,
                "warnings": warnings,
            }
        )

    for detection in flattened_detections:
        class_name = normalize_class_name(detection.get("class_name"))
        detection_id = str(detection.get("detection_id", "") or "")
        if detection_id in tracked_detection_ids:
            continue
        if not is_useful_traffic_class(class_name):
            if is_ignored_noisy_class(class_name):
                ignored_class_counts[class_name] += 1
            excluded_detection_ids.append(detection_id)
            continue
        warnings: list[str] = []
        image_path = str(detection.get("image_path", "") or "") or None
        crop_path = str(detection.get("crop_path", "") or "") or None
        if not image_path:
            warnings.append("missing_full_frame_path")
        elif not path_exists(run_dir, image_path):
            warnings.append("full_frame_path_missing_on_disk")
        if crop_path and not path_exists(run_dir, crop_path):
            warnings.append("crop_path_missing_on_disk")
        confidence = float(detection.get("confidence", 0.0) or 0.0)
        quality = "single_detection" if confidence >= 0.35 else "low_quality"
        append_record(
            {
                "object_record_id": f"obj_det_{detection_id}",
                "source_type": "detection",
                "track_id": None,
                "detection_id": detection_id,
                "frame_id": detection.get("frame_id"),
                "object_type": object_type_for_class(class_name),
                "class_name": class_name,
                "normalized_class_name": class_name,
                "search_class": search_class_for_class(class_name),
                "timestamp_seconds": round(float(detection.get("timestamp_seconds", 0.0) or 0.0), 6),
                "timestamp_text": str(detection.get("timestamp_text", "") or format_seconds_text(float(detection.get("timestamp_seconds", 0.0) or 0.0))),
                "first_seen_seconds": None,
                "last_seen_seconds": None,
                "duration_seconds": None,
                "confidence": round(confidence, 6),
                "bbox_xyxy": detection.get("bbox_xyxy"),
                "full_frame_path": image_path,
                "crop_path": crop_path,
                "contact_sheet_path": None,
                "vehicle_color": None,
                "verified_vehicle_color": None,
                "possible_vehicle_color": None,
                "color_confidence": "low",
                "color_status": "unknown",
                "color_source": "unknown",
                "color_warning": None,
                "verified_license_plate": None,
                "verified_plate_status": "none",
                "plate_confidence": None,
                "plate_source": "unknown",
                "plate_warning": None,
                "rejected_plate_reason": None,
                "possible_plate_text": None,
                "weak_ocr_text": [],
                "quality": quality,
                "warnings": warnings,
            }
        )

    records.sort(
        key=lambda item: (
            float(item.get("timestamp_seconds", 0.0) or 0.0),
            str(item.get("class_name", "") or ""),
            str(item.get("object_record_id", "") or ""),
        )
    )

    class_counts = Counter(str(item.get("class_name", "") or "unknown") for item in records)
    object_type_counts = Counter(str(item.get("object_type", "") or "unknown") for item in records)
    verified_color_counts = Counter(
        str(item.get("verified_vehicle_color", "") or "unknown")
        for item in records
        if item.get("verified_vehicle_color")
    )
    quality_counts = Counter(str(item.get("quality", "") or "unknown") for item in records)
    color_status_counts = Counter(str(item.get("color_status", "") or "unknown") for item in records)
    plate_status_counts = Counter(str(item.get("verified_plate_status", "") or "unknown") for item in records)
    full_frame_counts = {
        "records_with_full_frame": sum(1 for item in records if item.get("full_frame_path")),
        "records_with_full_frame_on_disk": sum(1 for item in records if path_exists(run_dir, item.get("full_frame_path"))),
        "records_missing_full_frame": sum(1 for item in records if not item.get("full_frame_path")),
    }
    crop_counts = {
        "records_with_crop": sum(1 for item in records if item.get("crop_path")),
        "records_with_crop_on_disk": sum(1 for item in records if path_exists(run_dir, item.get("crop_path"))),
        "records_missing_crop": sum(1 for item in records if not item.get("crop_path")),
    }
    report = {
        "status": "success",
        "source_files": {
            "yolo_detections": "03_yolo_detections.json",
            "tracks": "04B_tracks.json",
            "best_track_frames": "05_best_track_frames.json",
            "ocr_color_results": ocr_source_file,
            "video_info": "01_video_info.json",
        },
        "video_name": video_info.get("video_name"),
        "duration_text": video_info.get("duration_text"),
        "total_object_records": len(records),
        "track_records": sum(1 for item in records if item.get("source_type") == "track"),
        "detection_records": sum(1 for item in records if item.get("source_type") == "detection"),
        "class_counts": dict(class_counts),
        "object_type_counts": dict(object_type_counts),
        "verified_color_counts": dict(verified_color_counts),
        "quality_counts": dict(quality_counts),
        "color_status_counts": dict(color_status_counts),
        "plate_status_counts": dict(plate_status_counts),
        "records_with_verified_color": int(color_status_counts.get("verified", 0)),
        "records_with_possible_color": int(color_status_counts.get("possible", 0)),
        "records_with_unknown_color": int(color_status_counts.get("unknown", 0)),
        "color_conflict_count": int(color_status_counts.get("conflict", 0)),
        "tail_light_confusion_warning_count": sum(
            1 for item in records if str(item.get("color_warning", "") or "") == "possible_tail_light_color_confusion"
        ),
        "records_with_verified_plate": int(plate_status_counts.get("verified", 0)),
        "records_with_possible_plate": int(plate_status_counts.get("possible", 0)),
        "rejected_plate_count": int(plate_status_counts.get("rejected", 0)),
        "full_frame_counts": full_frame_counts,
        "crop_counts": crop_counts,
        "ignored_class_counts": dict(ignored_class_counts),
        "excluded_by_whitelist_detection_count": len(excluded_detection_ids),
        "image_path_ready": full_frame_counts["records_with_full_frame_on_disk"] == full_frame_counts["records_with_full_frame"],
        "warnings_count": sum(len(list(item.get("warnings", []))) for item in records),
    }
    payload = {
        "status": "success",
        "schema_version": "v2",
        "summary": {
            "total_object_records": len(records),
            "track_records": report["track_records"],
            "detection_records": report["detection_records"],
            "records_with_verified_color": report["records_with_verified_color"],
            "records_with_verified_plate": report["records_with_verified_plate"],
            "image_path_ready": report["image_path_ready"],
        },
        "records": records,
    }
    return payload, records, report


def write_traffic_index_outputs(run_dir: Path, payload: dict[str, Any], flat_records: list[dict[str, Any]], report: dict[str, Any]) -> None:
    write_json(run_dir / "07B_traffic_object_search_index.json", payload)
    write_json_any(run_dir / "07B_traffic_object_search_index_flat.json", flat_records)
    write_json(run_dir / "07B_traffic_object_search_index_report.json", report)


def parse_timestamp_query_to_seconds(query: str) -> float | None:
    normalized = query.strip()
    if not normalized:
        return None
    if re.fullmatch(r"\d+:\d{2}", normalized):
        minutes, seconds = normalized.split(":")
        return float(int(minutes) * 60 + int(seconds))
    if re.fullmatch(r"\d+:\d{2}:\d{2}", normalized):
        hours, minutes, seconds = normalized.split(":")
        return float(int(hours) * 3600 + int(minutes) * 60 + int(seconds))
    if re.fullmatch(r"\d+(\.\d+)?", normalized):
        return float(normalized)
    return None


def parse_search_query(query: str) -> dict[str, Any]:
    raw = str(query or "").strip().lower()
    tokens = [token for token in normalize_token(raw).split() if token]
    joined = " ".join(tokens)

    class_token = None
    for alias, canonical in sorted(CLASS_ALIASES.items(), key=lambda item: -len(item[0])):
        alias_tokens = normalize_token(alias).split()
        if alias_tokens and all(token in tokens for token in alias_tokens):
            class_token = canonical
            break

    color_token = None
    for token in tokens:
        normalized = normalize_color(token)
        if normalized:
            color_token = normalized
            break

    plate_token = None
    for piece in re.split(r"\s+", raw):
        if is_valid_indian_plate(piece):
            plate_token = normalize_plate(piece)
            break

    timestamp_seconds = parse_timestamp_query_to_seconds(raw)
    fallback_tokens = [token for token in tokens if token not in {class_token, color_token}]
    return {
        "raw": query,
        "tokens": tokens,
        "class_token": class_token,
        "color_token": color_token,
        "plate_token": plate_token,
        "timestamp_seconds": timestamp_seconds,
        "fallback_tokens": fallback_tokens,
        "joined": joined,
    }


def run_traffic_search(
    records: list[dict[str, Any]],
    query: str,
    *,
    top_k: int = 20,
    time_tolerance_seconds: float = 5.0,
    require_full_frame: bool = False,
    run_dir: Path | None = None,
    include_uncertain_colors: bool = True,
    include_possible_plates: bool = False,
) -> dict[str, Any]:
    parsed = parse_search_query(query)
    if not str(parsed["raw"]).strip():
        return {"blocked": False, "query": query, "matches": [], "reason": "empty_query", "parsed_query": parsed}
    query_upper = str(query or "").strip().upper()
    if query_upper in INVALID_OCR_TERMS:
        return {"blocked": True, "query": query, "matches": [], "reason": "invalid_ocr_term", "parsed_query": parsed}

    matches: list[dict[str, Any]] = []
    for record in records:
        if require_full_frame and not record.get("full_frame_path"):
            continue
        if require_full_frame and run_dir is not None and not path_exists(run_dir, record.get("full_frame_path")):
            continue

        explanation = {
            "matched_class": False,
            "matched_verified_color": False,
            "matched_possible_color": False,
            "matched_plate": False,
            "matched_timestamp": False,
            "matched_text_fallback": False,
        }

        if parsed["class_token"]:
            if record.get("search_class") != parsed["class_token"] and record.get("class_name") != parsed["class_token"]:
                continue
            explanation["matched_class"] = True

        if parsed["color_token"]:
            verified_color = record.get("verified_vehicle_color")
            possible_color = record.get("possible_vehicle_color")
            if verified_color == parsed["color_token"]:
                explanation["matched_verified_color"] = True
            elif include_uncertain_colors and possible_color == parsed["color_token"]:
                explanation["matched_possible_color"] = True
            else:
                continue

        if parsed["plate_token"]:
            verified_plate = normalize_plate(record.get("verified_license_plate"))
            possible_plate = normalize_plate(record.get("possible_plate_text"))
            if verified_plate == parsed["plate_token"]:
                explanation["matched_plate"] = True
            elif include_possible_plates and possible_plate == parsed["plate_token"]:
                explanation["matched_plate"] = True
            else:
                continue

        if parsed["timestamp_seconds"] is not None:
            timestamp_seconds = float(record.get("timestamp_seconds", 0.0) or 0.0)
            if abs(timestamp_seconds - float(parsed["timestamp_seconds"])) <= time_tolerance_seconds:
                explanation["matched_timestamp"] = True
            else:
                continue

        if not any(explanation.values()) and parsed["fallback_tokens"]:
            search_blob = " ".join(str(token) for token in record.get("searchable_tokens", [])) + " " + str(record.get("search_text", ""))
            if all(token in search_blob.lower() for token in parsed["fallback_tokens"]):
                explanation["matched_text_fallback"] = True
            else:
                continue

        score = float(record.get("confidence", 0.0) or 0.0)
        score += 2.0 if explanation["matched_class"] else 0.0
        score += 2.0 if explanation["matched_verified_color"] else 0.0
        score += 1.5 if explanation["matched_possible_color"] else 0.0
        score += 4.0 if explanation["matched_plate"] else 0.0
        score += 1.5 if explanation["matched_timestamp"] else 0.0
        score += 0.5 if explanation["matched_text_fallback"] else 0.0
        matches.append(
            {
                "record": record,
                "score": round(score, 6),
                "match_explanation": explanation,
            }
        )

    matches.sort(
        key=lambda item: (
            -float(item.get("score", 0.0) or 0.0),
            float(item.get("record", {}).get("timestamp_seconds", 0.0) or 0.0),
        )
    )
    return {"blocked": False, "query": query, "matches": matches[:top_k], "reason": "ok", "parsed_query": parsed}


def make_demo_queries(records: list[dict[str, Any]]) -> list[str]:
    class_counts = Counter(str(item.get("class_name", "") or "unknown") for item in records)
    verified_color_counts = Counter(
        str(item.get("verified_vehicle_color", "") or "").strip()
        for item in records
        if item.get("verified_vehicle_color")
    )
    possible_color_counts = Counter(
        str(item.get("possible_vehicle_color", "") or "").strip()
        for item in records
        if item.get("possible_vehicle_color")
    )
    combo_counts = Counter(
        f"{(item.get('verified_vehicle_color') or item.get('possible_vehicle_color'))} {item.get('class_name')}"
        for item in records
        if (item.get("verified_vehicle_color") or item.get("possible_vehicle_color")) and item.get("class_name")
    )
    plates = []
    timestamps = []
    for item in records:
        plate = str(item.get("verified_license_plate", "") or "").strip()
        if plate and plate not in plates:
            plates.append(plate)
        timestamp_text = str(item.get("timestamp_text", "") or "").strip()
        if timestamp_text and timestamp_text not in timestamps:
            timestamps.append(timestamp_text)

    queries: list[str] = []
    queries.extend([item for item, _count in class_counts.most_common(3) if item])
    queries.extend([item for item, _count in verified_color_counts.most_common(3) if item])
    for item, _count in possible_color_counts.most_common(5):
        if item and item not in queries:
            queries.append(item)
    queries.extend([item for item, _count in combo_counts.most_common(3) if item])
    for color_name in ["blue", "red", "white", "black", "grey"]:
        for class_name in ["car", "motorcycle"]:
            combo = f"{color_name} {class_name}"
            if combo in queries:
                continue
            if any(
                item.get("class_name") == class_name
                and (item.get("verified_vehicle_color") == color_name or item.get("possible_vehicle_color") == color_name)
                for item in records
            ):
                queries.append(combo)
    queries.extend(plates[:5])
    queries.extend(timestamps[:3])
    for item in GENERIC_TRAFFIC_QUERIES:
        if item not in queries:
            queries.append(item)
    return queries


def maybe_load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    return read_json(path)
