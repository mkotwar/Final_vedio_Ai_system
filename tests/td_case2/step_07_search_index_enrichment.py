from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from stage_checks import format_seconds_text, read_json, write_json


CLASS_NORMALIZATION = {
    "car": "car",
    "motorcycle": "motorcycle",
    "bike": "motorcycle",
    "truck": "truck",
    "bus": "bus",
    "vehicle": "vehicle",
    "van": "vehicle",
    "auto": "vehicle",
}
INVALID_SEARCH_TOKENS = {
    "UNANSWERABLE",
    "ANSWERABLE",
    "UNKNOWN",
    "NONE",
    "NULL",
    "NIL",
    "STOP",
    "VICTORY",
    "DEV",
    "LANCERAIL",
    "AIRAMBULANCE",
    "LANCERAILAIRAMBULANCE",
}
SHORT_ALLOWED_TERMS = {"car", "bus", "red", "blue"}


def _resolve_run_relative(run_dir: Path, path_value: str | None) -> Path | None:
    """Resolve a run-relative path safely."""

    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (run_dir / path).resolve()


def _relative_to_run(run_dir: Path, path: Path | None) -> str | None:
    """Convert an absolute path back to run-relative POSIX when possible."""

    if path is None:
        return None
    try:
        return path.relative_to(run_dir).as_posix()
    except ValueError:
        return str(path)


def _safe_stats(values: list[float]) -> dict[str, float]:
    """Return min/max/avg stats with zero defaults."""

    if not values:
        return {"min": 0.0, "max": 0.0, "avg": 0.0}
    return {
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "avg": round(sum(values) / len(values), 6),
    }


def is_invalid_search_token(token: str) -> bool:
    """Reject weak or invalid tokens from search_text."""

    normalized = str(token or "").strip()
    if not normalized:
        return True
    upper_value = normalized.upper()
    lower_value = normalized.lower()
    if upper_value in INVALID_SEARCH_TOKENS:
        return True
    if normalized.isdigit():
        return True
    if len(normalized) < 4 and lower_value not in SHORT_ALLOWED_TERMS:
        return True
    return False


def _normalize_vehicle_class(track_result: dict[str, Any], best_crop: dict[str, Any] | None) -> str:
    """Normalize vehicle class names into a small stable set."""

    dominant_class = str(track_result.get("dominant_class_name", "") or "").lower()
    if dominant_class in CLASS_NORMALIZATION:
        return CLASS_NORMALIZATION[dominant_class]
    if best_crop is not None:
        best_class = str(best_crop.get("class_name", "") or "").lower()
        if best_class in CLASS_NORMALIZATION:
            return CLASS_NORMALIZATION[best_class]
    return "vehicle"


def _pick_best_crop(track_result: dict[str, Any]) -> dict[str, Any] | None:
    """Choose the best crop according to verified plate, color, or score priority."""

    crop_results = list(track_result.get("crop_results", []))
    if not crop_results:
        return None

    verified_matches = [
        crop
        for crop in crop_results
        if bool(crop.get("verified_license_plate_valid"))
        and str(crop.get("verified_license_plate", "not_visible")) == str(track_result.get("verified_license_plate", "not_visible"))
    ]
    if verified_matches:
        return sorted(
            verified_matches,
            key=lambda item: (
                -float(item.get("plate_confidence", 0.0) or 0.0),
                -float(item.get("final_selection_score", 0.0) or 0.0),
            ),
        )[0]

    color_matches = [
        crop
        for crop in crop_results
        if str(crop.get("parsed_vehicle_color", "unknown")) == str(track_result.get("best_vehicle_color", "unknown"))
        and str(crop.get("parsed_vehicle_color", "unknown")) != "unknown"
    ]
    if color_matches:
        return sorted(color_matches, key=lambda item: -float(item.get("final_selection_score", 0.0) or 0.0))[0]

    return sorted(crop_results, key=lambda item: -float(item.get("final_selection_score", 0.0) or 0.0))[0]


def _find_step05_track(track_id: str, best_frames_payload: dict[str, Any]) -> dict[str, Any]:
    """Look up Step 05 track payload by track id."""

    for track_item in list(best_frames_payload.get("tracks", [])):
        if str(track_item.get("track_id", "")) == track_id:
            return track_item
    return {}


def _selected_detection_matches_crop(selected_detection: dict[str, Any], crop_item: dict[str, Any]) -> bool:
    """Match a Step 05 selected detection to a Step 06 crop result."""

    if str(selected_detection.get("selected_crop_path", "")) and str(crop_item.get("selected_crop_path", "")):
        if str(selected_detection.get("selected_crop_path", "")) == str(crop_item.get("selected_crop_path", "")):
            return True
    return (
        str(selected_detection.get("detection_id", "")) == str(crop_item.get("detection_id", ""))
        and str(selected_detection.get("frame_id", "")) == str(crop_item.get("frame_id", ""))
        and round(float(selected_detection.get("timestamp_seconds", 0.0) or 0.0), 3)
        == round(float(crop_item.get("timestamp_seconds", 0.0) or 0.0), 3)
    )


def _recover_full_frame_path(
    *,
    run_dir: Path,
    track_id: str,
    selection_group: str,
    frame_id: str,
) -> Path | None:
    """Try to recover a missing Step 05 selected full-frame path by filename pattern."""

    full_frames_dir = run_dir / "05_selected_track_full_frames"
    if not full_frames_dir.exists():
        return None
    pattern = f"{selection_group}_{track_id}_rank*_{frame_id}_full.jpg"
    matches = sorted(full_frames_dir.glob(pattern))
    return matches[0] if matches else None


def _link_best_full_frame(
    *,
    run_dir: Path,
    track_result: dict[str, Any],
    best_crop: dict[str, Any],
    step05_track: dict[str, Any],
) -> tuple[str | None, bool, str | None]:
    """Find the best full-frame path from Step 05 selected detections."""

    selected_detections = list(step05_track.get("selected_detections", []))

    candidate_detection: dict[str, Any] | None = None
    if bool(track_result.get("verified_license_plate_valid")):
        for detection in selected_detections:
            if _selected_detection_matches_crop(detection, best_crop):
                candidate_detection = detection
                break

    if candidate_detection is None and str(track_result.get("best_color_crop_path", "")):
        for detection in selected_detections:
            if str(detection.get("selected_crop_path", "")) == str(track_result.get("best_color_crop_path", "")):
                candidate_detection = detection
                break

    if candidate_detection is None and selected_detections:
        candidate_detection = sorted(
            selected_detections,
            key=lambda item: -float(item.get("final_selection_score", 0.0) or 0.0),
        )[0]

    if candidate_detection is None:
        return None, False, "step05_selected_detection_not_found"

    selected_full_frame_path_value = str(candidate_detection.get("selected_full_frame_path", "") or "")
    selected_full_frame_path = _resolve_run_relative(run_dir, selected_full_frame_path_value)
    if selected_full_frame_path is not None and selected_full_frame_path.exists():
        return _relative_to_run(run_dir, selected_full_frame_path), True, None

    recovered_path = _recover_full_frame_path(
        run_dir=run_dir,
        track_id=str(track_result.get("track_id", "")),
        selection_group=str(track_result.get("selection_group", "")),
        frame_id=str(candidate_detection.get("frame_id", best_crop.get("frame_id", ""))),
    )
    if recovered_path is not None and recovered_path.exists():
        return _relative_to_run(run_dir, recovered_path), True, None

    return None, False, "not_found"


def _compute_search_confidence(track_result: dict[str, Any], vehicle_class: str) -> float:
    """Calculate stable search confidence from track metadata."""

    score = 0.0
    if str(track_result.get("selection_group", "")) == "primary":
        score += 0.20
    else:
        score += 0.05
    if str(track_result.get("quality_label", "")) == "good":
        score += 0.20
    if bool(track_result.get("verified_license_plate_valid")):
        score += 0.35
    if str(track_result.get("best_vehicle_color", "unknown")) != "unknown":
        score += 0.15
    if vehicle_class != "vehicle" or str(track_result.get("dominant_class_name", "")):
        score += 0.10
    if int(track_result.get("verified_license_plate_evidence_count", 0) or 0) > 1:
        score += 0.10
    return round(min(1.0, score), 6)


def _color_terms(color_value: str) -> list[str]:
    """Return a small set of search terms for color aliases."""

    if not color_value or color_value == "unknown":
        return []
    normalized = color_value.lower()
    if normalized == "grey":
        return ["grey", "gray"]
    if normalized == "gray":
        return ["gray", "grey"]
    return [normalized]


def _build_search_terms(record: dict[str, Any], include_possible_ocr: bool) -> tuple[dict[str, list[str]], list[str], list[str]]:
    """Build source-grouped search terms and track filtered invalid OCR tokens."""

    grouped_terms = {
        "class_terms": [],
        "color_terms": [],
        "verified_plate_terms": [],
        "possible_ocr_terms": [],
        "timestamp_terms": [],
        "metadata_terms": [],
    }
    removed_invalid_terms: list[str] = []

    def add_grouped(group_name: str, value: str, *, validate: bool = False) -> None:
        normalized = value.strip()
        if not normalized:
            return
        if validate and is_invalid_search_token(normalized):
            if normalized not in removed_invalid_terms:
                removed_invalid_terms.append(normalized)
            return
        if normalized not in grouped_terms[group_name]:
            grouped_terms[group_name].append(normalized)

    add_grouped("metadata_terms", str(record["track_id"]))
    add_grouped("class_terms", str(record["vehicle_class"]))
    add_grouped("class_terms", str(record["dominant_class_name"]))
    for color_term in _color_terms(str(record["vehicle_color"])):
        add_grouped("color_terms", color_term)
    if bool(record["verified_license_plate_valid"]):
        add_grouped("verified_plate_terms", str(record["verified_license_plate"]))
    if include_possible_ocr:
        for candidate in list(record["possible_license_plate_candidates"]):
            add_grouped("possible_ocr_terms", str(candidate.get("text", "")), validate=True)
    add_grouped("timestamp_terms", str(record["best_timestamp_text"]))
    add_grouped("metadata_terms", str(record["selection_group"]))
    add_grouped("metadata_terms", str(record["quality_label"]))
    for value in dict(record.get("vehicle_attributes", {})).values():
        if isinstance(value, str):
            add_grouped("metadata_terms", value)
    for value in dict(record.get("scene_attributes", {})).values():
        if isinstance(value, str):
            add_grouped("metadata_terms", value)

    combined_terms: list[str] = []
    for group_name in [
        "class_terms",
        "color_terms",
        "verified_plate_terms",
        "possible_ocr_terms",
        "timestamp_terms",
        "metadata_terms",
    ]:
        for term in grouped_terms[group_name]:
            if term not in combined_terms:
                combined_terms.append(term)
    return grouped_terms, combined_terms, removed_invalid_terms


def _find_track_metadata(track_id: str, tracks_payload: dict[str, Any]) -> dict[str, Any]:
    """Look up Step 04B track metadata by track id."""

    for track_item in list(tracks_payload.get("tracks", [])):
        if str(track_item.get("track_id", "")) == track_id:
            return track_item
    return {}


def run_search_index_enrichment(
    *,
    run_dir: Path,
    index_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]] | None]:
    """Create searchable vehicle index payloads from verified Step 06 results."""

    results_payload = read_json(run_dir / "06_ocr_color_results_verified.json")
    report_payload = read_json(run_dir / "06_ocr_color_report_verified.json")
    tracks_payload = read_json(run_dir / "04B_tracks.json")
    best_frames_payload = read_json(run_dir / "05_best_track_frames.json")

    track_results = list(results_payload.get("track_results", []))
    if not track_results:
        empty_index = {
            "status": "no_track_results",
            "source_results_file": "06_ocr_color_results_verified.json",
            "source_report_file": "06_ocr_color_report_verified.json",
            "index_config": index_config,
            "summary": {
                "total_vehicle_records": 0,
                "searchable_records": 0,
                "primary_records": 0,
                "fallback_records": 0,
                "records_with_verified_plate": 0,
                "unique_verified_plate_count": 0,
                "records_with_color": 0,
                "records_with_possible_ocr": 0,
                "records_with_crop": 0,
                "records_with_full_frame": 0,
            },
            "records": [],
        }
        empty_report = {
            "status": "no_track_results",
            "total_vehicle_records": 0,
            "searchable_records": 0,
            "primary_records": 0,
            "fallback_records": 0,
            "records_with_verified_plate": 0,
            "unique_verified_plate_count": 0,
            "verified_license_plates": [],
            "records_with_color": 0,
            "color_counts": {},
            "vehicle_class_counts": {},
            "quality_group_counts": {},
            "search_confidence_stats": {"min": 0.0, "max": 0.0, "avg": 0.0},
            "example_exact_plate_searches": [],
            "example_color_searches": [],
            "recommendation": "No Step 06 track results were available for indexing.",
        }
        write_json(run_dir / "07_vehicle_search_index.json", empty_index)
        write_json(run_dir / "07_vehicle_search_index_report.json", empty_report)
        return empty_index, empty_report, [] if bool(index_config["save_flat_index"]) else None

    include_fallback = bool(index_config["include_fallback"])
    include_possible_ocr = bool(index_config["include_possible_ocr"])
    min_confidence_for_search = float(index_config["min_confidence_for_search"])
    require_color_for_color_index = bool(index_config["require_color_for_color_index"])
    require_verified_plate_for_plate_index = bool(index_config["require_verified_plate_for_plate_index"])

    records: list[dict[str, Any]] = []
    flat_records: list[dict[str, Any]] = []
    color_counts: Counter[str] = Counter()
    vehicle_class_counts: Counter[str] = Counter()
    quality_group_counts: Counter[str] = Counter()
    search_confidence_values: list[float] = []
    invalid_ocr_terms_removed_from_search_text_count = 0
    invalid_ocr_terms_removed_examples: list[dict[str, Any]] = []

    for record_index, track_result in enumerate(track_results, start=1):
        selection_group = str(track_result.get("selection_group", "fallback"))
        if selection_group == "fallback" and not include_fallback:
            continue

        best_crop = _pick_best_crop(track_result)
        if best_crop is None:
            continue
        step05_track = _find_step05_track(str(track_result.get("track_id", "")), best_frames_payload)

        confidence_summary = dict(track_result.get("confidence_summary", {}))
        average_confidence = float(confidence_summary.get("avg", 0.0) or 0.0)
        if average_confidence < min_confidence_for_search:
            continue

        crop_results = list(track_result.get("crop_results", []))
        timestamps = [float(item.get("timestamp_seconds", 0.0) or 0.0) for item in crop_results]
        start_timestamp_seconds = min(timestamps) if timestamps else 0.0
        end_timestamp_seconds = max(timestamps) if timestamps else 0.0
        best_timestamp_seconds = float(best_crop.get("timestamp_seconds", 0.0) or 0.0)
        track_metadata = _find_track_metadata(str(track_result.get("track_id", "")), tracks_payload)
        vehicle_class = _normalize_vehicle_class(track_result, best_crop)
        vehicle_color = str(track_result.get("best_vehicle_color", "unknown") or "unknown").lower()
        if require_color_for_color_index and vehicle_color == "unknown":
            color_search_enabled = False
        else:
            color_search_enabled = vehicle_color != "unknown"

        exact_plate_search_enabled = bool(track_result.get("verified_license_plate_valid"))
        if require_verified_plate_for_plate_index and not exact_plate_search_enabled:
            exact_plate_search_enabled = False

        possible_license_plate_candidates = list(track_result.get("possible_license_plate_candidates", [])) if include_possible_ocr else []
        weak_ocr_text = list(track_result.get("weak_ocr_text", [])) if include_possible_ocr else []
        weak_ocr_search_enabled = bool(possible_license_plate_candidates or weak_ocr_text)

        best_crop_path = _resolve_run_relative(run_dir, str(best_crop.get("selected_crop_path", "")))
        best_full_frame_path_value, full_frame_available, full_frame_missing_reason = _link_best_full_frame(
            run_dir=run_dir,
            track_result=track_result,
            best_crop=best_crop,
            step05_track=step05_track,
        )
        best_plate_crop_path = _resolve_run_relative(
            run_dir,
            str(track_result.get("verified_license_plate_crop_path") or best_crop.get("plate_crop_path") or ""),
        )
        debug_image_path = _resolve_run_relative(run_dir, str(best_crop.get("debug_image_path", "")))

        contact_sheet_path = None
        for track_item in list(best_frames_payload.get("tracks", [])):
            if str(track_item.get("track_id", "")) == str(track_result.get("track_id", "")):
                contact_sheet_path = _resolve_run_relative(run_dir, str(track_item.get("contact_sheet_path", "")))
                break

        record = {
            "search_record_id": f"veh_search_{record_index:06d}",
            "track_id": str(track_result.get("track_id", "")),
            "track_type": "vehicle",
            "vehicle_class": vehicle_class,
            "dominant_class_name": str(track_result.get("dominant_class_name", vehicle_class)),
            "selection_group": selection_group,
            "quality_label": str(track_result.get("quality_label", "")),
            "track_quality": str(track_metadata.get("track_quality", track_result.get("quality_label", ""))),
            "start_timestamp_seconds": round(start_timestamp_seconds, 6),
            "end_timestamp_seconds": round(end_timestamp_seconds, 6),
            "best_timestamp_seconds": round(best_timestamp_seconds, 6),
            "start_timestamp_text": format_seconds_text(start_timestamp_seconds),
            "end_timestamp_text": format_seconds_text(end_timestamp_seconds),
            "best_timestamp_text": format_seconds_text(best_timestamp_seconds),
            "verified_license_plate": str(track_result.get("verified_license_plate", "not_visible")),
            "verified_license_plate_valid": bool(track_result.get("verified_license_plate_valid")),
            "verified_license_plate_source": str(track_result.get("verified_license_plate_source", "none")),
            "verified_license_plate_confidence_level": str(track_result.get("verified_license_plate_confidence_level", "none")),
            "verified_license_plate_evidence_count": int(track_result.get("verified_license_plate_evidence_count", 0) or 0),
            "possible_license_plate_candidates": possible_license_plate_candidates,
            "weak_ocr_text": weak_ocr_text,
            "invalid_ocr_text": list(track_result.get("invalid_ocr_text", [])),
            "vehicle_color": vehicle_color if vehicle_color else "unknown",
            "vehicle_color_source": str(track_result.get("best_color_source", "unknown")),
            "all_candidate_colors": list(track_result.get("all_candidate_colors", [])),
            "vehicle_attributes": dict(track_result.get("vehicle_attributes", {})),
            "license_plate_attributes": dict(track_result.get("license_plate_attributes", {})),
            "scene_attributes": dict(track_result.get("scene_attributes", {})),
            "best_crop_path": _relative_to_run(run_dir, best_crop_path),
            "best_full_frame_path": best_full_frame_path_value,
            "full_frame_available": full_frame_available,
            "full_frame_missing_reason": full_frame_missing_reason,
            "best_plate_crop_path": _relative_to_run(run_dir, best_plate_crop_path),
            "debug_image_path": _relative_to_run(run_dir, debug_image_path),
            "contact_sheet_path": _relative_to_run(run_dir, contact_sheet_path),
            "confidence_summary": confidence_summary,
            "search_confidence": _compute_search_confidence(track_result, vehicle_class),
            "exact_plate_search_enabled": exact_plate_search_enabled,
            "color_search_enabled": color_search_enabled,
            "class_search_enabled": True,
            "weak_ocr_search_enabled": weak_ocr_search_enabled,
            "record_status": "searchable",
        }
        search_terms_by_source, search_terms, removed_invalid_terms = _build_search_terms(record, include_possible_ocr)
        record["search_terms_by_source"] = search_terms_by_source
        record["search_terms"] = search_terms
        record["search_text"] = " ".join(term.lower() for term in record["search_terms"])
        invalid_ocr_terms_removed_from_search_text_count += len(removed_invalid_terms)
        if removed_invalid_terms and len(invalid_ocr_terms_removed_examples) < 20:
            invalid_ocr_terms_removed_examples.append(
                {
                    "track_id": record["track_id"],
                    "removed_terms": removed_invalid_terms,
                }
            )
        records.append(record)

        flat_records.append(
            {
                "search_record_id": record["search_record_id"],
                "track_id": record["track_id"],
                "vehicle_class": record["vehicle_class"],
                "vehicle_color": record["vehicle_color"],
                "vehicle_make": record["vehicle_attributes"].get("make"),
                "vehicle_model": record["vehicle_attributes"].get("model"),
                "vehicle_body_type": record["vehicle_attributes"].get("body_type"),
                "verified_license_plate": record["verified_license_plate"],
                "has_verified_plate": record["verified_license_plate_valid"],
                "possible_ocr_text": " ".join(
                    str(item.get("text", "")) for item in record["possible_license_plate_candidates"] if str(item.get("text", ""))
                ),
                "selection_group": record["selection_group"],
                "quality_label": record["quality_label"],
                "best_timestamp_seconds": record["best_timestamp_seconds"],
                "best_timestamp_text": record["best_timestamp_text"],
                "best_crop_path": record["best_crop_path"],
                "best_full_frame_path": record["best_full_frame_path"],
                "best_plate_crop_path": record["best_plate_crop_path"],
                "contact_sheet_path": record["contact_sheet_path"],
                "search_confidence": record["search_confidence"],
                "search_text": record["search_text"],
            }
        )

        if record["vehicle_color"] != "unknown":
            color_counts[record["vehicle_color"]] += 1
        vehicle_class_counts[record["vehicle_class"]] += 1
        quality_group_counts[record["selection_group"]] += 1
        search_confidence_values.append(record["search_confidence"])

    verified_license_plates = sorted(
        {
            str(record["verified_license_plate"])
            for record in records
            if bool(record["verified_license_plate_valid"]) and str(record["verified_license_plate"]) != "not_visible"
        }
    )
    records_with_possible_ocr = sum(
        1 for record in records if bool(record["possible_license_plate_candidates"]) or bool(record["weak_ocr_text"])
    )
    records_with_crop = sum(1 for record in records if record["best_crop_path"])
    records_with_full_frame = sum(1 for record in records if record["full_frame_available"])
    records_missing_full_frame = len(records) - records_with_full_frame

    index_payload = {
        "status": "success",
        "source_results_file": "06_ocr_color_results_verified.json",
        "source_report_file": "06_ocr_color_report_verified.json",
        "index_config": index_config,
        "summary": {
            "total_vehicle_records": len(records),
            "searchable_records": len(records),
            "primary_records": quality_group_counts.get("primary", 0),
            "fallback_records": quality_group_counts.get("fallback", 0),
            "records_with_verified_plate": sum(1 for record in records if bool(record["verified_license_plate_valid"])),
            "unique_verified_plate_count": len(verified_license_plates),
        "records_with_color": sum(1 for record in records if record["vehicle_color"] != "unknown"),
        "records_with_possible_ocr": records_with_possible_ocr,
        "records_with_crop": records_with_crop,
        "records_with_full_frame": records_with_full_frame,
        },
        "records": records,
    }

    report_output = {
        "status": "success",
        "total_vehicle_records": len(records),
        "searchable_records": len(records),
        "primary_records": quality_group_counts.get("primary", 0),
        "fallback_records": quality_group_counts.get("fallback", 0),
        "records_with_verified_plate": sum(1 for record in records if bool(record["verified_license_plate_valid"])),
        "unique_verified_plate_count": len(verified_license_plates),
        "verified_license_plates": verified_license_plates,
        "records_with_color": sum(1 for record in records if record["vehicle_color"] != "unknown"),
        "records_with_full_frame": records_with_full_frame,
        "records_missing_full_frame": records_missing_full_frame,
        "color_counts": dict(sorted(color_counts.items())),
        "vehicle_class_counts": dict(sorted(vehicle_class_counts.items())),
        "quality_group_counts": dict(sorted(quality_group_counts.items())),
        "search_confidence_stats": _safe_stats(search_confidence_values),
        "invalid_ocr_terms_removed_from_search_text_count": invalid_ocr_terms_removed_from_search_text_count,
        "invalid_ocr_terms_removed_examples": invalid_ocr_terms_removed_examples[:10],
        "search_text_policy": {
            "verified_plate": "included_for_exact_search",
            "possible_ocr": "included_for_weak_search",
            "invalid_ocr": "excluded_from_search_text",
        },
        "example_exact_plate_searches": [
            {
                "query": record["verified_license_plate"],
                "matched_track_id": record["track_id"],
                "vehicle_class": record["vehicle_class"],
                "vehicle_color": record["vehicle_color"],
                "timestamp": record["best_timestamp_text"],
            }
            for record in records
            if bool(record["verified_license_plate_valid"])
        ][:10],
        "example_color_searches": [
            {
                "query": f"{color} {vehicle_class}",
                "matched_count": sum(
                    1 for record in records if record["vehicle_color"] == color and record["vehicle_class"] == vehicle_class
                ),
            }
            for color, vehicle_class in list({(record["vehicle_color"], record["vehicle_class"]) for record in records if record["vehicle_color"] != "unknown"})[:10]
        ],
        "recommendation": "Proceed to Step 08 query test harness / search validation.",
    }

    write_json(run_dir / "07_vehicle_search_index.json", index_payload)
    write_json(run_dir / "07_vehicle_search_index_report.json", report_output)

    flat_index_output: list[dict[str, Any]] | None = None
    if bool(index_config["save_flat_index"]):
        flat_index_output = flat_records
        (run_dir / "07_vehicle_search_index_flat.json").write_text(
            json.dumps(flat_records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return index_payload, report_output, flat_index_output
