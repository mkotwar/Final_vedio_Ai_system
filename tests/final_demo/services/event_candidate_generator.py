from __future__ import annotations

import os
from difflib import SequenceMatcher
from collections import defaultdict
from pathlib import Path
from typing import Any

from tests.final_demo.services.chunk_planner import read_json
from tests.final_demo.services.video_io import current_timestamp, write_json


ENV_FINAL_DEMO_DWELL_SECONDS = "FINAL_DEMO_DWELL_SECONDS"
ENV_FINAL_DEMO_EVENT_PLATE_DEDUP_ENABLED = "FINAL_DEMO_EVENT_PLATE_DEDUP_ENABLED"
ENV_FINAL_DEMO_EVENT_PLATE_DEDUP_TIME_WINDOW_SECONDS = "FINAL_DEMO_EVENT_PLATE_DEDUP_TIME_WINDOW_SECONDS"
ENV_FINAL_DEMO_EVENT_PLATE_DEDUP_TEXT_SIMILARITY = "FINAL_DEMO_EVENT_PLATE_DEDUP_TEXT_SIMILARITY"
ENV_FINAL_DEMO_EVENT_INCLUDE_FRAME_SCAN_UNREADABLE = "FINAL_DEMO_EVENT_INCLUDE_FRAME_SCAN_UNREADABLE"
DEFAULT_DWELL_SECONDS = 10.0
DEFAULT_EVENT_PLATE_DEDUP_ENABLED = True
DEFAULT_EVENT_PLATE_DEDUP_TIME_WINDOW_SECONDS = 2.0
DEFAULT_EVENT_PLATE_DEDUP_TEXT_SIMILARITY = 0.80
DEFAULT_EVENT_INCLUDE_FRAME_SCAN_UNREADABLE = False
ALLOWED_PLATE_EVENT_STATUSES = {"read_strong", "read_needs_review"}
ALLOWED_PLATE_EVENT_FORMATS = {
    "valid_indian_plate",
    "possible_indian_plate",
    "partial_indian_plate",
}


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def read_non_negative_float_env(env_name: str, default_value: float) -> float:
    raw_value = os.environ.get(env_name, str(default_value))
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {env_name} must be a valid number. Received: {raw_value!r}"
        ) from exc
    if value < 0:
        raise ValueError(
            f"Environment variable {env_name} must be greater than or equal to 0. Received: {value}"
        )
    return value


def read_bool_env(env_name: str, default_value: bool) -> bool:
    raw_value = os.environ.get(env_name)
    if raw_value is None or raw_value.strip() == "":
        return default_value
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(
        f"Environment variable {env_name} must be boolean-like. Received: {raw_value!r}"
    )


def round_time(value: Any) -> float:
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return 0.0


def normalize_track_lookup(tracks_payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    if not isinstance(tracks_payload, dict):
        return lookup
    for track in list(tracks_payload.get("clean_tracks") or []):
        if not isinstance(track, dict):
            continue
        track_id = str(track.get("clean_track_id") or track.get("source_track_id") or "")
        if track_id:
            lookup[track_id] = track
    return lookup


def normalize_attribute_lookup(attributes_payload: dict[str, Any] | None) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_attribute_id: dict[str, dict[str, Any]] = {}
    by_source_track_id: dict[str, dict[str, Any]] = {}
    if not isinstance(attributes_payload, dict):
        return by_attribute_id, by_source_track_id
    for attribute in list(attributes_payload.get("attributes") or []):
        if not isinstance(attribute, dict):
            continue
        attribute_track_id = str(attribute.get("attribute_track_id") or "")
        source_track_id = str(attribute.get("source_track_id") or "")
        if attribute_track_id:
            by_attribute_id[attribute_track_id] = attribute
        if source_track_id and source_track_id not in by_source_track_id:
            by_source_track_id[source_track_id] = attribute
    return by_attribute_id, by_source_track_id


def normalize_ocr_lookup(ocr_payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    if not isinstance(ocr_payload, dict):
        return lookup
    for item in list(ocr_payload.get("results") or []):
        if not isinstance(item, dict):
            continue
        if str(item.get("candidate_source") or "") == "frame_scan":
            continue
        attribute_track_id = str(item.get("attribute_track_id") or "")
        source_track_id = str(item.get("source_track_id") or "")
        if attribute_track_id:
            lookup[attribute_track_id] = item
        elif source_track_id:
            lookup[source_track_id] = item
    return lookup


def extract_frame_scan_ocr_results(ocr_payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not isinstance(ocr_payload, dict):
        return items
    for item in list(ocr_payload.get("results") or []):
        if not isinstance(item, dict):
            continue
        if str(item.get("candidate_source") or "") != "frame_scan":
            continue
        items.append(item)
    items.sort(
        key=lambda entry: (
            round_time(entry.get("best_ocr_timestamp") or entry.get("start_time")),
            str(entry.get("best_ocr_frame_id") or ""),
        )
    )
    return items


def build_ocr_event_audit(ocr_payload: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    results = list((ocr_payload or {}).get("results") or []) if isinstance(ocr_payload, dict) else []
    audit_rows: list[dict[str, Any]] = []
    possible_matches: list[dict[str, Any]] = []
    target_trace: dict[str, Any] = {
        "target": "HR38AE1442",
        "found": False,
        "possible_matches": [],
    }

    results_with_candidate_source_frame_scan = 0
    results_with_null_track_ids = 0
    results_with_candidate_text = 0
    results_with_allowed_status = 0
    results_with_allowed_format = 0
    eligible_for_plate_event_count = 0
    skipped_empty_text_count = 0
    skipped_status_count = 0
    skipped_format_count = 0

    for index, item in enumerate(results):
        if not isinstance(item, dict):
            continue
        candidate_source = str(item.get("candidate_source") or "")
        source_track_id = item.get("source_track_id")
        attribute_track_id = item.get("attribute_track_id")
        matched_source_track_id = item.get("matched_source_track_id")
        matched_attribute_track_id = item.get("matched_attribute_track_id")
        candidate_text = str(item.get("candidate_plate_text") or "")
        corrected_text = str(item.get("corrected_plate_text") or "")
        plate_status = str(item.get("plate_ocr_status") or "")
        plate_format_status = str(item.get("plate_format_status") or "")

        is_frame_scan_by_candidate_source = candidate_source == "frame_scan"
        is_frame_scan_by_null_track = not source_track_id and not attribute_track_id
        has_candidate_text = bool(candidate_text.strip())
        status_allowed = plate_status in ALLOWED_PLATE_EVENT_STATUSES
        format_allowed = plate_format_status in ALLOWED_PLATE_EVENT_FORMATS
        eligible_for_plate_event = has_candidate_text and status_allowed and format_allowed

        if is_frame_scan_by_candidate_source:
            results_with_candidate_source_frame_scan += 1
        if is_frame_scan_by_null_track:
            results_with_null_track_ids += 1
        if has_candidate_text:
            results_with_candidate_text += 1
        else:
            skipped_empty_text_count += 1
        if status_allowed:
            results_with_allowed_status += 1
        else:
            skipped_status_count += 1
        if format_allowed:
            results_with_allowed_format += 1
        else:
            skipped_format_count += 1
        if eligible_for_plate_event:
            eligible_for_plate_event_count += 1

        if not has_candidate_text:
            skip_reason = "empty_candidate_plate_text"
        elif not status_allowed:
            skip_reason = f"plate_ocr_status_not_allowed:{plate_status or 'missing'}"
        elif not format_allowed:
            skip_reason = f"plate_format_status_not_allowed:{plate_format_status or 'missing'}"
        else:
            skip_reason = "eligible_for_plate_event"

        row = {
            "index": index,
            "candidate_source": candidate_source,
            "source_track_id": source_track_id,
            "attribute_track_id": attribute_track_id,
            "matched_source_track_id": matched_source_track_id,
            "matched_attribute_track_id": matched_attribute_track_id,
            "class_name": item.get("class_name"),
            "vehicle_type": item.get("vehicle_type"),
            "candidate_plate_text": candidate_text,
            "corrected_plate_text": corrected_text,
            "plate_ocr_status": plate_status,
            "plate_format_status": plate_format_status,
            "ocr_confidence": item.get("ocr_confidence"),
            "final_plate_confidence": item.get("final_plate_confidence"),
            "is_frame_scan_by_candidate_source": is_frame_scan_by_candidate_source,
            "is_frame_scan_by_null_track": is_frame_scan_by_null_track,
            "has_candidate_text": has_candidate_text,
            "status_allowed": status_allowed,
            "format_allowed": format_allowed,
            "eligible_for_plate_event": eligible_for_plate_event,
            "skip_reason": skip_reason,
        }
        audit_rows.append(row)

        haystacks = [candidate_text.upper(), corrected_text.upper()]
        if any(target in haystacks for target in ["HR38AE1442", "HR38", "AE1442"]):
            possible_matches.append(
                {
                    "index": index,
                    "candidate_source": candidate_source,
                    "candidate_plate_text": candidate_text,
                    "corrected_plate_text": corrected_text,
                    "plate_ocr_status": plate_status,
                    "plate_format_status": plate_format_status,
                    "eligible_for_plate_event": eligible_for_plate_event,
                    "skip_reason": skip_reason,
                }
            )
        if candidate_text.upper() == "HR38AE1442" or corrected_text.upper() == "HR38AE1442":
            target_trace = {
                "target": "HR38AE1442",
                "found": True,
                "index": index,
                "candidate_source": candidate_source,
                "candidate_plate_text": candidate_text,
                "corrected_plate_text": corrected_text,
                "plate_ocr_status": plate_status,
                "plate_format_status": plate_format_status,
                "eligible_for_plate_event": eligible_for_plate_event,
                "skip_reason": skip_reason,
            }

    if not target_trace.get("found"):
        target_trace["possible_matches"] = possible_matches

    probable_bug = ""
    if (
        results_with_candidate_source_frame_scan >= 2
        and results_with_null_track_ids <= 1
    ):
        probable_bug = "Step 7B is counting frame_scan using null track ids instead of candidate_source."
    if target_trace.get("found") and bool(target_trace.get("eligible_for_plate_event")):
        probable_bug = "Eligible OCR result is built by audit but skipped in event creation path."
    elif target_trace.get("found") and not bool(target_trace.get("eligible_for_plate_event")):
        probable_bug = "Eligibility rule mismatch; check status/format/text fields."
    elif not target_trace.get("found"):
        probable_bug = "Step 7B is reading a different or stale 07A_plate_ocr_results.json."

    audit_payload = {
        "created_at": current_timestamp(),
        "ocr_results_source": "07A_plate_ocr_results.json",
        "ocr_results_total_loaded": len(audit_rows),
        "target_plate_trace": target_trace,
        "rows": audit_rows,
    }
    report_payload = {
        "created_at": current_timestamp(),
        "overall_status": "completed",
        "ocr_results_total_loaded": len(audit_rows),
        "results_with_candidate_source_frame_scan": results_with_candidate_source_frame_scan,
        "results_with_null_track_ids": results_with_null_track_ids,
        "results_with_candidate_text": results_with_candidate_text,
        "results_with_allowed_status": results_with_allowed_status,
        "results_with_allowed_format": results_with_allowed_format,
        "eligible_for_plate_event_count": eligible_for_plate_event_count,
        "skipped_empty_text_count": skipped_empty_text_count,
        "skipped_status_count": skipped_status_count,
        "skipped_format_count": skipped_format_count,
        "target_plate_found": bool(target_trace.get("found")),
        "target_plate_eligible": bool(target_trace.get("eligible_for_plate_event")),
        "target_plate_skip_reason": target_trace.get("skip_reason"),
        "probable_bug": probable_bug,
    }
    return audit_payload, report_payload


def read_event_settings() -> dict[str, Any]:
    return {
        "plate_dedup_enabled": read_bool_env(
            ENV_FINAL_DEMO_EVENT_PLATE_DEDUP_ENABLED,
            DEFAULT_EVENT_PLATE_DEDUP_ENABLED,
        ),
        "plate_dedup_time_window_seconds": round(
            read_non_negative_float_env(
                ENV_FINAL_DEMO_EVENT_PLATE_DEDUP_TIME_WINDOW_SECONDS,
                DEFAULT_EVENT_PLATE_DEDUP_TIME_WINDOW_SECONDS,
            ),
            3,
        ),
        "plate_dedup_text_similarity": round(
            read_non_negative_float_env(
                ENV_FINAL_DEMO_EVENT_PLATE_DEDUP_TEXT_SIMILARITY,
                DEFAULT_EVENT_PLATE_DEDUP_TEXT_SIMILARITY,
            ),
            3,
        ),
        "include_frame_scan_unreadable": read_bool_env(
            ENV_FINAL_DEMO_EVENT_INCLUDE_FRAME_SCAN_UNREADABLE,
            DEFAULT_EVENT_INCLUDE_FRAME_SCAN_UNREADABLE,
        ),
    }


def plate_format_rank(status: str) -> int:
    return {
        "valid_indian_plate": 4,
        "possible_indian_plate": 3,
        "partial_indian_plate": 2,
        "weak_pattern": 1,
        "non_plate_text": 0,
        "unreadable": -1,
        "not_available": -1,
    }.get(str(status or ""), -1)


def text_similarity(text_a: str, text_b: str) -> float:
    if not text_a or not text_b:
        return 0.0
    return round(SequenceMatcher(None, text_a, text_b).ratio(), 4)


def build_base_keywords(*parts: Any) -> list[str]:
    keywords: list[str] = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, list):
            for item in part:
                text = str(item).strip()
                if text and text not in keywords:
                    keywords.append(text)
            continue
        text = str(part).strip()
        if text and text not in keywords:
            keywords.append(text)
    return keywords


def track_confidence(track: dict[str, Any]) -> float:
    max_conf = track.get("max_confidence")
    if max_conf is not None:
        try:
            return round(max(0.25, min(0.95, float(max_conf))), 3)
        except (TypeError, ValueError):
            pass
    avg_conf = track.get("average_confidence")
    if avg_conf is not None:
        try:
            return round(max(0.25, min(0.95, float(avg_conf))), 3)
        except (TypeError, ValueError):
            pass
    return 0.60


def attribute_confidence(attribute: dict[str, Any]) -> float:
    value = attribute.get("attribute_confidence")
    try:
        return round(max(0.20, min(0.95, float(value))), 3)
    except (TypeError, ValueError):
        return 0.55


def ocr_event_confidence(ocr_item: dict[str, Any]) -> float:
    status = str(ocr_item.get("plate_ocr_status") or "")
    format_status = str(ocr_item.get("plate_format_status") or "")
    indian_plate_score = float(ocr_item.get("indian_plate_score") or 0.0)
    ocr_conf = float(ocr_item.get("ocr_confidence") or 0.0)
    plate_candidate_score = float(ocr_item.get("plate_candidate_score") or 0.0)
    if status == "read_strong":
        return round(min(0.95, 0.65 + indian_plate_score * 0.20 + ocr_conf * 0.10), 3)
    if status == "read_needs_review" and format_status == "valid_indian_plate":
        return round(min(0.75, 0.45 + indian_plate_score * 0.15 + ocr_conf * 0.10 + plate_candidate_score * 0.05), 3)
    if status == "read_needs_review" and format_status == "possible_indian_plate":
        return round(min(0.65, 0.35 + indian_plate_score * 0.15 + ocr_conf * 0.10 + plate_candidate_score * 0.05), 3)
    if status == "read_weak":
        return round(min(0.45, 0.20 + ocr_conf * 0.10 + plate_candidate_score * 0.10), 3)
    return round(min(0.30, 0.10 + plate_candidate_score * 0.10 + ocr_conf * 0.05), 3)


def resolve_vehicle_class_name(
    track: dict[str, Any] | None,
    attribute: dict[str, Any] | None,
    ocr_item: dict[str, Any] | None,
) -> str:
    for candidate in (
        ocr_item.get("class_name") if isinstance(ocr_item, dict) else None,
        attribute.get("class_name") if isinstance(attribute, dict) else None,
        track.get("class_name") if isinstance(track, dict) else None,
    ):
        text = str(candidate or "").strip().lower()
        if text:
            return text
    return "vehicle"


def build_event(
    *,
    event_type: str,
    event_family: str,
    class_name: str,
    source_track_id: str | None,
    attribute_track_id: str | None,
    start_time: float,
    end_time: float,
    title: str,
    description: str,
    confidence: float,
    risk_score: float,
    needs_review: bool,
    search_keywords: list[str],
    attributes: dict[str, Any],
    evidence: dict[str, Any],
    source_steps: list[str],
) -> dict[str, Any]:
    duration_seconds = round(max(0.0, end_time - start_time), 3)
    representative_timestamp = round(start_time if duration_seconds <= 0 else start_time + duration_seconds / 2.0, 3)
    return {
        "event_type": event_type,
        "event_family": event_family,
        "class_name": class_name,
        "source_track_id": source_track_id,
        "attribute_track_id": attribute_track_id,
        "start_time": round(start_time, 3),
        "end_time": round(end_time, 3),
        "representative_timestamp": representative_timestamp,
        "duration_seconds": duration_seconds,
        "title": title,
        "description": description,
        "confidence": round(confidence, 3),
        "risk_score": round(risk_score, 3),
        "needs_review": bool(needs_review),
        "search_keywords": search_keywords,
        "attributes": attributes,
        "evidence": evidence,
        "source_steps": source_steps,
        "status": "candidate",
    }


def build_person_events(
    track: dict[str, Any],
    attribute: dict[str, Any] | None,
    *,
    dwell_seconds: float,
) -> list[dict[str, Any]]:
    source_track_id = str(track.get("clean_track_id") or track.get("source_track_id") or "")
    attribute_track_id = str(attribute.get("attribute_track_id") or "") if isinstance(attribute, dict) else None
    start_time = round_time(track.get("start_time"))
    end_time = round_time(track.get("end_time"))
    duration_seconds = round(max(0.0, end_time - start_time), 3)
    confidence = track_confidence(track)
    review_flag = bool(track.get("needs_review")) or str(track.get("count_for_summary")) == "review"
    evidence = {
        "best_frame_id": track.get("best_frame_id"),
        "best_image_path": track.get("best_image_path"),
        "crop_path": attribute.get("attribute_crop_path") if isinstance(attribute, dict) else None,
        "plate_crop_path": None,
        "ocr_debug_crop_dir": None,
    }
    events = [
        build_event(
            event_type="person_track_observed",
            event_family="person",
            class_name="person",
            source_track_id=source_track_id,
            attribute_track_id=attribute_track_id,
            start_time=start_time,
            end_time=end_time,
            title="Person track observed",
            description="Person track observed in the scene.",
            confidence=confidence,
            risk_score=0.15,
            needs_review=review_flag,
            search_keywords=build_base_keywords("person", "track", track.get("cleanup_status")),
            attributes={
                "cleanup_status": track.get("cleanup_status"),
                "count_for_summary": track.get("count_for_summary"),
                "motion_status": attribute.get("motion_status") if isinstance(attribute, dict) else None,
            },
            evidence=evidence,
            source_steps=["05B_track_cleanup", "06_attribute_extraction"] if attribute else ["05B_track_cleanup"],
        )
    ]
    if start_time <= 1.0:
        events.append(
            build_event(
                event_type="person_entered_scene",
                event_family="person",
                class_name="person",
                source_track_id=source_track_id,
                attribute_track_id=attribute_track_id,
                start_time=start_time,
                end_time=min(end_time, start_time + 1.0),
                title="Person entered scene",
                description="Person appears near the start of the observed scene.",
                confidence=max(0.50, confidence - 0.05),
                risk_score=0.10,
                needs_review=review_flag,
                search_keywords=build_base_keywords("person", "entered", "scene"),
                attributes={"start_edge": attribute.get("start_edge") if isinstance(attribute, dict) else None},
                evidence=evidence,
                source_steps=["05B_track_cleanup", "06_attribute_extraction"] if attribute else ["05B_track_cleanup"],
            )
        )
    if isinstance(attribute, dict):
        if str(attribute.get("motion_status") or "") in {"mostly_stationary", "stationary"}:
            events.append(
                build_event(
                    event_type="person_stationary_candidate",
                    event_family="person",
                    class_name="person",
                    source_track_id=source_track_id,
                    attribute_track_id=attribute_track_id,
                    start_time=start_time,
                    end_time=end_time,
                    title="Person stationary candidate",
                    description="Person appears mostly stationary in the scene.",
                    confidence=max(0.50, attribute_confidence(attribute)),
                    risk_score=0.25,
                    needs_review=review_flag,
                    search_keywords=build_base_keywords("person", "stationary", "candidate"),
                    attributes={"motion_status": attribute.get("motion_status"), "speed_level": attribute.get("speed_level")},
                    evidence=evidence,
                    source_steps=["05B_track_cleanup", "06_attribute_extraction"],
                )
            )
        if duration_seconds >= dwell_seconds:
            events.append(
                build_event(
                    event_type="person_dwell_candidate",
                    event_family="person",
                    class_name="person",
                    source_track_id=source_track_id,
                    attribute_track_id=attribute_track_id,
                    start_time=start_time,
                    end_time=end_time,
                    title="Person dwell candidate",
                    description=f"Person remains in scene for at least {dwell_seconds:.1f} seconds.",
                    confidence=max(0.55, attribute_confidence(attribute)),
                    risk_score=0.30,
                    needs_review=review_flag,
                    search_keywords=build_base_keywords("person", "dwell", "candidate"),
                    attributes={"duration_seconds": duration_seconds},
                    evidence=evidence,
                    source_steps=["05B_track_cleanup", "06_attribute_extraction"],
                )
            )
        person_attrs = dict(attribute.get("person_attributes") or {})
        if person_attrs:
            events.append(
                build_event(
                    event_type="person_attribute_observed",
                    event_family="person",
                    class_name="person",
                    source_track_id=source_track_id,
                    attribute_track_id=attribute_track_id,
                    start_time=start_time,
                    end_time=end_time,
                    title="Person attributes observed",
                    description="Person attributes were extracted for search and review.",
                    confidence=attribute_confidence(attribute),
                    risk_score=0.10,
                    needs_review=review_flag,
                    search_keywords=build_base_keywords(
                        "person",
                        person_attrs.get("upper_clothing_color"),
                        person_attrs.get("lower_clothing_color"),
                        person_attrs.get("carried_object_candidates"),
                    ),
                    attributes={
                        "upper_clothing_color": person_attrs.get("upper_clothing_color"),
                        "lower_clothing_color": person_attrs.get("lower_clothing_color"),
                        "carried_object_candidates": person_attrs.get("carried_object_candidates"),
                    },
                    evidence=evidence,
                    source_steps=["05B_track_cleanup", "06_attribute_extraction"],
                )
            )
            if list(person_attrs.get("carried_object_candidates") or []):
                events.append(
                    build_event(
                        event_type="person_with_possible_carried_object",
                        event_family="person",
                        class_name="person",
                        source_track_id=source_track_id,
                        attribute_track_id=attribute_track_id,
                        start_time=start_time,
                        end_time=end_time,
                        title="Person with possible carried object",
                        description="Person may be carrying or closely associated with an object.",
                        confidence=max(0.50, attribute_confidence(attribute) - 0.05),
                        risk_score=0.20,
                        needs_review=True,
                        search_keywords=build_base_keywords("person", "carried_object", person_attrs.get("carried_object_candidates")),
                        attributes={"carried_object_candidates": person_attrs.get("carried_object_candidates")},
                        evidence=evidence,
                        source_steps=["05B_track_cleanup", "06_attribute_extraction"],
                    )
                )
    return events


def build_vehicle_events(track: dict[str, Any], attribute: dict[str, Any] | None, ocr_item: dict[str, Any] | None) -> list[dict[str, Any]]:
    source_track_id = str(track.get("clean_track_id") or track.get("source_track_id") or "")
    attribute_track_id = str(attribute.get("attribute_track_id") or "") if isinstance(attribute, dict) else None
    resolved_class_name = resolve_vehicle_class_name(track, attribute, ocr_item)
    start_time = round_time(track.get("start_time"))
    end_time = round_time(track.get("end_time"))
    confidence = track_confidence(track)
    review_flag = bool(track.get("needs_review")) or str(track.get("count_for_summary")) == "review"
    vehicle_attrs = dict(attribute.get("vehicle_attributes") or {}) if isinstance(attribute, dict) else {}
    evidence = {
        "best_frame_id": track.get("best_frame_id"),
        "best_image_path": track.get("best_image_path"),
        "crop_path": attribute.get("attribute_crop_path") if isinstance(attribute, dict) else None,
        "plate_crop_path": (
            ocr_item.get("selected_crop_path")
            if isinstance(ocr_item, dict) and ocr_item.get("selected_crop_path")
            else vehicle_attrs.get("possible_plate_crop_path")
        ),
        "ocr_debug_crop_dir": ocr_item.get("debug_crop_dir") if isinstance(ocr_item, dict) else None,
        "alternate_plate_crop_paths": [],
        "supporting_frame_ids": [],
        "supporting_timestamps": [],
    }
    events = [
        build_event(
            event_type="vehicle_track_observed",
            event_family="vehicle",
            class_name=resolved_class_name,
            source_track_id=source_track_id,
            attribute_track_id=attribute_track_id,
            start_time=start_time,
            end_time=end_time,
            title="Vehicle track observed",
            description="Vehicle track observed in the scene.",
            confidence=confidence,
            risk_score=0.20,
            needs_review=review_flag,
            search_keywords=build_base_keywords("vehicle", resolved_class_name, vehicle_attrs.get("vehicle_color")),
            attributes={"cleanup_status": track.get("cleanup_status"), "vehicle_type": vehicle_attrs.get("vehicle_type")},
            evidence=evidence,
            source_steps=["05B_track_cleanup", "06_attribute_extraction"] if attribute else ["05B_track_cleanup"],
        )
    ]
    if isinstance(attribute, dict) and vehicle_attrs:
        events.append(
            build_event(
                event_type="vehicle_attribute_observed",
                event_family="vehicle",
                class_name=resolved_class_name,
                source_track_id=source_track_id,
                attribute_track_id=attribute_track_id,
                start_time=start_time,
                end_time=end_time,
                title="Vehicle attributes observed",
                description="Vehicle attributes were extracted for search and review.",
                confidence=attribute_confidence(attribute),
                risk_score=0.10,
                needs_review=review_flag,
                search_keywords=build_base_keywords("vehicle", vehicle_attrs.get("vehicle_type"), vehicle_attrs.get("vehicle_color"), vehicle_attrs.get("vehicle_category")),
                attributes={
                    "vehicle_type": vehicle_attrs.get("vehicle_type"),
                    "vehicle_color": vehicle_attrs.get("vehicle_color"),
                    "vehicle_category": vehicle_attrs.get("vehicle_category"),
                },
                evidence=evidence,
                source_steps=["05B_track_cleanup", "06_attribute_extraction"],
            )
        )
        if str(attribute.get("motion_status") or "") in {"mostly_stationary", "stationary"}:
            events.append(
                build_event(
                    event_type="vehicle_stationary_candidate",
                    event_family="vehicle",
                    class_name=resolved_class_name,
                    source_track_id=source_track_id,
                    attribute_track_id=attribute_track_id,
                    start_time=start_time,
                    end_time=end_time,
                    title="Vehicle stationary candidate",
                    description="Vehicle appears mostly stationary in the scene.",
                    confidence=max(0.50, attribute_confidence(attribute)),
                    risk_score=0.25,
                    needs_review=review_flag,
                    search_keywords=build_base_keywords("vehicle", "stationary", vehicle_attrs.get("vehicle_type")),
                    attributes={"motion_status": attribute.get("motion_status"), "speed_level": attribute.get("speed_level")},
                    evidence=evidence,
                    source_steps=["05B_track_cleanup", "06_attribute_extraction"],
                )
            )
    if isinstance(ocr_item, dict):
        plate_format_status = str(ocr_item.get("plate_format_status") or "")
        plate_status = str(ocr_item.get("plate_ocr_status") or "")
        candidate_text = str(ocr_item.get("candidate_plate_text") or "")
        body_text_possible = bool(ocr_item.get("body_text_possible"))
        if plate_format_status == "non_plate_text" and body_text_possible:
            events.append(
                build_event(
                    event_type="vehicle_body_text_observed",
                    event_family="vehicle",
                    class_name=resolved_class_name,
                    source_track_id=source_track_id,
                    attribute_track_id=attribute_track_id,
                    start_time=round_time(ocr_item.get("best_ocr_timestamp") or start_time),
                    end_time=round_time(ocr_item.get("best_ocr_timestamp") or start_time) + 0.25,
                    title="Vehicle body text observed",
                    description="Detected text appears to be body/fleet/phone text rather than a licence plate.",
                    confidence=max(0.25, ocr_event_confidence(ocr_item)),
                    risk_score=0.10,
                    needs_review=True,
                    search_keywords=build_base_keywords("vehicle", "body_text", candidate_text),
                    attributes={
                        "candidate_plate_text": candidate_text,
                        "plate_ocr_status": plate_status,
                        "plate_format_status": plate_format_status,
                        "ocr_confidence": ocr_item.get("ocr_confidence"),
                        "body_text_possible": True,
                        "candidate_source": ocr_item.get("candidate_source"),
                        "source_detection_id": ocr_item.get("source_detection_id"),
                        "source_detection_class_name": ocr_item.get("source_detection_class_name"),
                        "class_source": ocr_item.get("class_source"),
                    },
                    evidence=evidence,
                    source_steps=["05B_track_cleanup", "06_attribute_extraction", "06A_plate_candidate_detection", "07A_plate_ocr"],
                )
            )
            return events
        if (
            plate_status in {"read_strong", "read_needs_review"}
            and plate_format_status in {"valid_indian_plate", "possible_indian_plate", "partial_indian_plate"}
            and candidate_text
        ):
            event_type = "vehicle_plate_candidate_observed" if plate_status == "read_strong" else "vehicle_plate_needs_review"
            title = "Vehicle plate candidate observed" if plate_status == "read_strong" else "Vehicle plate needs review"
            description = (
                f"{vehicle_attrs.get('vehicle_type') or track.get('class_name') or 'Vehicle'} has a candidate Indian licence plate reading "
                f"{candidate_text}. OCR {'appears strong' if plate_status == 'read_strong' else 'requires review'}."
            )
            needs_review = bool(ocr_item.get("needs_review")) or review_flag or plate_status != "read_strong"
            events.append(
                build_event(
                    event_type=event_type,
                    event_family="vehicle",
                    class_name=resolved_class_name,
                    source_track_id=source_track_id,
                    attribute_track_id=attribute_track_id,
                    start_time=round_time(ocr_item.get("best_ocr_timestamp") or start_time),
                    end_time=round_time(ocr_item.get("best_ocr_timestamp") or start_time) + 0.25,
                    title=title,
                    description=description,
                    confidence=ocr_event_confidence(ocr_item),
                    risk_score=0.20,
                    needs_review=needs_review,
                    search_keywords=build_base_keywords("vehicle", vehicle_attrs.get("vehicle_type"), "plate", candidate_text, "needs_review" if needs_review else "candidate"),
                    attributes={
                        "vehicle_type": vehicle_attrs.get("vehicle_type"),
                        "vehicle_color": vehicle_attrs.get("vehicle_color"),
                        "candidate_plate_text": candidate_text,
                        "plate_ocr_status": plate_status,
                        "plate_format_status": plate_format_status,
                        "ocr_confidence": ocr_item.get("ocr_confidence"),
                        "candidate_source": ocr_item.get("candidate_source"),
                        "source_detection_id": ocr_item.get("source_detection_id"),
                        "source_detection_class_name": ocr_item.get("source_detection_class_name"),
                        "matched_source_track_id": ocr_item.get("matched_source_track_id"),
                        "matched_attribute_track_id": ocr_item.get("matched_attribute_track_id"),
                        "matched_track_class_name": ocr_item.get("matched_track_class_name"),
                        "matched_track_vehicle_type": ocr_item.get("matched_track_vehicle_type"),
                        "matched_track_iou": ocr_item.get("matched_track_iou"),
                        "matched_track_time_delta": ocr_item.get("matched_track_time_delta"),
                        "class_source": ocr_item.get("class_source"),
                    },
                    evidence=evidence,
                    source_steps=["05B_track_cleanup", "06_attribute_extraction", "06A_plate_candidate_detection", "07A_plate_ocr"],
                )
            )
        elif (
            plate_status in {"unreadable", "not_available"}
            and str(ocr_item.get("candidate_source") or "track_based") != "frame_scan"
            and (review_flag or track.get("count_for_summary") in {True, "review"})
        ):
            events.append(
                build_event(
                    event_type="vehicle_plate_unreadable",
                    event_family="vehicle",
                    class_name=str(track.get("class_name") or "vehicle"),
                    source_track_id=source_track_id,
                    attribute_track_id=attribute_track_id,
                    start_time=start_time,
                    end_time=end_time,
                    title="Vehicle plate unreadable",
                    description="Vehicle appears useful for review, but no reliable plate reading was produced.",
                    confidence=max(0.10, ocr_event_confidence(ocr_item)),
                    risk_score=0.15,
                    needs_review=True,
                    search_keywords=build_base_keywords("vehicle", "plate", "unreadable", vehicle_attrs.get("vehicle_type")),
                    attributes={
                        "plate_ocr_status": plate_status,
                        "plate_format_status": plate_format_status,
                    },
                    evidence=evidence,
                    source_steps=["05B_track_cleanup", "06_attribute_extraction", "07A_plate_ocr"],
                )
            )
    return events


def build_object_events(track: dict[str, Any], attribute: dict[str, Any] | None) -> list[dict[str, Any]]:
    source_track_id = str(track.get("clean_track_id") or track.get("source_track_id") or "")
    attribute_track_id = str(attribute.get("attribute_track_id") or "") if isinstance(attribute, dict) else None
    start_time = round_time(track.get("start_time"))
    end_time = round_time(track.get("end_time"))
    class_name = str(track.get("class_name") or "object")
    events = [
        build_event(
            event_type="object_track_observed",
            event_family="object",
            class_name=class_name,
            source_track_id=source_track_id,
            attribute_track_id=attribute_track_id,
            start_time=start_time,
            end_time=end_time,
            title="Object track observed",
            description=f"{class_name} track observed in the scene.",
            confidence=track_confidence(track),
            risk_score=0.10,
            needs_review=bool(track.get("needs_review")) or str(track.get("count_for_summary")) == "review",
            search_keywords=build_base_keywords("object", class_name),
            attributes={"cleanup_status": track.get("cleanup_status")},
            evidence={
                "best_frame_id": track.get("best_frame_id"),
                "best_image_path": track.get("best_image_path"),
                "crop_path": attribute.get("attribute_crop_path") if isinstance(attribute, dict) else None,
                "plate_crop_path": None,
                "ocr_debug_crop_dir": None,
            },
            source_steps=["05B_track_cleanup", "06_attribute_extraction"] if attribute else ["05B_track_cleanup"],
        )
    ]
    if isinstance(attribute, dict):
        events.append(
            build_event(
                event_type="object_attribute_observed",
                event_family="object",
                class_name=class_name,
                source_track_id=source_track_id,
                attribute_track_id=attribute_track_id,
                start_time=start_time,
                end_time=end_time,
                title="Object attributes observed",
                description=f"{class_name} attributes were extracted.",
                confidence=attribute_confidence(attribute),
                risk_score=0.05,
                needs_review=bool(track.get("needs_review")),
                search_keywords=build_base_keywords("object", class_name),
                attributes=dict(attribute.get("basic_attributes") or {}),
                evidence={
                    "best_frame_id": track.get("best_frame_id"),
                    "best_image_path": track.get("best_image_path"),
                    "crop_path": attribute.get("attribute_crop_path"),
                    "plate_crop_path": None,
                    "ocr_debug_crop_dir": None,
                },
                source_steps=["05B_track_cleanup", "06_attribute_extraction"],
            )
        )
    return events


def build_untracked_plate_events(ocr_item: dict[str, Any]) -> list[dict[str, Any]]:
    plate_status = str(ocr_item.get("plate_ocr_status") or "")
    plate_format_status = str(ocr_item.get("plate_format_status") or "")
    candidate_text = str(ocr_item.get("candidate_plate_text") or "")
    matched_source_track_id = ocr_item.get("matched_source_track_id")
    matched_attribute_track_id = ocr_item.get("matched_attribute_track_id")
    source_track_id = matched_source_track_id or ocr_item.get("source_track_id")
    attribute_track_id = matched_attribute_track_id or ocr_item.get("attribute_track_id")
    if matched_source_track_id:
        class_name = str(
            ocr_item.get("matched_track_class_name")
            or ocr_item.get("class_name")
            or ocr_item.get("vehicle_type")
            or "vehicle"
        )
        vehicle_type = str(
            ocr_item.get("matched_track_vehicle_type")
            or ocr_item.get("vehicle_type")
            or class_name
        )
        class_source = "matched_clean_track"
    else:
        class_name = str(
            ocr_item.get("source_detection_class_name")
            or ocr_item.get("class_name")
            or ocr_item.get("vehicle_type")
            or "vehicle"
        )
        vehicle_type = class_name
        class_source = "source_yolo_detection"
    timestamp = round_time(ocr_item.get("best_ocr_timestamp") or ocr_item.get("start_time"))
    evidence = {
        "best_frame_id": ocr_item.get("best_ocr_frame_id"),
        "best_image_path": (
            list(ocr_item.get("frames_used_for_ocr") or [{}])[0].get("image_path")
            if list(ocr_item.get("frames_used_for_ocr") or [])
            else None
        ),
        "crop_path": None,
        "plate_crop_path": ocr_item.get("selected_crop_path") or ocr_item.get("plate_candidate_crop_path"),
        "ocr_debug_crop_dir": ocr_item.get("debug_crop_dir"),
        "alternate_plate_crop_paths": [],
        "supporting_frame_ids": [],
        "supporting_timestamps": [],
    }
    is_plate_candidate = (
        plate_status in {"read_strong", "read_needs_review"}
        and plate_format_status in {"valid_indian_plate", "possible_indian_plate", "partial_indian_plate"}
        and bool(candidate_text)
    )
    if not is_plate_candidate:
        return []
    if plate_status == "read_strong":
        event_type = "vehicle_plate_candidate_observed"
    else:
        event_type = "vehicle_plate_needs_review"
    title = "Vehicle plate candidate observed" if plate_status == "read_strong" else "Vehicle plate needs review"
    description = (
        f"{class_name} has a candidate Indian licence plate reading {candidate_text}. "
        f"OCR {'appears strong' if plate_status == 'read_strong' else 'requires review'}."
    )
    return [
        build_event(
            event_type=event_type,
            event_family="vehicle",
            class_name=class_name,
            source_track_id=source_track_id,
            attribute_track_id=attribute_track_id,
            start_time=timestamp,
            end_time=timestamp + 0.25,
            title=title,
            description=description,
            confidence=ocr_event_confidence(ocr_item),
            risk_score=0.20,
            needs_review=bool(ocr_item.get("needs_review")) or plate_status != "read_strong",
            search_keywords=build_base_keywords("vehicle", class_name, "plate", candidate_text, "frame_scan"),
            attributes={
                "vehicle_type": vehicle_type,
                "vehicle_color": ocr_item.get("vehicle_color"),
                "candidate_plate_text": candidate_text,
                "plate_ocr_status": plate_status,
                "plate_format_status": plate_format_status,
                "ocr_confidence": ocr_item.get("ocr_confidence"),
                "plate_candidate_score": ocr_item.get("plate_candidate_score"),
                "needs_review": ocr_item.get("needs_review"),
                "candidate_source": ocr_item.get("candidate_source"),
                "source_detection_id": ocr_item.get("source_detection_id"),
                "source_detection_class_name": ocr_item.get("source_detection_class_name"),
                "matched_source_track_id": matched_source_track_id,
                "matched_attribute_track_id": matched_attribute_track_id,
                "matched_track_class_name": ocr_item.get("matched_track_class_name"),
                "matched_track_vehicle_type": ocr_item.get("matched_track_vehicle_type"),
                "matched_track_iou": ocr_item.get("matched_track_iou"),
                "matched_track_time_delta": ocr_item.get("matched_track_time_delta"),
                "class_source": class_source,
            },
            evidence=evidence,
            source_steps=["06A_plate_candidate_detection", "07A_plate_ocr"],
        )
    ]


def build_multiple_people_events(person_tracks: list[dict[str, Any]], attribute_by_track: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    overlap_ranges: list[tuple[float, float, list[str], bool]] = []
    for index, track_a in enumerate(person_tracks):
        for track_b in person_tracks[index + 1 :]:
            start = max(round_time(track_a.get("start_time")), round_time(track_b.get("start_time")))
            end = min(round_time(track_a.get("end_time")), round_time(track_b.get("end_time")))
            if end - start < 1.0:
                continue
            review_flag = (
                bool(track_a.get("needs_review"))
                or bool(track_b.get("needs_review"))
                or str(track_a.get("count_for_summary")) == "review"
                or str(track_b.get("count_for_summary")) == "review"
            )
            overlap_ranges.append(
                (
                    round(start, 3),
                    round(end, 3),
                    [
                        str(track_a.get("clean_track_id") or track_a.get("source_track_id") or ""),
                        str(track_b.get("clean_track_id") or track_b.get("source_track_id") or ""),
                    ],
                    review_flag,
                )
            )
    events: list[dict[str, Any]] = []
    for start, end, source_track_ids, review_flag in overlap_ranges:
        events.append(
            build_event(
                event_type="multiple_people_present",
                event_family="person",
                class_name="person",
                source_track_id="+".join(source_track_ids),
                attribute_track_id=None,
                start_time=start,
                end_time=end,
                title="Multiple people present",
                description="Two or more person tracks overlap in time.",
                confidence=0.55,
                risk_score=0.15,
                needs_review=review_flag,
                search_keywords=build_base_keywords("person", "multiple_people", "overlap"),
                attributes={"source_track_ids": source_track_ids},
                evidence={"best_frame_id": None, "best_image_path": None, "crop_path": None, "plate_crop_path": None, "ocr_debug_crop_dir": None},
                source_steps=["05B_track_cleanup"],
            )
        )
    return events


def should_cluster_plate_events(
    cluster_seed: dict[str, Any],
    candidate: dict[str, Any],
    *,
    settings: dict[str, Any],
) -> bool:
    candidate_source = str(candidate.get("attributes", {}).get("candidate_source") or "")
    seed_source = str(cluster_seed.get("attributes", {}).get("candidate_source") or "")
    if candidate_source != "frame_scan" or seed_source != "frame_scan":
        return False
    candidate_text = str(candidate.get("attributes", {}).get("candidate_plate_text") or "")
    seed_text = str(cluster_seed.get("attributes", {}).get("candidate_plate_text") or "")
    if not candidate_text or not seed_text:
        return False
    time_delta = abs(
        round_time(candidate.get("start_time")) - round_time(cluster_seed.get("start_time"))
    )
    if time_delta > float(settings["plate_dedup_time_window_seconds"]):
        return False
    seed_track = str(cluster_seed.get("source_track_id") or cluster_seed.get("attributes", {}).get("matched_source_track_id") or "")
    candidate_track = str(candidate.get("source_track_id") or candidate.get("attributes", {}).get("matched_source_track_id") or "")
    if seed_track and candidate_track and seed_track == candidate_track:
        return True
    if candidate_text == seed_text:
        return True
    return text_similarity(candidate_text, seed_text) >= float(settings["plate_dedup_text_similarity"])


def representative_plate_event(events: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    sorted_events = sorted(
        events,
        key=lambda item: (
            plate_format_rank(str(item.get("attributes", {}).get("plate_format_status") or "")),
            float(item.get("confidence") or 0.0),
            float(item.get("attributes", {}).get("ocr_confidence") or 0.0),
            float(item.get("attributes", {}).get("plate_candidate_score") or 0.0),
        ),
        reverse=True,
    )
    representative = dict(sorted_events[0])
    alternate_texts = []
    for item in events:
        text = str(item.get("attributes", {}).get("candidate_plate_text") or "")
        if text and text not in alternate_texts:
            alternate_texts.append(text)
    return representative, alternate_texts


def cluster_plate_events(
    events: list[dict[str, Any]],
    *,
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not settings["plate_dedup_enabled"]:
        return events, {
            "raw_plate_ocr_events_before_dedup": 0,
            "plate_events_after_dedup": 0,
            "frame_scan_events_before_dedup": 0,
            "frame_scan_events_after_dedup": 0,
            "plate_event_clusters_created": 0,
            "duplicate_plate_events_suppressed": 0,
            "frame_scan_events_matched_to_tracks": 0,
            "frame_scan_events_unmatched": 0,
        }

    clustered: list[dict[str, Any]] = []
    plate_candidates = [
        event for event in events
        if str(event.get("event_type") or "") in {
            "vehicle_plate_candidate_observed",
            "vehicle_plate_needs_review",
            "weak_plate_text_observed",
        }
    ]
    non_plate = [event for event in events if event not in plate_candidates]
    frame_scan_candidates = [
        event for event in plate_candidates
        if str(event.get("attributes", {}).get("candidate_source") or "") == "frame_scan"
    ]
    track_candidates = [event for event in plate_candidates if event not in frame_scan_candidates]

    used: set[int] = set()
    cluster_count = 0
    matched_to_tracks = 0
    unmatched = 0
    for index, seed in enumerate(frame_scan_candidates):
        if index in used:
            continue
        group = [seed]
        used.add(index)
        for candidate_index in range(index + 1, len(frame_scan_candidates)):
            if candidate_index in used:
                continue
            candidate = frame_scan_candidates[candidate_index]
            if should_cluster_plate_events(seed, candidate, settings=settings):
                group.append(candidate)
                used.add(candidate_index)
        representative, alternate_texts = representative_plate_event(group)
        cluster_count += 1
        representative["cluster_id"] = f"plate_cluster_{cluster_count:06d}"
        representative["attributes"] = dict(representative.get("attributes") or {})
        representative["evidence"] = dict(representative.get("evidence") or {})
        representative["attributes"]["cluster_size"] = len(group)
        representative["cluster_size"] = len(group)
        representative["attributes"]["alternate_candidate_texts"] = alternate_texts
        representative["attributes"]["best_candidate_reason"] = "highest_plate_format_then_confidence"
        representative["candidate_source"] = representative["attributes"].get("candidate_source")
        representative["source_detection_id"] = representative["attributes"].get("source_detection_id")
        representative["source_detection_class_name"] = representative["attributes"].get("source_detection_class_name")
        representative["matched_source_track_id"] = representative["attributes"].get("matched_source_track_id")
        representative["class_source"] = representative["attributes"].get("class_source")
        representative["evidence"]["alternate_plate_crop_paths"] = list(
            dict.fromkeys(
                [
                    item.get("evidence", {}).get("plate_crop_path")
                    for item in group
                    if item.get("evidence", {}).get("plate_crop_path")
                ]
            )
        )
        representative["evidence"]["supporting_frame_ids"] = list(
            dict.fromkeys(
                [
                    item.get("evidence", {}).get("best_frame_id")
                    for item in group
                    if item.get("evidence", {}).get("best_frame_id")
                ]
            )
        )
        representative["evidence"]["supporting_timestamps"] = list(
            dict.fromkeys([round_time(item.get("start_time")) for item in group])
        )
        if representative["matched_source_track_id"]:
            matched_to_tracks += 1
        else:
            unmatched += 1
        clustered.append(representative)

    clustered.extend(track_candidates)
    clustered.extend(non_plate)
    return clustered, {
        "raw_plate_ocr_events_before_dedup": len(plate_candidates),
        "plate_events_after_dedup": len([item for item in clustered if str(item.get("event_type") or "") in {"vehicle_plate_candidate_observed", "vehicle_plate_needs_review", "weak_plate_text_observed"}]),
        "frame_scan_events_before_dedup": len(frame_scan_candidates),
        "frame_scan_events_after_dedup": len([item for item in clustered if str(item.get("attributes", {}).get("candidate_source") or "") == "frame_scan"]),
        "plate_event_clusters_created": cluster_count,
        "duplicate_plate_events_suppressed": len(frame_scan_candidates) - cluster_count,
        "frame_scan_events_matched_to_tracks": matched_to_tracks,
        "frame_scan_events_unmatched": unmatched,
    }


def dedupe_and_finalize_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[str, str, float, float], dict[str, Any]] = {}
    for event in events:
        key = (
            str(event["event_type"]),
            str(event.get("source_track_id") or ""),
            round_time(event["start_time"]),
            round_time(event["end_time"]),
        )
        existing = deduped.get(key)
        if existing is None or float(event["confidence"]) > float(existing["confidence"]):
            deduped[key] = event
    ordered = sorted(
        deduped.values(),
        key=lambda item: (
            round_time(item["start_time"]),
            str(item["event_family"]),
            str(item["event_type"]),
            str(item["source_track_id"]),
        ),
    )
    for index, event in enumerate(ordered, start=1):
        event["event_candidate_id"] = f"evt_cand_{index:06d}"
    return ordered


def update_run_manifest_for_event_candidates(run_manifest_path: Path) -> dict[str, Any]:
    run_manifest = read_json(run_manifest_path)
    completed_steps = list(run_manifest.get("completed_steps") or [])
    if "07B_event_candidate_generation" not in completed_steps:
        completed_steps.append("07B_event_candidate_generation")
    run_manifest["completed_steps"] = completed_steps
    run_manifest["next_step"] = "08_event_grouping_or_ranking"
    write_json(run_manifest_path, run_manifest)
    return run_manifest


def build_event_candidate_outputs(run_dir: Path) -> dict[str, Any]:
    settings = read_event_settings()
    dwell_seconds = round(read_non_negative_float_env(ENV_FINAL_DEMO_DWELL_SECONDS, DEFAULT_DWELL_SECONDS), 3)
    tracks_path = run_dir / "05B_clean_tracks.json"
    attributes_path = run_dir / "06_track_attributes.json"

    warnings: list[str] = []
    missing_inputs: list[str] = []
    if not tracks_path.exists():
        missing_inputs.append(str(tracks_path.name))
    if not attributes_path.exists():
        missing_inputs.append(str(attributes_path.name))

    tracks_payload = read_optional_json(tracks_path)
    attributes_payload = read_optional_json(attributes_path)
    ocr_results_payload = read_optional_json(run_dir / "07A_plate_ocr_results.json")
    ocr_report_payload = read_optional_json(run_dir / "07A_plate_ocr_report.json")
    plate_candidates_payload = read_optional_json(run_dir / "06A_plate_candidates.json")
    plate_candidate_report_payload = read_optional_json(run_dir / "06A_plate_candidate_report.json")
    detection_report_payload = read_optional_json(run_dir / "04_yolo_detection_report.json")
    tracking_focus_payload = read_optional_json(run_dir / "05_tracking_focus.json")
    video_info_payload = read_optional_json(run_dir / "01_video_info.json")
    audit_payload, audit_report_payload = build_ocr_event_audit(ocr_results_payload)

    if tracks_payload is None:
        warnings.append("Missing required input: 05B_clean_tracks.json")
    if attributes_payload is None:
        warnings.append("Missing required input: 06_track_attributes.json")

    if tracks_payload is None or attributes_payload is None:
        results_payload = {
            "created_at": current_timestamp(),
            "selected_track_source": "05B_clean_tracks",
            "selected_attribute_source": "06_track_attributes",
            "selected_plate_ocr_source": "07A_plate_ocr_results",
            "total_event_candidates": 0,
            "events": [],
            "warnings": warnings,
        }
        report_payload = {
            "created_at": current_timestamp(),
            "overall_status": "skipped_missing_inputs",
            "total_tracks_input": 0,
            "total_attributes_input": 0,
            "total_ocr_results_input": len(list((ocr_results_payload or {}).get("results") or [])),
            "total_event_candidates": 0,
            "events_by_family": {"person": 0, "vehicle": 0, "object": 0},
            "events_by_type": {},
            "events_needing_review": 0,
            "plate_candidate_events": 0,
            "body_text_events": 0,
            "unreadable_plate_events": 0,
            "warnings": warnings,
            "recommendations": ["Run Step 5B and Step 6 before Step 7B."],
        }
        return {
            "events_payload": results_payload,
            "report_payload": report_payload,
            "audit_payload": audit_payload,
            "audit_report_payload": audit_report_payload,
        }

    track_lookup = normalize_track_lookup(tracks_payload)
    attribute_by_id, attribute_by_track = normalize_attribute_lookup(attributes_payload)
    ocr_lookup = normalize_ocr_lookup(ocr_results_payload)
    frame_scan_ocr_results = extract_frame_scan_ocr_results(ocr_results_payload)

    events: list[dict[str, Any]] = []
    person_tracks: list[dict[str, Any]] = []
    for source_track_id, track in track_lookup.items():
        attribute = attribute_by_track.get(source_track_id)
        class_name = str(track.get("class_name") or attribute.get("class_name") if isinstance(attribute, dict) else "object")
        if class_name == "person":
            person_tracks.append(track)
            events.extend(build_person_events(track, attribute, dwell_seconds=dwell_seconds))
        elif class_name in {"car", "truck", "bus", "motorcycle", "bicycle"}:
            ocr_item = None
            if isinstance(attribute, dict):
                ocr_item = ocr_lookup.get(str(attribute.get("attribute_track_id") or "")) or ocr_lookup.get(source_track_id)
            events.extend(build_vehicle_events(track, attribute, ocr_item))
        else:
            events.extend(build_object_events(track, attribute))

    for ocr_item in frame_scan_ocr_results:
        events.extend(build_untracked_plate_events(ocr_item))

    events.extend(build_multiple_people_events(person_tracks, attribute_by_track))
    clustered_events, plate_cluster_stats = cluster_plate_events(events, settings=settings)
    finalized_events = dedupe_and_finalize_events(clustered_events)

    if not finalized_events:
        overall_status = "completed_no_events"
    else:
        overall_status = "completed"

    events_by_family: dict[str, int] = defaultdict(int)
    events_by_type: dict[str, int] = defaultdict(int)
    events_by_candidate_source: dict[str, int] = defaultdict(int)
    events_by_class: dict[str, int] = defaultdict(int)
    events_needing_review = 0
    plate_candidate_events = 0
    body_text_events = 0
    unreadable_plate_events = 0
    for event in finalized_events:
        events_by_family[str(event["event_family"])] += 1
        events_by_type[str(event["event_type"])] += 1
        events_by_class[str(event.get("class_name") or "unknown")] += 1
        candidate_source = str(event.get("attributes", {}).get("candidate_source") or "")
        if candidate_source:
            events_by_candidate_source[candidate_source] += 1
        if bool(event.get("needs_review")):
            events_needing_review += 1
        if str(event["event_type"]) in {"vehicle_plate_candidate_observed", "vehicle_plate_needs_review"}:
            plate_candidate_events += 1
        if str(event["event_type"]) == "vehicle_body_text_observed":
            body_text_events += 1
        if str(event["event_type"]) == "vehicle_plate_unreadable":
            unreadable_plate_events += 1

    recommendations: list[str] = []
    if body_text_events > 0:
        recommendations.append("Review vehicle body text events separately from licence plate candidates.")
    if unreadable_plate_events > 0:
        recommendations.append("Unreadable plate events may benefit from better plate crops or higher-resolution OCR.")
    if plate_candidate_report_payload and str(plate_candidate_report_payload.get("overall_status")) == "completed_no_plate_candidates":
        recommendations.append("No reliable Step 6A plate candidates were found for some vehicles.")

    results_payload = {
        "created_at": current_timestamp(),
        "selected_track_source": "05B_clean_tracks",
        "selected_attribute_source": "06_track_attributes",
        "selected_plate_ocr_source": "07A_plate_ocr_results" if isinstance(ocr_results_payload, dict) else None,
        "frame_scan_plate_ocr_results": len(frame_scan_ocr_results),
        "total_event_candidates": len(finalized_events),
        "events": finalized_events,
        "warnings": warnings,
    }
    report_payload = {
        "created_at": current_timestamp(),
        "overall_status": overall_status,
        "total_tracks_input": len(track_lookup),
        "total_attributes_input": len(attribute_by_id),
        "total_ocr_results_input": len(list((ocr_results_payload or {}).get("results") or [])),
        "frame_scan_plate_ocr_results": len(frame_scan_ocr_results),
        "total_event_candidates": len(finalized_events),
        "raw_plate_ocr_events_before_dedup": plate_cluster_stats["raw_plate_ocr_events_before_dedup"],
        "plate_events_after_dedup": plate_cluster_stats["plate_events_after_dedup"],
        "frame_scan_events_before_dedup": plate_cluster_stats["frame_scan_events_before_dedup"],
        "frame_scan_events_after_dedup": plate_cluster_stats["frame_scan_events_after_dedup"],
        "plate_event_clusters_created": plate_cluster_stats["plate_event_clusters_created"],
        "duplicate_plate_events_suppressed": plate_cluster_stats["duplicate_plate_events_suppressed"],
        "frame_scan_events_matched_to_tracks": plate_cluster_stats["frame_scan_events_matched_to_tracks"],
        "frame_scan_events_unmatched": plate_cluster_stats["frame_scan_events_unmatched"],
        "events_by_family": {
            "person": int(events_by_family.get("person", 0)),
            "vehicle": int(events_by_family.get("vehicle", 0)),
            "object": int(events_by_family.get("object", 0)),
        },
        "events_by_type": dict(sorted(events_by_type.items())),
        "events_by_candidate_source": dict(sorted(events_by_candidate_source.items())),
        "events_by_class": dict(sorted(events_by_class.items())),
        "events_needing_review": events_needing_review,
        "plate_candidate_events": plate_candidate_events,
        "body_text_events": body_text_events,
        "unreadable_plate_events": unreadable_plate_events,
        "warnings": warnings,
        "recommendations": recommendations,
    }
    return {
        "events_payload": results_payload,
        "report_payload": report_payload,
        "audit_payload": audit_payload,
        "audit_report_payload": audit_report_payload,
    }
