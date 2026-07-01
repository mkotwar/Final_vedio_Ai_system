from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from tests.final_demo.services.chunk_planner import read_json
from tests.final_demo.services.video_io import current_timestamp


ENV_FINAL_DEMO_TRACK_FOCUS = "FINAL_DEMO_TRACK_FOCUS"
ENV_FINAL_DEMO_TRACK_CLASSES = "FINAL_DEMO_TRACK_CLASSES"
DEFAULT_TRACK_FOCUS = "auto"

PERSON_CLASSES = ["person"]
VEHICLE_CLASSES = ["bicycle", "car", "motorcycle", "bus", "truck"]
CARRIED_OBJECT_CLASSES = ["backpack", "handbag", "suitcase"]
ANIMAL_CLASSES = ["dog", "cat", "horse", "sheep", "cow"]

FOCUS_PROFILES = {
    "person_security": ["person"],
    "traffic_road": ["person", "bicycle", "car", "motorcycle", "bus", "truck"],
    "vehicle_only": ["bicycle", "car", "motorcycle", "bus", "truck"],
    "parking": ["person", "car", "motorcycle", "bus", "truck"],
    "all_tender": ["person", "bicycle", "car", "motorcycle", "bus", "truck"],
}


def parse_track_classes_override() -> list[str] | None:
    raw_value = os.environ.get(ENV_FINAL_DEMO_TRACK_CLASSES, "")
    if not raw_value.strip():
        return None
    classes = [item.strip().lower() for item in raw_value.split(",") if item.strip()]
    if not classes:
        raise ValueError(
            f"Environment variable {ENV_FINAL_DEMO_TRACK_CLASSES} did not contain any valid class names."
        )
    return classes


def read_focus_mode() -> str:
    return str(
        os.environ.get(ENV_FINAL_DEMO_TRACK_FOCUS, DEFAULT_TRACK_FOCUS) or DEFAULT_TRACK_FOCUS
    ).strip().lower()


def collect_class_counts(
    detection_report_payload: dict[str, Any] | None,
    detections_payload: dict[str, Any],
) -> dict[str, int]:
    report_counts = (
        detection_report_payload.get("detections_by_class")
        if isinstance(detection_report_payload, dict)
        else None
    )
    if isinstance(report_counts, dict) and report_counts:
        return {
            str(class_name).lower(): int(count)
            for class_name, count in report_counts.items()
            if str(class_name).strip()
        }

    counts: dict[str, int] = {}
    for detection in list(detections_payload.get("detections") or []):
        if not isinstance(detection, dict):
            continue
        class_name = str(detection.get("class_name") or "").lower()
        if not class_name:
            continue
        counts[class_name] = counts.get(class_name, 0) + 1
    return counts


def choose_auto_profile(class_counts: dict[str, int]) -> tuple[str, float, str, list[str]]:
    warnings: list[str] = []
    total_person = sum(class_counts.get(class_name, 0) for class_name in PERSON_CLASSES)
    total_vehicle = sum(class_counts.get(class_name, 0) for class_name in VEHICLE_CLASSES)
    total_carried = sum(class_counts.get(class_name, 0) for class_name in CARRIED_OBJECT_CLASSES)
    total_animals = sum(class_counts.get(class_name, 0) for class_name in ANIMAL_CLASSES)
    total_all = sum(int(value) for value in class_counts.values())

    if total_all == 0:
        warnings.append("No detections available for auto tracking focus; using all_tender.")
        return (
            "all_tender",
            0.0,
            "no detections were available; selected all_tender as a safe fallback",
            warnings,
        )

    person_ratio = total_person / total_all
    vehicle_ratio = total_vehicle / total_all

    if vehicle_ratio >= 0.60 and person_ratio < 0.20:
        profile = "vehicle_only"
        confidence = 0.9 if vehicle_ratio >= 0.75 else 0.82
        reason = "vehicle detections dominate strongly and person detections are low; selected vehicle_only profile"
    elif vehicle_ratio >= 0.40 and total_person > 0:
        profile = "traffic_road"
        confidence = 0.74 if vehicle_ratio >= 0.55 else 0.66
        reason = "vehicle and person detections are both present; selected traffic_road profile"
    elif person_ratio >= 0.50:
        profile = "person_security"
        confidence = 0.85 if person_ratio >= 0.70 else 0.72
        reason = "person detections dominate the scene; selected person_security profile"
    else:
        profile = "all_tender"
        confidence = 0.48 if (total_carried > 0 or total_animals > 0) else 0.55
        reason = "no strong single scene pattern was detected; selected all_tender profile"

    if confidence < 0.60:
        warnings.append(
            "Tracking focus auto-selection confidence is low. User should review or select focus manually."
        )
    return profile, round(confidence, 3), reason, warnings


def resolve_tracking_focus(
    *,
    run_dir: Path,
    detections_payload: dict[str, Any],
) -> dict[str, Any]:
    detection_report_path = run_dir / "04_yolo_detection_report.json"
    detection_report_payload = read_json(detection_report_path) if detection_report_path.exists() else None
    class_counts = collect_class_counts(detection_report_payload, detections_payload)
    warnings: list[str] = []

    custom_classes = parse_track_classes_override()
    if custom_classes is not None:
        return {
            "created_at": current_timestamp(),
            "focus_mode": "custom_env_override",
            "selected_focus_profile": "custom",
            "selected_track_classes": custom_classes,
            "focus_confidence": 1.0,
            "class_counts": dict(sorted(class_counts.items())),
            "reason": "FINAL_DEMO_TRACK_CLASSES override was provided; using exactly those tracking classes",
            "warnings": warnings,
        }

    requested_focus = read_focus_mode()
    if requested_focus != "auto":
        if requested_focus == "custom":
            raise ValueError(
                "FINAL_DEMO_TRACK_FOCUS=custom requires FINAL_DEMO_TRACK_CLASSES to be set."
            )
        if requested_focus not in FOCUS_PROFILES:
            raise ValueError(
                f"Environment variable {ENV_FINAL_DEMO_TRACK_FOCUS} must be one of: "
                f"auto, custom, {', '.join(sorted(FOCUS_PROFILES))}"
            )
        return {
            "created_at": current_timestamp(),
            "focus_mode": requested_focus,
            "selected_focus_profile": requested_focus,
            "selected_track_classes": list(FOCUS_PROFILES[requested_focus]),
            "focus_confidence": 1.0,
            "class_counts": dict(sorted(class_counts.items())),
            "reason": f"Predefined tracking focus profile {requested_focus} was selected by environment variable",
            "warnings": warnings,
        }

    selected_profile, confidence, reason, auto_warnings = choose_auto_profile(class_counts)
    warnings.extend(auto_warnings)
    return {
        "created_at": current_timestamp(),
        "focus_mode": "auto",
        "selected_focus_profile": selected_profile,
        "selected_track_classes": list(FOCUS_PROFILES[selected_profile]),
        "focus_confidence": confidence,
        "class_counts": dict(sorted(class_counts.items())),
        "reason": reason,
        "warnings": warnings,
    }
