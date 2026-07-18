from __future__ import annotations

import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from .class_normalization import normalize_class_name
from .config import ObjectTrackingConfig
from .serialization import read_jsonl, write_json


TRACK_CLASSES = {"person", "car", "motorcycle", "bus", "truck"}


def object_group_for_detection(class_name: str | None) -> str:
    return normalize_class_name(class_name).object_group


def validate_object_tracking_config(
    config: ObjectTrackingConfig,
    *,
    vehicle_model_names: dict[int, str] | None = None,
    person_model_names: dict[int, str] | None = None,
    require_existing_paths: bool = True,
) -> dict[str, Any]:
    report = {
        "detection_mode": config.detection_mode,
        "enable_vehicle_tracking": config.enable_vehicle_tracking,
        "enable_person_tracking": config.enable_person_tracking,
        "vehicle_model_path": config.vehicle_model_path,
        "person_model_path": config.person_model_path,
        "vehicle_model_classes": _normalize_names(vehicle_model_names or {}),
        "person_model_classes": _normalize_names(person_model_names or {}),
        "supports_vehicle": False,
        "supports_person": False,
        "errors": [],
    }
    if require_existing_paths:
        if config.enable_vehicle_tracking and config.vehicle_model_path and not Path(config.vehicle_model_path).exists():
            report["errors"].append(f"vehicle_model_missing:{config.vehicle_model_path}")
        if config.enable_person_tracking and config.detection_mode == "dual" and config.person_model_path and not Path(config.person_model_path).exists():
            report["errors"].append(f"person_model_missing:{config.person_model_path}")
    vehicle_classes = {item["normalized_class_name"] for item in report["vehicle_model_classes"].values()}
    person_classes = {item["normalized_class_name"] for item in report["person_model_classes"].values()}
    report["supports_vehicle"] = bool(vehicle_classes & {"car", "motorcycle", "bus", "truck"})
    report["supports_person"] = "person" in vehicle_classes or "person" in person_classes
    if config.enable_person_tracking and not report["supports_person"]:
        report["errors"].append("person_tracking_enabled_but_no_person_detector")
    if config.enable_vehicle_tracking and vehicle_model_names is not None and not report["supports_vehicle"]:
        report["errors"].append("vehicle_tracking_enabled_but_no_vehicle_detector")
    if report["errors"]:
        raise ValueError("; ".join(report["errors"]))
    return report


def inspect_yolo_model(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "load_status": "not_configured", "class_names": {}, "error": None}
    model_path = Path(path)
    if not model_path.exists():
        return {"path": str(model_path), "exists": False, "load_status": "missing", "class_names": {}, "error": None}
    try:
        from ultralytics import YOLO  # type: ignore

        model = YOLO(str(model_path))
        names = getattr(model, "names", {}) or {}
        class_names = {int(key): str(value) for key, value in (names.items() if isinstance(names, dict) else enumerate(names))}
        return {"path": str(model_path), "exists": True, "load_status": "loaded", "class_names": class_names, "error": None}
    except Exception as exc:
        return {
            "path": str(model_path),
            "exists": True,
            "load_status": "load_failed",
            "class_names": {},
            "error": str(exc),
            "zip_entries": _zip_entries(model_path),
        }


def build_person_tracking_audit(
    *,
    run_dir: str | Path,
    vehicle_model_path: str | Path | None,
    person_model_path: str | Path | None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    vehicle_model = inspect_yolo_model(vehicle_model_path)
    person_model = inspect_yolo_model(person_model_path)
    vehicle_classes = _normalized_class_set(vehicle_model.get("class_names") or {})
    person_classes = _normalized_class_set(person_model.get("class_names") or {})
    root = Path(run_dir)
    detections = _read_optional_jsonl(root / "03_tracking" / "frame_detections.jsonl")
    if not detections:
        detections = _read_optional_jsonl(root / "02_detection" / "detection_packets.jsonl")
    if not detections:
        detections = _read_optional_jsonl(root / "02_detections" / "detection_packets.jsonl")
    tracks = _read_optional_jsonl(root / "04_lifecycle" / "completed_tracks.jsonl")
    searchable = _read_optional_jsonl(root / "09_searchable_objects" / "searchable_vehicle_records.jsonl")

    detection_counts = _count_detections(detections)
    step6_metrics = _read_step6_detection_metrics(root)
    track_counts = _count_tracks(tracks)
    record_counts = Counter(str(row.get("object_group") or ("person" if row.get("object_class") == "person" else "vehicle")) for row in searchable)
    root_causes: list[str] = []
    if "person" not in vehicle_classes:
        root_causes.append("configured_vehicle_detector_does_not_advertise_person_class")
    if person_model_path and person_model.get("load_status") == "load_failed":
        root_causes.append("configured_person_detector_failed_to_load")
    if not person_model_path:
        root_causes.append("person_detector_not_configured")
    if detection_counts["person_after_filter"] == 0:
        root_causes.append("no_person_detections_reached_filtered_detection_artifacts")
    if track_counts["person_completed"] == 0:
        root_causes.append("no_person_completed_tracks_in_lifecycle_artifact")
    if record_counts["person"] == 0:
        root_causes.append("no_person_records_in_searchable_object_artifact")

    audit = {
        "vehicle_detector_supports_person": "person" in vehicle_classes,
        "person_detector_configured": bool(person_model_path),
        "person_detector_load_status": person_model.get("load_status"),
        "person_detector_error": person_model.get("error"),
        "vehicle_model_path": str(vehicle_model_path) if vehicle_model_path else None,
        "vehicle_model_class_mapping": vehicle_model.get("class_names") or {},
        "person_model_path": str(person_model_path) if person_model_path else None,
        "person_model_class_mapping": person_model.get("class_names") or {},
        "person_model_zip_entries": person_model.get("zip_entries") or [],
        "model_raw_detections_all_classes": step6_metrics.get("raw_detections"),
        "filtered_detections_all_classes": step6_metrics.get("filtered_detections"),
        "person_detections_before_filter": step6_metrics.get("person_detections_before_filter", detection_counts["person_before_filter"]),
        "person_detections_after_filter": step6_metrics.get("person_detections_after_filter", detection_counts["person_after_filter"]),
        "person_tracks_created": track_counts["person_total"],
        "person_tracks_confirmed": track_counts["person_confirmed"],
        "person_tracks_completed": track_counts["person_completed"],
        "person_records_written": record_counts["person"],
        "vehicle_records_written": record_counts["vehicle"],
        "root_causes": root_causes,
    }
    if output_path is None:
        output_path = root / "person_tracking_audit.json"
    write_json(output_path, audit)
    return audit


def _normalize_names(names: dict[int, str]) -> dict[int, dict[str, Any]]:
    return {int(class_id): normalize_class_name(class_name, int(class_id)).to_dict() for class_id, class_name in names.items()}


def _normalized_class_set(names: dict[int, str]) -> set[str]:
    return {normalize_class_name(value, key).normalized_class_name for key, value in names.items()}


def _zip_entries(path: Path) -> list[str]:
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    except Exception:
        return []


def _read_optional_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [dict(item) for item in read_jsonl(path)]


def _read_step6_detection_metrics(root: Path) -> dict[str, Any]:
    report_path = root / "reports" / "step6_validation_result.json"
    if not report_path.exists():
        return {}
    try:
        from .serialization import read_json

        report = read_json(report_path)
    except Exception:
        return {}
    metrics = ((report.get("crop_collection_report") or {}).get("detection_metrics") or {})
    class_counts = metrics.get("class_counts") or {}
    person_after = int(class_counts.get("person") or 0)
    result: dict[str, Any] = {}
    if "raw_detections" in metrics:
        result["raw_detections"] = int(metrics.get("raw_detections") or 0)
    if "filtered_detections" in metrics:
        result["filtered_detections"] = int(metrics.get("filtered_detections") or 0)
    if person_after:
        result["person_detections_before_filter"] = person_after
        result["person_detections_after_filter"] = person_after
    return result


def _count_detections(rows: list[dict[str, Any]]) -> dict[str, int]:
    before = 0
    after = 0
    for row in rows:
        for detection in row.get("detections") or []:
            if normalize_class_name(detection.get("raw_class_name") or detection.get("class_name")).normalized_class_name == "person":
                after += 1
        if normalize_class_name(row.get("raw_class_name") or row.get("class_name")).normalized_class_name == "person":
            before += 1
            after += 1
    return {"person_before_filter": before or after, "person_after_filter": after}


def _count_tracks(rows: list[dict[str, Any]]) -> dict[str, int]:
    total = 0
    confirmed = 0
    completed = 0
    for row in rows:
        class_name = row.get("last_class_name") or _dominant_class(row.get("class_votes"))
        if normalize_class_name(class_name).normalized_class_name != "person":
            continue
        total += 1
        if str(row.get("status") or "") in {"confirmed", "temporarily_lost", "completed"}:
            confirmed += 1
        if str(row.get("status") or "") == "completed":
            completed += 1
    return {"person_total": total, "person_confirmed": confirmed, "person_completed": completed}


def _dominant_class(class_votes: Any) -> str | None:
    if not isinstance(class_votes, dict) or not class_votes:
        return None
    return sorted(class_votes.items(), key=lambda item: (-int(item[1] or 0), str(item[0])))[0][0]
