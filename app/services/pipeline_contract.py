"""Canonical file contract for the video processing pipeline.

Frames are persisted as a catalog plus optional per-frame files. Events are
persisted as both per-event files and a consolidated catalog consumed by
summary/search APIs.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from app.core.config import settings


FRAME_CATALOG_SUFFIX = "_frames.json"
EVENT_CATALOG_SUFFIX = "_events.json"


def frame_catalog_path(video_id: str) -> Path:
    """Return the canonical frame catalog path for a video."""
    return settings.METADATA_DIR / f"{video_id}{FRAME_CATALOG_SUFFIX}"


def event_catalog_path(video_id: str) -> Path:
    """Return the canonical consolidated event catalog path for a video."""
    return settings.METADATA_DIR / f"{video_id}{EVENT_CATALOG_SUFFIX}"


def frame_metadata_dir(video_id: str) -> Path:
    """Return the optional per-frame metadata directory for a video."""
    return settings.METADATA_DIR / video_id


def event_dir(video_id: str) -> Path:
    """Return the canonical per-event JSON directory for a video."""
    return settings.EVENTS_DIR / video_id


def read_json_file(path: Path) -> Any:
    """Read JSON from disk using the pipeline's UTF-8 contract."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_file(path: Path, payload: Any) -> None:
    """Write JSON to disk using the pipeline's UTF-8 contract."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def load_frame_records(video_id: str) -> List[Dict[str, Any]]:
    """Load frames from the canonical catalog, falling back to per-frame files."""
    catalog = frame_catalog_path(video_id)
    if catalog.exists():
        data = read_json_file(catalog)
        return data if isinstance(data, list) else []

    records: List[Dict[str, Any]] = []
    frames_dir = frame_metadata_dir(video_id)
    if frames_dir.exists():
        for path in sorted(frames_dir.glob("*.json")):
            data = read_json_file(path)
            if isinstance(data, dict):
                records.append(data)
    return records


def load_event_catalog(video_id: str) -> List[Dict[str, Any]]:
    """Load the canonical consolidated event catalog if it exists."""
    catalog = event_catalog_path(video_id)
    if not catalog.exists():
        return []
    data = read_json_file(catalog)
    return data if isinstance(data, list) else []


def write_event_catalog(video_id: str, events: List[Dict[str, Any]]) -> Path:
    """Write the canonical consolidated event catalog and return its path."""
    catalog = event_catalog_path(video_id)
    write_json_file(catalog, events)
    return catalog


def load_legacy_event_files(video_id: str) -> List[Dict[str, Any]]:
    """Load old per-event JSON records when the consolidated catalog is missing."""
    legacy_dir = event_dir(video_id)
    if not legacy_dir.exists():
        return []

    events: List[Dict[str, Any]] = []
    for path in sorted(legacy_dir.glob("evt_*.json")):
        data = read_json_file(path)
        if isinstance(data, dict):
            events.append(data)
    return events


def normalize_event_record(event: Dict[str, Any], video_id: str) -> Dict[str, Any]:
    """Normalize raw/legacy event JSON into the AggregatedEvent API contract."""
    source_frames = event.get("source_frames", []) or []
    first_frame_id = source_frames[0] if source_frames else None
    thumbnail_path = f"/api/v1/events/{video_id}/thumbnail/{first_frame_id}" if first_frame_id else event.get("thumbnail_path")

    return {
        "event_id": event.get("event_id", ""),
        "event_type": event.get("event_type", ""),
        "description": event.get("description") or event.get("summary", ""),
        "start_time": event.get("start_time") or event.get("timestamp_start_human", ""),
        "end_time": event.get("end_time") or event.get("timestamp_end_human", ""),
        "duration_seconds": event.get("duration_seconds", 0.0),
        "objects": event.get("objects", []),
        "activities": event.get("activities", []),
        "activity": event.get("activity", event.get("primary_activity", "")),
        "primary_object": event.get("primary_object", ""),
        "location_text": event.get("location_text", "the monitored area"),
        "scene_context": event.get("scene_context", ""),
        "real_world_time": event.get("real_world_time"),
        "actor_description": event.get("actor_description", ""),
        "participants": event.get("participants", []),
        "participant_count": event.get("participant_count", 0),
        "behavioral_flags": event.get("behavioral_flags", []),
        "confidence": event.get("confidence", 0.5),
        "narrative_sentence": event.get("narrative_sentence") or event.get("summary", ""),
        "thumbnail_path": thumbnail_path,
        "event_severity": event.get("event_severity", 15),
        "unified_text": event.get("unified_text", ""),
        "frame_events": event.get("frame_events", []),
    }
