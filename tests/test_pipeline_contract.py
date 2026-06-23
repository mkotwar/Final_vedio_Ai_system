import json

import pytest

from app.core.config import settings
from app.services.event_aggregation import EventAggregationService
from app.services.pipeline_contract import (
    event_catalog_path,
    event_dir,
    frame_catalog_path,
    write_json_file,
)
from app.services.summary_service import SummaryService


@pytest.fixture
def isolated_pipeline_dirs(tmp_path):
    original_events_dir = settings.EVENTS_DIR
    original_metadata_dir = settings.METADATA_DIR
    settings.EVENTS_DIR = tmp_path / "events"
    settings.METADATA_DIR = tmp_path / "metadata"
    settings.EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    settings.METADATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        yield
    finally:
        settings.EVENTS_DIR = original_events_dir
        settings.METADATA_DIR = original_metadata_dir


def _frame(video_id: str, idx: int, timestamp: float, caption: str = "A blue car driving on the street."):
    return {
        "frame_id": f"{video_id}_f{idx:04d}",
        "video_id": video_id,
        "timestamp_seconds": timestamp,
        "timestamp_human": f"00:00:0{idx - 1}",
        "timestamp_start_seconds": timestamp,
        "timestamp_end_seconds": timestamp + 1.0,
        "scene_type": "outdoor street",
        "caption": caption,
        "objects": [{"type": "vehicle", "subtype": "car", "color": "blue"}],
        "activities": ["driving"],
    }


def test_event_aggregation_writes_canonical_event_contract(isolated_pipeline_dirs):
    video_id = "contract-video"
    frames = [_frame(video_id, 1, 0.0), _frame(video_id, 2, 1.0)]

    raw_events = EventAggregationService.process_events(video_id, frames)

    assert len(raw_events) == 1
    assert (event_dir(video_id) / "evt_001.json").exists()
    assert event_catalog_path(video_id).exists()

    with open(event_catalog_path(video_id), "r", encoding="utf-8") as f:
        catalog = json.load(f)

    assert catalog[0]["event_id"] == "evt_001"
    assert catalog[0]["description"]
    assert catalog[0]["start_time"] == "00:00:00"
    assert catalog[0]["end_time"] == "00:00:02"


def test_summary_reads_canonical_event_catalog(isolated_pipeline_dirs):
    video_id = "summary-contract-video"
    catalog = [
        {
            "event_id": "evt_001",
            "event_type": "vehicle_movement",
            "description": "A blue car entered the monitored area.",
            "start_time": "00:00:00",
            "end_time": "00:00:03",
            "duration_seconds": 3.0,
        }
    ]
    write_json_file(event_catalog_path(video_id), catalog)

    events = SummaryService.load_events(video_id)

    assert len(events) == 1
    assert events[0].event_id == "evt_001"
    assert events[0].description == "A blue car entered the monitored area."


def test_summary_rebuilds_canonical_catalog_from_legacy_event_files(isolated_pipeline_dirs):
    video_id = "legacy-contract-video"
    legacy_dir = event_dir(video_id)
    legacy_dir.mkdir(parents=True, exist_ok=True)
    write_json_file(
        legacy_dir / "evt_001.json",
        {
            "event_id": "evt_001",
            "video_id": video_id,
            "event_type": "vehicle_entry",
            "summary": "A vehicle arrived at the main gate.",
            "timestamp_start_human": "00:00:00",
            "timestamp_end_human": "00:00:02",
            "duration_seconds": 2.0,
        },
    )

    events = SummaryService.load_events(video_id)

    assert len(events) == 1
    assert events[0].description == "A vehicle arrived at the main gate."
    assert event_catalog_path(video_id).exists()


def test_summary_regenerates_canonical_events_from_frame_catalog(isolated_pipeline_dirs):
    video_id = "frames-contract-video"
    write_json_file(frame_catalog_path(video_id), [_frame(video_id, 1, 0.0), _frame(video_id, 2, 1.0)])

    events = SummaryService.load_events(video_id)

    assert len(events) == 1
    assert "blue car" in events[0].description
    assert (event_dir(video_id) / "evt_001.json").exists()
    assert event_catalog_path(video_id).exists()
