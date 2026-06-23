"""Pytest integration tests for the video summary API endpoint.
"""

import json
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings


def test_summary_api_success(tmp_path):
    """Verify that the summary API successfully loads the consolidated events file and returns statistics, notable events, and a timeline."""
    original_metadata_dir = settings.METADATA_DIR
    settings.METADATA_DIR = tmp_path / "metadata"
    settings.METADATA_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        video_id = "mock-video-id"
        # Mock events data aligned to the AggregatedEvent schema
        mock_events = [
            {
                "event_id": "evt_001",
                "event_type": "vehicle_entry",
                "description": "A blue car driving down a busy city street.",
                "start_time": "00:00:00",
                "end_time": "00:00:03",
                "duration_seconds": 3.0
            },
            {
                "event_id": "evt_002",
                "event_type": "indoor_activity",
                "description": "Suspicious activity detected in office.",
                "start_time": "00:00:03",
                "end_time": "00:01:05",
                "duration_seconds": 62.0
            },
            {
                "event_id": "evt_003",
                "event_type": "motion_detected",
                "description": "Motion detected in empty corridor.",
                "start_time": "00:01:05",
                "end_time": "00:01:10",
                "duration_seconds": 5.0
            }
        ]
        
        # Save mock events array to the mocked metadata directory
        events_path = settings.METADATA_DIR / f"{video_id}_events.json"
        with open(events_path, "w", encoding="utf-8") as f:
            json.dump(mock_events, f)
            
        with TestClient(app) as client:
            response = client.get(f"/api/v1/videos/{video_id}/summary")
            assert response.status_code == 200
            data = response.json()
            assert data["video_id"] == video_id
            assert data["status"] == "success"
            assert len(data["timeline"]) == 3
            assert len(data["notable_events"]) >= 1  # evt_002 is > 60s or suspicious
            assert "evt_002" in [x["event_id"] for x in data["notable_events"]]
            
    finally:
        settings.METADATA_DIR = original_metadata_dir


def test_summary_api_no_events(tmp_path):
    """Verify that when no events catalog exists, the API returns a graceful status: no_events response."""
    original_metadata_dir = settings.METADATA_DIR
    settings.METADATA_DIR = tmp_path / "metadata"
    settings.METADATA_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        video_id = "non-existent-video-id"
        with TestClient(app) as client:
            response = client.get(f"/api/v1/videos/{video_id}/summary")
            assert response.status_code == 200
            data = response.json()
            assert data["video_id"] == video_id
            assert data["status"] == "no_events"
            assert data["overview"] == "No significant incidents detected."
            assert len(data["timeline"]) == 0
            assert len(data["notable_events"]) == 0
            
    finally:
        settings.METADATA_DIR = original_metadata_dir


def test_summary_api_fallback_to_individual_events(tmp_path):
    """Verify that when the consolidated events catalog does not exist, the API successfully
    falls back to individual events folder, rebuilds the catalog, saves it, and returns success.
    """
    original_metadata_dir = settings.METADATA_DIR
    original_events_dir = settings.EVENTS_DIR
    settings.METADATA_DIR = tmp_path / "metadata"
    settings.METADATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.EVENTS_DIR = tmp_path / "events"
    
    try:
        video_id = "legacy-video-id"
        legacy_events_dir = settings.EVENTS_DIR / video_id
        legacy_events_dir.mkdir(parents=True, exist_ok=True)
        
        # Create individual event files
        evt_data_1 = {
            "event_id": "evt_001",
            "video_id": video_id,
            "event_type": "vehicle_entry",
            "summary": "A vehicle arrived at the main gate.",
            "timestamp_start_human": "00:00:00",
            "timestamp_end_human": "00:00:02",
            "duration_seconds": 2.0
        }
        evt_data_2 = {
            "event_id": "evt_002",
            "video_id": video_id,
            "event_type": "pedestrian_crossing",
            "summary": "A pedestrian crossed the road.",
            "timestamp_start_human": "00:00:02",
            "timestamp_end_human": "00:00:05",
            "duration_seconds": 3.0
        }
        
        with open(legacy_events_dir / "evt_001.json", "w", encoding="utf-8") as f:
            json.dump(evt_data_1, f)
        with open(legacy_events_dir / "evt_002.json", "w", encoding="utf-8") as f:
            json.dump(evt_data_2, f)
            
        with TestClient(app) as client:
            # consolidated file does not exist initially
            consolidated_path = settings.METADATA_DIR / f"{video_id}_events.json"
            assert not consolidated_path.exists()
            
            response = client.get(f"/api/v1/videos/{video_id}/summary")
            assert response.status_code == 200
            data = response.json()
            
            # Verify the API responds with success and correct data
            assert data["video_id"] == video_id
            assert data["status"] == "success"
            assert len(data["timeline"]) == 2
            assert data["statistics"]["total_events"] == 2
            
            # Verify that the consolidated file was created as a side effect
            assert consolidated_path.exists()
            with open(consolidated_path, "r", encoding="utf-8") as f:
                catalog = json.load(f)
            assert len(catalog) == 2
            assert catalog[0]["event_id"] == "evt_001"
            assert catalog[0]["description"] == "A vehicle arrived at the main gate."
            assert catalog[1]["event_id"] == "evt_002"
            assert catalog[1]["description"] == "A pedestrian crossed the road."
            
    finally:
        settings.METADATA_DIR = original_metadata_dir
        settings.EVENTS_DIR = original_events_dir


def test_summary_api_fallback_to_frames_metadata(tmp_path):
    """Verify that when both the consolidated events catalog and individual events folder do not exist,
    the API dynamically generates events from the frame metadata JSON file, caches it, and returns success.
    """
    original_metadata_dir = settings.METADATA_DIR
    original_events_dir = settings.EVENTS_DIR
    settings.METADATA_DIR = tmp_path / "metadata"
    settings.METADATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.EVENTS_DIR = tmp_path / "events"
    settings.EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        video_id = "frame-legacy-video-id"
        
        # Create mock frame metadata list inside {video_id}_frames.json
        frames_data = [
            {
                "frame_id": f"{video_id}_f0001",
                "video_id": video_id,
                "timestamp_seconds": 0.0,
                "timestamp_human": "00:00:00",
                "timestamp_start_seconds": 0.0,
                "timestamp_end_seconds": 1.0,
                "scene_type": "outdoor street",
                "caption": "A blue car driving on the street.",
                "objects": [{"type": "vehicle", "subtype": "car", "color": "blue"}],
                "activities": ["driving"],
            },
            {
                "frame_id": f"{video_id}_f0002",
                "video_id": video_id,
                "timestamp_seconds": 1.0,
                "timestamp_human": "00:00:01",
                "timestamp_start_seconds": 1.0,
                "timestamp_end_seconds": 2.0,
                "scene_type": "outdoor street",
                "caption": "A blue car driving on the street.",
                "objects": [{"type": "vehicle", "subtype": "car", "color": "blue"}],
                "activities": ["driving"],
            }
        ]
        
        frames_path = settings.METADATA_DIR / f"{video_id}_frames.json"
        with open(frames_path, "w", encoding="utf-8") as f:
            json.dump(frames_data, f)
            
        with TestClient(app) as client:
            # Neither consolidated file nor events directory exists
            consolidated_path = settings.METADATA_DIR / f"{video_id}_events.json"
            events_dir = settings.EVENTS_DIR / video_id
            assert not consolidated_path.exists()
            assert not events_dir.exists()
            
            response = client.get(f"/api/v1/videos/{video_id}/summary")
            assert response.status_code == 200
            data = response.json()
            
            # Verify the API responds with success and correct data
            assert data["video_id"] == video_id
            assert data["status"] == "success"
            assert len(data["timeline"]) == 1  # 2 similar frames aggregated into 1 event
            assert data["statistics"]["total_events"] == 1
            
            # Verify that the events directory and consolidated catalog were created on-the-fly
            assert events_dir.exists()
            assert (events_dir / "evt_001.json").exists()
            assert consolidated_path.exists()
            
            with open(consolidated_path, "r", encoding="utf-8") as f:
                catalog = json.load(f)
            assert len(catalog) == 1
            assert catalog[0]["event_id"] == "evt_001"
            assert "blue car" in catalog[0]["description"]
            
    finally:
        settings.METADATA_DIR = original_metadata_dir
        settings.EVENTS_DIR = original_events_dir
