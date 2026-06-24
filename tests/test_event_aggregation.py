import json

import pytest

from app.core.config import settings
from app.services.event_aggregation import EventAggregationService


@pytest.fixture
def isolated_event_dirs(tmp_path):
    original_events_dir = settings.EVENTS_DIR
    original_metadata_dir = settings.METADATA_DIR
    settings.EVENTS_DIR = tmp_path / "events"
    settings.METADATA_DIR = tmp_path / "metadata"
    settings.METADATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        yield
    finally:
        settings.EVENTS_DIR = original_events_dir
        settings.METADATA_DIR = original_metadata_dir


def test_jaccard_similarity():
    """Verify Jaccard similarity computes intersection over union correctly."""
    # Test identical lists
    assert EventAggregationService.jaccard_similarity(["car", "person"], ["car", "person"]) == 1.0
    # Test empty lists
    assert EventAggregationService.jaccard_similarity([], []) == 1.0
    # Test partially overlapping lists
    assert EventAggregationService.jaccard_similarity(["car", "person"], ["car", "truck"]) == 1.0 / 3.0
    # Test case insensitivity and whitespace stripping
    assert EventAggregationService.jaccard_similarity([" CAR  ", "person"], ["car", "PERSON"]) == 1.0
    # Test completely disjoint lists
    assert EventAggregationService.jaccard_similarity(["car"], ["person"]) == 0.0


def test_calculate_similarity():
    """Verify calculation of average similarity on frame pairs."""
    frame1 = {
        "caption": "A blue car driving down a busy city street.",
        "scene_type": "outdoor street",
        "objects": [{"type": "vehicle", "subtype": "car", "color": "blue"}],
        "activities": ["driving"],
    }
    frame2 = {
        "caption": "A blue car driving down a busy city street.",
        "scene_type": "outdoor street",
        "objects": [{"type": "vehicle", "subtype": "car", "color": "blue"}],
        "activities": ["driving"],
    }
    # Identical frames should return 1.0 similarity
    assert EventAggregationService.calculate_similarity(frame1, frame2) == 1.0

    frame3 = {
        "caption": "A laptop open on a table in an office.",
        "scene_type": "indoor office",
        "objects": [{"type": "electronics", "subtype": "laptop", "color": "silver"}],
        "activities": ["working"],
    }
    # Completely different frames should return a low similarity score
    assert EventAggregationService.calculate_similarity(frame1, frame3) < 0.25


def test_extract_real_world_time_prefers_full_cctv_hour():
    frames = [
        {"ocr": {"detected_text": ["Office Cam", "1:02:24"]}},
        {"ocr": {"detected_text": ["2026-06-24 14:02:24"]}},
        {"ocr": {"detected_text": ["2026-06-24 14:02:25"]}},
    ]

    assert EventAggregationService._extract_real_world_time(frames) == "14:02:24"


def test_extract_real_world_time_ignores_single_weak_read():
    frames = [
        {"ocr": {"detected_text": ["Office Cam", "1:02:24"]}},
        {"ocr": {"detected_text": ["Empty office space"]}},
    ]

    assert EventAggregationService._extract_real_world_time(frames) is None


def test_event_aggregation_processing(isolated_event_dirs):
    """Verify frames are grouped into events and serialized to the target directory correctly."""
    original_threshold = settings.EVENT_SIMILARITY_THRESHOLD
    settings.EVENT_SIMILARITY_THRESHOLD = 0.60
    
    try:
        video_id = "test-video-uuid"
        
        # Create a list of frames that should group into two separate events
        # Event 1: Outdoor street at seconds 0, 1, 2
        # Event 2: Indoor office at seconds 3, 4
        frames = [
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
            },
            {
                "frame_id": f"{video_id}_f0003",
                "video_id": video_id,
                "timestamp_seconds": 2.0,
                "timestamp_human": "00:00:02",
                "timestamp_start_seconds": 2.0,
                "timestamp_end_seconds": 3.0,
                "scene_type": "outdoor street",
                "caption": "A blue car driving on the street with a pedestrian.",
                "objects": [
                    {"type": "vehicle", "subtype": "car", "color": "blue"},
                    {"type": "pedestrian", "subtype": "person", "color": "black"}
                ],
                "activities": ["driving", "walking"],
            },
            {
                "frame_id": f"{video_id}_f0004",
                "video_id": video_id,
                "timestamp_seconds": 3.0,
                "timestamp_human": "00:00:03",
                "timestamp_start_seconds": 3.0,
                "timestamp_end_seconds": 4.0,
                "scene_type": "indoor office",
                "caption": "A laptop open on a table.",
                "objects": [{"type": "electronics", "subtype": "laptop", "color": "silver"}],
                "activities": ["working"],
            },
            {
                "frame_id": f"{video_id}_f0005",
                "video_id": video_id,
                "timestamp_seconds": 4.0,
                "timestamp_human": "00:00:04",
                "timestamp_start_seconds": 4.0,
                "timestamp_end_seconds": 5.0,
                "scene_type": "indoor office",
                "caption": "A laptop open on a table.",
                "objects": [{"type": "electronics", "subtype": "laptop", "color": "silver"}],
                "activities": ["working"],
            },
        ]
        
        events = EventAggregationService.process_events(video_id, frames)
        
        # We expect exactly 2 events
        assert len(events) == 2
        
        evt1 = events[0]
        assert evt1["event_id"] == "evt_001"
        assert evt1["video_id"] == video_id
        assert evt1["start_time"] == "00:00:00"
        assert evt1["end_time"] == "00:00:03"
        assert evt1["timestamp_start_seconds"] == 0.0
        assert evt1["timestamp_end_seconds"] == 3.0
        assert evt1["duration_seconds"] == 3.0
        assert evt1["frame_count"] == 3
        assert evt1["event_type"] == "vehicle_movement"
        assert evt1["source_frames"] == [f"{video_id}_f0001", f"{video_id}_f0002", f"{video_id}_f0003"]
        assert len(evt1["objects"]) == 2
        
        evt2 = events[1]
        assert evt2["event_id"] == "evt_002"
        assert evt2["start_time"] == "00:00:03"
        assert evt2["end_time"] == "00:00:05"
        assert evt2["timestamp_start_seconds"] == 3.0
        assert evt2["timestamp_end_seconds"] == 5.0
        assert evt2["duration_seconds"] == 2.0
        assert evt2["frame_count"] == 2
        assert evt2["event_type"] == "normal_activity"
        assert evt2["source_frames"] == [f"{video_id}_f0004", f"{video_id}_f0005"]
        assert len(evt2["objects"]) == 1
        
        # Verify JSON writing
        evt_dir = settings.EVENTS_DIR / video_id
        assert evt_dir.exists()
        assert (evt_dir / "evt_001.json").exists()
        assert (evt_dir / "evt_002.json").exists()
        
        with open(evt_dir / "evt_001.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["event_id"] == "evt_001"
            assert data["frame_count"] == 3

        # Verify consolidated array catalog writing
        consolidated_file = settings.METADATA_DIR / f"{video_id}_events.json"
        assert consolidated_file.exists()
        with open(consolidated_file, "r", encoding="utf-8") as f:
            catalog = json.load(f)
            assert len(catalog) == 2
            assert catalog[0]["event_id"] == "evt_001"
            assert catalog[0]["event_type"] == "vehicle_movement"
            assert catalog[0]["description"] == "Vehicle movement involving person was detected at the monitored area. Other participants: blue car."
            assert catalog[0]["start_time"] == "00:00:00"
            assert catalog[0]["end_time"] == "00:00:03"
            
    finally:
        settings.EVENT_SIMILARITY_THRESHOLD = original_threshold


def test_person_is_selected_over_furniture(isolated_event_dirs):
    frame = {
        "frame_id": "f01",
        "timestamp_seconds": 0.0,
        "objects": [
            {"id": "desk_1", "type": "furniture", "subtype": "desk", "attributes": []},
            {"id": "chair_1", "type": "furniture", "subtype": "chair", "attributes": []},
            {"id": "person_1", "type": "person", "subtype": "employee", "attributes": []},
        ],
        "activities": ["walking"],
        "relationships": [],
        "location_context": [],
    }

    events = EventAggregationService.process_events("test_video_1", [frame])

    assert len(events) == 1
    assert events[0]["primary_object"] == "Person"
    assert events[0]["event_type"] == "pedestrian_activity"


def test_vehicle_is_selected_over_furniture(isolated_event_dirs):
    frame = {
        "frame_id": "f01",
        "timestamp_seconds": 0.0,
        "objects": [
            {"id": "monitor_1", "type": "electronics", "subtype": "monitor", "attributes": []},
            {"id": "vehicle_1", "type": "vehicle", "subtype": "car", "attributes": []},
        ],
        "activities": ["moving"],
        "relationships": [],
        "location_context": [],
    }

    events = EventAggregationService.process_events("test_video_2", [frame])

    assert len(events) == 1
    assert events[0]["primary_object"] == "Vehicle"


def test_empty_furniture_scene_is_not_classified_as_person_or_vehicle(isolated_event_dirs):
    frame = {
        "frame_id": "f01",
        "timestamp_seconds": 0.0,
        "scene_type": "office",
        "objects": [
            {"id": "desk_1", "type": "furniture", "subtype": "desk", "attributes": []},
            {"id": "chair_1", "type": "furniture", "subtype": "chair", "attributes": []},
        ],
        "activities": ["waiting"],
        "relationships": [],
        "location_context": [],
    }

    events = EventAggregationService.process_events("test_video_3", [frame])

    assert len(events) == 1
    assert events[0]["primary_object"] not in {"Person", "Vehicle"}


def test_empty_office_scene_does_not_create_event(isolated_event_dirs):
    frame = {
        "frame_id": "f01",
        "timestamp_seconds": 0.0,
        "timestamp_start_seconds": 0.0,
        "timestamp_end_seconds": 1.0,
        "scene_type": "office",
        "scene_description": "Empty office space with desks and equipment.",
        "caption": "No visible activity in the office space.",
        "objects": [],
        "activities": [],
        "keywords": ["office", "empty"],
    }

    events = EventAggregationService.process_events("test_video_empty", [frame])

    assert events == []


def test_same_person_id_is_deduplicated_across_appearance_variants(isolated_event_dirs):
    frames = [
        {
            "frame_id": "f01",
            "timestamp_seconds": 0.0,
            "objects": [
                {"id": "person_1", "type": "person", "subtype": "employee", "color": "purple", "attributes": ["wearing purple shirt"]},
            ],
            "activities": ["walking"],
            "caption": "Person walking in office.",
        },
        {
            "frame_id": "f02",
            "timestamp_seconds": 1.0,
            "objects": [
                {"id": "person_1", "type": "person", "subtype": "employee", "color": "purple shirt, blue jeans", "attributes": ["short hair"]},
            ],
            "activities": ["walking"],
            "caption": "Person walking near desk.",
        },
    ]

    events = EventAggregationService.process_events("test_video_dedupe", frames)

    assert len(events) == 1
    assert events[0]["participant_count"] == 1
    assert events[0]["participants"] == []
    assert len(events[0]["objects"]) == 1
