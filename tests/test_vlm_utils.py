import pytest
import json
from app.services.vlm_utils import (
    clean_json_response,
    normalize_metadata_dict,
    generate_search_text,
    format_timestamp_human_vlm,
)


def test_clean_json_response():
    """Verify clean_json_response removes markdown blocks (e.g. ```json ... ```) successfully."""
    raw_response = "```json\n{\n  \"status\": \"success\"\n}\n```"
    cleaned = clean_json_response(raw_response)
    parsed = json.loads(cleaned)
    assert parsed["status"] == "success"


def test_normalize_metadata_dict_pedestrian():
    """Verify normalize_metadata_dict normalizes pedestrian subtype to person."""
    raw_dict = {
        "scene_type": "outdoor",
        "objects": [{"id": "p1", "type": "person", "subtype": "pedestrian"}]
    }
    normalized = normalize_metadata_dict(raw_dict)
    assert normalized["objects"][0]["subtype"] == "person"


def test_normalize_metadata_dict_shopper():
    """Verify normalize_metadata_dict normalizes shopper subtype to customer."""
    raw_dict = {
        "scene_type": "indoor",
        "objects": [{"id": "c1", "type": "person", "subtype": "shopper"}]
    }
    normalized = normalize_metadata_dict(raw_dict)
    assert normalized["objects"][0]["subtype"] == "customer"


def test_generate_search_text_non_empty():
    """Verify generate_search_text produces non-empty output for typical metadata."""
    meta = {
        "scene_type": "indoor office",
        "scene_description": "A quiet office setting",
        "caption": "An empty meeting room",
        "activities": ["working", "typing"],
        "keywords": ["office", "desk"],
        "objects": [{"subtype": "person", "color": "blue", "attributes": ["sitting"]}]
    }
    search_text = generate_search_text(meta)
    assert len(search_text) > 0
    assert "indoor" in search_text
    assert "blue person" in search_text


def test_normalize_metadata_dict_activity_dicts():
    """Verify normalize_metadata_dict converts dict-based activities to strings."""
    raw_dict = {
        "scene_type": "outdoor",
        "activities": [
            {"subject_id": "person_1", "relation": "standing"},
            {"type": "walking"},
            "running"
        ]
    }
    normalized = normalize_metadata_dict(raw_dict)
    assert normalized["activities"] == ["standing", "walking", "running"]
