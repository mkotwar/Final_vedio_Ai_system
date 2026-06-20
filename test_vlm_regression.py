import pytest
import json
from app.services.qwen_vlm import QwenVLMService

def test_normalization_subtype_person():
    """CASE 1: Input 'adult male'/'pedestrian' normalizes to 'person'"""
    raw_dict = {
        "scene_type": "outdoor",
        "objects": [{"id": "p1", "type": "person", "subtype": "pedestrian"}]
    }
    normalized = QwenVLMService._normalize_metadata_dict(raw_dict)
    assert normalized["objects"][0]["subtype"] == "person"

def test_normalization_subtype_customer():
    """CASE 2: Input 'shopper' normalizes to 'customer'"""
    raw_dict = {
        "scene_type": "indoor",
        "objects": [{"id": "c1", "type": "person", "subtype": "shopper"}]
    }
    normalized = QwenVLMService._normalize_metadata_dict(raw_dict)
    assert normalized["objects"][0]["subtype"] == "customer"

def test_normalization_expanded():
    """Testing expanded normalizations"""
    test_cases = {
        "guard": "person",
        "security": "person",
        "staff": "employee",
        "worker": "employee",
        "female": "person",
        "man": "person"
    }
    
    for input_sub, expected in test_cases.items():
        raw_dict = {
            "scene_type": "indoor",
            "objects": [{"id": "x1", "type": "person", "subtype": input_sub}]
        }
        normalized = QwenVLMService._normalize_metadata_dict(raw_dict)
        assert normalized["objects"][0]["subtype"] == expected, f"Failed on {input_sub}"

def validate_regurgitation(raw_json_str: str) -> bool:
    """Helper mimicking the validation_runner_vlm check"""
    PLACEHOLDER_PATTERNS = ["e.g.", "example", "unique id"]
    lower_str = raw_json_str.lower()
    for pattern in PLACEHOLDER_PATTERNS:
        if pattern in lower_str:
            return True # Regurgitated
    return False

def test_schema_regurgitation_detection():
    """CASE 3: Output level validation of regurgitation patterns."""
    bad_output = '{"objects": [{"id": "unique id e.g. person_1"}]}'
    assert validate_regurgitation(bad_output) == True
    
    good_output = '{"objects": [{"id": "person_1"}]}'
    assert validate_regurgitation(good_output) == False

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
