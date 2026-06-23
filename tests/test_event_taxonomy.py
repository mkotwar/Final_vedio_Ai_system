from app.services.event_taxonomy import (
    get_event_search_boost,
    get_event_severity,
    get_notable_severity,
    is_notable_event_type,
    normalize_event_type,
)


def test_event_type_aliases_normalize_to_canonical_contract():
    assert normalize_event_type("vehicle_collision") == "collision_or_accident"
    assert normalize_event_type("restricted-area activity") == "intrusion"
    assert normalize_event_type("fire smoke detected") == "fire_incident"
    assert normalize_event_type("person_fall") == "fall_incident"
    assert normalize_event_type("none") == "normal_activity"


def test_event_severity_matches_aggregation_contract():
    assert get_event_severity("collision_or_accident") == 100
    assert get_event_severity("fire_incident") == 100
    assert get_event_severity("weapon_drawn") == 95
    assert get_event_severity("vehicle_movement") == 30
    assert get_event_severity("normal_activity") == 10
    assert get_event_severity("unknown_future_event") == 15


def test_notable_contract_supports_vlm_aliases():
    assert is_notable_event_type("collision")
    assert is_notable_event_type("restricted_area_activity")
    assert is_notable_event_type("physical_altercation")
    assert get_notable_severity("fire") == "critical"
    assert get_notable_severity("fall") == "medium"
    assert not is_notable_event_type("vehicle_movement")


def test_search_boost_contract_preserves_existing_priority():
    assert get_event_search_boost("weapon_drawn") == 1.00
    assert get_event_search_boost("fire_incident") == 0.75
    assert get_event_search_boost("collision") == 0.50
    assert get_event_search_boost("normal_activity") == 0.0
