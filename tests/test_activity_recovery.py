"""Unit tests for the ActivityRecoveryService.

Tests cover:
  - recover_from_attributes()  — attribute-level mining
  - recover_from_caption()     — caption regex mining
  - recover_from_keywords()    — keyword-level mining
  - recover()                  — full priority cascade
  - apply()                    — in-place dict mutation

All tests use synthetic frame data that mirrors patterns observed in the
real pipeline output in data/metadata/*_frames.json.
"""

import pytest
from app.services.activity_recovery import ActivityRecoveryService


# ------------------------------------------------------------------ #
# recover_from_attributes tests                                        #
# ------------------------------------------------------------------ #

class TestRecoverFromAttributes:

    def test_parked_position_attribute(self):
        objects = [{"type": "vehicle", "subtype": "sedan", "color": "silver",
                    "attributes": ["parked position angled towards camera", "closed windows"]}]
        result = ActivityRecoveryService.recover_from_attributes(objects)
        assert "vehicle parked" in result

    def test_stationary_attribute(self):
        objects = [{"type": "vehicle", "subtype": "sedan", "color": "silver",
                    "attributes": ["stationary", "taillights lit up"]}]
        result = ActivityRecoveryService.recover_from_attributes(objects)
        assert "vehicle stationary" in result

    def test_taillights_lit_attribute(self):
        """Taillights lit → vehicle stationary (common in CCTV footage)."""
        objects = [{"type": "vehicle", "subtype": "car", "color": "black",
                    "attributes": ["taillights lit up", "partially visible roof"]}]
        result = ActivityRecoveryService.recover_from_attributes(objects)
        assert "vehicle stationary" in result

    def test_person_seated_on_it(self):
        """'person seated on it' is a motorcycle/vehicle occupancy indicator."""
        objects = [{"type": "vehicle", "subtype": "motorcycle", "color": "black",
                    "attributes": ["rear lights visible", "person seated on it", "license plate attached"]}]
        result = ActivityRecoveryService.recover_from_attributes(objects)
        assert "vehicle occupied" in result

    def test_walking_away_attribute(self):
        objects = [{"type": "pedestrian", "subtype": "person walking away", "color": "pink",
                    "attributes": ["walking away", "carrying plastic bag"]}]
        result = ActivityRecoveryService.recover_from_attributes(objects)
        assert "walking" in result
        assert "carrying object" in result

    def test_walking_towards_attribute(self):
        objects = [{"type": "pedestrian", "subtype": "", "color": "blue",
                    "attributes": ["walking towards crosswalk", "short dark hair"]}]
        result = ActivityRecoveryService.recover_from_attributes(objects)
        assert "walking" in result

    def test_crossing_attribute(self):
        objects = [{"type": "pedestrian", "subtype": "person", "color": "pink",
                    "attributes": ["crossing road", "carrying bag"]}]
        result = ActivityRecoveryService.recover_from_attributes(objects)
        assert "crossing road" in result

    def test_empty_attributes_list(self):
        objects = [{"type": "vehicle", "subtype": "sedan", "color": "gray", "attributes": []}]
        result = ActivityRecoveryService.recover_from_attributes(objects)
        assert result == []

    def test_no_objects(self):
        result = ActivityRecoveryService.recover_from_attributes([])
        assert result == []

    def test_multiple_objects_combine(self):
        """Activities from multiple objects should all be collected."""
        objects = [
            {"type": "vehicle", "subtype": "sedan", "color": "silver",
             "attributes": ["stationary"]},
            {"type": "pedestrian", "subtype": "person", "color": "pink",
             "attributes": ["walking away"]},
        ]
        result = ActivityRecoveryService.recover_from_attributes(objects)
        assert "vehicle stationary" in result
        assert "walking" in result

    def test_no_duplicates(self):
        """Same pattern across multiple objects should not produce duplicates."""
        objects = [
            {"type": "vehicle", "subtype": "car", "color": "red", "attributes": ["stationary"]},
            {"type": "vehicle", "subtype": "truck", "color": "white", "attributes": ["stationary"]},
        ]
        result = ActivityRecoveryService.recover_from_attributes(objects)
        assert result.count("vehicle stationary") == 1

    def test_non_list_attributes_skipped(self):
        """Handles malformed attribute fields gracefully."""
        objects = [{"type": "vehicle", "subtype": "car", "color": "blue",
                    "attributes": "stationary"}]
        # Non-list attributes skipped — no crash
        result = ActivityRecoveryService.recover_from_attributes(objects)
        assert isinstance(result, list)

    def test_case_insensitive_matching(self):
        objects = [{"type": "vehicle", "subtype": "car", "color": "blue",
                    "attributes": ["STATIONARY"]}]
        result = ActivityRecoveryService.recover_from_attributes(objects)
        assert "vehicle stationary" in result


# ------------------------------------------------------------------ #
# recover_from_caption tests                                           #
# ------------------------------------------------------------------ #

class TestRecoverFromCaption:

    def test_parked_caption(self):
        caption = "A silver sedan parked at an angle near a concrete sidewalk under bright daylight."
        result = ActivityRecoveryService.recover_from_caption(caption)
        assert "vehicle parked" in result

    def test_parked_curbside_caption(self):
        caption = "a silver sedan parked curbside amidst a quiet urban street lined with large greenery."
        result = ActivityRecoveryService.recover_from_caption(caption)
        assert "vehicle parked" in result

    def test_driving_caption(self):
        caption = "A green trash bin lid viewed from within a car as it drives past marked roads."
        result = ActivityRecoveryService.recover_from_caption(caption)
        assert "driving" in result

    def test_walking_caption(self):
        caption = "A person walks along an urban street lined with tree shadows."
        result = ActivityRecoveryService.recover_from_caption(caption)
        assert "walking" in result

    def test_walking_variant_walking(self):
        caption = "A pedestrian is walking across the intersection."
        result = ActivityRecoveryService.recover_from_caption(caption)
        assert "walking" in result

    def test_crossing_caption(self):
        caption = "A person wearing pink walks across a city street at a designated crosswalk."
        result = ActivityRecoveryService.recover_from_caption(caption)
        assert "crossing road" in result

    def test_crossing_caption_intersection(self):
        caption = "A pedestrian is walking across the intersection."
        result = ActivityRecoveryService.recover_from_caption(caption)
        assert "crossing road" in result

    def test_standing_caption(self):
        caption = "Two guards standing near the entrance gate."
        result = ActivityRecoveryService.recover_from_caption(caption)
        assert "standing" in result

    def test_entering_caption(self):
        caption = "A red car entering the main gate at 8pm."
        result = ActivityRecoveryService.recover_from_caption(caption)
        assert "entering" in result

    def test_reversing_caption(self):
        caption = "A white truck reversing into the loading bay."
        result = ActivityRecoveryService.recover_from_caption(caption)
        assert "reversing" in result

    def test_empty_caption(self):
        result = ActivityRecoveryService.recover_from_caption("")
        assert result == []

    def test_no_description_available_caption(self):
        result = ActivityRecoveryService.recover_from_caption("No description available.")
        assert result == []

    def test_no_description_mixed_case(self):
        result = ActivityRecoveryService.recover_from_caption("no description available.")
        assert result == []

    def test_irrelevant_caption(self):
        caption = "Long tree shadows stretch across smooth asphalt roadway."
        result = ActivityRecoveryService.recover_from_caption(caption)
        assert result == []

    def test_no_duplicates_in_caption(self):
        """Multiple pattern matches for same label should produce one entry."""
        caption = "The car is parked and was parked yesterday."
        result = ActivityRecoveryService.recover_from_caption(caption)
        assert result.count("vehicle parked") == 1

    def test_case_insensitive_caption(self):
        caption = "A VEHICLE IS PARKED ON THE STREET."
        result = ActivityRecoveryService.recover_from_caption(caption)
        assert "vehicle parked" in result

    def test_seated_caption(self):
        caption = "A person seated on the motorcycle waits at the traffic light."
        result = ActivityRecoveryService.recover_from_caption(caption)
        assert "seated" in result

    def test_carrying_caption(self):
        caption = "A woman carrying a shopping bag walks through the parking lot."
        result = ActivityRecoveryService.recover_from_caption(caption)
        assert "carrying object" in result
        assert "walking" in result


# ------------------------------------------------------------------ #
# recover_from_keywords tests                                          #
# ------------------------------------------------------------------ #

class TestRecoverFromKeywords:

    def test_vehicle_parked_keyword(self):
        keywords = ["car parked", "urban scene", "shade from trees"]
        result = ActivityRecoveryService.recover_from_keywords(keywords)
        assert "vehicle parked" in result

    def test_pedestrian_crossing_keyword(self):
        keywords = ["pedestrian crossing", "street view", "urban setting"]
        result = ActivityRecoveryService.recover_from_keywords(keywords)
        assert "crossing road" in result

    def test_motorbike_rider_keyword(self):
        keywords = ["street view", "motorbike rider", "urban setting", "daytime lighting"]
        result = ActivityRecoveryService.recover_from_keywords(keywords)
        assert "riding motorcycle" in result

    def test_pedestrian_activity_keyword(self):
        keywords = ["pedestrian activity", "crossing", "urban setting"]
        result = ActivityRecoveryService.recover_from_keywords(keywords)
        assert "walking" in result

    def test_traveling_keyword(self):
        keywords = ["bus stop", "public transport", "traveling", "city life"]
        result = ActivityRecoveryService.recover_from_keywords(keywords)
        assert "moving" in result

    def test_empty_keywords(self):
        result = ActivityRecoveryService.recover_from_keywords([])
        assert result == []

    def test_none_keywords(self):
        result = ActivityRecoveryService.recover_from_keywords(None)
        assert result == []

    def test_no_matching_keywords(self):
        keywords = ["daylight", "shadows", "trees", "pavement"]
        result = ActivityRecoveryService.recover_from_keywords(keywords)
        assert result == []


# ------------------------------------------------------------------ #
# recover() — full priority cascade tests                              #
# ------------------------------------------------------------------ #

class TestRecover:

    def test_original_activities_are_normalized(self):
        frame = {
            "activities": ["crossing", "walking_with", "none"],
            "objects": [],
            "caption": "",
            "keywords": [],
        }
        activities, source = ActivityRecoveryService.recover(frame)
        assert activities == ["crossing road", "walking"]
        assert source == "original"

    def test_original_activities_preserved(self):
        """Non-empty activities must be returned as-is without recovery."""
        frame = {
            "activities": ["crossing road"],
            "objects": [],
            "caption": "",
            "keywords": [],
        }
        activities, source = ActivityRecoveryService.recover(frame)
        assert activities == ["crossing road"]
        assert source == "original"

    def test_attributes_take_priority_over_caption(self):
        """Attributes are checked before caption; if attrs succeed, caption is skipped."""
        frame = {
            "activities": [],
            "objects": [{"type": "vehicle", "subtype": "car", "color": "silver",
                         "attributes": ["stationary"]}],
            "caption": "A silver car parked on the street.",
            "keywords": [],
        }
        activities, source = ActivityRecoveryService.recover(frame)
        assert source == "attributes"
        assert "vehicle stationary" in activities

    def test_caption_fallback_when_attributes_empty(self):
        """Caption is used when object attributes yield nothing."""
        frame = {
            "activities": [],
            "objects": [{"type": "road surface", "subtype": "asphalt", "color": "gray",
                         "attributes": []}],
            "caption": "A silver sedan parked curbside amidst a quiet urban street.",
            "keywords": [],
        }
        activities, source = ActivityRecoveryService.recover(frame)
        assert source == "caption"
        assert "vehicle parked" in activities

    def test_keyword_fallback_when_attributes_and_caption_empty(self):
        """Keywords are used when both attributes and caption yield nothing."""
        frame = {
            "activities": [],
            "objects": [{"type": "road surface", "subtype": "asphalt", "color": "gray",
                         "attributes": []}],
            "caption": "Long tree shadows across the asphalt road.",
            "keywords": ["motorbike rider", "urban setting"],
        }
        activities, source = ActivityRecoveryService.recover(frame)
        assert source == "keywords"
        assert "riding motorcycle" in activities

    def test_none_returned_when_all_sources_empty(self):
        """Frames with only road/shadow objects genuinely have no recoverable activity."""
        frame = {
            "activities": [],
            "objects": [{"type": "roadway", "subtype": "asphalt", "color": "gray",
                         "attributes": []}],
            "caption": "Long tree shadows cast across a gray asphalt road.",
            "keywords": ["daylight", "shadows", "trees"],
        }
        activities, source = ActivityRecoveryService.recover(frame)
        assert source == "none"
        assert activities == []

    def test_real_pattern_sedan_parked_position(self):
        """Mirrors actual frame f0042 from data/metadata/755dfad5_frames.json."""
        frame = {
            "activities": [],
            "objects": [
                {"type": "vehicle", "subtype": "sedan", "color": "silver/grey metallic finish",
                 "attributes": ["closed windows", "parked position angled towards camera",
                                "shadow cast by nearby tree branches"]},
            ],
            "caption": "No description available.",
            "keywords": ["car parked", "urban scene", "shade from trees"],
        }
        activities, source = ActivityRecoveryService.recover(frame)
        assert source == "attributes"
        assert "vehicle parked" in activities

    def test_real_pattern_motorcycle_occupied(self):
        """Mirrors actual frame f0028 from data/metadata/755dfad5_frames.json."""
        frame = {
            "activities": [],
            "objects": [
                {"type": "vehicle", "subtype": "motorcycle", "color": "black",
                 "attributes": ["rear lights visible", "license plate attached",
                                "person seated on it"]},
            ],
            "caption": "No description available.",
            "keywords": ["street view", "motorbike rider", "urban setting"],
        }
        activities, source = ActivityRecoveryService.recover(frame)
        # attributes takes priority ("person seated on it" → "vehicle occupied")
        assert source == "attributes"
        assert "vehicle occupied" in activities

    def test_real_pattern_sedan_parked_caption(self):
        """Mirrors actual frame f0043 (sedan, parked in caption, no useful attrs)."""
        frame = {
            "activities": [],
            "objects": [
                {"type": "vehicle", "subtype": "sedan", "color": "silver",
                 "attributes": ["modern design", "sunroof visible", "license plate DL10CZ9339"]},
            ],
            "caption": "a silver sedan parked curbside amidst a quiet urban street lined with large greenery.",
            "keywords": ["car", "parking lot", "daylight"],
        }
        # "modern design", "sunroof visible", "license plate" don't match any attribute rule
        # → falls through to caption → "parked" matches
        activities, source = ActivityRecoveryService.recover(frame)
        assert source in ("attributes", "caption")
        assert "vehicle parked" in activities

    def test_real_pattern_walking_caption(self):
        """Mirrors actual frame f0024 (person walks, no person object type detected)."""
        frame = {
            "activities": [],
            "objects": [
                {"type": "road surface", "subtype": "asphalt pavement", "color": "gray",
                 "attributes": ["smooth", "wide lane marked by white lines on one side."]},
                {"type": "sidewalk", "subtype": "concrete edge", "color": "brownish-gray",
                 "attributes": ["adjacent to road.", "partially visible at top right corner."]},
            ],
            "caption": "A person walks along an urban street lined with tree shadows stretching across the sunlit asphalt.",
            "keywords": ["street view", "shadow play", "urban landscape", "daytime outdoor setting"],
        }
        activities, source = ActivityRecoveryService.recover(frame)
        assert source == "caption"
        assert "walking" in activities

    def test_real_pattern_bus_traveling(self):
        """Mirrors actual frame f0021 (bus + person, no activity extracted by VLM)."""
        frame = {
            "activities": [],
            "objects": [
                {"type": "bus", "subtype": "public transportation vehicle", "color": "green/yellow",
                 "attributes": ["digital display", "route number indicator",
                                "advertisements on side panel"]},
                {"type": "person", "subtype": "", "color": "", "attributes": []},
            ],
            "caption": "A digital display board attached to a green-yellow public transit bus shows route information.",
            "keywords": ["bus stop", "public transport", "traveling", "city life"],
        }
        activities, source = ActivityRecoveryService.recover(frame)
        # No useful attribute, caption has no walking/driving verbs → keywords "traveling" → "moving"
        assert source == "keywords"
        assert "moving" in activities


# ------------------------------------------------------------------ #
# apply() tests                                                        #
# ------------------------------------------------------------------ #

class TestApply:

    def test_apply_mutates_activities_when_empty(self):
        parsed = {
            "activities": [],
            "objects": [{"type": "vehicle", "subtype": "car", "color": "gray",
                         "attributes": ["stationary"]}],
            "caption": "",
            "keywords": [],
        }
        result = ActivityRecoveryService.apply(parsed)
        assert result["activities"] == ["vehicle stationary"]
        assert result["activity_recovery_source"] == "attributes"

    def test_apply_sets_none_source_when_unrecoverable(self):
        parsed = {
            "activities": [],
            "objects": [{"type": "road", "subtype": "asphalt", "color": "gray",
                         "attributes": []}],
            "caption": "Long tree shadows on road.",
            "keywords": ["daylight", "shadows"],
        }
        result = ActivityRecoveryService.apply(parsed)
        assert result["activities"] == []
        assert result["activity_recovery_source"] == "none"

    def test_apply_does_not_set_recovery_source_for_original(self):
        parsed = {
            "activities": ["crossing road"],
            "objects": [],
            "caption": "",
            "keywords": [],
        }
        result = ActivityRecoveryService.apply(parsed)
        assert result["activities"] == ["crossing road"]
        assert "activity_recovery_source" not in result

    def test_apply_returns_same_dict_reference(self):
        """apply() mutates and returns the input dict."""
        parsed = {
            "activities": [],
            "objects": [{"type": "vehicle", "subtype": "car", "color": "gray",
                         "attributes": ["parked"]}],
            "caption": "",
            "keywords": [],
        }
        returned = ActivityRecoveryService.apply(parsed)
        assert returned is parsed

    def test_apply_caption_recovery_source_set(self):
        parsed = {
            "activities": [],
            "objects": [{"type": "roadway", "subtype": "asphalt", "color": "gray",
                         "attributes": []}],
            "caption": "A blue motorcycle driving down the street.",
            "keywords": [],
        }
        result = ActivityRecoveryService.apply(parsed)
        assert "driving" in result["activities"]
        assert result["activity_recovery_source"] == "caption"

    def test_apply_keyword_recovery_source_set(self):
        parsed = {
            "activities": [],
            "objects": [{"type": "roadway", "subtype": "asphalt", "color": "gray",
                         "attributes": []}],
            "caption": "Long tree shadows cast across a smooth asphalt road.",
            "keywords": ["pedestrian crossing", "urban scene"],
        }
        result = ActivityRecoveryService.apply(parsed)
        assert "crossing road" in result["activities"]
        assert result["activity_recovery_source"] == "keywords"
