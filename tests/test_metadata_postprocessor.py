"""Tests for MetadataPostprocessor.

Covers all six rules:
  1. Duplicate IDs
  2. Missing / generic IDs
  3. Relationship reference repair
  4. Location context reference repair
  5. Missing actor recovery
  6. Invalid reference removal
"""

import pytest
from copy import deepcopy

from app.schemas.frame import (
    FrameRichMetadata,
    ObjectMetadata,
    RelationshipMetadata,
    LocationContextMetadata,
)
from app.services.metadata_postprocessor import MetadataPostprocessor


def _make_meta(**overrides) -> FrameRichMetadata:
    """Helper to build a minimal FrameRichMetadata with sensible defaults."""
    defaults = dict(
        frame_id="test_frame",
        video_id="test_video",
        timestamp_seconds=0.0,
        timestamp_human="00:00:00",
        frame_path="frames/test.jpg",
        scene_type="test",
        objects=[],
        activities=[],
        relationships=[],
        location_context=[],
    )
    defaults.update(overrides)
    return FrameRichMetadata(**defaults)


# ──────────────────────────────────────────────────────────────────────── #
# Rule 1: Duplicate Object IDs                                           #
# ──────────────────────────────────────────────────────────────────────── #

class TestDuplicateIDs:
    def test_duplicate_ids_are_deduplicated(self):
        meta = _make_meta(objects=[
            ObjectMetadata(id="desk", type="furniture", subtype="desk"),
            ObjectMetadata(id="desk", type="furniture", subtype="desk"),
        ])
        result = MetadataPostprocessor.process(meta)
        ids = [o.id for o in result.objects]
        assert len(set(ids)) == len(ids), f"IDs are not unique: {ids}"
        assert "desk_1" in ids
        assert "desk_2" in ids

    def test_triple_duplicate(self):
        meta = _make_meta(objects=[
            ObjectMetadata(id="chair", type="furniture"),
            ObjectMetadata(id="chair", type="furniture"),
            ObjectMetadata(id="chair", type="furniture"),
        ])
        result = MetadataPostprocessor.process(meta)
        ids = [o.id for o in result.objects]
        assert ids == ["chair_1", "chair_2", "chair_3"]

    def test_no_duplicates_unchanged(self):
        meta = _make_meta(objects=[
            ObjectMetadata(id="desk_1", type="furniture"),
            ObjectMetadata(id="chair_1", type="furniture"),
        ])
        result = MetadataPostprocessor.process(meta)
        ids = [o.id for o in result.objects]
        assert "desk_1" in ids
        assert "chair_1" in ids


# ──────────────────────────────────────────────────────────────────────── #
# Rule 2: Normalize Generic IDs                                          #
# ──────────────────────────────────────────────────────────────────────── #

class TestGenericIDs:
    def test_bare_person_gets_indexed(self):
        meta = _make_meta(objects=[
            ObjectMetadata(id="person", type="person", subtype="customer"),
        ])
        result = MetadataPostprocessor.process(meta)
        assert result.objects[0].id == "person_1"

    def test_mixed_generic_and_indexed(self):
        meta = _make_meta(objects=[
            ObjectMetadata(id="desk", type="furniture"),
            ObjectMetadata(id="monitor", type="device"),
        ])
        result = MetadataPostprocessor.process(meta)
        ids = [o.id for o in result.objects]
        assert "desk_1" in ids
        assert "monitor_1" in ids


# ──────────────────────────────────────────────────────────────────────── #
# Rule 3: Relationship Reference Repair                                  #
# ──────────────────────────────────────────────────────────────────────── #

class TestRelationshipRepair:
    def test_generic_ref_repaired(self):
        meta = _make_meta(
            objects=[
                ObjectMetadata(id="person", type="person"),
                ObjectMetadata(id="desk", type="furniture"),
            ],
            relationships=[
                RelationshipMetadata(subject_id="person", target_id="desk", relation="standing_near"),
            ],
        )
        result = MetadataPostprocessor.process(meta)
        rel = result.relationships[0]
        assert rel.subject_id == "person_1"
        assert rel.target_id == "desk_1"

    def test_already_indexed_ref_preserved(self):
        meta = _make_meta(
            objects=[
                ObjectMetadata(id="person_1", type="person"),
                ObjectMetadata(id="desk_1", type="furniture"),
            ],
            relationships=[
                RelationshipMetadata(subject_id="person_1", target_id="desk_1", relation="standing_near"),
            ],
        )
        result = MetadataPostprocessor.process(meta)
        rel = result.relationships[0]
        assert rel.subject_id == "person_1"
        assert rel.target_id == "desk_1"

    def test_phantom_person_ref_creates_object_and_repairs(self):
        """VLM creates person_1 in relationships but not in objects."""
        meta = _make_meta(
            objects=[
                ObjectMetadata(id="desk", type="furniture"),
            ],
            activities=["standing"],
            relationships=[
                RelationshipMetadata(subject_id="person_1", target_id="desk", relation="standing_near"),
            ],
        )
        result = MetadataPostprocessor.process(meta)
        # person_1 should now exist in objects (recovered from activity + ref)
        obj_ids = [o.id for o in result.objects]
        assert "person_1" in obj_ids
        # Relationship should be intact
        rel = result.relationships[0]
        assert rel.subject_id == "person_1"


# ──────────────────────────────────────────────────────────────────────── #
# Rule 4: Location Reference Repair                                     #
# ──────────────────────────────────────────────────────────────────────── #

class TestLocationRepair:
    def test_generic_location_ref_repaired(self):
        meta = _make_meta(
            objects=[
                ObjectMetadata(id="counter", type="furniture"),
            ],
            location_context=[
                LocationContextMetadata(object_id="counter", location="center_area"),
            ],
        )
        result = MetadataPostprocessor.process(meta)
        loc = result.location_context[0]
        assert loc.object_id == "counter_1"

    def test_customer_ref_maps_to_person(self):
        """VLM uses 'customer' as object_id but object is 'person_1'."""
        meta = _make_meta(
            objects=[
                ObjectMetadata(id="person", type="person", subtype="customer"),
            ],
            activities=["talking"],
            location_context=[
                LocationContextMetadata(object_id="person", location="near_counter"),
            ],
        )
        result = MetadataPostprocessor.process(meta)
        loc = result.location_context[0]
        assert loc.object_id == "person_1"


# ──────────────────────────────────────────────────────────────────────── #
# Rule 5: Missing Actor Recovery                                         #
# ──────────────────────────────────────────────────────────────────────── #

class TestMissingActorRecovery:
    def test_walking_activity_creates_person(self):
        meta = _make_meta(
            objects=[
                ObjectMetadata(id="desk", type="furniture"),
            ],
            activities=["walking"],
        )
        result = MetadataPostprocessor.process(meta)
        obj_types = [(o.type, o.subtype) for o in result.objects]
        assert ("person", "person") in obj_types

    def test_no_recovery_when_person_exists(self):
        meta = _make_meta(
            objects=[
                ObjectMetadata(id="customer_1", type="person", subtype="customer"),
            ],
            activities=["standing"],
        )
        original_count = len(meta.objects)
        result = MetadataPostprocessor.process(meta)
        # Should not add another person
        person_count = sum(1 for o in result.objects if o.type.lower() == "person")
        assert person_count == 1

    def test_no_recovery_for_none_activity(self):
        meta = _make_meta(
            objects=[
                ObjectMetadata(id="desk", type="furniture"),
            ],
            activities=["none"],
        )
        result = MetadataPostprocessor.process(meta)
        person_count = sum(1 for o in result.objects if o.type.lower() == "person")
        assert person_count == 0

    def test_empty_activities_no_recovery_if_count_zero(self):
        meta = _make_meta(
            objects=[ObjectMetadata(id="desk", type="furniture")],
            activities=[],
            people_count=0,
        )
        result = MetadataPostprocessor.process(meta)
        person_count = sum(1 for o in result.objects if o.type.lower() == "person")
        assert person_count == 0

    def test_people_count_creates_person(self):
        meta = _make_meta(
            objects=[ObjectMetadata(id="desk", type="furniture")],
            activities=[],
            people_count=1,
        )
        result = MetadataPostprocessor.process(meta)
        person_count = sum(1 for o in result.objects if o.type.lower() == "person")
        assert person_count == 1
        obj_ids = [o.id for o in result.objects]
        assert "person_1" in obj_ids
        
    def test_location_context_reassigned_on_recovery(self):
        meta = _make_meta(
            objects=[ObjectMetadata(id="desk", type="furniture")],
            activities=["standing"],
            people_count=1,
            location_context=[LocationContextMetadata(object_id="", location="center_area")],
        )
        result = MetadataPostprocessor.process(meta)
        loc = result.location_context[0]
        assert loc.object_id == "person_1"



# ──────────────────────────────────────────────────────────────────────── #
# Rule 6: Invalid Reference Removal                                      #
# ──────────────────────────────────────────────────────────────────────── #

class TestInvalidReferenceRemoval:
    def test_completely_dangling_relationship_removed(self):
        meta = _make_meta(
            objects=[
                ObjectMetadata(id="desk", type="furniture"),
            ],
            relationships=[
                RelationshipMetadata(subject_id="ghost_1", target_id="phantom_2", relation="haunting"),
            ],
        )
        result = MetadataPostprocessor.process(meta)
        assert len(result.relationships) == 0

    def test_dangling_location_removed(self):
        meta = _make_meta(
            objects=[
                ObjectMetadata(id="desk", type="furniture"),
            ],
            location_context=[
                LocationContextMetadata(object_id="nonexistent", location="center_area"),
            ],
        )
        result = MetadataPostprocessor.process(meta)
        assert len(result.location_context) == 0

    def test_empty_relation_removed(self):
        meta = _make_meta(
            objects=[
                ObjectMetadata(id="desk", type="furniture"),
                ObjectMetadata(id="chair", type="furniture"),
            ],
            relationships=[
                RelationshipMetadata(subject_id="desk", target_id="chair", relation=""),
            ],
        )
        result = MetadataPostprocessor.process(meta)
        assert len(result.relationships) == 0

    def test_valid_relationship_kept(self):
        meta = _make_meta(
            objects=[
                ObjectMetadata(id="person_1", type="person"),
                ObjectMetadata(id="desk_1", type="furniture"),
            ],
            relationships=[
                RelationshipMetadata(subject_id="person_1", target_id="desk_1", relation="standing_near"),
            ],
        )
        result = MetadataPostprocessor.process(meta)
        assert len(result.relationships) == 1


# ──────────────────────────────────────────────────────────────────────── #
# Integration: Full V3 output scenario                                   #
# ──────────────────────────────────────────────────────────────────────── #

class TestIntegration:
    def test_v3_jewelry_store_scenario(self):
        """Simulate the actual V3 experiment output for the jewelry store frame."""
        meta = _make_meta(
            scene_type="jewelry_store",
            objects=[
                ObjectMetadata(id="counter", type="desk", subtype="cashier", condition="occupied"),
                ObjectMetadata(id="glass_cases", type="display_cases", subtype="jewelry"),
                ObjectMetadata(id="computer_monitor", type="electronic_device", subtype="monitor"),
                ObjectMetadata(id="laptop", type="electronic_device", subtype="laptop"),
                ObjectMetadata(id="door", type="entrance", subtype="main"),
            ],
            activities=[
                "standing",
                "standing_near",
            ],
            relationships=[
                RelationshipMetadata(subject_id="person_1", target_id="counter", relation="talking_to"),
                RelationshipMetadata(subject_id="person_2", target_id="glass_cases", relation="facing"),
            ],
            location_context=[
                LocationContextMetadata(object_id="counter", location="near_counter"),
                LocationContextMetadata(object_id="glass_cases", location="center_area"),
                LocationContextMetadata(object_id="door", location="right_side"),
            ],
        )
        result = MetadataPostprocessor.process(meta)

        # All object IDs should be unique and indexed
        obj_ids = [o.id for o in result.objects]
        assert len(set(obj_ids)) == len(obj_ids)

        # Activities should be plain strings
        for act in result.activities:
            assert isinstance(act, str), f"Activity is not a string: {act}"

        # person_1, person_2 should exist as objects (recovered from refs)
        assert "person_1" in obj_ids
        assert "person_2" in obj_ids

        # Relationships should reference valid objects
        for rel in result.relationships:
            assert rel.subject_id in obj_ids, f"Dangling subject: {rel.subject_id}"
            assert rel.target_id in obj_ids, f"Dangling target: {rel.target_id}"

        # Location context should reference valid objects
        for loc in result.location_context:
            assert loc.object_id in obj_ids, f"Dangling loc ref: {loc.object_id}"

    def test_empty_room_scenario(self):
        """Empty room: no activities, no relationships, no location context."""
        meta = _make_meta(
            scene_type="office",
            objects=[
                ObjectMetadata(id="desk", type="furniture"),
                ObjectMetadata(id="chair", type="furniture"),
                ObjectMetadata(id="monitor", type="device"),
            ],
            activities=[],
            relationships=[],
            location_context=[],
        )
        result = MetadataPostprocessor.process(meta)
        assert len(result.objects) == 3
        assert len(result.activities) == 0
        assert len(result.relationships) == 0
        assert len(result.location_context) == 0
