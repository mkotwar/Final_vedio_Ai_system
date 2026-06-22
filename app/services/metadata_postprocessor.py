"""Metadata Postprocessor – stabilizes VLM output before Event Aggregation.

Sits between VLM generation and Event Aggregation.  Repairs schema drift,
deduplicates object IDs, fixes dangling references, and recovers missing
actors without touching the VLM prompt or the Event Aggregator.

Usage::

    from app.services.metadata_postprocessor import MetadataPostprocessor
    rich_meta = MetadataPostprocessor.process(rich_meta)
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set

from app.schemas.frame import (
    FrameRichMetadata,
    ObjectMetadata,
    RelationshipMetadata,
    LocationContextMetadata,
)


# Activities that imply a human actor must exist in the objects list.
_PERSON_ACTIVITIES: frozenset = frozenset({
    "standing", "walking", "running", "sitting", "talking",
    "interacting", "waiting", "working", "driving",
    "entering", "exiting",
})

_APPROVED_ACTIVITIES: frozenset = frozenset({
    "standing", "walking", "running", "sitting", "talking",
    "interacting", "waiting", "working", "driving",
    "entering", "exiting", "none",
})

# Regex to strip a trailing _N suffix so we can find the base name.
_TRAILING_INDEX_RE = re.compile(r"^(.+?)_(\d+)$")


class MetadataPostprocessor:
    """Post-processes a single ``FrameRichMetadata`` instance in-place."""

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def process(metadata: FrameRichMetadata) -> FrameRichMetadata:
        """Apply all repair rules and return the (mutated) metadata."""

        # Rule 5 first – recover missing actors (may add objects)
        _recover_missing_actors(metadata)

        # Rule 1 + 2 – deduplicate & normalize object IDs
        id_mapping = _deduplicate_object_ids(metadata)

        # Rule 3 – repair relationship references
        _repair_relationship_refs(metadata, id_mapping)

        # Rule 4 – repair location context references
        _repair_location_refs(metadata, id_mapping)

        # Rule 6 – drop references that still can't be resolved
        _remove_invalid_references(metadata)

        return metadata


# ──────────────────────────────────────────────────────────────────────── #
# Internal helpers                                                        #
# ──────────────────────────────────────────────────────────────────────── #


def _base_name(raw_id: str) -> str:
    """Return the base name of an ID, stripping any trailing ``_N`` index.

    ``"desk_2"`` → ``"desk"``, ``"person"`` → ``"person"``
    """
    m = _TRAILING_INDEX_RE.match(raw_id)
    return m.group(1) if m else raw_id


def _deduplicate_object_ids(metadata: FrameRichMetadata) -> Dict[str, str]:
    """Rule 1 + 2: Ensure every object has a unique ID.

    Returns a mapping ``{old_id: new_id}`` for IDs that changed so that
    downstream reference-repair can use it.
    """
    objects = metadata.objects
    if not objects:
        return {}

    id_mapping: Dict[str, str] = {}  # old_id → new_id
    base_counters: Counter = Counter()
    new_objects: List[ObjectMetadata] = []

    # First pass: count how many times each base name appears.
    for obj in objects:
        base = _base_name(obj.id) if obj.id else (obj.subtype or obj.type or "object")
        base_counters[base] += 1

    # Second pass: assign unique IDs.
    assign_counters: Counter = Counter()
    for obj in objects:
        old_id = obj.id
        base = _base_name(old_id) if old_id else (obj.subtype or obj.type or "object")
        base = base.lower().strip().replace(" ", "_")

        # Always number if the base appears more than once OR if the id
        # is a bare generic name without an index.
        needs_index = (
            base_counters[base] > 1
            or (old_id and not _TRAILING_INDEX_RE.match(old_id) and base_counters.get(base, 0) >= 1)
        )

        assign_counters[base] += 1
        new_id = f"{base}_{assign_counters[base]}" if needs_index else old_id

        if old_id and old_id != new_id:
            id_mapping[old_id] = new_id

        obj.id = new_id
        new_objects.append(obj)

    metadata.objects = new_objects
    return id_mapping


def _resolve_ref(ref_id: str, id_mapping: Dict[str, str], object_ids: Set[str]) -> Optional[str]:
    """Try to resolve a reference ID to a valid object ID.

    Strategy:
    1. Direct match in current object IDs.
    2. Lookup in id_mapping (old_id → new_id).
    3. Fuzzy: if ref_id is a bare name (e.g. ``"person"``), find the first
       object whose base name matches (e.g. ``"person_1"``).
    """
    if ref_id in object_ids:
        return ref_id

    if ref_id in id_mapping:
        mapped = id_mapping[ref_id]
        if mapped in object_ids:
            return mapped

    # Fuzzy: bare name → first matching indexed id
    ref_base = _base_name(ref_id).lower().strip()
    for oid in sorted(object_ids):
        if _base_name(oid).lower().strip() == ref_base:
            return oid

    return None


def _repair_relationship_refs(
    metadata: FrameRichMetadata, id_mapping: Dict[str, str]
) -> None:
    """Rule 3: Repair dangling subject_id / target_id in relationships."""
    object_ids = {obj.id for obj in metadata.objects if obj.id}

    for rel in metadata.relationships:
        if rel.subject_id:
            resolved = _resolve_ref(rel.subject_id, id_mapping, object_ids)
            if resolved:
                rel.subject_id = resolved

        if rel.target_id:
            resolved = _resolve_ref(rel.target_id, id_mapping, object_ids)
            if resolved:
                rel.target_id = resolved


def _repair_location_refs(
    metadata: FrameRichMetadata, id_mapping: Dict[str, str]
) -> None:
    """Rule 4: Repair dangling object_id in location_context."""
    object_ids = {obj.id for obj in metadata.objects if obj.id}

    for loc in metadata.location_context:
        if loc.object_id:
            resolved = _resolve_ref(loc.object_id, id_mapping, object_ids)
            if resolved:
                loc.object_id = resolved


def _remove_invalid_references(metadata: FrameRichMetadata) -> None:
    """Rule 6: Drop relationships and location entries that still
    reference non-existent object IDs after repair."""
    object_ids = {obj.id for obj in metadata.objects if obj.id}

    metadata.relationships = [
        rel for rel in metadata.relationships
        if (
            rel.subject_id in object_ids
            and rel.target_id in object_ids
            and rel.relation
        )
    ]

    metadata.location_context = [
        loc for loc in metadata.location_context
        if loc.object_id in object_ids and loc.location
    ]


def _recover_missing_actors(metadata: FrameRichMetadata) -> None:
    """Rule 5: Recover missing actors based on people_count or activity triggers."""
    # Check if any person-type object already exists
    has_person = any(
        obj.type.lower() in ("person",) or obj.subtype.lower() in ("person", "customer", "employee")
        for obj in metadata.objects
    )

    if has_person:
        return

    person_activities = [a for a in metadata.activities if isinstance(a, str) and a.lower().strip() in _PERSON_ACTIVITIES]
    # Also check dict activities
    for a in metadata.activities:
        if isinstance(a, dict):
            label = a.get("type", "") or a.get("relation", "")
            if str(label).lower().strip() in _PERSON_ACTIVITIES:
                person_activities.append(str(label))

    # Rule A & B: Trigger if people_count > 0 OR we have person activities
    needs_recovery = (metadata.people_count > 0) or len(person_activities) > 0

    if not needs_recovery:
        return

    # Also check relationship/location refs for person-like IDs
    # to avoid double-creating
    all_refs: Set[str] = set()
    for rel in metadata.relationships:
        if rel.subject_id:
            all_refs.add(rel.subject_id)
        if rel.target_id:
            all_refs.add(rel.target_id)
    for loc in metadata.location_context:
        if loc.object_id:
            all_refs.add(loc.object_id)

    person_refs = [r for r in all_refs if "person" in r.lower() or "customer" in r.lower() or "employee" in r.lower()]

    synthesized_id = None
    if person_refs:
        # Create person objects for each unique person reference
        existing_ids = {obj.id for obj in metadata.objects}
        for ref in sorted(set(person_refs)):
            if ref not in existing_ids:
                metadata.objects.append(ObjectMetadata(
                    id=ref,
                    type="person",
                    subtype="person",
                    condition="unknown",
                ))
                if not synthesized_id:
                    synthesized_id = ref
    else:
        # No person refs at all — create a single generic person
        synthesized_id = "person_1"
        metadata.objects.append(ObjectMetadata(
            id=synthesized_id,
            type="person",
            subtype="person",
            condition="unknown",
        ))
        
    # Rule C: If exactly one location_context exists, assign it to the new person
    if synthesized_id and len(metadata.location_context) == 1:
        if not metadata.location_context[0].object_id:
            metadata.location_context[0].object_id = synthesized_id
