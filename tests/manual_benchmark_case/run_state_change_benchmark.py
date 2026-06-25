import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT_PATH = SCRIPT_PATH.parents[2]
OUTPUT_ROOT = PROJECT_ROOT_PATH / "tests" / "manual_benchmark_case" / "data" / "output"

RELATIONSHIP_GRAPH_PATH = OUTPUT_ROOT / "relationship_graph.json"
STATE_CHANGES_PATH = OUTPUT_ROOT / "state_changes.json"


STATE_BY_RELATIONSHIP = {
    "walking_together": ("walking", "People moved together with close spacing and similar direction."),
    "following": ("following", "One person moved behind another along a similar path."),
    "meeting": ("standing", "People were close enough to indicate a meeting or pause."),
    "distance": ("proximal_distance_observed", "Distance between two tracked entities was measured."),
    "near_object": ("person_near_object", "A person was near an object."),
    "touch_object": ("person_touch_object", "A person was close enough to touch or overlap the object."),
    "leave_object": ("person_leaves_object", "A person moved away after being near the object."),
    "pick_object": ("object_picked_or_moved", "The object moved after person-object contact."),
    "enter_vehicle": ("person_enters_vehicle", "A person disappeared or ended near a vehicle while the vehicle remained tracked."),
    "exit_vehicle": ("person_exits_vehicle", "A person appeared near a vehicle that was already tracked."),
    "load_object": ("object_loaded_into_vehicle", "A person, object, and vehicle were close in the same frame."),
    "entered": ("zone_entered", "A tracked entity entered a zone."),
    "left": ("zone_left", "A tracked entity left a zone."),
    "parked": ("vehicle_parked", "A vehicle stayed stationary inside a zone."),
    "inside": ("object_inside_zone", "An object was observed inside a zone."),
    "outside": ("object_outside_zone", "An object zone was unknown or outside defined frame zones."),
    "stationary": ("object_stationary", "An object stayed stationary inside a zone."),
}

END_STATE_BY_RELATIONSHIP = {
    "walking_together": ("walking_together_ended", "The walking-together relationship ended."),
    "following": ("following_ended", "The following relationship ended."),
    "meeting": ("meeting_ended", "The meeting/proximity interval ended."),
    "near_object": ("person_no_longer_near_object", "The person-object proximity interval ended."),
    "touch_object": ("person_no_longer_touching_object", "The person-object touch interval ended."),
    "parked": ("vehicle_no_longer_confirmed_parked", "The parked interval ended."),
    "stationary": ("object_stationary_interval_ended", "The stationary interval ended."),
}

TRANSITION_REASONS = {
    ("walking", "standing"): "Movement changed from walking-like motion to standing/meeting.",
    ("standing", "walking"): "Standing/meeting ended and walking-like motion resumed.",
    ("person_near_object", "person_touch_object"): "Person-object proximity escalated to contact.",
    ("person_touch_object", "object_stationary"): "Object remained stationary after contact.",
    ("object_stationary", "person_leaves_object"): "Person left while the object remained stationary.",
    ("person_leaves_object", "person_near_object"): "A person approached after another person left.",
    ("person_touch_object", "object_picked_or_moved"): "Object moved after touch/contact.",
    ("person_near_object", "object_loaded_into_vehicle"): "Object proximity progressed into vehicle loading.",
    ("zone_entered", "vehicle_parked"): "Vehicle entered a zone and became parked.",
    ("object_inside_zone", "object_stationary"): "Object inside a zone became or remained stationary.",
}


def _format_timestamp(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def _load_relationship_graph() -> Dict[str, Any]:
    if not RELATIONSHIP_GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Missing Phase 2 input: {RELATIONSHIP_GRAPH_PATH}. "
            "Run run_relationship_graph_benchmark.py first."
        )
    payload = json.loads(RELATIONSHIP_GRAPH_PATH.read_text(encoding="utf-8"))
    relationships = payload.get("relationships")
    if not isinstance(relationships, list):
        raise ValueError("relationship_graph.json does not contain a relationships list.")
    return payload


def _track_key(track: Dict[str, Any]) -> Optional[str]:
    if track.get("entity_type") == "zone":
        return f"zone:{track.get('zone', 'unknown')}"
    track_id = track.get("track_id")
    if track_id is None:
        return None
    return f"track:{track_id}"


def _supporting_tracks(relationship: Dict[str, Any]) -> List[Dict[str, Any]]:
    tracks = []
    for role in ("source", "target"):
        entity = relationship.get(role, {})
        if entity.get("entity_type") == "zone":
            tracks.append({"role": role, "entity_type": "zone", "zone": entity.get("zone", "unknown")})
        else:
            tracks.append(
                {
                    "role": role,
                    "track_id": entity.get("track_id"),
                    "entity_type": entity.get("entity_type"),
                    "class_name": entity.get("class_name"),
                }
            )
    return tracks


def _frames_from_relationship(relationship: Dict[str, Any]) -> List[str]:
    frame_ids = relationship.get("evidence", {}).get("frame_ids", [])
    if frame_ids:
        return list(dict.fromkeys(frame_ids))
    frames = [
        relationship.get("start", {}).get("frame_id"),
        relationship.get("end", {}).get("frame_id"),
    ]
    return [frame_id for frame_id in dict.fromkeys(frames) if frame_id]


def _direct_state_change(
    relationship: Dict[str, Any],
    state_name: str,
    reason: str,
    boundary: str,
    confidence_multiplier: float,
) -> Dict[str, Any]:
    boundary_data = relationship.get(boundary, {})
    timestamp_seconds = float(boundary_data.get("timestamp_seconds", 0.0))
    return {
        "state": state_name,
        "timestamp": {
            "seconds": timestamp_seconds,
            "human": boundary_data.get("timestamp_human") or _format_timestamp(timestamp_seconds),
        },
        "frames": _frames_from_relationship(relationship),
        "reason": reason,
        "confidence": round(float(relationship.get("confidence", 0.0)) * confidence_multiplier, 3),
        "supporting_tracks": _supporting_tracks(relationship),
        "source_relationship_ids": [relationship.get("relationship_id")],
        "evidence": {
            "relationship_type": relationship.get("relationship_type"),
            "relationship_start": relationship.get("start"),
            "relationship_end": relationship.get("end"),
            "relationship_evidence": relationship.get("evidence", {}),
            "boundary": boundary,
            "no_vlm": True,
        },
    }


def _direct_state_changes(relationship: Dict[str, Any]) -> List[Dict[str, Any]]:
    relation_type = relationship.get("relationship_type")
    changes = []
    if relation_type in STATE_BY_RELATIONSHIP:
        state_name, reason = STATE_BY_RELATIONSHIP[relation_type]
        changes.append(_direct_state_change(relationship, state_name, reason, "start", 1.0))
    if relation_type in END_STATE_BY_RELATIONSHIP:
        state_name, reason = END_STATE_BY_RELATIONSHIP[relation_type]
        changes.append(_direct_state_change(relationship, state_name, reason, "end", 0.88))
    return changes


def _relationship_state_name(relationship: Dict[str, Any]) -> Optional[str]:
    entry = STATE_BY_RELATIONSHIP.get(relationship.get("relationship_type"))
    if entry is None:
        return None
    return entry[0]


def _relationship_time(relationship: Dict[str, Any], boundary: str = "start") -> float:
    return float(relationship.get(boundary, {}).get("timestamp_seconds", 0.0))


def _relationship_frames(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    frames = _frames_from_relationship(a) + _frames_from_relationship(b)
    return list(dict.fromkeys(frames))


def _shared_entity_keys(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    a_keys = {_track_key(a.get("source", {})), _track_key(a.get("target", {}))}
    b_keys = {_track_key(b.get("source", {})), _track_key(b.get("target", {}))}
    return sorted(key for key in a_keys.intersection(b_keys) if key)


def _merge_supporting_tracks(a: Dict[str, Any], b: Dict[str, Any]) -> List[Dict[str, Any]]:
    merged = []
    seen = set()
    for item in _supporting_tracks(a) + _supporting_tracks(b):
        key = json.dumps(item, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _transition_state_change(previous: Dict[str, Any], current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    previous_state = _relationship_state_name(previous)
    current_state = _relationship_state_name(current)
    if previous_state is None or current_state is None or previous_state == current_state:
        return None

    shared_keys = _shared_entity_keys(previous, current)
    if not shared_keys:
        return None

    reason = TRANSITION_REASONS.get(
        (previous_state, current_state),
        f"State changed from {previous_state} to {current_state} for shared tracked entity context.",
    )
    timestamp_seconds = _relationship_time(current, "start")
    confidence = min(
        float(previous.get("confidence", 0.0)),
        float(current.get("confidence", 0.0)),
    )
    return {
        "state": f"{previous_state}_to_{current_state}",
        "timestamp": {
            "seconds": timestamp_seconds,
            "human": current.get("start", {}).get("timestamp_human") or _format_timestamp(timestamp_seconds),
        },
        "frames": _relationship_frames(previous, current),
        "reason": reason,
        "confidence": round(confidence * 0.92, 3),
        "supporting_tracks": _merge_supporting_tracks(previous, current),
        "source_relationship_ids": [
            previous.get("relationship_id"),
            current.get("relationship_id"),
        ],
        "evidence": {
            "previous_relationship_type": previous.get("relationship_type"),
            "current_relationship_type": current.get("relationship_type"),
            "previous_state": previous_state,
            "current_state": current_state,
            "shared_entity_keys": shared_keys,
            "previous_relationship": {
                "start": previous.get("start"),
                "end": previous.get("end"),
                "confidence": previous.get("confidence"),
            },
            "current_relationship": {
                "start": current.get("start"),
                "end": current.get("end"),
                "confidence": current.get("confidence"),
            },
            "no_vlm": True,
        },
    }


def _ordered_relationships_for_key(relationships: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    rows = []
    for relationship in relationships:
        keys = {_track_key(relationship.get("source", {})), _track_key(relationship.get("target", {}))}
        if key in keys:
            rows.append(relationship)
    return sorted(
        rows,
        key=lambda item: (
            _relationship_time(item, "start"),
            _relationship_time(item, "end"),
            str(item.get("relationship_id", "")),
        ),
    )


def _transition_state_changes(relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    entity_keys = sorted(
        {
            key
            for relationship in relationships
            for key in (_track_key(relationship.get("source", {})), _track_key(relationship.get("target", {})))
            if key
        }
    )
    changes = []
    seen = set()
    for key in entity_keys:
        ordered = _ordered_relationships_for_key(relationships, key)
        for previous, current in zip(ordered, ordered[1:]):
            change = _transition_state_change(previous, current)
            if change is None:
                continue
            dedupe_key = (
                tuple(change["source_relationship_ids"]),
                change["state"],
                change["timestamp"]["seconds"],
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            changes.append(change)
    return changes


def _missing_or_disappear_notes(relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    notes = []
    object_relationships = [
        relationship
        for relationship in relationships
        if relationship.get("source", {}).get("entity_type") == "object"
        or relationship.get("target", {}).get("entity_type") == "object"
    ]
    if object_relationships:
        notes.append(
            {
                "state": "object_disappearance_not_inferred",
                "reason": (
                    "relationship_graph.json contains relationship intervals but not a full absent/present timeline, "
                    "so disappearance is not asserted in this no-VLM phase."
                ),
                "supporting_relationship_count": len(object_relationships),
            }
        )
    return notes


def _build_state_changes(graph: Dict[str, Any]) -> Dict[str, Any]:
    relationships = sorted(
        graph.get("relationships", []),
        key=lambda item: (
            _relationship_time(item, "start"),
            _relationship_time(item, "end"),
            str(item.get("relationship_id", "")),
        ),
    )
    state_changes = []
    for relationship in relationships:
        state_changes.extend(_direct_state_changes(relationship))
    state_changes.extend(_transition_state_changes(relationships))

    state_changes = sorted(
        state_changes,
        key=lambda item: (
            float(item["timestamp"]["seconds"]),
            item["state"],
            ",".join(str(value) for value in item["source_relationship_ids"]),
        ),
    )
    for index, change in enumerate(state_changes, start=1):
        change["state_change_id"] = f"sc_{index:05d}"

    by_state: Dict[str, int] = {}
    for change in state_changes:
        by_state[change["state"]] = by_state.get(change["state"], 0) + 1

    return {
        "benchmark": "state_change_builder_phase_3",
        "input": str(RELATIONSHIP_GRAPH_PATH),
        "output": str(STATE_CHANGES_PATH),
        "summary": {
            "relationship_count": len(relationships),
            "state_change_count": len(state_changes),
            "state_change_count_by_state": dict(sorted(by_state.items())),
            "no_vlm": True,
            "method": "deterministic_relationship_state_transitions",
        },
        "state_changes": state_changes,
        "notes": _missing_or_disappear_notes(relationships),
    }


def main() -> None:
    start = time.perf_counter()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    graph = _load_relationship_graph()
    payload = _build_state_changes(graph)
    payload["summary"]["wall_clock_runtime_seconds"] = time.perf_counter() - start
    STATE_CHANGES_PATH.write_text(json.dumps(payload, indent=4), encoding="utf-8")

    print("STATE_CHANGE_BENCHMARK_START")
    print(json.dumps({"state_changes": str(STATE_CHANGES_PATH)}))
    print("STATE_CHANGE_BENCHMARK_END")


if __name__ == "__main__":
    main()
