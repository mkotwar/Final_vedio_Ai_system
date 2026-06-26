from collections import defaultdict
from typing import Any, Dict, List, Tuple


PERSON_TERMS = {"person", "pedestrian", "employee", "customer", "guard", "security", "rider", "driver", "passenger"}
VEHICLE_TERMS = {"vehicle", "car", "truck", "bus", "motorcycle", "bicycle", "bike", "scooter"}


def _entity_type(obj: Dict[str, Any]) -> str:
    obj_type = str(obj.get("type", "")).lower()
    subtype = str(obj.get("subtype", "")).lower()
    text = f"{obj_type} {subtype}"
    if any(term in text for term in PERSON_TERMS):
        return "person"
    if any(term in text for term in VEHICLE_TERMS):
        return "vehicle"
    return "object"


def _actor_key(obj: Dict[str, Any], entity_type: str) -> str:
    obj_id = str(obj.get("id", "")).strip()
    if obj_id:
        return f"{entity_type}:{obj_id.lower()}"
    color = str(obj.get("color", "")).strip().lower()
    subtype = str(obj.get("subtype", "")).strip().lower()
    obj_type = str(obj.get("type", "")).strip().lower()
    return f"{entity_type}:{obj_type}:{subtype}:{color}"


def _event_lookup(events: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    frame_to_events: Dict[str, List[str]] = defaultdict(list)
    for event in events:
        event_id = str(event.get("event_id", ""))
        for frame_id in event.get("source_frames", []) or []:
            frame_to_events[frame_id].append(event_id)
    return frame_to_events


def _activity_flags(frame: Dict[str, Any], obj: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    activities = [str(item).lower() for item in frame.get("activities", [])]
    attributes = [str(item).lower() for item in obj.get("attributes", [])]
    joined = " ".join(activities + attributes)

    if "carrying" in joined or "holding" in joined:
        flags.append("carrying_object")
    if "running" in joined:
        flags.append("running")
    if "standing" in joined:
        flags.append("standing")
    if "walking" in joined:
        flags.append("walking")
    if "enter" in joined or "arriv" in joined:
        flags.append("entry_like")
    if "exit" in joined or "depart" in joined or "leave" in joined:
        flags.append("exit_like")
    return flags


def build_actor_states(frames: List[Dict[str, Any]], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    frame_to_events = _event_lookup(events)
    actors: Dict[str, Dict[str, Any]] = {}

    for frame in sorted(frames, key=lambda item: float(item.get("timestamp_seconds", 0.0))):
        frame_id = str(frame.get("frame_id", ""))
        timestamp_seconds = float(frame.get("timestamp_seconds", 0.0))
        timestamp_human = frame.get("timestamp_human", "")
        activities = frame.get("activities", []) or []
        location = frame.get("scene_description", "") or frame.get("caption", "")
        related_events = frame_to_events.get(frame_id, [])

        for obj in frame.get("objects", []) or []:
            if not isinstance(obj, dict):
                continue

            entity_type = _entity_type(obj)
            actor_id = _actor_key(obj, entity_type)
            actor = actors.setdefault(
                actor_id,
                {
                    "actor_id": actor_id,
                    "entity_type": entity_type,
                    "source_object_id": obj.get("id", ""),
                    "type": obj.get("type", ""),
                    "subtype": obj.get("subtype", ""),
                    "dominant_color": obj.get("color", ""),
                    "attributes": sorted({str(item) for item in obj.get("attributes", [])}),
                    "first_seen_seconds": timestamp_seconds,
                    "last_seen_seconds": timestamp_seconds,
                    "first_seen_human": timestamp_human,
                    "last_seen_human": timestamp_human,
                    "observation_count": 0,
                    "activities": [],
                    "activity_counts": defaultdict(int),
                    "flags": [],
                    "event_ids": [],
                    "observations": [],
                },
            )

            actor["last_seen_seconds"] = timestamp_seconds
            actor["last_seen_human"] = timestamp_human
            actor["observation_count"] += 1
            actor["dominant_color"] = actor["dominant_color"] or obj.get("color", "")

            for item in activities:
                value = str(item)
                actor["activity_counts"][value] += 1

            flags = _activity_flags(frame, obj)
            actor["flags"] = sorted(set(actor["flags"]).union(flags))
            actor["event_ids"] = sorted(set(actor["event_ids"]).union(related_events))
            actor["observations"].append(
                {
                    "frame_id": frame_id,
                    "timestamp_seconds": timestamp_seconds,
                    "timestamp_human": timestamp_human,
                    "activities": activities,
                    "event_ids": related_events,
                    "location_hint": location,
                    "flags": flags,
                }
            )

    serialized_actors: List[Dict[str, Any]] = []
    for actor in actors.values():
        ranked_activities = sorted(
            actor["activity_counts"].items(),
            key=lambda item: (-item[1], item[0]),
        )
        actor["activities"] = [name for name, _count in ranked_activities[:6]]
        actor["activity_counts"] = dict(ranked_activities)
        actor["presence_duration_seconds"] = round(actor["last_seen_seconds"] - actor["first_seen_seconds"], 2)
        serialized_actors.append(actor)

    serialized_actors.sort(key=lambda item: (item["entity_type"], item["actor_id"]))
    counts: Dict[str, int] = defaultdict(int)
    for actor in serialized_actors:
        counts[actor["entity_type"]] += 1

    return {
        "summary": {
            "actor_count": len(serialized_actors),
            "actor_type_counts": dict(counts),
        },
        "actors": serialized_actors,
    }
