from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EvidenceUnit:
    evidence_id: str
    time_start: float
    time_end: float
    actors: List[str]
    evidence_type: str
    objects: List[str]
    location: str
    event_id: str
    description: str


PERSON_TERMS = {"person", "pedestrian", "employee", "customer", "guard", "security", "man", "woman", "rider", "driver", "passenger"}
VEHICLE_TERMS = {"vehicle", "car", "truck", "bus", "motorcycle", "bicycle", "bike", "scooter", "van"}


def _event_seconds(event: Dict[str, Any]) -> Tuple[float, float]:
    return float(event.get("duration_seconds", 0.0) or 0.0), 0.0


def _timeline_index(actor_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    by_global_id: Dict[str, Dict[str, Any]] = {}
    by_source_id: Dict[str, str] = {}
    for actor in actor_payload.get("actors", []):
        global_id = actor.get("global_actor_id", "")
        by_global_id[global_id] = actor
        attributes = actor.get("attributes", {})
        for source_object_id in attributes.get("source_object_ids", []) or []:
            source_id = str(source_object_id).strip().lower()
            if source_id:
                by_source_id[source_id] = global_id
    return {"by_global_id": by_global_id, "by_source_id": by_source_id}


def _entity_type(obj: Dict[str, Any]) -> str:
    obj_type = str(obj.get("type", "")).lower()
    subtype = str(obj.get("subtype", "")).lower()
    text = f"{obj_type} {subtype}"
    if any(term in text for term in PERSON_TERMS):
        return "actor"
    if any(term in text for term in VEHICLE_TERMS):
        return "vehicle"
    return "object"


def _fallback_object_key(obj: Dict[str, Any]) -> str:
    entity_type = _entity_type(obj)
    subtype = str(obj.get("subtype", "")).strip().lower()
    color = str(obj.get("color", "")).strip().lower()
    obj_type = str(obj.get("type", "")).strip().lower()
    return f"{entity_type}:{obj_type}:{subtype}:{color}"


def _lookup_global_id(obj: Dict[str, Any], actor_index: Dict[str, Dict[str, Any]]) -> Optional[str]:
    object_id = str(obj.get("id", "")).strip().lower()
    if object_id:
        return actor_index["by_source_id"].get(object_id)

    entity_type = _entity_type(obj)
    subtype = str(obj.get("subtype", "")).strip().lower()
    color = str(obj.get("color", "")).strip().lower()
    attributes = {str(item).strip().lower() for item in obj.get("attributes", []) or []}

    for global_id, actor in actor_index["by_global_id"].items():
        if actor.get("entity_type") != entity_type:
            continue
        attr_block = actor.get("attributes", {})
        if str(attr_block.get("subtype", "")).strip().lower() != subtype:
            continue
        existing_color = str(attr_block.get("upper_color", "")).strip().lower()
        if color and existing_color and color != existing_color:
            continue
        existing_attributes = {str(item).strip().lower() for item in attr_block.get("attributes", []) or []}
        if attributes and existing_attributes and not attributes.intersection(existing_attributes):
            continue
        return global_id
    return None


def _event_window_seconds(event: Dict[str, Any]) -> Tuple[float, float]:
    start_text = str(event.get("start_time", "")).strip()
    end_text = str(event.get("end_time", "")).strip()
    start = _time_to_seconds(start_text)
    end = _time_to_seconds(end_text)
    if end < start:
        end = start + float(event.get("duration_seconds", 0.0) or 0.0)
    return start, end


def _time_to_seconds(value: str) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    parts = text.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
            return float(hours * 3600 + minutes * 60 + seconds)
        if len(parts) == 2:
            minutes, seconds = map(int, parts)
            return float(minutes * 60 + seconds)
        return float(text)
    except Exception:
        return 0.0


def _make_evidence_description(
    evidence_type: str,
    actors: List[str],
    objects: List[str],
    location: str,
) -> str:
    actor_text = ", ".join(actors) if actors else "Unknown actor"
    object_text = ", ".join(objects) if objects else "no named object"
    if evidence_type == "enter":
        return f"{actor_text} entered or arrived at {location}."
    if evidence_type == "exit":
        return f"{actor_text} exited or departed from {location}."
    if evidence_type == "wait":
        return f"{actor_text} remained present at {location}."
    if evidence_type == "approach":
        return f"{actor_text} approached {object_text} at {location}."
    if evidence_type == "remain_near":
        return f"{actor_text} remained near {object_text} at {location}."
    if evidence_type == "carried":
        return f"{actor_text} carried or held {object_text} at {location}."
    if evidence_type == "object_moved":
        return f"{object_text} appears to have moved or been handled at {location}."
    if evidence_type == "group_formed":
        return f"{actor_text} formed a group presence at {location}."
    if evidence_type == "vehicle_arrived":
        return f"{actor_text} arrived at {location}."
    if evidence_type == "vehicle_departed":
        return f"{actor_text} departed from {location}."
    return f"{actor_text} was involved in {evidence_type} at {location}."


class EvidenceGraphBuilder:
    def __init__(self) -> None:
        self._evidence_counter = 0
        self._fallback_object_map: Dict[str, str] = {}
        self._fallback_object_counter = 0

    def build(self, events: List[Dict[str, Any]], actor_timeline: Dict[str, Any]) -> Dict[str, Any]:
        actor_index = _timeline_index(actor_timeline)
        evidence_units: List[EvidenceUnit] = []
        nodes: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []

        for actor in actor_timeline.get("actors", []):
            global_id = actor.get("global_actor_id", "")
            nodes[f"actor:{global_id}"] = {
                "node_id": f"actor:{global_id}",
                "node_type": actor.get("entity_type", "actor"),
                "global_actor_id": global_id,
                "attributes": actor.get("attributes", {}),
            }

        for event in events:
            location = str(event.get("location_text", "") or "the monitored area")
            start_seconds, end_seconds = _event_window_seconds(event)
            event_id = str(event.get("event_id", ""))
            event_actors, event_vehicles, event_objects = self._resolve_event_entities(event, actor_index, nodes)
            activities = [str(item).lower() for item in event.get("activities", []) or []]
            behavioral_flags = [str(item).lower() for item in event.get("behavioral_flags", []) or []]

            if event_actors:
                if any(flag in behavioral_flags for flag in ("access_event",)) or any(term in " ".join(activities) for term in ("enter", "arriv")):
                    evidence_units.append(self._make_unit(start_seconds, end_seconds, event_actors, "enter", [], location, event_id))
                if any(flag in behavioral_flags for flag in ("egress_event",)) or any(term in " ".join(activities) for term in ("exit", "depart", "leave")):
                    evidence_units.append(self._make_unit(start_seconds, end_seconds, event_actors, "exit", [], location, event_id))
                if any(flag in behavioral_flags for flag in ("prolonged_activity", "loitering")) or float(event.get("duration_seconds", 0.0) or 0.0) >= 20.0:
                    evidence_units.append(self._make_unit(start_seconds, end_seconds, event_actors, "wait", [], location, event_id))

            if event_actors and event_objects:
                evidence_units.append(self._make_unit(start_seconds, end_seconds, event_actors, "approach", event_objects, location, event_id))
                if float(event.get("duration_seconds", 0.0) or 0.0) >= 10.0:
                    evidence_units.append(self._make_unit(start_seconds, end_seconds, event_actors, "remain_near", event_objects, location, event_id))
                if "carrying_object" in behavioral_flags or any("carrying" in activity or "holding" in activity for activity in activities):
                    evidence_units.append(self._make_unit(start_seconds, end_seconds, event_actors, "carried", event_objects, location, event_id))

            if event_objects and any("moving" in " ".join(str(attr).lower() for attr in obj.get("attributes", []) or []) for obj in event.get("objects", []) or []):
                evidence_units.append(self._make_unit(start_seconds, end_seconds, [], "object_moved", event_objects, location, event_id))

            if len(event_actors) >= 2 or "multi_person" in behavioral_flags or int(event.get("participant_count", 0) or 0) >= 3:
                evidence_units.append(self._make_unit(start_seconds, end_seconds, event_actors, "group_formed", [], location, event_id))

            if event_vehicles:
                vehicle_text = " ".join(activities + behavioral_flags)
                if any(term in vehicle_text for term in ("enter", "arriv", "vehicle_movement")):
                    evidence_units.append(self._make_unit(start_seconds, end_seconds, event_vehicles, "vehicle_arrived", [], location, event_id))
                if any(term in vehicle_text for term in ("exit", "depart", "egress_event")):
                    evidence_units.append(self._make_unit(start_seconds, end_seconds, event_vehicles, "vehicle_departed", [], location, event_id))

        for unit in evidence_units:
            node_id = f"evidence:{unit.evidence_id}"
            nodes[node_id] = {
                "node_id": node_id,
                "node_type": "evidence_unit",
                **asdict(unit),
            }
            location_node = f"location:{unit.location}"
            nodes.setdefault(
                location_node,
                {
                    "node_id": location_node,
                    "node_type": "location",
                    "name": unit.location,
                },
            )
            edges.append(
                {
                    "edge_type": "moved_to",
                    "source": node_id,
                    "target": location_node,
                }
            )
            for actor_id in unit.actors:
                edges.append(
                    {
                        "edge_type": self._edge_for_evidence(unit.evidence_type),
                        "source": f"actor:{actor_id}",
                        "target": node_id,
                    }
                )
            for object_id in unit.objects:
                object_node = f"object:{object_id}"
                nodes.setdefault(
                    object_node,
                    {
                        "node_id": object_node,
                        "node_type": "object",
                        "name": object_id,
                    },
                )
                edges.append(
                    {
                        "edge_type": self._object_edge_for_evidence(unit.evidence_type),
                        "source": node_id,
                        "target": object_node,
                    }
                )

        return {
            "summary": {
                "evidence_unit_count": len(evidence_units),
                "node_count": len(nodes),
                "edge_count": len(edges),
                "actor_node_count": sum(1 for node in nodes.values() if node.get("node_type") in {"actor", "vehicle", "object"}),
                "location_node_count": sum(1 for node in nodes.values() if node.get("node_type") == "location"),
            },
            "evidence_units": [asdict(unit) for unit in evidence_units],
            "nodes": list(nodes.values()),
            "edges": edges,
        }

    def _resolve_event_entities(
        self,
        event: Dict[str, Any],
        actor_index: Dict[str, Dict[str, Any]],
        nodes: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[str], List[str], List[str]]:
        actors: List[str] = []
        vehicles: List[str] = []
        objects: List[str] = []
        for obj in event.get("objects", []) or []:
            if not isinstance(obj, dict):
                continue
            entity_type = _entity_type(obj)
            global_id = _lookup_global_id(obj, actor_index)
            if global_id is None and entity_type == "object":
                global_id = self._fallback_global_object(obj)
            if global_id is None:
                continue
            node_id = f"actor:{global_id}"
            nodes.setdefault(
                node_id,
                {
                    "node_id": node_id,
                    "node_type": entity_type,
                    "global_actor_id": global_id,
                    "attributes": {
                        "subtype": obj.get("subtype", ""),
                        "upper_color": obj.get("color", ""),
                        "attributes": obj.get("attributes", []),
                    },
                },
            )
            if entity_type == "actor":
                actors.append(global_id)
            elif entity_type == "vehicle":
                vehicles.append(global_id)
            else:
                objects.append(global_id)
        return sorted(set(actors)), sorted(set(vehicles)), sorted(set(objects))

    def _fallback_global_object(self, obj: Dict[str, Any]) -> str:
        key = _fallback_object_key(obj)
        global_id = self._fallback_object_map.get(key)
        if global_id is not None:
            return global_id
        self._fallback_object_counter += 1
        global_id = f"global_object_{self._fallback_object_counter}"
        self._fallback_object_map[key] = global_id
        return global_id

    def _make_unit(
        self,
        start_seconds: float,
        end_seconds: float,
        actors: List[str],
        evidence_type: str,
        objects: List[str],
        location: str,
        event_id: str,
    ) -> EvidenceUnit:
        self._evidence_counter += 1
        evidence_id = f"ev_{self._evidence_counter:03d}"
        return EvidenceUnit(
            evidence_id=evidence_id,
            time_start=start_seconds,
            time_end=end_seconds,
            actors=actors,
            evidence_type=evidence_type,
            objects=objects,
            location=location,
            event_id=event_id,
            description=_make_evidence_description(evidence_type, actors, objects, location),
        )

    @staticmethod
    def _edge_for_evidence(evidence_type: str) -> str:
        if evidence_type == "approach":
            return "approached"
        if evidence_type == "remain_near":
            return "remained_near"
        if evidence_type == "carried":
            return "carried"
        return "moved_to"

    @staticmethod
    def _object_edge_for_evidence(evidence_type: str) -> str:
        if evidence_type in {"approach", "remain_near", "carried"}:
            return "interacted_with"
        if evidence_type == "object_moved":
            return "moved_to"
        return "interacted_with"
