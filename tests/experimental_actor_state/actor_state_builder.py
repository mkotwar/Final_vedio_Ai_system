from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


PERSON_TERMS = {"person", "pedestrian", "employee", "customer", "guard", "security", "man", "woman", "rider", "driver", "passenger"}
VEHICLE_TERMS = {"vehicle", "car", "truck", "bus", "motorcycle", "bicycle", "bike", "scooter", "van"}
CLOTHING_TERMS = {
    "shirt",
    "jacket",
    "hoodie",
    "coat",
    "sweater",
    "jeans",
    "pants",
    "trousers",
    "dress",
    "skirt",
    "uniform",
    "cap",
    "hat",
    "helmet",
    "bag",
    "backpack",
}


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


def _entity_type(obj: Dict[str, Any]) -> str:
    obj_type = str(obj.get("type", "")).lower()
    subtype = str(obj.get("subtype", "")).lower()
    text = f"{obj_type} {subtype}"
    if any(term in text for term in PERSON_TERMS):
        return "actor"
    if any(term in text for term in VEHICLE_TERMS):
        return "vehicle"
    return "object"


def _normalized_attributes(obj: Dict[str, Any]) -> List[str]:
    values = []
    for item in obj.get("attributes", []) or []:
        value = str(item).strip().lower()
        if value:
            values.append(value)
    return sorted(set(values))


def _clothing_attributes(attributes: List[str]) -> List[str]:
    results = []
    for item in attributes:
        if any(term in item for term in CLOTHING_TERMS):
            results.append(item)
    return sorted(set(results))


def _upper_color(color: str, attributes: List[str]) -> str:
    normalized = str(color or "").strip().lower()
    if normalized:
        return normalized
    for item in attributes:
        tokens = item.split()
        if tokens:
            return tokens[0]
    return ""


def _global_prefix(entity_type: str) -> str:
    if entity_type == "actor":
        return "global_actor"
    if entity_type == "vehicle":
        return "global_vehicle"
    return "global_object"


def _event_activity(event: Dict[str, Any], obj: Dict[str, Any]) -> str:
    attributes = " ".join(_normalized_attributes(obj))
    activities = [str(item) for item in event.get("activities", []) or []]
    if activities:
        return activities[0]
    if attributes:
        return attributes.split(",")[0].strip()
    return "present"


def _similarity_jaccard(a: List[str], b: List[str]) -> float:
    if not a and not b:
        return 1.0
    set_a = set(a)
    set_b = set(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a.intersection(set_b)) / len(set_a.union(set_b))


@dataclass
class ActorObservation:
    time: float
    event_id: str
    event_type: str
    activity: str
    start_time: str
    end_time: str
    location_text: str
    confidence: float
    source_object_id: str


@dataclass
class GlobalEntity:
    global_actor_id: str
    entity_type: str
    subtype: str
    upper_color: str
    source_object_ids: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    clothing: List[str] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    matched_by_id_count: int = 0
    soft_match_count: int = 0
    unmatched_observation_count: int = 0
    timeline: List[ActorObservation] = field(default_factory=list)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "global_actor_id": self.global_actor_id,
            "entity_type": self.entity_type,
            "attributes": {
                "subtype": self.subtype,
                "upper_color": self.upper_color,
                "clothing": self.clothing,
                "attributes": self.attributes,
                "source_object_ids": self.source_object_ids,
            },
            "timeline": [
                {
                    "time": item.time,
                    "event_id": item.event_id,
                    "event_type": item.event_type,
                    "activity": item.activity,
                    "start_time": item.start_time,
                    "end_time": item.end_time,
                    "location_text": item.location_text,
                    "confidence": item.confidence,
                    "source_object_id": item.source_object_id,
                }
                for item in sorted(self.timeline, key=lambda obs: (obs.time, obs.event_id))
            ],
            "continuity_stats": {
                "first_seen_seconds": self.first_seen,
                "last_seen_seconds": self.last_seen,
                "presence_duration_seconds": round(max(0.0, self.last_seen - self.first_seen), 2),
                "observation_count": len(self.timeline),
                "matched_by_id_count": self.matched_by_id_count,
                "soft_match_count": self.soft_match_count,
                "unmatched_observation_count": self.unmatched_observation_count,
            },
        }


class ActorStateBuilder:
    EXACT_ID_SCORE = 1.0
    SUBTYPE_WEIGHT = 0.40
    COLOR_WEIGHT = 0.20
    CLOTHING_WEIGHT = 0.20
    ATTRIBUTE_WEIGHT = 0.15
    TEMPORAL_WEIGHT = 0.05
    SOFT_MATCH_THRESHOLD = 0.58
    TEMPORAL_WINDOW_SECONDS = 45.0

    def __init__(self) -> None:
        self._entities: List[GlobalEntity] = []
        self._id_lookup: Dict[Tuple[str, str], str] = {}
        self._counters = {"actor": 0, "vehicle": 0, "object": 0}

    def build(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        ordered_events = sorted(events, key=lambda item: _time_to_seconds(item.get("start_time", "")))
        for event in ordered_events:
            self._consume_event(event)
        payload_entities = [entity.to_payload() for entity in self._entities]
        payload_entities.sort(key=lambda item: item["global_actor_id"])
        return {
            "summary": self._summary(payload_entities),
            "actors": payload_entities,
        }

    def _summary(self, payload_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        counts = {"actor": 0, "vehicle": 0, "object": 0}
        total_observations = 0
        matched_by_id = 0
        soft_matches = 0
        for item in payload_entities:
            entity_type = item.get("entity_type", "object")
            counts[entity_type] = counts.get(entity_type, 0) + 1
            stats = item.get("continuity_stats", {})
            total_observations += int(stats.get("observation_count", 0))
            matched_by_id += int(stats.get("matched_by_id_count", 0))
            soft_matches += int(stats.get("soft_match_count", 0))
        return {
            "total_actors": len(payload_entities),
            "entity_type_counts": counts,
            "total_observations": total_observations,
            "matched_by_id_count": matched_by_id,
            "soft_match_count": soft_matches,
        }

    def _consume_event(self, event: Dict[str, Any]) -> None:
        event_start_seconds = _time_to_seconds(event.get("start_time", ""))
        event_end_seconds = _time_to_seconds(event.get("end_time", ""))
        for obj in event.get("objects", []) or []:
            if not isinstance(obj, dict):
                continue
            self._consume_object(event, obj, event_start_seconds, event_end_seconds)

    def _consume_object(
        self,
        event: Dict[str, Any],
        obj: Dict[str, Any],
        event_start_seconds: float,
        event_end_seconds: float,
    ) -> None:
        entity_type = _entity_type(obj)
        object_id = str(obj.get("id", "")).strip().lower()
        global_id = None
        matched_by_id = False
        matched_soft = False

        if object_id:
            global_id = self._id_lookup.get((entity_type, object_id))
            if global_id is not None:
                matched_by_id = True

        if global_id is None:
            candidate = self._find_soft_match(entity_type, obj, event_start_seconds)
            if candidate is not None:
                global_id = candidate.global_actor_id
                matched_soft = True

        if global_id is None:
            entity = self._create_entity(entity_type, obj, event_start_seconds)
        else:
            entity = self._entity_by_id(global_id)
            if entity is None:
                entity = self._create_entity(entity_type, obj, event_start_seconds)
                global_id = entity.global_actor_id

        self._apply_observation(
            entity=entity,
            event=event,
            obj=obj,
            start_seconds=event_start_seconds,
            end_seconds=event_end_seconds,
            matched_by_id=matched_by_id,
            matched_soft=matched_soft,
        )

        if object_id:
            self._id_lookup[(entity_type, object_id)] = entity.global_actor_id

    def _create_entity(self, entity_type: str, obj: Dict[str, Any], first_seen: float) -> GlobalEntity:
        self._counters[entity_type] += 1
        global_id = f"{_global_prefix(entity_type)}_{self._counters[entity_type]}"
        attributes = _normalized_attributes(obj)
        entity = GlobalEntity(
            global_actor_id=global_id,
            entity_type=entity_type,
            subtype=str(obj.get("subtype", "")).strip().lower(),
            upper_color=_upper_color(str(obj.get("color", "")), attributes),
            source_object_ids=[str(obj.get("id", "")).strip()] if str(obj.get("id", "")).strip() else [],
            attributes=attributes,
            clothing=_clothing_attributes(attributes),
            first_seen=first_seen,
            last_seen=first_seen,
            unmatched_observation_count=1,
        )
        self._entities.append(entity)
        return entity

    def _entity_by_id(self, global_id: str) -> Optional[GlobalEntity]:
        for entity in self._entities:
            if entity.global_actor_id == global_id:
                return entity
        return None

    def _find_soft_match(self, entity_type: str, obj: Dict[str, Any], event_start_seconds: float) -> Optional[GlobalEntity]:
        best_score = 0.0
        best_entity: Optional[GlobalEntity] = None
        for entity in self._entities:
            if entity.entity_type != entity_type:
                continue
            score = self._soft_match_score(entity, obj, event_start_seconds)
            if score > best_score:
                best_score = score
                best_entity = entity
        if best_score >= self.SOFT_MATCH_THRESHOLD:
            return best_entity
        return None

    def _soft_match_score(self, entity: GlobalEntity, obj: Dict[str, Any], event_start_seconds: float) -> float:
        subtype = str(obj.get("subtype", "")).strip().lower()
        attributes = _normalized_attributes(obj)
        clothing = _clothing_attributes(attributes)
        color = _upper_color(str(obj.get("color", "")), attributes)

        subtype_score = 1.0 if subtype and subtype == entity.subtype else 0.0
        color_score = 1.0 if color and color == entity.upper_color else 0.0
        clothing_score = _similarity_jaccard(clothing, entity.clothing)
        attribute_score = _similarity_jaccard(attributes, entity.attributes)

        time_gap = max(0.0, event_start_seconds - entity.last_seen)
        if time_gap <= self.TEMPORAL_WINDOW_SECONDS:
            temporal_score = 1.0 - min(time_gap / self.TEMPORAL_WINDOW_SECONDS, 1.0)
        else:
            temporal_score = 0.0

        return (
            (subtype_score * self.SUBTYPE_WEIGHT)
            + (color_score * self.COLOR_WEIGHT)
            + (clothing_score * self.CLOTHING_WEIGHT)
            + (attribute_score * self.ATTRIBUTE_WEIGHT)
            + (temporal_score * self.TEMPORAL_WEIGHT)
        )

    def _apply_observation(
        self,
        entity: GlobalEntity,
        event: Dict[str, Any],
        obj: Dict[str, Any],
        start_seconds: float,
        end_seconds: float,
        matched_by_id: bool,
        matched_soft: bool,
    ) -> None:
        object_id = str(obj.get("id", "")).strip()
        if object_id and object_id not in entity.source_object_ids:
            entity.source_object_ids.append(object_id)

        incoming_attributes = _normalized_attributes(obj)
        entity.attributes = sorted(set(entity.attributes).union(incoming_attributes))
        entity.clothing = sorted(set(entity.clothing).union(_clothing_attributes(incoming_attributes)))
        entity.upper_color = entity.upper_color or _upper_color(str(obj.get("color", "")), incoming_attributes)
        entity.subtype = entity.subtype or str(obj.get("subtype", "")).strip().lower()
        entity.last_seen = max(entity.last_seen, end_seconds)

        if matched_by_id:
            entity.matched_by_id_count += 1
        elif matched_soft:
            entity.soft_match_count += 1
        else:
            entity.unmatched_observation_count += 1

        entity.timeline.append(
            ActorObservation(
                time=start_seconds,
                event_id=str(event.get("event_id", "")),
                event_type=str(event.get("event_type", "")),
                activity=_event_activity(event, obj),
                start_time=str(event.get("start_time", "")),
                end_time=str(event.get("end_time", "")),
                location_text=str(event.get("location_text", "")),
                confidence=float(event.get("confidence", 0.0) or 0.0),
                source_object_id=object_id,
            )
        )
