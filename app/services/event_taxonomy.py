"""Central contract for event types used across aggregation, summary, and search."""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class EventTypeDefinition:
    event_type: str
    label: str
    severity: int
    category: str
    notable: bool = False
    notable_severity: str = "low"
    search_boost: float = 0.0


EVENT_NORMAL_ACTIVITY = "normal_activity"
EVENT_VEHICLE_MOVEMENT = "vehicle_movement"
EVENT_PEDESTRIAN_ACTIVITY = "pedestrian_activity"
EVENT_COLLISION_OR_ACCIDENT = "collision_or_accident"
EVENT_FIRE_INCIDENT = "fire_incident"
EVENT_ROBBERY_INCIDENT = "robbery_incident"
EVENT_MEDICAL_EMERGENCY = "medical_emergency"
EVENT_WEAPON_DRAWN = "weapon_drawn"
EVENT_FALL_INCIDENT = "fall_incident"
EVENT_INTRUSION = "intrusion"
EVENT_LOITERING = "loitering"
EVENT_PHYSICAL_ALTERCATION = "physical_altercation"
EVENT_ABANDONED_OBJECT = "abandoned_object"
EVENT_VEHICLE_SPEEDING = "vehicle_speeding"


EVENT_TAXONOMY: Dict[str, EventTypeDefinition] = {
    EVENT_COLLISION_OR_ACCIDENT: EventTypeDefinition(
        EVENT_COLLISION_OR_ACCIDENT, "Collision or accident", 100, "safety",
        notable=True, notable_severity="high", search_boost=0.50,
    ),
    EVENT_FIRE_INCIDENT: EventTypeDefinition(
        EVENT_FIRE_INCIDENT, "Fire incident", 100, "safety",
        notable=True, notable_severity="critical", search_boost=0.75,
    ),
    EVENT_ROBBERY_INCIDENT: EventTypeDefinition(
        EVENT_ROBBERY_INCIDENT, "Robbery or theft", 100, "security",
        notable=True, notable_severity="high", search_boost=0.50,
    ),
    EVENT_MEDICAL_EMERGENCY: EventTypeDefinition(
        EVENT_MEDICAL_EMERGENCY, "Medical emergency", 95, "safety",
        notable=True, notable_severity="high", search_boost=0.50,
    ),
    EVENT_WEAPON_DRAWN: EventTypeDefinition(
        EVENT_WEAPON_DRAWN, "Weapon detected", 95, "security",
        notable=True, notable_severity="high", search_boost=1.00,
    ),
    EVENT_FALL_INCIDENT: EventTypeDefinition(
        EVENT_FALL_INCIDENT, "Fall incident", 90, "safety",
        notable=True, notable_severity="medium", search_boost=0.50,
    ),
    EVENT_INTRUSION: EventTypeDefinition(
        EVENT_INTRUSION, "Intrusion", 85, "security",
        notable=True, notable_severity="high", search_boost=0.50,
    ),
    EVENT_LOITERING: EventTypeDefinition(
        EVENT_LOITERING, "Loitering", 60, "security",
        notable=True, notable_severity="medium",
    ),
    EVENT_VEHICLE_MOVEMENT: EventTypeDefinition(
        EVENT_VEHICLE_MOVEMENT, "Vehicle movement", 30, "traffic",
    ),
    EVENT_PEDESTRIAN_ACTIVITY: EventTypeDefinition(
        EVENT_PEDESTRIAN_ACTIVITY, "Pedestrian activity", 20, "people",
    ),
    EVENT_NORMAL_ACTIVITY: EventTypeDefinition(
        EVENT_NORMAL_ACTIVITY, "Normal activity", 10, "normal",
    ),
    EVENT_PHYSICAL_ALTERCATION: EventTypeDefinition(
        EVENT_PHYSICAL_ALTERCATION, "Physical altercation", 90, "security",
        notable=True, notable_severity="high", search_boost=0.50,
    ),
    EVENT_ABANDONED_OBJECT: EventTypeDefinition(
        EVENT_ABANDONED_OBJECT, "Abandoned object", 70, "security",
        notable=True, notable_severity="medium",
    ),
    EVENT_VEHICLE_SPEEDING: EventTypeDefinition(
        EVENT_VEHICLE_SPEEDING, "Vehicle speeding", 55, "traffic",
        notable=True, notable_severity="medium",
    ),
}


EVENT_TYPE_ALIASES = {
    "accident": EVENT_COLLISION_OR_ACCIDENT,
    "collision": EVENT_COLLISION_OR_ACCIDENT,
    "crash": EVENT_COLLISION_OR_ACCIDENT,
    "vehicle_collision": EVENT_COLLISION_OR_ACCIDENT,
    "fire": EVENT_FIRE_INCIDENT,
    "smoke": EVENT_FIRE_INCIDENT,
    "fire_smoke_detected": EVENT_FIRE_INCIDENT,
    "weapon": EVENT_WEAPON_DRAWN,
    "weapon_visible": EVENT_WEAPON_DRAWN,
    "fall": EVENT_FALL_INCIDENT,
    "person_fall": EVENT_FALL_INCIDENT,
    "restricted_area_activity": EVENT_INTRUSION,
    "fight": EVENT_PHYSICAL_ALTERCATION,
    "abandonment": EVENT_ABANDONED_OBJECT,
    "speeding": EVENT_VEHICLE_SPEEDING,
}


def normalize_event_type(event_type: str) -> str:
    cleaned = str(event_type or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not cleaned or cleaned == "none":
        return EVENT_NORMAL_ACTIVITY
    return EVENT_TYPE_ALIASES.get(cleaned, cleaned)


def get_event_definition(event_type: str) -> EventTypeDefinition:
    normalized = normalize_event_type(event_type)
    return EVENT_TAXONOMY.get(
        normalized,
        EventTypeDefinition(normalized, normalized.replace("_", " ").title(), 15, "unknown"),
    )


def get_event_severity(event_type: str) -> int:
    return get_event_definition(event_type).severity


def get_event_search_boost(event_type: str) -> float:
    return get_event_definition(event_type).search_boost


def is_notable_event_type(event_type: str) -> bool:
    return get_event_definition(event_type).notable


def get_notable_severity(event_type: str) -> str:
    return get_event_definition(event_type).notable_severity
