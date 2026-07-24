from __future__ import annotations


SUPPORTED_VEHICLE_BODY_TYPES = (
    "HATCHBACK",
    "SEDAN",
    "SUV",
    "MUV",
    "COUPE",
    "CONVERTIBLE",
    "PICKUP",
    "VAN",
    "MINIVAN",
    "TRUCK",
    "BUS",
    "THREE_WHEELER",
    "MOTORCYCLE",
    "UNKNOWN",
)

_ALIASES = {
    "hatchback": "HATCHBACK",
    "sedan": "SEDAN",
    "saloon": "SEDAN",
    "suv": "SUV",
    "sport utility vehicle": "SUV",
    "muv": "MUV",
    "utility vehicle": "MUV",
    "multi utility vehicle": "MUV",
    "coupe": "COUPE",
    "convertible": "CONVERTIBLE",
    "pickup": "PICKUP",
    "pickup truck": "PICKUP",
    "van": "VAN",
    "minivan": "MINIVAN",
    "mini van": "MINIVAN",
    "truck": "TRUCK",
    "bus": "BUS",
    "three wheeler": "THREE_WHEELER",
    "threewheeler": "THREE_WHEELER",
    "three wheeled": "THREE_WHEELER",
    "auto rickshaw": "THREE_WHEELER",
    "motorcycle": "MOTORCYCLE",
    "bike": "MOTORCYCLE",
    "motorbike": "MOTORCYCLE",
    "unknown": "UNKNOWN",
}


def normalize_vehicle_body_type(value: str) -> str:
    normalized = " ".join(str(value).strip().lower().replace("_", " ").replace("-", " ").split())
    if not normalized:
        return "UNKNOWN"
    if normalized in _ALIASES:
        return _ALIASES[normalized]
    upper = normalized.upper()
    if upper in SUPPORTED_VEHICLE_BODY_TYPES:
        return upper
    return "UNKNOWN"
