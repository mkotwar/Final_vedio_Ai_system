from __future__ import annotations

from enum import Enum


class VehicleClass(str, Enum):
    THREE_WHEELER = "3WHEELER"
    BUS = "BUS"
    CAR = "CAR"
    MOTORCYCLE = "MOTORCYCLE"
    TRUCK = "TRUCK"
    UNKNOWN = "UNKNOWN"


RUNTIME_VEHICLE_CLASSES = ("3wheeler", "bus", "car", "motorcycle", "truck", "unknown")

_RUNTIME_CLASS_MAP = {
    "3wheeler": "3wheeler",
    "3 wheeler": "3wheeler",
    "3-wheeler": "3wheeler",
    "bus": "bus",
    "car": "car",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "bike": "motorcycle",
    "truck": "truck",
    "lorry": "truck",
    "unknown": "unknown",
}

_CANONICAL_CLASS_MAP = {
    "3wheeler": VehicleClass.THREE_WHEELER,
    "bus": VehicleClass.BUS,
    "car": VehicleClass.CAR,
    "motorcycle": VehicleClass.MOTORCYCLE,
    "truck": VehicleClass.TRUCK,
    "unknown": VehicleClass.UNKNOWN,
}


def _normalize_key(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def normalize_runtime_vehicle_class(value: str | None) -> str | None:
    normalized = _normalize_key(value)
    if not normalized:
        return None
    return _RUNTIME_CLASS_MAP.get(normalized)


def normalize_vehicle_class(value: str | None) -> VehicleClass:
    runtime_class = normalize_runtime_vehicle_class(value)
    if runtime_class is None:
        return VehicleClass.UNKNOWN
    return _CANONICAL_CLASS_MAP[runtime_class]


def is_supported_vehicle_class(value: str | None) -> bool:
    normalized = normalize_runtime_vehicle_class(value)
    return normalized is not None and normalized != "unknown"
