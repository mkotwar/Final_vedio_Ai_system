from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


NORMALIZED_CLASSES = {
    "person",
    "car",
    "motorcycle",
    "bicycle",
    "bus",
    "truck",
    "3wheeler",
    "other_vehicle",
    "other_object",
}

VEHICLE_CLASSES = {"car", "motorcycle", "bicycle", "bus", "truck", "3wheeler", "other_vehicle"}
ANPR_VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "3wheeler", "other_vehicle"}

_CLASS_SYNONYMS = {
    "person": "person",
    "pedestrian": "person",
    "human": "person",
    "car": "car",
    "auto": "car",
    "automobile": "car",
    "sedan": "car",
    "suv": "car",
    "van": "car",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "motor bike": "motorcycle",
    "bike": "motorcycle",
    "scooter": "motorcycle",
    "two wheeler": "motorcycle",
    "two_wheeler": "motorcycle",
    "2wheeler": "motorcycle",
    "bicycle": "bicycle",
    "cycle": "bicycle",
    "bus": "bus",
    "truck": "truck",
    "lorry": "truck",
    "3wheeler": "3wheeler",
    "three wheeler": "3wheeler",
    "rickshaw": "3wheeler",
    "auto rickshaw": "3wheeler",
}


@dataclass(frozen=True)
class NormalizedClass:
    raw_class_id: int | None
    raw_class_name: str | None
    normalized_class_name: str
    object_group: str
    is_vehicle: bool
    is_anpr_eligible: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_class_id": self.raw_class_id,
            "raw_class_name": self.raw_class_name,
            "normalized_class_name": self.normalized_class_name,
            "object_group": self.object_group,
            "is_vehicle": self.is_vehicle,
            "is_anpr_eligible": self.is_anpr_eligible,
        }


def normalize_class_name(raw_class_name: str | None, raw_class_id: int | None = None) -> NormalizedClass:
    normalized_input = _normalize_text(raw_class_name)
    normalized = _CLASS_SYNONYMS.get(normalized_input)
    if normalized is None:
        normalized = "other_object"
    object_group = "person" if normalized == "person" else "vehicle" if normalized in VEHICLE_CLASSES else "object"
    return NormalizedClass(
        raw_class_id=raw_class_id,
        raw_class_name=raw_class_name,
        normalized_class_name=normalized,
        object_group=object_group,
        is_vehicle=normalized in VEHICLE_CLASSES,
        is_anpr_eligible=normalized in ANPR_VEHICLE_CLASSES,
    )


def normalize_model_names(names: dict[int, str] | list[str] | tuple[str, ...]) -> dict[int, dict[str, Any]]:
    items = names.items() if isinstance(names, dict) else enumerate(names)
    return {int(class_id): normalize_class_name(str(class_name), int(class_id)).to_dict() for class_id, class_name in items}


def is_vehicle_class(class_name: str | None) -> bool:
    return normalize_class_name(class_name).is_vehicle


def is_anpr_eligible_class(class_name: str | None) -> bool:
    return normalize_class_name(class_name).is_anpr_eligible


def object_group_for_class(class_name: str | None) -> str:
    return normalize_class_name(class_name).object_group


def _normalize_text(value: str | None) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())
