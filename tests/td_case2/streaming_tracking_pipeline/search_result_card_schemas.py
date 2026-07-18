from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .serialization import dataclass_to_dict


@dataclass(frozen=True)
class VehicleResultCard:
    rank: int
    record_id: str
    source_id: str
    track_id: int
    track_generation: int
    title: str
    subtitle: str
    time_label: str
    plate_label: str
    colour_label: str
    status_badge: str
    confidence_label: str
    object_class: str | None
    colour: str | None
    plate_text: str | None
    plate_status: str
    plate_confidence: float | None
    first_seen_sec: float | None
    last_seen_sec: float | None
    duration_sec: float | None
    thumbnail_path: str | None
    secondary_image_path: str | None
    search_score: float
    matched_filters: list[str] = field(default_factory=list)
    matched_tokens: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass(frozen=True)
class VehicleResultCardPackage:
    raw_query: str
    parsed_query: dict[str, Any]
    total_matches: int
    returned_cards: int
    cards: list[VehicleResultCard]
    runtime_sec: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)
