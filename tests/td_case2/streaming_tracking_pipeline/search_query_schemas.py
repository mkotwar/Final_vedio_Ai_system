from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .serialization import dataclass_to_dict


@dataclass(frozen=True)
class VehicleSearchQuery:
    raw_query: str
    object_classes: list[str] = field(default_factory=list)
    colours: list[str] = field(default_factory=list)
    plate_text: str | None = None
    plate_prefix: str | None = None
    plate_statuses: list[str] = field(default_factory=list)
    start_time_sec: float | None = None
    end_time_sec: float | None = None
    track_id: int | None = None
    track_generation: int | None = None
    free_text_tokens: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass(frozen=True)
class VehicleSearchResult:
    rank: int
    score: float
    record_id: str
    source_id: str
    track_id: int
    track_generation: int
    object_class: str | None
    colour: str | None
    plate_text: str | None
    plate_status: str
    first_seen_sec: float | None
    last_seen_sec: float | None
    representative_vehicle_crop_path: str | None
    representative_plate_crop_path: str | None
    matched_filters: list[str] = field(default_factory=list)
    matched_tokens: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    score_components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass(frozen=True)
class VehicleSearchResponse:
    query: VehicleSearchQuery
    total_records_searched: int
    total_matches: int
    results: list[VehicleSearchResult]
    runtime_sec: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)
