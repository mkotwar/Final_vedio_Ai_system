from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .serialization import dataclass_to_dict


@dataclass(frozen=True)
class SearchableVehicleRecord:
    record_id: str
    source_id: str
    video_path: str | None
    track_id: int
    track_generation: int
    object_class: str | None
    first_frame_index: int | None
    last_frame_index: int | None
    first_seen_sec: float | None
    last_seen_sec: float | None
    duration_sec: float | None
    plate_text: str | None
    plate_status: str
    plate_confidence: float
    plate_support_count: int
    normalized_colour: str | None
    raw_colour: str | None
    representative_frame_index: int | None
    representative_timestamp_sec: float | None
    representative_vehicle_crop_path: str | None
    representative_plate_crop_path: str | None
    selected_frame_index: int | None = None
    selected_timestamp_sec: float | None = None
    primary_crop_paths: list[str] = field(default_factory=list)
    fallback_crop_paths: list[str] = field(default_factory=list)
    object_type: str = "vehicle"
    object_group: str = "vehicle"
    full_frame_path: str | None = None
    object_crop_path: str | None = None
    event_eligible: bool = True
    raw_class_name: str | None = None
    normalized_class_name: str | None = None
    dominant_colour: str | None = None
    colour_confidence: float | None = None
    colour_coverage: float | None = None
    colour_method: str | None = None
    colour_region: str | None = None
    colour_warnings: list[str] = field(default_factory=list)
    upper_clothing_color: str | None = None
    lower_clothing_color: str | None = None
    dominant_clothing_color: str | None = None
    clothing_color_confidence: float | None = None
    clothing_color_status: str | None = None
    vehicle_colour_status: str | None = None
    search_text: str = ""
    searchable_tokens: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)
