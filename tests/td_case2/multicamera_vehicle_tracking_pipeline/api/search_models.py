from __future__ import annotations

from datetime import date as calendar_date, datetime, time
from enum import Enum
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..enrichment.plate_text_normalizer import normalize_registration_text
from ..enrichment.vehicle_colour_mapping import SUPPORTED_VEHICLE_COLOURS, normalize_vehicle_colour
from ..persistence.vehicle_class_mapping import VehicleClass, normalize_vehicle_class
from .response_models import MediaReference, PlateResult


_CAMERA_CODE_PATTERN = re.compile(r"^[A-Z0-9_:-]+$")


class SearchResultScope(str, Enum):
    LOCAL_TRACKS = "LOCAL_TRACKS"
    GLOBAL_VEHICLES = "GLOBAL_VEHICLES"
    ALL = "ALL"


class PlateMatchType(str, Enum):
    EXACT = "EXACT"
    CONTAINS = "CONTAINS"
    STARTS_WITH = "STARTS_WITH"
    ENDS_WITH = "ENDS_WITH"


class VehicleSearchSortBy(str, Enum):
    RELEVANCE = "RELEVANCE"
    FIRST_SEEN = "FIRST_SEEN"
    LAST_SEEN = "LAST_SEEN"
    CONFIDENCE = "CONFIDENCE"
    PLATE = "PLATE"


class VehicleSearchSortOrder(str, Enum):
    ASC = "ASC"
    DESC = "DESC"


class VehicleSearchQuery(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    run_code: str | None = None
    result_scope: SearchResultScope = SearchResultScope.ALL
    vehicle_class: str | None = None
    colour: str | None = None
    plate: str | None = None
    plate_match_type: PlateMatchType = PlateMatchType.CONTAINS
    camera_codes: tuple[str, ...] = ()
    date: calendar_date | None = None
    start_time: time | None = None
    end_time: time | None = None
    minimum_confidence: float | None = Field(default=0.5, ge=0, le=1)
    multi_camera_only: bool = False
    verified_plate_only: bool = False
    limit: int = Field(default=25, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    sort_by: VehicleSearchSortBy = VehicleSearchSortBy.RELEVANCE
    sort_order: VehicleSearchSortOrder = VehicleSearchSortOrder.DESC

    @field_validator("run_code", "plate", mode="before")
    @classmethod
    def _empty_to_none(cls, value: Any) -> Any:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("vehicle_class", mode="before")
    @classmethod
    def _normalize_vehicle_class(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        normalized = normalize_vehicle_class(text)
        if normalized == VehicleClass.UNKNOWN and text.upper() != VehicleClass.UNKNOWN.value:
            raise ValueError("Unsupported vehicle class.")
        return normalized.value

    @field_validator("colour", mode="before")
    @classmethod
    def _normalize_colour(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        normalized = normalize_vehicle_colour(text)
        if normalized == "UNKNOWN" and text.upper() != "UNKNOWN":
            raise ValueError(f"Unsupported colour. Expected one of: {', '.join(SUPPORTED_VEHICLE_COLOURS)}.")
        return normalized

    @field_validator("plate", mode="after")
    @classmethod
    def _normalize_plate(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_registration_text(value, country_profile="INDIA").cleaned_text
        return normalized or None

    @field_validator("camera_codes", mode="before")
    @classmethod
    def _parse_camera_codes(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            raw_values = value.split(",")
        elif isinstance(value, (list, tuple, set)):
            raw_values = list(value)
        else:
            raise ValueError("camera_codes must be a comma-separated string or list.")
        cleaned: list[str] = []
        for item in raw_values:
            text = str(item).strip().upper()
            if not text:
                continue
            if not _CAMERA_CODE_PATTERN.fullmatch(text):
                raise ValueError("Invalid camera code.")
            if text not in cleaned:
                cleaned.append(text)
        return tuple(cleaned)

    @model_validator(mode="after")
    def _validate_time_range(self) -> "VehicleSearchQuery":
        if self.start_time and self.end_time and self.start_time > self.end_time:
            raise ValueError("start_time must not be after end_time.")
        return self

    def resolved_window(self) -> tuple[str | None, str | None]:
        if self.date is None and self.start_time is None and self.end_time is None:
            return None, None
        requested_date = self.date or calendar_date.today()
        start_value = self.start_time or time.min
        end_value = self.end_time or time.max.replace(microsecond=0)
        start_at = datetime.combine(requested_date, start_value)
        end_at = datetime.combine(requested_date, end_value)
        return start_at.isoformat(), end_at.isoformat()

    def applied_filters(self) -> dict[str, Any]:
        filters: dict[str, Any] = {
            "run_code": self.run_code,
            "result_scope": self.result_scope,
            "vehicle_class": self.vehicle_class,
            "colour": self.colour,
            "plate": self.plate,
            "plate_match_type": self.plate_match_type if self.plate else None,
            "camera_codes": list(self.camera_codes) or None,
            "date": self.date.isoformat() if self.date else None,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "minimum_confidence": self.minimum_confidence,
            "multi_camera_only": self.multi_camera_only or None,
            "verified_plate_only": self.verified_plate_only or None,
            "sort_by": self.sort_by,
            "sort_order": self.sort_order,
        }
        return {key: value for key, value in filters.items() if value is not None}


class NaturalLanguageSearchRequest(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    query: str
    run_code: str | None = None
    result_scope: SearchResultScope | None = None
    default_time_tolerance_minutes: int = Field(default=15, ge=1, le=180)
    limit: int = Field(default=25, ge=1, le=100)
    offset: int = Field(default=0, ge=0)

    @field_validator("query", "run_code", mode="before")
    @classmethod
    def _strip_text_fields(cls, value: Any) -> Any:
        if value is None:
            return None
        return str(value).strip()

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str) -> str:
        if not value:
            raise ValueError("query must not be blank.")
        if len(value) < 2:
            raise ValueError("query must contain at least 2 characters.")
        if len(value) > 500:
            raise ValueError("query must not exceed 500 characters.")
        return value


class ParsedVehicleSearchIntent(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    vehicle_class: str | None = None
    colour: str | None = None
    plate: str | None = None
    plate_match_type: PlateMatchType | None = None
    camera_codes: list[str] = Field(default_factory=list)
    date: calendar_date | None = None
    start_time: time | None = None
    end_time: time | None = None
    target_time: time | None = None
    time_tolerance_minutes: int | None = Field(default=None, ge=1, le=180)
    minimum_confidence: float | None = Field(default=None, ge=0, le=1)
    multi_camera_only: bool | None = None
    verified_plate_only: bool | None = None
    result_scope: SearchResultScope | None = None
    sort_by: VehicleSearchSortBy | None = None
    sort_order: VehicleSearchSortOrder | None = None
    clarification_required: bool = False
    clarification_message: str | None = None

    @field_validator("vehicle_class", mode="before")
    @classmethod
    def _normalize_intent_vehicle_class(cls, value: Any) -> str | None:
        return VehicleSearchQuery._normalize_vehicle_class(value)

    @field_validator("colour", mode="before")
    @classmethod
    def _normalize_intent_colour(cls, value: Any) -> str | None:
        return VehicleSearchQuery._normalize_colour(value)

    @field_validator("plate", mode="before")
    @classmethod
    def _normalize_intent_plate_before(cls, value: Any) -> Any:
        return VehicleSearchQuery._empty_to_none(value)

    @field_validator("plate", mode="after")
    @classmethod
    def _normalize_intent_plate_after(cls, value: str | None) -> str | None:
        return VehicleSearchQuery._normalize_plate(value)

    @field_validator("camera_codes", mode="before")
    @classmethod
    def _parse_intent_camera_codes(cls, value: Any) -> list[str]:
        parsed = VehicleSearchQuery._parse_camera_codes(value)
        return list(parsed)

    @field_validator("clarification_message", mode="before")
    @classmethod
    def _strip_clarification_message(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @model_validator(mode="after")
    def _validate_intent(self) -> "ParsedVehicleSearchIntent":
        if self.start_time and self.end_time and self.start_time > self.end_time:
            raise ValueError("start_time must not be after end_time.")
        if self.clarification_required and not self.clarification_message:
            object.__setattr__(self, "clarification_message", "Please refine the natural-language search request.")
        return self

    def applied_filters(self) -> dict[str, Any]:
        payload = {
            "vehicle_class": self.vehicle_class,
            "colour": self.colour,
            "plate": self.plate,
            "plate_match_type": self.plate_match_type,
            "camera_codes": self.camera_codes or None,
            "date": self.date.isoformat() if self.date else None,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "target_time": self.target_time.isoformat() if self.target_time else None,
            "time_tolerance_minutes": self.time_tolerance_minutes,
            "minimum_confidence": self.minimum_confidence,
            "multi_camera_only": self.multi_camera_only,
            "verified_plate_only": self.verified_plate_only,
            "result_scope": self.result_scope,
            "sort_by": self.sort_by,
            "sort_order": self.sort_order,
            "clarification_required": self.clarification_required,
            "clarification_message": self.clarification_message,
        }
        return {key: value for key, value in payload.items() if value is not None}


class VehicleSearchResultType(str, Enum):
    LOCAL_TRACK = "LOCAL_TRACK"
    GLOBAL_VEHICLE = "GLOBAL_VEHICLE"


class VehicleSearchResultItem(BaseModel):
    result_type: VehicleSearchResultType
    global_vehicle_code: str | None = None
    track_uuid: str | None = None
    class_name: str | None = None
    colour: str | None = None
    plate_result: PlateResult | None = None
    plate: str | None = None
    plate_status: str | None = None
    camera_codes: list[str] = Field(default_factory=list)
    first_seen_at: str | None = None
    last_seen_at: str | None = None
    confidence: float | None = None
    member_track_count: int | None = None
    primary_media: MediaReference | None = None
    primary_vehicle_media: MediaReference | None = None
    primary_plate_media: MediaReference | None = None
    match_reasons: list[str] = Field(default_factory=list)
    relevance_score: float | None = None


class VehicleSearchPagination(BaseModel):
    limit: int
    offset: int
    returned: int
    total: int
    has_more: bool


class VehicleSearchResponse(BaseModel):
    filters: dict[str, Any]
    pagination: VehicleSearchPagination
    results: list[VehicleSearchResultItem]


class NaturalLanguageParserMetadata(BaseModel):
    provider: str
    model: str | None = None
    fallback_used: bool


class NaturalLanguageParseResponse(BaseModel):
    original_query: str
    parser: NaturalLanguageParserMetadata
    parsed_intent: ParsedVehicleSearchIntent
    interpreted_filters: dict[str, Any]
    clarification_required: bool
    clarification_message: str | None = None


class NaturalLanguageSearchResponse(BaseModel):
    original_query: str
    parser: NaturalLanguageParserMetadata
    clarification_required: bool
    clarification_message: str | None = None
    interpreted_filters: dict[str, Any]
    pagination: VehicleSearchPagination
    results: list[VehicleSearchResultItem]
