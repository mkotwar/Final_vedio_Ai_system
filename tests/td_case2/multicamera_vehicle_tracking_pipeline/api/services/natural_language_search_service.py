from __future__ import annotations

import json
import logging
import re
import time as time_module
from dataclasses import dataclass
from datetime import date as calendar_date, datetime, time, timedelta
from typing import Any, Protocol

from ..errors import ApiError
from ..search_models import (
    NaturalLanguageParseResponse,
    NaturalLanguageParserMetadata,
    NaturalLanguageSearchRequest,
    NaturalLanguageSearchResponse,
    ParsedVehicleSearchIntent,
    PlateMatchType,
    SearchResultScope,
    VehicleSearchPagination,
    VehicleSearchQuery,
)
from ..settings import ApiSettings
from .vehicle_search_service import VehicleSearchService


LOGGER = logging.getLogger("multicamera_vehicle_api")
SAFE_CAMERA_CODE_PATTERN = re.compile(r"\bCAM[_:-]?[A-Z0-9]+\b", re.IGNORECASE)
PLATE_TOKEN_PATTERN = re.compile(r"\b[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{3,4}\b", re.IGNORECASE)
TIME_FRAGMENT_PATTERN = r"\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?"

CLASS_ALIASES: dict[str, str] = {
    "3 wheeler": "3WHEELER",
    "3-wheeler": "3WHEELER",
    "3wheeler": "3WHEELER",
    "auto": "3WHEELER",
    "autorickshaw": "3WHEELER",
    "bus": "BUS",
    "buses": "BUS",
    "car": "CAR",
    "cars": "CAR",
    "motorbike": "MOTORCYCLE",
    "motorbike ": "MOTORCYCLE",
    "motorcycle": "MOTORCYCLE",
    "motorcycles": "MOTORCYCLE",
    "bike": "MOTORCYCLE",
    "bikes": "MOTORCYCLE",
    "truck": "TRUCK",
    "trucks": "TRUCK",
}
COLOUR_ALIASES: dict[str, str] = {
    "black": "BLACK",
    "white": "WHITE",
    "silver": "SILVER",
    "grey": "GREY",
    "gray": "GREY",
    "red": "RED",
    "blue": "BLUE",
    "green": "GREEN",
    "yellow": "YELLOW",
    "orange": "ORANGE",
    "brown": "BROWN",
    "beige": "BEIGE",
    "purple": "PURPLE",
}


@dataclass(frozen=True, slots=True)
class NaturalLanguageSearchContext:
    run_code: str | None
    result_scope: SearchResultScope | None
    default_time_tolerance_minutes: int
    available_camera_codes: tuple[str, ...] = ()


class NaturalLanguageParserProvider(Protocol):
    provider_name: str
    model_name: str | None

    def parse_vehicle_search(
        self,
        query: str,
        context: NaturalLanguageSearchContext,
    ) -> dict[str, Any]:
        ...


class ProviderUnavailableError(RuntimeError):
    pass


class GeminiNaturalLanguageParserProvider:
    provider_name = "gemini"

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str,
        timeout_seconds: int,
        max_retries: int,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def parse_vehicle_search(
        self,
        query: str,
        context: NaturalLanguageSearchContext,
    ) -> dict[str, Any]:
        if not self.api_key.strip():
            raise ProviderUnavailableError("Gemini API key is missing.")
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional dependency
            raise ProviderUnavailableError("The google-genai package is not installed.") from exc

        client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(timeout=max(1, self.timeout_seconds) * 1000),
        )
        prompt = _build_parser_prompt(query, context)
        attempts = max(1, int(self.max_retries) + 1)
        last_error: Exception | None = None
        for attempt in range(attempts):
            started = time_module.perf_counter()
            try:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        response_mime_type="application/json",
                    ),
                )
                text = str(getattr(response, "text", "") or "").strip()
                if not text:
                    raise ValueError("Gemini returned an empty response.")
                return json.loads(text)
            except Exception as exc:  # pragma: no cover - exercised with fakes
                last_error = exc
                LOGGER.warning(
                    "Natural-language Gemini parse failed on attempt %s after %.3fs: %s",
                    attempt + 1,
                    time_module.perf_counter() - started,
                    _sanitize_provider_error(str(exc), self.api_key),
                )
                if attempt + 1 >= attempts:
                    break
        raise ProviderUnavailableError("Gemini natural-language parsing failed.") from last_error


class NaturalLanguageSearchService:
    def __init__(
        self,
        repository: Any,
        vehicle_search_service: VehicleSearchService,
        *,
        settings: ApiSettings,
        parser_provider: NaturalLanguageParserProvider | None = None,
    ) -> None:
        self.repository = repository
        self.vehicle_search_service = vehicle_search_service
        self.settings = settings
        self.parser_provider = parser_provider

    def parse_only(self, request: NaturalLanguageSearchRequest) -> NaturalLanguageParseResponse:
        self._ensure_feature_enabled()
        parsed_intent, metadata = self.parse_query(request)
        merged_intent = self.merge_context(request, parsed_intent)
        return NaturalLanguageParseResponse(
            original_query=request.query,
            parser=metadata,
            parsed_intent=merged_intent,
            interpreted_filters=self._intent_to_filters(request, merged_intent),
            clarification_required=merged_intent.clarification_required,
            clarification_message=merged_intent.clarification_message,
        )

    def search(self, request: NaturalLanguageSearchRequest) -> NaturalLanguageSearchResponse:
        self._ensure_feature_enabled()
        parsed_intent, metadata = self.parse_query(request)
        merged_intent = self.merge_context(request, parsed_intent)

        if merged_intent.clarification_required:
            return NaturalLanguageSearchResponse(
                original_query=request.query,
                parser=metadata,
                clarification_required=True,
                clarification_message=merged_intent.clarification_message,
                interpreted_filters=self._intent_to_filters(request, merged_intent),
                pagination=VehicleSearchPagination(limit=request.limit, offset=request.offset, returned=0, total=0, has_more=False),
                results=[],
            )

        structured_query = self.build_structured_query(request, merged_intent)
        response_payload = self.vehicle_search_service.search(structured_query)
        return NaturalLanguageSearchResponse(
            original_query=request.query,
            parser=metadata,
            clarification_required=False,
            clarification_message=None,
            interpreted_filters=structured_query.applied_filters(),
            pagination=VehicleSearchPagination.model_validate(response_payload["pagination"]),
            results=response_payload["results"],
        )

    def parse_query(
        self,
        request: NaturalLanguageSearchRequest,
    ) -> tuple[ParsedVehicleSearchIntent, NaturalLanguageParserMetadata]:
        context = NaturalLanguageSearchContext(
            run_code=request.run_code,
            result_scope=request.result_scope,
            default_time_tolerance_minutes=request.default_time_tolerance_minutes,
        )
        fallback_intent = self._fallback_parse(request.query, context)
        if self.parser_provider is None:
            return fallback_intent, NaturalLanguageParserMetadata(provider="deterministic_fallback", model=None, fallback_used=True)

        try:
            raw_payload = self.parser_provider.parse_vehicle_search(request.query, context)
            parsed = ParsedVehicleSearchIntent.model_validate(raw_payload)
            return parsed, NaturalLanguageParserMetadata(
                provider=self.parser_provider.provider_name,
                model=self.parser_provider.model_name,
                fallback_used=False,
            )
        except Exception as exc:
            LOGGER.warning("Natural-language provider output rejected; using fallback parser: %s", type(exc).__name__)
            return fallback_intent, NaturalLanguageParserMetadata(
                provider=getattr(self.parser_provider, "provider_name", "provider"),
                model=getattr(self.parser_provider, "model_name", None),
                fallback_used=True,
            )

    def merge_context(
        self,
        request: NaturalLanguageSearchRequest,
        parsed_intent: ParsedVehicleSearchIntent,
    ) -> ParsedVehicleSearchIntent:
        merged = parsed_intent.model_copy(deep=True)
        if request.result_scope is not None:
            merged.result_scope = request.result_scope
        if merged.target_time and merged.time_tolerance_minutes is None:
            merged.time_tolerance_minutes = request.default_time_tolerance_minutes
        if merged.sort_by is None:
            merged.sort_by = None
        if merged.sort_order is None:
            merged.sort_order = None
        if request.run_code is None and (merged.target_time or merged.start_time or merged.end_time):
            merged.clarification_required = True
            merged.clarification_message = "Please choose a processing run before using time-based natural-language search."
        if request.run_code is None and merged.camera_codes:
            merged.clarification_required = True
            merged.clarification_message = "Please choose a processing run before filtering by camera."
        return ParsedVehicleSearchIntent.model_validate(merged.model_dump())

    def build_structured_query(
        self,
        request: NaturalLanguageSearchRequest,
        merged_intent: ParsedVehicleSearchIntent,
    ) -> VehicleSearchQuery:
        if not request.run_code:
            raise ApiError(
                code="NATURAL_LANGUAGE_RUN_REQUIRED",
                message="Natural-language search requires a selected processing run.",
                status_code=400,
            )

        validated_cameras = self._validate_camera_codes_for_run(request.run_code, merged_intent.camera_codes)
        resolved_date = merged_intent.date
        if resolved_date is None and (merged_intent.target_time or merged_intent.start_time or merged_intent.end_time):
            resolved_date = self._resolve_run_date(request.run_code)
            if resolved_date is None:
                raise ApiError(
                    code="NATURAL_LANGUAGE_RUN_DATE_UNAVAILABLE",
                    message="The selected run does not expose a usable date for time-based natural-language search.",
                    status_code=400,
                )

        normalized_intent = self.normalize_intent(merged_intent, resolved_date=resolved_date)
        return VehicleSearchQuery(
            run_code=request.run_code,
            result_scope=request.result_scope or normalized_intent.result_scope or SearchResultScope.ALL,
            vehicle_class=normalized_intent.vehicle_class,
            colour=normalized_intent.colour,
            plate=normalized_intent.plate,
            plate_match_type=normalized_intent.plate_match_type or PlateMatchType.CONTAINS,
            camera_codes=validated_cameras,
            date=normalized_intent.date,
            start_time=normalized_intent.start_time,
            end_time=normalized_intent.end_time,
            minimum_confidence=normalized_intent.minimum_confidence if normalized_intent.minimum_confidence is not None else 0.5,
            multi_camera_only=bool(normalized_intent.multi_camera_only),
            verified_plate_only=bool(normalized_intent.verified_plate_only),
            limit=request.limit,
            offset=request.offset,
            sort_by=normalized_intent.sort_by or "RELEVANCE",
            sort_order=normalized_intent.sort_order or "DESC",
        )

    def normalize_intent(
        self,
        merged_intent: ParsedVehicleSearchIntent,
        *,
        resolved_date: calendar_date | None,
    ) -> ParsedVehicleSearchIntent:
        payload = merged_intent.model_dump()
        payload["date"] = resolved_date
        if merged_intent.target_time is not None:
            tolerance_minutes = merged_intent.time_tolerance_minutes or 15
            window_start, window_end = _expand_target_time(merged_intent.target_time, tolerance_minutes)
            payload["start_time"] = window_start
            payload["end_time"] = window_end
        payload["target_time"] = merged_intent.target_time
        payload["time_tolerance_minutes"] = merged_intent.time_tolerance_minutes
        return ParsedVehicleSearchIntent.model_validate(payload)

    def _ensure_feature_enabled(self) -> None:
        if not self.settings.natural_language_search_enabled:
            raise ApiError(
                code="NATURAL_LANGUAGE_SEARCH_DISABLED",
                message="Natural-language search is not enabled.",
                status_code=503,
            )

    def _validate_camera_codes_for_run(self, run_code: str, camera_codes: list[str]) -> list[str]:
        if not camera_codes:
            return []
        _, page = self.repository.list_run_cameras(run_code=run_code, page=1, page_size=500)
        valid_codes = {
            str(item.get("camera_code")).upper()
            for item in getattr(page, "items", [])
            if item.get("camera_code")
        }
        unknown = [code for code in camera_codes if code.upper() not in valid_codes]
        if unknown:
            raise ApiError(
                code="UNKNOWN_CAMERA_CODE",
                message="One or more camera codes are not available for the selected run.",
                status_code=400,
                details={"camera_codes": unknown},
            )
        return camera_codes

    def _resolve_run_date(self, run_code: str) -> calendar_date | None:
        run = self.repository.find_run_by_code(run_code)
        if not run:
            raise ApiError(code="RUN_NOT_FOUND", message="Processing run was not found.", status_code=404)
        for field_name in ("started_at", "created_at", "completed_at"):
            value = run.get(field_name)
            if not value:
                continue
            try:
                return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
            except ValueError:
                continue
        return None

    def _fallback_parse(
        self,
        query: str,
        context: NaturalLanguageSearchContext,
    ) -> ParsedVehicleSearchIntent:
        lower = query.strip().lower()
        payload: dict[str, Any] = {
            "camera_codes": _extract_camera_codes(query),
        }

        plate_ending = re.search(r"\bplate\s+(?:ending|ends)\s+in\s+([a-z0-9]+)\b", lower)
        plate_start = re.search(r"\bplate\s+starts?\s+with\s+([a-z0-9]+)\b", lower)
        plate_contains = re.search(r"\bplate\s+contains?\s+([a-z0-9]+)\b", lower)
        exact_plate = PLATE_TOKEN_PATTERN.search(query.upper())

        if plate_ending:
            payload["plate"] = plate_ending.group(1)
            payload["plate_match_type"] = PlateMatchType.ENDS_WITH
        elif plate_start:
            payload["plate"] = plate_start.group(1)
            payload["plate_match_type"] = PlateMatchType.STARTS_WITH
        elif plate_contains:
            payload["plate"] = plate_contains.group(1)
            payload["plate_match_type"] = PlateMatchType.CONTAINS
        elif exact_plate:
            payload["plate"] = exact_plate.group(0)
            payload["plate_match_type"] = PlateMatchType.EXACT

        for phrase, canonical in CLASS_ALIASES.items():
            if re.search(rf"\b{re.escape(phrase)}\b", lower):
                payload["vehicle_class"] = canonical
                break

        for phrase, canonical in COLOUR_ALIASES.items():
            if re.search(rf"\b{re.escape(phrase)}\b", lower):
                payload["colour"] = canonical
                break

        if "verified plate" in lower or "verified plates" in lower:
            payload["verified_plate_only"] = True

        if "both cameras" in lower or "multiple cameras" in lower or "same vehicle across" in lower or "seen in cam_" in lower:
            payload["multi_camera_only"] = True
        if "same vehicle across" in lower or "seen in both cameras" in lower or ("same vehicle" in lower and len(payload["camera_codes"]) >= 2):
            payload["result_scope"] = SearchResultScope.GLOBAL_VEHICLES

        between = re.search(rf"\bbetween\s+({TIME_FRAGMENT_PATTERN})\s+and\s+({TIME_FRAGMENT_PATTERN})\b", lower, re.IGNORECASE)
        around = re.search(rf"\baround\s+({TIME_FRAGMENT_PATTERN})\b", lower, re.IGNORECASE)
        after = re.search(rf"\bafter\s+({TIME_FRAGMENT_PATTERN})\b", lower, re.IGNORECASE)
        before = re.search(rf"\bbefore\s+({TIME_FRAGMENT_PATTERN})\b", lower, re.IGNORECASE)

        if between:
            payload["start_time"] = _parse_time_fragment(between.group(1))
            payload["end_time"] = _parse_time_fragment(between.group(2), meridiem_hint=_extract_meridiem(between.group(1)))
        elif around:
            payload["target_time"] = _parse_time_fragment(around.group(1))
            payload["time_tolerance_minutes"] = context.default_time_tolerance_minutes
        elif after:
            payload["start_time"] = _parse_time_fragment(after.group(1))
        elif before:
            payload["end_time"] = _parse_time_fragment(before.group(1))

        return ParsedVehicleSearchIntent.model_validate(payload)

    def _intent_to_filters(
        self,
        request: NaturalLanguageSearchRequest,
        intent: ParsedVehicleSearchIntent,
    ) -> dict[str, Any]:
        filters = intent.applied_filters()
        if request.run_code:
            filters["run_code"] = request.run_code
        if request.result_scope:
            filters["result_scope"] = request.result_scope
        filters["limit"] = request.limit
        filters["offset"] = request.offset
        return filters


def _build_parser_prompt(query: str, context: NaturalLanguageSearchContext) -> str:
    camera_codes = ", ".join(context.available_camera_codes) if context.available_camera_codes else "Only camera codes explicitly named by the operator."
    return (
        "You convert natural-language vehicle search requests into strict JSON only.\n"
        "Return one JSON object with only these keys: "
        "vehicle_class, colour, plate, plate_match_type, camera_codes, date, start_time, end_time, target_time, "
        "time_tolerance_minutes, minimum_confidence, multi_camera_only, verified_plate_only, result_scope, sort_by, sort_order, "
        "clarification_required, clarification_message.\n"
        "Never return SQL, table names, code, markdown, or explanations.\n"
        f"Supported result_scope values: {[item.value for item in SearchResultScope]}.\n"
        f"Supported plate_match_type values: {[item.value for item in PlateMatchType]}.\n"
        "Supported vehicle_class values: ['3WHEELER', 'BUS', 'CAR', 'MOTORCYCLE', 'TRUCK', 'UNKNOWN'].\n"
        "Supported colour values: ['BLACK', 'WHITE', 'SILVER', 'GREY', 'RED', 'BLUE', 'GREEN', 'YELLOW', 'ORANGE', 'BROWN', 'BEIGE', 'PURPLE', 'UNKNOWN'].\n"
        "Interpret 'ending in 6268' as plate='6268' and plate_match_type='ENDS_WITH'.\n"
        "Interpret a full plate as plate_match_type='EXACT'.\n"
        "Interpret 'both cameras' as multi_camera_only=true.\n"
        "Interpret 'verified plates' as verified_plate_only=true.\n"
        "Interpret 'same vehicle across cameras' as result_scope='GLOBAL_VEHICLES' and multi_camera_only=true.\n"
        f"Default time tolerance minutes: {context.default_time_tolerance_minutes}.\n"
        f"Explicit run_code context: {context.run_code or 'NONE'}.\n"
        f"Explicit result_scope context: {context.result_scope.value if context.result_scope else 'NONE'}.\n"
        f"Available camera context: {camera_codes}.\n"
        "If essential context is missing, set clarification_required=true and provide a short clarification_message.\n"
        f"Operator query: {query}"
    )


def _sanitize_provider_error(message: str, api_key: str | None) -> str:
    sanitized = str(message or "")
    if api_key:
        sanitized = sanitized.replace(api_key, "[REDACTED_API_KEY]")
    return sanitized[:400]


def _extract_camera_codes(query: str) -> list[str]:
    seen: list[str] = []
    for match in SAFE_CAMERA_CODE_PATTERN.findall(query.upper()):
        normalized = match.replace("-", "_")
        if normalized not in seen:
            seen.append(normalized)
    return seen


def _extract_meridiem(fragment: str) -> str | None:
    normalized = fragment.lower().replace(".", "").strip()
    if normalized.endswith("am"):
        return "am"
    if normalized.endswith("pm"):
        return "pm"
    return None


def _parse_time_fragment(fragment: str, meridiem_hint: str | None = None) -> time:
    cleaned = re.sub(r"\s+", " ", fragment.strip().lower().replace(".", ""))
    meridiem = _extract_meridiem(cleaned) or meridiem_hint
    numbers = cleaned.replace("am", "").replace("pm", "").strip()
    if ":" in numbers:
        hour_text, minute_text = numbers.split(":", 1)
        hour = int(hour_text)
        minute = int(minute_text)
    else:
        hour = int(numbers)
        minute = 0
    if meridiem == "pm" and hour < 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        raise ValueError("Unsupported time fragment.")
    return time(hour=hour, minute=minute)


def _expand_target_time(target_time: time, tolerance_minutes: int) -> tuple[time, time]:
    anchor = datetime.combine(calendar_date(2026, 1, 1), target_time)
    start = max(anchor - timedelta(minutes=tolerance_minutes), datetime.combine(anchor.date(), time.min))
    end = min(anchor + timedelta(minutes=tolerance_minutes), datetime.combine(anchor.date(), time.max.replace(microsecond=0)))
    return start.time(), end.time()
