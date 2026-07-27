from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from ..dependencies import get_natural_language_search_service, get_vehicle_search_service, sanitize_payload
from ..search_models import (
    NaturalLanguageParseResponse,
    NaturalLanguageSearchRequest,
    NaturalLanguageSearchResponse,
    VehicleSearchQuery,
    VehicleSearchResponse,
    VehicleSearchResultItem,
)
from ..services.natural_language_search_service import NaturalLanguageSearchService
from ..services.vehicle_search_service import VehicleSearchService


router = APIRouter(prefix="/search", tags=["search"])


def _build_vehicle_search_query(
    run_code: str | None = Query(default=None),
    result_scope: str = Query(default="ALL"),
    vehicle_class: str | None = Query(default=None),
    colour: str | None = Query(default=None),
    plate: str | None = Query(default=None),
    plate_match_type: str = Query(default="CONTAINS"),
    camera_codes: str | None = Query(default=None),
    date: str | None = Query(default=None),
    start_time: str | None = Query(default=None),
    end_time: str | None = Query(default=None),
    minimum_confidence: float | None = Query(default=0.5, ge=0, le=1),
    multi_camera_only: bool = Query(default=False),
    verified_plate_only: bool = Query(default=False),
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    sort_by: str = Query(default="RELEVANCE"),
    sort_order: str = Query(default="DESC"),
) -> VehicleSearchQuery:
    try:
        return VehicleSearchQuery.model_validate(
            {
                "run_code": run_code,
                "result_scope": result_scope,
                "vehicle_class": vehicle_class,
                "colour": colour,
                "plate": plate,
                "plate_match_type": plate_match_type,
                "camera_codes": camera_codes,
                "date": date,
                "start_time": start_time,
                "end_time": end_time,
                "minimum_confidence": minimum_confidence,
                "multi_camera_only": multi_camera_only,
                "verified_plate_only": verified_plate_only,
                "limit": limit,
                "offset": offset,
                "sort_by": sort_by,
                "sort_order": sort_order,
            }
        )
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc


@router.get("/vehicles", response_model=VehicleSearchResponse)
def search_vehicles(
    query: VehicleSearchQuery = Depends(_build_vehicle_search_query),
    service: VehicleSearchService = Depends(get_vehicle_search_service),
) -> VehicleSearchResponse:
    payload = sanitize_payload(service.search(query))
    payload["results"] = [VehicleSearchResultItem.model_validate(item).model_dump() for item in payload.get("results", [])]
    return VehicleSearchResponse.model_validate(payload)


@router.post("/natural-language", response_model=NaturalLanguageSearchResponse)
def search_vehicles_natural_language(
    request: NaturalLanguageSearchRequest,
    service: NaturalLanguageSearchService = Depends(get_natural_language_search_service),
) -> NaturalLanguageSearchResponse:
    payload = sanitize_payload(service.search(request).model_dump())
    payload["results"] = [VehicleSearchResultItem.model_validate(item).model_dump() for item in payload.get("results", [])]
    return NaturalLanguageSearchResponse.model_validate(payload)


@router.post("/natural-language/parse", response_model=NaturalLanguageParseResponse)
def parse_vehicle_search_natural_language(
    request: NaturalLanguageSearchRequest,
    service: NaturalLanguageSearchService = Depends(get_natural_language_search_service),
) -> NaturalLanguageParseResponse:
    payload = sanitize_payload(service.parse_only(request).model_dump())
    return NaturalLanguageParseResponse.model_validate(payload)
