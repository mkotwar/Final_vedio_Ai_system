from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Query

from ..dependencies import get_global_vehicle_service, sanitize_payload
from ..pagination import PaginatedResponse
from ..response_models import GlobalVehicleDetailResponse, GlobalVehicleListItem, TrackListItem
from ..services.global_vehicle_service import GlobalVehicleService


router = APIRouter(prefix="/global-vehicles", tags=["global-vehicles"])


@router.get("", response_model=PaginatedResponse[GlobalVehicleListItem])
def list_global_vehicles(
    run_code: str | None = None,
    status: str | None = None,
    vehicle_class: str | None = None,
    colour: str | None = None,
    plate: str | None = None,
    minimum_confidence: float | None = Query(default=None, ge=0, le=1),
    minimum_camera_count: int | None = Query(default=None, ge=1),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
    sort_by: Literal["first_seen_at", "last_seen_at", "camera_count", "track_count", "identity_confidence"] = "first_seen_at",
    sort_order: Literal["asc", "desc"] = "desc",
    service: GlobalVehicleService = Depends(get_global_vehicle_service),
) -> PaginatedResponse[GlobalVehicleListItem]:
    repository_sort = "identity_confidence" if sort_by == "identity_confidence" else sort_by
    page_result = service.list_global_vehicles(
        run_code=run_code,
        status=status,
        vehicle_class=vehicle_class,
        colour=colour,
        plate=plate,
        minimum_confidence=minimum_confidence,
        minimum_camera_count=minimum_camera_count,
        page=page,
        page_size=page_size,
        sort_by=repository_sort,
        sort_order=sort_order,
    )
    return PaginatedResponse[GlobalVehicleListItem](
        items=[GlobalVehicleListItem.model_validate(sanitize_payload(item)) for item in page_result.items],
        page=page_result.page,
        page_size=page_result.page_size,
        total=page_result.total,
        has_next=page_result.has_next,
    )


@router.get("/{global_vehicle_code}", response_model=GlobalVehicleDetailResponse)
def get_global_vehicle(
    global_vehicle_code: str,
    service: GlobalVehicleService = Depends(get_global_vehicle_service),
) -> GlobalVehicleDetailResponse:
    return GlobalVehicleDetailResponse.model_validate(sanitize_payload(service.get_global_vehicle(global_vehicle_code)))


@router.get("/{global_vehicle_code}/tracks")
def list_global_vehicle_tracks(
    global_vehicle_code: str,
    service: GlobalVehicleService = Depends(get_global_vehicle_service),
) -> list[dict]:
    return sanitize_payload(service.list_member_tracks(global_vehicle_code))
