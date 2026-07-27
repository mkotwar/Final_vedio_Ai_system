from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Query

from ..dependencies import get_media_service, get_track_service, sanitize_payload
from ..pagination import PaginatedResponse
from ..response_models import MediaDeliveryResponse, MediaReference, ObservationItem, TrackDetailResponse, TrackListItem
from ..services.media_service import MediaService
from ..services.track_service import TrackService


router = APIRouter(tags=["tracks"])


@router.get("/runs/{run_code}/tracks", response_model=PaginatedResponse[TrackListItem])
def list_tracks(
    run_code: str,
    camera_code: str | None = None,
    vehicle_class: str | None = None,
    colour: str | None = None,
    plate: str | None = None,
    plate_status: str | None = None,
    lifecycle_state: str | None = None,
    minimum_confidence: float | None = Query(default=0.5, ge=0, le=1),
    has_media: bool | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
    sort_by: Literal["first_seen_at", "last_seen_at", "observation_count", "best_detection_confidence", "track_uuid"] = "first_seen_at",
    sort_order: Literal["asc", "desc"] = "desc",
    service: TrackService = Depends(get_track_service),
) -> PaginatedResponse[TrackListItem]:
    page_result = service.list_tracks(
        run_code,
        camera_code=camera_code,
        vehicle_class=vehicle_class,
        colour=colour,
        plate=plate,
        plate_status=plate_status,
        lifecycle_state=lifecycle_state,
        minimum_confidence=minimum_confidence,
        has_media=has_media,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    return PaginatedResponse[TrackListItem](
        items=[TrackListItem.model_validate(sanitize_payload(item)) for item in page_result.items],
        page=page_result.page,
        page_size=page_result.page_size,
        total=page_result.total,
        has_next=page_result.has_next,
    )


@router.get("/tracks/{track_uuid}", response_model=TrackDetailResponse)
def get_track(track_uuid: str, service: TrackService = Depends(get_track_service)) -> TrackDetailResponse:
    return TrackDetailResponse.model_validate(sanitize_payload(service.get_track(track_uuid)))


@router.get("/tracks/{track_uuid}/observations", response_model=PaginatedResponse[ObservationItem])
def list_observations(
    track_uuid: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
    key_only: bool = False,
    start_frame: int | None = Query(default=None, ge=0),
    end_frame: int | None = Query(default=None, ge=0),
    sort_order: Literal["asc", "desc"] = "asc",
    service: TrackService = Depends(get_track_service),
) -> PaginatedResponse[ObservationItem]:
    page_result = service.list_observations(
        track_uuid,
        page=page,
        page_size=page_size,
        key_only=key_only,
        start_frame=start_frame,
        end_frame=end_frame,
        sort_order=sort_order,
    )
    return PaginatedResponse[ObservationItem](
        items=[ObservationItem.model_validate(sanitize_payload(item)) for item in page_result.items],
        page=page_result.page,
        page_size=page_result.page_size,
        total=page_result.total,
        has_next=page_result.has_next,
    )


@router.get("/tracks/{track_uuid}/media", response_model=list[MediaReference])
def list_track_media(track_uuid: str, service: MediaService = Depends(get_media_service)) -> list[MediaReference]:
    return [MediaReference.model_validate(sanitize_payload(item)) for item in service.list_media(track_uuid)]
