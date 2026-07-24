from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Query

from ..dependencies import get_camera_service, sanitize_payload
from ..pagination import PaginatedResponse
from ..response_models import CameraDetailResponse, CameraListItem
from ..services.camera_service import CameraService


router = APIRouter(prefix="/runs/{run_code}/cameras", tags=["cameras"])


@router.get("", response_model=PaginatedResponse[CameraListItem])
def list_cameras(
    run_code: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
    status: str | None = None,
    camera_code: str | None = None,
    sort_by: Literal["camera_code", "camera_name", "frames_processed", "completed_track_count"] = "camera_code",
    sort_order: Literal["asc", "desc"] = "asc",
    service: CameraService = Depends(get_camera_service),
) -> PaginatedResponse[CameraListItem]:
    page_result = service.list_cameras(
        run_code,
        page=page,
        page_size=page_size,
        status=status,
        camera_code=camera_code,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    return PaginatedResponse[CameraListItem](
        items=[CameraListItem.model_validate(sanitize_payload(item)) for item in page_result.items],
        page=page_result.page,
        page_size=page_result.page_size,
        total=page_result.total,
        has_next=page_result.has_next,
    )


@router.get("/{camera_code}", response_model=CameraDetailResponse)
def get_camera(run_code: str, camera_code: str, service: CameraService = Depends(get_camera_service)) -> CameraDetailResponse:
    return CameraDetailResponse.model_validate(sanitize_payload(service.get_camera(run_code, camera_code)))
