from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Query

from ..dependencies import sanitize_payload
from ..pagination import PaginatedResponse
from ..response_models import RunDetailResponse, RunListItem
from ..services.run_service import RunService
from ..dependencies import get_run_service


router = APIRouter(prefix="/runs", tags=["runs"])


@router.get("", response_model=PaginatedResponse[RunListItem])
def list_runs(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
    status: str | None = None,
    run_code: str | None = None,
    sort_by: Literal["created_at", "started_at", "completed_at", "run_code", "status"] = "created_at",
    sort_order: Literal["asc", "desc"] = "desc",
    service: RunService = Depends(get_run_service),
) -> PaginatedResponse[RunListItem]:
    page_result = service.list_runs(
        page=page,
        page_size=page_size,
        status=status,
        run_code=run_code,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    return PaginatedResponse[RunListItem](
        items=[RunListItem.model_validate(sanitize_payload(item)) for item in page_result.items],
        page=page_result.page,
        page_size=page_result.page_size,
        total=page_result.total,
        has_next=page_result.has_next,
    )


@router.get("/{run_code}", response_model=RunDetailResponse)
def get_run(run_code: str, service: RunService = Depends(get_run_service)) -> RunDetailResponse:
    return RunDetailResponse.model_validate(sanitize_payload(service.get_run(run_code)))
