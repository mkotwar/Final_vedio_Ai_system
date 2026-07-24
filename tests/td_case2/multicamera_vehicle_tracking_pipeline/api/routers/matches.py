from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Query

from ..dependencies import get_match_service, sanitize_payload
from ..pagination import PaginatedResponse
from ..response_models import MatchDetailResponse, MatchListItem
from ..services.match_service import MatchService


router = APIRouter(prefix="/cross-camera-matches", tags=["cross-camera-matches"])


@router.get("", response_model=PaginatedResponse[MatchListItem])
def list_matches(
    run_code: str | None = None,
    decision: str | None = None,
    rule_version: str | None = None,
    minimum_score: float | None = Query(default=None, ge=0, le=1),
    camera_code: str | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
    sort_by: Literal["updated_at", "overall_score", "decision"] = "updated_at",
    sort_order: Literal["asc", "desc"] = "desc",
    service: MatchService = Depends(get_match_service),
) -> PaginatedResponse[MatchListItem]:
    page_result = service.list_matches(
        run_code=run_code,
        decision=decision,
        rule_version=rule_version,
        minimum_score=minimum_score,
        camera_code=camera_code,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    return PaginatedResponse[MatchListItem](
        items=[MatchListItem.model_validate(sanitize_payload(item)) for item in page_result.items],
        page=page_result.page,
        page_size=page_result.page_size,
        total=page_result.total,
        has_next=page_result.has_next,
    )


@router.get("/{match_id}", response_model=MatchDetailResponse)
def get_match(match_id: str, service: MatchService = Depends(get_match_service)) -> MatchDetailResponse:
    return MatchDetailResponse.model_validate(sanitize_payload(service.get_match(match_id)))
