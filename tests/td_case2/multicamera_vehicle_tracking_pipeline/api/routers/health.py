from __future__ import annotations

from fastapi import APIRouter, Depends

from ..dependencies import get_repository, sanitize_payload
from ..response_models import HealthResponse
from ...persistence.api_read_repository import AnalyticsReadRepository


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(repository: AnalyticsReadRepository = Depends(get_repository)) -> HealthResponse:
    return HealthResponse.model_validate(sanitize_payload(repository.health()))
