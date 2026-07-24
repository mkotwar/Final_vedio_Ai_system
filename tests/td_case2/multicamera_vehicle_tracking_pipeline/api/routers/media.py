from __future__ import annotations

from fastapi import APIRouter, Depends

from ..dependencies import get_media_service, sanitize_payload
from ..response_models import MediaDeliveryResponse
from ..services.media_service import MediaService


router = APIRouter(prefix="/media", tags=["media"])


@router.get("/{media_id}", response_model=MediaDeliveryResponse)
def get_media(media_id: str, service: MediaService = Depends(get_media_service)) -> MediaDeliveryResponse:
    return MediaDeliveryResponse.model_validate(sanitize_payload(service.get_media_reference(media_id)))
