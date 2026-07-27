from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from ..dependencies import get_media_service, sanitize_payload
from ..response_models import MediaDeliveryResponse, MediaSignedUrlResponse
from ..services.media_service import MediaService


router = APIRouter(prefix="/media", tags=["media"])


@router.get("/{media_id}", response_model=MediaDeliveryResponse)
def get_media(media_id: str, service: MediaService = Depends(get_media_service)) -> MediaDeliveryResponse:
    return MediaDeliveryResponse.model_validate(sanitize_payload(service.get_media_reference(media_id)))


@router.get("/{media_id}/content")
def get_media_content(media_id: str, service: MediaService = Depends(get_media_service)) -> FileResponse:
    return service.get_media_content_response(media_id)


@router.get("/{media_id}/url", response_model=MediaSignedUrlResponse)
def get_media_url(media_id: str, service: MediaService = Depends(get_media_service)) -> MediaSignedUrlResponse:
    return MediaSignedUrlResponse.model_validate(sanitize_payload(service.get_signed_url(media_id)))
