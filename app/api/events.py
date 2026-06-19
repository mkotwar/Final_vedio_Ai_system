from fastapi import APIRouter, Path, HTTPException
from fastapi.responses import FileResponse
from loguru import logger

from app.core.config import settings

router = APIRouter(prefix="/api/v1/events", tags=["Events"])

@router.get("/{video_id}/thumbnail/{frame_id}", summary="Get Event Thumbnail")
async def get_event_thumbnail(
    video_id: str = Path(..., description="The unique UUID of the video"),
    frame_id: str = Path(..., description="The frame ID of the event thumbnail"),
):
    """Serve the JPEG thumbnail for a specific event."""
    frame_path = settings.FRAMES_DIR / video_id / f"{frame_id}.jpg"
    if not frame_path.exists():
        logger.warning(f"Thumbnail not found: {frame_path}")
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    
    return FileResponse(path=frame_path, media_type="image/jpeg")
