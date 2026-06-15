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
    # The frame_id usually looks like <video_id>_f0001, but the actual file is frame_0001.jpg
    try:
        frame_idx_str = frame_id.split("_f")[-1]
        frame_filename = f"frame_{frame_idx_str}.jpg"
    except Exception:
        frame_filename = f"{frame_id}.jpg"
        
    frame_path = settings.FRAMES_DIR / video_id / frame_filename
    if not frame_path.exists():
        logger.warning(f"Thumbnail not found: {frame_path}")
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    
    return FileResponse(path=frame_path, media_type="image/jpeg")
