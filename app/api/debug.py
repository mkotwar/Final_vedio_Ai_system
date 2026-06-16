import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from loguru import logger
from app.core.config import settings

router = APIRouter(prefix="/api/v1/debug", tags=["debug"])

@router.get("/sampling/{video_id}")
async def get_sampling_metrics(video_id: str):
    """
    Retrieves the saved SamplingMetrics for a specific video.
    Useful for historical comparison and analytics dashboards.
    """
    sampling_path = settings.METADATA_DIR / f"{video_id}_sampling.json"
    
    if not sampling_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Sampling telemetry not found for video {video_id}. It may not have been processed or sampling was disabled."
        )
        
    try:
        with open(sampling_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as exc:
        logger.error(f"Failed to read sampling telemetry for {video_id}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to read telemetry data.")
