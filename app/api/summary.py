from fastapi import APIRouter, Path, HTTPException
from loguru import logger

from app.schemas.summary import SummaryResponse
from app.services.summary_service import SummaryService

router = APIRouter(prefix="/api/v1/videos", tags=["Summary"])


@router.get("/{video_id}/summary", response_model=SummaryResponse)
async def get_video_summary(
    video_id: str = Path(..., description="The unique UUID of the video"),
) -> SummaryResponse:
    """
    Retrieve the event-centric deterministic summary of a video.
    
    This endpoint loads pre-aggregated high-level events, computes statistics, 
    detects peak periods, extracts notable anomalies, and builds a chronological timeline.
    If no events are found, it gracefully returns an empty summary.
    """
    try:
        logger.info(f"Generating summary for video_id: {video_id}")
        # Generating summary is currently deterministic and synchronous.
        # Can easily be made async if it starts relying on external IO heavily.
        return SummaryService.generate_summary(video_id)
    except Exception as e:
        logger.error(f"Error generating summary for video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during summary generation")

@router.get("/{video_id}/report")
async def get_investigation_report(
    video_id: str = Path(..., description="The unique UUID of the video"),
):
    """
    Generate an export-ready, structured Investigation Report.
    """
    from app.services.report_service import ReportService
    from app.schemas.report import InvestigationReport
    try:
        logger.info(f"Generating investigation report for video_id: {video_id}")
        return ReportService.generate_report(video_id)
    except Exception as e:
        logger.error(f"Error generating report for video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during report generation")
