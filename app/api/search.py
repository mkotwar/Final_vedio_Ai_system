"""API router for Semantic Event Search.
"""

from fastapi import APIRouter, HTTPException
from loguru import logger

from app.schemas.search import SearchRequest, SearchResponse
from app.services.search_service import SearchService


router = APIRouter(prefix="/api/v1/search", tags=["Search"])

@router.get("/reprocess_test_video")
def reprocess_test_video():
    import json
    from app.services.event_aggregation import EventAggregationService
    from app.services.search_service import SearchService
    from app.core.config import settings
    from app.services.pipeline_contract import frame_catalog_path
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    
    try:
        video_id = "284e527c-888c-4c80-96c8-3cd7d50731b3"
        frames_path = frame_catalog_path(video_id)
        
        with open(frames_path, "r", encoding="utf-8") as f:
            frames = json.load(f)
            
        new_events = EventAggregationService.process_events(video_id, frames)
        
        client = SearchService.get_client()
        client.delete(
            collection_name=settings.QDRANT_COLLECTION,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="video_id",
                        match=MatchValue(value=video_id)
                    )
                ]
            )
        )
        
        SearchService.index_events(video_id, new_events)
        
        return {"status": "success", "events_generated": len(new_events)}
    except Exception as e:
        import traceback
        return {"status": "error", "error": traceback.format_exc()}


@router.post("", response_model=SearchResponse)
async def search_events(request: SearchRequest) -> SearchResponse:
    """
    Search indexed video events using natural language.
    
    This endpoint vectorizes the input query, retrieves matching events from Qdrant
    based on cosine similarity distance, and ranks the results. Results can be optionally
    filtered by a list of video IDs.
    """
    try:
        logger.info(f"Executing semantic search query: '{request.query}' (limit: {request.limit}, threshold: {request.score_threshold})")
        results = SearchService.search_events(
            query=request.query,
            limit=request.limit,
            video_ids=request.video_ids,
            score_threshold=request.score_threshold
        )
        return SearchResponse(query=request.query, results=results)
    except Exception as e:
        logger.error(f"Error during semantic search execution: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during search execution")
