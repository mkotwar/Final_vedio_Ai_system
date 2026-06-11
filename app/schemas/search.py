"""Pydantic schemas for Semantic Event Search.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Any


class SearchRequest(BaseModel):
    """Schema representing a semantic search query request."""
    
    query: str = Field(..., min_length=1, description="Natural language search query")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of results to return")
    video_ids: Optional[List[str]] = Field(default=None, description="Optional list of video UUIDs to filter search results")
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum similarity score (0‑1) required for a result")


class SearchResultItem(BaseModel):
    """Schema representing a single search result hit."""
    
    score: float = Field(..., description="Cosine similarity confidence score")
    event_id: str = Field(..., description="ID of the matching event")
    video_id: str = Field(..., description="ID of the source video")
    event_type: str = Field(..., description="Category of the event")
    description: str = Field(..., description="Text description of the event")
    start_time: str = Field(..., description="Human-readable start time (HH:MM:SS)")
    end_time: str = Field(..., description="Human-readable end time (HH:MM:SS)")
    duration_seconds: float = Field(..., description="Duration of the event in seconds")
    objects: List[Any] = Field(default_factory=list, description="Objects detected in this event")
    activities: List[str] = Field(default_factory=list, description="Activities detected in this event")
    thumbnail_path: Optional[str] = Field(None, description="Path to the visual thumbnail for this event")
    severity: Optional[str] = Field(None, description="Event severity string indicator")
    narrative: Optional[str] = Field(None, description="Investigator narrative generated for the event")
    match_reasons: List[str] = Field(default_factory=list, description="Explainability reasons for why this event matched")


class SearchResponse(BaseModel):
    """Schema representing the complete semantic search query response."""
    
    query: str = Field(..., description="The original search query")
    results: List[SearchResultItem] = Field(..., description="List of ranked matching events")
