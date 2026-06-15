"""Pydantic schemas for the Incident Correlation Engine."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import uuid

# We can reuse TimelineEntry or just use a simple string for the incident timeline
class IncidentChain(BaseModel):
    """A macro-incident composed of multiple correlated micro-events."""
    incident_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    incident_type: str = Field(..., description="The classified macro-incident type (e.g. road_rage_incident)")
    severity: str = Field(..., description="Severity: LOW, MEDIUM, HIGH, CRITICAL")
    description: str = Field(..., description="A synthesized narrative of the entire incident chain")
    start_time: str = Field(..., description="Start time of the first event in the chain")
    end_time: str = Field(..., description="End time of the last event in the chain")
    chain_events: List[Dict[str, Any]] = Field(default_factory=list, description="Raw event data making up the chain")
    timeline: List[str] = Field(default_factory=list, description="Chronological breakdown of the chain")
    recommendations: List[str] = Field(default_factory=list, description="Automated actions based on the chain")
    poster_frame: Optional[str] = None
    poster_timestamp: Optional[str] = None
    poster_event_id: Optional[str] = None
