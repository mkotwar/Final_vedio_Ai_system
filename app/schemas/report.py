"""Pydantic schemas for Investigation Reports."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ReportTimelineEntry(BaseModel):
    timestamp: str = Field(..., description="Chronological timestamp (e.g., 00:00:03)")
    description: str = Field(..., description="Investigator-readable narrative of the event")

class ReportCriticalFinding(BaseModel):
    event_type: str = Field(..., description="Type of incident")
    severity: str = Field(..., description="Severity level")
    description: str = Field(..., description="Description of the finding")
    timestamp: str = Field(..., description="When it occurred")

class InvestigationReport(BaseModel):
    video_id: str = Field(..., description="Source video ID")
    title: str = Field(..., description="Report Title")
    generated_at: str = Field(..., description="ISO 8601 Datetime of generation")
    risk_level: str = Field(..., description="Overall Risk Level: LOW, MEDIUM, HIGH, CRITICAL")
    executive_summary: str = Field(..., description="3-6 sentences focusing on what, where, who, and significance")
    critical_findings: List[ReportCriticalFinding] = Field(default_factory=list, description="Notable events ranked by severity")
    incident_summary: List[str] = Field(default_factory=list, description="List of unique incidents found")
    timeline: List[ReportTimelineEntry] = Field(default_factory=list, description="Investigator-readable timeline")
    recommendations: List[str] = Field(default_factory=list, description="Rule-based recommendations for action")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Statistical breakdown of activity")
