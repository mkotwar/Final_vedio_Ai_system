"""Service for generating structured investigation reports."""

import datetime
from typing import List

from loguru import logger
from app.schemas.report import (
    InvestigationReport,
    ReportTimelineEntry,
    ReportCriticalFinding,
)
from app.services.summary_service import SummaryService

class ReportService:
    """Generates investigation-grade reports from video metadata."""

    @classmethod
    def generate_report(cls, video_id: str) -> InvestigationReport:
        """Orchestrates the creation of an Investigation Report."""
        logger.info(f"Generating investigation report for video {video_id}")
        
        # 1. Fetch raw data from existing summary service
        summary = SummaryService.generate_summary(video_id)
        
        # 2. Risk Assessment (Part 3)
        risk_level = cls._calculate_risk_level(summary)
        
        # 3. Critical Findings (Part 5)
        findings = cls._extract_critical_findings(summary)
        
        # 4. Timeline Formatting (Part 6)
        timeline = cls._format_timeline(summary)
        
        # 5. Recommendations (Part 7)
        recommendations = cls._generate_recommendations(summary)
        
        # 6. Incident Summary
        incident_summary = [f"{inc.incident_type.replace('_', ' ').title()}" for inc in summary.incidents]
        
        return InvestigationReport(
            video_id=video_id,
            title=f"Investigation Report: {video_id}",
            generated_at=datetime.datetime.utcnow().isoformat() + "Z",
            risk_level=risk_level,
            executive_summary=summary.overview,  # Uses the 3-6 sentence narrative overview
            critical_findings=findings,
            incident_summary=list(set(incident_summary)),
            timeline=timeline,
            recommendations=recommendations,
            statistics=summary.statistics.model_dump()
        )

    @classmethod
    def _calculate_risk_level(cls, summary) -> str:
        """
        severity < 30: LOW
        severity 30-70: MEDIUM
        severity > 70: HIGH
        Multiple critical incidents: CRITICAL
        """
        max_severity = 0
        critical_count = 0
        
        severity_map = {
            "low": 20,
            "medium": 50,
            "high": 80,
            "critical": 100
        }
        
        for inc in summary.incidents:
            val = severity_map.get(inc.severity.lower(), 20)
            max_severity = max(max_severity, val)
            if val >= 80:
                critical_count += 1
                
        for not_event in summary.notable_events:
            val = severity_map.get(not_event.severity.lower(), 20)
            max_severity = max(max_severity, val)
            if val >= 80:
                critical_count += 1
                
        if critical_count >= 2:
            return "CRITICAL"
        elif max_severity > 70:
            return "HIGH"
        elif max_severity >= 30:
            return "MEDIUM"
        else:
            return "LOW"

    @classmethod
    def _extract_critical_findings(cls, summary) -> List[ReportCriticalFinding]:
        """Convert notable events and incidents into structured findings."""
        findings = []
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        
        for n in summary.notable_events:
            findings.append(ReportCriticalFinding(
                event_type=n.event_type.replace('_', ' ').title(),
                severity=n.severity.upper(),
                description=n.reason or n.description,
                timestamp=n.timestamp
            ))
            
        for i in summary.incidents:
            findings.append(ReportCriticalFinding(
                event_type=i.incident_type.replace('_', ' ').title(),
                severity=i.severity.upper(),
                description=i.description,
                timestamp=i.start_time or "Unknown"
            ))
            
        # Deduplicate and sort by severity
        seen = set()
        unique_findings = []
        for f in findings:
            key = f"{f.event_type}-{f.timestamp}"
            if key not in seen:
                seen.add(key)
                unique_findings.append(f)
                
        unique_findings.sort(key=lambda x: severity_order.get(x.severity.lower(), 4))
        return unique_findings

    @classmethod
    def _format_timeline(cls, summary) -> List[ReportTimelineEntry]:
        """Generate investigator-readable timeline."""
        timeline = []
        for entry in summary.timeline:
            # Extract start time from time_range (e.g. '00:00:03 - 00:00:10')
            start_time = entry.time_range.split(" - ")[0] if " - " in entry.time_range else entry.time_range
            desc = entry.description.strip()
            if not desc:
                desc = f"{entry.event_type.replace('_', ' ').title()} observed."
            
            # Format: 00:00:03 Individual enters kitchen
            timeline.append(ReportTimelineEntry(
                timestamp=start_time,
                description=desc
            ))
        return timeline

    @classmethod
    def _generate_recommendations(cls, summary) -> List[str]:
        """Apply rule-based recommendations based on identified events."""
        recommendations = set()
        
        # Combine all event types detected
        event_types = set()
        for i in summary.incidents:
            event_types.add(i.incident_type.lower())
        for n in summary.notable_events:
            event_types.add(n.event_type.lower())
        for t in summary.timeline:
            event_types.add(t.event_type.lower())
            
        for et in event_types:
            if "fall" in et or "medical" in et:
                recommendations.add("Medical assistance and wellness check recommended.")
            elif "collision" in et or "crash" in et or "accident" in et:
                recommendations.add("Vehicle damage assessment and insurance review recommended.")
            elif "fire" in et or "smoke" in et:
                recommendations.add("Immediate emergency response and fire safety review required.")
            elif "intrusion" in et or "unauthorized" in et:
                recommendations.add("Security sweep and access control audit recommended.")
            elif "weapon" in et:
                recommendations.add("Immediate law enforcement escalation required. Lockdown recommended.")
            elif "abandon" in et:
                recommendations.add("Security sweep recommended to clear abandoned object.")
                
        # If no critical events, provide a generic recommendation
        if not recommendations:
            if cls._calculate_risk_level(summary) in ["LOW", "MEDIUM"]:
                recommendations.add("No immediate action required. Continue routine monitoring.")
            else:
                recommendations.add("Review flagged events for potential security implications.")
                
        return list(recommendations)
