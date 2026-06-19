"""Service for correlating isolated events into macro-incident chains."""

import uuid
from typing import List, Dict, Any, Tuple
from loguru import logger

from app.schemas.summary import AggregatedEvent
from app.schemas.incident import IncidentChain

class IncidentEngine:
    """Detects temporal relationships between events to build incident chains."""

    # Default correlation window in seconds
    CORRELATION_WINDOW_SECONDS = 60

    @staticmethod
    def _time_to_seconds(time_str: str) -> int:
        """Convert HH:MM:SS to seconds."""
        try:
            parts = time_str.split(":")
            if len(parts) == 3:
                h, m, s = map(int, parts)
                return h * 3600 + m * 60 + s
            elif len(parts) == 2:
                m, s = map(int, parts)
                return m * 60 + s
            return int(time_str)
        except Exception:
            return 0

    @classmethod
    def correlate_events(cls, events: List[AggregatedEvent]) -> List[IncidentChain]:
        """Process a list of events and group them into incident chains."""
        if not events:
            return []

        # Sort events chronologically
        sorted_events = sorted(events, key=lambda x: cls._time_to_seconds(x.start_time))
        
        chains = []
        current_chain = [sorted_events[0]]
        
        for event in sorted_events[1:]:
            prev_event = current_chain[-1]
            prev_end = cls._time_to_seconds(prev_event.end_time)
            curr_start = cls._time_to_seconds(event.start_time)
            
            # If the current event starts within CORRELATION_WINDOW_SECONDS of the previous event ending
            if curr_start - prev_end <= cls.CORRELATION_WINDOW_SECONDS:
                current_chain.append(event)
            else:
                chains.append(current_chain)
                current_chain = [event]
                
        if current_chain:
            chains.append(current_chain)

        incident_chains = []
        for chain in chains:
            incident = cls._evaluate_chain(chain)
            if incident:
                incident_chains.append(incident)

        # Sort final incidents by severity (Critical -> High -> Medium -> Low)
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        incident_chains.sort(key=lambda x: severity_order.get(x.severity, 4))
        
        return incident_chains

    @classmethod
    def _evaluate_chain(cls, chain: List[AggregatedEvent]) -> IncidentChain:
        """Analyze a grouped chain of events and classify the macro-incident."""
        
        # 1. Gather all tags, types, and flags from the chain
        all_types = set()
        all_flags = set()
        max_severity_val = 0
        severity_map = {"critical": 100, "high": 80, "medium": 50, "low": 20, "none": 0}
        reverse_sev_map = {100: "CRITICAL", 80: "HIGH", 50: "MEDIUM", 20: "LOW", 0: "LOW"}
        
        timeline_texts = []
        
        for e in chain:
            all_types.add(e.event_type.lower())
            for flag in e.behavioral_flags:
                all_flags.add(flag.lower())
                
            for fe in e.frame_events:
                fe_type = fe.event_type.lower().strip()
                if fe_type and fe_type != "none":
                    all_types.add(fe_type)
                    fe_sev = severity_map.get(fe.severity.lower(), 20)
                    if fe_sev > max_severity_val:
                        max_severity_val = fe_sev
            
            # Check main event severity too
            e_sev = 15 if not hasattr(e, "event_severity") else e.event_severity
            if e_sev >= 80:
                max_severity_val = max(max_severity_val, 80)
            elif e_sev >= 50:
                max_severity_val = max(max_severity_val, 50)
                
            desc = e.narrative_sentence or e.description
            timeline_texts.append(f"{e.start_time}: {desc}")

        # 2. Rule Matching
        incident_type = None
        recommendations = []
        description = ""
        
        # Helper to check intersections
        has_collision = bool(all_types.intersection({"collision", "vehicle_collision", "crash", "accident"})) or "vehicle_collision" in all_flags
        has_fall = bool(all_types.intersection({"fall", "person_fall"})) or "person_fall" in all_flags
        has_fight = bool(all_types.intersection({"fight", "physical_altercation", "argument"})) or "physical_altercation" in all_flags
        has_medical = bool(all_types.intersection({"medical_emergency", "medical_assistance"}))
        has_fire = bool(all_types.intersection({"fire", "smoke", "fire_smoke_detected"})) or "fire_smoke" in all_flags
        has_intrusion = bool(all_types.intersection({"intrusion", "unauthorized_access", "restricted_area"})) or "intrusion_detected" in all_flags
        has_abandon = bool(all_types.intersection({"abandoned_object", "abandonment"})) or "abandoned_object" in all_flags

        if has_collision and (has_fall or has_medical):
            incident_type = "vehicle_collision_with_injury"
            max_severity_val = max(max_severity_val, 100) # Elevate to critical
            description = "A vehicle collision occurred resulting in a potential injury or medical emergency."
            recommendations = ["Dispatch emergency medical services.", "Dispatch law enforcement for accident report."]
            
        elif has_collision and has_fight:
            incident_type = "road_rage_incident"
            max_severity_val = max(max_severity_val, 100)
            description = "A vehicle collision escalated into a physical altercation or argument."
            recommendations = ["Dispatch law enforcement immediately.", "Secure the area."]
            
        elif has_fire:
            incident_type = "fire_emergency"
            max_severity_val = max(max_severity_val, 100)
            description = "Smoke or fire detected in the monitored area."
            recommendations = ["Trigger fire alarm.", "Dispatch fire department.", "Initiate evacuation protocols."]
            
        elif has_intrusion and has_abandon:
            incident_type = "intrusion_sequence"
            max_severity_val = max(max_severity_val, 80)
            description = "An unauthorized intrusion was followed by an object being abandoned."
            recommendations = ["Dispatch security for sweep.", "Do not approach the abandoned object.", "Review access logs."]
            
        elif has_medical or (has_fall and len(chain) > 1): # Fall with extended duration/subsequent events
            incident_type = "medical_emergency"
            max_severity_val = max(max_severity_val, 80)
            description = "A person fell or required medical assistance."
            recommendations = ["Dispatch medical personnel.", "Conduct wellness check."]
            
        # 3. Fallback to the highest severity singular event type if no macro-rule matches
        if not incident_type:
            # We only care about chains that contain at least something notable/high severity
            if max_severity_val >= 50:
                # Find the most severe type to name the incident
                for fe in chain:
                    for frame_evt in fe.frame_events:
                        if severity_map.get(frame_evt.severity.lower(), 0) == max_severity_val:
                            incident_type = frame_evt.event_type
                            description = frame_evt.description
                            break
                    if incident_type:
                        break
                
                if not incident_type:
                    incident_type = chain[0].event_type
                    description = chain[0].narrative_sentence or chain[0].description
            else:
                return None # Skip routine activity chains

        if not recommendations:
            if max_severity_val >= 80:
                recommendations.append("Immediate review required.")
            else:
                recommendations.append("Monitor situation.")

        severity_str = "LOW"
        if max_severity_val >= 100:
            severity_str = "CRITICAL"
        elif max_severity_val >= 80:
            severity_str = "HIGH"
        elif max_severity_val >= 50:
            severity_str = "MEDIUM"

        return IncidentChain(
            incident_type=incident_type,
            severity=severity_str,
            description=description,
            start_time=chain[0].start_time,
            end_time=chain[-1].end_time,
            chain_events=[e.model_dump() for e in chain],
            timeline=timeline_texts,
            recommendations=recommendations
        )
