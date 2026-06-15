"""Service for building causal narratives from aggregated event timelines using an LLM Reasoner."""

import json
from typing import List, Dict, Any, Optional
from loguru import logger
from pydantic import BaseModel, Field

from app.core.config import settings
from app.schemas.summary import AggregatedEvent
from app.schemas.incident import IncidentChain

class IncidentAnalysis(BaseModel):
    primary_incident: str = Field(description="The overarching name of the incident (e.g., 'Theft and Pursuit', 'Vehicle Collision')")
    severity: str = Field(description="Severity of the incident. Must be one of: CRITICAL, HIGH, MEDIUM, LOW")
    description: str = Field(description="A detailed causal summary explaining what happened and how events connect.")
    causal_chain: List[str] = Field(description="A list of 3-5 concise bullet points mapping the chronology and causality of the incident.")
    recommendations: List[str] = Field(description="Recommended actions for security personnel.")
    source_candidate_chain_ids: List[int] = Field(description="List of Candidate Chain IDs that make up this final validated incident.")

class InvestigationResult(BaseModel):
    incidents: List[IncidentAnalysis]

class NarrativeBuilderService:
    """Uses LLMs to validate semantic candidate chains and generate narrative chains."""
    
    @classmethod
    def gemini_available(cls) -> bool:
        """Safely check if the Gemini package is installed and API key is configured."""
        if not settings.GEMINI_API_KEY:
            return False
            
        try:
            import google.genai
            return True
        except ImportError:
            return False
            
    @classmethod
    def generate_narrative_from_events(cls, events: List[AggregatedEvent]) -> List[IncidentChain]:
        """Convert a list of events into structured candidate chains and validate them using an LLM."""
        if not events:
            return []
            
        logger.info("[INFO] NarrativeBuilder initialized")
            
        from app.services.incident_engine import IncidentEngine
        
        if not cls.gemini_available():
            if not settings.GEMINI_API_KEY:
                logger.warning("[WARNING] GEMINI_API_KEY missing. Falling back to IncidentEngine.")
            else:
                logger.warning("[WARNING] Gemini package not installed. Falling back to IncidentEngine.")
            return IncidentEngine.correlate_events(events)
            
        logger.info(f"Using Correlation Engine to build candidate chains for {len(events)} events.")
        
        # 1. Ask Correlation Engine to group events semantically into candidate chains
        candidate_chains = IncidentEngine.build_candidate_chains(events)
        
        if not candidate_chains:
            return []
            
        logger.info(f"Correlation Engine generated {len(candidate_chains)} candidate chains. Sending to LLM ({settings.NARRATIVE_MODEL_ID}) for validation.")
        
        # 2. Format the candidate chains for the LLM
        chains_prompt = cls._format_candidate_chains_for_prompt(candidate_chains)
        
        # 3. Call the LLM Validator
        analysis_list = cls._call_llm_validator(chains_prompt)
        
        if analysis_list is None:
            # Fallback triggered because the LLM call failed
            return IncidentEngine.correlate_events(events)
            
        if not analysis_list:
            logger.info("LLM Reasoner found no incidents.")
            return []
            
        # 4. Map the LLM output back to IncidentChain objects
        final_incidents = []
        for analysis in analysis_list:
            # Reconstruct the raw events that belong to this final validated incident
            merged_events = []
            for chain_id in analysis.source_candidate_chain_ids:
                # 1-indexed in the prompt
                idx = chain_id - 1
                if 0 <= idx < len(candidate_chains):
                    merged_events.extend(candidate_chains[idx])
                    
            if not merged_events:
                continue
                
            # Sort chronologically just in case the LLM merged them out of order
            merged_events.sort(key=lambda x: x.start_time)

            final_incidents.append(IncidentChain(
                incident_type=analysis.primary_incident,
                severity=analysis.severity.upper() if analysis.severity.upper() in ["CRITICAL", "HIGH", "MEDIUM", "LOW"] else "MEDIUM",
                description=analysis.description,
                start_time=merged_events[0].start_time,
                end_time=merged_events[-1].end_time,
                chain_events=[e.model_dump() for e in merged_events],
                timeline=analysis.causal_chain,
                recommendations=analysis.recommendations
            ))
            
        logger.info(f"LLM Validator finalized {len(final_incidents)} macro-incidents.")
        return final_incidents

    @classmethod
    def generate_investigation_report(cls, timeline_text: str) -> dict:
        """Call Gemini to generate a professional Investigation Report."""
        if not cls.gemini_available():
            logger.warning("Gemini API not configured. Cannot generate investigation report.")
            return {}

        prompt = (
            "You are a professional CCTV investigator.\n"
            "Analyze the timeline and incident chains.\n"
            "Produce:\n"
            "1. Executive Summary\n"
            "2. Chronological Narrative\n"
            "3. Key Findings\n"
            "4. Recommended Actions\n\n"
            "Focus on: theft, collisions, falls, injuries, pursuits, arrests, fire, suspicious behavior.\n"
            "Use investigator language. Do not produce generic AI summaries.\n"
            "Return structured JSON only, strictly matching this format:\n"
            "{\n"
            '  "executive_summary": "2-5 sentences",\n'
            '  "incident_narrative": "Story-like reconstruction",\n'
            '  "key_findings": ["Finding 1", "Finding 2"],\n'
            '  "recommendations": ["Action 1", "Action 2"]\n'
            "}\n\n"
            f"Timeline:\n{timeline_text}\n"
        )
        
        try:
            from google import genai
            client = genai.Client(api_key=settings.GEMINI_API_KEY)
            resp = client.models.generate_content(
                model=settings.NARRATIVE_MODEL_ID,
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            raw_text = resp.text.strip()
            
            # Simple JSON cleanup if it includes markdown blocks
            if raw_text.startswith("```json"):
                raw_text = raw_text.replace("```json", "", 1)
            if raw_text.endswith("```"):
                raw_text = raw_text[: -3]
                
            report_data = json.loads(raw_text.strip())
            return report_data
        except Exception as e:
            logger.error(f"[ERROR] Gemini Investigation Report generation failed: {e}")
            return {}

    @classmethod
    def _format_events_for_prompt(cls, events: List[AggregatedEvent]) -> str:
        """Serialize a flat list of events into a human-readable timeline for the LLM."""
        lines = []
        for e_idx, e in enumerate(events):
            desc = e.narrative_sentence or e.description
            flags = ", ".join(e.behavioral_flags) if e.behavioral_flags else "none"
            activities = ", ".join(e.activities) if e.activities else "none"
            lines.append(f"Event {e_idx + 1}")
            lines.append(f"Time: {e.start_time} - {e.end_time}")
            lines.append(f"Actors: {e.actor_description or 'Unknown'}")
            lines.append(f"Activities: {activities}")
            lines.append(f"Behavioral Flags: {flags}")
            lines.append(f"Scene: {e.scene_context or 'Unknown'}")
            lines.append(f"Caption: {desc}\n")
            
        return "\n".join(lines)

    @classmethod
    def _format_candidate_chains_for_prompt(cls, candidate_chains: List[List[AggregatedEvent]]) -> str:
        """Serialize candidate chains into a human-readable format for the LLM."""
        lines = []
        for c_idx, chain in enumerate(candidate_chains):
            lines.append(f"--- CANDIDATE CHAIN {c_idx + 1} ---")
            for e_idx, e in enumerate(chain):
                desc = e.narrative_sentence or e.description
                flags = ", ".join(e.behavioral_flags) if e.behavioral_flags else "none"
                activities = ", ".join(e.activities) if e.activities else "none"
                lines.append(f"  Event {e_idx + 1}")
                lines.append(f"  Time: {e.start_time} - {e.end_time}")
                lines.append(f"  Actors: {e.actor_description or 'Unknown'}")
                lines.append(f"  Activities: {activities}")
                lines.append(f"  Behavioral Flags: {flags}")
                lines.append(f"  Scene: {e.scene_context or 'Unknown'}")
                lines.append(f"  Caption: {desc}\n")
            lines.append("")
            
        prompt_text = "\n".join(lines)
        logger.debug(f"NarrativeBuilder Validation Prompt:\n{prompt_text}")
        return prompt_text
        
    @classmethod
    def _call_llm_validator(cls, chains_text: str) -> Optional[List[IncidentAnalysis]]:
        """Call Gemini to validate candidate chains."""
        prompt = (
            "You are an AI investigation analyst.\n"
            "The correlation engine has produced 'Candidate Incident Chains' by clustering video events based on semantic similarity.\n"
            "Your task is to validate these chains. If a Candidate Chain represents a cohesive incident on its own, validate it.\n"
            "If multiple Candidate Chains actually belong to the same overarching macro-incident (e.g., theft in Chain 1, and pursuit in Chain 2), merge them by specifying multiple source chain IDs.\n"
            "If a chain represents only normal routine activity (e.g., people just walking by normally), ignore it.\n"
            "Use only provided observations. Focus on: theft, pursuit, police intervention, arrest, collisions, injuries, falls, medical emergencies, fire, suspicious behavior.\n"
            "Return structured JSON only.\n\n"
            f"Candidate Chains:\n{chains_text}\n\n"
        )
        
        try:
            from google import genai
            client = genai.Client(api_key=settings.GEMINI_API_KEY)
            response = client.models.generate_content(
                model=settings.NARRATIVE_MODEL_ID,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=InvestigationResult,
                    temperature=0.1,
                ),
            )
            
            if response.text:
                data = json.loads(response.text)
                result = InvestigationResult(**data)
                return result.incidents
            return []
            
        except Exception as e:
            logger.error(f"[ERROR] Gemini validation failed. Falling back to IncidentEngine. Details: {e}")
            return None
