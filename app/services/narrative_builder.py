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

class InvestigationResult(BaseModel):
    incidents: List[IncidentAnalysis]

class NarrativeBuilderService:
    """Uses LLMs to reason about temporal events and generate narrative chains."""
    
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
        """Convert a list of events into a structured timeline and reason about them using an LLM."""
        if not events:
            return []
            
        logger.info("[INFO] NarrativeBuilder initialized")
            
        if not cls.gemini_available():
            if not settings.GEMINI_API_KEY:
                logger.warning("[WARNING] GEMINI_API_KEY missing. Falling back to IncidentEngine.")
            else:
                logger.warning("[WARNING] Gemini package not installed. Falling back to IncidentEngine.")
                
            from app.services.incident_engine import IncidentEngine
            return IncidentEngine.correlate_events(events)
            
        logger.info(f"Using LLM Reasoner ({settings.NARRATIVE_MODEL_ID}) to analyze {len(events)} events.")
        
        # 1. Format the events into a clean chronological timeline
        timeline_prompt = cls._format_events_for_prompt(events)
        
        # 2. Call the LLM Reasoner
        analysis_list = cls._call_llm_reasoner(timeline_prompt)
        
        if analysis_list is None:
            # Fallback triggered because the LLM call failed
            from app.services.incident_engine import IncidentEngine
            return IncidentEngine.correlate_events(events)
            
        if not analysis_list:
            logger.info("LLM Reasoner found no incidents.")
            return []
            
        # 3. Map the LLM output back to IncidentChain objects
        chains = []
        for analysis in analysis_list:
            chains.append(IncidentChain(
                incident_type=analysis.primary_incident,
                severity=analysis.severity.upper() if analysis.severity.upper() in ["CRITICAL", "HIGH", "MEDIUM", "LOW"] else "MEDIUM",
                description=analysis.description,
                start_time=events[0].start_time,
                end_time=events[-1].end_time,
                chain_events=[e.model_dump() for e in events],
                timeline=analysis.causal_chain,
                recommendations=analysis.recommendations
            ))
            
        logger.info(f"LLM Reasoner identified {len(chains)} macro-incidents.")
        return chains

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
        """Serialize events into a compact, human-readable format for the LLM."""
        lines = []
        for idx, e in enumerate(events):
            desc = e.narrative_sentence or e.description
            flags = ", ".join(e.behavioral_flags) if e.behavioral_flags else "none"
            activities = ", ".join(e.activities) if e.activities else "none"
            lines.append(f"Event {idx + 1}")
            lines.append(f"Time:\n{e.start_time} - {e.end_time}")
            lines.append(f"Actors:\n{e.actor_description or 'Unknown'}")
            lines.append(f"Activities:\n{activities}")
            lines.append(f"Behavioral Flags:\n{flags}")
            lines.append(f"Scene:\n{e.scene_context or 'Unknown'}")
            lines.append(f"Caption:\n{desc}\n")
            
        prompt_text = "\n".join(lines)
        # Temporary DEBUG log
        logger.debug(f"NarrativeBuilder Semantic Prompt:\n{prompt_text}")
        return prompt_text
        
    @classmethod
    def _call_llm_reasoner(cls, timeline_text: str) -> Optional[List[IncidentAnalysis]]:
        """Call Gemini to reason about the timeline."""
        prompt = (
            "You are an AI investigation analyst.\n"
            "Use only provided observations. Do not rely on predefined event categories. "
            "Infer incidents from behavior and chronology.\n"
            "Focus on: theft, pursuit, police intervention, arrest, collisions, injuries, falls, medical emergencies, fire, suspicious behavior.\n"
            "Return structured JSON only.\n\n"
            f"Timeline:\n{timeline_text}\n\n"
            "Identify distinct major incidents. If the events only show completely normal routine activity (e.g., people just walking by normally), return an empty incidents list."
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
            logger.error(f"[ERROR] Gemini request failed. Falling back to IncidentEngine. Details: {e}")
            return None
