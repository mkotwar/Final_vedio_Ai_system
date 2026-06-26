import json
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from app.core.config import settings


class InvestigationReasoningOutput(BaseModel):
    summary: str = Field(description="High-level investigative summary grounded in evidence only.")
    important_activities: List[str] = Field(default_factory=list, description="Most important observed activities.")
    suspicious_observations: List[str] = Field(default_factory=list, description="Potentially unusual observations grounded in evidence.")
    timeline_summary: List[str] = Field(default_factory=list, description="Chronological summary of the video evidence.")
    risk_assessment: str = Field(description="Objective risk assessment based only on observed evidence.")
    supporting_evidence: List[str] = Field(default_factory=list, description="Evidence items supporting the conclusions.")


def gemini_available() -> bool:
    if not settings.GEMINI_API_KEY:
        return False
    try:
        import google.genai  # noqa: F401
        return True
    except ImportError:
        return False


def _timeline_lines(evidence_graph: Dict[str, Any]) -> List[str]:
    rows = []
    for item in sorted(
        evidence_graph.get("evidence_units", []),
        key=lambda unit: (float(unit.get("time_start", 0.0)), unit.get("evidence_id", "")),
    ):
        rows.append(
            f"{item.get('time_start', 0.0):06.1f}s -> {item.get('time_end', 0.0):06.1f}s | "
            f"{item.get('evidence_type')} | actors={item.get('actors', [])} | "
            f"objects={item.get('objects', [])} | location={item.get('location', '')} | "
            f"event={item.get('event_id', '')} | {item.get('description', '')}"
        )
    return rows


def _compact_actors(actor_timeline: Dict[str, Any]) -> List[Dict[str, Any]]:
    compact = []
    for actor in actor_timeline.get("actors", []):
        compact.append(
            {
                "global_actor_id": actor.get("global_actor_id"),
                "entity_type": actor.get("entity_type"),
                "attributes": actor.get("attributes", {}),
                "timeline": actor.get("timeline", [])[:12],
                "continuity_stats": actor.get("continuity_stats", {}),
            }
        )
    return compact


def _compact_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact = []
    for event in events:
        compact.append(
            {
                "event_id": event.get("event_id"),
                "event_type": event.get("event_type"),
                "start_time": event.get("start_time"),
                "end_time": event.get("end_time"),
                "duration_seconds": event.get("duration_seconds"),
                "description": event.get("description"),
                "activities": event.get("activities", []),
                "location_text": event.get("location_text"),
                "behavioral_flags": event.get("behavioral_flags", []),
                "confidence": event.get("confidence"),
                "participants": event.get("participants", []),
                "objects": event.get("objects", [])[:10],
            }
        )
    return compact


def build_evidence_prompt(
    actor_timeline: Dict[str, Any],
    evidence_graph: Dict[str, Any],
    events: List[Dict[str, Any]],
) -> str:
    actor_summary = actor_timeline.get("summary", {})
    graph_summary = evidence_graph.get("summary", {})
    timeline_lines = _timeline_lines(evidence_graph)[:40]

    payload = {
        "actor_summary": actor_summary,
        "graph_summary": graph_summary,
        "actors": _compact_actors(actor_timeline)[:20],
        "aggregated_events": _compact_events(events)[:20],
        "evidence_timeline": timeline_lines,
    }

    return (
        "You are an investigative reasoning assistant for surveillance analysis.\n"
        "Use only the structured evidence provided below.\n"
        "Do not invent facts. Do not assign fixed incident labels unless the evidence explicitly supports them.\n"
        "Prefer objective language such as observed, appears, remained, approached, carried, gathered, entered, exited.\n"
        "Return JSON only with exactly these keys:\n"
        "{\n"
        '  "summary": "string",\n'
        '  "important_activities": ["string"],\n'
        '  "suspicious_observations": ["string"],\n'
        '  "timeline_summary": ["string"],\n'
        '  "risk_assessment": "string",\n'
        '  "supporting_evidence": ["string"]\n'
        "}\n\n"
        "VIDEO EVIDENCE:\n"
        f"{json.dumps(payload, indent=2)}\n"
    )


def run_llm_reasoning(prompt: str) -> Dict[str, Any]:
    if not gemini_available():
        raise RuntimeError(
            "Gemini is not available for this benchmark. "
            "Set GEMINI_API_KEY and install the google genai package."
        )

    from google import genai

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=settings.NARRATIVE_MODEL_ID,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=InvestigationReasoningOutput,
            temperature=0.1,
        ),
    )

    raw_text = (response.text or "").strip()
    if not raw_text:
        raise RuntimeError("LLM returned an empty response.")

    parsed = json.loads(raw_text)
    validated = InvestigationReasoningOutput(**parsed)
    return validated.model_dump()
