import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.core.config import settings
from app.services.embedding_service import EmbeddingService
from app.services.pipeline_contract import event_catalog_path


def _time_to_seconds(value: str) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    parts = text.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
            return float(hours * 3600 + minutes * 60 + seconds)
        if len(parts) == 2:
            minutes, seconds = map(int, parts)
            return float(minutes * 60 + seconds)
        return float(text)
    except Exception:
        return 0.0


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
            f"[CUR:{item.get('evidence_id')}] "
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


def build_current_evidence_payload(
    actor_timeline: Dict[str, Any],
    evidence_graph: Dict[str, Any],
    events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "actor_summary": actor_timeline.get("summary", {}),
        "graph_summary": evidence_graph.get("summary", {}),
        "actors": _compact_actors(actor_timeline)[:20],
        "aggregated_events": _compact_events(events)[:20],
        "evidence_timeline": _timeline_lines(evidence_graph)[:40],
    }


def build_query_document(
    actor_timeline: Dict[str, Any],
    evidence_graph: Dict[str, Any],
    events: List[Dict[str, Any]],
) -> str:
    parts: List[str] = []
    for event in events:
        parts.append(str(event.get("event_type", "")))
        parts.append(str(event.get("description", "")))
        parts.extend(str(item) for item in event.get("activities", []) or [])
        parts.extend(str(item) for item in event.get("behavioral_flags", []) or [])
        for obj in event.get("objects", []) or []:
            if isinstance(obj, dict):
                parts.append(str(obj.get("type", "")))
                parts.append(str(obj.get("subtype", "")))
                parts.extend(str(item) for item in obj.get("attributes", []) or [])
    for item in evidence_graph.get("evidence_units", []) or []:
        parts.append(str(item.get("evidence_type", "")))
        parts.append(str(item.get("description", "")))
    return " ".join(part for part in parts if part).strip()


def _historical_event_document(video_id: str, event: Dict[str, Any]) -> Dict[str, Any]:
    objects = []
    for obj in event.get("objects", []) or []:
        if isinstance(obj, dict):
            objects.append(str(obj.get("subtype", "") or obj.get("type", "")))
        else:
            objects.append(str(obj))
    text = " ".join(
        [
            str(event.get("event_type", "")),
            str(event.get("description", "")),
            " ".join(str(item) for item in event.get("activities", []) or []),
            " ".join(str(item) for item in event.get("behavioral_flags", []) or []),
            " ".join(objects),
            str(event.get("location_text", "")),
            str(event.get("narrative_sentence", "")),
        ]
    ).strip()
    return {
        "video_id": video_id,
        "event_id": event.get("event_id", ""),
        "start_time": event.get("start_time", ""),
        "end_time": event.get("end_time", ""),
        "event_type": event.get("event_type", ""),
        "description": event.get("description", ""),
        "activities": event.get("activities", []),
        "behavioral_flags": event.get("behavioral_flags", []),
        "location_text": event.get("location_text", ""),
        "objects": objects,
        "document": text,
    }


def load_historical_event_corpus(metadata_dir: Path, current_video_id: str) -> List[Dict[str, Any]]:
    corpus: List[Dict[str, Any]] = []
    for path in sorted(metadata_dir.glob("*_events.json")):
        video_id = path.name.removesuffix("_events.json")
        if video_id == current_video_id:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        for event in payload:
            if isinstance(event, dict):
                corpus.append(_historical_event_document(video_id, event))
    return corpus


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve_similar_historical_events(
    actor_timeline: Dict[str, Any],
    evidence_graph: Dict[str, Any],
    events: List[Dict[str, Any]],
    metadata_dir: Path,
    current_video_id: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    corpus = load_historical_event_corpus(metadata_dir, current_video_id)
    if not corpus:
        return []

    query_doc = build_query_document(actor_timeline, evidence_graph, events)
    query_embedding = EmbeddingService.generate_embeddings(query_doc)
    corpus_embeddings = EmbeddingService.generate_embeddings([item["document"] for item in corpus])

    ranked: List[Dict[str, Any]] = []
    for item, vector in zip(corpus, corpus_embeddings):
        similarity = _cosine_similarity(query_embedding, vector)
        ranked.append({**item, "similarity": round(float(similarity), 6)})

    ranked.sort(key=lambda row: (-row["similarity"], row["video_id"], row["event_id"]))
    results = []
    for index, item in enumerate(ranked[:top_k], start=1):
        result = dict(item)
        result["retrieval_id"] = f"hist_{index:03d}"
        results.append(result)
    return results


def build_reasoning_prompt(
    current_payload: Dict[str, Any],
    retrieved_events: List[Dict[str, Any]],
    include_retrieval: bool,
) -> str:
    prompt_lines = [
        "You are an investigative reasoning assistant for surveillance analysis.",
        "Use only the structured evidence provided below.",
        "Do not invent facts. Do not assign fixed incident labels unless the evidence explicitly supports them.",
        "Prefer objective language such as observed, appears, remained, approached, carried, gathered, entered, exited.",
        "Retrieved historical evidence is reference context only. Do not claim it happened in the current video.",
        "When you use evidence in supporting_evidence, cite evidence IDs like [CUR:ev_001] or retrieval IDs like [RET:hist_001].",
        "Return JSON only with exactly these keys:",
        "{",
        '  "summary": "string",',
        '  "important_activities": ["string"],',
        '  "suspicious_observations": ["string"],',
        '  "timeline_summary": ["string"],',
        '  "risk_assessment": "string",',
        '  "supporting_evidence": ["string"]',
        "}",
        "",
        "CURRENT VIDEO EVIDENCE:",
        json.dumps(current_payload, indent=2),
    ]

    if include_retrieval:
        prompt_lines.extend(
            [
                "",
                "RETRIEVED HISTORICAL EVIDENCE:",
                json.dumps(
                    [
                        {
                            "retrieval_id": f"[RET:{item['retrieval_id']}]",
                            "video_id": item.get("video_id"),
                            "event_id": item.get("event_id"),
                            "similarity": item.get("similarity"),
                            "event_type": item.get("event_type"),
                            "description": item.get("description"),
                            "activities": item.get("activities", []),
                            "behavioral_flags": item.get("behavioral_flags", []),
                            "location_text": item.get("location_text", ""),
                            "objects": item.get("objects", []),
                        }
                        for item in retrieved_events
                    ],
                    indent=2,
                ),
                "",
                "Task: Generate an investigative summary for the current video evidence. You may use retrieved evidence only as comparison context.",
            ]
        )
    else:
        prompt_lines.extend(
            [
                "",
                "Task: Generate an investigative summary for the current video evidence only.",
            ]
        )

    return "\n".join(prompt_lines) + "\n"


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
        config={"response_mime_type": "application/json", "temperature": 0.1},
    )

    raw_text = (response.text or "").strip()
    if not raw_text:
        raise RuntimeError("LLM returned an empty response.")
    return json.loads(raw_text)


def _citation_count(result: Dict[str, Any], prefix: str) -> int:
    texts = []
    texts.append(str(result.get("summary", "")))
    texts.append(str(result.get("risk_assessment", "")))
    texts.extend(str(item) for item in result.get("important_activities", []))
    texts.extend(str(item) for item in result.get("suspicious_observations", []))
    texts.extend(str(item) for item in result.get("timeline_summary", []))
    texts.extend(str(item) for item in result.get("supporting_evidence", []))
    return sum(text.count(prefix) for text in texts)


def compare_reasoning_outputs(
    baseline: Dict[str, Any],
    augmented: Dict[str, Any],
    retrieved_events: List[Dict[str, Any]],
    baseline_prompt: str,
    augmented_prompt: str,
) -> Dict[str, Any]:
    avg_similarity = 0.0
    if retrieved_events:
        avg_similarity = sum(float(item.get("similarity", 0.0)) for item in retrieved_events) / len(retrieved_events)
    metrics = {
        "retrieval_count": len(retrieved_events),
        "retrieval_max_similarity": max((float(item.get("similarity", 0.0)) for item in retrieved_events), default=0.0),
        "retrieval_avg_similarity": round(avg_similarity, 6),
        "baseline_prompt_chars": len(baseline_prompt),
        "augmented_prompt_chars": len(augmented_prompt),
        "baseline_summary_chars": len(str(baseline.get("summary", ""))),
        "augmented_summary_chars": len(str(augmented.get("summary", ""))),
        "baseline_important_activity_count": len(baseline.get("important_activities", [])),
        "augmented_important_activity_count": len(augmented.get("important_activities", [])),
        "baseline_suspicious_count": len(baseline.get("suspicious_observations", [])),
        "augmented_suspicious_count": len(augmented.get("suspicious_observations", [])),
        "baseline_timeline_count": len(baseline.get("timeline_summary", [])),
        "augmented_timeline_count": len(augmented.get("timeline_summary", [])),
        "baseline_supporting_evidence_count": len(baseline.get("supporting_evidence", [])),
        "augmented_supporting_evidence_count": len(augmented.get("supporting_evidence", [])),
        "baseline_current_citation_count": _citation_count(baseline, "[CUR:"),
        "augmented_current_citation_count": _citation_count(augmented, "[CUR:"),
        "augmented_retrieval_citation_count": _citation_count(augmented, "[RET:"),
        "summary_char_delta": len(str(augmented.get("summary", ""))) - len(str(baseline.get("summary", ""))),
        "suspicious_count_delta": len(augmented.get("suspicious_observations", [])) - len(baseline.get("suspicious_observations", [])),
        "supporting_evidence_delta": len(augmented.get("supporting_evidence", [])) - len(baseline.get("supporting_evidence", [])),
    }
    metrics["retrieval_used_in_reasoning"] = metrics["augmented_retrieval_citation_count"] > 0
    return metrics
