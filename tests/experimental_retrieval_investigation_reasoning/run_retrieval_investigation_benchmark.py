import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT_PATH = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

if str(os.environ.get("DEBUG", "")).lower() not in {"", "0", "1", "true", "false", "yes", "no", "on", "off"}:
    os.environ["DEBUG"] = "true"

from app.core.config import PROJECT_ROOT, settings
from app.services.pipeline_contract import event_catalog_path
from tests.experimental_retrieval_investigation_reasoning.retrieval_reasoner import (
    build_current_evidence_payload,
    build_reasoning_prompt,
    compare_reasoning_outputs,
    gemini_available,
    retrieve_similar_historical_events,
    run_llm_reasoning,
)


BENCHMARK_VIDEO_ID = os.getenv("BENCHMARK_VIDEO_ID", "11111111-2222-4333-8444-555555555555")
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
CASE_ROOT = PROJECT_ROOT / "tests" / "experimental_retrieval_investigation_reasoning"
DATA_ROOT = CASE_ROOT / "data"
OUTPUT_ROOT = DATA_ROOT / "output"
ACTOR_TIMELINE_PATH = PROJECT_ROOT / "tests" / "experimental_actor_state" / "data" / "output" / "actor_state_timeline.json"
EVIDENCE_GRAPH_PATH = PROJECT_ROOT / "tests" / "experimental_evidence_graph" / "data" / "output" / "evidence_graph.json"
BASELINE_JSON_PATH = OUTPUT_ROOT / "retrieval_investigation_baseline.json"
AUGMENTED_JSON_PATH = OUTPUT_ROOT / "retrieval_investigation_augmented.json"
RETRIEVAL_JSON_PATH = OUTPUT_ROOT / "retrieval_candidates.json"
METRICS_JSON_PATH = OUTPUT_ROOT / "retrieval_benchmark_metrics.json"
REPORT_MD_PATH = OUTPUT_ROOT / "RETRIEVAL_INVESTIGATION_REPORT.md"
BASELINE_PROMPT_PATH = OUTPUT_ROOT / "baseline_prompt.txt"
AUGMENTED_PROMPT_PATH = OUTPUT_ROOT / "augmented_prompt.txt"


def _ensure_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_events(video_id: str) -> List[Dict[str, Any]]:
    path = event_catalog_path(video_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing aggregated events catalog: {path}. "
            "Run the existing pipeline first."
        )
    payload = _load_json(path, [])
    if not isinstance(payload, list):
        raise ValueError(f"Expected list payload in {path}")
    return payload


def _load_actor_timeline() -> Dict[str, Any]:
    if not ACTOR_TIMELINE_PATH.exists():
        raise FileNotFoundError(
            f"Missing actor timeline input: {ACTOR_TIMELINE_PATH}. "
            "Run tests\\experimental_actor_state\\run_actor_state_benchmark.py first."
        )
    payload = _load_json(ACTOR_TIMELINE_PATH, {})
    if not isinstance(payload, dict) or "actors" not in payload:
        raise ValueError(f"Invalid actor timeline payload in {ACTOR_TIMELINE_PATH}")
    return payload


def _load_evidence_graph() -> Dict[str, Any]:
    if not EVIDENCE_GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Missing evidence graph input: {EVIDENCE_GRAPH_PATH}. "
            "Run tests\\experimental_evidence_graph\\run_evidence_graph_benchmark.py first."
        )
    payload = _load_json(EVIDENCE_GRAPH_PATH, {})
    if not isinstance(payload, dict) or "evidence_units" not in payload:
        raise ValueError(f"Invalid evidence graph payload in {EVIDENCE_GRAPH_PATH}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def _write_text(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _write_report(metrics: Dict[str, Any], retrieved: List[Dict[str, Any]]) -> None:
    lines = [
        "# Retrieval Investigation Benchmark Report",
        "",
        f"- Video ID: `{BENCHMARK_VIDEO_ID}`",
        f"- Model: `{settings.NARRATIVE_MODEL_ID}`",
        f"- Gemini available: `{gemini_available()}`",
        f"- Retrieved historical events: `{metrics['retrieval_count']}`",
        f"- Retrieval max similarity: `{metrics['retrieval_max_similarity']:.4f}`",
        f"- Retrieval avg similarity: `{metrics['retrieval_avg_similarity']:.4f}`",
        f"- Baseline summary chars: `{metrics['baseline_summary_chars']}`",
        f"- Augmented summary chars: `{metrics['augmented_summary_chars']}`",
        f"- Baseline suspicious count: `{metrics['baseline_suspicious_count']}`",
        f"- Augmented suspicious count: `{metrics['augmented_suspicious_count']}`",
        f"- Augmented retrieval citation count: `{metrics['augmented_retrieval_citation_count']}`",
        f"- Retrieval used in reasoning: `{metrics['retrieval_used_in_reasoning']}`",
        "",
        "## Retrieved Evidence",
        "",
    ]
    for item in retrieved:
        lines.append(
            f"- `[RET:{item['retrieval_id']}]` video=`{item['video_id']}` event=`{item['event_id']}` "
            f"similarity=`{item['similarity']:.4f}` type=`{item['event_type']}` description=`{item['description']}`"
        )
    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    _ensure_dirs()
    events = _load_events(BENCHMARK_VIDEO_ID)
    actor_timeline = _load_actor_timeline()
    evidence_graph = _load_evidence_graph()

    current_payload = build_current_evidence_payload(actor_timeline, evidence_graph, events)
    retrieved_events = retrieve_similar_historical_events(
        actor_timeline=actor_timeline,
        evidence_graph=evidence_graph,
        events=events,
        metadata_dir=settings.METADATA_DIR,
        current_video_id=BENCHMARK_VIDEO_ID,
        top_k=RETRIEVAL_TOP_K,
    )

    baseline_prompt = build_reasoning_prompt(current_payload, retrieved_events, include_retrieval=False)
    augmented_prompt = build_reasoning_prompt(current_payload, retrieved_events, include_retrieval=True)
    _write_text(BASELINE_PROMPT_PATH, baseline_prompt)
    _write_text(AUGMENTED_PROMPT_PATH, augmented_prompt)

    baseline_result = run_llm_reasoning(baseline_prompt)
    augmented_result = run_llm_reasoning(augmented_prompt)
    metrics = compare_reasoning_outputs(
        baseline=baseline_result,
        augmented=augmented_result,
        retrieved_events=retrieved_events,
        baseline_prompt=baseline_prompt,
        augmented_prompt=augmented_prompt,
    )

    _write_json(BASELINE_JSON_PATH, baseline_result)
    _write_json(AUGMENTED_JSON_PATH, augmented_result)
    _write_json(RETRIEVAL_JSON_PATH, retrieved_events)
    _write_json(METRICS_JSON_PATH, metrics)
    _write_report(metrics, retrieved_events)

    summary = {
        "video_id": BENCHMARK_VIDEO_ID,
        "gemini_available": gemini_available(),
        "model_id": settings.NARRATIVE_MODEL_ID,
        "top_k": RETRIEVAL_TOP_K,
        "metrics_path": str(METRICS_JSON_PATH),
        "baseline_path": str(BASELINE_JSON_PATH),
        "augmented_path": str(AUGMENTED_JSON_PATH),
        "retrieval_path": str(RETRIEVAL_JSON_PATH),
        "report_path": str(REPORT_MD_PATH),
    }
    print("RETRIEVAL_INVESTIGATION_BENCHMARK_SUMMARY_START")
    print(json.dumps(summary, indent=2))
    print("RETRIEVAL_INVESTIGATION_BENCHMARK_SUMMARY_END")


if __name__ == "__main__":
    main()
