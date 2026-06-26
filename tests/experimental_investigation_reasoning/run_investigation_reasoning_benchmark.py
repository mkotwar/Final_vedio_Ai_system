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
from tests.experimental_investigation_reasoning.investigation_reasoner import (
    build_evidence_prompt,
    gemini_available,
    run_llm_reasoning,
)


BENCHMARK_VIDEO_ID = os.getenv("BENCHMARK_VIDEO_ID", "11111111-2222-4333-8444-555555555555")
CASE_ROOT = PROJECT_ROOT / "tests" / "experimental_investigation_reasoning"
DATA_ROOT = CASE_ROOT / "data"
OUTPUT_ROOT = DATA_ROOT / "output"
ACTOR_TIMELINE_PATH = PROJECT_ROOT / "tests" / "experimental_actor_state" / "data" / "output" / "actor_state_timeline.json"
EVIDENCE_GRAPH_PATH = PROJECT_ROOT / "tests" / "experimental_evidence_graph" / "data" / "output" / "evidence_graph.json"
REASONING_JSON_PATH = OUTPUT_ROOT / "investigation_reasoning.json"
SUMMARY_JSON_PATH = OUTPUT_ROOT / "investigation_reasoning_summary.json"
REPORT_MD_PATH = OUTPUT_ROOT / "INVESTIGATION_REASONING_REPORT.md"
PROMPT_TXT_PATH = OUTPUT_ROOT / "investigation_reasoning_prompt.txt"


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


def _write_report(video_id: str, result: Dict[str, Any], summary: Dict[str, Any]) -> None:
    lines = [
        "# Investigation Reasoning Report",
        "",
        f"- Video ID: `{video_id}`",
        f"- Aggregated events input: `{event_catalog_path(video_id)}`",
        f"- Actor timeline input: `{ACTOR_TIMELINE_PATH}`",
        f"- Evidence graph input: `{EVIDENCE_GRAPH_PATH}`",
        f"- Gemini available: `{summary['gemini_available']}`",
        f"- Model: `{summary['model_id']}`",
        "",
        "## Summary",
        "",
        result.get("summary", ""),
        "",
        "## Important Activities",
        "",
    ]
    for item in result.get("important_activities", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Suspicious Observations", ""])
    for item in result.get("suspicious_observations", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Timeline Summary", ""])
    for item in result.get("timeline_summary", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Risk Assessment", "", result.get("risk_assessment", ""), "", "## Supporting Evidence", ""])
    for item in result.get("supporting_evidence", []):
        lines.append(f"- {item}")
    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    _ensure_dirs()
    events = _load_events(BENCHMARK_VIDEO_ID)
    actor_timeline = _load_actor_timeline()
    evidence_graph = _load_evidence_graph()

    prompt = build_evidence_prompt(actor_timeline, evidence_graph, events)
    _write_text(PROMPT_TXT_PATH, prompt)

    result = run_llm_reasoning(prompt)
    summary = {
        "video_id": BENCHMARK_VIDEO_ID,
        "input_event_catalog": str(event_catalog_path(BENCHMARK_VIDEO_ID)),
        "input_actor_timeline": str(ACTOR_TIMELINE_PATH),
        "input_evidence_graph": str(EVIDENCE_GRAPH_PATH),
        "gemini_available": gemini_available(),
        "model_id": settings.NARRATIVE_MODEL_ID,
        "important_activity_count": len(result.get("important_activities", [])),
        "suspicious_observation_count": len(result.get("suspicious_observations", [])),
        "timeline_summary_count": len(result.get("timeline_summary", [])),
        "supporting_evidence_count": len(result.get("supporting_evidence", [])),
        "outputs": {
            "reasoning_json": str(REASONING_JSON_PATH),
            "summary_json": str(SUMMARY_JSON_PATH),
            "report_md": str(REPORT_MD_PATH),
            "prompt_txt": str(PROMPT_TXT_PATH),
        },
    }

    _write_json(REASONING_JSON_PATH, result)
    _write_json(SUMMARY_JSON_PATH, summary)
    _write_report(BENCHMARK_VIDEO_ID, result, summary)

    print("INVESTIGATION_REASONING_BENCHMARK_SUMMARY_START")
    print(json.dumps(summary, indent=2))
    print("INVESTIGATION_REASONING_BENCHMARK_SUMMARY_END")


if __name__ == "__main__":
    main()
