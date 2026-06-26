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

from app.core.config import PROJECT_ROOT
from app.services.pipeline_contract import event_catalog_path
from tests.experimental_evidence_graph.evidence_graph_builder import EvidenceGraphBuilder


BENCHMARK_VIDEO_ID = os.getenv("BENCHMARK_VIDEO_ID", "11111111-2222-4333-8444-555555555555")
CASE_ROOT = PROJECT_ROOT / "tests" / "experimental_evidence_graph"
DATA_ROOT = CASE_ROOT / "data"
OUTPUT_ROOT = DATA_ROOT / "output"
ACTOR_TIMELINE_PATH = PROJECT_ROOT / "tests" / "experimental_actor_state" / "data" / "output" / "actor_state_timeline.json"
EVIDENCE_GRAPH_PATH = OUTPUT_ROOT / "evidence_graph.json"
SUMMARY_JSON_PATH = OUTPUT_ROOT / "evidence_graph_summary.json"
REPORT_MD_PATH = OUTPUT_ROOT / "EVIDENCE_GRAPH_REPORT.md"


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
        raise ValueError(f"Expected a list of events in {path}")
    return payload


def _load_actor_timelines() -> Dict[str, Any]:
    if not ACTOR_TIMELINE_PATH.exists():
        raise FileNotFoundError(
            f"Missing actor timeline input: {ACTOR_TIMELINE_PATH}. "
            "Run tests\\experimental_actor_state\\run_actor_state_benchmark.py first."
        )
    payload = _load_json(ACTOR_TIMELINE_PATH, {})
    if not isinstance(payload, dict) or "actors" not in payload:
        raise ValueError(f"Invalid actor timeline payload in {ACTOR_TIMELINE_PATH}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def _write_report(video_id: str, graph: Dict[str, Any], summary: Dict[str, Any]) -> None:
    lines = [
        "# Evidence Graph Report",
        "",
        f"- Video ID: `{video_id}`",
        f"- Aggregated events input: `{event_catalog_path(video_id)}`",
        f"- Actor timeline input: `{ACTOR_TIMELINE_PATH}`",
        f"- Evidence units: `{summary['evidence_unit_count']}`",
        f"- Node count: `{summary['node_count']}`",
        f"- Edge count: `{summary['edge_count']}`",
        "",
        "## Evidence Units",
        "",
    ]
    for item in graph.get("evidence_units", [])[:40]:
        lines.append(
            f"- `{item['evidence_id']}` type=`{item['evidence_type']}` actors=`{item['actors']}` "
            f"objects=`{item['objects']}` window=`{item['time_start']:.1f}-{item['time_end']:.1f}` "
            f"description=`{item['description']}`"
        )
    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    _ensure_dirs()
    events = _load_events(BENCHMARK_VIDEO_ID)
    actor_timeline = _load_actor_timelines()

    builder = EvidenceGraphBuilder()
    graph = builder.build(events, actor_timeline)

    summary = {
        "video_id": BENCHMARK_VIDEO_ID,
        "input_event_catalog": str(event_catalog_path(BENCHMARK_VIDEO_ID)),
        "input_actor_timeline": str(ACTOR_TIMELINE_PATH),
        "evidence_unit_count": graph.get("summary", {}).get("evidence_unit_count", 0),
        "node_count": graph.get("summary", {}).get("node_count", 0),
        "edge_count": graph.get("summary", {}).get("edge_count", 0),
        "actor_node_count": graph.get("summary", {}).get("actor_node_count", 0),
        "location_node_count": graph.get("summary", {}).get("location_node_count", 0),
        "outputs": {
            "evidence_graph": str(EVIDENCE_GRAPH_PATH),
            "summary_json": str(SUMMARY_JSON_PATH),
            "report_md": str(REPORT_MD_PATH),
        },
    }

    _write_json(EVIDENCE_GRAPH_PATH, graph)
    _write_json(SUMMARY_JSON_PATH, summary)
    _write_report(BENCHMARK_VIDEO_ID, graph, summary)

    print("EVIDENCE_GRAPH_BENCHMARK_SUMMARY_START")
    print(json.dumps(summary, indent=2))
    print("EVIDENCE_GRAPH_BENCHMARK_SUMMARY_END")


if __name__ == "__main__":
    main()
