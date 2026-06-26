import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT_PATH = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from app.core.config import PROJECT_ROOT, settings
from app.services.frame import FrameExtractionService
from app.services.pipeline_contract import event_catalog_path, frame_catalog_path, frame_metadata_dir
from app.services.status_service import JobStatusService
from tests.experimental_surveillance_reasoning.actor_state_builder import build_actor_states
from tests.experimental_surveillance_reasoning.evidence_graph_builder import build_evidence_graph
from tests.experimental_surveillance_reasoning.investigation_reasoner import reason_over_surveillance


INPUT_VIDEO_PATH = Path(os.getenv("BENCHMARK_INPUT_VIDEO", r"C:\Mukul K\test_video\V_ai_test_2min.mp4"))
BENCHMARK_VIDEO_ID = os.getenv("BENCHMARK_VIDEO_ID", "22222222-3333-4444-8555-999999999999")
CASE_ROOT = PROJECT_ROOT / "tests" / "experimental_surveillance_reasoning"
DATA_ROOT = CASE_ROOT / "data"
INPUT_ROOT = DATA_ROOT / "input"
OUTPUT_ROOT = DATA_ROOT / "output"
PROJECT_VIDEO_PATH = settings.VIDEOS_DIR / f"{BENCHMARK_VIDEO_ID}{INPUT_VIDEO_PATH.suffix.lower()}"
PROJECT_METADATA_PATH = settings.METADATA_DIR / f"{BENCHMARK_VIDEO_ID}.json"
INPUT_COPY_PATH = INPUT_ROOT / INPUT_VIDEO_PATH.name
ACTOR_STATES_PATH = OUTPUT_ROOT / "actor_states.json"
EVIDENCE_GRAPH_PATH = OUTPUT_ROOT / "evidence_graph.json"
INVESTIGATION_REASONING_PATH = OUTPUT_ROOT / "investigation_reasoning.json"
SUMMARY_JSON_PATH = OUTPUT_ROOT / "experimental_surveillance_summary.json"
SUMMARY_MD_PATH = OUTPUT_ROOT / "experimental_surveillance_summary.md"


def _ensure_dirs() -> None:
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def _copy_input_video() -> None:
    if not INPUT_VIDEO_PATH.exists():
        raise FileNotFoundError(f"Input video not found: {INPUT_VIDEO_PATH}")
    if INPUT_VIDEO_PATH.resolve() != INPUT_COPY_PATH.resolve():
        shutil.copy2(INPUT_VIDEO_PATH, INPUT_COPY_PATH)
    if INPUT_VIDEO_PATH.resolve() != PROJECT_VIDEO_PATH.resolve():
        shutil.copy2(INPUT_VIDEO_PATH, PROJECT_VIDEO_PATH)


def _write_project_metadata() -> None:
    metadata = {
        "video_id": BENCHMARK_VIDEO_ID,
        "filename": INPUT_VIDEO_PATH.name,
        "upload_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "file_size": PROJECT_VIDEO_PATH.stat().st_size,
        "upload_duration_ms": 0.0,
    }
    with open(PROJECT_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def _safe_remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


def _clean_previous_artifacts() -> None:
    _safe_remove(settings.FRAMES_DIR / BENCHMARK_VIDEO_ID)
    _safe_remove(frame_metadata_dir(BENCHMARK_VIDEO_ID))
    _safe_remove(settings.EVENTS_DIR / BENCHMARK_VIDEO_ID)
    _safe_remove(frame_catalog_path(BENCHMARK_VIDEO_ID))
    _safe_remove(event_catalog_path(BENCHMARK_VIDEO_ID))
    _safe_remove(PROJECT_METADATA_PATH)
    for output_path in (
        ACTOR_STATES_PATH,
        EVIDENCE_GRAPH_PATH,
        INVESTIGATION_REASONING_PATH,
        SUMMARY_JSON_PATH,
        SUMMARY_MD_PATH,
    ):
        _safe_remove(output_path)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


async def run_event_candidate_layer() -> Dict[str, Any]:
    return await FrameExtractionService.extract_frames(BENCHMARK_VIDEO_ID)


def _write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def _write_summary(summary: Dict[str, Any]) -> None:
    _write_json(SUMMARY_JSON_PATH, summary)
    lines: List[str] = [
        "# Experimental Surveillance Reasoning Summary",
        "",
        f"- Input video: `{summary['input_video_path']}`",
        f"- Input copy: `{summary['input_copy_path']}`",
        f"- Video ID: `{summary['video_id']}`",
        f"- Existing pipeline runtime: `{summary['existing_pipeline_runtime_seconds']:.2f}s`",
        f"- Existing frame metadata count: `{summary['frame_metadata_count']}`",
        f"- Existing aggregated event count: `{summary['aggregated_event_count']}`",
        f"- Actor states: `{summary['actor_state_count']}`",
        f"- Evidence graph nodes: `{summary['evidence_graph_nodes']}`",
        f"- Evidence graph edges: `{summary['evidence_graph_edges']}`",
        f"- Investigation hypotheses: `{summary['investigation_hypothesis_count']}`",
        "",
        "## Outputs",
        "",
        f"- Actor states: `{ACTOR_STATES_PATH}`",
        f"- Evidence graph: `{EVIDENCE_GRAPH_PATH}`",
        f"- Investigation reasoning: `{INVESTIGATION_REASONING_PATH}`",
    ]
    for finding in summary.get("top_hypotheses", []):
        lines.append(
            f"- Hypothesis `{finding['hypothesis_type']}` confidence=`{finding['confidence']:.2f}` events=`{finding['event_ids']}`"
        )
    with open(SUMMARY_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


async def main() -> None:
    _ensure_dirs()
    _clean_previous_artifacts()
    _copy_input_video()
    _write_project_metadata()
    JobStatusService.initialize(BENCHMARK_VIDEO_ID)

    start = time.perf_counter()
    _stats = await run_event_candidate_layer()
    pipeline_runtime_seconds = time.perf_counter() - start

    frames = _load_json(frame_catalog_path(BENCHMARK_VIDEO_ID), [])
    events = _load_json(event_catalog_path(BENCHMARK_VIDEO_ID), [])

    actor_states = build_actor_states(frames, events)
    evidence_graph = build_evidence_graph(frames, events, actor_states)
    investigation_reasoning = reason_over_surveillance(frames, events, actor_states, evidence_graph)

    _write_json(ACTOR_STATES_PATH, actor_states)
    _write_json(EVIDENCE_GRAPH_PATH, evidence_graph)
    _write_json(INVESTIGATION_REASONING_PATH, investigation_reasoning)

    summary = {
        "input_video_path": str(INPUT_VIDEO_PATH),
        "input_copy_path": str(INPUT_COPY_PATH),
        "video_id": BENCHMARK_VIDEO_ID,
        "existing_pipeline_runtime_seconds": pipeline_runtime_seconds,
        "frame_metadata_count": len(frames),
        "aggregated_event_count": len(events),
        "actor_state_count": len(actor_states.get("actors", [])),
        "evidence_graph_nodes": evidence_graph.get("summary", {}).get("node_count", 0),
        "evidence_graph_edges": evidence_graph.get("summary", {}).get("edge_count", 0),
        "investigation_hypothesis_count": len(investigation_reasoning.get("hypotheses", [])),
        "top_hypotheses": investigation_reasoning.get("hypotheses", [])[:8],
        "outputs": {
            "actor_states": str(ACTOR_STATES_PATH),
            "evidence_graph": str(EVIDENCE_GRAPH_PATH),
            "investigation_reasoning": str(INVESTIGATION_REASONING_PATH),
            "frame_catalog": str(frame_catalog_path(BENCHMARK_VIDEO_ID)),
            "event_catalog": str(event_catalog_path(BENCHMARK_VIDEO_ID)),
        },
    }
    _write_summary(summary)

    print("EXPERIMENTAL_SURVEILLANCE_REASONING_SUMMARY_START")
    print(json.dumps(summary, indent=2))
    print("EXPERIMENTAL_SURVEILLANCE_REASONING_SUMMARY_END")


if __name__ == "__main__":
    asyncio.run(main())
