import asyncio
import json
import logging
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT_PATH = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

if str(os.environ.get("DEBUG", "")).lower() not in {"", "0", "1", "true", "false", "yes", "no", "on", "off"}:
    os.environ["DEBUG"] = "true"

from app.core.config import PROJECT_ROOT, settings
from app.services.frame import FrameExtractionService
from app.services.pipeline_contract import event_catalog_path, frame_catalog_path, frame_metadata_dir
from app.services.status_service import JobStatusService
from tests.experimental_actor_state.actor_state_builder import ActorStateBuilder
from tests.experimental_evidence_graph.evidence_graph_builder import EvidenceGraphBuilder
from tests.experimental_investigation_reasoning.investigation_reasoner import build_evidence_prompt
from tests.experimental_retrieval_investigation_reasoning.retrieval_reasoner import (
    build_current_evidence_payload,
    build_reasoning_prompt,
    compare_reasoning_outputs,
    retrieve_similar_historical_events,
    run_llm_reasoning,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("full_surveillance_intelligence")

DEFAULT_INPUT_VIDEO = r"C:\Mukul K\test_video\V_ai_test_2min.mp4"
INPUT_VIDEO_PATH = Path(os.getenv("BENCHMARK_INPUT_VIDEO", DEFAULT_INPUT_VIDEO))
BENCHMARK_VIDEO_ID = os.getenv("BENCHMARK_VIDEO_ID") or f"full-intel-{uuid.uuid4()}"
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))

CASE_ROOT = PROJECT_ROOT / "tests" / "full_surveillance_intelligence"
DATA_ROOT = CASE_ROOT / "data"
INPUT_ROOT = DATA_ROOT / "input"
OUTPUT_ROOT = DATA_ROOT / "output"
INPUT_COPY_PATH = INPUT_ROOT / INPUT_VIDEO_PATH.name
PROJECT_VIDEO_PATH = settings.VIDEOS_DIR / f"{BENCHMARK_VIDEO_ID}{INPUT_VIDEO_PATH.suffix.lower()}"
PROJECT_METADATA_PATH = settings.METADATA_DIR / f"{BENCHMARK_VIDEO_ID}.json"

ACTOR_TIMELINE_PATH = OUTPUT_ROOT / "actor_state_timeline.json"
EVIDENCE_GRAPH_PATH = OUTPUT_ROOT / "evidence_graph.json"
INVESTIGATION_REASONING_PATH = OUTPUT_ROOT / "investigation_reasoning.json"
RETRIEVAL_CANDIDATES_PATH = OUTPUT_ROOT / "retrieval_candidates.json"
RETRIEVAL_AUGMENTED_REASONING_PATH = OUTPUT_ROOT / "retrieval_augmented_reasoning.json"
SUMMARY_JSON_PATH = OUTPUT_ROOT / "full_surveillance_intelligence_summary.json"
REPORT_MD_PATH = OUTPUT_ROOT / "FULL_SURVEILLANCE_INTELLIGENCE_REPORT.md"
BASELINE_PROMPT_PATH = OUTPUT_ROOT / "investigation_reasoning_prompt.txt"
AUGMENTED_PROMPT_PATH = OUTPUT_ROOT / "retrieval_augmented_reasoning_prompt.txt"


class BenchmarkValidationError(RuntimeError):
    pass


def _ensure_dirs() -> None:
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    settings.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    settings.METADATA_DIR.mkdir(parents=True, exist_ok=True)


def _copy_input_video() -> None:
    if not INPUT_VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Input video not found: {INPUT_VIDEO_PATH}. "
            "Set BENCHMARK_INPUT_VIDEO to a valid local video path."
        )
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
    _safe_remove(settings.METADATA_DIR / f"{BENCHMARK_VIDEO_ID}_status.json")
    _safe_remove(PROJECT_METADATA_PATH)
    for output_path in (
        ACTOR_TIMELINE_PATH,
        EVIDENCE_GRAPH_PATH,
        INVESTIGATION_REASONING_PATH,
        RETRIEVAL_CANDIDATES_PATH,
        RETRIEVAL_AUGMENTED_REASONING_PATH,
        SUMMARY_JSON_PATH,
        REPORT_MD_PATH,
        BASELINE_PROMPT_PATH,
        AUGMENTED_PROMPT_PATH,
    ):
        _safe_remove(output_path)


def _load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def _write_text(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _validate_frames(frames: List[Dict[str, Any]]) -> None:
    if not isinstance(frames, list) or not frames:
        raise BenchmarkValidationError(
            "Frame metadata is missing or empty. "
            f"Expected non-empty frame catalog at {frame_catalog_path(BENCHMARK_VIDEO_ID)}."
        )


def _validate_events(events: List[Dict[str, Any]]) -> None:
    if not isinstance(events, list) or not events:
        raise BenchmarkValidationError(
            "Aggregated events are missing or empty. "
            f"Expected non-empty event catalog at {event_catalog_path(BENCHMARK_VIDEO_ID)}."
        )


def _validate_actor_timeline(actor_timeline: Dict[str, Any]) -> None:
    actors = actor_timeline.get("actors", [])
    if not isinstance(actors, list) or not actors:
        raise BenchmarkValidationError("Actor timelines are empty. ActorStateBuilder produced no global entities.")


def _validate_evidence_graph(evidence_graph: Dict[str, Any]) -> None:
    evidence_units = evidence_graph.get("evidence_units", [])
    nodes = evidence_graph.get("nodes", [])
    edges = evidence_graph.get("edges", [])
    if not evidence_units or not nodes or not edges:
        raise BenchmarkValidationError("Evidence graph is empty. Expected evidence units, nodes, and edges.")


def _validate_reasoning_output(result: Dict[str, Any], label: str) -> None:
    if not isinstance(result, dict):
        raise BenchmarkValidationError(f"{label} output is not a JSON object.")
    required_keys = {
        "summary",
        "important_activities",
        "suspicious_observations",
        "timeline_summary",
        "risk_assessment",
        "supporting_evidence",
    }
    missing = [key for key in required_keys if key not in result]
    if missing:
        raise BenchmarkValidationError(f"{label} output is missing required keys: {missing}")
    if not str(result.get("summary", "")).strip():
        raise BenchmarkValidationError(f"{label} output is empty. Summary text is required.")


def _continuity_statistics(actor_timeline: Dict[str, Any]) -> Dict[str, Any]:
    actors = actor_timeline.get("actors", [])
    total_observations = sum(len(actor.get("timeline", [])) for actor in actors)
    multi_event_actors = sum(
        1
        for actor in actors
        if len({item.get("event_id") for item in actor.get("timeline", [])}) >= 2
    )
    longest = 0.0
    for actor in actors:
        stats = actor.get("continuity_stats", {})
        longest = max(longest, float(stats.get("presence_duration_seconds", 0.0) or 0.0))
    return {
        "total_observations": total_observations,
        "multi_event_actors": multi_event_actors,
        "longest_presence_seconds": round(longest, 2),
    }


def _processing_stats(stats: Dict[str, Any], frames: List[Dict[str, Any]], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    durations = [float(event.get("duration_seconds", 0.0) or 0.0) for event in events]
    return {
        "video_id": BENCHMARK_VIDEO_ID,
        "frame_metadata_count": len(frames),
        "aggregated_event_count": len(events),
        "frames_extracted": int(stats.get("frames_extracted", stats.get("total_frames_extracted", 0)) or 0),
        "frames_retained_for_coverage": int(stats.get("frames_retained_for_coverage", 0) or 0),
        "frames_sent_to_qwen": int(stats.get("frames_sent_to_qwen", 0) or 0),
        "frames_filtered_before_vlm": int(stats.get("frames_filtered_before_vlm", 0) or 0),
        "frames_skipped": int(stats.get("frames_skipped", 0) or 0),
        "reduction_percent": float(stats.get("reduction_percent", 0.0) or 0.0),
        "avg_event_duration_seconds": round(sum(durations) / len(durations), 2) if durations else 0.0,
        "max_event_duration_seconds": round(max(durations), 2) if durations else 0.0,
    }


def _final_narrative(baseline: Dict[str, Any], augmented: Dict[str, Any]) -> str:
    base = str(baseline.get("summary", "")).strip()
    aug = str(augmented.get("summary", "")).strip()
    risk = str(augmented.get("risk_assessment", "")).strip()
    if base and aug and base != aug:
        return f"{aug}\n\nBaseline comparison: {base}\n\nRisk assessment: {risk}"
    if aug:
        return f"{aug}\n\nRisk assessment: {risk}"
    return base


def _write_report(summary: Dict[str, Any]) -> None:
    baseline = summary["investigation_reasoning"]
    augmented = summary["retrieval_augmented_reasoning"]
    retrieved = summary["retrieval_candidates"]
    processing = summary["production_pipeline_statistics"]
    actor_stats = summary["actor_continuity_statistics"]
    evidence_stats = summary["evidence_graph_statistics"]
    retrieval_metrics = summary["retrieval_metrics"]

    lines = [
        "# Full Surveillance Intelligence Report",
        "",
        "## Video Information",
        "",
        f"- Input video: `{summary['input_video_path']}`",
        f"- Input copy: `{summary['input_copy_path']}`",
        f"- Video ID: `{summary['video_id']}`",
        f"- Retrieval top-k: `{summary['retrieval_top_k']}`",
        "",
        "## Production Pipeline Statistics",
        "",
        f"- Frame metadata count: `{processing['frame_metadata_count']}`",
        f"- Aggregated event count: `{processing['aggregated_event_count']}`",
        f"- Frames extracted: `{processing['frames_extracted']}`",
        f"- Frames retained for coverage: `{processing['frames_retained_for_coverage']}`",
        f"- Frames sent to Qwen: `{processing['frames_sent_to_qwen']}`",
        f"- Frames filtered before VLM: `{processing['frames_filtered_before_vlm']}`",
        f"- Frames skipped: `{processing['frames_skipped']}`",
        f"- Sampling reduction: `{processing['reduction_percent']:.2f}%`",
        f"- Average aggregated event duration: `{processing['avg_event_duration_seconds']:.2f}s`",
        f"- Max aggregated event duration: `{processing['max_event_duration_seconds']:.2f}s`",
        "",
        "## Stage Timings",
        "",
        f"- Production extraction pipeline: `{summary['timings']['production_pipeline_seconds']:.2f}s`",
        f"- Actor state builder: `{summary['timings']['actor_state_seconds']:.2f}s`",
        f"- Evidence graph builder: `{summary['timings']['evidence_graph_seconds']:.2f}s`",
        f"- Investigation reasoning: `{summary['timings']['investigation_reasoning_seconds']:.2f}s`",
        f"- Retrieval augmented reasoning: `{summary['timings']['retrieval_augmented_reasoning_seconds']:.2f}s`",
        f"- Total benchmark runtime: `{summary['timings']['total_runtime_seconds']:.2f}s`",
        "",
        "## Candidate and Aggregated Event Statistics",
        "",
        f"- Candidate proxy: frames sent to Qwen = `{processing['frames_sent_to_qwen']}`",
        f"- Aggregated events produced: `{processing['aggregated_event_count']}`",
        "",
        "## Actor Continuity Statistics",
        "",
        f"- Total actors: `{summary['actor_timeline']['summary'].get('total_actors', 0)}`",
        f"- Entity type counts: `{summary['actor_timeline']['summary'].get('entity_type_counts', {})}`",
        f"- Total observations: `{actor_stats['total_observations']}`",
        f"- Multi-event actors: `{actor_stats['multi_event_actors']}`",
        f"- Longest presence: `{actor_stats['longest_presence_seconds']:.2f}s`",
        "",
        "## Evidence Graph Statistics",
        "",
        f"- Evidence units: `{evidence_stats.get('evidence_unit_count', 0)}`",
        f"- Node count: `{evidence_stats.get('node_count', 0)}`",
        f"- Edge count: `{evidence_stats.get('edge_count', 0)}`",
        f"- Actor node count: `{evidence_stats.get('actor_node_count', 0)}`",
        f"- Location node count: `{evidence_stats.get('location_node_count', 0)}`",
        "",
        "## Investigation Summary",
        "",
        augmented.get("summary", ""),
        "",
        "## Important Activities",
        "",
    ]
    for item in augmented.get("important_activities", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Suspicious Observations", ""])
    for item in augmented.get("suspicious_observations", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Risk Assessment", "", augmented.get("risk_assessment", ""), "", "## Supporting Evidence", ""])
    for item in augmented.get("supporting_evidence", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Timeline Summary", ""])
    for item in augmented.get("timeline_summary", []):
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Retrieved Historical Similarities",
            "",
            f"- Retrieved events: `{retrieval_metrics.get('retrieval_count', 0)}`",
            f"- Max similarity: `{retrieval_metrics.get('retrieval_max_similarity', 0.0):.4f}`",
            f"- Avg similarity: `{retrieval_metrics.get('retrieval_avg_similarity', 0.0):.4f}`",
            f"- Retrieval citations used: `{retrieval_metrics.get('augmented_retrieval_citation_count', 0)}`",
            "",
        ]
    )
    for item in retrieved:
        lines.append(
            f"- `[RET:{item['retrieval_id']}]` video=`{item.get('video_id')}` "
            f"event=`{item.get('event_id')}` similarity=`{float(item.get('similarity', 0.0)):.4f}` "
            f"type=`{item.get('event_type')}` description=`{item.get('description')}`"
        )
    lines.extend(["", "## Final Investigation Narrative", "", _final_narrative(baseline, augmented), "", "## Outputs", ""])
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")

    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


async def _run_production_pipeline() -> Dict[str, Any]:
    logger.info("Running production extraction pipeline...")
    return await FrameExtractionService.extract_frames(BENCHMARK_VIDEO_ID)


async def main() -> None:
    os.environ["BENCHMARK_VIDEO_ID"] = BENCHMARK_VIDEO_ID
    overall_start = time.perf_counter()

    _ensure_dirs()
    _clean_previous_artifacts()
    _copy_input_video()
    _write_project_metadata()
    JobStatusService.initialize(BENCHMARK_VIDEO_ID)

    production_start = time.perf_counter()
    stats = await _run_production_pipeline()
    production_seconds = time.perf_counter() - production_start

    frames = _load_json(frame_catalog_path(BENCHMARK_VIDEO_ID), [])
    events = _load_json(event_catalog_path(BENCHMARK_VIDEO_ID), [])
    _validate_frames(frames)
    _validate_events(events)
    logger.info("Aggregated events generated: %s", len(events))

    logger.info("Building actor timelines...")
    actor_start = time.perf_counter()
    actor_timeline = ActorStateBuilder().build(events)
    actor_seconds = time.perf_counter() - actor_start
    _validate_actor_timeline(actor_timeline)
    logger.info("Actor timelines built: %s", len(actor_timeline.get("actors", [])))

    logger.info("Building evidence graph...")
    evidence_start = time.perf_counter()
    evidence_graph = EvidenceGraphBuilder().build(events, actor_timeline)
    evidence_seconds = time.perf_counter() - evidence_start
    _validate_evidence_graph(evidence_graph)
    logger.info("Evidence units generated: %s", len(evidence_graph.get("evidence_units", [])))

    logger.info("Running investigation reasoning...")
    reasoning_start = time.perf_counter()
    baseline_prompt = build_evidence_prompt(actor_timeline, evidence_graph, events)
    baseline_result = run_llm_reasoning(baseline_prompt)
    reasoning_seconds = time.perf_counter() - reasoning_start
    _validate_reasoning_output(baseline_result, "Investigation reasoning")

    logger.info("Running retrieval augmented reasoning...")
    retrieval_start = time.perf_counter()
    current_payload = build_current_evidence_payload(actor_timeline, evidence_graph, events)
    retrieved_events = retrieve_similar_historical_events(
        actor_timeline=actor_timeline,
        evidence_graph=evidence_graph,
        events=events,
        metadata_dir=settings.METADATA_DIR,
        current_video_id=BENCHMARK_VIDEO_ID,
        top_k=RETRIEVAL_TOP_K,
    )
    augmented_prompt = build_reasoning_prompt(current_payload, retrieved_events, include_retrieval=True)
    augmented_result = run_llm_reasoning(augmented_prompt)
    retrieval_seconds = time.perf_counter() - retrieval_start
    _validate_reasoning_output(augmented_result, "Retrieval augmented reasoning")

    retrieval_metrics = compare_reasoning_outputs(
        baseline=baseline_result,
        augmented=augmented_result,
        retrieved_events=retrieved_events,
        baseline_prompt=baseline_prompt,
        augmented_prompt=augmented_prompt,
    )

    logger.info("Persisting benchmark outputs...")
    _write_json(ACTOR_TIMELINE_PATH, actor_timeline)
    _write_json(EVIDENCE_GRAPH_PATH, evidence_graph)
    _write_json(INVESTIGATION_REASONING_PATH, baseline_result)
    _write_json(RETRIEVAL_CANDIDATES_PATH, retrieved_events)
    _write_json(RETRIEVAL_AUGMENTED_REASONING_PATH, augmented_result)
    _write_text(BASELINE_PROMPT_PATH, baseline_prompt)
    _write_text(AUGMENTED_PROMPT_PATH, augmented_prompt)

    summary = {
        "input_video_path": str(INPUT_VIDEO_PATH),
        "input_copy_path": str(INPUT_COPY_PATH),
        "video_id": BENCHMARK_VIDEO_ID,
        "retrieval_top_k": RETRIEVAL_TOP_K,
        "production_pipeline_statistics": _processing_stats(stats, frames, events),
        "actor_timeline": actor_timeline,
        "actor_continuity_statistics": _continuity_statistics(actor_timeline),
        "evidence_graph_statistics": evidence_graph.get("summary", {}),
        "investigation_reasoning": baseline_result,
        "retrieval_candidates": retrieved_events,
        "retrieval_augmented_reasoning": augmented_result,
        "retrieval_metrics": retrieval_metrics,
        "timings": {
            "production_pipeline_seconds": round(production_seconds, 2),
            "actor_state_seconds": round(actor_seconds, 2),
            "evidence_graph_seconds": round(evidence_seconds, 2),
            "investigation_reasoning_seconds": round(reasoning_seconds, 2),
            "retrieval_augmented_reasoning_seconds": round(retrieval_seconds, 2),
            "total_runtime_seconds": round(time.perf_counter() - overall_start, 2),
        },
        "outputs": {
            "frame_catalog": str(frame_catalog_path(BENCHMARK_VIDEO_ID)),
            "event_catalog": str(event_catalog_path(BENCHMARK_VIDEO_ID)),
            "actor_state_timeline": str(ACTOR_TIMELINE_PATH),
            "evidence_graph": str(EVIDENCE_GRAPH_PATH),
            "investigation_reasoning": str(INVESTIGATION_REASONING_PATH),
            "retrieval_candidates": str(RETRIEVAL_CANDIDATES_PATH),
            "retrieval_augmented_reasoning": str(RETRIEVAL_AUGMENTED_REASONING_PATH),
            "final_report": str(REPORT_MD_PATH),
        },
    }
    _write_json(SUMMARY_JSON_PATH, summary)
    _write_report(summary)
    logger.info("Final report generated.")

    print("FULL_SURVEILLANCE_INTELLIGENCE_SUMMARY_START")
    print(json.dumps(summary, indent=2))
    print("FULL_SURVEILLANCE_INTELLIGENCE_SUMMARY_END")


if __name__ == "__main__":
    asyncio.run(main())
