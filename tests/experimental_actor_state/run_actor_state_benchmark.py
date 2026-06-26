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
from tests.experimental_actor_state.actor_state_builder import ActorStateBuilder


BENCHMARK_VIDEO_ID = os.getenv("BENCHMARK_VIDEO_ID", "11111111-2222-4333-8444-555555555555")
CASE_ROOT = PROJECT_ROOT / "tests" / "experimental_actor_state"
DATA_ROOT = CASE_ROOT / "data"
OUTPUT_ROOT = DATA_ROOT / "output"
TIMELINE_JSON_PATH = OUTPUT_ROOT / "actor_state_timeline.json"
SUMMARY_JSON_PATH = OUTPUT_ROOT / "actor_state_summary.json"
REPORT_MD_PATH = OUTPUT_ROOT / "ACTOR_STATE_REPORT.md"


def _ensure_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def _load_events(video_id: str) -> List[Dict[str, Any]]:
    path = event_catalog_path(video_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing aggregated events catalog: {path}. "
            "Run the existing pipeline first so data/metadata/<video_id>_events.json exists."
        )
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of aggregated events in {path}")
    return payload


def _continuity_statistics(timeline_payload: Dict[str, Any]) -> Dict[str, Any]:
    actors = timeline_payload.get("actors", [])
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


def _write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def _write_report(video_id: str, actor_timeline: Dict[str, Any], summary: Dict[str, Any]) -> None:
    lines = [
        "# Actor State Benchmark Report",
        "",
        f"- Video ID: `{video_id}`",
        f"- Aggregated events input: `{event_catalog_path(video_id)}`",
        f"- Total actors: `{summary['total_actors']}`",
        f"- Total observations: `{summary['continuity_statistics']['total_observations']}`",
        f"- Multi-event actors: `{summary['continuity_statistics']['multi_event_actors']}`",
        f"- Longest presence: `{summary['continuity_statistics']['longest_presence_seconds']:.2f}s`",
        f"- Entity type counts: `{summary['entity_type_counts']}`",
        f"- Matched by ID count: `{summary['matched_by_id_count']}`",
        f"- Soft match count: `{summary['soft_match_count']}`",
        "",
        "## Actor Timelines",
        "",
    ]
    for actor in actor_timeline.get("actors", []):
        attributes = actor.get("attributes", {})
        lines.append(
            f"- `{actor['global_actor_id']}` type=`{actor.get('entity_type')}` "
            f"subtype=`{attributes.get('subtype', '')}` color=`{attributes.get('upper_color', '')}` "
            f"events=`{len(actor.get('timeline', []))}`"
        )
    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    _ensure_dirs()
    events = _load_events(BENCHMARK_VIDEO_ID)
    builder = ActorStateBuilder()
    actor_timeline = builder.build(events)
    continuity = _continuity_statistics(actor_timeline)

    summary = {
        "video_id": BENCHMARK_VIDEO_ID,
        "input_event_catalog": str(event_catalog_path(BENCHMARK_VIDEO_ID)),
        "total_events": len(events),
        "total_actors": actor_timeline.get("summary", {}).get("total_actors", 0),
        "entity_type_counts": actor_timeline.get("summary", {}).get("entity_type_counts", {}),
        "matched_by_id_count": actor_timeline.get("summary", {}).get("matched_by_id_count", 0),
        "soft_match_count": actor_timeline.get("summary", {}).get("soft_match_count", 0),
        "continuity_statistics": continuity,
        "outputs": {
            "actor_state_timeline": str(TIMELINE_JSON_PATH),
            "actor_state_summary": str(SUMMARY_JSON_PATH),
            "actor_state_report": str(REPORT_MD_PATH),
        },
    }

    _write_json(TIMELINE_JSON_PATH, actor_timeline)
    _write_json(SUMMARY_JSON_PATH, summary)
    _write_report(BENCHMARK_VIDEO_ID, actor_timeline, summary)

    print("ACTOR_STATE_BENCHMARK_SUMMARY_START")
    print(json.dumps(summary, indent=2))
    print("ACTOR_STATE_BENCHMARK_SUMMARY_END")


if __name__ == "__main__":
    main()
