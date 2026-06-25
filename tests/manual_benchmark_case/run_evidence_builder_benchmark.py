import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT_PATH = SCRIPT_PATH.parents[2]
OUTPUT_ROOT = PROJECT_ROOT_PATH / "tests" / "manual_benchmark_case" / "data" / "output"

STATE_CHANGES_PATH = OUTPUT_ROOT / "state_changes.json"
EVIDENCE_PACKAGE_PATH = OUTPUT_ROOT / "evidence_package.json"
REASONING_INPUT_ROOT = OUTPUT_ROOT / "reasoning_inputs" / "evidence_builder"

TIMELINE_CONTEXT_SECONDS = 5.0
MAX_KEYFRAMES_PER_PACKAGE = 8


ENTITY_APPEAR_STATES = {
    "zone_entered",
    "person_exits_vehicle",
}
ENTITY_DISAPPEAR_STATES = {
    "zone_left",
    "person_enters_vehicle",
    "walking_together_ended",
    "following_ended",
    "meeting_ended",
    "person_no_longer_near_object",
    "person_no_longer_touching_object",
    "vehicle_no_longer_confirmed_parked",
    "object_stationary_interval_ended",
}
POSE_CHANGE_STATES = {
    "walking",
    "standing",
    "following",
}
RELATIONSHIP_CHANGE_STATES = {
    "proximal_distance_observed",
    "person_near_object",
    "person_touch_object",
    "person_leaves_object",
    "person_no_longer_near_object",
    "person_no_longer_touching_object",
    "meeting_ended",
    "walking_together_ended",
    "following_ended",
}
OBJECT_CHANGE_STATES = {
    "person_touch_object",
    "object_picked_or_moved",
    "object_loaded_into_vehicle",
    "object_inside_zone",
    "object_outside_zone",
    "object_stationary",
    "object_stationary_interval_ended",
}
ZONE_CHANGE_STATES = {
    "zone_entered",
    "zone_left",
    "vehicle_parked",
    "vehicle_no_longer_confirmed_parked",
    "object_inside_zone",
    "object_outside_zone",
    "object_stationary",
}


def _load_state_changes() -> Dict[str, Any]:
    if not STATE_CHANGES_PATH.exists():
        raise FileNotFoundError(
            f"Missing Phase 3 input: {STATE_CHANGES_PATH}. "
            "Run run_state_change_benchmark.py first."
        )
    payload = json.loads(STATE_CHANGES_PATH.read_text(encoding="utf-8"))
    state_changes = payload.get("state_changes")
    if not isinstance(state_changes, list):
        raise ValueError("state_changes.json does not contain a state_changes list.")
    return payload


def _safe_prepare_reasoning_dir() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    resolved = REASONING_INPUT_ROOT.resolve()
    output_resolved = OUTPUT_ROOT.resolve()
    if not str(resolved).startswith(str(output_resolved)):
        raise RuntimeError(f"Refusing to clean unexpected path: {resolved}")
    if REASONING_INPUT_ROOT.exists():
        shutil.rmtree(REASONING_INPUT_ROOT)
    REASONING_INPUT_ROOT.mkdir(parents=True, exist_ok=True)


def _track_key(track: Dict[str, Any]) -> Optional[str]:
    if track.get("entity_type") == "zone":
        return f"zone:{track.get('zone', 'unknown')}"
    track_id = track.get("track_id")
    if track_id is None:
        return None
    return f"track:{track_id}"


def _state_track_keys(change: Dict[str, Any]) -> Set[str]:
    return {
        key
        for key in (_track_key(track) for track in change.get("supporting_tracks", []))
        if key
    }


def _timestamp_seconds(change: Dict[str, Any]) -> float:
    return float(change.get("timestamp", {}).get("seconds", 0.0))


def _state_categories(state: str) -> List[str]:
    categories = []
    if state in ENTITY_APPEAR_STATES:
        categories.append("entity_appears")
    if state in ENTITY_DISAPPEAR_STATES:
        categories.append("entity_disappears")
    if state in POSE_CHANGE_STATES or "_to_walking" in state or "walking_to_" in state or "_to_standing" in state or "standing_to_" in state:
        categories.append("pose_changes")
    if state in RELATIONSHIP_CHANGE_STATES or "_to_" in state:
        categories.append("relationship_changes")
    if state in OBJECT_CHANGE_STATES or "object_" in state:
        categories.append("object_changes")
    if state in ZONE_CHANGE_STATES or "zone_" in state:
        categories.append("zone_changes")
    return categories


def _candidate_reason(change: Dict[str, Any]) -> str:
    categories = _state_categories(str(change.get("state", "")))
    if not categories:
        return "not_selected_no_evidence_frame_rule_matched"
    return ", ".join(categories)


def _is_evidence_trigger(change: Dict[str, Any]) -> bool:
    return bool(_state_categories(str(change.get("state", ""))))


def _related_timeline(change: Dict[str, Any], all_changes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    center_time = _timestamp_seconds(change)
    center_keys = _state_track_keys(change)
    rows = []
    for candidate in all_changes:
        if abs(_timestamp_seconds(candidate) - center_time) > TIMELINE_CONTEXT_SECONDS:
            continue
        candidate_keys = _state_track_keys(candidate)
        if center_keys and candidate_keys and not center_keys.intersection(candidate_keys):
            continue
        rows.append(_compact_state_change(candidate))
    return sorted(rows, key=lambda item: (item["timestamp_seconds"], item["state_change_id"]))


def _compact_state_change(change: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "state_change_id": change.get("state_change_id"),
        "state": change.get("state"),
        "timestamp_seconds": _timestamp_seconds(change),
        "timestamp_human": change.get("timestamp", {}).get("human"),
        "frames": change.get("frames", []),
        "reason": change.get("reason"),
        "confidence": change.get("confidence"),
        "source_relationship_ids": change.get("source_relationship_ids", []),
    }


def _selected_keyframes(timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keyframes = []
    seen = set()
    for item in timeline:
        categories = _state_categories(str(item.get("state", "")))
        if not categories:
            continue
        for frame_id in item.get("frames", []):
            key = (frame_id, item.get("state_change_id"))
            if key in seen:
                continue
            seen.add(key)
            keyframes.append(
                {
                    "frame_id": frame_id,
                    "timestamp_seconds": item.get("timestamp_seconds"),
                    "timestamp_human": item.get("timestamp_human"),
                    "state_change_id": item.get("state_change_id"),
                    "selection_rule": categories,
                    "candidate_reason": item.get("reason"),
                    "selected_because": "frame belongs to a state change matching evidence-frame rules",
                }
            )
            if len(keyframes) >= MAX_KEYFRAMES_PER_PACKAGE:
                return keyframes
    return keyframes


def _relationships(change: Dict[str, Any], timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    relationship_ids = []
    for item in timeline:
        relationship_ids.extend(item.get("source_relationship_ids", []))
    relationship_ids.extend(change.get("source_relationship_ids", []))
    return [
        {
            "relationship_id": relationship_id,
            "source": "state_changes.source_relationship_ids",
        }
        for relationship_id in dict.fromkeys(relationship_ids)
        if relationship_id
    ]


def _tracks(change: Dict[str, Any], timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    seen = set()
    for track in change.get("supporting_tracks", []):
        key = json.dumps(track, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        rows.append(track)
    return rows


def _ocr_package() -> Dict[str, Any]:
    return {
        "available": False,
        "detected_text": [],
        "license_plates": [],
        "reason": "state_changes.json does not carry OCR text; Phase 4 uses only the requested input and does not run OCR or VLM.",
    }


def _build_package(index: int, change: Dict[str, Any], all_changes: List[Dict[str, Any]]) -> Dict[str, Any]:
    timeline = _related_timeline(change, all_changes)
    keyframes = _selected_keyframes(timeline)
    package_id = f"evidence_pkg_{index:05d}"
    return {
        "package_id": package_id,
        "state_change_id": change.get("state_change_id"),
        "candidate_reason": _candidate_reason(change),
        "timeline": timeline,
        "keyframes": keyframes,
        "relationships": _relationships(change, timeline),
        "tracks": _tracks(change, timeline),
        "OCR": _ocr_package(),
        "source_state_change": _compact_state_change(change),
        "selection_policy": {
            "selected_representative_frames": False,
            "selected_only_evidence_frames": True,
            "allowed_frame_reasons": [
                "entity_appears",
                "entity_disappears",
                "pose_changes",
                "relationship_changes",
                "object_changes",
                "zone_changes",
            ],
            "no_vlm": True,
        },
    }


def _build_evidence_payload(state_payload: Dict[str, Any]) -> Dict[str, Any]:
    state_changes = sorted(
        state_payload.get("state_changes", []),
        key=lambda item: (_timestamp_seconds(item), str(item.get("state_change_id", ""))),
    )
    packages = [
        _build_package(index, change, state_changes)
        for index, change in enumerate(state_changes, start=1)
        if _is_evidence_trigger(change)
    ]

    by_reason: Dict[str, int] = {}
    for package in packages:
        for reason in [item.strip() for item in package["candidate_reason"].split(",")]:
            by_reason[reason] = by_reason.get(reason, 0) + 1

    return {
        "benchmark": "evidence_builder_phase_4",
        "input": str(STATE_CHANGES_PATH),
        "output": str(EVIDENCE_PACKAGE_PATH),
        "reasoning_inputs_folder": str(REASONING_INPUT_ROOT),
        "summary": {
            "state_change_count": len(state_changes),
            "evidence_package_count": len(packages),
            "keyframe_count": sum(len(package["keyframes"]) for package in packages),
            "candidate_reason_count": dict(sorted(by_reason.items())),
            "no_vlm": True,
            "method": "deterministic_evidence_frame_selection_from_state_changes",
        },
        "packages": packages,
    }


def _write_reasoning_inputs(packages: List[Dict[str, Any]]) -> None:
    for package in packages:
        path = REASONING_INPUT_ROOT / f"{package['package_id']}.json"
        package["reasoning_input_path"] = str(path)
        path.write_text(json.dumps(package, indent=4), encoding="utf-8")


def main() -> None:
    start = time.perf_counter()
    _safe_prepare_reasoning_dir()
    state_payload = _load_state_changes()
    evidence_payload = _build_evidence_payload(state_payload)
    _write_reasoning_inputs(evidence_payload["packages"])
    evidence_payload["summary"]["wall_clock_runtime_seconds"] = time.perf_counter() - start
    EVIDENCE_PACKAGE_PATH.write_text(json.dumps(evidence_payload, indent=4), encoding="utf-8")

    print("EVIDENCE_BUILDER_BENCHMARK_START")
    print(
        json.dumps(
            {
                "evidence_package": str(EVIDENCE_PACKAGE_PATH),
                "reasoning_inputs": str(REASONING_INPUT_ROOT),
            }
        )
    )
    print("EVIDENCE_BUILDER_BENCHMARK_END")


if __name__ == "__main__":
    main()
