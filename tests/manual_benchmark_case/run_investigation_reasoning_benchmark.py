import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT_PATH = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

OUTPUT_ROOT = PROJECT_ROOT_PATH / "tests" / "manual_benchmark_case" / "data" / "output"
ENTITY_TIMELINES_PATH = OUTPUT_ROOT / "entity_timelines.json"
RELATIONSHIP_GRAPH_PATH = OUTPUT_ROOT / "relationship_graph.json"
STATE_CHANGES_PATH = OUTPUT_ROOT / "state_changes.json"
EVIDENCE_PACKAGE_PATH = OUTPUT_ROOT / "evidence_package.json"
EVIDENCE_REASONING_INPUT_ROOT = OUTPUT_ROOT / "reasoning_inputs" / "evidence_builder"
INVESTIGATION_REASONING_INPUT_ROOT = OUTPUT_ROOT / "reasoning_inputs" / "investigation_reasoning"
INVESTIGATION_SUMMARY_JSON_PATH = OUTPUT_ROOT / "investigation_summary.json"
INVESTIGATION_SUMMARY_MD_PATH = OUTPUT_ROOT / "investigation_summary.md"
FRAME_ROOT = PROJECT_ROOT_PATH / "data" / "frames"

MAX_PACKAGES = int(os.getenv("INVESTIGATION_MAX_PACKAGES", "4"))
MAX_KEYFRAMES = int(os.getenv("INVESTIGATION_MAX_KEYFRAMES", "4"))
MAX_TIMELINE_ITEMS = int(os.getenv("INVESTIGATION_MAX_TIMELINE_ITEMS", "16"))


REQUIRED_OUTPUT_KEYS = {
    "incident_summary": str,
    "timeline": list,
    "people_involved": list,
    "vehicles_involved": list,
    "objects_involved": list,
    "confidence": (int, float),
    "evidence_used": list,
    "investigation_recommendation": str,
}


def _load_evidence_package() -> Dict[str, Any]:
    if not EVIDENCE_PACKAGE_PATH.exists():
        raise FileNotFoundError(_missing_dependency_message())
    payload = json.loads(EVIDENCE_PACKAGE_PATH.read_text(encoding="utf-8"))
    packages = payload.get("packages")
    if not isinstance(packages, list):
        raise ValueError("evidence_package.json does not contain a packages list.")
    return payload


def _missing_dependency_message() -> str:
    required_steps = [
        (
            "Phase 1",
            ENTITY_TIMELINES_PATH,
            "python.exe tests\\manual_benchmark_case\\run_entity_timeline_benchmark.py",
        ),
        (
            "Phase 2",
            RELATIONSHIP_GRAPH_PATH,
            "python.exe tests\\manual_benchmark_case\\run_relationship_graph_benchmark.py",
        ),
        (
            "Phase 3",
            STATE_CHANGES_PATH,
            "python.exe tests\\manual_benchmark_case\\run_state_change_benchmark.py",
        ),
        (
            "Phase 4",
            EVIDENCE_PACKAGE_PATH,
            "python.exe tests\\manual_benchmark_case\\run_evidence_builder_benchmark.py",
        ),
    ]
    missing = [(phase, path, command) for phase, path, command in required_steps if not path.exists()]
    lines = [
        f"Missing Phase 5 input: {EVIDENCE_PACKAGE_PATH}.",
        "Phase 5 consumes Phase 4 evidence packages and does not regenerate earlier benchmark artifacts.",
        "Missing prerequisite artifacts:",
    ]
    for phase, path, _command in missing:
        lines.append(f"- {phase}: {path}")
    lines.extend(["Run the benchmark chain in order:", ""])
    lines.extend(command for _phase, _path, command in required_steps)
    return "\n".join(lines)


def _safe_prepare_reasoning_dir() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    resolved = INVESTIGATION_REASONING_INPUT_ROOT.resolve()
    output_resolved = OUTPUT_ROOT.resolve()
    if not str(resolved).startswith(str(output_resolved)):
        raise RuntimeError(f"Refusing to clean unexpected path: {resolved}")
    if INVESTIGATION_REASONING_INPUT_ROOT.exists():
        shutil.rmtree(INVESTIGATION_REASONING_INPUT_ROOT)
    INVESTIGATION_REASONING_INPUT_ROOT.mkdir(parents=True, exist_ok=True)


def _load_reasoning_input_packages(evidence_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    packages = []
    for package in evidence_payload.get("packages", []):
        path_value = package.get("reasoning_input_path")
        path = Path(path_value) if path_value else EVIDENCE_REASONING_INPUT_ROOT / f"{package.get('package_id')}.json"
        if path.exists():
            packages.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            packages.append(package)
    return packages


def _frame_path_for_id(frame_id: str) -> Optional[Path]:
    if "_f" not in frame_id:
        return None
    video_id, frame_suffix = frame_id.rsplit("_f", 1)
    try:
        frame_number = int(frame_suffix)
    except ValueError:
        return None
    path = FRAME_ROOT / video_id / f"frame_{frame_number:04d}.jpg"
    return path if path.exists() else None


def _collect_keyframes(packages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keyframes = []
    seen = set()
    for package in packages:
        for keyframe in package.get("keyframes", []):
            frame_id = keyframe.get("frame_id")
            if not frame_id or frame_id in seen:
                continue
            seen.add(frame_id)
            frame_path = _frame_path_for_id(frame_id)
            keyframes.append(
                {
                    **keyframe,
                    "package_id": package.get("package_id"),
                    "frame_path": str(frame_path) if frame_path else None,
                    "frame_available": frame_path is not None,
                }
            )
            if len(keyframes) >= MAX_KEYFRAMES:
                return keyframes
    return keyframes


def _render_evidence_strip(keyframes: List[Dict[str, Any]]) -> Path:
    width = 336
    height = 189
    panels = []
    selected = keyframes[:MAX_KEYFRAMES] or [
        {
            "frame_id": "no_frame_available",
            "timestamp_human": "00:00:00",
            "selection_rule": ["structured_evidence_only"],
            "frame_path": None,
        }
    ]

    for index, keyframe in enumerate(selected, start=1):
        frame_path = Path(keyframe["frame_path"]) if keyframe.get("frame_path") else None
        image = cv2.imread(str(frame_path)) if frame_path and frame_path.exists() else None
        if image is None:
            image = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(
                image,
                "STRUCTURED EVIDENCE",
                (28, 116),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

        label = f"{index}: {keyframe.get('frame_id')} | {keyframe.get('timestamp_human')}"
        cv2.rectangle(image, (0, 0), (width, 34), (0, 0, 0), thickness=-1)
        cv2.putText(image, label[:72], (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(image)

    rows = []
    for start in range(0, len(panels), 2):
        row_panels = panels[start:start + 2]
        while len(row_panels) < 2:
            row_panels.append(np.zeros((height, width, 3), dtype=np.uint8))
        rows.append(cv2.hconcat(row_panels))
    strip = cv2.vconcat(rows)
    out_path = INVESTIGATION_REASONING_INPUT_ROOT / "investigation_evidence_frames.jpg"
    cv2.imwrite(str(out_path), strip)
    return out_path


def _track_key(track: Dict[str, Any]) -> str:
    if track.get("entity_type") == "zone":
        return f"zone:{track.get('zone', 'unknown')}"
    return f"track:{track.get('track_id')}:{track.get('entity_type')}:{track.get('class_name')}"


def _summarize_tracks(packages: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped = {"people": [], "vehicles": [], "objects": [], "zones": [], "other": []}
    seen = set()
    for package in packages:
        for track in package.get("tracks", []):
            key = _track_key(track)
            if key in seen:
                continue
            seen.add(key)
            entity_type = track.get("entity_type")
            if entity_type == "person":
                grouped["people"].append(track)
            elif entity_type == "vehicle":
                grouped["vehicles"].append(track)
            elif entity_type == "object":
                grouped["objects"].append(track)
            elif entity_type == "zone":
                grouped["zones"].append(track)
            else:
                grouped["other"].append(track)
    return grouped


def _compact_track(track: Dict[str, Any]) -> Dict[str, Any]:
    if track.get("entity_type") == "zone":
        return {"role": track.get("role"), "type": "zone", "zone": track.get("zone", "unknown")}
    return {
        "role": track.get("role"),
        "track_id": track.get("track_id"),
        "type": track.get("entity_type"),
        "class": track.get("class_name"),
    }


def _compact_timeline_item(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": item.get("state_change_id"),
        "t": item.get("timestamp_human"),
        "state": item.get("state"),
        "frames": item.get("frames", [])[:3],
        "reason": item.get("reason"),
        "conf": item.get("confidence"),
    }


def _compact_packages(packages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact = []
    for package in packages[:MAX_PACKAGES]:
        compact.append(
            {
                "id": package.get("package_id"),
                "why": package.get("candidate_reason"),
                "state": _compact_timeline_item(package.get("source_state_change", {})),
                "keyframes": [
                    {
                        "frame": item.get("frame_id"),
                        "t": item.get("timestamp_human"),
                        "rule": item.get("selection_rule"),
                    }
                    for item in package.get("keyframes", [])[:3]
                ],
                "relationships": [
                    item.get("relationship_id")
                    for item in package.get("relationships", [])[:6]
                    if item.get("relationship_id")
                ],
                "tracks": [_compact_track(track) for track in package.get("tracks", [])],
                "ocr": package.get("OCR", {}).get("detected_text", [])[:5],
            }
        )
    return compact


def _timeline_from_packages(packages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    seen = set()
    for package in packages:
        for item in package.get("timeline", []):
            key = item.get("state_change_id")
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "state_change_id": key,
                    "state": item.get("state"),
                    "timestamp_seconds": item.get("timestamp_seconds"),
                    "timestamp_human": item.get("timestamp_human"),
                    "frames": item.get("frames", []),
                    "reason": item.get("reason"),
                    "confidence": item.get("confidence"),
                }
            )
    return sorted(rows, key=lambda item: (float(item.get("timestamp_seconds") or 0.0), str(item.get("state_change_id"))))


def _build_structured_evidence(evidence_payload: Dict[str, Any], packages: List[Dict[str, Any]], keyframes: List[Dict[str, Any]]) -> Dict[str, Any]:
    compact_packages = _compact_packages(packages)
    compact_timeline = [_compact_timeline_item(item) for item in _timeline_from_packages(packages)[:MAX_TIMELINE_ITEMS]]
    tracks = _summarize_tracks(packages[:MAX_PACKAGES])
    return {
        "summary": {
            "package_count": len(packages),
            "selected_package_count": len(compact_packages),
            "keyframe_count": len(keyframes),
            "no_vlm_in_prior_phases": evidence_payload.get("summary", {}).get("no_vlm", True),
        },
        "evidence_package_count": len(packages),
        "packages_used": [package.get("package_id") for package in packages[:MAX_PACKAGES]],
        "timeline": compact_timeline,
        "tracks": {
            "people": [_compact_track(track) for track in tracks["people"]],
            "vehicles": [_compact_track(track) for track in tracks["vehicles"]],
            "objects": [_compact_track(track) for track in tracks["objects"]],
            "zones": [_compact_track(track) for track in tracks["zones"]],
        },
        "keyframes": [
            {
                "frame_id": item.get("frame_id"),
                "timestamp_human": item.get("timestamp_human"),
                "selection_rule": item.get("selection_rule"),
                "frame_available": item.get("frame_available"),
            }
            for item in keyframes
        ],
        "OCR": {
            "available": any(package.get("OCR", {}).get("available") for package in packages),
            "detected_text": [
                text
                for package in packages[:MAX_PACKAGES]
                for text in package.get("OCR", {}).get("detected_text", [])[:5]
            ],
        },
        "relationship_graph_refs": sorted(
            {
                relationship.get("relationship_id")
                for package in packages[:MAX_PACKAGES]
                for relationship in package.get("relationships", [])[:6]
                if relationship.get("relationship_id")
            }
        ),
        "state_changes": [package["state"] for package in compact_packages],
        "compact_evidence_packages": compact_packages,
    }


def _build_prompt(structured_evidence: Dict[str, Any]) -> str:
    facts_json = json.dumps(structured_evidence, separators=(",", ":"), ensure_ascii=False)
    return (
        "You are an investigation assistant.\n"
        "You are provided evidence frames, entity histories, relationship graph references, timeline, OCR, and state changes.\n"
        "Qwen must NOT detect objects, people, vehicles, text, actions, or scene contents from the images.\n"
        "Use the images only as evidence-frame context. Base every conclusion only on the structured evidence JSON below.\n"
        f"STRUCTURED_EVIDENCE_JSON={facts_json}\n"
        "Return strict JSON only. No markdown. No commentary.\n"
        "Schema:\n"
        "{"
        "\"incident_summary\":\"string\","
        "\"timeline\":[{\"timestamp\":\"HH:MM:SS\",\"state\":\"string\",\"description\":\"string\",\"evidence_ids\":[\"string\"]}],"
        "\"people_involved\":[{\"track_id\":number,\"role\":\"string\",\"evidence_ids\":[\"string\"]}],"
        "\"vehicles_involved\":[{\"track_id\":number,\"role\":\"string\",\"evidence_ids\":[\"string\"]}],"
        "\"objects_involved\":[{\"track_id\":number,\"class_name\":\"string\",\"role\":\"string\",\"evidence_ids\":[\"string\"]}],"
        "\"confidence\":0.0,"
        "\"evidence_used\":[{\"evidence_id\":\"string\",\"frames\":[\"string\"],\"tracks\":[\"string\"],\"reason\":\"string\"}],"
        "\"investigation_recommendation\":\"string\""
        "}\n"
        "If evidence is insufficient, say so in incident_summary and lower confidence. Do not invent missing detections."
    )


def _parse_qwen_json(raw_output: str) -> Tuple[Optional[Dict[str, Any]], str, Optional[str]]:
    from app.services.vlm_utils import clean_json_response

    try:
        cleaned = clean_json_response(raw_output)
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed, cleaned, None
        return None, cleaned, "parsed_output_not_dict"
    except Exception as exc:
        return None, raw_output, str(exc)


def _normalize_summary(parsed: Optional[Dict[str, Any]], fallback_evidence: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(parsed, dict):
        parsed = {}

    normalized: Dict[str, Any] = {}
    for key, expected_type in REQUIRED_OUTPUT_KEYS.items():
        value = parsed.get(key)
        if key == "confidence":
            try:
                normalized[key] = max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                normalized[key] = 0.0
        elif not isinstance(value, expected_type):
            normalized[key] = [] if expected_type is list else ""
        else:
            normalized[key] = value

    if not normalized["timeline"]:
        normalized["timeline"] = [
            {
                "timestamp": item.get("timestamp_human") or item.get("t"),
                "state": item.get("state"),
                "description": item.get("reason"),
                "evidence_ids": [item.get("state_change_id") or item.get("id")],
            }
            for item in fallback_evidence.get("timeline", [])[:12]
        ]
    if not normalized["evidence_used"]:
        normalized["evidence_used"] = [
            {
                "evidence_id": package.get("package_id") or package.get("id"),
                "frames": [frame.get("frame_id") or frame.get("frame") for frame in package.get("keyframes", [])],
                "tracks": [str(track) for track in package.get("tracks", [])],
                "reason": package.get("candidate_reason") or package.get("why"),
            }
            for package in fallback_evidence.get("compact_evidence_packages", [])[:12]
        ]
    if not normalized["incident_summary"]:
        normalized["incident_summary"] = "Qwen did not return a valid incident summary; structured evidence is preserved for review."
    if not normalized["investigation_recommendation"]:
        normalized["investigation_recommendation"] = "Review the listed evidence frames and state changes manually."
    return normalized


async def _run_qwen_reasoning(image_path: Path, prompt: str) -> Dict[str, Any]:
    from app.core.config import settings
    from app.services.qwen_vlm_hf import NativeQwenTransformersService

    old_batch_size = settings.BATCH_SIZE
    runtime_seconds = 0.0
    try:
        settings.BATCH_SIZE = 1
        start = time.perf_counter()
        raw_outputs = await NativeQwenTransformersService._async_hf_generate([image_path], [prompt])
        runtime_seconds = time.perf_counter() - start
    except Exception as exc:
        runtime_seconds = time.perf_counter() - start if "start" in locals() else 0.0
        return {
            "raw_output": "",
            "cleaned_output": "",
            "parsed_output": None,
            "parse_error": f"{type(exc).__name__}: {exc}",
            "success": False,
            "runtime_seconds": runtime_seconds,
        }
    finally:
        settings.BATCH_SIZE = old_batch_size

    raw_output = raw_outputs[0] if raw_outputs else ""
    parsed, cleaned, error = _parse_qwen_json(raw_output)
    return {
        "raw_output": raw_output,
        "cleaned_output": cleaned,
        "parsed_output": parsed,
        "parse_error": error,
        "success": isinstance(parsed, dict) and error is None,
        "runtime_seconds": runtime_seconds,
    }


def _write_markdown(summary_payload: Dict[str, Any]) -> None:
    result = summary_payload["investigation_summary"]
    lines = [
        "# Investigation Summary",
        "",
        f"- Confidence: `{result.get('confidence')}`",
        f"- Qwen JSON success: `{summary_payload['qwen_result']['success']}`",
        f"- Evidence packages used: `{len(summary_payload['structured_evidence']['packages_used'])}`",
        f"- Evidence frames used: `{len(summary_payload['structured_evidence']['keyframes'])}`",
        "",
        "## Incident Summary",
        "",
        str(result.get("incident_summary", "")),
        "",
        "## Timeline",
        "",
    ]
    for item in result.get("timeline", []):
        lines.append(
            f"- `{item.get('timestamp')}` {item.get('state')}: {item.get('description')} "
            f"(evidence={item.get('evidence_ids')})"
        )
    lines.extend(["", "## People Involved", ""])
    lines.extend(_entity_lines(result.get("people_involved", []), "track_id"))
    lines.extend(["", "## Vehicles Involved", ""])
    lines.extend(_entity_lines(result.get("vehicles_involved", []), "track_id"))
    lines.extend(["", "## Objects Involved", ""])
    lines.extend(_entity_lines(result.get("objects_involved", []), "track_id"))
    lines.extend(["", "## Evidence Used", ""])
    for item in result.get("evidence_used", []):
        lines.append(f"- `{item.get('evidence_id')}` frames={item.get('frames')} tracks={item.get('tracks')} reason={item.get('reason')}")
    lines.extend(["", "## Recommendation", "", str(result.get("investigation_recommendation", ""))])
    INVESTIGATION_SUMMARY_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def _entity_lines(rows: List[Dict[str, Any]], id_key: str) -> List[str]:
    if not rows:
        return ["- None"]
    return [f"- `{row.get(id_key)}` role={row.get('role')} evidence={row.get('evidence_ids')}" for row in rows]


async def main() -> None:
    start = time.perf_counter()
    _safe_prepare_reasoning_dir()
    evidence_payload = _load_evidence_package()
    packages = _load_reasoning_input_packages(evidence_payload)
    keyframes = _collect_keyframes(packages)
    image_path = _render_evidence_strip(keyframes)
    structured_evidence = _build_structured_evidence(evidence_payload, packages, keyframes)
    prompt = _build_prompt(structured_evidence)

    prompt_path = INVESTIGATION_REASONING_INPUT_ROOT / "investigation_prompt.json"
    prompt_path.write_text(
        json.dumps(
            {
                "prompt": prompt,
                "structured_evidence": structured_evidence,
                "evidence_frame_strip": str(image_path),
                "instruction": "Qwen must reason only from structured evidence and must not detect.",
            },
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    qwen_result = await _run_qwen_reasoning(image_path, prompt)
    investigation_summary = _normalize_summary(qwen_result["parsed_output"], structured_evidence)
    summary_payload = {
        "benchmark": "investigation_reasoning_phase_5",
        "input": {
            "evidence_package": str(EVIDENCE_PACKAGE_PATH),
            "reasoning_inputs": str(EVIDENCE_REASONING_INPUT_ROOT),
        },
        "output": {
            "investigation_summary_json": str(INVESTIGATION_SUMMARY_JSON_PATH),
            "investigation_summary_md": str(INVESTIGATION_SUMMARY_MD_PATH),
            "reasoning_prompt": str(prompt_path),
            "evidence_frame_strip": str(image_path),
        },
        "summary": {
            "no_detection_instruction": True,
            "qwen_role": "reasoning_only",
            "wall_clock_runtime_seconds": time.perf_counter() - start,
            "packages_used": len(structured_evidence["packages_used"]),
            "keyframes_used": len(structured_evidence["keyframes"]),
            "qwen_success": qwen_result["success"],
        },
        "structured_evidence": structured_evidence,
        "investigation_summary": investigation_summary,
        "qwen_result": qwen_result,
    }
    INVESTIGATION_SUMMARY_JSON_PATH.write_text(json.dumps(summary_payload, indent=4, ensure_ascii=False), encoding="utf-8")
    _write_markdown(summary_payload)

    print("INVESTIGATION_REASONING_BENCHMARK_START")
    print(
        json.dumps(
            {
                "investigation_summary_json": str(INVESTIGATION_SUMMARY_JSON_PATH),
                "investigation_summary_md": str(INVESTIGATION_SUMMARY_MD_PATH),
                "reasoning_inputs": str(INVESTIGATION_REASONING_INPUT_ROOT),
                "qwen_success": qwen_result["success"],
            }
        )
    )
    print("INVESTIGATION_REASONING_BENCHMARK_END")


if __name__ == "__main__":
    asyncio.run(main())
