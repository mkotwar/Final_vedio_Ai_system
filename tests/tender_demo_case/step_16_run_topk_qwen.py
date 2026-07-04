from __future__ import annotations

import importlib.util
import json
import os
import re
from pathlib import Path
from typing import Any


DEFAULT_FAST_COMPACT_QWEN_SCHEMA = True


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_required_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Top-K Qwen input file: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = payload.get("items")
        if not isinstance(items, list):
            raise ValueError("Expected an 'items' list in 15_topk_vlm_inputs.json")
        return items
    if isinstance(payload, list):
        return payload
    raise ValueError("15_topk_vlm_inputs.json must contain either a list or an object with an 'items' list.")


def _load_tender_demo_vlm_factory():
    try:
        from tests.tender_demo_case.tender_demo_vlm_adapter import create_tender_demo_vlm
        return create_tender_demo_vlm
    except ModuleNotFoundError:
        adapter_path = Path(__file__).resolve().parent / "tender_demo_vlm_adapter.py"
        spec = importlib.util.spec_from_file_location("tender_demo_vlm_adapter", adapter_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load tender demo VLM adapter from: {adapter_path}")
        adapter_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter_module)
        return adapter_module.create_tender_demo_vlm


def _record_identity(item: dict[str, Any]) -> str:
    preferred_keys = [
        "topk_vlm_input_id",
        "input_id",
        "topk_vlm_output_id",
    ]
    for key in preferred_keys:
        value = str(item.get(key, "")).strip()
        if value:
            return value
    clip_id = str(item.get("source_clip_id", "")).strip()
    current_time = str(item.get("current_time", "")).strip()
    if clip_id and current_time:
        return f"{clip_id}@{current_time}"
    return clip_id


def _resolve_media_path(path_value: Any) -> Path | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        return path if path.exists() else None
    candidate = _repo_root() / path
    if candidate.exists():
        return candidate
    return path if path.exists() else None


def _load_existing_raw_outputs(output_path: Path) -> dict[str, str]:
    if not output_path.exists():
        return {}
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    items = payload.get("items", []) if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return {}
    raw_by_record_id: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        record_id = _record_identity(item)
        raw_text = str(item.get("raw_vlm_output", "") or "")
        if record_id and raw_text.strip():
            raw_by_record_id[record_id] = raw_text
    return raw_by_record_id


def _safe_bool_env(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _read_prompt_override_text() -> str | None:
    raw_value = os.environ.get("TENDER_DEMO_STEP16_PROMPT_FILE", "").strip()
    if not raw_value:
        return None

    candidate = Path(raw_value).expanduser()
    if not candidate.is_absolute():
        candidate = (_repo_root() / candidate).resolve()

    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(f"Step 16 prompt override file does not exist: {candidate}")

    prompt_text = candidate.read_text(encoding="utf-8").strip()
    if not prompt_text:
        raise ValueError(f"Step 16 prompt override file is empty: {candidate}")
    return prompt_text


def _compact_step16_prompt() -> str:
    return """You analyze CCTV, surveillance, shop, office, road, parking, warehouse, public-area, and traffic imagery.

The image is a temporal strip:

PREVIOUS | CURRENT | NEXT

Focus on the CURRENT panel.
Use PREVIOUS and NEXT only to understand what changed.

Return ONLY one valid JSON object.
No markdown.
No code block.
No explanation.
Use double quotes only.
Do not use trailing commas.
Keep every value short.
Arrays must contain short strings only.
Do not return nested person/object arrays.

JSON schema:
{
  "scene_type": "shop|office|road|street|parking|warehouse|public_area|indoor|outdoor|unknown",
  "caption": "short factual sentence about CURRENT panel",
  "people_count": 0,
  "vehicle_count": 0,
  "main_objects": ["person"],
  "main_activities": ["walking"],
  "main_interactions": ["person near counter"],
  "traffic_state": "moving_traffic|stopped_vehicle|parked_vehicle|pedestrian_activity|mixed_motion|unclear",

  "weapon_like_object_visible": "yes|no|unclear",
  "grabbing_or_restraining_visible": "yes|no|unclear",
  "threat_or_control_posture_visible": "yes|no|unclear",
  "fall_or_person_down_visible": "yes|no|unclear",
  "fight_or_assault_visible": "yes|no|unclear",
  "object_taken_or_removed_visible": "yes|no|unclear",
  "display_or_counter_interaction": "yes|no|unclear",
  "intrusion_or_boundary_crossing": "yes|no|unclear",
  "abandoned_object_visible": "yes|no|unclear",
  "collision_or_near_miss_visible": "yes|no|unclear",
  "fire_smoke_or_hazard_visible": "yes|no|unclear",

  "suspicious_activity": "yes|no|unclear",
  "primary_event_label": "normal_activity|possible_theft|possible_robbery|possible_weapon_visible|possible_assault_or_grabbing|possible_fight|possible_fall|possible_intrusion|possible_abandoned_object|possible_collision|possible_traffic_violation|possible_fire_or_hazard|traffic_activity|uncertain_activity",
  "risk_level": "low|medium|high|unknown",
  "confidence": "low|medium|high",
  "description": "one short factual sentence",
  "search_keywords": ["short", "tags"]
}

Rules:
1. Report only visible evidence.
2. Do not invent robbery, weapon, fight, fall, collision, or theft.
3. If unclear, write "unclear".
4. If a person reaches into a counter, display, bag, drawer, shelf, vehicle, or restricted area, mention it.
5. If one person grabs, blocks, pulls, pushes, surrounds, or restrains another person, mention it.
6. If a small object in hand may be important but unclear, say weapon_like_object_visible="unclear".
7. If interaction with a counter, display, bag, drawer, shelf, cashier area, door, gate, parked vehicle, or another person looks unusual or potentially important, do not default to normal_activity. Use an appropriate possible_* label or uncertain_activity.
8. Keep output compact and parseable JSON only."""


def _legacy_step16_prompt() -> str:
    return _compact_step16_prompt()


def _build_selection_context_lines(item: dict[str, Any]) -> list[str]:
    selection_reasons = item.get("selection_reasons", [])
    if not isinstance(selection_reasons, list):
        selection_reasons = []
    ranking_reasons = item.get("ranking_reasons", [])
    if not isinstance(ranking_reasons, list):
        ranking_reasons = []

    yolo = item.get("yolo", {})
    if not isinstance(yolo, dict):
        yolo = {}

    yolo_top_classes = []
    for entry in yolo.get("top_classes", [])[:4]:
        if isinstance(entry, dict):
            class_name = str(entry.get("class_name", "")).strip()
            if class_name:
                yolo_top_classes.append(class_name)

    context_lines: list[str] = ["Selection hints from earlier pipeline steps:"]
    person_max = yolo.get("person_max")
    vehicle_max = yolo.get("vehicle_max")
    important_object_max = yolo.get("important_object_max")
    if person_max is not None or vehicle_max is not None or important_object_max is not None:
        context_lines.append(
            f"- Detection hints: people_max={person_max or 0}, vehicle_max={vehicle_max or 0}, important_object_max={important_object_max or 0}"
        )
    if yolo_top_classes:
        context_lines.append(f"- Top detected classes: {', '.join(yolo_top_classes)}")

    reason_flags = {str(reason).strip().lower() for reason in selection_reasons + ranking_reasons if str(reason).strip()}
    if {"shop_counter_display_context", "object_interaction_context", "person_object_interaction_possible"} & reason_flags:
        context_lines.append(
            "- This clip was selected because a shop/display/counter or person-object interaction may be important. If the CURRENT panel shows reaching, taking, hiding, passing, or handling an item near a counter/display, avoid labeling it as routine normal activity."
        )
    if {"person_person_interaction_possible", "multiple_people"} & reason_flags:
        context_lines.append(
            "- Multiple people or close person-person interaction may matter here. If the CURRENT panel shows grabbing, blocking, crowding, controlling, chasing, or confrontation, prefer a possible incident label over normal_activity."
        )
    if {"high_motion", "adaptive_high_change", "adaptive_motion_change"} & reason_flags:
        context_lines.append(
            "- This moment has elevated motion/change. Check whether the CURRENT panel shows a transition such as sudden movement, approach, departure, running, item movement, or vehicle conflict."
        )
    if any(term in reason_flags for term in {"moving_vehicle", "vehicle_present", "traffic_scene"}):
        context_lines.append(
            "- For traffic/road scenes, distinguish normal traffic flow from near-collision, sudden stop, intrusion, or unusual pedestrian-vehicle interaction."
        )
    return context_lines if len(context_lines) > 1 else []


def _build_prompt_for_item(item: dict[str, Any]) -> str:
    prompt = get_tender_demo_step16_prompt()
    context_lines = _build_selection_context_lines(item)
    motion_hints = item.get("motion_state_hints", {})
    if (not isinstance(motion_hints, dict) or not motion_hints) and not context_lines:
        return prompt

    appended_sections: list[str] = []
    if context_lines:
        appended_sections.append("\n".join(context_lines))

    if isinstance(motion_hints, dict) and motion_hints:
        motion_lines = ["Rule-based motion evidence from YOLO frame comparison:"]
        for entry in motion_hints.get("objects_in_motion", [])[:4]:
            if not isinstance(entry, dict):
                continue
            motion_lines.append(
                f"- {entry.get('class_name', 'object')}: {entry.get('motion_state', 'moving')}"
                f", direction {entry.get('direction', 'unknown')}, confidence {entry.get('confidence', 'low')}"
            )
        for entry in motion_hints.get("stationary_objects", [])[:3]:
            if not isinstance(entry, dict):
                continue
            motion_lines.append(
                f"- {entry.get('class_name', 'object')}: {entry.get('motion_state', 'stationary_or_parked')}"
                f", confidence {entry.get('confidence', 'low')}"
            )
        if motion_hints.get("motion_summary"):
            motion_lines.append(f"Summary: {motion_hints.get('motion_summary')}")
        motion_lines.extend(
            [
                "Use this evidence when describing motion.",
                "Do not call a vehicle parked if motion evidence says moving.",
                "Only call a vehicle parked/stationary if motion evidence says stationary_or_parked.",
                "If visual appearance and motion evidence conflict, say motion is unclear rather than parked.",
            ]
        )
        appended_sections.append("\n".join(motion_lines))
    return prompt + "\n\n" + "\n\n".join(appended_sections)


def get_tender_demo_step16_prompt() -> str:
    prompt_override = _read_prompt_override_text()
    if prompt_override:
        return prompt_override
    if _safe_bool_env("TENDER_DEMO_FAST_COMPACT_QWEN_SCHEMA", DEFAULT_FAST_COMPACT_QWEN_SCHEMA):
        return _compact_step16_prompt()
    return _legacy_step16_prompt()


def _extract_json_text(raw_output: str) -> str:
    cleaned = str(raw_output or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace >= first_brace:
        cleaned = cleaned[first_brace : last_brace + 1]
    elif first_brace != -1:
        cleaned = cleaned[first_brace:]
    return cleaned.strip()


def _repair_json_candidate(text: str) -> str:
    repaired = str(text or "").strip()
    repaired = repaired.replace("\u201c", '"').replace("\u201d", '"')
    repaired = repaired.replace("\u2018", "'").replace("\u2019", "'")
    repaired = repaired.replace("\r", " ").replace("\n", " ")
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    if repaired.count("{") > repaired.count("}"):
        repaired += "}" * (repaired.count("{") - repaired.count("}"))
    if repaired.count("[") > repaired.count("]"):
        repaired += "]" * (repaired.count("[") - repaired.count("]"))
    return re.sub(r"\s+", " ", repaired).strip()


def _first_useful_sentence(text: str) -> str:
    cleaned = re.sub(r"```(?:json)?", " ", str(text or ""), flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    for sentence in sentences:
        candidate = sentence.strip(" \t\r\n-:;,.")
        if len(candidate) >= 8:
            return candidate[:160]
    return cleaned[:160]


def _build_fallback_parsed_json(raw_text: str) -> dict[str, Any]:
    sentence = _first_useful_sentence(raw_text)
    description = re.sub(r"\s+", " ", str(raw_text or "")).strip()[:250]
    if not description:
        description = sentence or "No usable VLM text was returned."
    return {
        "scene_type": "unknown",
        "caption": sentence or description,
        "people_count": 0,
        "vehicle_count": 0,
        "main_objects": [],
        "main_activities": [],
        "main_interactions": [],
        "motion_summary": "",
        "moving_objects": [],
        "stationary_objects": [],
        "traffic_state": "unclear",
        "primary_event_label": "uncertain_activity",
        "event_label": "uncertain_activity",
        "suspicious_activity": "unclear",
        "risk_level": "unknown",
        "description": description,
        "search_keywords": [],
        "keywords": [],
        }


def _normalize_parsed_json_schema(parsed_json: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(parsed_json) if isinstance(parsed_json, dict) else {}

    primary_event_label = str(normalized.get("primary_event_label", "")).strip()
    event_label = str(normalized.get("event_label", "")).strip()
    if primary_event_label and not event_label:
        normalized["event_label"] = primary_event_label
    elif event_label and not primary_event_label:
        normalized["primary_event_label"] = event_label

    search_keywords = normalized.get("search_keywords")
    keywords = normalized.get("keywords")
    if isinstance(search_keywords, list) and not isinstance(keywords, list):
        normalized["keywords"] = search_keywords
    elif isinstance(keywords, list) and not isinstance(search_keywords, list):
        normalized["search_keywords"] = keywords
    elif not isinstance(search_keywords, list) and not isinstance(keywords, list):
        normalized["search_keywords"] = []
        normalized["keywords"] = []

    for key in ["main_objects", "main_activities", "main_interactions", "moving_objects", "stationary_objects"]:
        if not isinstance(normalized.get(key), list):
            normalized[key] = []

    return normalized


def parse_qwen_json_output(raw_text: str) -> tuple[bool, dict[str, Any] | None, str | None, bool]:
    cleaned = _extract_json_text(raw_text)
    parse_errors: list[str] = []

    if cleaned:
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return True, _normalize_parsed_json_schema(parsed), None, False
            parse_errors.append("Parsed JSON was not an object.")
        except json.JSONDecodeError as exc:
            parse_errors.append(str(exc))

        repaired = _repair_json_candidate(cleaned)
        if repaired:
            try:
                parsed = json.loads(repaired)
                if isinstance(parsed, dict):
                    return True, _normalize_parsed_json_schema(parsed), None, False
                parse_errors.append("Repaired JSON was not an object.")
            except json.JSONDecodeError as exc:
                parse_errors.append(str(exc))
    else:
        parse_errors.append("No JSON object found in Qwen output.")

    fallback = _normalize_parsed_json_schema(_build_fallback_parsed_json(raw_text))
    return False, fallback, " | ".join(parse_errors), True


def run_qwen_on_topk_vlm_inputs(run_dir: Path) -> list[dict[str, Any]]:
    requested_backend = os.environ.get("TENDER_DEMO_VLM_BACKEND", "qwen").strip().lower() or "qwen"
    print(f"[tender-demo] Starting Step 16: run {requested_backend} on Top-K VLM inputs")

    manifest_path = run_dir / "15_topk_vlm_inputs.json"
    items = _load_required_manifest(manifest_path)
    if not items:
        raise ValueError("15_topk_vlm_inputs.json contains no selected Top-K VLM inputs.")

    print(f"[tender-demo] Total Top-K VLM inputs: {len(items)}")

    output_path = run_dir / "16_topk_vlm_outputs.json"
    existing_raw_outputs = _load_existing_raw_outputs(output_path)

    vlm = None
    vlm_health: dict[str, Any] = {}
    resolved_backend = requested_backend
    adapter_error: str | None = None
    try:
        create_tender_demo_vlm = _load_tender_demo_vlm_factory()
        vlm = create_tender_demo_vlm()
        if hasattr(vlm, "health_check"):
            try:
                vlm_health = vlm.health_check()
            except Exception:
                vlm_health = {}
        resolved_backend = str(vlm_health.get("backend", requested_backend)).strip() or requested_backend
    except Exception as exc:
        adapter_error = str(exc)
        if existing_raw_outputs:
            print("[tender-demo] Selected VLM adapter unavailable. Reusing existing raw outputs for reparsing.")
        else:
            raise

    valid_requests: list[tuple[dict[str, Any], Path]] = []
    results: list[dict[str, Any]] = []

    for index, item in enumerate(items, start=1):
        record = dict(item)
        prompt = _build_prompt_for_item(item)
        record["topk_vlm_output_id"] = f"topk_vlm_output_{index:06d}"
        record["prompt"] = prompt
        record["raw_vlm_output"] = ""
        record["parsed_json"] = None
        record["parse_success"] = False
        record["parse_error"] = None
        record["fallback_used"] = False

        strip_path_value = item.get("strip_path")
        strip_path = _resolve_media_path(strip_path_value)
        if strip_path is None:
            record["parse_error"] = f"Missing strip image path: {strip_path_value}"
            results.append(record)
            continue

        valid_requests.append((record, strip_path))
        results.append(record)

    if valid_requests and vlm is not None:
        image_paths = [strip_path for _, strip_path in valid_requests]
        prompts = [record.get("prompt", get_tender_demo_step16_prompt()) for record, _ in valid_requests]
        try:
            raw_outputs = vlm.generate_batch(image_paths=image_paths, prompts=prompts)
        except Exception as exc:
            raw_outputs = []
            error_message = f"VLM batch generation failed: {exc}"
            for record, _ in valid_requests:
                record["parse_error"] = error_message
                record["parsed_json"] = _build_fallback_parsed_json("")
                record["fallback_used"] = True
        else:
            for (record, _), raw_output in zip(valid_requests, raw_outputs):
                raw_text = raw_output if isinstance(raw_output, str) else str(raw_output)
                parse_success, parsed_json, parse_error, fallback_used = parse_qwen_json_output(raw_text)
                record["raw_vlm_output"] = raw_text
                record["parsed_json"] = parsed_json
                record["parse_success"] = parse_success
                record["parse_error"] = parse_error
                record["fallback_used"] = fallback_used

            if len(raw_outputs) < len(valid_requests):
                for record, _ in valid_requests[len(raw_outputs) :]:
                    record["parse_error"] = "Adapter returned fewer outputs than requested."
                    record["parsed_json"] = _build_fallback_parsed_json("")
                    record["fallback_used"] = True
    else:
        print("[tender-demo] No valid strip images were found for Qwen inference.")

    if existing_raw_outputs:
        for record in results:
            if str(record.get("raw_vlm_output", "")).strip():
                continue
            record_id = _record_identity(record)
            existing_raw_text = existing_raw_outputs.get(record_id, "")
            if not existing_raw_text.strip():
                continue
            parse_success, parsed_json, parse_error, fallback_used = parse_qwen_json_output(existing_raw_text)
            record["raw_vlm_output"] = existing_raw_text
            record["parsed_json"] = parsed_json
            record["parse_success"] = parse_success
            record["parse_error"] = parse_error or record.get("parse_error") or adapter_error
            record["fallback_used"] = fallback_used

    successful_outputs = sum(1 for item in results if item.get("parse_success") is True)
    failed_outputs = len(results) - successful_outputs
    fallback_outputs = sum(1 for item in results if item.get("fallback_used") is True)
    empty_outputs = sum(1 for item in results if not str(item.get("raw_vlm_output", "")).strip())

    payload = {
        "vlm_backend": resolved_backend,
        "requested_vlm_backend": requested_backend,
        "vlm_health": vlm_health,
        "total_inputs": len(items),
        "successful_outputs": successful_outputs,
        "failed_outputs": failed_outputs,
        "fallback_outputs": fallback_outputs,
        "empty_outputs": empty_outputs,
        "items": results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[tender-demo] Selected VLM backend: {resolved_backend}")
    print(f"[tender-demo] Successful strict parses: {successful_outputs}")
    print(f"[tender-demo] Failed strict parses: {failed_outputs}")
    print(f"[tender-demo] Fallback parses used: {fallback_outputs}")
    print(f"[tender-demo] Empty outputs: {empty_outputs}")
    print(f"[tender-demo] Output path: {output_path}")

    return results
