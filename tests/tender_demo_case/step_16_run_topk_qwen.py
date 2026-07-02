from __future__ import annotations

import importlib.util
import json
import os
import re
from pathlib import Path
from typing import Any


DEFAULT_FAST_COMPACT_QWEN_SCHEMA = True


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


def _load_tender_demo_qwen_vlm():
    try:
        from tests.tender_demo_case.tender_demo_vlm_adapter import TenderDemoQwenVLM
        return TenderDemoQwenVLM
    except ModuleNotFoundError:
        adapter_path = Path(__file__).resolve().parent / "tender_demo_vlm_adapter.py"
        spec = importlib.util.spec_from_file_location("tender_demo_vlm_adapter", adapter_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load tender demo VLM adapter from: {adapter_path}")
        adapter_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter_module)
        return adapter_module.TenderDemoQwenVLM


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
    raw_by_clip_id: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        clip_id = str(item.get("source_clip_id", "")).strip()
        raw_text = str(item.get("raw_vlm_output", "") or "")
        if clip_id and raw_text.strip():
            raw_by_clip_id[clip_id] = raw_text
    return raw_by_clip_id


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


def _legacy_step16_prompt() -> str:
    return """You analyze CCTV/security imagery for a tender-demo video summarization system.

The image is a 3-panel temporal strip:

PREVIOUS | CURRENT | NEXT

Focus mainly on the CURRENT panel.
Use PREVIOUS and NEXT only as temporal context.

Return only valid JSON.
Do not use markdown.
Do not add explanation outside JSON.

JSON schema:

{
"scene_type": "shop | street | office | warehouse | home | vehicle | outdoor | unknown",
"caption": "one concise sentence describing the CURRENT panel",
"people_count": 0,
"visible_people": [
{
"id": "person_1",
"appearance": "brief visual description",
"pose_or_action": "standing | walking | bending | reaching | sitting | running | unknown",
"location": "brief location in scene"
}
],
"objects": [
{
"name": "object name",
"location": "brief location",
"possible_relevance": "normal | potentially_relevant | unknown"
}
],
"activities": [
{
"activity_type": "person_object_interaction | walking | standing | bending | reaching | crowding | unclear",
"description": "brief description",
"actors": ["person_1"]
}
],
"events": [
{
"event_type": "normal_activity | possible_theft | possible_robbery | suspicious_reaching | object_removed | fall | fight | intrusion | unclear",
"description": "brief event description",
"severity": "low | medium | high | unknown"
}
],
"suspicious_activity": "yes | no | unclear",
"risk_level": "low | medium | high | unknown",
"event_label": "normal_activity | possible_theft_or_robbery | suspicious_activity | uncertain_activity",
"confidence": "low | medium | high",
"keywords": []
}

Hard rules:

* If a person bends over, reaches into a counter/display case, hides an object, removes an item, or interacts unusually with a display/counter, mention it clearly.
* If the scene looks normal, describe the visible normal activity only.
* Do not say robbery/theft unless visual evidence suggests it.
* Do not say there is no theft/no robbery/no assault unless directly asked.
* If unsure, use suspicious_activity = "unclear" and event_label = "uncertain_activity".
* Always return parseable JSON."""


def _compact_step16_prompt() -> str:
    return """You analyze CCTV/security imagery for a fast tender-demo video summarization system.

The image is a 3-panel temporal strip:

PREVIOUS | CURRENT | NEXT

Focus on the CURRENT panel.
Use PREVIOUS and NEXT only as context.

Return ONLY valid JSON.
No markdown.
No explanation.
No code block.

JSON schema:
{
"scene_type": "street|road|parking|shop|office|warehouse|indoor|outdoor|unknown",
"caption": "short sentence describing current panel",
"people_count": 0,
"main_objects": ["person", "vehicle", "bag"],
"main_activities": ["walking", "standing", "riding", "carrying", "running"],
"motion_summary": "short sentence describing movement",
"moving_objects": ["car", "person"],
"stationary_objects": [],
"traffic_state": "moving_traffic|parked_or_stopped_vehicle|pedestrian_activity|mixed_motion|unclear",
"event_label": "normal_activity|possible_theft_or_robbery|possible_fight|possible_collision|possible_fall|possible_intrusion|traffic_activity|uncertain_activity",
"suspicious_activity": "yes|no|unclear",
"risk_level": "low|medium|high|unknown",
"description": "one factual sentence about what is visible",
"keywords": ["short", "searchable", "tags"]
}

Rules:
* Focus on CURRENT panel.
* Use PREVIOUS and NEXT only as context.
* Mention visible people, vehicles, bags, bicycles, motorcycles, collisions, fights, falls, running, crowding, or unusual interactions.
* If normal road/parking/shop activity is visible, say normal_activity.
* Never call a vehicle parked unless motion evidence or clear visual evidence supports it.
* If unsure, use suspicious_activity="unclear" and event_label="uncertain_activity".
* Keep output short.
* Return only JSON."""


def _build_prompt_for_item(item: dict[str, Any]) -> str:
    prompt = get_tender_demo_step16_prompt()
    motion_hints = item.get("motion_state_hints", {})
    if not isinstance(motion_hints, dict) or not motion_hints:
        return prompt

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
    return prompt + "\n\n" + "\n".join(motion_lines)


def get_tender_demo_step16_prompt() -> str:
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
        description = sentence or "No usable Qwen text was returned."
    return {
        "scene_type": "unknown",
        "caption": sentence or description,
        "people_count": 0,
        "main_objects": [],
        "main_activities": [],
        "motion_summary": "",
        "moving_objects": [],
        "stationary_objects": [],
        "traffic_state": "unclear",
        "event_label": "uncertain_activity",
        "suspicious_activity": "unclear",
        "risk_level": "unknown",
        "description": description,
        "keywords": [],
    }


def parse_qwen_json_output(raw_text: str) -> tuple[bool, dict[str, Any] | None, str | None, bool]:
    cleaned = _extract_json_text(raw_text)
    parse_errors: list[str] = []

    if cleaned:
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return True, parsed, None, False
            parse_errors.append("Parsed JSON was not an object.")
        except json.JSONDecodeError as exc:
            parse_errors.append(str(exc))

        repaired = _repair_json_candidate(cleaned)
        if repaired:
            try:
                parsed = json.loads(repaired)
                if isinstance(parsed, dict):
                    return True, parsed, None, False
                parse_errors.append("Repaired JSON was not an object.")
            except json.JSONDecodeError as exc:
                parse_errors.append(str(exc))
    else:
        parse_errors.append("No JSON object found in Qwen output.")

    fallback = _build_fallback_parsed_json(raw_text)
    return False, fallback, " | ".join(parse_errors), True


def run_qwen_on_topk_vlm_inputs(run_dir: Path) -> list[dict[str, Any]]:
    print("[tender-demo] Starting Step 16: run Qwen on Top-K VLM inputs")

    manifest_path = run_dir / "15_topk_vlm_inputs.json"
    items = _load_required_manifest(manifest_path)
    if not items:
        raise ValueError("15_topk_vlm_inputs.json contains no selected Top-K VLM inputs.")

    print(f"[tender-demo] Total Top-K VLM inputs: {len(items)}")

    output_path = run_dir / "16_topk_vlm_outputs.json"
    existing_raw_outputs = _load_existing_raw_outputs(output_path)

    vlm = None
    adapter_error: str | None = None
    try:
        TenderDemoQwenVLM = _load_tender_demo_qwen_vlm()
        vlm = TenderDemoQwenVLM()
    except Exception as exc:
        adapter_error = str(exc)
        if existing_raw_outputs:
            print("[tender-demo] Qwen adapter unavailable. Reusing existing raw outputs for reparsing.")
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
        strip_path = Path(str(strip_path_value)) if strip_path_value else None
        if strip_path is None or not strip_path.exists():
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
            error_message = f"Qwen batch generation failed: {exc}"
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
            clip_id = str(record.get("source_clip_id", "")).strip()
            existing_raw_text = existing_raw_outputs.get(clip_id, "")
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
        "total_inputs": len(items),
        "successful_outputs": successful_outputs,
        "failed_outputs": failed_outputs,
        "fallback_outputs": fallback_outputs,
        "empty_outputs": empty_outputs,
        "items": results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[tender-demo] Successful strict parses: {successful_outputs}")
    print(f"[tender-demo] Failed strict parses: {failed_outputs}")
    print(f"[tender-demo] Fallback parses used: {fallback_outputs}")
    print(f"[tender-demo] Empty outputs: {empty_outputs}")
    print(f"[tender-demo] Output path: {output_path}")

    return results
