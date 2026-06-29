from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


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


def get_tender_demo_step16_prompt() -> str:
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


def _extract_json_text(raw_output: str) -> str:
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1 :].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        pass

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return cleaned[first_brace : last_brace + 1].strip()
    return cleaned


def _parse_output(raw_output: str) -> tuple[Any | None, bool, str | None]:
    cleaned = _extract_json_text(raw_output)
    try:
        parsed = json.loads(cleaned)
        return parsed, True, None
    except json.JSONDecodeError as exc:
        return None, False, str(exc)


def run_qwen_on_topk_vlm_inputs(run_dir: Path) -> list[dict[str, Any]]:
    print("[tender-demo] Starting Step 16: run Qwen on Top-K VLM inputs")

    manifest_path = run_dir / "15_topk_vlm_inputs.json"
    items = _load_required_manifest(manifest_path)
    if not items:
        raise ValueError("15_topk_vlm_inputs.json contains no selected Top-K VLM inputs.")

    print(f"[tender-demo] Total Top-K VLM inputs: {len(items)}")

    prompt = get_tender_demo_step16_prompt()
    output_path = run_dir / "16_topk_vlm_outputs.json"

    TenderDemoQwenVLM = _load_tender_demo_qwen_vlm()
    vlm = TenderDemoQwenVLM()

    valid_requests: list[tuple[dict[str, Any], Path]] = []
    results: list[dict[str, Any]] = []

    for index, item in enumerate(items, start=1):
        record = dict(item)
        record["topk_vlm_output_id"] = f"topk_vlm_output_{index:06d}"
        record["prompt"] = prompt
        record["raw_vlm_output"] = ""
        record["parsed_json"] = None
        record["parse_success"] = False
        record["parse_error"] = None

        strip_path_value = item.get("strip_path")
        strip_path = Path(str(strip_path_value)) if strip_path_value else None
        if strip_path is None or not strip_path.exists():
            record["parse_error"] = f"Missing strip image path: {strip_path_value}"
            results.append(record)
            continue

        valid_requests.append((record, strip_path))
        results.append(record)

    if valid_requests:
        image_paths = [strip_path for _, strip_path in valid_requests]
        prompts = [prompt] * len(valid_requests)
        try:
            raw_outputs = vlm.generate_batch(image_paths=image_paths, prompts=prompts)
        except Exception as exc:
            raw_outputs = []
            error_message = f"Qwen batch generation failed: {exc}"
            for record, _ in valid_requests:
                record["parse_error"] = error_message
        else:
            for (record, _), raw_output in zip(valid_requests, raw_outputs):
                raw_text = raw_output if isinstance(raw_output, str) else str(raw_output)
                parsed_json, parse_success, parse_error = _parse_output(raw_text)
                record["raw_vlm_output"] = raw_text
                record["parsed_json"] = parsed_json
                record["parse_success"] = parse_success
                record["parse_error"] = parse_error

            if len(raw_outputs) < len(valid_requests):
                for record, _ in valid_requests[len(raw_outputs) :]:
                    record["parse_error"] = "Adapter returned fewer outputs than requested."
    else:
        print("[tender-demo] No valid strip images were found for Qwen inference.")

    successful_outputs = sum(1 for item in results if item.get("parse_success") is True)
    failed_outputs = len(results) - successful_outputs

    payload = {
        "total_inputs": len(items),
        "successful_outputs": successful_outputs,
        "failed_outputs": failed_outputs,
        "items": results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[tender-demo] Successful parses: {successful_outputs}")
    print(f"[tender-demo] Failed parses: {failed_outputs}")
    print(f"[tender-demo] Output path: {output_path}")

    return results
