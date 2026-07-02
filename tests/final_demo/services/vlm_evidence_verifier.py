from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from tests.final_demo.services.chunk_planner import read_json
from tests.final_demo.services.final_demo_vlm_adapter import (
    ENV_FINAL_DEMO_VLM_VERIFY_ENABLED,
    FinalDemoVLMAdapter,
)
from tests.final_demo.services.video_io import current_timestamp, write_json


ENV_FINAL_DEMO_VLM_MAX_CANDIDATES = "FINAL_DEMO_VLM_MAX_CANDIDATES"
ENV_FINAL_DEMO_VLM_INPUT_MODE = "FINAL_DEMO_VLM_INPUT_MODE"
ENV_FINAL_DEMO_VLM_DEBUG_FULL = "FINAL_DEMO_VLM_DEBUG_FULL"

DEFAULT_VLM_MAX_CANDIDATES = 10
DEFAULT_VLM_INPUT_MODE = "crop_or_frame"
DEFAULT_VLM_DEBUG_FULL = False


def read_bool_env(env_name: str, default_value: bool) -> bool:
    raw_value = os.environ.get(env_name)
    if raw_value is None or raw_value.strip() == "":
        return default_value
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Environment variable {env_name} must be boolean-like. Received: {raw_value!r}")


def read_positive_int_env(env_name: str, default_value: int) -> int:
    raw_value = os.environ.get(env_name, str(default_value))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {env_name} must be integer. Received: {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"Environment variable {env_name} must be greater than 0. Received: {raw_value!r}")
    return value


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_round(value: Any, digits: int = 3) -> float:
    return round(as_float(value, 0.0), digits)


def clean_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def has_text(value: Any) -> bool:
    return bool(str(value or "").strip())


def priority_rank(priority: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(str(priority or ""), 9)


def try_import_pil() -> Any:
    try:
        from PIL import Image, ImageDraw, ImageOps  # type: ignore
    except Exception:
        return None
    return {"Image": Image, "ImageDraw": ImageDraw, "ImageOps": ImageOps}


def choose_source_paths(record: dict[str, Any], candidate: dict[str, Any]) -> tuple[str, str]:
    evidence = dict(record.get("evidence") or {})
    source_image_path = str(
        candidate.get("image_path")
        or evidence.get("image_path")
        or evidence.get("crop_path")
        or evidence.get("subject_crop_path")
        or evidence.get("object_crop_path")
        or ""
    )
    source_crop_path = str(
        candidate.get("crop_path")
        or evidence.get("crop_path")
        or evidence.get("subject_crop_path")
        or evidence.get("object_crop_path")
        or ""
    )
    return source_image_path, source_crop_path


def prepare_vlm_input_image(
    input_dir: Path,
    index: int,
    candidate: dict[str, Any],
    record: dict[str, Any],
    input_mode: str,
) -> tuple[str, dict[str, Any]]:
    debug_row: dict[str, Any] = {
        "source_evidence_id": str(candidate.get("source_evidence_id") or ""),
        "reason": "",
    }
    source_image_path, source_crop_path = choose_source_paths(record, candidate)
    selected_source = source_crop_path or source_image_path
    if not selected_source:
        debug_row["reason"] = "missing_source_image_and_crop"
        return "", debug_row
    selected_path = Path(selected_source)
    if not selected_path.exists():
        debug_row["reason"] = f"selected_input_missing:{selected_path}"
        return "", debug_row
    pil_modules = try_import_pil()
    if pil_modules is None:
        debug_row["reason"] = "pil_not_available_copy_source_image"
        return str(selected_path), debug_row
    Image = pil_modules["Image"]
    ImageDraw = pil_modules["ImageDraw"]
    ImageOps = pil_modules["ImageOps"]
    output_path = input_dir / f"vlm_verify_{index:06d}.jpg"
    try:
        source_image = Image.open(source_image_path).convert("RGB") if source_image_path and Path(source_image_path).exists() else None
        crop_image = Image.open(source_crop_path).convert("RGB") if source_crop_path and Path(source_crop_path).exists() else None
        if input_mode == "crop_or_frame" and crop_image is not None:
            chosen = crop_image.copy()
            chosen.thumbnail((1280, 1280))
            chosen.save(output_path, format="JPEG", quality=90)
            debug_row["reason"] = "used_crop_image"
            return str(output_path), debug_row
        if source_image is None and crop_image is not None:
            chosen = crop_image.copy()
            chosen.thumbnail((1280, 1280))
            chosen.save(output_path, format="JPEG", quality=90)
            debug_row["reason"] = "used_crop_image_only"
            return str(output_path), debug_row
        if source_image is None:
            debug_row["reason"] = "no_loadable_source_image"
            return "", debug_row
        if crop_image is None:
            chosen = source_image.copy()
            chosen.thumbnail((1280, 1280))
            chosen.save(output_path, format="JPEG", quality=90)
            debug_row["reason"] = "used_full_frame_only"
            return str(output_path), debug_row
        left = ImageOps.contain(source_image, (960, 720))
        right = ImageOps.contain(crop_image, (960, 720))
        canvas_width = left.width + right.width
        canvas_height = max(left.height, right.height) + 60
        canvas = Image.new("RGB", (canvas_width, canvas_height), color=(18, 18, 18))
        draw = ImageDraw.Draw(canvas)
        canvas.paste(left, (0, 60))
        canvas.paste(right, (left.width, 60))
        label = f"{candidate.get('candidate_reason', 'verification')} | {candidate.get('suggested_prompt_type', '')}"
        draw.text((12, 18), label, fill=(240, 240, 240))
        canvas.save(output_path, format="JPEG", quality=90)
        debug_row["reason"] = "used_side_by_side_frame_and_crop"
        return str(output_path), debug_row
    except Exception as exc:
        debug_row["reason"] = f"image_prepare_failed:{exc}"
        return "", debug_row


def build_plate_prompt(record: dict[str, Any]) -> str:
    attributes = dict(record.get("attributes") or {})
    plate_text = str(attributes.get("plate_text") or attributes.get("candidate_plate_text") or "")
    return (
        "You are verifying CCTV evidence.\n"
        "Look only at the image.\n"
        "Do not guess.\n"
        "Return only valid JSON.\n\n"
        "Task:\n"
        "Verify whether a vehicle number plate is visible and whether the candidate text is supported by the image.\n\n"
        f"Candidate plate text:\n{plate_text}\n\n"
        'Return JSON:\n'
        '{\n'
        '"visible_plate": true/false,\n'
        '"plate_text_supported": true/false/null,\n'
        '"readable_text": "",\n'
        '"text_confidence": 0.0,\n'
        '"vehicle_visible": true/false,\n'
        '"vehicle_type_observed": "",\n'
        '"evidence_quality": "good | medium | poor",\n'
        '"verification_status": "verified | contradicted | inconclusive",\n'
        '"observations": [],\n'
        '"do_not_guess": true\n'
        '}\n'
    )


def build_vehicle_prompt(record: dict[str, Any]) -> str:
    attributes = dict(record.get("attributes") or {})
    possible_vehicle_classes = clean_list(attributes.get("possible_vehicle_classes")) or clean_list(attributes.get("possible_vehicle_classes"))
    if not possible_vehicle_classes:
        possible_vehicle_classes = clean_list(attributes.get("possible_vehicle_types")) or clean_list(attributes.get("possible_vehicle_classes"))
    if not possible_vehicle_classes:
        possible_vehicle_classes = clean_list(attributes.get("possible_vehicle_classes")) + clean_list(attributes.get("possible_vehicle_types"))
    if not possible_vehicle_classes:
        possible_vehicle_classes = clean_list(attributes.get("possible_vehicle_classes"))
    return (
        "You are verifying CCTV vehicle evidence.\n"
        "Look only at the image.\n"
        "Do not guess.\n"
        "Return only valid JSON.\n\n"
        "Task:\n"
        "Verify the visible vehicle type.\n\n"
        f"Candidate system label:\n{record.get('class_name', '')}\n\n"
        f"Possible vehicle classes:\n{possible_vehicle_classes}\n\n"
        'Return JSON:\n'
        '{\n'
        '"vehicle_visible": true/false,\n'
        '"vehicle_type_supported": true/false/null,\n'
        '"observed_vehicle_type": "",\n'
        '"possible_vehicle_types": [],\n'
        '"evidence_quality": "good | medium | poor",\n'
        '"verification_status": "verified | contradicted | inconclusive",\n'
        '"observations": [],\n'
        '"do_not_guess": true\n'
        '}\n'
    )


def build_object_prompt(record: dict[str, Any]) -> str:
    return (
        "You are verifying CCTV object evidence.\n"
        "Look only at the image.\n"
        "Do not guess.\n"
        "Return only valid JSON.\n\n"
        "Task:\n"
        "Verify whether the highlighted/cropped item is truly the candidate object, or whether it looks like part of a vehicle / false detection.\n\n"
        f"Candidate object label:\n{record.get('class_name', '')}\n\n"
        f"Review reason:\n{record.get('review_reason', '')}\n\n"
        'Return JSON:\n'
        '{\n'
        '"object_visible": true/false,\n'
        '"candidate_object_supported": true/false/null,\n'
        '"observed_object_type": "",\n'
        '"looks_like_vehicle_or_vehicle_part": true/false/null,\n'
        '"false_positive_possible": true/false,\n'
        '"evidence_quality": "good | medium | poor",\n'
        '"verification_status": "verified | contradicted | inconclusive",\n'
        '"observations": [],\n'
        '"do_not_guess": true\n'
        '}\n'
    )


def build_person_prompt(record: dict[str, Any]) -> str:
    attributes = dict(record.get("attributes") or {})
    return (
        "You are verifying CCTV person attribute evidence.\n"
        "Look only at the image.\n"
        "Do not guess.\n"
        "Return only valid JSON.\n\n"
        "Task:\n"
        "Verify visible person clothing color.\n\n"
        f"Candidate top clothing color:\n{attributes.get('top_clothing_color') or attributes.get('normalized_top_color') or ''}\n\n"
        f"Candidate bottom clothing color:\n{attributes.get('bottom_clothing_color') or attributes.get('normalized_bottom_color') or ''}\n\n"
        'Return JSON:\n'
        '{\n'
        '"person_visible": true/false,\n'
        '"top_color_supported": true/false/null,\n'
        '"bottom_color_supported": true/false/null,\n'
        '"observed_top_color": "",\n'
        '"observed_bottom_color": "",\n'
        '"carried_object_visible": true/false/null,\n'
        '"carried_object_type": "",\n'
        '"evidence_quality": "good | medium | poor",\n'
        '"verification_status": "verified | contradicted | inconclusive",\n'
        '"observations": [],\n'
        '"do_not_guess": true\n'
        '}\n'
    )


def build_explain_prompt() -> str:
    return (
        "You are verifying one CCTV evidence image.\n"
        "Look only at the image.\n"
        "Do not guess.\n"
        "Return only valid JSON.\n\n"
        "Task:\n"
        "Describe what is visibly supported by this evidence image.\n\n"
        'Return JSON:\n'
        '{\n'
        '"main_visible_entities": [],\n'
        '"visible_activity": "",\n'
        '"evidence_quality": "good | medium | poor",\n'
        '"verification_status": "verified | inconclusive",\n'
        '"observations": [],\n'
        '"do_not_guess": true\n'
        '}\n'
    )


def determine_prompt_type(record: dict[str, Any], candidate: dict[str, Any]) -> str:
    attributes = dict(record.get("attributes") or {})
    suggested_prompt_type = str(candidate.get("suggested_prompt_type") or "")
    if suggested_prompt_type == "verify_plate" or has_text(attributes.get("plate_text")) or has_text(attributes.get("candidate_plate_text")):
        return "verify_plate"
    if bool(attributes.get("vehicle_subtype_needs_review")) or clean_list(attributes.get("possible_vehicle_classes")):
        return "verify_vehicle_type"
    if suggested_prompt_type == "verify_object_type" or str(record.get("status") or "") == "possible_vehicle_misclassification":
        return "verify_object_type"
    if suggested_prompt_type == "verify_person_attribute":
        return "verify_person_attribute"
    return "explain_event"


def build_prompt(record: dict[str, Any], prompt_type: str) -> str:
    if prompt_type == "verify_plate":
        return build_plate_prompt(record)
    if prompt_type == "verify_vehicle_type":
        return build_vehicle_prompt(record)
    if prompt_type == "verify_object_type":
        return build_object_prompt(record)
    if prompt_type == "verify_person_attribute":
        return build_person_prompt(record)
    return build_explain_prompt()


def normalize_verification_status(status: Any) -> str:
    normalized = str(status or "").strip().lower()
    if normalized in {"verified", "contradicted", "inconclusive", "skipped", "error"}:
        return normalized
    return "inconclusive"


def build_final_recommendation(
    verification_status: str,
    verification_payload: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    original_needs_review = bool(record.get("needs_review"))
    if verification_status == "verified" and not original_needs_review:
        return {
            "keep_original_status": False,
            "suggested_status": "confirmed",
            "suggested_label": str(record.get("class_name") or ""),
            "reason": "VLM supports the original evidence and the rule-based record was not review-only.",
        }
    if verification_status == "verified" and original_needs_review:
        return {
            "keep_original_status": True,
            "suggested_status": "review",
            "suggested_label": str(record.get("class_name") or ""),
            "reason": "VLM supports evidence, but original rule-based record still needs review.",
        }
    if verification_status == "contradicted":
        return {
            "keep_original_status": False,
            "suggested_status": "reject",
            "suggested_label": str(record.get("class_name") or ""),
            "reason": "VLM contradicted the key candidate evidence.",
        }
    if verification_status == "inconclusive":
        return {
            "keep_original_status": True,
            "suggested_status": "review",
            "suggested_label": str(record.get("class_name") or ""),
            "reason": "VLM could not confidently verify the evidence.",
        }
    if verification_status == "skipped":
        return {
            "keep_original_status": True,
            "suggested_status": "review",
            "suggested_label": str(record.get("class_name") or ""),
            "reason": "VLM verification was skipped.",
        }
    return {
        "keep_original_status": True,
        "suggested_status": "inconclusive",
        "suggested_label": str(record.get("class_name") or ""),
        "reason": "VLM verification returned an error.",
    }


def build_observations(parsed_json: dict[str, Any]) -> list[str]:
    observations = parsed_json.get("observations")
    if isinstance(observations, list):
        return [str(item) for item in observations if str(item or "").strip()]
    return []


def parse_verification_result(
    record: dict[str, Any],
    inference_result: dict[str, Any],
) -> tuple[str, float, list[str], list[str], list[str], list[str]]:
    parsed_json = dict(inference_result.get("parsed_json") or {})
    verification_status = normalize_verification_status(parsed_json.get("verification_status"))
    observations = build_observations(parsed_json)
    verified_labels: list[str] = []
    contradicted_labels: list[str] = []
    safety_notes: list[str] = []
    confidence = 0.0
    if verification_status == "verified":
        confidence = 0.80
    elif verification_status == "contradicted":
        confidence = 0.75
    elif verification_status == "inconclusive":
        confidence = 0.40
    if parsed_json.get("plate_text_supported") is True:
        verified_labels.append(str(record.get("attributes", {}).get("candidate_plate_text") or record.get("attributes", {}).get("plate_text") or record.get("class_name") or ""))
        confidence = max(confidence, as_float(parsed_json.get("text_confidence"), 0.0))
    if parsed_json.get("plate_text_supported") is False:
        contradicted_labels.append(str(record.get("attributes", {}).get("candidate_plate_text") or record.get("attributes", {}).get("plate_text") or ""))
    if parsed_json.get("candidate_object_supported") is True:
        verified_labels.append(str(record.get("class_name") or ""))
    if parsed_json.get("candidate_object_supported") is False:
        contradicted_labels.append(str(record.get("class_name") or ""))
    if parsed_json.get("vehicle_type_supported") is True and has_text(parsed_json.get("observed_vehicle_type")):
        verified_labels.append(str(parsed_json.get("observed_vehicle_type") or ""))
    if parsed_json.get("vehicle_type_supported") is False:
        contradicted_labels.append(str(record.get("class_name") or ""))
    if parsed_json.get("top_color_supported") is True and has_text(parsed_json.get("observed_top_color")):
        verified_labels.append(str(parsed_json.get("observed_top_color") or ""))
    if parsed_json.get("top_color_supported") is False:
        contradicted_labels.append(str(record.get("attributes", {}).get("top_clothing_color") or record.get("attributes", {}).get("normalized_top_color") or ""))
    if parsed_json.get("do_not_guess") is not True:
        safety_notes.append("Model output did not explicitly preserve do_not_guess=true.")
    if parsed_json.get("evidence_quality") == "poor":
        safety_notes.append("Evidence quality was reported as poor.")
    return verification_status, min(max(confidence, 0.0), 1.0), verified_labels, contradicted_labels, observations, safety_notes


def create_rule_based_status(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "entity_family": str(record.get("entity_family") or ""),
        "entity_type": str(record.get("entity_type") or ""),
        "class_name": str(record.get("class_name") or ""),
        "match_strength": str(record.get("match_strength") or ""),
        "needs_review": bool(record.get("needs_review")),
        "review_reason": str(record.get("review_reason") or ""),
        "confidence": safe_round(record.get("confidence")),
        "attributes": dict(record.get("attributes") or {}),
        "evidence": dict(record.get("evidence") or {}),
    }


def select_candidates(
    ranked_payload: dict[str, Any],
    max_candidates: int,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]]]:
    global_ranked_evidence = list(ranked_payload.get("global_ranked_evidence") or [])
    evidence_lookup = {
        str(item.get("evidence_id") or ""): item
        for item in global_ranked_evidence
    }
    top_vlm_candidates = list(ranked_payload.get("top_vlm_candidates") or [])
    debug_rows: list[dict[str, Any]] = []
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    top_vlm_candidates.sort(
        key=lambda item: (
            priority_rank(str(item.get("priority") or "")),
            next(
                (
                    int(record.get("rank") or 999999)
                    for record in global_ranked_evidence
                    if str(record.get("evidence_id") or "") == str(item.get("source_evidence_id") or "")
                ),
                999999,
            ),
            -next(
                (
                    as_float(record.get("ranking_score"), 0.0)
                    for record in global_ranked_evidence
                    if str(record.get("evidence_id") or "") == str(item.get("source_evidence_id") or "")
                ),
                0.0,
            ),
        )
    )
    for candidate in top_vlm_candidates:
        source_evidence_id = str(candidate.get("source_evidence_id") or "")
        if source_evidence_id not in evidence_lookup:
            debug_rows.append(
                {
                    "source_vlm_candidate_id": str(candidate.get("vlm_candidate_id") or ""),
                    "decision": "skipped_missing_source_evidence",
                }
            )
            continue
        selected.append((candidate, evidence_lookup[source_evidence_id]))
        debug_rows.append(
            {
                "source_vlm_candidate_id": str(candidate.get("vlm_candidate_id") or ""),
                "decision": "selected",
                "source_evidence_id": source_evidence_id,
            }
        )
        if len(selected) >= max_candidates:
            break
    return selected, debug_rows


def build_vlm_verification_outputs(
    run_dir: Path,
    *,
    debug_full: bool = False,
) -> dict[str, Any]:
    ranked_path = run_dir / "15_ranked_evidence.json"
    if not ranked_path.exists():
        raise FileNotFoundError(f"Missing required Step 16 input: {ranked_path}")
    ranked_payload = read_json(ranked_path)
    input_dir = run_dir / "16_vlm_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    max_candidates = read_positive_int_env(ENV_FINAL_DEMO_VLM_MAX_CANDIDATES, DEFAULT_VLM_MAX_CANDIDATES)
    input_mode = os.environ.get(ENV_FINAL_DEMO_VLM_INPUT_MODE, DEFAULT_VLM_INPUT_MODE).strip() or DEFAULT_VLM_INPUT_MODE
    debug_full_enabled = debug_full or read_bool_env(ENV_FINAL_DEMO_VLM_DEBUG_FULL, DEFAULT_VLM_DEBUG_FULL)

    adapter = FinalDemoVLMAdapter()
    selected_candidates, selection_debug = select_candidates(ranked_payload, max_candidates)

    verifications: list[dict[str, Any]] = []
    input_debug: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    model_errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    recommendations: list[str] = []
    vlm_inputs_created = 0
    vlm_calls_attempted = 0
    vlm_calls_succeeded = 0
    vlm_calls_failed = 0

    if not selected_candidates:
        warnings.append("No top VLM candidates were available.")
    if not adapter.enabled:
        warnings.append("VLM disabled.")
        recommendations.append("Set FINAL_DEMO_VLM_VERIFY_ENABLED=1 to run actual verification.")

    for index, (candidate, record) in enumerate(selected_candidates, start=1):
        prompt_type = determine_prompt_type(record, candidate)
        prompt = build_prompt(record, prompt_type)
        input_image_path, input_debug_row = prepare_vlm_input_image(input_dir, index, candidate, record, input_mode)
        input_debug.append(input_debug_row)
        if input_image_path:
            vlm_inputs_created += 1
        source_image_path, source_crop_path = choose_source_paths(record, candidate)

        verification_status = "skipped"
        verification_confidence = 0.0
        parsed_json: dict[str, Any] = {}
        raw_output = ""
        verified_labels: list[str] = []
        contradicted_labels: list[str] = []
        observations: list[str] = []
        safety_notes: list[str] = []
        if not input_image_path:
            verification_status = "skipped"
            safety_notes.append("Missing candidate image or crop.")
            warnings.append("Some VLM candidates were skipped because image inputs were missing.")
        elif not adapter.enabled:
            verification_status = "skipped"
            safety_notes.append("VLM disabled by environment.")
        else:
            vlm_calls_attempted += 1
            inference_result = adapter.verify_image(Path(input_image_path), prompt)
            raw_output = str(inference_result.get("raw_output") or "")
            parsed_json = dict(inference_result.get("parsed_json") or {})
            if inference_result.get("error"):
                error_text = str(inference_result.get("error") or "")
                if error_text.startswith("json_") or "json" in error_text:
                    parse_errors.append({"verification_id": f"vlm_verify_{index:06d}", "error": error_text})
                else:
                    model_errors.append({"verification_id": f"vlm_verify_{index:06d}", "error": error_text})
                verification_status = "error"
                safety_notes.append(error_text)
                vlm_calls_failed += 1
            else:
                verification_status, verification_confidence, verified_labels, contradicted_labels, observations, safety_notes = parse_verification_result(record, inference_result)
                vlm_calls_succeeded += 1
        final_recommendation = build_final_recommendation(verification_status, parsed_json, record)
        verifications.append(
            {
                "verification_id": f"vlm_verify_{index:06d}",
                "source_vlm_candidate_id": str(candidate.get("vlm_candidate_id") or ""),
                "source_evidence_id": str(candidate.get("source_evidence_id") or ""),
                "source_search_id": str(record.get("source_search_id") or ""),
                "source_record_type": str(record.get("source_record_type") or ""),
                "candidate_reason": str(candidate.get("candidate_reason") or ""),
                "suggested_prompt_type": prompt_type,
                "priority": str(candidate.get("priority") or ""),
                "input_image_path": str(input_image_path or ""),
                "source_image_path": str(source_image_path or ""),
                "source_crop_path": str(source_crop_path or ""),
                "start_time": safe_round(record.get("start_time")),
                "end_time": safe_round(record.get("end_time")),
                "representative_timestamp": safe_round(record.get("representative_timestamp")),
                "rule_based_status": create_rule_based_status(record),
                "vlm_prompt": prompt,
                "vlm_raw_output": raw_output,
                "vlm_parsed_json": parsed_json,
                "verification_status": verification_status,
                "verification_confidence": round(verification_confidence, 3),
                "verified_labels": verified_labels,
                "contradicted_labels": contradicted_labels,
                "observations": observations,
                "safety_notes": safety_notes,
                "final_recommendation": final_recommendation,
            }
        )

    verifications_by_status: dict[str, int] = {}
    verifications_by_prompt_type: dict[str, int] = {}
    for item in verifications:
        verifications_by_status[item["verification_status"]] = verifications_by_status.get(item["verification_status"], 0) + 1
        verifications_by_prompt_type[item["suggested_prompt_type"]] = verifications_by_prompt_type.get(item["suggested_prompt_type"], 0) + 1

    if verifications and verifications_by_status.get("inconclusive", 0) == len(verifications):
        warnings.append("All verification results were inconclusive.")
    if parse_errors:
        warnings.append("Some VLM outputs could not be parsed as JSON.")
    if model_errors:
        warnings.append("Some VLM calls failed.")
    if verifications_by_prompt_type.get("verify_plate", 0) > 0 and verifications_by_status.get("inconclusive", 0) >= verifications_by_prompt_type.get("verify_plate", 0):
        recommendations.append("If many plate results are inconclusive, improve plate crop quality.")
    if verifications_by_prompt_type.get("verify_object_type", 0) > 0 and verifications_by_status.get("contradicted", 0) >= max(1, verifications_by_prompt_type.get("verify_object_type", 0) // 2):
        recommendations.append("If many object results are contradicted, train or customize the object detector.")
    if verifications_by_prompt_type.get("verify_vehicle_type", 0) > 0 and verifications_by_status.get("inconclusive", 0) >= max(1, verifications_by_prompt_type.get("verify_vehicle_type", 0) // 2):
        recommendations.append("If many vehicle type results are inconclusive, use higher-resolution frames or a dedicated traffic model.")

    results_payload = {
        "created_at": current_timestamp(),
        "source": {"ranked_evidence": "15_ranked_evidence.json"},
        "vlm_enabled": adapter.enabled,
        "model_id": adapter.model_id,
        "verification_version": "final_demo_vlm_verify_v1",
        "verifications": verifications,
    }
    report_payload = {
        "overall_status": (
            "skipped"
            if not adapter.enabled
            else ("completed_with_errors" if model_errors or parse_errors else "completed")
        ),
        "vlm_enabled": adapter.enabled,
        "model_id": adapter.model_id,
        "ranked_evidence_loaded": len(list(ranked_payload.get("global_ranked_evidence") or [])),
        "top_vlm_candidates_loaded": len(list(ranked_payload.get("top_vlm_candidates") or [])),
        "candidates_selected": len(selected_candidates),
        "vlm_inputs_created": vlm_inputs_created,
        "vlm_calls_attempted": vlm_calls_attempted,
        "vlm_calls_succeeded": vlm_calls_succeeded,
        "vlm_calls_failed": vlm_calls_failed,
        "verifications_created": len(verifications),
        "verifications_by_status": dict(sorted(verifications_by_status.items())),
        "verifications_by_prompt_type": dict(sorted(verifications_by_prompt_type.items())),
        "verified_count": verifications_by_status.get("verified", 0),
        "contradicted_count": verifications_by_status.get("contradicted", 0),
        "inconclusive_count": verifications_by_status.get("inconclusive", 0),
        "skipped_count": verifications_by_status.get("skipped", 0),
        "error_count": verifications_by_status.get("error", 0),
        "plate_verifications": verifications_by_prompt_type.get("verify_plate", 0),
        "object_type_verifications": verifications_by_prompt_type.get("verify_object_type", 0),
        "vehicle_type_verifications": verifications_by_prompt_type.get("verify_vehicle_type", 0),
        "person_attribute_verifications": verifications_by_prompt_type.get("verify_person_attribute", 0),
        "warnings": warnings,
        "recommendations": recommendations,
    }
    debug_payload = {
        "created_at": current_timestamp(),
        "candidate_selection_decisions": selection_debug[: (len(selection_debug) if debug_full_enabled else 80)],
        "input_image_creation_decisions": input_debug[: (len(input_debug) if debug_full_enabled else 80)],
        "prompt_type_decisions": [
            {
                "verification_id": item.get("verification_id"),
                "prompt_type": item.get("suggested_prompt_type"),
                "candidate_reason": item.get("candidate_reason"),
            }
            for item in verifications[: (len(verifications) if debug_full_enabled else 80)]
        ],
        "raw_model_errors": model_errors[: (len(model_errors) if debug_full_enabled else 80)],
        "parse_errors": parse_errors[: (len(parse_errors) if debug_full_enabled else 80)],
        "skipped_candidates": [
            {
                "verification_id": item.get("verification_id"),
                "reason": item.get("safety_notes"),
            }
            for item in verifications
            if str(item.get("verification_status") or "") == "skipped"
        ][: (len(verifications) if debug_full_enabled else 80)],
    }
    return {
        "results_payload": results_payload,
        "report_payload": report_payload,
        "debug_payload": debug_payload,
    }


def update_run_manifest_for_vlm_verification(run_manifest_path: Path) -> dict[str, Any]:
    run_manifest = read_json(run_manifest_path)
    completed_steps = list(run_manifest.get("completed_steps") or [])
    if "16_vlm_evidence_verification" not in completed_steps:
        completed_steps.append("16_vlm_evidence_verification")
    run_manifest["completed_steps"] = completed_steps
    run_manifest["next_step"] = "17_final_summary_or_ui"
    write_json(run_manifest_path, run_manifest)
    return run_manifest
