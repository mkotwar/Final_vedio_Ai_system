from __future__ import annotations

import hashlib
import importlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from config import ENV_STEP14_DEVICE
from device_manager import resolve_device
from stage_checks import read_json, write_json
from step_09_search_result_packaging import write_json_any
from qwen_api_client import call_qwen_api_with_image
from qwen_4bit import (
    build_qwen_4bit_load_config,
    capture_cuda_memory_snapshot,
    release_qwen_resources,
    verify_model_loaded_in_4bit,
)


INPUT_FILE_PRIMARY = "14_vlm_event_reviews.json"
SOURCE_FILE = "13_vlm_event_inputs.json"
MODEL_NAME = "Qwen2.5-VL-7B-Instruct"
PROMPT_VERSION = "step14_v1"
ALLOWED_REVIEW_DECISIONS = {"event_visible", "normal_context", "uncertain"}
ALLOWED_EVENT_TYPES = {"collision", "near_miss", "sudden_stop", "traffic_congestion", "normal_traffic", "other", "uncertain"}
ALLOWED_RISK_LEVELS = {"high", "medium", "low", "none"}
NORMALISH_STEP11_5_DECISIONS = {"no", "fallback_normal_context"}
ABNORMAL_TERMS = (
    "collision",
    "accident",
    "contact",
    "impact",
    "crash",
    "near miss",
    "blocked",
    "blockage",
    "fallen",
    "overturned",
    "dangerous",
    "obstruction",
    "person on the ground",
    "vehicle on the ground",
)
NORMAL_TRAFFIC_TERMS = (
    "normal traffic",
    "traffic moving",
    "vehicle driving",
    "vehicle moving",
    "person riding",
    "motorcycle riding",
    "brake lights",
    "cars on the road",
    "traffic is present",
)
REVIEW_PROMPT_TEMPLATE = (
    "You are reviewing CCTV/traffic/security footage. "
    "The image may be a 3-panel temporal strip labeled PREVIOUS, CURRENT / EVENT CENTER, NEXT, or a contact sheet. "
    "Review only what is visibly present. Do not assume event truth from metadata. Candidate labels are only weak hints. "
    "If the Step 11.5 filter decision is fallback_normal_context or no, treat this as a normal review moment unless the image clearly contradicts it. "
    "Do not say collision or accident unless visible contact, clear aftermath, a fallen vehicle or person, a blocked lane, or clearly dangerous abnormal behavior is visible. "
    "If a collision, crash, impact, rollover, fire, explosion, or dangerous aftermath is clearly visible, preserve it as a standalone high-priority event and do not relabel it as normal traffic. "
    "Normal traffic, normal riding, normal driving, brake lights alone, or vehicles close in traffic should be described as normal traffic, not an event. "
    "Return only JSON with keys vlm_input_id, review_decision, event_visible, event_type, risk_level, summary_caption, what_is_visible, why_decision, objects_seen, needs_human_review, confidence. "
    "review_decision must be one of event_visible, normal_context, uncertain. "
    "event_type must be one of collision, near_miss, sudden_stop, traffic_congestion, normal_traffic, other, uncertain. "
    "risk_level must be one of high, medium, low, none. "
    "Use event_visible only when visible evidence clearly supports an event. "
    "Use normal_context for normal traffic or no clear abnormal event. "
    "Use uncertain for ambiguity without clear proof."
)


def _find_spec(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _normalize_rel_path(path_value: str | None) -> str | None:
    if not path_value:
        return None
    normalized = str(path_value).strip().replace("\\", "/")
    return normalized or None


def _resolve_run_path(run_dir: Path, path_value: str | None) -> Path | None:
    normalized = _normalize_rel_path(path_value)
    if not normalized:
        return None
    path = Path(normalized)
    if path.is_absolute():
        return path
    return (run_dir / path).resolve()


def _try_parse_json(raw_text: str) -> tuple[bool, dict[str, Any] | None]:
    cleaned = raw_text.strip()
    if not cleaned:
        return False, None
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        payload = json.loads(cleaned)
        return isinstance(payload, dict), payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(cleaned[start : end + 1])
                return isinstance(payload, dict), payload if isinstance(payload, dict) else None
            except json.JSONDecodeError:
                return False, None
    return False, None


def _cuda_memory_mb(torch_module: Any) -> float | None:
    if not torch_module.cuda.is_available():
        return None
    return round(float(torch_module.cuda.memory_allocated()) / (1024**2), 2)


def _step11_5_maps(run_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    candidate_map: dict[str, dict[str, Any]] = {}
    candidates_path = run_dir / "11_5_vlm_filtered_event_candidates.json"
    report_path = run_dir / "11_5_vlm_filter_report.json"
    candidate_payload: dict[str, Any] = {}
    report_payload: dict[str, Any] = {}
    if candidates_path.exists():
        candidate_payload = read_json(candidates_path)
        for item in list(candidate_payload.get("candidate_events", [])):
            candidate_id = str(item.get("candidate_event_id", "") or "")
            if candidate_id:
                candidate_map[candidate_id] = item
    if report_path.exists():
        report_payload = read_json(report_path)
    return candidate_map, candidate_payload, report_payload


def _choose_review_image(vlm_input: dict[str, Any], run_dir: Path, require_strip: bool) -> tuple[str | None, Path | None, str | None]:
    media = dict(vlm_input.get("media", {}))
    strip_path = _normalize_rel_path(media.get("temporal_strip_path"))
    contact_sheet_path = _normalize_rel_path(media.get("contact_sheet_path"))
    resolved_strip = _resolve_run_path(run_dir, strip_path)
    resolved_contact_sheet = _resolve_run_path(run_dir, contact_sheet_path)
    if resolved_strip is not None and resolved_strip.exists():
        return strip_path, resolved_strip, "temporal_strip"
    if require_strip:
        return None, None, None
    if resolved_contact_sheet is not None and resolved_contact_sheet.exists():
        return contact_sheet_path, resolved_contact_sheet, "contact_sheet"
    return None, None, None


def _summarize_step11_5_context(source_candidate_ids: list[str], step11_5_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    decisions: list[str] = []
    visible_types: list[str] = []
    reasons: list[str] = []
    event_likelihoods: list[float] = []
    final_truths: list[str] = []
    for candidate_id in source_candidate_ids:
        item = step11_5_map.get(candidate_id)
        if not item:
            continue
        vlm_filter = dict(item.get("vlm_filter", {}))
        decision = str(vlm_filter.get("decision", "") or "")
        if decision:
            decisions.append(decision)
        visible_type = str(vlm_filter.get("visible_event_type", "") or "")
        if visible_type:
            visible_types.append(visible_type)
        reason = str(vlm_filter.get("short_reason", "") or "").strip()
        if reason:
            reasons.append(reason)
        likelihood = vlm_filter.get("event_likelihood")
        if isinstance(likelihood, (int, float)):
            event_likelihoods.append(float(likelihood))
        final_truth = str(item.get("final_event_truth", "") or "")
        if final_truth:
            final_truths.append(final_truth)
    return {
        "source": "11_5_vlm_filtered_event_candidates.json" if decisions or reasons or final_truths else None,
        "candidate_count_found": len(decisions),
        "filter_decisions": decisions,
        "visible_event_types": visible_types,
        "short_reasons": reasons[:5],
        "event_likelihood_average": round(sum(event_likelihoods) / len(event_likelihoods), 6) if event_likelihoods else None,
        "final_event_truths": final_truths,
        "all_normalish": bool(decisions) and all(decision in NORMALISH_STEP11_5_DECISIONS for decision in decisions),
    }


def _contains_any_term(text: str, phrases: tuple[str, ...]) -> bool:
    normalized = text.lower()
    return any(phrase in normalized for phrase in phrases)


def _normalize_objects_seen(raw_value: Any) -> list[str]:
    if isinstance(raw_value, list):
        values = raw_value
    elif isinstance(raw_value, str):
        values = [part.strip() for part in raw_value.split(",")]
    else:
        values = []
    normalized: list[str] = []
    for item in values:
        value = str(item or "").strip().lower()
        if value and value not in normalized:
            normalized.append(value)
    return normalized[:12]


def _normalize_review_payload(
    *,
    vlm_input: dict[str, Any],
    parsed_payload: dict[str, Any] | None,
    raw_output_text: str,
    parsed_ok: bool,
    inference_time_seconds: float,
    image_path_used: str | None,
    image_source_type: str | None,
    step11_5_context: dict[str, Any],
) -> dict[str, Any]:
    payload = parsed_payload or {}
    review_decision = str(payload.get("review_decision", "uncertain") or "").strip().lower()
    if review_decision not in ALLOWED_REVIEW_DECISIONS:
        review_decision = "uncertain"
    event_visible = bool(payload.get("event_visible")) if isinstance(payload.get("event_visible"), bool) else review_decision == "event_visible"
    event_type = str(payload.get("event_type", "uncertain") or "").strip().lower()
    if event_type not in ALLOWED_EVENT_TYPES:
        event_type = "uncertain"
    risk_level = str(payload.get("risk_level", "low") or "").strip().lower()
    if risk_level not in ALLOWED_RISK_LEVELS:
        risk_level = "low"
    summary_caption = str(payload.get("summary_caption", "") or "").strip() or "Review did not produce a concise caption."
    what_is_visible = str(payload.get("what_is_visible", "") or "").strip() or "Visible scene description was not provided."
    why_decision = str(payload.get("why_decision", "") or "").strip() or "Decision rationale was not provided."
    needs_human_review = bool(payload.get("needs_human_review")) if isinstance(payload.get("needs_human_review"), bool) else review_decision == "uncertain"
    try:
        confidence = float(payload.get("confidence", 0.5) or 0.5)
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    objects_seen = _normalize_objects_seen(payload.get("objects_seen"))

    combined_reasoning = " ".join([summary_caption, what_is_visible, why_decision]).lower()
    mentions_abnormal = _contains_any_term(combined_reasoning, ABNORMAL_TERMS)
    mentions_normal_traffic = _contains_any_term(combined_reasoning, NORMAL_TRAFFIC_TERMS)
    step11_5_all_normalish = bool(step11_5_context.get("all_normalish"))

    if review_decision == "event_visible":
        event_visible = True
    if event_visible and review_decision != "event_visible":
        review_decision = "event_visible"
    if review_decision == "normal_context":
        event_visible = False
        event_type = "normal_traffic"
        risk_level = "none"
    if review_decision == "uncertain" and risk_level == "none":
        risk_level = "low"
    if not event_visible and review_decision == "event_visible":
        review_decision = "uncertain"
    if review_decision == "event_visible" and not mentions_abnormal:
        if step11_5_all_normalish or mentions_normal_traffic:
            review_decision = "normal_context"
            event_visible = False
            event_type = "normal_traffic"
            risk_level = "none"
            needs_human_review = False
            confidence = min(confidence, 0.45)
            why_decision = f"{why_decision} Downgraded to normal context because no clear visible abnormal evidence was described."
    if step11_5_all_normalish and review_decision == "uncertain" and not mentions_abnormal:
        review_decision = "normal_context"
        event_visible = False
        event_type = "normal_traffic"
        risk_level = "none"
        needs_human_review = False
        confidence = min(confidence, 0.45)
    if review_decision == "normal_context" and confidence > 0.69:
        confidence = 0.69
    if review_decision == "uncertain" and confidence < 0.30:
        confidence = 0.30
    if review_decision == "event_visible" and confidence < 0.70:
        confidence = 0.70
    if review_decision == "normal_context" and event_type != "normal_traffic":
        event_type = "normal_traffic"
    if review_decision == "event_visible" and risk_level == "none":
        risk_level = "medium"

    return {
        "vlm_input_id": str(vlm_input.get("vlm_input_id", "") or ""),
        "review_decision": review_decision,
        "event_visible": event_visible,
        "event_type": event_type,
        "risk_level": risk_level,
        "summary_caption": summary_caption,
        "what_is_visible": what_is_visible,
        "why_decision": why_decision,
        "objects_seen": objects_seen,
        "needs_human_review": needs_human_review,
        "confidence": round(confidence, 6),
        "image_path_used": image_path_used,
        "image_source_type": image_source_type,
        "raw_output_text": raw_output_text,
        "parsed_json_ok": parsed_ok,
        "inference_time_seconds": round(inference_time_seconds, 3),
    }


def _build_prompt(vlm_input: dict[str, Any], step11_5_context: dict[str, Any]) -> str:
    source_event_types = ", ".join(str(item).replace("_", " ") for item in list(vlm_input.get("source_event_types", [])) if str(item).strip())
    if not source_event_types:
        source_event_types = "unknown candidate"
    step11_5_decisions = ", ".join(step11_5_context.get("filter_decisions", [])) or "not available"
    best_timestamp_text = str(vlm_input.get("best_timestamp_text", "") or "")
    return (
        f"{REVIEW_PROMPT_TEMPLATE} "
        f"VLM input id: {vlm_input.get('vlm_input_id', '')}. "
        f"Candidate hint types: {source_event_types}. "
        f"Best timestamp: {best_timestamp_text}. "
        f"Step 11.5 filter decisions: {step11_5_decisions}. "
        f"Answer for vlm_input_id={vlm_input.get('vlm_input_id', '')}."
    )


def _cache_key(vlm_input_id: str, image_path_used: str | None, model_name: str) -> str:
    raw = f"{vlm_input_id}|{image_path_used or ''}|{model_name}|{PROMPT_VERSION}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _load_model_components(review_config: dict[str, Any]) -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not _find_spec("qwen_vl_utils"):
        raise RuntimeError("qwen_vl_utils is required for Step 14 but is not installed.")
    if not _find_spec("bitsandbytes"):
        raise RuntimeError("bitsandbytes is required for Step 14 4-bit Qwen inference but is not installed.")
    from qwen_vl_utils import process_vision_info

    model_path = str(review_config["model_path"])
    if "3B" in model_path:
        raise RuntimeError("Step 14 is configured with a 3B model path. Qwen2.5-VL-7B-Instruct is required.")
    _decision = resolve_device(
        component_name="Step 14 local Qwen",
        override_env_names=(ENV_STEP14_DEVICE,),
        require_cuda=True,
    )
    load_config = build_qwen_4bit_load_config(model_path, torch_module=torch)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    memory_before = capture_cuda_memory_snapshot(torch)

    load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": True,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if load_config["quantization_config"] is not None:
        model_kwargs["quantization_config"] = load_config["quantization_config"]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
    model_load_time_seconds = time.perf_counter() - load_start
    verification = verify_model_loaded_in_4bit(model)
    memory_after = capture_cuda_memory_snapshot(torch)
    return processor, model, process_vision_info, {
        "model_load_time_seconds": model_load_time_seconds,
        "precision_label": str(load_config["precision_label"]),
        "checkpoint_type": str(load_config["checkpoint_type"]),
        "compute_dtype_name": str(load_config["compute_dtype_name"]),
        "quantization_type": str(load_config["quantization_type"]),
        "double_quant": bool(load_config["double_quant"]),
        "is_loaded_in_4bit": bool(verification["is_loaded_in_4bit"]),
        "linear_4bit_module_count": int(verification["linear_4bit_module_count"]),
        "cuda_memory_allocated_before_load_mb": memory_before["allocated_mb"],
        "cuda_memory_reserved_before_load_mb": memory_before["reserved_mb"],
        "cuda_memory_allocated_after_load_mb": memory_after["allocated_mb"],
        "cuda_memory_reserved_after_load_mb": memory_after["reserved_mb"],
        "processor_class": str(load_config["processor_info"]["processor_class"]),
    }


def _run_single_inference(
    *,
    processor: Any,
    model: Any,
    process_vision_info: Any,
    image_path: Path,
    prompt_text: str,
    max_new_tokens: int,
) -> tuple[str, bool, dict[str, Any] | None, float]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    model_inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    model_inputs = model_inputs.to(model.device)
    infer_start = time.perf_counter()
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    trimmed_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    decoded = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    inference_time_seconds = time.perf_counter() - infer_start
    raw_output_text = decoded[0].strip() if decoded else ""
    parsed_ok, parsed_payload = _try_parse_json(raw_output_text)
    return raw_output_text, parsed_ok, parsed_payload, inference_time_seconds


def _build_flat_review(review_item: dict[str, Any]) -> dict[str, Any]:
    model_review = dict(review_item.get("model_review", {}))
    return {
        "vlm_input_id": review_item.get("vlm_input_id"),
        "source_candidate_ids": review_item.get("source_candidate_ids"),
        "best_timestamp_text": review_item.get("best_timestamp_text"),
        "review_decision": model_review.get("review_decision"),
        "event_visible": model_review.get("event_visible"),
        "event_type": model_review.get("event_type"),
        "risk_level": model_review.get("risk_level"),
        "confidence": model_review.get("confidence"),
        "summary_caption": model_review.get("summary_caption"),
        "needs_human_review": model_review.get("needs_human_review"),
        "temporal_strip_path": review_item.get("temporal_strip_path"),
    }


def _build_final_summary(
    *,
    video_info: dict[str, Any],
    reviews: list[dict[str, Any]],
) -> dict[str, Any]:
    model_reviews = [dict(item.get("model_review", {})) for item in reviews]
    event_reviews = [item for item in reviews if dict(item.get("model_review", {})).get("event_visible") is True]
    normal_count = sum(1 for item in model_reviews if item.get("review_decision") == "normal_context")
    uncertain_count = sum(1 for item in model_reviews if item.get("review_decision") == "uncertain")
    risk_counts = Counter(str(item.get("risk_level", "none") or "none") for item in model_reviews)

    if not event_reviews:
        return {
            "overall_status": "normal_no_clear_event_detected",
            "headline": "No clear abnormal event detected",
            "summary": "The reviewed moments appear to show normal traffic context. No clear collision, accident, or suspicious event was visible in the selected review frames.",
            "event_count": 0,
            "normal_context_count": normal_count,
            "uncertain_count": uncertain_count,
            "high_risk_event_count": 0,
            "medium_risk_event_count": 0,
            "low_risk_event_count": 0,
            "recommended_action": "No immediate action from reviewed moments. Human review optional if required.",
            "video_name": video_info.get("video_name"),
            "duration_text": video_info.get("duration_text"),
        }

    key_events: list[dict[str, Any]] = []
    timeline: list[dict[str, Any]] = []
    for review in event_reviews:
        model_review = dict(review.get("model_review", {}))
        key_events.append(
            {
                "vlm_input_id": review.get("vlm_input_id"),
                "timestamp": review.get("best_timestamp_text"),
                "event_type": model_review.get("event_type"),
                "risk_level": model_review.get("risk_level"),
                "summary_caption": model_review.get("summary_caption"),
            }
        )
        timeline.append(
            {
                "timestamp": review.get("best_timestamp_text"),
                "caption": model_review.get("summary_caption"),
            }
        )

    collision_like_events = [
        item
        for item in key_events
        if str(item.get("event_type", "") or "").strip().lower() in {"collision", "near_miss", "sudden_stop"}
    ]
    headline = "Visible event evidence detected in reviewed moments"
    summary = "One or more reviewed moments show visible event evidence that should be retained for follow-up review."
    overall_status = "event_visible_in_reviewed_moments"
    if collision_like_events:
        event_label = "collision"
        if any(str(item.get("event_type", "") or "").strip().lower() == "near_miss" for item in collision_like_events):
            event_label = "collision / near-miss"
        timestamps = ", ".join(str(item.get("timestamp") or "") for item in collision_like_events[:6] if str(item.get("timestamp") or ""))
        headline = "Collision evidence detected in reviewed moments"
        summary = (
            f"The reviewed moments show visible {event_label} evidence. "
            f"Key timestamps: {timestamps or 'see key events'}."
        )
        overall_status = "collision_detected_in_reviewed_moments"
    return {
        "overall_status": overall_status,
        "headline": headline,
        "summary": summary,
        "event_count": len(event_reviews),
        "normal_context_count": normal_count,
        "uncertain_count": uncertain_count,
        "high_risk_event_count": int(risk_counts.get("high", 0)),
        "medium_risk_event_count": int(risk_counts.get("medium", 0)),
        "low_risk_event_count": int(risk_counts.get("low", 0)),
        "key_events": key_events,
        "timestamps": [item.get("timestamp") for item in key_events],
        "risk_levels": [item.get("risk_level") for item in key_events],
        "short_timeline": timeline,
        "recommended_action": "Human review recommended for the visible event moments.",
        "video_name": video_info.get("video_name"),
        "duration_text": video_info.get("duration_text"),
    }


def run_vlm_event_review(
    *,
    run_dir: Path,
    review_config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    payload = read_json(run_dir / SOURCE_FILE)
    input_report = read_json(run_dir / "13_vlm_event_input_report.json")
    selected_candidates_payload = read_json(run_dir / "12_selected_top_event_candidates.json")
    video_info = read_json(run_dir / "01_video_info.json")
    step11_5_map, step11_5_payload, step11_5_report = _step11_5_maps(run_dir)

    source_inputs = list(payload.get("vlm_inputs", []))
    max_inputs = int(review_config["max_inputs"])
    vlm_inputs = source_inputs[:max_inputs]
    warnings: list[str] = []
    backend = str(review_config.get("vlm_backend", "local_qwen") or "local_qwen").strip().lower()
    api_provider = str(review_config.get("api_provider", "") or "")
    api_model = str(review_config.get("api_model", "") or "")
    cache_dir = run_dir / "14_vlm_review_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    total_inference_time_seconds = 0.0
    model_load_time_seconds = 0.0
    model_dtype = ""
    cuda_after_load_mb: float | None = None
    cuda_after_mb: float | None = None
    cuda_reserved_after_load_mb: float | None = None
    cuda_reserved_after_mb: float | None = None
    cuda_before_load_mb: float | None = None
    cuda_reserved_before_load_mb: float | None = None
    checkpoint_type: str | None = None
    is_loaded_in_4bit: bool | None = None
    quantization_type: str | None = None
    double_quant: bool | None = None
    compute_dtype_name: str | None = None
    linear_4bit_module_count = 0
    cache_used_count = 0
    used_strip_count = 0
    used_contact_sheet_count = 0
    api_success_count = 0
    api_failed_count = 0
    parse_success_count = 0
    parse_failed_count = 0
    total_latency_seconds = 0.0

    review_candidates: list[dict[str, Any]] = []
    skipped_inputs: list[dict[str, Any]] = []
    for vlm_input in vlm_inputs:
        image_path_used, resolved_image_path, image_source_type = _choose_review_image(
            vlm_input,
            run_dir,
            bool(review_config["require_strip"]),
        )
        if image_source_type == "temporal_strip":
            used_strip_count += 1
        elif image_source_type == "contact_sheet":
            used_contact_sheet_count += 1
        if resolved_image_path is None:
            warning = f"{vlm_input.get('vlm_input_id')}: no strip/contact sheet available for Step 14 review."
            warnings.append(warning)
            skipped_inputs.append(
                {
                    "vlm_input_id": vlm_input.get("vlm_input_id"),
                    "warning": warning,
                }
            )
            continue
        review_candidates.append(
            {
                "vlm_input": vlm_input,
                "image_path_used": image_path_used,
                "resolved_image_path": resolved_image_path,
                "image_source_type": image_source_type,
            }
        )

    uncached_review_count = 0
    for item in review_candidates:
        vlm_input_id = str(dict(item["vlm_input"]).get("vlm_input_id", "") or "")
        cache_path = cache_dir / f"{_cache_key(vlm_input_id, item['image_path_used'], MODEL_NAME)}.json"
        if not (bool(review_config["use_cache"]) and cache_path.exists()):
            uncached_review_count += 1

    processor = model = process_vision_info = None
    try:
        if backend == "disabled":
            warnings.append("VLM skipped because TD_CASE2_VLM_BACKEND=disabled.")
        elif backend == "local_qwen" and uncached_review_count > 0:
            processor, model, process_vision_info, load_metadata = _load_model_components(
                review_config
            )
            model_load_time_seconds = float(load_metadata["model_load_time_seconds"])
            model_dtype = str(load_metadata["precision_label"])
            checkpoint_type = str(load_metadata["checkpoint_type"])
            compute_dtype_name = str(load_metadata["compute_dtype_name"])
            is_loaded_in_4bit = bool(load_metadata["is_loaded_in_4bit"])
            quantization_type = str(load_metadata["quantization_type"])
            double_quant = bool(load_metadata["double_quant"])
            linear_4bit_module_count = int(load_metadata["linear_4bit_module_count"])
            cuda_before_load_mb = load_metadata["cuda_memory_allocated_before_load_mb"]
            cuda_reserved_before_load_mb = load_metadata["cuda_memory_reserved_before_load_mb"]
            cuda_after_load_mb = load_metadata["cuda_memory_allocated_after_load_mb"]
            cuda_reserved_after_load_mb = load_metadata["cuda_memory_reserved_after_load_mb"]

        reviews: list[dict[str, Any]] = []
        flat_reviews: list[dict[str, Any]] = []

        if backend == "disabled":
            final_summary_payload = {
            "overall_status": "vlm_skipped",
            "headline": "VLM review skipped",
            "summary": "Step 14 VLM review was skipped because TD_CASE2_VLM_BACKEND=disabled.",
            "event_count": 0,
            "normal_context_count": 0,
            "uncertain_count": 0,
            "high_risk_event_count": 0,
            "medium_risk_event_count": 0,
            "low_risk_event_count": 0,
            "recommended_action": "Run the separate VLM pipeline with local_qwen or api_qwen when event summaries are needed.",
            "video_name": video_info.get("video_name"),
            "duration_text": video_info.get("duration_text"),
        }
            output_payload = {
            "status": "skipped",
            "source_file": SOURCE_FILE,
            "model": MODEL_NAME,
            "vlm_backend": backend,
            "config": {**dict(review_config), "model_path": str(review_config["model_path"]), "model_dtype": model_dtype, "prompt_version": PROMPT_VERSION},
            "summary": {
                "inputs_loaded": len(vlm_inputs),
                "inputs_reviewed": 0,
                "inputs_skipped": len(vlm_inputs),
                "event_visible_count": 0,
                "normal_context_count": 0,
                "uncertain_count": 0,
                "used_temporal_strip_count": used_strip_count,
                "used_contact_sheet_count": used_contact_sheet_count,
                "ready_for_demo_report_ui": True,
            },
            "reviews": [],
        }
            report_payload = {
            "status": "skipped",
            "vlm_backend": backend,
            "api_provider": None,
            "api_model": None,
            "inputs_loaded": len(vlm_inputs),
            "inputs_reviewed": 0,
            "inputs_skipped": len(vlm_inputs),
            "event_visible_count": 0,
            "normal_context_count": 0,
            "uncertain_count": 0,
            "risk_counts": {},
            "event_type_counts": {},
            "model_path": str(review_config["model_path"]),
            "model_load_time_seconds": 0.0,
            "checkpoint_type": None,
            "model_precision_label": "",
            "compute_dtype": None,
            "is_loaded_in_4bit": None,
            "quantization_type": None,
            "double_quant": None,
            "linear_4bit_module_count": 0,
            "api_success_count": 0,
            "api_failed_count": 0,
            "parse_success_count": 0,
            "parse_failed_count": 0,
            "average_latency_seconds": 0.0,
            "total_latency_seconds": 0.0,
            "estimated_cost_usd": None,
            "total_inference_time_seconds": 0.0,
            "average_inference_time_seconds": 0.0,
            "cuda_memory_allocated_before_load_mb": None,
            "cuda_memory_reserved_before_load_mb": None,
            "cuda_memory_allocated_after_load_mb": None,
            "cuda_memory_reserved_after_load_mb": None,
            "cuda_memory_allocated_after_mb": None,
            "cache_used_count": 0,
            "used_temporal_strip_count": used_strip_count,
            "used_contact_sheet_count": used_contact_sheet_count,
            "source_inputs_ready_for_vlm": input_report.get("inputs_ready_for_vlm"),
            "selected_candidates_count": selected_candidates_payload.get("selected_count"),
            "step11_5_available": bool(step11_5_payload),
            "step11_5_filter_summary": step11_5_report.get("decision_counts", {}),
            "warnings": warnings,
            "recommendation": final_summary_payload.get("recommended_action"),
        }
            write_json(run_dir / INPUT_FILE_PRIMARY, output_payload)
            write_json_any(run_dir / "14_vlm_event_reviews_flat.json", [])
            write_json(run_dir / "14_final_video_summary.json", final_summary_payload)
            write_json(run_dir / "14_vlm_event_review_report.json", report_payload)
            return output_payload, [], final_summary_payload, report_payload

        for item in review_candidates:
            vlm_input = dict(item["vlm_input"])
            vlm_input_id = str(vlm_input.get("vlm_input_id", "") or "")
            image_path_used = item["image_path_used"]
            resolved_image_path = item["resolved_image_path"]
            image_source_type = item["image_source_type"]
            source_candidate_ids = [str(candidate_id or "") for candidate_id in list(vlm_input.get("source_candidate_ids", []))]
            step11_5_context = _summarize_step11_5_context(source_candidate_ids, step11_5_map)
            prompt_text = _build_prompt(vlm_input, step11_5_context)
            cache_path = cache_dir / f"{_cache_key(vlm_input_id, image_path_used, MODEL_NAME)}.json"
            cached_payload: dict[str, Any] | None = None

            if bool(review_config["use_cache"]) and cache_path.exists():
                try:
                    cached_payload = read_json(cache_path)
                except Exception as exc:
                    warnings.append(f"{vlm_input_id}: cache read failed ({exc}). Re-running inference.")

            if cached_payload is None:
                if backend == "api_qwen":
                    api_result = call_qwen_api_with_image(prompt_text=prompt_text, image_path=resolved_image_path)
                    total_latency_seconds += float(api_result.get("latency_seconds", 0.0) or 0.0)
                    if api_result.get("status") == "success":
                        api_success_count += 1
                    else:
                        api_failed_count += 1
                        warnings.append(f"{vlm_input_id}: API failure ({api_result.get('error_message')}).")
                    parsed_ok, parsed_payload = _try_parse_json(str(api_result.get("assistant_text", "") or ""))
                    if parsed_ok:
                        parse_success_count += 1
                    else:
                        parse_failed_count += 1
                    normalized_review = _normalize_review_payload(
                        vlm_input=vlm_input,
                        parsed_payload=parsed_payload,
                        raw_output_text=str(api_result.get("assistant_text", "") or ""),
                        parsed_ok=parsed_ok,
                        inference_time_seconds=float(api_result.get("latency_seconds", 0.0) or 0.0),
                        image_path_used=image_path_used,
                        image_source_type=image_source_type,
                        step11_5_context=step11_5_context,
                    )
                    normalized_review["api_request_metadata"] = api_result.get("request_metadata")
                    normalized_review["api_provider"] = api_result.get("provider")
                    normalized_review["api_model"] = api_result.get("model")
                    normalized_review["api_raw_response_text"] = api_result.get("raw_response_text")
                    normalized_review["api_error_message"] = api_result.get("error_message")
                else:
                    raw_output_text, parsed_ok, parsed_payload, inference_time_seconds = _run_single_inference(
                        processor=processor,
                        model=model,
                        process_vision_info=process_vision_info,
                        image_path=resolved_image_path,
                        prompt_text=prompt_text,
                        max_new_tokens=int(review_config["max_new_tokens"]),
                    )
                    if parsed_ok:
                        parse_success_count += 1
                    else:
                        parse_failed_count += 1
                    normalized_review = _normalize_review_payload(
                        vlm_input=vlm_input,
                        parsed_payload=parsed_payload,
                        raw_output_text=raw_output_text,
                        parsed_ok=parsed_ok,
                        inference_time_seconds=inference_time_seconds,
                        image_path_used=image_path_used,
                        image_source_type=image_source_type,
                        step11_5_context=step11_5_context,
                    )
                cached_payload = normalized_review
                if bool(review_config["use_cache"]):
                    write_json(cache_path, cached_payload)
            else:
                cache_used_count += 1
                if bool(cached_payload.get("parsed_json_ok")):
                    parse_success_count += 1
                else:
                    parse_failed_count += 1
                normalized_review = _normalize_review_payload(
                    vlm_input=vlm_input,
                    parsed_payload=cached_payload,
                    raw_output_text=str(cached_payload.get("raw_output_text", "") or ""),
                    parsed_ok=bool(cached_payload.get("parsed_json_ok")),
                    inference_time_seconds=float(cached_payload.get("inference_time_seconds", 0.0) or 0.0),
                    image_path_used=image_path_used,
                    image_source_type=image_source_type,
                    step11_5_context=step11_5_context,
                )

            total_inference_time_seconds += float(normalized_review.get("inference_time_seconds", 0.0) or 0.0)
            review_record = {
            "vlm_input_id": vlm_input_id,
            "source_candidate_ids": source_candidate_ids,
            "source_event_types": list(vlm_input.get("source_event_types", [])),
            "best_timestamp_text": vlm_input.get("best_timestamp_text"),
            "context_start_seconds": vlm_input.get("context_start_seconds"),
            "context_end_seconds": vlm_input.get("context_end_seconds"),
            "temporal_strip_path": dict(vlm_input.get("media", {})).get("temporal_strip_path"),
            "contact_sheet_path": dict(vlm_input.get("media", {})).get("contact_sheet_path"),
            "step11_5_filter_context": step11_5_context,
            "model_review": {
                "vlm_input_id": normalized_review.get("vlm_input_id"),
                "review_decision": normalized_review.get("review_decision"),
                "event_visible": normalized_review.get("event_visible"),
                "event_type": normalized_review.get("event_type"),
                "risk_level": normalized_review.get("risk_level"),
                "summary_caption": normalized_review.get("summary_caption"),
                "what_is_visible": normalized_review.get("what_is_visible"),
                "why_decision": normalized_review.get("why_decision"),
                "objects_seen": normalized_review.get("objects_seen"),
                "needs_human_review": normalized_review.get("needs_human_review"),
                "confidence": normalized_review.get("confidence"),
            },
            "raw_output_text": normalized_review.get("raw_output_text"),
            "parsed_json_ok": normalized_review.get("parsed_json_ok"),
            "inference_time_seconds": normalized_review.get("inference_time_seconds"),
            "image_path_used": normalized_review.get("image_path_used"),
            "image_source_type": normalized_review.get("image_source_type"),
            "ready_for_final_summary": True,
        }
            reviews.append(review_record)
            flat_reviews.append(_build_flat_review(review_record))

        if model is not None:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    memory_after = capture_cuda_memory_snapshot(torch)
                    cuda_after_mb = memory_after["allocated_mb"]
                    cuda_reserved_after_mb = memory_after["reserved_mb"]
            except Exception:
                cuda_after_mb = None
                cuda_reserved_after_mb = None

        review_decision_counts = Counter(str(item.get("model_review", {}).get("review_decision", "uncertain")) for item in reviews)
        risk_counts = Counter(str(item.get("model_review", {}).get("risk_level", "none")) for item in reviews)
        event_type_counts = Counter(str(item.get("model_review", {}).get("event_type", "uncertain")) for item in reviews)
        average_inference_time_seconds = round(total_inference_time_seconds / len(reviews), 3) if reviews else 0.0
        final_summary_payload = _build_final_summary(video_info=video_info, reviews=reviews)

        output_payload = {
            "status": "success",
            "source_file": SOURCE_FILE,
            "model": MODEL_NAME,
            "config": {
                **dict(review_config),
                "model_path": str(review_config["model_path"]),
                "model_dtype": model_dtype,
                "prompt_version": PROMPT_VERSION,
            },
            "summary": {
                "inputs_loaded": len(vlm_inputs),
                "inputs_reviewed": len(reviews),
                "inputs_skipped": len(skipped_inputs),
                "event_visible_count": int(sum(1 for item in reviews if item.get("model_review", {}).get("event_visible") is True)),
                "normal_context_count": int(review_decision_counts.get("normal_context", 0)),
                "uncertain_count": int(review_decision_counts.get("uncertain", 0)),
                "used_temporal_strip_count": used_strip_count,
                "used_contact_sheet_count": used_contact_sheet_count,
                "ready_for_demo_report_ui": True,
            },
            "reviews": reviews,
        }
        report_payload = {
            "status": "success",
            "vlm_backend": backend,
            "api_provider": api_provider if backend == "api_qwen" else None,
            "api_model": api_model if backend == "api_qwen" else None,
            "inputs_loaded": len(vlm_inputs),
            "inputs_reviewed": len(reviews),
            "inputs_skipped": len(skipped_inputs),
            "event_visible_count": int(sum(1 for item in reviews if item.get("model_review", {}).get("event_visible") is True)),
            "normal_context_count": int(review_decision_counts.get("normal_context", 0)),
            "uncertain_count": int(review_decision_counts.get("uncertain", 0)),
            "risk_counts": dict(risk_counts),
            "event_type_counts": dict(event_type_counts),
            "model_path": str(review_config["model_path"]),
            "model_device": str(getattr(model, "device", "cuda:0")) if model is not None else None,
            "model_load_time_seconds": round(model_load_time_seconds, 3),
            "checkpoint_type": checkpoint_type,
            "model_precision_label": model_dtype,
            "compute_dtype": compute_dtype_name,
            "is_loaded_in_4bit": is_loaded_in_4bit,
            "quantization_type": quantization_type,
            "double_quant": double_quant,
            "linear_4bit_module_count": linear_4bit_module_count,
            "api_success_count": api_success_count,
            "api_failed_count": api_failed_count,
            "parse_success_count": parse_success_count,
            "parse_failed_count": parse_failed_count,
            "average_latency_seconds": round(total_latency_seconds / len(reviews), 3) if reviews else 0.0,
            "total_latency_seconds": round(total_latency_seconds, 3),
            "estimated_cost_usd": None,
            "total_inference_time_seconds": round(total_inference_time_seconds, 3),
            "average_inference_time_seconds": average_inference_time_seconds,
            "cuda_memory_allocated_before_load_mb": cuda_before_load_mb,
            "cuda_memory_reserved_before_load_mb": cuda_reserved_before_load_mb,
            "cuda_memory_allocated_after_load_mb": cuda_after_load_mb,
            "cuda_memory_reserved_after_load_mb": cuda_reserved_after_load_mb,
            "cuda_memory_allocated_after_mb": cuda_after_mb if cuda_after_mb is not None else cuda_after_load_mb,
            "cache_used_count": cache_used_count,
            "used_temporal_strip_count": used_strip_count,
            "used_contact_sheet_count": used_contact_sheet_count,
            "source_inputs_ready_for_vlm": input_report.get("inputs_ready_for_vlm"),
            "selected_candidates_count": selected_candidates_payload.get("selected_count"),
            "step11_5_available": bool(step11_5_payload),
            "step11_5_filter_summary": step11_5_report.get("decision_counts", {}),
            "warnings": warnings,
            "recommendation": final_summary_payload.get("recommended_action"),
        }

        write_json(run_dir / INPUT_FILE_PRIMARY, output_payload)
        write_json_any(run_dir / "14_vlm_event_reviews_flat.json", flat_reviews)
        write_json(run_dir / "14_final_video_summary.json", final_summary_payload)
        write_json(run_dir / "14_vlm_event_review_report.json", report_payload)
        return output_payload, flat_reviews, final_summary_payload, report_payload
    finally:
        if model is not None or processor is not None:
            loaded_model = model
            loaded_processor = processor
            loaded_process_vision_info = process_vision_info
            model = None
            processor = None
            process_vision_info = None
            del loaded_model, loaded_processor, loaded_process_vision_info
            release_qwen_resources()
