from __future__ import annotations

import hashlib
import importlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from config import ENV_STEP11_5_DEVICE
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


INPUT_FILE_PRIMARY = "11_5_vlm_filtered_event_candidates.json"
SOURCE_FILE = "11_full_scene_event_candidates.json"
PROMPT_VERSION = "step11_5_v2"
FILTER_PROMPT = (
    "You are reviewing CCTV/traffic full-scene frames. "
    "The rule engine marked this as a possible event, but it may be false. "
    "Look only at what is visibly present. "
    "Do not assume collision, accident, or suspicious activity unless it is visible. "
    "Decide whether this frame/context shows a meaningful visible abnormal event or strong event evidence worth sending to a larger VLM. "
    "Answer yes only if the image visibly shows a meaningful abnormal event or strong event evidence such as visible collision/contact, "
    "a near miss with clearly dangerous close interaction, a vehicle stopped or blocked in an abnormal way, a person fallen, "
    "a vehicle fallen or overturned, a traffic obstruction, a visible dangerous vehicle-person interaction, or another clear unusual incident. "
    "Answer no for normal scenes such as normal traffic flow, a person riding a motorcycle normally, a vehicle driving normally, "
    "a parked or stopped vehicle with no abnormal context, brake lights alone, crowded but normal traffic, unclear event, "
    "or when only metadata suggests collision but the image does not visibly show it. "
    "Answer uncertain only when there is some visual concern but not enough proof. "
    "Do not answer yes just because a vehicle, motorcycle, person, brake light, or road traffic is visible. "
    "Use event_likelihood ranges: yes usually 0.70 to 1.00, uncertain usually 0.40 to 0.69, no usually 0.00 to 0.39. "
    "Return only JSON with keys decision, event_likelihood, visible_event_type, short_reason, should_keep. "
    "decision must be one of yes, no, uncertain."
)
MODEL_NAME = "Qwen2.5-VL-3B-Instruct"
TRAFFIC_PRIORITY = {
    "possible_collision_or_near_miss": 1.0,
    "sudden_stop": 0.92,
    "vehicle_person_interaction": 0.86,
    "traffic_congestion_or_dense_vehicle_activity": 0.70,
    "unusual_motion_spike": 0.62,
    "object_density_spike": 0.45,
    "track_start_stop_activity": 0.30,
    "stationary_vehicle": 0.20,
}
TRAFFIC_SAFETY_TYPES = {"possible_collision_or_near_miss", "sudden_stop", "vehicle_person_interaction"}


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


def _frame_path_for_candidate(candidate: dict[str, Any], run_dir: Path) -> tuple[str | None, Path | None]:
    representative = dict(candidate.get("representative_frame", {}))
    path_candidates = [representative.get("image_path"), *list(candidate.get("full_frame_paths", []))]
    seen: set[str] = set()
    for path_value in path_candidates:
        normalized = _normalize_rel_path(path_value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        resolved = _resolve_run_path(run_dir, normalized)
        if resolved is not None and resolved.exists():
            return normalized, resolved
    return None, None


def _event_preselection_score(candidate: dict[str, Any]) -> float:
    event_type = str(candidate.get("event_type", "") or "")
    score = float(candidate.get("candidate_score", 0.0) or 0.0) * 10.0
    score += TRAFFIC_PRIORITY.get(event_type, 0.15) * 5.0
    if event_type in TRAFFIC_SAFETY_TYPES:
        score += 2.5
    if dict(candidate.get("representative_frame", {})).get("image_path"):
        score += 1.5
    if list(candidate.get("full_frame_paths", [])):
        score += 1.0
    if event_type == "unusual_motion_spike":
        score += 0.5
    return round(score, 6)


def _preselect_candidates(candidates: list[dict[str, Any]], run_dir: Path, max_candidates_to_check: int) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for candidate in candidates:
        rel_path, resolved_path = _frame_path_for_candidate(candidate, run_dir)
        item = dict(candidate)
        item["_step11_5_image_path"] = rel_path
        item["_step11_5_image_resolved"] = resolved_path
        item["_step11_5_preselection_score"] = _event_preselection_score(item)
        enriched.append(item)

    enriched.sort(
        key=lambda item: (
            -float(item.get("_step11_5_preselection_score", 0.0) or 0.0),
            -float(item.get("candidate_score", 0.0) or 0.0),
            float(item.get("best_timestamp_seconds", 0.0) or 0.0),
        )
    )

    selected: list[dict[str, Any]] = []
    selected_timestamps: list[float] = []
    for candidate in enriched:
        if len(selected) >= max_candidates_to_check:
            break
        timestamp = float(candidate.get("best_timestamp_seconds", 0.0) or 0.0)
        event_type = str(candidate.get("event_type", "") or "")
        image_ok = candidate.get("_step11_5_image_resolved") is not None
        if not image_ok:
            continue
        gap_limit = 5.0 if event_type in TRAFFIC_SAFETY_TYPES else 8.0
        near_duplicate = any(abs(timestamp - existing) < gap_limit for existing in selected_timestamps)
        if near_duplicate and event_type != "unusual_motion_spike":
            continue
        selected.append(candidate)
        selected_timestamps.append(timestamp)

    if len(selected) < max_candidates_to_check:
        selected_ids = {str(item.get("candidate_event_id", "") or "") for item in selected}
        for candidate in enriched:
            if len(selected) >= max_candidates_to_check:
                break
            candidate_id = str(candidate.get("candidate_event_id", "") or "")
            if candidate_id in selected_ids:
                continue
            if candidate.get("_step11_5_image_resolved") is None:
                continue
            selected.append(candidate)
            selected_ids.add(candidate_id)
    return selected


NORMAL_ACTIVITY_PHRASES = (
    "riding a motorcycle",
    "driving on the road",
    "vehicle is visible",
    "car is moving",
    "brake lights are on",
    "traffic is present",
)
ABNORMAL_EVENT_TERMS = (
    "collision",
    "near miss",
    "blocked",
    "stopped dangerously",
    "fallen",
    "accident",
    "obstruction",
    "unusual position",
    "dangerous",
)
EVENT_LIKE_REASON_TERMS = ABNORMAL_EVENT_TERMS + (
    "impact",
    "crash",
    "overturned",
    "down on the road",
    "person on the ground",
    "vehicle-person interaction",
)


def _cache_key(candidate_event_id: str, image_path: str | None, model_name: str) -> str:
    raw = f"{candidate_event_id}|{image_path or ''}|{model_name}|{PROMPT_VERSION}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _normalize_decision(raw_value: Any) -> str:
    value = str(raw_value or "").strip().lower()
    if value in {"yes", "no", "uncertain"}:
        return value
    return "uncertain"


def _contains_any_term(text: str, phrases: tuple[str, ...]) -> bool:
    normalized = text.lower()
    return any(phrase in normalized for phrase in phrases)


def _normalize_decision_and_likelihood(decision: str, event_likelihood: float, short_reason: str) -> tuple[str, float]:
    reason_is_event_like = _contains_any_term(short_reason, EVENT_LIKE_REASON_TERMS)
    if decision == "no" and event_likelihood > 0.39:
        return "no", 0.30
    if decision == "uncertain" and not (0.40 <= event_likelihood <= 0.69):
        return "uncertain", 0.55
    if decision == "yes" and event_likelihood < 0.70:
        if reason_is_event_like:
            return "yes", 0.70
        if event_likelihood >= 0.40:
            return "uncertain", 0.55
        return "no", 0.30
    return decision, event_likelihood


def _apply_reason_safety_correction(decision: str, short_reason: str, should_keep: bool) -> tuple[str, str, bool]:
    reason_mentions_normal_only = _contains_any_term(short_reason, NORMAL_ACTIVITY_PHRASES)
    reason_mentions_abnormal = _contains_any_term(short_reason, ABNORMAL_EVENT_TERMS)
    if decision == "yes" and reason_mentions_normal_only and not reason_mentions_abnormal:
        corrected_reason = f"{short_reason} This appears to be normal traffic, not a meaningful event."
        return "no", corrected_reason, False
    return decision, short_reason, should_keep


def _normalize_filter_payload(
    *,
    raw_output_text: str,
    parsed_ok: bool,
    parsed_payload: dict[str, Any] | None,
    model_name: str,
    image_path_used: str | None,
    inference_time_seconds: float,
    checked: bool,
) -> dict[str, Any]:
    payload = parsed_payload or {}
    decision = _normalize_decision(payload.get("decision"))
    try:
        event_likelihood = float(payload.get("event_likelihood", 0.5) or 0.5)
    except (TypeError, ValueError):
        event_likelihood = 0.5
    event_likelihood = max(0.0, min(1.0, event_likelihood))
    short_reason = str(payload.get("short_reason", "") or "").strip() or "Model response did not include a clear reason."
    visible_event_type = str(payload.get("visible_event_type", "") or "").strip() or "unclear"
    should_keep_value = payload.get("should_keep")
    if isinstance(should_keep_value, bool):
        should_keep = should_keep_value
    else:
        should_keep = decision in {"yes", "uncertain"}
    decision, event_likelihood = _normalize_decision_and_likelihood(decision, event_likelihood, short_reason)
    should_keep = should_keep and decision in {"yes", "uncertain"}
    decision, short_reason, should_keep = _apply_reason_safety_correction(decision, short_reason, should_keep)
    if decision == "no":
        should_keep = False
        if event_likelihood > 0.39:
            event_likelihood = 0.30
    elif decision == "uncertain":
        should_keep = True
        if not (0.40 <= event_likelihood <= 0.69):
            event_likelihood = 0.55
    elif decision == "yes":
        should_keep = True
        if event_likelihood < 0.70:
            event_likelihood = 0.70
    return {
        "checked": checked,
        "decision": decision,
        "event_likelihood": round(event_likelihood, 6),
        "visible_event_type": visible_event_type,
        "short_reason": short_reason,
        "should_keep": should_keep,
        "model": model_name,
        "image_path_used": image_path_used,
        "inference_time_seconds": round(inference_time_seconds, 3),
        "raw_output_text": raw_output_text,
        "parsed_json_ok": parsed_ok,
    }


def _load_model_components(filter_config: dict[str, Any]) -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not _find_spec("qwen_vl_utils"):
        raise RuntimeError("qwen_vl_utils is required for Step 11.5 but is not installed.")
    if not _find_spec("bitsandbytes"):
        raise RuntimeError("bitsandbytes is required for Step 11.5 4-bit Qwen inference but is not installed.")
    from qwen_vl_utils import process_vision_info

    model_path = str(filter_config["model_path"])
    _decision = resolve_device(
        component_name="Step 11.5 local Qwen",
        override_env_names=(ENV_STEP11_5_DEVICE,),
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
    max_new_tokens: int,
) -> tuple[str, bool, dict[str, Any] | None, float]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": FILTER_PROMPT},
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


def run_lightweight_vlm_filter(
    *,
    run_dir: Path,
    filter_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Filter Step 11 candidates with a lightweight VLM pass."""

    payload = read_json(run_dir / SOURCE_FILE)
    source_candidates = list(payload.get("candidate_events", []))
    checked_candidates = _preselect_candidates(
        source_candidates,
        run_dir,
        int(filter_config["max_candidates_to_check"]),
    )
    checked_ids = {str(item.get("candidate_event_id", "") or "") for item in checked_candidates}
    warnings: list[str] = []
    backend = str(filter_config.get("vlm_backend", "local_qwen") or "local_qwen").strip().lower()
    api_provider = str(filter_config.get("api_provider", "") or "")
    api_model = str(filter_config.get("api_model", "") or "")
    cache_dir = run_dir / "11_5_vlm_filter_cache"
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
    decision_counts: Counter[str] = Counter()
    api_success_count = 0
    api_failed_count = 0
    total_api_latency_seconds = 0.0
    fallback_used = False
    errors_summary: list[str] = []

    by_id: dict[str, dict[str, Any]] = {}
    for original in source_candidates:
        candidate = dict(original)
        candidate["vlm_filter"] = {
            "checked": False,
            "decision": "not_checked",
            "event_likelihood": None,
            "visible_event_type": "",
            "short_reason": "Candidate was not sent to Step 11.5 VLM.",
            "should_keep": False,
            "model": MODEL_NAME,
            "image_path_used": None,
            "inference_time_seconds": 0.0,
            "raw_output_text": "",
            "parsed_json_ok": False,
        }
        by_id[str(candidate.get("candidate_event_id", "") or "")] = candidate

    processor = model = process_vision_info = None
    try:
        if backend == "disabled":
            fallback_used = True
            warnings.append("VLM skipped because TD_CASE2_VLM_BACKEND=disabled. Using deterministic fallback candidate selection.")
        elif backend == "local_qwen" and checked_candidates:
            processor, model, process_vision_info, load_metadata = _load_model_components(
                filter_config
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

        accepted_yes: list[dict[str, Any]] = []
        uncertain_candidates: list[dict[str, Any]] = []
        rejected_no: list[dict[str, Any]] = []
        flat_results: list[dict[str, Any]] = []

        for candidate in checked_candidates:
            candidate_id = str(candidate.get("candidate_event_id", "") or "")
            image_path_used = _normalize_rel_path(candidate.get("_step11_5_image_path"))
            resolved_image_path = candidate.get("_step11_5_image_resolved")
            if resolved_image_path is None:
                warnings.append(f"{candidate_id}: no full-scene image could be resolved for VLM filtering.")
                continue
            cache_path = cache_dir / f"{_cache_key(candidate_id, image_path_used, MODEL_NAME)}.json"
            cache_payload: dict[str, Any] | None = None
            if bool(filter_config["use_cache"]) and cache_path.exists():
                try:
                    cache_payload = read_json(cache_path)
                except Exception as exc:
                    warnings.append(f"{candidate_id}: cache read failed ({exc}). Re-running inference.")

            if cache_payload is None:
                if backend == "disabled":
                    cache_payload = {
                        "checked": False,
                        "decision": "skipped_backend_disabled",
                        "event_likelihood": None,
                        "visible_event_type": "skipped",
                        "short_reason": "VLM skipped because TD_CASE2_VLM_BACKEND=disabled.",
                        "should_keep": False,
                        "model": MODEL_NAME,
                        "image_path_used": image_path_used,
                        "inference_time_seconds": 0.0,
                        "raw_output_text": "",
                        "parsed_json_ok": False,
                    }
                elif backend == "api_qwen":
                    api_result = call_qwen_api_with_image(prompt_text=FILTER_PROMPT, image_path=resolved_image_path)
                    total_api_latency_seconds += float(api_result.get("latency_seconds", 0.0) or 0.0)
                    if api_result.get("status") == "success":
                        api_success_count += 1
                    else:
                        api_failed_count += 1
                        error_message = str(api_result.get("error_message", "") or "Unknown API failure.")
                        errors_summary.append(f"{candidate_id}: {error_message}")
                    parsed_ok, parsed_payload = _try_parse_json(str(api_result.get("assistant_text", "") or ""))
                    cache_payload = _normalize_filter_payload(
                        raw_output_text=str(api_result.get("assistant_text", "") or ""),
                        parsed_ok=parsed_ok,
                        parsed_payload=parsed_payload,
                        model_name=api_model or MODEL_NAME,
                        image_path_used=image_path_used,
                        inference_time_seconds=float(api_result.get("latency_seconds", 0.0) or 0.0),
                        checked=True,
                    )
                    cache_payload["api_request_metadata"] = api_result.get("request_metadata")
                    cache_payload["api_provider"] = api_result.get("provider")
                    cache_payload["api_model"] = api_result.get("model")
                    cache_payload["api_raw_response_text"] = api_result.get("raw_response_text")
                    cache_payload["api_error_message"] = api_result.get("error_message")
                else:
                    raw_output_text, parsed_ok, parsed_payload, inference_time_seconds = _run_single_inference(
                        processor=processor,
                        model=model,
                        process_vision_info=process_vision_info,
                        image_path=resolved_image_path,
                        max_new_tokens=int(filter_config["max_new_tokens"]),
                    )
                    cache_payload = _normalize_filter_payload(
                        raw_output_text=raw_output_text,
                        parsed_ok=parsed_ok,
                        parsed_payload=parsed_payload,
                        model_name=MODEL_NAME,
                        image_path_used=image_path_used,
                        inference_time_seconds=inference_time_seconds,
                        checked=True,
                    )
                if bool(filter_config["use_cache"]):
                    write_json(cache_path, cache_payload)
            else:
                cache_payload = _normalize_filter_payload(
                    raw_output_text=str(cache_payload.get("raw_output_text", "") or ""),
                    parsed_ok=bool(cache_payload.get("parsed_json_ok")),
                    parsed_payload={
                        "decision": cache_payload.get("decision"),
                        "event_likelihood": cache_payload.get("event_likelihood"),
                        "visible_event_type": cache_payload.get("visible_event_type"),
                        "short_reason": cache_payload.get("short_reason"),
                        "should_keep": cache_payload.get("should_keep"),
                    },
                    model_name=MODEL_NAME,
                    image_path_used=image_path_used,
                    inference_time_seconds=float(cache_payload.get("inference_time_seconds", 0.0) or 0.0),
                    checked=True,
                )
            total_inference_time_seconds += float(cache_payload.get("inference_time_seconds", 0.0) or 0.0)

            decision = str(cache_payload.get("decision", "uncertain"))
            decision_counts[decision] += 1
            candidate_out = by_id[candidate_id]
            candidate_out["vlm_filter"] = cache_payload
            if decision == "yes":
                accepted_yes.append(candidate_out)
            elif decision == "uncertain":
                uncertain_candidates.append(candidate_out)
            else:
                rejected_no.append(candidate_out)
            flat_results.append(
                {
                    "candidate_event_id": candidate_id,
                    "event_type": candidate_out.get("event_type"),
                    "best_timestamp_text": candidate_out.get("best_timestamp_text"),
                    "candidate_score": candidate_out.get("candidate_score"),
                    "filter_decision": cache_payload.get("decision"),
                    "event_likelihood": cache_payload.get("event_likelihood"),
                    "short_reason": cache_payload.get("short_reason"),
                    "should_keep": cache_payload.get("should_keep"),
                    "image_path_used": cache_payload.get("image_path_used"),
                    "inference_time_seconds": cache_payload.get("inference_time_seconds"),
                    "representative_frame_path": candidate_out.get("representative_frame", {}).get("image_path"),
                }
            )
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

        if backend == "disabled":
            selected_final = checked_candidates[: int(filter_config["max_filtered_events"])]
            for candidate in selected_final:
                candidate_id = str(candidate.get("candidate_event_id", "") or "")
                candidate_out = by_id[candidate_id]
                candidate_out["vlm_filter"] = {
                    "checked": False,
                    "decision": "skipped_backend_disabled",
                    "event_likelihood": None,
                    "visible_event_type": "skipped",
                    "short_reason": "VLM skipped because TD_CASE2_VLM_BACKEND=disabled. Selected deterministically.",
                    "should_keep": True,
                    "model": MODEL_NAME,
                    "image_path_used": candidate.get("_step11_5_image_path"),
                    "inference_time_seconds": 0.0,
                    "raw_output_text": "",
                    "parsed_json_ok": False,
                }
                flat_results.append(
                    {
                        "candidate_event_id": candidate_id,
                        "event_type": candidate_out.get("event_type"),
                        "best_timestamp_text": candidate_out.get("best_timestamp_text"),
                        "candidate_score": candidate_out.get("candidate_score"),
                        "filter_decision": "skipped_backend_disabled",
                        "event_likelihood": None,
                        "short_reason": candidate_out["vlm_filter"]["short_reason"],
                        "should_keep": True,
                        "image_path_used": candidate_out["vlm_filter"]["image_path_used"],
                        "inference_time_seconds": 0.0,
                        "representative_frame_path": candidate_out.get("representative_frame", {}).get("image_path"),
                    }
                )
            accepted_yes = list(selected_final)
        else:
            selected_final = accepted_yes[: int(filter_config["max_filtered_events"])]
        if bool(filter_config["allow_uncertain_backfill"]) and len(selected_final) < int(filter_config["min_filtered_events"]):
            for candidate in uncertain_candidates:
                if len(selected_final) >= int(filter_config["max_filtered_events"]):
                    break
                selected_final.append(candidate)
                if len(selected_final) >= int(filter_config["min_filtered_events"]):
                    break

        fallback_normal_context_count = 0
        if bool(filter_config["allow_normal_context_backfill"]) and len(selected_final) < int(filter_config["min_filtered_events"]):
            already_selected = {str(item.get("candidate_event_id", "") or "") for item in selected_final}
            fallback_pool = sorted(
                source_candidates,
                key=lambda item: (
                    -_event_preselection_score(item),
                    -float(item.get("candidate_score", 0.0) or 0.0),
                    float(item.get("best_timestamp_seconds", 0.0) or 0.0),
                ),
            )
            for original in fallback_pool:
                if len(selected_final) >= int(filter_config["min_filtered_events"]):
                    break
                candidate_id = str(original.get("candidate_event_id", "") or "")
                if candidate_id in already_selected:
                    continue
                candidate = by_id[candidate_id]
                candidate["vlm_filter"] = {
                    "checked": candidate_id in checked_ids,
                    "decision": "fallback_normal_context",
                    "event_likelihood": candidate.get("vlm_filter", {}).get("event_likelihood"),
                    "visible_event_type": candidate.get("vlm_filter", {}).get("visible_event_type", "normal_context"),
                    "short_reason": "Included as fallback normal context to preserve downstream minimum candidate count.",
                    "should_keep": True,
                    "model": MODEL_NAME,
                    "image_path_used": candidate.get("vlm_filter", {}).get("image_path_used"),
                    "inference_time_seconds": candidate.get("vlm_filter", {}).get("inference_time_seconds", 0.0),
                    "raw_output_text": candidate.get("vlm_filter", {}).get("raw_output_text", ""),
                    "parsed_json_ok": candidate.get("vlm_filter", {}).get("parsed_json_ok", False),
                }
                candidate["final_event_truth"] = "normal_context_or_uncertain_candidate"
                selected_final.append(candidate)
                already_selected.add(candidate_id)
                fallback_normal_context_count += 1
                decision_counts["fallback_normal_context"] += 1
                flat_results.append(
                    {
                        "candidate_event_id": candidate_id,
                        "event_type": candidate.get("event_type"),
                        "best_timestamp_text": candidate.get("best_timestamp_text"),
                        "candidate_score": candidate.get("candidate_score"),
                        "filter_decision": "fallback_normal_context",
                        "event_likelihood": candidate["vlm_filter"].get("event_likelihood"),
                        "short_reason": candidate["vlm_filter"].get("short_reason"),
                        "should_keep": True,
                        "image_path_used": candidate["vlm_filter"].get("image_path_used"),
                        "inference_time_seconds": candidate["vlm_filter"].get("inference_time_seconds"),
                        "representative_frame_path": candidate.get("representative_frame", {}).get("image_path"),
                    }
                )

        selected_final = selected_final[: int(filter_config["max_filtered_events"])]
        event_type_counts_before = Counter(str(item.get("event_type", "") or "unknown") for item in source_candidates)
        event_type_counts_after = Counter(str(item.get("event_type", "") or "unknown") for item in selected_final)
        average_inference_time_seconds = round(total_inference_time_seconds / len(checked_candidates), 3) if checked_candidates else 0.0
        average_api_latency_seconds = round(total_api_latency_seconds / max(1, api_success_count + api_failed_count), 3) if (api_success_count + api_failed_count) else 0.0

        output_payload = {
            "status": "success" if backend != "disabled" else "skipped",
            "source_file": SOURCE_FILE,
            "filter_model": MODEL_NAME,
            "vlm_backend": backend,
            "config": {
                **dict(filter_config),
                "model_path": str(filter_config["model_path"]),
                "prompt_style": "single_full_scene_frame",
                "model_dtype": model_dtype,
            },
            "summary": {
                "input_candidate_count": len(source_candidates),
                "candidates_checked_by_vlm": len(checked_candidates),
                "accepted_yes_count": len(accepted_yes),
                "uncertain_count": len(uncertain_candidates),
                "rejected_no_count": len(rejected_no),
                "fallback_normal_context_count": fallback_normal_context_count,
                "final_filtered_candidate_count": len(selected_final),
                "ready_for_step12_event_ranking": len(selected_final) > 0,
            },
            "candidate_events": selected_final,
        }
        report_payload = {
            "status": "success" if backend != "disabled" else "skipped",
            "source_file": SOURCE_FILE,
            "vlm_backend": backend,
            "api_provider": api_provider if backend == "api_qwen" else None,
            "api_model": api_model if backend == "api_qwen" else None,
            "input_candidate_count": len(source_candidates),
            "candidates_checked_by_vlm": len(checked_candidates),
            "accepted_yes_count": len(accepted_yes),
            "uncertain_count": len(uncertain_candidates),
            "rejected_no_count": len(rejected_no),
            "fallback_normal_context_count": fallback_normal_context_count,
            "final_filtered_candidate_count": len(selected_final),
            "selected_for_step12_count": len(selected_final),
            "min_filtered_events": int(filter_config["min_filtered_events"]),
            "max_filtered_events": int(filter_config["max_filtered_events"]),
            "max_candidates_to_check": int(filter_config["max_candidates_to_check"]),
            "model_path": str(filter_config["model_path"]),
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
            "total_api_latency_seconds": round(total_api_latency_seconds, 3),
            "average_api_latency_seconds": average_api_latency_seconds,
            "total_inference_time_seconds": round(total_inference_time_seconds, 3),
            "average_inference_time_seconds": average_inference_time_seconds,
            "cuda_memory_allocated_before_load_mb": cuda_before_load_mb,
            "cuda_memory_reserved_before_load_mb": cuda_reserved_before_load_mb,
            "cuda_memory_allocated_after_load_mb": cuda_after_load_mb,
            "cuda_memory_reserved_after_load_mb": cuda_reserved_after_load_mb,
            "fallback_used": fallback_used,
            "errors_summary": errors_summary,
            "cuda_memory_allocated_after_mb": cuda_after_mb if cuda_after_mb is not None else cuda_after_load_mb,
            "decision_counts": dict(decision_counts),
            "event_type_counts_before": dict(event_type_counts_before),
            "event_type_counts_after": dict(event_type_counts_after),
            "top_filtered_candidates": [
                {
                    "candidate_event_id": item.get("candidate_event_id"),
                    "event_type": item.get("event_type"),
                    "best_timestamp_text": item.get("best_timestamp_text"),
                    "candidate_score": item.get("candidate_score"),
                    "filter_decision": item.get("vlm_filter", {}).get("decision"),
                    "event_likelihood": item.get("vlm_filter", {}).get("event_likelihood"),
                    "short_reason": item.get("vlm_filter", {}).get("short_reason"),
                }
                for item in selected_final[:10]
            ],
            "rejected_examples": [
                {
                    "candidate_event_id": item.get("candidate_event_id"),
                    "event_type": item.get("event_type"),
                    "best_timestamp_text": item.get("best_timestamp_text"),
                    "candidate_score": item.get("candidate_score"),
                    "filter_decision": item.get("vlm_filter", {}).get("decision"),
                    "short_reason": item.get("vlm_filter", {}).get("short_reason"),
                }
                for item in rejected_no[:8]
            ],
            "warnings": warnings,
            "ready_for_step12_event_ranking": len(selected_final) > 0,
        }

        write_json(run_dir / INPUT_FILE_PRIMARY, output_payload)
        write_json(run_dir / "11_5_vlm_filter_report.json", report_payload)
        write_json_any(run_dir / "11_5_vlm_filter_results_flat.json", flat_results)
        return output_payload, report_payload, flat_results
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
