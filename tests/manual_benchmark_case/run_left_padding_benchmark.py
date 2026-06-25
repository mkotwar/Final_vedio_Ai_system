import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT_PATH = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from app.core.config import PROJECT_ROOT, settings
from app.services.qwen_vlm_hf import NativeQwenTransformersService
from app.services.vlm_utils import clean_json_response
from qwen_vl_utils import process_vision_info


CASE_ROOT = PROJECT_ROOT / "tests" / "manual_benchmark_case"
OUTPUT_ROOT = CASE_ROOT / "data" / "output"
REASONING_INPUT_ROOT = OUTPUT_ROOT / "reasoning_inputs"
SOURCE_SUMMARY_PATH = OUTPUT_ROOT / "event_candidate_benchmark_summary.json"
TARGET_INPUT_VIDEO = os.getenv("BENCHMARK_INPUT_VIDEO", r"C:\Mukul K\test_video\V_ai_test_2min.mp4")

SUMMARY_JSON_PATH = OUTPUT_ROOT / "left_padding_benchmark.json"
SUMMARY_MD_PATH = OUTPUT_ROOT / "LEFT_PADDING_BENCHMARK.md"

TARGET_MODES = ("candidate_only", "candidate_plus_periodic10s")
TARGET_BATCH_SIZES = (1, 4)
MAX_NEW_TOKENS = 150


def _ensure_output_dir() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def _load_summary() -> Dict[str, Any]:
    if not SOURCE_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Missing source benchmark summary: {SOURCE_SUMMARY_PATH}. "
            "Run run_event_candidate_reasoning_benchmark.py first."
        )
    return json.loads(SOURCE_SUMMARY_PATH.read_text(encoding="utf-8"))


def _build_reasoning_prompt(structured_context: Dict[str, Any]) -> str:
    facts_json = json.dumps(structured_context, separators=(",", ":"))
    return (
        "You are an investigation assistant.\n"
        f"Known facts: {facts_json}\n"
        "Analyze:\n"
        "1. likely event\n"
        "2. notable behavior\n"
        "3. interaction\n"
        "4. investigate?\n"
        'Return compact JSON only with keys event_type, notable, interaction, investigate, why. '
        'Use true/false booleans and very short strings.'
    )


def _batch4_variant_name(mode: str) -> str:
    return "strip_tokens150_batch4"


def _variant_name_for_batch(mode: str, batch_size: int) -> str:
    return f"strip_tokens150_batch{batch_size}"


def _image_path_for_job(mode: str, event_id: str) -> Path:
    return REASONING_INPUT_ROOT / f"{mode}_{_batch4_variant_name(mode)}_{event_id}.jpg"


def _load_jobs(mode: str) -> List[Dict[str, Any]]:
    source = _load_summary()
    rows = source["modes"][mode][_batch4_variant_name(mode)]["results"]
    jobs: List[Dict[str, Any]] = []
    for row in rows:
        image_path = _image_path_for_job(mode, row["event_id"])
        if not image_path.exists():
            raise FileNotFoundError(f"Missing reasoning input image: {image_path}")
        jobs.append(
            {
                "event_id": row["event_id"],
                "mode": mode,
                "periodic": row["periodic"],
                "image_path": image_path,
                "structured_context": row["structured_context"],
                "prompt": _build_reasoning_prompt(row["structured_context"]),
            }
        )
    return jobs


def _safe_json_parse(text: str) -> bool:
    try:
        cleaned = clean_json_response(text)
        parsed = json.loads(cleaned)
        return isinstance(parsed, dict)
    except Exception:
        return False


def _classify_failure(raw_output: str, success: bool) -> str:
    if success:
        return "json_success"
    raw = (raw_output or "").strip()
    if not raw:
        return "empty_response"
    if any(ord(ch) > 127 for ch in raw):
        return "non_english_garbage"
    return "non_json_garbage"


def _token_count(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(text or "", add_special_tokens=False, return_attention_mask=False)
    return len(encoded["input_ids"])


def _collect_transformers_warnings():
    from transformers.generation import utils as generation_utils

    original_warning = generation_utils.logger.warning
    captured: List[str] = []

    def wrapper(message, *args, **kwargs):
        text = str(message)
        captured.append(text)
        return original_warning(message, *args, **kwargs)

    generation_utils.logger.warning = wrapper
    return generation_utils.logger, original_warning, captured


def _run_batch(batch_jobs: List[Dict[str, Any]], batch_size: int) -> Dict[str, Any]:
    service = NativeQwenTransformersService
    processor = service._processor
    model = service._model

    image_paths = [job["image_path"] for job in batch_jobs]
    prompts = [job["prompt"] for job in batch_jobs]
    messages_batch = []
    for path, prompt in zip(image_paths, prompts):
        messages_batch.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{path.absolute()}"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        )

    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_batch]
    image_inputs, video_inputs = process_vision_info(messages_batch)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    model_inputs = inputs.to(service._device)

    warning_logger, original_warning, warnings = _collect_transformers_warnings()
    start = time.perf_counter()
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated_ids = model.generate(**model_inputs, max_new_tokens=settings.QWEN_MAX_NEW_TOKENS)
    finally:
        warning_logger.warning = original_warning
    latency_seconds = time.perf_counter() - start

    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]
    outputs = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    input_ids_cpu = model_inputs.input_ids.detach().cpu()
    attention_mask_cpu = model_inputs.attention_mask.detach().cpu()
    generated_ids_cpu = generated_ids.detach().cpu()

    rows: List[Dict[str, Any]] = []
    for idx, job in enumerate(batch_jobs):
        decoded_text = outputs[idx]
        success = _safe_json_parse(decoded_text)
        row = {
            "event_id": job["event_id"],
            "periodic": job["periodic"],
            "prompt_token_count": _token_count(processor.tokenizer, job["prompt"]),
            "padded_sequence_length": int(input_ids_cpu[idx].shape[0]),
            "valid_sequence_length": int(attention_mask_cpu[idx].sum().item()),
            "attention_mask_shape": list(model_inputs.attention_mask.shape),
            "generated_token_count": int(generated_ids_cpu[idx].shape[0] - input_ids_cpu[idx].shape[0]),
            "decoded_output_length": len(decoded_text),
            "decoded_output": decoded_text,
            "json_parse_success": success,
            "failure_type": _classify_failure(decoded_text, success),
        }
        rows.append(row)

    return {
        "event_ids": [job["event_id"] for job in batch_jobs],
        "latency_seconds": latency_seconds,
        "warnings": warnings,
        "rows": rows,
    }


def _run_config(jobs: List[Dict[str, Any]], batch_size: int) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    for start_idx in range(0, len(jobs), batch_size):
        runs.append(_run_batch(jobs[start_idx:start_idx + batch_size], batch_size))

    rows = [row for run in runs for row in run["rows"]]
    success_count = sum(1 for row in rows if row["json_parse_success"])
    warnings = [warning for run in runs for warning in run["warnings"]]
    failure_breakdown: Dict[str, int] = {}
    for row in rows:
        failure_type = row["failure_type"]
        failure_breakdown[failure_type] = failure_breakdown.get(failure_type, 0) + 1

    return {
        "batch_size": batch_size,
        "latency_seconds": sum(run["latency_seconds"] for run in runs),
        "successful_responses": success_count,
        "failed_responses": len(rows) - success_count,
        "warnings": warnings,
        "warning_count": len(warnings),
        "failure_breakdown": failure_breakdown,
        "rows": rows,
    }


def _extract_before(summary: Dict[str, Any], mode: str, batch_size: int) -> Dict[str, Any]:
    variant_name = _variant_name_for_batch(mode, batch_size)
    before = summary["modes"][mode][variant_name]
    rows = []
    for row in before["results"]:
        rows.append(
            {
                "event_id": row["event_id"],
                "periodic": row["periodic"],
                "decoded_output": row["raw_output"],
                "decoded_output_length": len(row["raw_output"] or ""),
                "json_parse_success": row["success"],
                "failure_type": row["failure_category"],
                "generated_token_count": row["output_tokens"],
            }
        )
    return {
        "batch_size": batch_size,
        "latency_seconds": before["wall_clock_runtime_seconds"],
        "successful_responses": before["successful_responses"],
        "failed_responses": before["failed_responses"],
        "failure_breakdown": before["failure_breakdown"],
        "warning_count": None,
        "warnings": ["Not captured in baseline summary; prior HF warning was observed in logs for batch_size > 1."],
        "rows": rows,
    }


def _compare(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "success_before": before["successful_responses"],
        "success_after": after["successful_responses"],
        "failed_before": before["failed_responses"],
        "failed_after": after["failed_responses"],
        "latency_before_seconds": before["latency_seconds"],
        "latency_after_seconds": after["latency_seconds"],
        "warning_count_before": before["warning_count"],
        "warning_count_after": after["warning_count"],
        "failure_breakdown_before": before["failure_breakdown"],
        "failure_breakdown_after": after["failure_breakdown"],
    }


def _write_markdown(result: Dict[str, Any]) -> None:
    lines = [
        "# LEFT_PADDING_BENCHMARK",
        "",
        "## Configuration",
        "",
        f"- Tokenizer padding before: `{result['configuration']['padding_side_before']}`",
        f"- Tokenizer padding after: `{result['configuration']['padding_side_after']}`",
        f"- Pad token before: `{result['configuration']['pad_token_before']}`",
        f"- Pad token after: `{result['configuration']['pad_token_after']}`",
        f"- EOS token: `{result['configuration']['eos_token']}`",
        f"- `max_new_tokens`: `{result['configuration']['max_new_tokens']}`",
        f"- Changed variable only: `processor.tokenizer.padding_side = \"left\"`",
        "",
        "## Conclusion",
        "",
        f"- Did changing only tokenizer left padding eliminate the batch-generation corruption? "
        f"`{result['conclusion']['eliminated_batch_corruption']}`",
        f"- Short answer: {result['conclusion']['summary']}",
        "",
        "## Before vs After",
        "",
    ]

    for mode in TARGET_MODES:
        lines.append(f"### {mode}")
        lines.append("")
        for batch_size in TARGET_BATCH_SIZES:
            comparison = result["comparisons"][mode][f"batch{batch_size}"]
            lines.extend(
                [
                    f"- batch_size={batch_size}",
                    f"  - success: `{comparison['success_before']} -> {comparison['success_after']}`",
                    f"  - failed: `{comparison['failed_before']} -> {comparison['failed_after']}`",
                    f"  - latency: `{comparison['latency_before_seconds']:.2f}s -> {comparison['latency_after_seconds']:.2f}s`",
                    f"  - warnings: `{comparison['warning_count_before']} -> {comparison['warning_count_after']}`",
                    f"  - failures before: `{comparison['failure_breakdown_before']}`",
                    f"  - failures after: `{comparison['failure_breakdown_after']}`",
                ]
            )
        lines.append("")

    lines.extend(
        [
            "## Remaining Failures",
            "",
        ]
    )

    remaining_failures = result["conclusion"]["remaining_failures"]
    if not remaining_failures:
        lines.append("- None.")
    else:
        for item in remaining_failures:
            lines.append(
                f"- {item['mode']} batch_size={item['batch_size']} event `{item['event_id']}` -> `{item['failure_type']}`"
            )

    lines.extend(
        [
            "",
            "## Warning Comparison",
            "",
        ]
    )
    for mode in TARGET_MODES:
        for batch_size in TARGET_BATCH_SIZES:
            after = result["after"][mode][f"batch{batch_size}"]
            lines.append(
                f"- {mode} batch_size={batch_size}: warning_count=`{after['warning_count']}`, "
                f"warnings=`{after['warnings']}`"
            )

    SUMMARY_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _ensure_output_dir()
    baseline_summary = _load_summary()

    old_debug = settings.DEBUG
    old_max_new_tokens = settings.QWEN_MAX_NEW_TOKENS
    settings.DEBUG = False
    settings.QWEN_MAX_NEW_TOKENS = MAX_NEW_TOKENS

    try:
        NativeQwenTransformersService.load_model()
        tokenizer = NativeQwenTransformersService._processor.tokenizer
        padding_side_before = tokenizer.padding_side
        pad_token_before = tokenizer.pad_token

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        after_results: Dict[str, Any] = {}
        for mode in TARGET_MODES:
            jobs = _load_jobs(mode)
            after_results[mode] = {}
            for batch_size in TARGET_BATCH_SIZES:
                after_results[mode][f"batch{batch_size}"] = _run_config(jobs, batch_size)

        padding_side_after = tokenizer.padding_side
        pad_token_after = tokenizer.pad_token

    finally:
        settings.DEBUG = old_debug
        settings.QWEN_MAX_NEW_TOKENS = old_max_new_tokens

    before_results: Dict[str, Any] = {}
    for mode in TARGET_MODES:
        before_results[mode] = {}
        for batch_size in TARGET_BATCH_SIZES:
            before_results[mode][f"batch{batch_size}"] = _extract_before(baseline_summary, mode, batch_size)

    comparisons: Dict[str, Any] = {}
    remaining_failures: List[Dict[str, Any]] = []
    for mode in TARGET_MODES:
        comparisons[mode] = {}
        for batch_size in TARGET_BATCH_SIZES:
            key = f"batch{batch_size}"
            comparisons[mode][key] = _compare(before_results[mode][key], after_results[mode][key])
            for row in after_results[mode][key]["rows"]:
                if not row["json_parse_success"]:
                    remaining_failures.append(
                        {
                            "mode": mode,
                            "batch_size": batch_size,
                            "event_id": row["event_id"],
                            "failure_type": row["failure_type"],
                        }
                    )

    batch4_clean = (
        after_results["candidate_only"]["batch4"]["failed_responses"] == 0
        and after_results["candidate_plus_periodic10s"]["batch4"]["failed_responses"] == 0
    )
    result = {
        "configuration": {
            "input_video": TARGET_INPUT_VIDEO,
            "padding_side_before": padding_side_before,
            "padding_side_after": padding_side_after,
            "pad_token_before": pad_token_before,
            "pad_token_after": pad_token_after,
            "eos_token": NativeQwenTransformersService._processor.tokenizer.eos_token,
            "max_new_tokens": MAX_NEW_TOKENS,
            "changed_variable_only": "processor.tokenizer.padding_side = 'left'",
        },
        "before": before_results,
        "after": after_results,
        "comparisons": comparisons,
        "conclusion": {
            "eliminated_batch_corruption": batch4_clean,
            "summary": (
                "Yes. Left padding alone removed the batch-generation corruption in this benchmark."
                if batch4_clean
                else "No. Left padding improved things but did not fully remove corruption."
            ),
            "remaining_failures": remaining_failures,
        },
    }

    SUMMARY_JSON_PATH.write_text(json.dumps(result, indent=4, ensure_ascii=False), encoding="utf-8")
    _write_markdown(result)

    print("LEFT_PADDING_BENCHMARK_START")
    print(json.dumps({"json": str(SUMMARY_JSON_PATH), "markdown": str(SUMMARY_MD_PATH)}))
    print("LEFT_PADDING_BENCHMARK_END")


if __name__ == "__main__":
    main()
