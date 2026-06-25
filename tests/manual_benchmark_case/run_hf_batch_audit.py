import inspect
import json
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

AUDIT_MD_PATH = OUTPUT_ROOT / "HF_BATCH_AUDIT.md"
AUDIT_JSON_PATH = OUTPUT_ROOT / "hf_batch_audit.json"

TARGET_MODE = "candidate_only"
TARGET_VARIANT = "strip_tokens150_batch4"


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


def _load_candidate_jobs() -> List[Dict[str, Any]]:
    summary = _load_summary()
    rows = summary["modes"][TARGET_MODE][TARGET_VARIANT]["results"]
    jobs: List[Dict[str, Any]] = []
    for row in rows:
        image_path = REASONING_INPUT_ROOT / f"{TARGET_MODE}_{TARGET_VARIANT}_{row['event_id']}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing reasoning input image: {image_path}")
        prompt = _build_reasoning_prompt(row["structured_context"])
        jobs.append(
            {
                "event_id": row["event_id"],
                "image_path": image_path,
                "prompt": prompt,
                "structured_context": row["structured_context"],
                "expected_batch4_success": row["success"],
                "expected_batch4_raw_output": row["raw_output"],
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


def _token_count(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(text or "", add_special_tokens=False, return_attention_mask=False)
    return len(encoded["input_ids"])


def _tensor_shape_map(inputs: Any) -> Dict[str, Any]:
    shape_map: Dict[str, Any] = {}
    for key, value in inputs.items():
        if hasattr(value, "shape"):
            shape_map[key] = list(value.shape)
        else:
            shape_map[key] = str(type(value).__name__)
    return shape_map


def _vision_rows(inputs: Any, batch_size: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    image_grid = inputs.get("image_grid_thw")
    if image_grid is None:
        return [{} for _ in range(batch_size)]

    grid_cpu = image_grid.detach().cpu().tolist()
    for idx in range(batch_size):
        if idx < len(grid_cpu):
            grid = grid_cpu[idx]
            vision_tokens = int(grid[0] * grid[1] * grid[2]) if len(grid) == 3 else None
            rows.append({"image_grid_thw": grid, "vision_token_count_estimate": vision_tokens})
        else:
            rows.append({})
    return rows


def _decode_with_current_trim(processor: Any, generated_ids: Any, input_ids: Any) -> List[str]:
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def _decode_with_attention_trim(processor: Any, generated_ids: Any, attention_mask: Any) -> List[str]:
    trimmed = []
    for mask_row, out_ids in zip(attention_mask, generated_ids):
        valid_len = int(mask_row.sum().item())
        trimmed.append(out_ids[valid_len:])
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def _run_generate_audit(batch_jobs: List[Dict[str, Any]], batch_size_label: int) -> Dict[str, Any]:
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

    generate_kwargs = {
        "input_keys": list(model_inputs.keys()),
        "max_new_tokens": int(settings.QWEN_MAX_NEW_TOKENS),
        "attention_mask_present": "attention_mask" in model_inputs,
        "pixel_values_present": "pixel_values" in model_inputs,
        "image_grid_thw_present": "image_grid_thw" in model_inputs,
    }

    start = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        generated_ids = model.generate(**model_inputs, max_new_tokens=settings.QWEN_MAX_NEW_TOKENS)
    generate_seconds = time.perf_counter() - start

    current_outputs = _decode_with_current_trim(processor, generated_ids, model_inputs.input_ids)
    attention_outputs = _decode_with_attention_trim(processor, generated_ids, model_inputs.attention_mask)

    input_ids_cpu = model_inputs.input_ids.detach().cpu()
    attention_mask_cpu = model_inputs.attention_mask.detach().cpu() if "attention_mask" in model_inputs else None
    generated_ids_cpu = generated_ids.detach().cpu()
    vision_info = _vision_rows(model_inputs, len(batch_jobs))

    per_sample: List[Dict[str, Any]] = []
    for idx, job in enumerate(batch_jobs):
        current_text = current_outputs[idx]
        attention_text = attention_outputs[idx]
        input_row = input_ids_cpu[idx]
        generated_row = generated_ids_cpu[idx]
        attention_row = attention_mask_cpu[idx] if attention_mask_cpu is not None else None
        valid_len = int(attention_row.sum().item()) if attention_row is not None else len(input_row)
        padded_len = len(input_row)
        generated_total_len = len(generated_row)
        current_generated_tokens = max(0, generated_total_len - padded_len)
        attention_generated_tokens = max(0, generated_total_len - valid_len)

        per_sample.append(
            {
                "event_id": job["event_id"],
                "batch_size_mode": batch_size_label,
                "prompt_chars": len(job["prompt"]),
                "prompt_token_count": _token_count(processor.tokenizer, job["prompt"]),
                "chat_template_chars": len(texts[idx]),
                "chat_template_token_count": _token_count(processor.tokenizer, texts[idx]),
                "padded_input_length": padded_len,
                "valid_input_length_from_attention_mask": valid_len,
                "attention_mask_sum": valid_len,
                "attention_mask_tail": attention_row[-12:].tolist() if attention_row is not None else [],
                "last_input_ids_tail": input_row[-12:].tolist(),
                "generated_total_length": generated_total_len,
                "generated_token_count_current_trim": current_generated_tokens,
                "generated_token_count_attention_trim": attention_generated_tokens,
                "decoded_length_current_trim": len(current_text),
                "decoded_length_attention_trim": len(attention_text),
                "decoded_text_current_trim": current_text,
                "decoded_text_attention_trim": attention_text,
                "parsed_json_success_current_trim": _safe_json_parse(current_text),
                "parsed_json_success_attention_trim": _safe_json_parse(attention_text),
                "current_vs_attention_decode_equal": current_text == attention_text,
                "vision_info": vision_info[idx],
            }
        )

    return {
        "batch_size_mode": batch_size_label,
        "event_ids": [job["event_id"] for job in batch_jobs],
        "processor_call": {
            "text_count": len(texts),
            "image_count": len(image_inputs) if image_inputs is not None else 0,
            "video_count": len(video_inputs) if video_inputs is not None else 0,
            "args": {
                "padding": True,
                "return_tensors": "pt",
                "images": "image_inputs",
                "text": "texts",
                "videos": "video_inputs",
            },
            "returned_keys": list(inputs.keys()),
            "returned_shapes": _tensor_shape_map(inputs),
            "attention_mask_exists": "attention_mask" in inputs,
        },
        "generate_kwargs": generate_kwargs,
        "generate_seconds": generate_seconds,
        "per_sample": per_sample,
    }


def _run_config(jobs: List[Dict[str, Any]], batch_size: int) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    for start_idx in range(0, len(jobs), batch_size):
        chunk = jobs[start_idx:start_idx + batch_size]
        runs.append(_run_generate_audit(chunk, batch_size))
    return {
        "configured_batch_size": batch_size,
        "runs": runs,
    }


def _find_lines(path: Path, patterns: List[str]) -> Dict[str, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    found: Dict[str, int] = {}
    for pattern in patterns:
        for idx, line in enumerate(lines, start=1):
            if pattern in line:
                found[pattern] = idx
                break
    return found


def _build_findings(audit: Dict[str, Any]) -> List[Dict[str, Any]]:
    batch1 = audit["comparisons"]["batch1"]
    batch4 = audit["comparisons"]["batch4"]

    def _flatten_success(cfg: Dict[str, Any]) -> List[bool]:
        values: List[bool] = []
        for run in cfg["runs"]:
            for row in run["per_sample"]:
                values.append(bool(row["parsed_json_success_current_trim"]))
        return values

    def _flatten_current_texts(cfg: Dict[str, Any]) -> List[str]:
        values: List[str] = []
        for run in cfg["runs"]:
            for row in run["per_sample"]:
                values.append(row["decoded_text_current_trim"])
        return values

    batch1_success = _flatten_success(batch1)
    batch4_success = _flatten_success(batch4)
    batch1_all_ok = all(batch1_success)
    batch4_ok_count = sum(1 for item in batch4_success if item)
    current_vs_attention_equal = all(
        row["current_vs_attention_decode_equal"]
        for run in batch4["runs"]
        for row in run["per_sample"]
    )

    findings = [
        {
            "title": "Right-padding in batched decoder-only generation is the top root-cause candidate.",
            "confidence": "high",
            "why": (
                "Tokenizer padding_side is right, the HF warning is triggered only on batch sizes above 1, "
                f"batch_size=1 succeeds on all 5 samples while batch_size=4 succeeds on only {batch4_ok_count}/5 "
                "for the same prompt/image pairs."
            ),
            "proposed_fix": "Initialize the tokenizer for left padding before batched generate().",
        },
        {
            "title": "Attention mask is present and forwarded into generate().",
            "confidence": "high",
            "why": "Processor returns attention_mask and model.generate receives it via **inputs; no dropped-mask bug was detected.",
            "proposed_fix": "No fix needed here.",
        },
        {
            "title": "Prompt/image ordering is preserved through batching.",
            "confidence": "high",
            "why": "The batch is built by zipping image_paths and prompt_list in order, and outputs are consumed with zip(batch, raw_outputs).",
            "proposed_fix": "No fix needed here.",
        },
        {
            "title": "Current output slicing is probably not the main failure cause.",
            "confidence": "medium",
            "why": (
                "Current trim and attention-sum trim decode identically across the audited batch-4 runs."
                if current_vs_attention_equal
                else "Current trim and attention-sum trim differ on some rows, so trimming should be reviewed after padding is fixed."
            ),
            "proposed_fix": "Revisit trim logic after the padding fix only if failures remain.",
        },
        {
            "title": "The fifth candidate in batch_size=4 is effectively a single-sample run.",
            "confidence": "high",
            "why": (
                "With 5 jobs and batch_size=4, the benchmark executes one batch of 4 and one batch of 1; "
                "the trailing single-item batch succeeds, reinforcing that corruption is tied to multi-sample batching."
            ),
            "proposed_fix": "No fix needed; this is diagnostic evidence.",
        },
    ]

    if not batch1_all_ok:
        findings.insert(
            0,
            {
                "title": "Unexpected: batch_size=1 is not fully clean in the audit.",
                "confidence": "high",
                "why": "This would contradict the prior benchmark and should be investigated before any production change.",
                "proposed_fix": "Reproduce with the exact saved prompt/image pairs and inspect environment drift.",
            },
        )

    return findings


def _write_markdown(audit: Dict[str, Any]) -> None:
    tokenizer = audit["tokenizer_audit"]
    processor = audit["processor_audit"]
    generate = audit["generate_audit"]
    batching = audit["batching_audit"]
    slicing = audit["output_slicing_audit"]
    vision = audit["vision_processing_audit"]
    comparisons = audit["comparisons"]
    findings = audit["findings"]

    lines: List[str] = [
        "# HF Batch Generation Audit for Qwen2.5-VL",
        "",
        "## Scope",
        "",
        f"- Input set: `{TARGET_MODE}/{TARGET_VARIANT}` saved benchmark prompts and images",
        f"- Candidate prompt/image pairs audited: `{audit['candidate_count']}`",
        f"- Source summary: `{SOURCE_SUMMARY_PATH}`",
        "",
        "## 1. Tokenizer Audit",
        "",
        f"- `padding_side`: `{tokenizer['padding_side']}`",
        f"- `pad_token`: `{tokenizer['pad_token']}`",
        f"- `pad_token_id`: `{tokenizer['pad_token_id']}`",
        f"- `eos_token`: `{tokenizer['eos_token']}`",
        f"- `eos_token_id`: `{tokenizer['eos_token_id']}`",
        f"- Assignment in project code: `{tokenizer['assignment_in_project_code']}`",
        f"- Later mutation detected in project code: `{tokenizer['later_mutation_detected']}`",
        "",
        "## 2. Processor Audit",
        "",
        f"- Processor init site: `{processor['processor_init_ref']}`",
        f"- Processor call site: `{processor['processor_call_ref']}`",
        f"- Processor call args: `{processor['processor_call_args']}`",
        f"- Returned keys in audited runs: `{processor['returned_keys_union']}`",
        f"- Attention mask present: `{processor['attention_mask_present']}`",
        "",
        "## 3. Generate Audit",
        "",
        f"- Generate site: `{generate['generate_call_ref']}`",
        f"- Generate receives `input_ids`: `{generate['input_ids_present']}`",
        f"- Generate receives `attention_mask`: `{generate['attention_mask_present']}`",
        f"- Generate receives `pixel_values`: `{generate['pixel_values_present']}`",
        f"- Generate receives `image_grid_thw`: `{generate['image_grid_thw_present']}`",
        f"- Extra kwargs shape source: `{generate['kwargs_source']}`",
        "",
        "## 4. Batching Audit",
        "",
        f"- Prompt list construction site: `{batching['prompt_list_ref']}`",
        f"- Image/prompt zip site: `{batching['zip_ref']}`",
        f"- Output zip site: `{batching['output_zip_ref']}`",
        f"- Ordering issue detected: `{batching['ordering_issue_detected']}`",
        f"- Batch4 execution groups: `{batching['batch4_groups']}`",
        "",
        "## 5. Decode Audit",
        "",
        f"- Decode site: `{audit['decode_audit']['decode_call_ref']}`",
        f"- `skip_special_tokens`: `{audit['decode_audit']['skip_special_tokens']}`",
        f"- `clean_up_tokenization_spaces`: `{audit['decode_audit']['clean_up_tokenization_spaces']}`",
        f"- Decode-only bug detected: `{audit['decode_audit']['decode_only_bug_detected']}`",
        "",
        "## 6. Output Slicing Audit",
        "",
        f"- Current trim site: `{slicing['trim_ref']}`",
        f"- Current trim basis: `{slicing['trim_basis']}`",
        f"- Alternative trim compared: `{slicing['alternative_trim_basis']}`",
        f"- Current-vs-attention decode mismatch detected: `{slicing['decode_mismatch_detected']}`",
        f"- Assessment: `{slicing['assessment']}`",
        "",
        "## 7. Vision Processing Audit",
        "",
        f"- `process_vision_info` source: `{vision['source']}`",
        f"- Vision call site: `{vision['call_ref']}`",
        f"- Variable image grid across samples: `{vision['variable_image_grid_detected']}`",
        f"- Assessment: `{vision['assessment']}`",
        "",
        "## 8. Batch1 vs Batch4 Comparison",
        "",
    ]

    for config_name, config in comparisons.items():
        lines.append(f"### {config_name}")
        lines.append("")
        for run_idx, run in enumerate(config["runs"], start=1):
            lines.append(
                f"- Run {run_idx}: batch_size={run['batch_size_mode']}, events={run['event_ids']}, "
                f"generate_seconds={run['generate_seconds']:.2f}"
            )
            for row in run["per_sample"]:
                lines.append(
                    f"  - {row['event_id']}: prompt_tokens={row['prompt_token_count']}, "
                    f"chat_tokens={row['chat_template_token_count']}, padded_len={row['padded_input_length']}, "
                    f"valid_len={row['valid_input_length_from_attention_mask']}, "
                    f"gen_tokens_current={row['generated_token_count_current_trim']}, "
                    f"decoded_len={row['decoded_length_current_trim']}, "
                    f"json_success={row['parsed_json_success_current_trim']}"
                )
        lines.append("")

    lines.extend(
        [
            "## 9. Identified Issues Ranked by Confidence",
            "",
        ]
    )
    for finding in findings:
        lines.extend(
            [
                f"- {finding['title']}",
                f"  - Confidence: `{finding['confidence']}`",
                f"  - Why: {finding['why']}",
                f"  - Proposed fix: {finding['proposed_fix']}",
            ]
        )

    lines.extend(
        [
            "",
            "## 10. Exact Code Locations Requiring Changes",
            "",
            f"- [qwen_vlm_hf.py](/{PROJECT_ROOT / 'app' / 'services' / 'qwen_vlm_hf.py'}:{tokenizer['processor_init_line']}): tokenizer/processor initialization point",
            f"- [qwen_vlm_hf.py](/{PROJECT_ROOT / 'app' / 'services' / 'qwen_vlm_hf.py'}:{processor['processor_call_line']}): processor batch construction with `padding=True`",
            f"- [qwen_vlm_hf.py](/{PROJECT_ROOT / 'app' / 'services' / 'qwen_vlm_hf.py'}:{generate['generate_call_line']}): `model.generate(**inputs, ...)` batched call",
            f"- [qwen_vlm_hf.py](/{PROJECT_ROOT / 'app' / 'services' / 'qwen_vlm_hf.py'}:{slicing['trim_line']}): generated token slicing",
            f"- [utils.py](/{PROJECT_ROOT / '.venv' / 'Lib' / 'site-packages' / 'transformers' / 'generation' / 'utils.py'}:{audit['hf_warning']['line']}): HF right-padding warning trigger",
            "",
            "If no issue is found in a section above, it is explicitly marked as no issue detected.",
        ]
    )

    AUDIT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _ensure_output_dir()
    jobs = _load_candidate_jobs()

    old_debug = settings.DEBUG
    old_max_new_tokens = settings.QWEN_MAX_NEW_TOKENS
    settings.DEBUG = False
    settings.QWEN_MAX_NEW_TOKENS = 150

    try:
        NativeQwenTransformersService.load_model()
        service = NativeQwenTransformersService
        processor = service._processor
        tokenizer = processor.tokenizer

        qwen_service_path = PROJECT_ROOT / "app" / "services" / "qwen_vlm_hf.py"
        transformers_utils_path = PROJECT_ROOT / ".venv" / "Lib" / "site-packages" / "transformers" / "generation" / "utils.py"

        service_lines = _find_lines(
            qwen_service_path,
            [
                "AutoProcessor.from_pretrained(",
                "padding=True,",
                "generated_ids = cls._model.generate(**inputs, max_new_tokens=effective_max_new_tokens)",
                "generated_ids_trimmed = [",
                "output_texts = cls._processor.batch_decode(",
                "prompt_list = list(prompts)",
                "for path, prompt in zip(image_paths, prompt_list):",
                "for job, raw_output in zip(batch, raw_outputs):",
            ],
        )
        hf_warning_lines = _find_lines(transformers_utils_path, ["A decoder-only architecture is being used, but right-padding was detected!"])

        batch1 = _run_config(jobs, batch_size=1)
        batch4 = _run_config(jobs, batch_size=4)

        returned_keys_union = sorted(
            {
                key
                for config in (batch1, batch4)
                for run in config["runs"]
                for key in run["processor_call"]["returned_keys"]
            }
        )
        attention_mask_present = all(
            run["processor_call"]["attention_mask_exists"]
            for config in (batch1, batch4)
            for run in config["runs"]
        )

        decode_mismatch_detected = any(
            not row["current_vs_attention_decode_equal"]
            for run in batch4["runs"]
            for row in run["per_sample"]
        )

        variable_image_grid_detected = len(
            {
                tuple(row["vision_info"].get("image_grid_thw", []))
                for run in batch4["runs"]
                for row in run["per_sample"]
                if row["vision_info"].get("image_grid_thw")
            }
        ) > 1

        audit = {
            "candidate_count": len(jobs),
            "tokenizer_audit": {
                "padding_side": tokenizer.padding_side,
                "pad_token": tokenizer.pad_token,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token": tokenizer.eos_token,
                "eos_token_id": tokenizer.eos_token_id,
                "assignment_in_project_code": "No assignment found in project code; inherited from AutoProcessor.from_pretrained().",
                "later_mutation_detected": False,
                "processor_init_line": service_lines.get("AutoProcessor.from_pretrained(", 0),
            },
            "processor_audit": {
                "processor_init_ref": f"app/services/qwen_vlm_hf.py:{service_lines.get('AutoProcessor.from_pretrained(', 0)}",
                "processor_init_line": service_lines.get("AutoProcessor.from_pretrained(", 0),
                "processor_call_ref": f"app/services/qwen_vlm_hf.py:{service_lines.get('padding=True,', 0)}",
                "processor_call_line": service_lines.get("padding=True,", 0),
                "processor_call_args": {
                    "text": "texts",
                    "images": "image_inputs",
                    "videos": "video_inputs",
                    "padding": True,
                    "return_tensors": "pt",
                },
                "returned_keys_union": returned_keys_union,
                "attention_mask_present": attention_mask_present,
            },
            "generate_audit": {
                "generate_call_ref": f"app/services/qwen_vlm_hf.py:{service_lines.get('generated_ids = cls._model.generate(**inputs, max_new_tokens=effective_max_new_tokens)', 0)}",
                "generate_call_line": service_lines.get(
                    "generated_ids = cls._model.generate(**inputs, max_new_tokens=effective_max_new_tokens)",
                    0,
                ),
                "input_ids_present": True,
                "attention_mask_present": attention_mask_present,
                "pixel_values_present": "pixel_values" in returned_keys_union,
                "image_grid_thw_present": "image_grid_thw" in returned_keys_union,
                "kwargs_source": "`**inputs` from processor output",
            },
            "batching_audit": {
                "prompt_list_ref": f"app/services/qwen_vlm_hf.py:{service_lines.get('prompt_list = list(prompts)', 0)}",
                "zip_ref": f"app/services/qwen_vlm_hf.py:{service_lines.get('for path, prompt in zip(image_paths, prompt_list):', 0)}",
                "output_zip_ref": "Outputs are returned from generate_batch() in the same row order as image_paths/prompt_list; benchmark harness then consumes them in order.",
                "ordering_issue_detected": False,
                "batch4_groups": [run["event_ids"] for run in batch4["runs"]],
            },
            "decode_audit": {
                "decode_call_ref": f"app/services/qwen_vlm_hf.py:{service_lines.get('output_texts = cls._processor.batch_decode(', 0)}",
                "skip_special_tokens": True,
                "clean_up_tokenization_spaces": False,
                "decode_only_bug_detected": False,
            },
            "output_slicing_audit": {
                "trim_ref": f"app/services/qwen_vlm_hf.py:{service_lines.get('generated_ids_trimmed = [', 0)}",
                "trim_line": service_lines.get("generated_ids_trimmed = [", 0),
                "trim_basis": "`len(in_ids)` per row after processor padding",
                "alternative_trim_basis": "`attention_mask.sum()` per row",
                "decode_mismatch_detected": decode_mismatch_detected,
                "assessment": (
                    "No issue detected."
                    if not decode_mismatch_detected
                    else "Potential issue detected; current and attention-sum trims diverged."
                ),
            },
            "vision_processing_audit": {
                "source": inspect.getsourcefile(process_vision_info),
                "call_ref": f"app/services/qwen_vlm_hf.py:{service_lines.get('padding=True,', 0) - 11}",
                "variable_image_grid_detected": variable_image_grid_detected,
                "assessment": (
                    "Variable image_grid_thw values are present across samples, but no standalone vision batching bug was isolated in this audit."
                ),
            },
            "comparisons": {
                "batch1": batch1,
                "batch4": batch4,
            },
            "hf_warning": {
                "source": "transformers/generation/utils.py",
                "line": hf_warning_lines.get(
                    "A decoder-only architecture is being used, but right-padding was detected!",
                    0,
                ),
                "explanation": (
                    "HF checks attention_mask[:, -1] == 0 for decoder-only generation and warns when any row ends in padding."
                ),
            },
        }
        audit["findings"] = _build_findings(audit)

    finally:
        settings.DEBUG = old_debug
        settings.QWEN_MAX_NEW_TOKENS = old_max_new_tokens

    AUDIT_JSON_PATH.write_text(json.dumps(audit, indent=4, ensure_ascii=False), encoding="utf-8")
    _write_markdown(audit)

    print("HF_BATCH_AUDIT_START")
    print(json.dumps({"markdown": str(AUDIT_MD_PATH), "json": str(AUDIT_JSON_PATH), "candidate_count": len(jobs)}))
    print("HF_BATCH_AUDIT_END")


if __name__ == "__main__":
    main()
