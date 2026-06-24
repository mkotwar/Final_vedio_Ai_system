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
from app.services.ocr import OCRService
from app.services.qwen_vlm_hf import NativeQwenTransformersService
from app.services.vlm_prompt import VLM_FRAME_METADATA_PROMPT
from qwen_vl_utils import process_vision_info


CASE_ROOT = PROJECT_ROOT / "tests" / "manual_benchmark_case"
OUTPUT_ROOT = CASE_ROOT / "data" / "output"
SUMMARY_JSON_PATH = OUTPUT_ROOT / "hf_stage_profile_summary.json"
SUMMARY_MD_PATH = OUTPUT_ROOT / "hf_stage_profile_summary.md"

PROFILE_VIDEO_ID = "11111111-2222-4333-8444-666666666666"
FRAME_PATHS = [
    PROJECT_ROOT / "data" / "frames" / PROFILE_VIDEO_ID / "frame_0020.jpg",
    PROJECT_ROOT / "data" / "frames" / PROFILE_VIDEO_ID / "frame_0021.jpg",
]


def _ensure_output_dir() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def _validate_inputs() -> List[Path]:
    missing = [path for path in FRAME_PATHS if not path.exists()]
    if missing:
        missing_str = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Profile frame(s) not found. Run the event-driven benchmark first or update FRAME_PATHS. "
            f"Missing: {missing_str}"
        )
    return FRAME_PATHS


def _build_messages(image_paths: List[Path]) -> List[List[Dict[str, Any]]]:
    messages_batch = []
    for path in image_paths:
        messages_batch.append(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"file://{path.absolute()}",
                        },
                        {"type": "text", "text": VLM_FRAME_METADATA_PROMPT},
                    ],
                }
            ]
        )
    return messages_batch


def _output_token_counts(processor: Any, output_texts: List[str]) -> List[int]:
    tokenizer = processor.tokenizer
    counts: List[int] = []
    for text in output_texts:
        if not text:
            counts.append(0)
            continue
        encoded = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
        counts.append(len(encoded["input_ids"]))
    return counts


def _profile_batch(image_paths: List[Path], warm_label: str) -> Dict[str, Any]:
    service = NativeQwenTransformersService
    processor = service._processor
    model = service._model
    effective_max_new_tokens = service._effective_max_new_tokens()
    messages_batch = _build_messages(image_paths)
    batch_size = len(image_paths)

    stage: Dict[str, float] = {}
    batch_start = time.perf_counter()

    t0 = time.perf_counter()
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages_batch
    ]
    stage["template_build_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    image_inputs, video_inputs = process_vision_info(messages_batch)
    stage["vision_processing_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(service._device)
    stage["tensor_preparation_ms"] = (time.perf_counter() - t0) * 1000.0

    mem_alloc_before = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    mem_res_before = torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0.0

    t0 = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        generated_ids = model.generate(**inputs, max_new_tokens=effective_max_new_tokens)
    stage["generate_ms"] = (time.perf_counter() - t0) * 1000.0

    mem_alloc_after = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    mem_res_after = torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0.0

    t0 = time.perf_counter()
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    stage["decode_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    ocr_results = [OCRService.extract_text(path) for path in image_paths]
    stage["ocr_total_ms"] = (time.perf_counter() - t0) * 1000.0

    batch_runtime_ms = (time.perf_counter() - batch_start) * 1000.0
    output_chars = [len(text) for text in output_texts]
    output_tokens = _output_token_counts(processor, output_texts)
    total_stage_ms = (
        stage["template_build_ms"]
        + stage["vision_processing_ms"]
        + stage["tensor_preparation_ms"]
        + stage["generate_ms"]
        + stage["decode_ms"]
    )
    total_tokens = sum(output_tokens)
    tokens_per_second = (total_tokens / (stage["generate_ms"] / 1000.0)) if stage["generate_ms"] > 0 else 0.0

    return {
        "label": warm_label,
        "batch_size": batch_size,
        "frame_paths": [str(path) for path in image_paths],
        "input_ids_shape": list(inputs.input_ids.shape),
        "pixel_values_shape": list(inputs.pixel_values.shape) if "pixel_values" in inputs else [],
        "stage_ms": stage,
        "batch_runtime_ms": batch_runtime_ms,
        "avg_runtime_per_frame_ms": batch_runtime_ms / max(1, batch_size),
        "stage_breakdown_percent": {
            "template_build": (stage["template_build_ms"] / total_stage_ms) * 100 if total_stage_ms else 0.0,
            "vision_processing": (stage["vision_processing_ms"] / total_stage_ms) * 100 if total_stage_ms else 0.0,
            "tensor_preparation": (stage["tensor_preparation_ms"] / total_stage_ms) * 100 if total_stage_ms else 0.0,
            "generate": (stage["generate_ms"] / total_stage_ms) * 100 if total_stage_ms else 0.0,
            "decode": (stage["decode_ms"] / total_stage_ms) * 100 if total_stage_ms else 0.0,
        },
        "output_chars": output_chars,
        "output_tokens": output_tokens,
        "total_output_tokens": total_tokens,
        "generate_tokens_per_second": tokens_per_second,
        "ocr_detected_text_counts": [len(item.get("detected_text", [])) for item in ocr_results],
        "gpu_memory_allocated_before_gb": mem_alloc_before,
        "gpu_memory_allocated_after_gb": mem_alloc_after,
        "gpu_memory_reserved_before_gb": mem_res_before,
        "gpu_memory_reserved_after_gb": mem_res_after,
        "effective_max_new_tokens": effective_max_new_tokens,
    }


def _build_conclusion(load_ms: float, one_frame: Dict[str, Any], two_frame: Dict[str, Any]) -> Dict[str, Any]:
    one_gen = one_frame["stage_ms"]["generate_ms"]
    one_vision = one_frame["stage_ms"]["vision_processing_ms"] + one_frame["stage_ms"]["tensor_preparation_ms"]
    one_ocr = one_frame["stage_ms"]["ocr_total_ms"]
    one_total = one_frame["avg_runtime_per_frame_ms"]

    dominant_stage = max(
        (
            ("generate", one_gen),
            ("vision_plus_tensor", one_vision),
            ("ocr", one_ocr),
        ),
        key=lambda item: item[1],
    )[0]

    answer_lines = [
        "This profile measures warm one-frame and two-frame HF runs separately from model load.",
        f"Model load is a one-time cost of {load_ms / 1000.0:.2f}s and should not be confused with steady-state frame latency.",
        f"Warm 1-frame average runtime is {one_total / 1000.0:.2f}s/frame.",
        f"Inside that 1-frame run: generate={one_gen / 1000.0:.2f}s, vision+tensor={(one_vision) / 1000.0:.2f}s, ocr={one_ocr / 1000.0:.2f}s.",
        f"The dominant steady-state stage is {dominant_stage}.",
    ]

    if dominant_stage == "generate":
        answer_lines.append(
            "So the ~5s/frame behavior is mainly model generation time, not OCR and not image preprocessing."
        )
    elif dominant_stage == "vision_plus_tensor":
        answer_lines.append(
            "So the ~5s/frame behavior is mainly vision preprocessing / tensor preparation, not text generation."
        )
    else:
        answer_lines.append(
            "So the ~5s/frame behavior is mainly OCR overhead, not the VLM itself."
        )

    if two_frame["avg_runtime_per_frame_ms"] < one_total:
        answer_lines.append(
            "Batching improves per-frame cost, which means there is some shared overhead amortization across frames."
        )
    else:
        answer_lines.append(
            "Batching does not materially reduce per-frame cost here, so the dominant work scales almost linearly with frames."
        )

    return {
        "dominant_stage": dominant_stage,
        "answer_lines": answer_lines,
    }


def main() -> None:
    _ensure_output_dir()
    image_paths = _validate_inputs()

    old_debug = settings.DEBUG
    settings.DEBUG = False

    try:
        t0 = time.perf_counter()
        NativeQwenTransformersService.load_model()
        model_load_ms = (time.perf_counter() - t0) * 1000.0

        one_frame = _profile_batch([image_paths[0]], "warm_batch_1")
        two_frame = _profile_batch(image_paths[:2], "warm_batch_2")
        conclusion = _build_conclusion(model_load_ms, one_frame, two_frame)
    finally:
        settings.DEBUG = old_debug

    summary = {
        "model_id": settings.QWEN_MODEL_ID,
        "engine": "native_hf",
        "device": NativeQwenTransformersService._device,
        "model_load_ms": model_load_ms,
        "profiles": [one_frame, two_frame],
        "conclusion": conclusion,
    }

    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    md_lines = [
        "# HF Stage Profile Summary",
        "",
        f"- Model: `{settings.QWEN_MODEL_ID}`",
        f"- Device: `{NativeQwenTransformersService._device}`",
        f"- One-time model load: `{model_load_ms / 1000.0:.2f}s`",
        f"- Warm 1-frame runtime: `{one_frame['avg_runtime_per_frame_ms'] / 1000.0:.2f}s/frame`",
        f"- Warm 2-frame runtime: `{two_frame['avg_runtime_per_frame_ms'] / 1000.0:.2f}s/frame`",
        f"- Dominant steady-state stage: `{conclusion['dominant_stage']}`",
        "",
        "## Answer",
        "",
    ]
    md_lines.extend(f"- {line}" for line in conclusion["answer_lines"])

    with open(SUMMARY_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print("HF_STAGE_PROFILE_SUMMARY_START")
    print(json.dumps(summary))
    print("HF_STAGE_PROFILE_SUMMARY_END")


if __name__ == "__main__":
    main()
