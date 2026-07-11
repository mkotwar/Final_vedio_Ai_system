from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

MODEL_PATH = Path(r"C:\Mukul K\models\Qwen2.5-VL-3B-Instruct")
BENCHMARK_IMAGE_PATH = Path(
    r"C:\Mukul K\vinfo1\video-search-engine\tests\td_case2\debug_runs\anpr_test_5min_20260707_122758\13_vlm_event_inputs\vlm_event_input_000001_strip.jpg"
)
RECOMMENDED_DOWNLOAD_COMMAND = (
    'huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir "C:\\Mukul K\\models\\Qwen2.5-VL-3B-Instruct"'
)
PROMPT_TEXT = (
    "Look at this CCTV traffic image. Does it show a meaningful visible event? "
    "Answer only JSON with keys: decision, short_reason. "
    "decision must be one of yes, no, uncertain."
)


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _find_spec(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _cuda_memory_mb(torch_module: Any) -> float | None:
    if not torch_module.cuda.is_available():
        return None
    return round(float(torch_module.cuda.memory_allocated()) / (1024**2), 2)


def _print_env_info(torch_module: Any, transformers_module: Any) -> None:
    print(f"PYTHON_EXECUTABLE={sys.executable}")
    print(f"TORCH_VERSION={torch_module.__version__}")
    print(f"TRANSFORMERS_VERSION={transformers_module.__version__}")
    print(f"CUDA_AVAILABLE={_bool_text(torch_module.cuda.is_available())}")
    if torch_module.cuda.is_available():
        props = torch_module.cuda.get_device_properties(0)
        print(f"GPU_NAME={torch_module.cuda.get_device_name(0)}")
        print(f"GPU_VRAM_GB={round(float(props.total_memory) / (1024**3), 2)}")
    else:
        print("GPU_NAME=")
        print("GPU_VRAM_GB=0")
    print(f"HAS_QWEN_VL_UTILS={_bool_text(_find_spec('qwen_vl_utils'))}")
    print(f"HAS_ACCELERATE={_bool_text(_find_spec('accelerate'))}")


def _load_readiness() -> dict[str, Any]:
    from transformers import AutoConfig, AutoProcessor

    found_local_model = MODEL_PATH.exists()
    can_load_config = False
    can_load_processor = False
    error_messages: list[str] = []

    if found_local_model:
        try:
            AutoConfig.from_pretrained(str(MODEL_PATH), local_files_only=True, trust_remote_code=True)
            can_load_config = True
        except Exception as exc:  # pragma: no cover - environment-specific
            error_messages.append(f"config_load_failed={exc}")
        try:
            AutoProcessor.from_pretrained(str(MODEL_PATH), local_files_only=True, trust_remote_code=True)
            can_load_processor = True
        except Exception as exc:  # pragma: no cover - environment-specific
            error_messages.append(f"processor_load_failed={exc}")

    return {
        "found_local_model": found_local_model,
        "can_load_config": can_load_config,
        "can_load_processor": can_load_processor,
        "ready_for_benchmark": found_local_model and can_load_config and can_load_processor,
        "error_messages": error_messages,
    }


def _try_parse_json(raw_text: str) -> tuple[bool, Any]:
    cleaned = raw_text.strip()
    if not cleaned:
        return False, None
    try:
        return True, json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            candidate = cleaned[start : end + 1]
            try:
                return True, json.loads(candidate)
            except json.JSONDecodeError:
                return False, None
    return False, None


def _benchmark_one_image() -> None:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    readiness = _load_readiness()
    if not readiness["ready_for_benchmark"]:
        print("BENCHMARK_RAN=false")
        print("BENCHMARK_REASON=model_not_ready")
        return
    if not BENCHMARK_IMAGE_PATH.exists():
        print("BENCHMARK_RAN=false")
        print(f"BENCHMARK_REASON=missing_image:{BENCHMARK_IMAGE_PATH}")
        return
    if not _find_spec("qwen_vl_utils"):
        print("BENCHMARK_RAN=false")
        print("BENCHMARK_REASON=qwen_vl_utils_not_installed")
        return

    from qwen_vl_utils import process_vision_info

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        cuda_before_mb = _cuda_memory_mb(torch)
    else:
        cuda_before_mb = None

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    total_start = time.perf_counter()
    load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(MODEL_PATH), local_files_only=True, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    model_load_time = time.perf_counter() - load_start

    preprocess_start = time.perf_counter()
    with Image.open(BENCHMARK_IMAGE_PATH) as img:
        img.verify()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(BENCHMARK_IMAGE_PATH)},
                {"type": "text", "text": PROMPT_TEXT},
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
    preprocess_time = time.perf_counter() - preprocess_start

    infer_start = time.perf_counter()
    generated_ids = model.generate(**model_inputs, max_new_tokens=96)
    trimmed_ids = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    decoded = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    inference_time = time.perf_counter() - infer_start
    total_time = time.perf_counter() - total_start

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        cuda_after_mb = _cuda_memory_mb(torch)
    else:
        cuda_after_mb = None

    raw_output_text = decoded[0].strip() if decoded else ""
    parsed_ok, parsed_json = _try_parse_json(raw_output_text)

    print("BENCHMARK_RAN=true")
    print(f"BENCHMARK_IMAGE_PATH={BENCHMARK_IMAGE_PATH}")
    print(f"MODEL_DTYPE={str(dtype).replace('torch.', '')}")
    print(f"MODEL_LOAD_TIME_SECONDS={model_load_time:.3f}")
    print(f"IMAGE_PREPROCESS_TIME_SECONDS={preprocess_time:.3f}")
    print(f"INFERENCE_TIME_SECONDS={inference_time:.3f}")
    print(f"TOTAL_BENCHMARK_TIME_SECONDS={total_time:.3f}")
    print(f"CUDA_MEMORY_ALLOCATED_BEFORE_MB={cuda_before_mb if cuda_before_mb is not None else 'NA'}")
    print(f"CUDA_MEMORY_ALLOCATED_AFTER_MB={cuda_after_mb if cuda_after_mb is not None else 'NA'}")
    print(f"RAW_OUTPUT_TEXT={raw_output_text}")
    print(f"PARSED_JSON_OK={_bool_text(parsed_ok)}")
    print(f"PARSED_JSON={json.dumps(parsed_json, ensure_ascii=True) if parsed_ok else 'null'}")


def main() -> None:
    import torch
    import transformers

    _print_env_info(torch, transformers)

    readiness = _load_readiness()
    print(f"FOUND_LOCAL_MODEL={_bool_text(readiness['found_local_model'])}")
    print(f"MODEL_PATH={MODEL_PATH}")
    print(f"CAN_LOAD_PROCESSOR={_bool_text(readiness['can_load_processor'])}")
    print(f"READY_FOR_BENCHMARK={_bool_text(readiness['ready_for_benchmark'])}")

    if not readiness["found_local_model"]:
        print("MODEL_MISSING=true")
        print(f"RECOMMENDED_DOWNLOAD_COMMAND={RECOMMENDED_DOWNLOAD_COMMAND}")
    else:
        print("MODEL_MISSING=false")

    if readiness["error_messages"]:
        for message in readiness["error_messages"]:
            print(f"LOAD_ERROR={message}")

    if os.environ.get("BENCHMARK_QWEN3B", "").strip() == "1":
        _benchmark_one_image()
    else:
        print("BENCHMARK_RAN=false")
        print("BENCHMARK_REASON=presence_check_mode")


if __name__ == "__main__":
    main()
