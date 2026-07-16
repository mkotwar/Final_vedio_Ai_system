from __future__ import annotations

import argparse
import time
from pathlib import Path

from qwen_4bit import (
    build_qwen_4bit_load_config,
    capture_cuda_memory_snapshot,
    release_qwen_resources,
    verify_model_loaded_in_4bit,
)


def _print_line(label: str, value: object) -> None:
    print(f"{label}: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate local Qwen runtime 4-bit loading without running generation.")
    parser.add_argument("--model-path", required=True, help="Path to a local Qwen checkpoint directory.")
    parser.add_argument("--processor-only", action="store_true", help="Only validate offline processor resolution.")
    parser.add_argument("--load-model", action="store_true", help="Load the model into runtime 4-bit NF4 and verify it.")
    args = parser.parse_args()

    if not args.processor_only and not args.load_model:
        parser.error("Specify at least one of --processor-only or --load-model.")

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model_path = Path(args.model_path).expanduser()
    try:
        load_config = build_qwen_4bit_load_config(model_path, torch_module=torch)
        _print_line("model path", model_path)
        _print_line("checkpoint type", load_config["checkpoint_type"])
        _print_line("CUDA device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
        _print_line("compute dtype", load_config["compute_dtype_name"])
        _print_line("quantization method", load_config["quantization_type"])
        _print_line("double quant", load_config["double_quant"])
        _print_line("processor class", load_config["processor_info"]["processor_class"])

        if args.processor_only and not args.load_model:
            _print_line("PASS", "processor offline validation succeeded")
            return 0

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        memory_before = capture_cuda_memory_snapshot(torch)
        load_started = time.perf_counter()
        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True)
        model_kwargs = {
            "local_files_only": True,
            "trust_remote_code": True,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
        if load_config["quantization_config"] is not None:
            model_kwargs["quantization_config"] = load_config["quantization_config"]
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_path), **model_kwargs)
        load_seconds = time.perf_counter() - load_started
        verification = verify_model_loaded_in_4bit(model)
        memory_after = capture_cuda_memory_snapshot(torch)

        _print_line("GPU memory before", memory_before)
        _print_line("GPU memory after", memory_after)
        _print_line("model load time", round(load_seconds, 3))
        _print_line("is_loaded_in_4bit", verification["is_loaded_in_4bit"])
        _print_line("4bit linear modules", verification["linear_4bit_module_count"])
        _print_line("PASS", "runtime 4-bit model load succeeded")
        loaded_model = model
        loaded_processor = processor
        model = None
        processor = None
        del loaded_model, loaded_processor
        release_qwen_resources()
        return 0
    except Exception as exc:
        _print_line("FAIL", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
