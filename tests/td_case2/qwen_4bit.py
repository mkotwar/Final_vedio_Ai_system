from __future__ import annotations

import gc
import importlib
import json
import os
from pathlib import Path
from typing import Any


ENV_QWEN_LOAD_IN_4BIT = "TD_CASE2_QWEN_LOAD_IN_4BIT"
ENV_QWEN_4BIT_QUANT_TYPE = "TD_CASE2_QWEN_4BIT_QUANT_TYPE"
ENV_QWEN_4BIT_DOUBLE_QUANT = "TD_CASE2_QWEN_4BIT_DOUBLE_QUANT"
ENV_QWEN_4BIT_COMPUTE_DTYPE = "TD_CASE2_QWEN_4BIT_COMPUTE_DTYPE"
DEFAULT_QWEN_LOAD_IN_4BIT = True
DEFAULT_QWEN_4BIT_QUANT_TYPE = "nf4"
DEFAULT_QWEN_4BIT_DOUBLE_QUANT = True
DEFAULT_QWEN_4BIT_COMPUTE_DTYPE = "auto"
SUPPORTED_COMPUTE_DTYPES = {"auto", "bfloat16", "bf16", "float16", "fp16"}


def _find_spec(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _read_bool_env(env_name: str, default_value: bool) -> bool:
    raw_value = os.environ.get(env_name)
    if raw_value is None or raw_value.strip() == "":
        return default_value
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Environment variable {env_name} must be boolean-like. Received: {raw_value!r}")


def _read_compute_dtype_name() -> str:
    value = os.environ.get(ENV_QWEN_4BIT_COMPUTE_DTYPE, DEFAULT_QWEN_4BIT_COMPUTE_DTYPE).strip().lower()
    if value not in SUPPORTED_COMPUTE_DTYPES:
        raise ValueError(
            f"Environment variable {ENV_QWEN_4BIT_COMPUTE_DTYPE} must be one of "
            f"{sorted(SUPPORTED_COMPUTE_DTYPES)}. Received: {value!r}"
        )
    return value


def _resolve_compute_dtype(torch_module: Any) -> tuple[Any, str]:
    requested = _read_compute_dtype_name()
    if requested == "auto":
        bf16_supported = bool(
            torch_module.cuda.is_available()
            and getattr(torch_module.cuda, "is_bf16_supported", lambda: False)()
        )
        if bf16_supported:
            return torch_module.bfloat16, "bfloat16"
        return torch_module.float16, "float16"
    if requested in {"bfloat16", "bf16"}:
        return torch_module.bfloat16, "bfloat16"
    return torch_module.float16, "float16"


def _read_model_config(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local Qwen checkpoint directory not found: {model_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Local Qwen checkpoint config not found: {config_path}")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to read Qwen checkpoint configuration: {config_path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Qwen checkpoint config is not a JSON object: {config_path}")
    return payload


def _verify_processor_offline(model_path: Path) -> dict[str, Any]:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    return {
        "processor_class": processor.__class__.__name__,
        "processor_offline_ready": True,
    }


def _preflight_runtime_quantization(model_path: Path, torch_module: Any) -> dict[str, Any]:
    if not _read_bool_env(ENV_QWEN_LOAD_IN_4BIT, DEFAULT_QWEN_LOAD_IN_4BIT):
        raise RuntimeError(
            f"Environment variable {ENV_QWEN_LOAD_IN_4BIT}=0 disables 4-bit loading, "
            "but full-precision fallback is disabled to protect GPU memory."
        )
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA is required for local Qwen 4-bit loading but torch.cuda.is_available() is false.")
    if not _find_spec("bitsandbytes"):
        raise RuntimeError("bitsandbytes is required for local Qwen 4-bit loading but is not installed.")
    if not _find_spec("accelerate"):
        raise RuntimeError("accelerate is required for local Qwen 4-bit loading but is not installed.")
    if not _find_spec("qwen_vl_utils"):
        raise RuntimeError("qwen_vl_utils is required for local Qwen 4-bit loading but is not installed.")
    processor_info = _verify_processor_offline(model_path)
    return processor_info


def _is_prequantized_nf4(quantization: dict[str, Any]) -> bool:
    load_in_4bit = quantization.get("load_in_4bit", quantization.get("_load_in_4bit"))
    return bool(
        load_in_4bit is True
        and str(quantization.get("quant_method", "")).lower() == "bitsandbytes"
        and str(quantization.get("bnb_4bit_quant_type", "")).lower() == "nf4"
        and quantization.get("bnb_4bit_use_double_quant") is True
    )


def validate_prequantized_nf4_checkpoint(model_path: str | Path) -> dict[str, Any]:
    """Validate that a local Qwen checkpoint is already pre-quantized BitsAndBytes NF4."""

    checkpoint_path = Path(model_path).expanduser()
    config = _read_model_config(checkpoint_path)
    quantization = dict(config.get("quantization_config") or {})
    if not _is_prequantized_nf4(quantization):
        raise RuntimeError(
            "Qwen checkpoint is not strict pre-quantized BitsAndBytes NF4 with double quantization: "
            f"{checkpoint_path}"
        )
    return quantization


def build_qwen_4bit_load_config(
    model_path: str | Path,
    *,
    torch_module: Any,
) -> dict[str, Any]:
    """Build a safe local-Qwen 4-bit runtime loading plan for normal or prequantized checkpoints."""

    resolved_model_path = Path(model_path).expanduser()
    config = _read_model_config(resolved_model_path)
    processor_info = _preflight_runtime_quantization(resolved_model_path, torch_module)
    quantization = dict(config.get("quantization_config") or {})
    quant_type = os.environ.get(ENV_QWEN_4BIT_QUANT_TYPE, DEFAULT_QWEN_4BIT_QUANT_TYPE).strip().lower() or DEFAULT_QWEN_4BIT_QUANT_TYPE
    if quant_type != "nf4":
        raise RuntimeError(
            f"Only NF4 runtime quantization is supported for local Qwen loading. Received: {quant_type!r}"
        )
    double_quant = _read_bool_env(ENV_QWEN_4BIT_DOUBLE_QUANT, DEFAULT_QWEN_4BIT_DOUBLE_QUANT)
    compute_dtype, compute_dtype_name = _resolve_compute_dtype(torch_module)

    if _is_prequantized_nf4(quantization):
        precision_label = f"4bit_nf4_prequantized_{compute_dtype_name}"
        return {
            "checkpoint_type": "prequantized_nf4",
            "quantization_config": None,
            "compute_dtype": compute_dtype,
            "compute_dtype_name": compute_dtype_name,
            "precision_label": precision_label,
            "is_prequantized": True,
            "load_in_4bit_requested": True,
            "quantization_type": "nf4",
            "double_quant": True,
            "processor_info": processor_info,
            "config_path": str(resolved_model_path / "config.json"),
            "model_path": str(resolved_model_path),
        }

    if quantization:
        raise RuntimeError(
            "Qwen checkpoint has an unsupported quantization_config. "
            "Only normal full-precision checkpoints or prequantized NF4 checkpoints are supported: "
            f"{resolved_model_path}"
        )

    from transformers import BitsAndBytesConfig

    runtime_quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=double_quant,
    )
    precision_label = f"4bit_nf4_runtime_{compute_dtype_name}"
    return {
        "checkpoint_type": "normal_runtime_quantized",
        "quantization_config": runtime_quantization,
        "compute_dtype": compute_dtype,
        "compute_dtype_name": compute_dtype_name,
        "precision_label": precision_label,
        "is_prequantized": False,
        "load_in_4bit_requested": True,
        "quantization_type": "nf4",
        "double_quant": double_quant,
        "processor_info": processor_info,
        "config_path": str(resolved_model_path / "config.json"),
        "model_path": str(resolved_model_path),
    }


def verify_model_loaded_in_4bit(model: Any) -> dict[str, Any]:
    """Verify that a loaded local-Qwen model is actually running in BitsAndBytes 4-bit mode."""

    is_loaded_in_4bit = bool(getattr(model, "is_loaded_in_4bit", False))
    has_4bit_linear = False
    linear_4bit_count = 0
    for module in model.modules():
        class_name = module.__class__.__name__.lower()
        module_name = module.__class__.__module__.lower()
        if "4bit" in class_name and "bitsandbytes" in module_name:
            has_4bit_linear = True
            linear_4bit_count += 1
    if not is_loaded_in_4bit:
        raise RuntimeError(
            "Qwen model was requested in 4-bit mode but is_loaded_in_4bit is false. "
            "Full-precision fallback is disabled to protect GPU memory."
        )
    if not has_4bit_linear:
        raise RuntimeError(
            "Qwen model was requested in 4-bit mode but no BitsAndBytes 4-bit linear layers were found. "
            "Full-precision fallback is disabled to protect GPU memory."
        )
    return {
        "is_loaded_in_4bit": is_loaded_in_4bit,
        "has_4bit_linear_layers": has_4bit_linear,
        "linear_4bit_module_count": linear_4bit_count,
    }


def capture_cuda_memory_snapshot(torch_module: Any) -> dict[str, float | None]:
    """Capture allocated and reserved GPU memory in MB."""

    if not torch_module.cuda.is_available():
        return {
            "allocated_mb": None,
            "reserved_mb": None,
        }
    return {
        "allocated_mb": round(float(torch_module.cuda.memory_allocated()) / (1024**2), 2),
        "reserved_mb": round(float(torch_module.cuda.memory_reserved()) / (1024**2), 2),
    }


def release_qwen_resources(*objects: Any) -> None:
    """Run post-release garbage collection and CUDA cache cleanup."""

    for obj in objects:
        del obj
    gc.collect()
    torch_module = importlib.import_module("torch")
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
