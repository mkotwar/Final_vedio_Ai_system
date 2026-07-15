from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def validate_prequantized_nf4_checkpoint(model_path: str | Path) -> dict[str, Any]:
    """Validate that a local Qwen checkpoint is pre-quantized BitsAndBytes NF4."""

    checkpoint_path = Path(model_path).expanduser()
    config_path = checkpoint_path / "config.json"
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Local 4-bit Qwen checkpoint directory not found: {checkpoint_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Local 4-bit Qwen checkpoint config not found: {config_path}")

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to read Qwen checkpoint configuration: {config_path}") from exc

    quantization = dict(config.get("quantization_config", {}))
    load_in_4bit = quantization.get("load_in_4bit", quantization.get("_load_in_4bit"))
    expected = {
        "load_in_4bit": load_in_4bit is True,
        "quant_method": str(quantization.get("quant_method", "")).lower() == "bitsandbytes",
        "quant_type": str(quantization.get("bnb_4bit_quant_type", "")).lower() == "nf4",
        "double_quant": quantization.get("bnb_4bit_use_double_quant") is True,
    }
    failed = [name for name, valid in expected.items() if not valid]
    if failed:
        raise RuntimeError(
            f"Qwen checkpoint is not strict pre-quantized BitsAndBytes NF4 with double quantization "
            f"({', '.join(failed)}): {checkpoint_path}"
        )
    return quantization
