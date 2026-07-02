from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


ENV_FINAL_DEMO_VLM_VERIFY_ENABLED = "FINAL_DEMO_VLM_VERIFY_ENABLED"
ENV_FINAL_DEMO_QWEN_MODEL_ID = "FINAL_DEMO_QWEN_MODEL_ID"
ENV_FINAL_DEMO_QWEN_DEVICE = "FINAL_DEMO_QWEN_DEVICE"
ENV_FINAL_DEMO_QWEN_DTYPE = "FINAL_DEMO_QWEN_DTYPE"
ENV_FINAL_DEMO_QWEN_BATCH_SIZE = "FINAL_DEMO_QWEN_BATCH_SIZE"
ENV_FINAL_DEMO_QWEN_MAX_NEW_TOKENS = "FINAL_DEMO_QWEN_MAX_NEW_TOKENS"
ENV_FINAL_DEMO_QWEN_TEMPERATURE = "FINAL_DEMO_QWEN_TEMPERATURE"

DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_QWEN_DEVICE = "cuda"
DEFAULT_QWEN_DTYPE = "auto"
DEFAULT_QWEN_BATCH_SIZE = 1
DEFAULT_QWEN_MAX_NEW_TOKENS = 256
DEFAULT_QWEN_TEMPERATURE = 0.0


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


def read_float_env(env_name: str, default_value: float) -> float:
    raw_value = os.environ.get(env_name, str(default_value))
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {env_name} must be numeric. Received: {raw_value!r}") from exc


def read_int_env(env_name: str, default_value: int) -> int:
    raw_value = os.environ.get(env_name, str(default_value))
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {env_name} must be integer. Received: {raw_value!r}") from exc


def extract_first_json_object(text: str) -> tuple[dict[str, Any], str | None]:
    raw_text = str(text or "").strip()
    if not raw_text:
        return {}, "empty_output"
    start_index = raw_text.find("{")
    if start_index < 0:
        return {}, "no_json_object_found"
    depth = 0
    end_index = -1
    for index in range(start_index, len(raw_text)):
        character = raw_text[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                end_index = index + 1
                break
    if end_index < 0:
        return {}, "unterminated_json_object"
    candidate = raw_text[start_index:end_index]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return {}, f"json_decode_error:{exc.msg}"
    if not isinstance(parsed, dict):
        return {}, "json_root_not_object"
    return parsed, None


class FinalDemoVLMAdapter:
    def __init__(self) -> None:
        self.enabled = read_bool_env(ENV_FINAL_DEMO_VLM_VERIFY_ENABLED, False)
        self.model_id = os.environ.get(ENV_FINAL_DEMO_QWEN_MODEL_ID, DEFAULT_QWEN_MODEL_ID).strip() or DEFAULT_QWEN_MODEL_ID
        self.device = os.environ.get(ENV_FINAL_DEMO_QWEN_DEVICE, DEFAULT_QWEN_DEVICE).strip() or DEFAULT_QWEN_DEVICE
        self.dtype = os.environ.get(ENV_FINAL_DEMO_QWEN_DTYPE, DEFAULT_QWEN_DTYPE).strip() or DEFAULT_QWEN_DTYPE
        self.batch_size = max(1, read_int_env(ENV_FINAL_DEMO_QWEN_BATCH_SIZE, DEFAULT_QWEN_BATCH_SIZE))
        self.max_new_tokens = max(1, read_int_env(ENV_FINAL_DEMO_QWEN_MAX_NEW_TOKENS, DEFAULT_QWEN_MAX_NEW_TOKENS))
        self.temperature = max(0.0, read_float_env(ENV_FINAL_DEMO_QWEN_TEMPERATURE, DEFAULT_QWEN_TEMPERATURE))
        self._processor: Any = None
        self._model: Any = None
        self._load_error = ""

    def ensure_loaded(self) -> tuple[bool, str]:
        if not self.enabled:
            return False, "VLM disabled by environment."
        if self._model is not None and self._processor is not None:
            return True, ""
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor  # type: ignore
        except Exception as exc:
            self._load_error = f"transformers_import_failed:{exc}"
            return False, self._load_error
        try:
            processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            model = AutoModelForVision2Seq.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype=None if self.dtype == "auto" else self.dtype,
                device_map=self.device if self.device not in {"cpu", ""} else None,
            )
            if self.device == "cpu":
                model = model.to("cpu")
            self._processor = processor
            self._model = model
            return True, ""
        except Exception as exc:
            self._load_error = f"model_load_failed:{exc}"
            return False, self._load_error

    def verify_image(self, image_path: Path, prompt: str) -> dict[str, Any]:
        if not self.enabled:
            return {
                "ok": False,
                "model_id": self.model_id,
                "raw_output": "",
                "parsed_json": {},
                "error": "vlm_disabled",
            }
        loaded, load_error = self.ensure_loaded()
        if not loaded:
            return {
                "ok": False,
                "model_id": self.model_id,
                "raw_output": "",
                "parsed_json": {},
                "error": load_error or "model_not_loaded",
            }
        try:
            from PIL import Image  # type: ignore
        except Exception as exc:
            return {
                "ok": False,
                "model_id": self.model_id,
                "raw_output": "",
                "parsed_json": {},
                "error": f"pil_import_failed:{exc}",
            }
        if not image_path.exists():
            return {
                "ok": False,
                "model_id": self.model_id,
                "raw_output": "",
                "parsed_json": {},
                "error": f"missing_input_image:{image_path}",
            }
        try:
            image = Image.open(image_path).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            prompt_text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(
                text=[prompt_text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
            if self.device == "cpu":
                inputs = {key: value.to("cpu") for key, value in inputs.items()}
            elif hasattr(self._model, "device") and str(getattr(self._model, "device", "")):
                inputs = {key: value.to(self._model.device) for key, value in inputs.items()}
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0.0,
            )
            output_text = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )[0]
            parsed_json, parse_error = extract_first_json_object(output_text)
            return {
                "ok": parse_error is None,
                "model_id": self.model_id,
                "raw_output": output_text,
                "parsed_json": parsed_json,
                "error": parse_error or "",
            }
        except Exception as exc:
            return {
                "ok": False,
                "model_id": self.model_id,
                "raw_output": "",
                "parsed_json": {},
                "error": f"inference_failed:{exc}",
            }
