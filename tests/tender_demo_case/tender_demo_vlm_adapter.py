from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover - depends on installed transformers version
    BitsAndBytesConfig = None


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_BATCH_SIZE = 1
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_MIN_PIXELS = 256 * 28 * 28
DEFAULT_MAX_PIXELS = 512 * 28 * 28

MODEL_ID_ALIASES = {
    "qwen2.5vl:7b": DEFAULT_MODEL_ID,
    "qwen2.5-vl:7b": DEFAULT_MODEL_ID,
    "qwen2-vl-7b": DEFAULT_MODEL_ID,
}


def _read_env_str(name: str, default: str) -> str:
    value = os.environ.get(name, default).strip()
    return value or default


def _read_env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer. Received: {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"Environment variable {name} must be greater than 0. Received: {value}")
    return value


def _read_env_bool(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Environment variable {name} must be a boolean-like value. Received: {raw_value!r}")


def _resolve_model_id(model_id: str) -> str:
    normalized = (model_id or "").strip()
    if not normalized:
        return DEFAULT_MODEL_ID
    return MODEL_ID_ALIASES.get(normalized.lower(), normalized)


def _resolve_local_snapshot(model_id: str) -> Path | None:
    if model_id != DEFAULT_MODEL_ID:
        return None

    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen2.5-VL-7B-Instruct"
    ref_path = cache_root / "refs" / "main"
    if not ref_path.exists():
        return None

    commit_hash = ref_path.read_text(encoding="utf-8").strip()
    if not commit_hash:
        return None

    snapshot_path = cache_root / "snapshots" / commit_hash
    if snapshot_path.exists():
        return snapshot_path
    return None


def get_strict_json_smoke_prompt() -> str:
    return """Analyze this CCTV/security temporal strip image.

The image may contain 3 panels:
PREVIOUS | CURRENT | NEXT

Analyze the CURRENT panel as the main moment.
Use PREVIOUS and NEXT only as temporal context.

Return ONLY one valid JSON object.
No markdown.
No explanation.
No comments.
Do not wrap output in ```json.

Required JSON schema:

{
"scene_type": "street|entrance|parking_area|corridor|office|shop|warehouse|indoor|outdoor|unknown",
"caption": "one objective sentence",
"people_count": 0,
"objects": [
{
"id": "person_1",
"type": "person|vehicle|animal|object",
"subtype": "man|woman|child|car|truck|bus|motorcycle|bicycle|bag|backpack|box|phone|weapon|other|unknown",
"color": "brown|red|orange|yellow|green|blue|purple|pink|white|grey|black|unknown",
"condition": "standing|walking|running|sitting|lying|bending|moving|stationary|parked|unknown"
}
],
"activities": [],
"relationships": [],
"events": [
{
"event_type": "normal_activity|person_object_interaction|loitering|abandoned_object|object_removed|possible_theft|possible_robbery|weapon_visible|physical_altercation|collision|fall|medical_emergency|fire|smoke|crowd_formation",
"description": "objective visual evidence only",
"actors": [],
"severity": "low|medium|high|critical"
}
],
"keywords": []
}

Rules:

* Report only visible facts.
* Do not invent crimes, intentions, identities, or hidden actions.
* If no suspicious event is visible, use "events": [].
* Use "possible_theft" or "possible_robbery" only when clear visual evidence exists.
* Use "physical_altercation" only for visible fighting, grabbing, pushing, striking, or wrestling.
* Use "fall" only if a person is visibly falling or lying after a fall.
* Keep the JSON concise.
* All required top-level keys must always be present."""


def _clean_json_output(raw_output: str) -> str:
    cleaned_output = raw_output.strip()
    if cleaned_output.startswith("```"):
        first_newline = cleaned_output.find("\n")
        if first_newline != -1:
            cleaned_output = cleaned_output[first_newline + 1 :].strip()
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[:-3].strip()
    return cleaned_output


class TenderDemoQwenVLM:
    def __init__(self) -> None:
        requested_model_id = _read_env_str("TENDER_DEMO_QWEN_MODEL_ID", DEFAULT_MODEL_ID)
        self.model_id = _resolve_model_id(requested_model_id)

        requested_device = os.environ.get("TENDER_DEMO_QWEN_DEVICE")
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = (requested_device.strip() if requested_device else default_device).lower()
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            print("[tender-demo-vlm] CUDA requested but not available; falling back to cpu")
            self.device = "cpu"

        self.batch_size = _read_env_int("TENDER_DEMO_QWEN_BATCH_SIZE", DEFAULT_BATCH_SIZE)
        self.max_new_tokens = _read_env_int(
            "TENDER_DEMO_QWEN_MAX_NEW_TOKENS",
            DEFAULT_MAX_NEW_TOKENS,
        )
        self.local_files_only = _read_env_bool("TENDER_DEMO_QWEN_LOCAL_FILES_ONLY", False)

        self.model: Any | None = None
        self.processor: Any | None = None
        self.resolved_model_source: str | None = None
        self.quantization_enabled = False

    def load_model(self) -> None:
        if self.model is not None and self.processor is not None:
            print("[tender-demo-vlm] Reusing already loaded model")
            return

        local_snapshot = _resolve_local_snapshot(self.model_id)
        model_source = str(local_snapshot) if local_snapshot is not None else self.model_id
        self.resolved_model_source = model_source

        print(f"[tender-demo-vlm] Resolved model id: {self.model_id}")
        if local_snapshot is not None:
            print(f"[tender-demo-vlm] Resolved local snapshot path: {local_snapshot}")
        print(f"[tender-demo-vlm] Model source: {model_source}")
        print(f"[tender-demo-vlm] Device: {self.device}")
        print(f"[tender-demo-vlm] Local files only: {self.local_files_only}")
        print(f"[tender-demo-vlm] Batch size: {self.batch_size}")
        print(f"[tender-demo-vlm] Max new tokens: {self.max_new_tokens}")

        self.processor = AutoProcessor.from_pretrained(
            model_source,
            min_pixels=DEFAULT_MIN_PIXELS,
            max_pixels=DEFAULT_MAX_PIXELS,
            local_files_only=self.local_files_only,
        )

        model_kwargs: dict[str, Any] = {
            "local_files_only": self.local_files_only,
        }

        if self.device.startswith("cuda"):
            quantization_loaded = False
            if BitsAndBytesConfig is not None:
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model_source,
                        quantization_config=quantization_config,
                        device_map={"": "cuda:0"},
                        **model_kwargs,
                    )
                    quantization_loaded = True
                    self.quantization_enabled = True
                    print("[tender-demo-vlm] 4-bit quantization enabled")
                except Exception as exc:
                    print(f"[tender-demo-vlm] 4-bit quantization load failed; falling back to normal CUDA load: {exc}")

            if not quantization_loaded:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_source,
                    torch_dtype=torch.bfloat16,
                    device_map={"": "cuda:0"},
                    **model_kwargs,
                )
                self.quantization_enabled = False
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_source,
                **model_kwargs,
            )
            self.model.to(self.device)
            self.quantization_enabled = False

        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

        print(f"[tender-demo-vlm] Quantization enabled: {self.quantization_enabled}")

    def generate_batch(
        self,
        image_paths: list[Path],
        prompts: list[str],
        max_new_tokens: int | None = None,
    ) -> list[str]:
        if len(image_paths) != len(prompts):
            raise ValueError("image_paths and prompts must have the same length.")
        if not image_paths:
            return []

        self.load_model()
        assert self.model is not None
        assert self.processor is not None

        resolved_paths = [Path(path).expanduser().resolve() for path in image_paths]
        for image_path in resolved_paths:
            if not image_path.exists():
                raise FileNotFoundError(f"Image path does not exist: {image_path}")
            if not image_path.is_file():
                raise FileNotFoundError(f"Image path is not a file: {image_path}")

        effective_max_new_tokens = max_new_tokens or self.max_new_tokens
        outputs: list[str] = []

        for batch_start in range(0, len(resolved_paths), self.batch_size):
            batch_image_paths = resolved_paths[batch_start : batch_start + self.batch_size]
            batch_prompts = prompts[batch_start : batch_start + self.batch_size]

            messages_batch = []
            for image_path, prompt in zip(batch_image_paths, batch_prompts):
                messages_batch.append(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": str(image_path)},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
                )

            start_time = time.perf_counter()
            texts = [
                self.processor.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for message in messages_batch
            ]
            image_inputs, video_inputs = process_vision_info(messages_batch)
            model_inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            model_inputs = model_inputs.to(self.device)

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=effective_max_new_tokens,
                )

            generated_ids_trimmed = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            batch_outputs = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            elapsed_seconds = time.perf_counter() - start_time

            for output_text in batch_outputs:
                print(
                    f"[tender-demo-vlm] Generation time: {elapsed_seconds:.2f}s | "
                    f"Output length: {len(output_text)} chars"
                )

            outputs.extend(batch_outputs)

        return outputs

    def generate_one(self, image_path: Path, prompt: str) -> str:
        outputs = self.generate_batch([image_path], [prompt])
        return outputs[0]

    def health_check(self) -> dict[str, Any]:
        return {
            "model_loaded": self.model is not None and self.processor is not None,
            "model_id": self.model_id,
            "resolved_model_source": self.resolved_model_source,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_new_tokens": self.max_new_tokens,
            "local_files_only": self.local_files_only,
            "cuda_available": torch.cuda.is_available(),
            "quantization_enabled": self.quantization_enabled,
        }


if __name__ == "__main__":
    image_path_raw = os.environ.get("TENDER_DEMO_TEST_IMAGE", "").strip()
    if not image_path_raw:
        print(
            "[tender-demo-vlm] Missing TENDER_DEMO_TEST_IMAGE. "
            "Set it to a temporal strip image path before running the smoke test."
        )
        raise SystemExit(1)

    image_path = Path(image_path_raw).expanduser().resolve()
    if not image_path.exists():
        print(f"[tender-demo-vlm] Test image does not exist: {image_path}")
        raise SystemExit(1)

    adapter = TenderDemoQwenVLM()
    adapter.load_model()
    print(f"[tender-demo-vlm] Health check: {adapter.health_check()}")
    prompt = get_strict_json_smoke_prompt()
    output = adapter.generate_one(image_path=image_path, prompt=prompt)
    print("[tender-demo-vlm] Raw output:")
    print(output)

    cleaned_output = _clean_json_output(output)
    try:
        json.loads(cleaned_output)
        print("[SUCCESS] Strict JSON parse succeeded")
    except json.JSONDecodeError as exc:
        print("[ERROR] Strict JSON parse failed")
        print(f"[ERROR] Parse error: {exc}")
        print(f"[ERROR] Raw output was: {output}")
