from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .anpr_schemas import (
    FlorenceColourResult,
    FlorenceOcrResult,
    normalize_raw_plate_text,
    normalize_vehicle_colour,
)
from .config import FlorenceConfig
from .crop_selection import SelectedCropJob
from .plate_detection import resolve_image_path


@dataclass
class FlorenceModelBundle:
    model: Any
    processor: Any
    device: str
    adapter_applied: bool = False


def _resolve_device(configured: str) -> str:
    if configured != "auto":
        return configured
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _torch_dtype(dtype_name: str) -> Any:
    import torch  # type: ignore

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    normalized = dtype_name.strip().lower()
    if normalized not in mapping:
        raise ValueError(f"Unsupported Florence dtype: {dtype_name}")
    return mapping[normalized]


def _extract_text(post_processed: Any, task_prompt: str) -> str:
    if post_processed is None:
        return ""
    if isinstance(post_processed, str):
        return post_processed
    if isinstance(post_processed, dict):
        if task_prompt in post_processed:
            return str(post_processed[task_prompt] or "")
        for value in post_processed.values():
            if isinstance(value, str) and value.strip():
                return value
        return str(post_processed)
    if isinstance(post_processed, list):
        return " ".join(str(item) for item in post_processed if str(item).strip())
    return str(post_processed)


class FlorenceInferenceEngine:
    """Shared Florence-2 inference wrapper for raw plate OCR and vehicle colour."""

    def __init__(
        self,
        config: FlorenceConfig,
        *,
        bundle: FlorenceModelBundle | None = None,
        run_dir: str | Path | None = None,
    ) -> None:
        self.config = config
        self.bundle = bundle
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.metrics = {
            "florence_load_attempts": 0,
            "florence_load_errors": 0,
            "ocr_attempts": 0,
            "ocr_successes": 0,
            "ocr_empty_outputs": 0,
            "colour_attempts": 0,
            "colour_successes": 0,
            "colour_empty_outputs": 0,
            "inference_errors": 0,
        }

    def load(self) -> FlorenceModelBundle:
        if self.bundle is not None:
            return self.bundle
        if not self.config.enabled:
            raise RuntimeError("Florence inference is disabled.")
        if not self.config.base_model_path:
            raise FileNotFoundError("florence.base_model_path is required when Florence inference is enabled.")
        self.metrics["florence_load_attempts"] += 1
        model_path = resolve_image_path(self.config.base_model_path, run_dir=self.run_dir)
        if self.config.local_files_only and not model_path.exists():
            self.metrics["florence_load_errors"] += 1
            raise FileNotFoundError(f"Florence base model not found locally: {self.config.base_model_path}")
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency-specific
            self.metrics["florence_load_errors"] += 1
            raise RuntimeError("transformers is required for real Step 7 Florence inference.") from exc
        kwargs: dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
            "local_files_only": self.config.local_files_only,
            "attn_implementation": "eager",
        }
        if self.config.load_in_4bit or self.config.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig  # type: ignore
            except Exception as exc:  # pragma: no cover - dependency-specific
                self.metrics["florence_load_errors"] += 1
                raise RuntimeError("bitsandbytes quantization requested but unavailable.") from exc
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=self.config.load_in_4bit,
                load_in_8bit=self.config.load_in_8bit,
            )
        else:
            kwargs["torch_dtype"] = _torch_dtype(self.config.dtype)
        device = _resolve_device(self.config.device)
        try:
            model = AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs)
            if not (self.config.load_in_4bit or self.config.load_in_8bit):
                model = model.to(device)
            processor = AutoProcessor.from_pretrained(
                str(model_path),
                trust_remote_code=self.config.trust_remote_code,
                local_files_only=self.config.local_files_only,
            )
            adapter_applied = False
            if self.config.adapter_path:
                adapter_path = resolve_image_path(self.config.adapter_path, run_dir=self.run_dir)
                if self.config.local_files_only and not adapter_path.exists():
                    raise FileNotFoundError(f"Florence adapter not found locally: {self.config.adapter_path}")
                from peft import PeftModel  # type: ignore

                model = PeftModel.from_pretrained(model, str(adapter_path), local_files_only=self.config.local_files_only)
                adapter_applied = True
            model.eval()
            self.bundle = FlorenceModelBundle(model=model, processor=processor, device=device, adapter_applied=adapter_applied)
            return self.bundle
        except Exception:
            self.metrics["florence_load_errors"] += 1
            raise

    def run_ocr(self, candidate: Any) -> FlorenceOcrResult:
        self.metrics["ocr_attempts"] += 1
        prompt = self.config.ocr_task_prompt
        if not self.config.enabled:
            return self._ocr_result(candidate, "", "model_disabled", prompt)
        image_path = getattr(candidate, "plate_crop_path", None)
        if not image_path:
            return self._ocr_result(candidate, "", "input_missing", prompt)
        resolved = resolve_image_path(str(image_path), run_dir=self.run_dir)
        if not resolved.exists():
            return self._ocr_result(candidate, "", "input_missing", prompt)
        try:
            raw = self._generate(resolved, prompt, use_adapter=True)
        except FileNotFoundError:
            return self._ocr_result(candidate, "", "load_error", prompt)
        except Exception as exc:
            self.metrics["inference_errors"] += 1
            return self._ocr_result(candidate, "", "inference_error", prompt, {"error": str(exc)})
        status = "success" if normalize_raw_plate_text(raw) else "empty_output"
        if status == "success":
            self.metrics["ocr_successes"] += 1
        else:
            self.metrics["ocr_empty_outputs"] += 1
        return self._ocr_result(candidate, raw, status, prompt)

    def run_colour(self, job: SelectedCropJob) -> FlorenceColourResult:
        self.metrics["colour_attempts"] += 1
        prompt = self.config.colour_task_prompt
        if not self.config.enabled:
            return self._colour_result(job, "", "model_disabled", prompt)
        resolved = resolve_image_path(job.vehicle_crop_path, run_dir=self.run_dir)
        if not resolved.exists():
            return self._colour_result(job, "", "input_missing", prompt)
        try:
            raw = self._generate(resolved, prompt, use_adapter=False)
        except FileNotFoundError:
            return self._colour_result(job, "", "load_error", prompt)
        except Exception as exc:
            self.metrics["inference_errors"] += 1
            return self._colour_result(job, "", "inference_error", prompt, {"error": str(exc)})
        colour = normalize_vehicle_colour(raw)
        status = "success" if colour != "unknown" else "empty_output"
        if status == "success":
            self.metrics["colour_successes"] += 1
        else:
            self.metrics["colour_empty_outputs"] += 1
        return self._colour_result(job, raw, status, prompt)

    def _generate(self, image_path: Path, prompt: str, *, use_adapter: bool) -> str:
        bundle = self.load()
        image = Image.open(image_path).convert("RGB")
        inputs = bundle.processor(text=prompt, images=image, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(bundle.device)
        inputs = self._cast_floating_inputs(inputs, bundle.device)
        generate_kwargs = dict(inputs)
        if bundle.adapter_applied and not use_adapter and hasattr(bundle.model, "disable_adapter"):
            with bundle.model.disable_adapter():
                generated_ids = bundle.model.generate(
                    **generate_kwargs,
                    max_new_tokens=self.config.max_new_tokens,
                    num_beams=self.config.num_beams,
                    do_sample=self.config.do_sample,
                    use_cache=False,
                )
        else:
            generated_ids = bundle.model.generate(
                **generate_kwargs,
                max_new_tokens=self.config.max_new_tokens,
                num_beams=self.config.num_beams,
                do_sample=self.config.do_sample,
                use_cache=False,
            )
        decoded = bundle.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        post_processed = decoded
        if hasattr(bundle.processor, "post_process_generation"):
            post_processed = bundle.processor.post_process_generation(
                decoded,
                task=prompt,
                image_size=(image.width, image.height),
            )
        return _extract_text(post_processed, prompt).strip()

    def _cast_floating_inputs(self, inputs: Any, device: str) -> Any:
        if not str(device).startswith("cuda"):
            return inputs
        try:
            import torch  # type: ignore
        except Exception:
            return inputs
        dtype = _torch_dtype(self.config.dtype)
        for key, value in list(inputs.items()):
            if hasattr(value, "dtype") and torch.is_floating_point(value):
                inputs[key] = value.to(device=device, dtype=dtype)
        return inputs

    def _ocr_result(self, candidate: Any, raw_text: str, status: str, prompt: str, metadata: dict[str, Any] | None = None) -> FlorenceOcrResult:
        return FlorenceOcrResult(
            source_id=candidate.source_id,
            track_id=candidate.track_id,
            track_generation=candidate.track_generation,
            crop_role=candidate.crop_role,
            crop_rank=candidate.crop_rank,
            frame_index=candidate.frame_index,
            plate_rank=candidate.plate_rank,
            plate_crop_path=candidate.plate_crop_path,
            raw_text=raw_text,
            normalized_text=normalize_raw_plate_text(raw_text),
            status=status,
            prompt=prompt,
            metadata=metadata or {},
        )

    def _colour_result(self, job: SelectedCropJob, raw_text: str, status: str, prompt: str, metadata: dict[str, Any] | None = None) -> FlorenceColourResult:
        return FlorenceColourResult(
            source_id=job.source_id,
            track_id=job.track_id,
            track_generation=job.track_generation,
            crop_role=job.crop_role,
            crop_rank=job.crop_rank,
            frame_index=job.frame_index,
            vehicle_crop_path=job.vehicle_crop_path,
            raw_text=raw_text,
            normalized_colour=normalize_vehicle_colour(raw_text),
            status=status,
            prompt=prompt,
            metadata=metadata or {},
        )
