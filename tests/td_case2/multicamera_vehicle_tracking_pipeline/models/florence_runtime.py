from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


os.environ.setdefault("HF_MODULES_CACHE", str(Path(tempfile.gettempdir()) / "codex_hf_modules_cache"))


try:
    import torch
    from peft import PeftModel
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    PeftModel = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoProcessor = None  # type: ignore[assignment]


class FlorenceRuntimeError(RuntimeError):
    """Raised when Florence runtime loading or inference fails."""


@dataclass(frozen=True, slots=True)
class FlorenceRuntimeDependencies:
    torch_module: Any
    auto_model_cls: Any
    auto_processor_cls: Any
    peft_model_cls: Any
    image_cls: Any


class FlorenceRuntime:
    def __init__(
        self,
        *,
        model_path: Path,
        adapter_path: Path | None,
        processor_path: Path | None,
        device: str,
        dtype: str,
        local_files_only: bool,
        trust_remote_code: bool,
        dependencies: FlorenceRuntimeDependencies | None = None,
    ) -> None:
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.processor_path = processor_path
        self.device = device
        self.dtype = dtype
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code
        self.dependencies = dependencies or FlorenceRuntimeDependencies(
            torch_module=torch,
            auto_model_cls=AutoModelForCausalLM,
            auto_processor_cls=AutoProcessor,
            peft_model_cls=PeftModel,
            image_cls=Image,
        )
        self.model = None
        self.processor = None
        self.loaded = False

    def load(self) -> None:
        if self.loaded:
            return
        if self.dependencies.auto_model_cls is None or self.dependencies.auto_processor_cls is None:
            raise FlorenceRuntimeError("Florence runtime dependencies are not installed.")
        _ensure_writable_huggingface_modules_cache()
        torch_module = self.dependencies.torch_module
        model_kwargs: dict[str, object] = {
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }
        if self.dtype == "float16" and torch_module is not None:
            model_kwargs["torch_dtype"] = torch_module.float16
        elif self.dtype == "bfloat16" and torch_module is not None:
            model_kwargs["torch_dtype"] = torch_module.bfloat16
        try:
            self.model = self.dependencies.auto_model_cls.from_pretrained(str(self.model_path), attn_implementation="eager", **model_kwargs)
            self.processor = self.dependencies.auto_processor_cls.from_pretrained(
                str(self.processor_path or self.adapter_path or self.model_path),
                trust_remote_code=self.trust_remote_code,
                local_files_only=self.local_files_only,
            )
            if self.adapter_path is not None and self.dependencies.peft_model_cls is not None:
                self.model = self.dependencies.peft_model_cls.from_pretrained(self.model, str(self.adapter_path), local_files_only=self.local_files_only)
            if hasattr(self.model, "to"):
                self.model = self.model.to(self.device)
            if hasattr(self.model, "eval"):
                self.model.eval()
        except Exception as exc:
            raise FlorenceRuntimeError(f"Failed to load Florence runtime: {exc}") from exc
        self.loaded = True

    def run_image_task(
        self,
        *,
        image_path: Path,
        prompt: str,
        disable_adapter: bool = False,
    ) -> str:
        self.load()
        if self.processor is None or self.model is None or self.dependencies.image_cls is None:
            raise FlorenceRuntimeError("Florence runtime is not loaded.")
        if not image_path.exists():
            raise FileNotFoundError(f"Florence input image does not exist: {image_path}")
        try:
            image = self.dependencies.image_cls.open(image_path).convert("RGB")
        except Exception as exc:
            raise FlorenceRuntimeError(f"Failed to load Florence input image: {exc}") from exc
        try:
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            if hasattr(inputs, "to"):
                inputs = inputs.to(self.device)
            context_manager = self._inference_context(disable_adapter=disable_adapter)
            with context_manager:
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=512,
                    do_sample=False,
                    num_beams=3,
                    use_cache=False,
                )
            decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        except Exception as exc:
            raise FlorenceRuntimeError(f"Florence inference failed: {exc}") from exc
        return str(decoded)

    def close(self) -> None:
        self.model = None
        self.processor = None
        self.loaded = False

    def _inference_context(self, *, disable_adapter: bool):
        torch_module = self.dependencies.torch_module
        if disable_adapter and hasattr(self.model, "disable_adapter"):
            return self.model.disable_adapter()
        if torch_module is None:
            return _NullContext()
        return torch_module.inference_mode()


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _ensure_writable_huggingface_modules_cache() -> Path:
    configured = os.getenv("HF_MODULES_CACHE")
    if configured not in (None, ""):
        cache_path = Path(configured).expanduser()
    else:
        cache_path = Path(tempfile.gettempdir()) / "codex_hf_modules_cache"
        os.environ["HF_MODULES_CACHE"] = str(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path
