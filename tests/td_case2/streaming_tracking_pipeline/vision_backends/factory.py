from __future__ import annotations

from pathlib import Path
from typing import Any

from ..crop_selection import SelectedCropJob
from ..florence_inference import FlorenceInferenceEngine
from .base import VisionInferenceBackend
from .florence_backend import FlorenceVisionBackend
from .gemini_backend import GeminiVisionBackend


class DisabledVisionBackend(VisionInferenceBackend):
    def __init__(self, florence_engine: FlorenceInferenceEngine | None = None) -> None:
        self.florence_engine = florence_engine or FlorenceInferenceEngine(config=type("DisabledConfig", (), {"enabled": False})())

    @property
    def backend_name(self) -> str:
        return "disabled"

    @property
    def metrics(self) -> dict[str, Any]:
        return {"disabled_requests": 0}

    def load(self) -> Any:
        return None

    def run_ocr(self, candidate: Any):
        return self.florence_engine._ocr_result(candidate, "", "model_disabled", "<OCR>")  # noqa: SLF001

    def run_colour(self, job: SelectedCropJob):
        return self.florence_engine._colour_result(job, "", "model_disabled", "<VQA>")  # noqa: SLF001


class FallbackVisionBackend(VisionInferenceBackend):
    def __init__(self, primary: VisionInferenceBackend, secondary: VisionInferenceBackend | None = None) -> None:
        self.primary = primary
        self.secondary = secondary
        self._metrics = {
            "vision_fallback_loads": 0,
            "vision_fallback_invocations": 0,
        }

    @property
    def backend_name(self) -> str:
        if self.secondary is None:
            return self.primary.backend_name
        return "auto"

    @property
    def metrics(self) -> dict[str, Any]:
        return {**self.primary.metrics, **(self.secondary.metrics if self.secondary is not None else {}), **self._metrics}

    def load(self) -> Any:
        try:
            return self.primary.load()
        except Exception:
            if self.secondary is None:
                raise
            self._metrics["vision_fallback_loads"] += 1
            return self.secondary.load()

    def run_ocr(self, candidate: Any):
        result = self.primary.run_ocr(candidate)
        if result.status in {"success", "empty_output", "model_disabled", "input_missing"} or self.secondary is None:
            return self._with_backend_metadata(result, selected=self.primary.backend_name, fallback_from=None)
        self._metrics["vision_fallback_invocations"] += 1
        fallback = self.secondary.run_ocr(candidate)
        return self._with_backend_metadata(fallback, selected=self.secondary.backend_name, fallback_from=self.primary.backend_name)

    def run_colour(self, job: SelectedCropJob):
        result = self.primary.run_colour(job)
        if result.status in {"success", "empty_output", "model_disabled", "input_missing"} or self.secondary is None:
            return self._with_backend_metadata(result, selected=self.primary.backend_name, fallback_from=None)
        self._metrics["vision_fallback_invocations"] += 1
        fallback = self.secondary.run_colour(job)
        return self._with_backend_metadata(fallback, selected=self.secondary.backend_name, fallback_from=self.primary.backend_name)

    def _with_backend_metadata(self, result: Any, *, selected: str, fallback_from: str | None) -> Any:
        metadata = dict(getattr(result, "metadata", {}) or {})
        metadata.setdefault("vision_backend", selected)
        if fallback_from:
            metadata["vision_fallback_from"] = fallback_from
        return type(result)(**{**result.to_dict(), "metadata": metadata})


def create_vision_backend(
    *,
    vision_config: Any,
    florence_config: Any,
    gemini_config: Any,
    run_dir: str | Path | None = None,
    florence_bundle: Any = None,
    gemini_client_factory: Any = None,
) -> VisionInferenceBackend:
    mode = str(getattr(vision_config, "backend_mode", "auto") or "auto").strip().lower()
    florence_backend = FlorenceVisionBackend(FlorenceInferenceEngine(florence_config, bundle=florence_bundle, run_dir=run_dir))
    gemini_backend = GeminiVisionBackend(config=gemini_config, run_dir=run_dir, client_factory=gemini_client_factory)
    if mode == "disabled":
        return DisabledVisionBackend(florence_backend.engine)
    if mode == "florence":
        return florence_backend
    if mode == "gemini":
        return gemini_backend
    return FallbackVisionBackend(florence_backend, gemini_backend)
