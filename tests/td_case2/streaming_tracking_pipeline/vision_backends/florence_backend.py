from __future__ import annotations

from typing import Any

from ..florence_inference import FlorenceInferenceEngine
from .base import VisionInferenceBackend


class FlorenceVisionBackend(VisionInferenceBackend):
    def __init__(self, engine: FlorenceInferenceEngine) -> None:
        self.engine = engine

    @property
    def backend_name(self) -> str:
        return "florence"

    @property
    def metrics(self) -> dict[str, Any]:
        return self.engine.metrics

    def load(self) -> Any:
        return self.engine.load()

    def run_ocr(self, candidate: Any):
        return self.engine.run_ocr(candidate)

    def run_colour(self, job):
        return self.engine.run_colour(job)
