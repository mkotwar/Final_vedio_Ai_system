from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..anpr_schemas import FlorenceColourResult, FlorenceOcrResult
from ..crop_selection import SelectedCropJob


class VisionInferenceBackend(ABC):
    @property
    @abstractmethod
    def backend_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def metrics(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def load(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def run_ocr(self, candidate: Any) -> FlorenceOcrResult:
        raise NotImplementedError

    @abstractmethod
    def run_colour(self, job: SelectedCropJob) -> FlorenceColourResult:
        raise NotImplementedError
