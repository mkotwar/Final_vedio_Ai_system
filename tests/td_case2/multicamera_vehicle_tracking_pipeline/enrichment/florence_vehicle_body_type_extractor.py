from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..models.florence_runtime import FlorenceRuntime, FlorenceRuntimeError
from .florence_body_type_response_parser import parse_florence_body_type_response
from .vehicle_body_type_models import VehicleBodyTypeResult


@dataclass(frozen=True, slots=True)
class FlorenceVehicleBodyTypeExtractor:
    runtime: FlorenceRuntime
    prompt: str
    allowed_body_types: Sequence[str]
    minimum_confidence: float
    default_confidence_when_missing: float

    def extract(
        self,
        image_path: Path,
        *,
        source_storage_uri: str | None = None,
    ) -> VehicleBodyTypeResult:
        try:
            raw_output = self.runtime.run_image_task(
                image_path=image_path,
                prompt=self.prompt,
                disable_adapter=True,
            )
        except FileNotFoundError:
            return VehicleBodyTypeResult(
                canonical_body_type="UNKNOWN",
                raw_output="",
                confidence=0.0,
                status="IMAGE_MISSING",
                source_storage_uri=source_storage_uri,
            )
        except FlorenceRuntimeError as exc:
            message = str(exc)
            status = "IMAGE_INVALID" if "load Florence input image" in message else "MODEL_ERROR"
            return VehicleBodyTypeResult(
                canonical_body_type="UNKNOWN",
                raw_output=message,
                confidence=0.0,
                status=status,
                model_name=self.runtime.model_path.name,
                adapter_name=self.runtime.adapter_path.name if self.runtime.adapter_path is not None else None,
                source_storage_uri=source_storage_uri,
            )
        parsed = parse_florence_body_type_response(
            raw_output,
            allowed_body_types=self.allowed_body_types,
            default_confidence=self.default_confidence_when_missing,
        )
        status = parsed.status
        canonical = parsed.canonical_body_type
        if canonical == "UNKNOWN":
            status = "UNKNOWN_RESULT" if parsed.status == "SUCCESS" else parsed.status
        elif parsed.confidence < self.minimum_confidence:
            status = "LOW_CONFIDENCE"
            canonical = "UNKNOWN"
        else:
            status = "SUCCESS"
        return VehicleBodyTypeResult(
            canonical_body_type=canonical if canonical in self.allowed_body_types else "UNKNOWN",
            raw_output=raw_output,
            confidence=parsed.confidence,
            status=status,
            backend="florence",
            model_name=self.runtime.model_path.name,
            adapter_name=self.runtime.adapter_path.name if self.runtime.adapter_path is not None else None,
            source_storage_uri=source_storage_uri,
        )
