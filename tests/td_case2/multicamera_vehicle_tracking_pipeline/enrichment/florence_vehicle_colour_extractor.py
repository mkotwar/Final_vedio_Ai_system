from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..models.florence_runtime import FlorenceRuntime, FlorenceRuntimeError
from .florence_colour_response_parser import parse_florence_colour_response
from .vehicle_colour_models import VehicleColourResult


@dataclass(frozen=True, slots=True)
class FlorenceVehicleColourExtractor:
    runtime: FlorenceRuntime
    prompt: str
    allowed_colours: Sequence[str]
    minimum_confidence: float

    def extract(
        self,
        image_path: Path,
        *,
        track_uuid: str,
        camera_code: str,
        source_storage_uri: str,
    ) -> VehicleColourResult:
        try:
            raw_output = self.runtime.run_image_task(
                image_path=image_path,
                prompt=self.prompt,
                disable_adapter=True,
            )
        except FileNotFoundError:
            return VehicleColourResult(
                canonical_colour="UNKNOWN",
                raw_output="",
                confidence=0.0,
                status="IMAGE_MISSING",
                source_storage_uri=source_storage_uri,
                metadata={"track_uuid": track_uuid, "camera_code": camera_code},
            )
        except FlorenceRuntimeError as exc:
            return VehicleColourResult(
                canonical_colour="UNKNOWN",
                raw_output=str(exc),
                confidence=0.0,
                status="MODEL_ERROR",
                source_storage_uri=source_storage_uri,
                metadata={"track_uuid": track_uuid, "camera_code": camera_code},
            )
        parsed = parse_florence_colour_response(
            raw_output,
            allowed_colours=self.allowed_colours,
            default_confidence=self.minimum_confidence,
        )
        status = parsed.status
        if parsed.primary_colour == "UNKNOWN":
            status = "UNKNOWN_RESULT" if parsed.status == "SUCCESS" else parsed.status
        elif parsed.confidence < self.minimum_confidence:
            status = "LOW_CONFIDENCE"
        else:
            status = "SUCCESS"
        canonical = parsed.primary_colour if status == "SUCCESS" else "UNKNOWN" if parsed.primary_colour == "UNKNOWN" else parsed.primary_colour
        return VehicleColourResult(
            canonical_colour=canonical if canonical in self.allowed_colours else "UNKNOWN",
            raw_output=raw_output,
            confidence=parsed.confidence,
            status=status,
            secondary_colour=parsed.secondary_colour,
            backend="florence",
            model_name=self.runtime.model_path.name,
            adapter_name=self.runtime.adapter_path.name if self.runtime.adapter_path is not None else None,
            source_storage_uri=source_storage_uri,
            metadata={"track_uuid": track_uuid, "camera_code": camera_code},
        )

