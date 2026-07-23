from __future__ import annotations

from pathlib import Path

from ..models.florence_runtime import FlorenceRuntime, FlorenceRuntimeError
from .anpr_config import AnprOcrConfig, AnprValidationConfig
from .india_registration_validator import validate_indian_registration
from .plate_image_preprocessor import preprocess_plate_image
from .plate_models import PlateCandidate, PlateOcrResult
from .plate_text_normalizer import normalize_registration_text


class FlorencePlateOcrExtractor:
    def __init__(self, *, runtime: FlorenceRuntime, ocr_config: AnprOcrConfig, validation_config: AnprValidationConfig) -> None:
        self.runtime = runtime
        self.ocr_config = ocr_config
        self.validation_config = validation_config

    def extract(self, candidate: PlateCandidate) -> PlateOcrResult:
        attempted_paths = [candidate.local_file_path]
        if self.ocr_config.retry_with_preprocessing:
            attempted_paths.append(
                preprocess_plate_image(
                    candidate.local_file_path,
                    output_path=candidate.local_file_path.with_name(candidate.local_file_path.stem + "_preprocessed.jpg"),
                )
            )
        last_raw = ""
        for image_path in attempted_paths[: self.ocr_config.maximum_retries + 1]:
            try:
                parsed = self.runtime.run_image_task(image_path=image_path, prompt=self.ocr_config.task_prompt)
            except (FlorenceRuntimeError, FileNotFoundError) as exc:
                last_raw = str(exc)
                continue
            raw_text = _extract_raw_text(parsed)
            last_raw = raw_text
            normalized = normalize_registration_text(raw_text, country_profile=self.validation_config.country_profile)
            validation = validate_indian_registration(
                normalized,
                ocr_confidence=max(candidate.detector_confidence, self.ocr_config.minimum_confidence),
                minimum_length=self.validation_config.minimum_normalized_length,
                maximum_length=self.validation_config.maximum_normalized_length,
            )
            status = validation.status
            verification_status = "VERIFIED" if validation.is_verified else ("UNVERIFIED" if validation.normalized_text else "UNKNOWN")
            confidence = max(0.0, min(1.0, candidate.detector_confidence + validation.confidence_adjustment))
            if validation.normalized_text or image_path == attempted_paths[min(len(attempted_paths), self.ocr_config.maximum_retries + 1) - 1]:
                return PlateOcrResult(
                    raw_text=raw_text,
                    normalized_text=validation.normalized_text,
                    confidence=confidence,
                    status=status,
                    verification_status=verification_status,
                    country_profile=self.validation_config.country_profile,
                    backend=self.ocr_config.backend,
                    model_name=self.runtime.model_path.name if hasattr(self.runtime, "model_path") else None,
                    adapter_name=self.runtime.adapter_path.name if getattr(self.runtime, "adapter_path", None) is not None else None,
                    source_vehicle_track_id=candidate.track_uuid,
                    source_plate_storage_uri=candidate.relative_storage_uri,
                    source_vehicle_storage_uri=candidate.source_vehicle_storage_uri,
                    metadata={
                        "matched_pattern": validation.matched_pattern,
                        "candidate_values": list(validation.candidate_values),
                        "reasons": list(validation.reasons),
                        "source_vehicle_role": candidate.source_vehicle_role,
                    },
                )
        return PlateOcrResult(
            raw_text=last_raw or "UNKNOWN",
            normalized_text=None,
            confidence=0.0,
            status="MODEL_ERROR" if last_raw else "UNKNOWN",
            verification_status="UNKNOWN",
            country_profile=self.validation_config.country_profile,
            backend=self.ocr_config.backend,
            model_name=self.runtime.model_path.name if hasattr(self.runtime, "model_path") else None,
            adapter_name=self.runtime.adapter_path.name if getattr(self.runtime, "adapter_path", None) is not None else None,
            source_vehicle_track_id=candidate.track_uuid,
            source_plate_storage_uri=candidate.relative_storage_uri,
            source_vehicle_storage_uri=candidate.source_vehicle_storage_uri,
            metadata={"source_vehicle_role": candidate.source_vehicle_role},
        )


def _extract_raw_text(parsed: str) -> str:
    text = str(parsed or "").strip()
    for prefix in ("<OCR>", "</OCR>", "<s>", "</s>", "Plate:", "PLATE:"):
        text = text.replace(prefix, " ")
    return " ".join(text.split()).strip() or "UNKNOWN"
