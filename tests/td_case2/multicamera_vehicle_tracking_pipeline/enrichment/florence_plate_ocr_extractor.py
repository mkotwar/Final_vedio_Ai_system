from __future__ import annotations

from pathlib import Path

from ..models.florence_runtime import FlorenceRuntime, FlorenceRuntimeError
from .anpr_config import AnprOcrConfig, AnprValidationConfig
from .india_registration_validator import validate_indian_registration
from .plate_image_preprocessor import generate_plate_variants
from .plate_models import PlateCandidate, PlateOcrAttempt, PlateOcrResult
from .plate_text_normalizer import normalize_registration_text


class FlorencePlateOcrExtractor:
    def __init__(self, *, runtime: FlorenceRuntime, ocr_config: AnprOcrConfig, validation_config: AnprValidationConfig) -> None:
        self.runtime = runtime
        self.ocr_config = ocr_config
        self.validation_config = validation_config

    def extract_attempts(self, candidate: PlateCandidate) -> list[PlateOcrAttempt]:
        variants = generate_plate_variants(
            candidate.local_file_path,
            output_directory=candidate.local_file_path.parent,
            max_variants=self.ocr_config.max_variants_per_candidate,
        )
        attempts: list[PlateOcrAttempt] = []
        for variant in variants[: self.ocr_config.maximum_retries + 1 if self.ocr_config.maximum_retries >= 0 else len(variants)]:
            try:
                parsed = self.runtime.run_image_task(image_path=variant.output_path, prompt=self.ocr_config.task_prompt)
                raw_text = _extract_raw_text(parsed)
            except (FlorenceRuntimeError, FileNotFoundError) as exc:
                raw_text = str(exc)
            normalized = normalize_registration_text(raw_text, country_profile=self.validation_config.country_profile)
            confidence_seed = max(candidate.detector_confidence, self.ocr_config.minimum_confidence)
            validation = validate_indian_registration(
                normalized,
                ocr_confidence=confidence_seed,
                minimum_length=self.validation_config.minimum_normalized_length,
                maximum_length=self.validation_config.maximum_normalized_length,
            )
            verification_status = "VERIFIED" if validation.is_verified else ("UNVERIFIED" if validation.normalized_text else "UNKNOWN")
            confidence = max(0.0, min(1.0, confidence_seed + validation.confidence_adjustment))
            attempts.append(
                PlateOcrAttempt(
                    candidate_storage_uri=candidate.relative_storage_uri,
                    source_vehicle_storage_uri=candidate.source_vehicle_storage_uri,
                    source_vehicle_role=candidate.source_vehicle_role,
                    source_image_kind=candidate.source_image_kind,
                    candidate_source=candidate.candidate_source,
                    preprocessing_variant=variant.variant_name,
                    frame_number=candidate.frame_number,
                    video_time_seconds=candidate.video_time_seconds,
                    detector_confidence=candidate.detector_confidence,
                    raw_text=raw_text,
                    normalized_text=validation.normalized_text,
                    confidence=confidence,
                    status=validation.status,
                    verification_status=verification_status,
                    metadata={
                        "matched_pattern": validation.matched_pattern,
                        "candidate_values": list(validation.candidate_values),
                        "reasons": list(validation.reasons),
                        "source_vehicle_role": candidate.source_vehicle_role,
                        "variant_path": variant.output_path.name,
                        "variant_metadata": variant.metadata,
                        "heuristic_region_name": candidate.heuristic_region_name,
                    },
                )
            )
        return attempts

    def extract(self, candidate: PlateCandidate) -> PlateOcrResult:
        attempts = self.extract_attempts(candidate)
        if not attempts:
            return PlateOcrResult(
                raw_text="UNKNOWN",
                normalized_text=None,
                confidence=0.0,
                status="UNKNOWN",
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
        best = _select_best_attempt(attempts)
        return PlateOcrResult(
            raw_text=best.raw_text,
            normalized_text=best.normalized_text,
            confidence=best.confidence,
            status=best.status,
            verification_status=best.verification_status,
            country_profile=self.validation_config.country_profile,
            backend=self.ocr_config.backend,
            model_name=self.runtime.model_path.name if hasattr(self.runtime, "model_path") else None,
            adapter_name=self.runtime.adapter_path.name if getattr(self.runtime, "adapter_path", None) is not None else None,
            source_vehicle_track_id=candidate.track_uuid,
            source_plate_storage_uri=candidate.relative_storage_uri,
            source_vehicle_storage_uri=candidate.source_vehicle_storage_uri,
            metadata=best.metadata,
        )


def _select_best_attempt(attempts: list[PlateOcrAttempt]) -> PlateOcrAttempt:
    return sorted(
        attempts,
        key=lambda attempt: (
            1 if attempt.verification_status == "VERIFIED" else 0,
            1 if attempt.status == "PARTIAL" else 0,
            1 if bool(attempt.normalized_text) else 0,
            attempt.confidence,
        ),
        reverse=True,
    )[0]


def _extract_raw_text(parsed: str) -> str:
    text = str(parsed or "").strip()
    for prefix in ("<OCR>", "</OCR>", "<s>", "</s>", "Plate:", "PLATE:"):
        text = text.replace(prefix, " ")
    return " ".join(text.split()).strip() or "UNKNOWN"
