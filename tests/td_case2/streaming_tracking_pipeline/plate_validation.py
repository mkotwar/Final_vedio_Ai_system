from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Any

from .plate_normalization import extract_plate_substrings, normalize_plate_ocr_text
from .plate_validation_schemas import PlateTextCandidate, PlateValidationConfig


INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK",
    "UP", "WB",
}
OCR_ALTERNATIVES = {
    "O": "0",
    "0": "O",
    "I": "1",
    "1": "I",
    "L": "1",
    "S": "5",
    "5": "S",
    "B": "8",
    "8": "B",
    "Z": "2",
    "2": "Z",
    "G": "6",
    "6": "G",
    "Q": "0",
}
STRICT_RE = re.compile(r"^([A-Z]{2})([0-9]{1,2})([A-Z]{1,3})([0-9]{1,4})$")
RELAXED_RE = re.compile(r"^([A-Z0-9]{2})([A-Z0-9]{1,2})([A-Z0-9]{1,3})([A-Z0-9]{1,4})$")


@dataclass(frozen=True)
class PlateFormatValidation:
    text: str
    format_status: str
    format_score: float
    rejection_reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def validate_plate_format(text: str) -> PlateFormatValidation:
    value = str(text or "").upper()
    if len(value) < 4:
        return PlateFormatValidation(value, "not_plate_like", 0.0, ["too_short"])
    if value.isdigit():
        return PlateFormatValidation(value, "partial_plate", 0.25, ["numeric_only"])
    strict = STRICT_RE.fullmatch(value)
    if strict and strict.group(1) in INDIAN_STATE_CODES:
        score = 1.0
        if len(strict.group(3)) > 2:
            score = 0.92
        return PlateFormatValidation(value, "strict_format_match", score, metadata={"groups": strict.groups()})
    relaxed = RELAXED_RE.fullmatch(value)
    if relaxed and _state_like(relaxed.group(1)):
        score = 0.78 if _contains_ocr_ambiguous(value) else 0.70
        return PlateFormatValidation(value, "relaxed_format_match", score, ["requires_review"], {"groups": relaxed.groups()})
    if 5 <= len(value) <= 11 and any(ch.isalpha() for ch in value) and any(ch.isdigit() for ch in value):
        return PlateFormatValidation(value, "partial_plate", 0.45, ["partial_or_nonstandard_format"])
    return PlateFormatValidation(value, "not_plate_like", 0.05, ["format_mismatch"])


def generate_controlled_variants(
    text: str,
    *,
    maximum_substitutions_per_candidate: int = 2,
    maximum_generated_variants: int = 20,
) -> list[tuple[str, list[dict[str, Any]]]]:
    value = str(text or "").upper()
    positions = [(index, OCR_ALTERNATIVES[ch]) for index, ch in enumerate(value) if ch in OCR_ALTERNATIVES]
    variants: list[tuple[str, list[dict[str, Any]]]] = [(value, [])]
    for count in range(1, max(0, maximum_substitutions_per_candidate) + 1):
        for combo in itertools.combinations(positions, count):
            chars = list(value)
            substitutions: list[dict[str, Any]] = []
            for index, replacement in combo:
                original = chars[index]
                chars[index] = replacement
                substitutions.append({"index": index, "from": original, "to": replacement})
            variants.append(("".join(chars), substitutions))
            if len(variants) >= maximum_generated_variants:
                return variants
    return variants


def best_validated_variant(text: str, config: PlateValidationConfig) -> tuple[PlateFormatValidation, str | None, list[dict[str, Any]]]:
    original = validate_plate_format(text)
    best = original
    best_text: str | None = None
    best_substitutions: list[dict[str, Any]] = []
    for variant, substitutions in generate_controlled_variants(
        text,
        maximum_substitutions_per_candidate=config.maximum_substitutions_per_candidate,
        maximum_generated_variants=config.maximum_generated_variants,
    ):
        validated = validate_plate_format(variant)
        penalty = 0.04 * len(substitutions)
        adjusted = PlateFormatValidation(
            validated.text,
            validated.format_status,
            max(0.0, round(validated.format_score - penalty, 6)),
            list(validated.rejection_reasons),
            dict(validated.metadata),
        )
        if adjusted.format_score > best.format_score and _status_rank(adjusted.format_status) >= _status_rank(best.format_status):
            best = adjusted
            best_text = variant if substitutions else None
            best_substitutions = substitutions
    return best, best_text, best_substitutions


def candidates_from_ocr_record(record: dict[str, Any], config: PlateValidationConfig, evidence: dict[str, Any] | None = None) -> list[PlateTextCandidate]:
    normalized = normalize_plate_ocr_text(str(record.get("raw_text") or record.get("normalized_text") or ""))
    extracted_values = extract_plate_substrings(normalized.normalized_text)
    if not extracted_values and normalized.normalized_text:
        extracted_values = [normalized.normalized_text]
    candidates: list[PlateTextCandidate] = []
    for extracted in extracted_values:
        validation, corrected_text, substitutions = best_validated_variant(extracted, config)
        reasons = list(validation.rejection_reasons)
        if normalized.removed_words:
            reasons.append("removed_unrelated_ocr_words")
        if record.get("status") == "empty_output":
            reasons.append("ocr_empty")
        elif record.get("status") not in {None, "success"}:
            reasons.append("ocr_failed")
        metadata = {
            "normalization": {"removed_words": list(normalized.removed_words)},
            "source_ocr_status": record.get("status"),
            "plate_rank": record.get("plate_rank"),
        }
        if evidence:
            metadata["evidence"] = evidence
        candidates.append(
            PlateTextCandidate(
                source_id=str(record.get("source_id", "")),
                track_id=int(record.get("track_id", 0) or 0),
                track_generation=int(record.get("track_generation", 0) or 0),
                raw_ocr_text=str(record.get("raw_text") or ""),
                normalized_text=normalized.normalized_text,
                extracted_text=extracted,
                corrected_text=corrected_text,
                substitutions=substitutions,
                format_status=validation.format_status,
                format_score=round(validation.format_score, 6),
                ocr_confidence=_optional_float(record.get("ocr_confidence")),
                plate_detection_confidence=_optional_float((evidence or {}).get("plate_detection_confidence")),
                crop_role=record.get("crop_role"),
                crop_rank=int(record["crop_rank"]) if record.get("crop_rank") is not None else None,
                frame_index=int(record["frame_index"]) if record.get("frame_index") is not None else None,
                timestamp_sec=_optional_float((evidence or {}).get("timestamp_sec")),
                plate_crop_path=record.get("plate_crop_path"),
                source_vehicle_crop_path=(evidence or {}).get("vehicle_crop_path"),
                rejection_reasons=reasons,
                metadata=metadata,
            )
        )
    return sorted(candidates, key=lambda item: (-item.format_score, -len(item.text_for_selection()), item.text_for_selection()))


def _state_like(value: str) -> bool:
    if value in INDIAN_STATE_CODES:
        return True
    corrected = "".join(OCR_ALTERNATIVES.get(ch, ch) for ch in value)
    return corrected in INDIAN_STATE_CODES


def _contains_ocr_ambiguous(value: str) -> bool:
    return any(ch in OCR_ALTERNATIVES for ch in value)


def _status_rank(status: str) -> int:
    return {
        "not_plate_like": 0,
        "partial_plate": 1,
        "relaxed_format_match": 2,
        "strict_format_match": 3,
    }.get(status, 0)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
