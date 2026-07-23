from __future__ import annotations

import re

from .plate_models import NormalizedRegistrationText, PlateValidationResult


VALID_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HP", "HR",
    "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR",
    "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB",
}

STRICT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("STANDARD", re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")),
    ("BH", re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")),
    ("NO_SERIES", re.compile(r"^[A-Z]{2}[0-9]{1,2}[0-9]{4}$")),
)


def validate_indian_registration(
    normalized: NormalizedRegistrationText,
    *,
    ocr_confidence: float,
    minimum_length: int,
    maximum_length: int,
) -> PlateValidationResult:
    reasons: list[str] = []
    for candidate in normalized.candidate_values:
        if len(candidate) < minimum_length:
            reasons.append("too_short")
            continue
        if len(candidate) > maximum_length:
            reasons.append("too_long")
            continue
        for pattern_name, pattern in STRICT_PATTERNS:
            if not pattern.match(candidate):
                continue
            if pattern_name != "BH" and candidate[:2] not in VALID_STATE_CODES:
                reasons.append("invalid_state_code")
                continue
            verified = ocr_confidence >= 0.5
            return PlateValidationResult(
                normalized_text=candidate,
                is_verified=verified,
                status="VERIFIED" if verified else "PROBABLE",
                confidence_adjustment=0.0 if verified else -0.15,
                matched_pattern=pattern_name,
                reasons=tuple(["format_match"] + (["low_confidence"] if not verified else [])),
                candidate_values=normalized.candidate_values,
            )
    status = "UNKNOWN" if not normalized.cleaned_text else "PARTIAL"
    return PlateValidationResult(
        normalized_text=None,
        is_verified=False,
        status=status,
        confidence_adjustment=-0.25,
        matched_pattern=None,
        reasons=tuple(dict.fromkeys(reasons or ["no_pattern_match"])),
        candidate_values=normalized.candidate_values,
    )
