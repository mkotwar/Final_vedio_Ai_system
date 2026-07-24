from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Sequence

from .vehicle_body_type_mapping import normalize_vehicle_body_type


@dataclass(frozen=True, slots=True)
class ParsedVehicleBodyType:
    canonical_body_type: str
    confidence: float
    status: str
    raw_output: str


def parse_florence_body_type_response(
    raw_output: str,
    *,
    allowed_body_types: Sequence[str],
    default_confidence: float,
) -> ParsedVehicleBodyType:
    cleaned = str(raw_output or "").strip()
    if not cleaned:
        return ParsedVehicleBodyType("UNKNOWN", float(default_confidence), "PARSE_ERROR", cleaned)
    json_candidate = _strip_markdown_fences(cleaned)
    try:
        parsed = json.loads(json_candidate)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        label = parsed.get("body_type", parsed.get("vehicle_body_type", parsed.get("label", parsed.get("type", ""))))
        canonical = normalize_vehicle_body_type(str(label))
        confidence = _coerce_confidence(parsed.get("confidence"), default_confidence=default_confidence)
        if canonical not in allowed_body_types:
            canonical = "UNKNOWN"
        status = "SUCCESS" if canonical != "UNKNOWN" else "UNKNOWN_RESULT"
        return ParsedVehicleBodyType(canonical, confidence, status, cleaned)
    label = _extract_plain_label(cleaned)
    canonical = normalize_vehicle_body_type(label)
    if canonical not in allowed_body_types:
        canonical = "UNKNOWN"
    status = "SUCCESS" if canonical != "UNKNOWN" else "UNKNOWN_RESULT"
    return ParsedVehicleBodyType(canonical, float(default_confidence), status, cleaned)


def _strip_markdown_fences(value: str) -> str:
    match = re.match(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", value, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else value


def _extract_plain_label(value: str) -> str:
    first_line = value.splitlines()[0].strip()
    if ":" in first_line:
        first_line = first_line.split(":", 1)[1]
    token = re.split(r"[.;,()]", first_line, maxsplit=1)[0]
    return token.strip()


def _coerce_confidence(value: object, *, default_confidence: float) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return float(default_confidence)
    if confidence < 0.0:
        return 0.0
    if confidence > 1.0:
        return 1.0
    return confidence
