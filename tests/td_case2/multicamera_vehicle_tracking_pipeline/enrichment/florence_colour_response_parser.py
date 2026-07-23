from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Sequence

from .vehicle_colour_mapping import normalize_vehicle_colour


@dataclass(frozen=True, slots=True)
class ParsedFlorenceColour:
    primary_colour: str
    secondary_colour: str | None
    confidence: float
    status: str
    raw_output: str


def parse_florence_colour_response(
    raw_output: str,
    *,
    allowed_colours: Sequence[str],
    default_confidence: float,
) -> ParsedFlorenceColour:
    cleaned = str(raw_output or "").strip()
    if not cleaned:
        return ParsedFlorenceColour("UNKNOWN", None, float(default_confidence), "PARSE_ERROR", cleaned)
    json_candidate = _strip_markdown_fences(cleaned)
    try:
        parsed = json.loads(json_candidate)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        primary = normalize_vehicle_colour(str(parsed.get("primary_colour", "")))
        secondary_raw = parsed.get("secondary_colour")
        secondary = None if secondary_raw in (None, "", "null") else normalize_vehicle_colour(str(secondary_raw))
        confidence = _coerce_confidence(parsed.get("confidence"), default_confidence=default_confidence)
        if primary not in allowed_colours:
            primary = "UNKNOWN"
        if secondary is not None and secondary not in allowed_colours:
            secondary = None
        status = "SUCCESS" if primary != "UNKNOWN" else "UNKNOWN_RESULT"
        return ParsedFlorenceColour(primary, secondary, confidence, status, cleaned)
    label = _extract_plain_label(cleaned)
    primary = normalize_vehicle_colour(label)
    if primary not in allowed_colours:
        primary = "UNKNOWN"
    status = "SUCCESS" if primary != "UNKNOWN" else "UNKNOWN_RESULT"
    return ParsedFlorenceColour(primary, None, float(default_confidence), status, cleaned)


def _strip_markdown_fences(value: str) -> str:
    match = re.match(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", value, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else value


def _extract_plain_label(value: str) -> str:
    first_line = value.splitlines()[0]
    token = re.split(r"[:.,;()]", first_line, maxsplit=1)[0]
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

