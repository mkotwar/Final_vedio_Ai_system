from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Sequence

from .florence_text_cleaning import clean_florence_text
from .vehicle_colour_mapping import normalize_vehicle_colour


@dataclass(frozen=True, slots=True)
class ParsedFlorenceColour:
    primary_colour: str
    secondary_colour: str | None
    confidence: float
    status: str
    raw_output: str
    cleaned_output: str


def parse_florence_colour_response(
    raw_output: str,
    *,
    allowed_colours: Sequence[str],
    default_confidence: float,
) -> ParsedFlorenceColour:
    cleaned = str(raw_output or "").strip()
    if not cleaned:
        return ParsedFlorenceColour("UNKNOWN", None, float(default_confidence), "PARSE_ERROR", cleaned, "")
    json_candidate = clean_florence_text(_strip_markdown_fences(cleaned))
    try:
        parsed = json.loads(json_candidate)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        primary = normalize_vehicle_colour(
            str(
                parsed.get(
                    "primary_colour",
                    parsed.get(
                        "primary_color",
                        parsed.get("colour", parsed.get("color", parsed.get("label", ""))),
                    ),
                )
            )
        )
        secondary_raw = parsed.get("secondary_colour")
        secondary = None if secondary_raw in (None, "", "null") else normalize_vehicle_colour(str(secondary_raw))
        confidence = _coerce_confidence(parsed.get("confidence"), default_confidence=default_confidence)
        if primary not in allowed_colours:
            primary = "UNKNOWN"
        if secondary is not None and secondary not in allowed_colours:
            secondary = None
        status = "SUCCESS" if primary != "UNKNOWN" else "UNKNOWN_RESULT"
        return ParsedFlorenceColour(primary, secondary, confidence, status, cleaned, json_candidate)
    label = _extract_plain_label(json_candidate)
    if normalize_vehicle_colour(label) == "UNKNOWN":
        label = _extract_explanatory_colour(json_candidate)
    primary = normalize_vehicle_colour(label)
    if primary not in allowed_colours:
        primary = "UNKNOWN"
    status = "SUCCESS" if primary != "UNKNOWN" else "UNKNOWN_RESULT"
    return ParsedFlorenceColour(primary, None, float(default_confidence), status, cleaned, json_candidate)


def _strip_markdown_fences(value: str) -> str:
    match = re.match(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", value, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else value


def _extract_plain_label(value: str) -> str:
    first_line = value.splitlines()[0]
    if ":" in first_line:
        first_line = first_line.split(":", 1)[1]
    token = re.split(r"[.,;()]", first_line, maxsplit=1)[0]
    return token.strip()


def _extract_explanatory_colour(value: str) -> str:
    lowered = value.lower()
    key_match = re.search(
        r"\b(?:primary\s+colour|primary\s+color|vehicle\s+colour|vehicle\s+color|colour|color|label)\b\s*[:\-]?\s*([a-zA-Z ]+)",
        lowered,
    )
    if key_match:
        candidate = key_match.group(1).strip()
        first = re.split(r"[.,;()]", candidate, maxsplit=1)[0].strip()
        if normalize_vehicle_colour(first) != "UNKNOWN":
            return first
    colour_words = [
        match.group(0)
        for match in re.finditer(
            r"\b(?:black|white|silver|grey|gray|red|blue|green|yellow|orange|brown|beige|purple|navy|maroon|cream|gold|violet)\b",
            lowered,
        )
    ]
    unique = {normalize_vehicle_colour(word) for word in colour_words}
    if len(unique) == 1:
        return colour_words[0]
    return ""


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
