from __future__ import annotations

import re

from .search_query_schemas import VehicleSearchQuery


OBJECT_CLASS_ALIASES = {
    "person": "person",
    "persons": "person",
    "people": "person",
    "pedestrian": "person",
    "pedestrians": "person",
    "car": "car",
    "cars": "car",
    "truck": "truck",
    "trucks": "truck",
    "3wheeler": "3wheeler",
    "3wheelers": "3wheeler",
    "rickshaw": "3wheeler",
    "rickshaws": "3wheeler",
    "bus": "bus",
    "buses": "bus",
    "motorcycle": "motorcycle",
    "motorcycles": "motorcycle",
    "bike": "motorcycle",
    "bikes": "motorcycle",
    "motorbike": "motorcycle",
    "motorbikes": "motorcycle",
    "scooter": "motorcycle",
    "scooters": "motorcycle",
    "two": "two",
    "wheeler": "wheeler",
    "bicycle": "bicycle",
    "bicycles": "bicycle",
    "cycle": "bicycle",
    "cycles": "bicycle",
}

COLOURS = {
    "black",
    "blue",
    "gray",
    "green",
    "grey",
    "maroon",
    "pink",
    "red",
    "silver",
    "white",
    "yellow",
    "orange",
    "brown",
    "beige",
    "purple",
    "multicolour",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "between",
    "find",
    "for",
    "in",
    "of",
    "plate",
    "plates",
    "second",
    "seconds",
    "show",
    "the",
    "to",
    "vehicle",
    "vehicles",
    "shirt",
    "shirts",
    "jacket",
    "jackets",
    "trouser",
    "trousers",
    "pants",
    "clothing",
    "wearing",
    "upper",
    "lower",
    "dominant",
    "object",
    "objects",
    "with",
    "without",
}


def parse_vehicle_search_query(raw_query: str) -> VehicleSearchQuery:
    text = " ".join(str(raw_query or "").strip().split())
    lowered = text.lower()
    consumed_spans: list[tuple[int, int]] = []
    warnings: list[str] = []

    start_time, end_time = _parse_between_time(lowered, consumed_spans)
    track_id, track_generation = _parse_track_identity(lowered, consumed_spans)
    plate_statuses = _parse_plate_statuses(lowered, consumed_spans)
    object_classes = _parse_classes(lowered, consumed_spans)
    _parse_two_wheeler(lowered, consumed_spans, object_classes)
    colours = _parse_colours(lowered, consumed_spans)
    plate_text, plate_prefix = _parse_plate_terms(text, lowered, consumed_spans)

    free_text_tokens = _remaining_tokens(lowered, consumed_spans)
    if not text:
        warnings.append("empty_query")
    if "grey" in colours:
        colours = ["gray" if colour == "grey" else colour for colour in colours]

    return VehicleSearchQuery(
        raw_query=text,
        object_classes=_dedupe(object_classes),
        colours=_dedupe(colours),
        plate_text=plate_text,
        plate_prefix=plate_prefix,
        plate_statuses=_dedupe(plate_statuses),
        start_time_sec=start_time,
        end_time_sec=end_time,
        track_id=track_id,
        track_generation=track_generation,
        free_text_tokens=free_text_tokens,
        warnings=warnings,
    )


def _parse_between_time(lowered: str, consumed_spans: list[tuple[int, int]]) -> tuple[float | None, float | None]:
    pattern = re.compile(r"\bbetween\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\s+(seconds?|secs?|minutes?|mins?)\b")
    match = pattern.search(lowered)
    if not match:
        return None, None
    start = float(match.group(1))
    end = float(match.group(2))
    unit = match.group(3)
    if unit.startswith(("minute", "min")):
        start *= 60.0
        end *= 60.0
    consumed_spans.append(match.span())
    return (min(start, end), max(start, end))


def _parse_track_identity(lowered: str, consumed_spans: list[tuple[int, int]]) -> tuple[int | None, int | None]:
    track_id = None
    generation = None
    track_match = re.search(r"\btrack[_\s-]*(\d+)\b", lowered)
    if track_match:
        track_id = int(track_match.group(1))
        consumed_spans.append(track_match.span())
    generation_match = re.search(r"\b(?:generation|gen)[_\s-]*(\d+)\b", lowered)
    if generation_match:
        generation = int(generation_match.group(1))
        consumed_spans.append(generation_match.span())
    return track_id, generation


def _parse_plate_statuses(lowered: str, consumed_spans: list[tuple[int, int]]) -> list[str]:
    statuses: list[str] = []
    phrase_map = {
        r"\bverified\s+plates?\b": "verified",
        r"\bverified\s+ocr\b": "verified",
        r"\bverified\b": "verified",
        r"\bweak\s+ocr\b": "weak",
        r"\bweak\s+plates?\b": "weak",
        r"\bwithout\s+plates?\b": "no_plate_detected",
        r"\bno\s+plates?\b": "no_plate_detected",
        r"\bmissing\s+plates?\b": "no_plate_detected",
        r"\binvalid\s+plates?\b": "invalid",
    }
    for pattern, status in phrase_map.items():
        for match in re.finditer(pattern, lowered):
            statuses.append(status)
            consumed_spans.append(match.span())
    return statuses


def _parse_classes(lowered: str, consumed_spans: list[tuple[int, int]]) -> list[str]:
    classes: list[str] = []
    for match in re.finditer(r"\b[a-z]+\b", lowered):
        value = OBJECT_CLASS_ALIASES.get(match.group(0))
        if value and value not in {"two", "wheeler"}:
            classes.append(value)
            consumed_spans.append(match.span())
    return classes


def _parse_two_wheeler(lowered: str, consumed_spans: list[tuple[int, int]], classes: list[str]) -> None:
    for match in re.finditer(r"\btwo[-_\s]*wheelers?\b|\b2[-_\s]*wheelers?\b", lowered):
        classes.append("motorcycle")
        consumed_spans.append(match.span())


def _parse_colours(lowered: str, consumed_spans: list[tuple[int, int]]) -> list[str]:
    colours: list[str] = []
    for match in re.finditer(r"\b[a-z]+\b", lowered):
        token = match.group(0)
        if token in COLOURS:
            colours.append(token)
            consumed_spans.append(match.span())
    return colours


def _parse_plate_terms(text: str, lowered: str, consumed_spans: list[tuple[int, int]]) -> tuple[str | None, str | None]:
    for match in re.finditer(r"\b[a-zA-Z0-9]{2,12}\b", text):
        token = match.group(0)
        lowered_token = token.lower()
        if lowered_token in STOPWORDS or lowered_token in OBJECT_CLASS_ALIASES or lowered_token in COLOURS:
            continue
        if not (re.search(r"[a-zA-Z]", token) and re.search(r"\d", token)):
            continue
        normalized = re.sub(r"[^A-Za-z0-9]+", "", token).upper()
        consumed_spans.append(match.span())
        if len(normalized) >= 6:
            return normalized, None
        return None, normalized
    return None, None


def _remaining_tokens(lowered: str, consumed_spans: list[tuple[int, int]]) -> list[str]:
    masked = list(lowered)
    for start, end in consumed_spans:
        for index in range(start, end):
            masked[index] = " "
    tokens = re.findall(r"\b[a-z0-9]+\b", "".join(masked))
    return _dedupe([token for token in tokens if token not in STOPWORDS])


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    retained: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            retained.append(value)
    return retained
