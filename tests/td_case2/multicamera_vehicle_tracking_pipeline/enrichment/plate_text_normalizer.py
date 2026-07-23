from __future__ import annotations

import re

from .plate_models import NormalizedRegistrationText


_PREFIX_PATTERN = re.compile(r"^(PLATE|REGISTRATION|NUMBER|NO|REGNO|PLATENO)+", re.IGNORECASE)
_SAFE_SEPARATORS = re.compile(r"[\s\-_:\.]+")


def normalize_registration_text(raw_text: str, *, country_profile: str) -> NormalizedRegistrationText:
    text = str(raw_text or "").strip()
    cleaned = text.upper().replace("\n", " ").replace("\r", " ")
    cleaned = _PREFIX_PATTERN.sub("", cleaned).strip()
    cleaned = _SAFE_SEPARATORS.sub("", cleaned)
    cleaned = re.sub(r"[^A-Z0-9]", "", cleaned)
    transformations: list[str] = []
    if cleaned != text:
        transformations.append("canonical_cleanup")
    candidates = [cleaned] if cleaned else []
    ambiguity_flags: list[str] = []
    for candidate in list(candidates):
        corrected = _generate_position_aware_candidates(candidate, ambiguity_flags)
        for item in corrected:
            if item not in candidates:
                candidates.append(item)
    return NormalizedRegistrationText(
        raw_text=text,
        cleaned_text=cleaned,
        candidate_values=tuple(item for item in candidates if item),
        transformations=tuple(transformations),
        ambiguity_flags=tuple(dict.fromkeys(ambiguity_flags)),
    )


def _generate_position_aware_candidates(value: str, ambiguity_flags: list[str]) -> list[str]:
    if not value:
        return []
    candidates: list[str] = []
    chars = list(value)
    for index, char in enumerate(chars):
        replacement = None
        if index in {0, 1}:
            if char == "0":
                replacement = "O"
            elif char == "1":
                replacement = "I"
        elif index in {2, 3}:
            if char == "O":
                replacement = "0"
            elif char == "I":
                replacement = "1"
            elif char == "L":
                replacement = "1"
        else:
            if char == "0" and index < len(chars) - 4:
                replacement = "O"
            elif char == "8" and index < len(chars) - 4:
                replacement = "B"
            elif char == "5" and index < len(chars) - 4:
                replacement = "S"
            elif char == "2" and index < len(chars) - 4:
                replacement = "Z"
            elif char == "6" and index < len(chars) - 4:
                replacement = "G"
        if replacement is None:
            continue
        cloned = list(chars)
        cloned[index] = replacement
        ambiguity_flags.append(f"{char}->{replacement}@{index}")
        candidates.append("".join(cloned))
    return candidates
