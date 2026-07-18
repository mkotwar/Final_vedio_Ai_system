from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


UNRELATED_OCR_WORDS = {
    "IND",
    "INDIA",
    "BHARAT",
    "SPORT",
    "SPORTS",
    "RE",
    "REG",
    "REGD",
    "GOVT",
    "GOVERNMENT",
    "POLICE",
    "PRIVATE",
}


@dataclass(frozen=True)
class NormalizedPlateText:
    raw_text: str
    normalized_text: str
    removed_words: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_plate_ocr_text(raw_text: str) -> NormalizedPlateText:
    raw = str(raw_text or "")
    upper = raw.upper()
    words = re.findall(r"[A-Z0-9]+", upper)
    removed: list[str] = []
    kept: list[str] = []
    for word in words:
        if word in UNRELATED_OCR_WORDS:
            removed.append(word)
            continue
        kept.append(word)
    normalized = re.sub(r"[^A-Z0-9]+", "", "".join(kept))
    return NormalizedPlateText(raw_text=raw, normalized_text=normalized, removed_words=removed)


def extract_plate_substrings(normalized_text: str) -> list[str]:
    text = re.sub(r"[^A-Z0-9]+", "", str(normalized_text or "").upper())
    if not text:
        return []
    candidates: set[str] = set()
    if len(text) <= 11:
        return [text]
    for start in range(0, max(1, len(text) - 5)):
        if not text[start : start + 2].isalpha():
            continue
        for length in range(min(11, len(text) - start), 5, -1):
            chunk = text[start : start + length]
            if re.fullmatch(r"[A-Z]{2}[A-Z0-9]{1,2}[A-Z]{1,3}[A-Z0-9]{1,4}", chunk):
                candidates.add(chunk)
    if text and not candidates:
        candidates.add(text)
    return sorted(candidates, key=lambda item: (-_plate_shape_score(item), -len(item), item))[:30]


def _plate_shape_score(text: str) -> float:
    if re.fullmatch(r"[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}", text):
        return 1.0
    if re.fullmatch(r"[A-Z0-9]{6,11}", text) and any(ch.isalpha() for ch in text) and any(ch.isdigit() for ch in text):
        return 0.6
    return 0.2
