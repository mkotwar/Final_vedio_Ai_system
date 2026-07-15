from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CANONICAL_COLORS = {
    "black",
    "white",
    "gray",
    "silver",
    "red",
    "orange",
    "yellow",
    "gold",
    "brown",
    "beige",
    "green",
    "blue",
    "purple",
    "pink",
}

# This is a normalization vocabulary, not a constrained Florence output list.
# Florence remains free to caption the image; explicitly stated shades are folded
# into terms that an investigator is likely to type into search.
SHADE_TO_CANONICAL = {
    "pearl white": "white",
    "off white": "white",
    "off-white": "white",
    "ivory": "white",
    "cream": "white",
    "alabaster": "white",
    "snow white": "white",
    "white": "white",
    "jet black": "black",
    "matte black": "black",
    "midnight black": "black",
    "black": "black",
    "silver gray": "gray",
    "silver grey": "gray",
    "charcoal": "gray",
    "graphite": "gray",
    "gunmetal": "gray",
    "slate gray": "gray",
    "slate grey": "gray",
    "dark gray": "gray",
    "dark grey": "gray",
    "light gray": "gray",
    "light grey": "gray",
    "gray": "gray",
    "grey": "gray",
    "metallic silver": "silver",
    "silver": "silver",
    "burgundy": "red",
    "maroon": "red",
    "crimson": "red",
    "scarlet": "red",
    "wine red": "red",
    "ruby": "red",
    "red": "red",
    "navy blue": "blue",
    "navy": "blue",
    "metallic blue": "blue",
    "cobalt": "blue",
    "cerulean": "blue",
    "azure": "blue",
    "teal": "blue",
    "turquoise": "blue",
    "blue": "blue",
    "forest green": "green",
    "emerald": "green",
    "olive": "green",
    "lime": "green",
    "green": "green",
    "champagne gold": "gold",
    "champagne": "gold",
    "golden": "gold",
    "gold": "gold",
    "bronze": "brown",
    "copper": "brown",
    "chocolate": "brown",
    "coffee": "brown",
    "brown": "brown",
    "tan": "beige",
    "sand": "beige",
    "khaki": "beige",
    "beige": "beige",
    "mustard": "yellow",
    "lemon": "yellow",
    "yellow": "yellow",
    "amber": "orange",
    "rust": "orange",
    "orange": "orange",
    "violet": "purple",
    "plum": "purple",
    "lavender": "purple",
    "purple": "purple",
    "magenta": "pink",
    "rose": "pink",
    "pink": "pink",
}

_VEHICLE_NOUNS = (
    "car|vehicle|sedan|hatchback|suv|truck|pickup|van|minivan|bus|"
    "minibus|motorcycle|motorbike|scooter|auto rickshaw"
)
_COLOR_PHRASE_PATTERNS = (
    re.compile(rf"\b(?P<phrase>(?:[a-z][a-z-]*\s+){{1,5}})(?:{_VEHICLE_NOUNS})\b", re.IGNORECASE),
    re.compile(
        rf"\b(?:{_VEHICLE_NOUNS})\s+(?:is|appears|looks|painted|finished)\s+"
        r"(?P<phrase>(?:[a-z][a-z-]*\s*){1,4})",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?P<phrase>(?:[a-z][a-z-]*\s*){1,4})(?:paint|body|finish)\b", re.IGNORECASE),
)


def normalize_color_phrase(value: str | None) -> str | None:
    """Map a free-form Florence shade phrase to one common search color."""

    normalized = re.sub(r"[^a-z -]+", " ", str(value or "").lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return None

    matches: list[tuple[int, int, str]] = []
    for shade, canonical in SHADE_TO_CANONICAL.items():
        match = re.search(rf"\b{re.escape(shade)}\b", normalized)
        if match:
            matches.append((match.start(), -len(shade), canonical))
    return min(matches)[2] if matches else None


def extract_florence_vehicle_color(caption: str | None) -> tuple[str | None, str | None]:
    """Extract a vehicle-linked free-form shade and its canonical search color."""

    text = str(caption or "")
    for pattern in _COLOR_PHRASE_PATTERNS:
        for match in pattern.finditer(text):
            raw_phrase = re.sub(r"\s+", " ", match.group("phrase")).strip(" ,.-")
            canonical = normalize_color_phrase(raw_phrase)
            if canonical:
                return raw_phrase.lower(), canonical
    return None, None


def _classify_hsv_pixels(hsv: np.ndarray) -> np.ndarray:
    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    labels = np.full(hue.shape, "gray", dtype="<U8")

    # Low-saturation road, glass and compression noise should remain gray rather
    # than becoming a falsely confident blue/green vehicle color.
    achromatic = saturation < 65
    labels[achromatic & (value < 58)] = "black"
    labels[achromatic & (value > 195)] = "white"
    labels[achromatic & (value >= 58) & (value <= 195)] = "gray"

    chromatic = ~achromatic
    labels[chromatic & (value < 48)] = "black"
    labels[chromatic & ((hue <= 8) | (hue >= 171))] = "red"
    labels[chromatic & (hue >= 9) & (hue <= 20)] = "orange"
    labels[chromatic & (hue >= 21) & (hue <= 34)] = "yellow"
    labels[chromatic & (hue >= 35) & (hue <= 85)] = "green"
    labels[chromatic & (hue >= 86) & (hue <= 130)] = "blue"
    labels[chromatic & (hue >= 131) & (hue <= 160)] = "purple"
    labels[chromatic & (hue >= 161) & (hue <= 170)] = "pink"

    labels[chromatic & (hue >= 15) & (hue <= 25) & (value >= 125) & (value < 195)] = "gold"
    labels[chromatic & (hue >= 5) & (hue <= 25) & (value < 125)] = "brown"
    return labels


def dominant_vehicle_color(image_or_path: np.ndarray | Path | str) -> tuple[str, float]:
    """Estimate a canonical dominant color from the central vehicle crop."""

    if isinstance(image_or_path, np.ndarray):
        image = image_or_path
    else:
        image = cv2.imread(str(image_or_path))
    if image is None or image.size == 0:
        raise ValueError("Cannot estimate vehicle color from an empty image.")

    height, width = image.shape[:2]
    y0, y1 = int(height * 0.10), max(int(height * 0.90), 1)
    x0, x1 = int(width * 0.10), max(int(width * 0.90), 1)
    central = image[y0:y1, x0:x1]
    resized = cv2.resize(central, (96, 96), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    labels = _classify_hsv_pixels(hsv)

    yy, xx = np.mgrid[-1:1:96j, -1:1:96j]
    weights = np.exp(-1.8 * (xx * xx + yy * yy))
    scores = {color: float(weights[labels == color].sum()) for color in CANONICAL_COLORS}
    chromatic_colors = CANONICAL_COLORS - {"black", "white", "gray", "silver"}
    chromatic_best = max(chromatic_colors, key=lambda color: scores[color])
    achromatic_best = max(("black", "white", "gray"), key=lambda color: scores[color])
    total = max(float(weights.sum()), 1.0)

    # A modest chromatic surface should beat dark windows and bright reflections.
    color = chromatic_best if scores[chromatic_best] / total >= 0.16 else achromatic_best
    confidence = scores[color] / total
    return color, round(float(confidence), 4)


def resolve_vehicle_color(caption: str | None, image_path: Path | str) -> dict[str, Any]:
    """Resolve caption color first, with image analysis as a guaranteed fallback."""

    raw_phrase, canonical = extract_florence_vehicle_color(caption)
    if canonical:
        return {
            "color": canonical,
            "raw_prediction": raw_phrase,
            "source": "florence_caption_normalized",
            "confidence": "medium",
            "image_confidence": None,
        }

    fallback_color, fallback_confidence = dominant_vehicle_color(image_path)
    return {
        "color": fallback_color,
        "raw_prediction": None,
        "source": "image_dominant_color",
        "confidence": "medium" if fallback_confidence >= 0.35 else "low",
        "image_confidence": fallback_confidence,
    }
