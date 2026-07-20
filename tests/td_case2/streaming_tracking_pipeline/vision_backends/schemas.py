from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


VISION_BACKEND_MODES = {"auto", "florence", "gemini", "disabled"}


@dataclass(frozen=True)
class VisionBackendHealth:
    backend: str
    available: bool
    detail: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeminiStructuredOcr:
    raw_text: str = ""
    confidence: float = 0.0
    notes: str = ""


@dataclass(frozen=True)
class GeminiStructuredColour:
    raw_text: str = ""
    normalized_colour: str = "unknown"
    confidence: float = 0.0
    notes: str = ""
