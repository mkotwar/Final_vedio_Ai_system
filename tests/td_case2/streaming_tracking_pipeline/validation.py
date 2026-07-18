"""Small validation helpers for streaming tracking pipeline schemas and config."""

from __future__ import annotations

import math
from enum import Enum
from pathlib import Path
from typing import TypeVar


TEnum = TypeVar("TEnum", bound=Enum)


def validate_non_empty_string(value: str, field_name: str) -> str:
    """Return a stripped non-empty string or raise ValueError."""

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be blank.")
    return stripped


def validate_finite_float(value: float, field_name: str) -> float:
    """Return a finite float or raise ValueError."""

    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite float.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite.")
    return parsed


def validate_probability(value: float, field_name: str) -> float:
    """Return a finite probability in the inclusive 0..1 range."""

    parsed = validate_finite_float(value, field_name)
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1.")
    return parsed


def validate_non_negative_int(value: int, field_name: str) -> int:
    """Return a non-negative integer or raise ValueError."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative.")
    return value


def validate_positive_int(value: int, field_name: str) -> int:
    """Return a positive integer or raise ValueError."""

    parsed = validate_non_negative_int(value, field_name)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive.")
    return parsed


def validate_positive_float(value: float, field_name: str) -> float:
    """Return a finite positive float or raise ValueError."""

    parsed = validate_finite_float(value, field_name)
    if parsed <= 0.0:
        raise ValueError(f"{field_name} must be positive.")
    return parsed


def normalize_optional_path(value: str | Path | None) -> str | None:
    """Normalize an optional path without checking whether it exists."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return str(Path(text).expanduser())


def validate_allowed_value(value: str, allowed_values: set[str] | tuple[str, ...], field_name: str) -> str:
    """Validate a string against a constrained set of values."""

    normalized = validate_non_empty_string(value, field_name).lower()
    allowed = {str(item).lower() for item in allowed_values}
    if normalized not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise ValueError(f"{field_name} must be one of: {allowed_text}.")
    return normalized


def validate_enum_value(value: str | TEnum, enum_type: type[TEnum], field_name: str) -> TEnum:
    """Coerce a string or enum member into the requested enum type."""

    if isinstance(value, enum_type):
        return value
    normalized = validate_non_empty_string(str(value), field_name).lower()
    try:
        return enum_type(normalized)
    except ValueError as exc:
        allowed_text = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{field_name} must be one of: {allowed_text}.") from exc
