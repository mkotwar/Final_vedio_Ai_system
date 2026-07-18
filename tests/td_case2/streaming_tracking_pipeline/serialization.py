"""JSON-safe serialization helpers for streaming tracking pipeline records."""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


RUNTIME_FIELD_NAMES = {"frame"}


def to_json_safe(value: Any, *, exclude_runtime_fields: bool = True) -> Any:
    """Convert supported Python values into JSON-safe values."""

    if is_dataclass(value) and not isinstance(value, type):
        return dataclass_to_dict(value, exclude_runtime_fields=exclude_runtime_fields)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [to_json_safe(item, exclude_runtime_fields=exclude_runtime_fields) for item in value]
    if isinstance(value, list):
        return [to_json_safe(item, exclude_runtime_fields=exclude_runtime_fields) for item in value]
    if isinstance(value, dict):
        return {
            str(to_json_safe(key, exclude_runtime_fields=exclude_runtime_fields)): to_json_safe(
                item,
                exclude_runtime_fields=exclude_runtime_fields,
            )
            for key, item in value.items()
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported value for JSON serialization: {type(value).__name__}.")


def dataclass_to_dict(value: Any, *, exclude_runtime_fields: bool = True) -> dict[str, Any]:
    """Convert a dataclass instance to a JSON-safe dictionary."""

    if not is_dataclass(value) or isinstance(value, type):
        raise TypeError("dataclass_to_dict expects a dataclass instance.")
    payload: dict[str, Any] = {}
    for field_info in fields(value):
        if exclude_runtime_fields and field_info.name in RUNTIME_FIELD_NAMES:
            continue
        payload[field_info.name] = to_json_safe(
            getattr(value, field_info.name),
            exclude_runtime_fields=exclude_runtime_fields,
        )
    return payload


def write_json(path: str | Path, value: Any) -> Path:
    """Write deterministic UTF-8 JSON, creating parent directories."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(to_json_safe(value), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def read_json(path: str | Path) -> Any:
    """Read JSON from disk."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, values: list[Any]) -> Path:
    """Write JSON Lines with one serialized value per line."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(to_json_safe(value), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return output_path


def read_jsonl(path: str | Path) -> list[Any]:
    """Read JSON Lines from disk."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
