from __future__ import annotations

from pathlib import Path
from typing import Any


def parse_simple_yaml(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = [item.strip() for item in line.split(":", 1)]
        if value.lower() in {"true", "false"}:
            payload[key] = value.lower() == "true"
        else:
            try:
                if "." in value:
                    payload[key] = float(value)
                else:
                    payload[key] = int(value)
            except ValueError:
                payload[key] = value.strip("\"'")
    return payload


def write_tracker_yaml(*, source_yaml: Path, destination_yaml: Path, overrides: dict[str, Any]) -> dict[str, Any]:
    resolved = parse_simple_yaml(source_yaml)
    resolved.update(overrides)
    lines = []
    for key, value in resolved.items():
        if isinstance(value, bool):
            serialized = "true" if value else "false"
        else:
            serialized = str(value)
        lines.append(f"{key}: {serialized}")
    destination_yaml.parent.mkdir(parents=True, exist_ok=True)
    destination_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return resolved
