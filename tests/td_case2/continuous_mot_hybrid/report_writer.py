from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_html_from_markdown(path: Path, markdown_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"<html><body><pre>{markdown_text}</pre></body></html>", encoding="utf-8")


def write_final_report(
    *,
    report_path: Path,
    summary: dict[str, Any],
    sections: dict[str, dict[str, Any]],
) -> str:
    payload = {"status": "success", "summary": summary, "sections": sections}
    write_json(report_path.with_suffix(".json"), payload)
    lines = [
        "# Continuous MOT Hybrid Report",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    for section_name, section_payload in sections.items():
        lines.append(f"## {section_name.replace('_', ' ').title()}")
        for key, value in section_payload.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    markdown_text = "\n".join(lines).rstrip() + "\n"
    write_markdown(report_path.with_suffix(".md"), lines)
    write_html_from_markdown(report_path.with_suffix(".html"), markdown_text)
    return markdown_text
