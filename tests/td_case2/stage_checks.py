from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any


def write_json(output_path: Path, payload: dict[str, Any]) -> Path:
    """Write a JSON payload with stable formatting."""

    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def read_json(input_path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in: {input_path}")
    return payload


def format_seconds_text(total_seconds: float) -> str:
    """Convert seconds into a simple human-readable timestamp."""

    rounded_seconds = max(0, int(round(total_seconds)))
    hours, remainder = divmod(rounded_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def build_failure_payload(exc: Exception) -> dict[str, str]:
    """Return a compact failure record for the stage gate report."""

    lines = traceback.format_exception_only(type(exc), exc)
    traceback_lines = traceback.format_exc().strip().splitlines()
    return {
        "status": "failed",
        "error_message": str(exc),
        "traceback_short": "\n".join(traceback_lines[-8:]) if traceback_lines else "".join(lines).strip(),
    }


def update_stage_gate_report(run_dir: Path, step_name: str, step_payload: dict[str, Any]) -> Path:
    """Create or update the td_case2 stage gate report."""

    report_path = run_dir / "00_stage_gate_report.json"
    if report_path.exists():
        report = read_json(report_path)
    else:
        report = {
            "overall_status": "success",
            "steps": {},
        }

    steps = report.setdefault("steps", {})
    if not isinstance(steps, dict):
        steps = {}
        report["steps"] = steps

    steps[step_name] = step_payload

    overall_status = "success"
    for item in steps.values():
        if isinstance(item, dict) and item.get("status") == "failed":
            overall_status = "failed"
            break
    report["overall_status"] = overall_status

    return write_json(report_path, report)
