from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def now_seconds() -> float:
    return time.perf_counter()


def build_step_result(step_id: int, step_name: str, started_at: float, status: str = "success") -> dict[str, Any]:
    duration_seconds = round(max(0.0, now_seconds() - started_at), 3)
    return {
        "step_id": step_id,
        "step_name": step_name,
        "duration_seconds": duration_seconds,
        "status": status,
    }


def build_parallel_branch_result(
    name: str,
    steps: list[int],
    started_at: float,
    status: str = "success",
) -> dict[str, Any]:
    return {
        "name": name,
        "steps": steps,
        "duration_seconds": round(max(0.0, now_seconds() - started_at), 3),
        "status": status,
    }


def build_parallel_section_result(
    section_name: str,
    started_at: float,
    branches: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "section_name": section_name,
        "duration_seconds": round(max(0.0, now_seconds() - started_at), 3),
        "branches": branches,
    }


def compute_slowest_steps(steps: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    sorted_steps = sorted(
        [step for step in steps if isinstance(step, dict)],
        key=lambda item: float(item.get("duration_seconds", 0.0) or 0.0),
        reverse=True,
    )
    return [
        {
            "step_id": item.get("step_id"),
            "step_name": item.get("step_name"),
            "duration_seconds": item.get("duration_seconds", 0.0),
            "status": item.get("status", "unknown"),
        }
        for item in sorted_steps[:limit]
    ]


def write_runtime_metrics(run_dir: Path, metrics: dict[str, Any]) -> Path:
    output_path = run_dir / "20_runtime_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return output_path
