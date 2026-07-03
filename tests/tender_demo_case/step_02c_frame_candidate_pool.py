from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_required_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required frame candidate pool input file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in JSON file: {path}")
    return [item for item in payload if isinstance(item, dict)]


def _load_optional_adaptive_items(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "02b_adaptive_frames.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return []
    items = payload.get("items", [])
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def create_frame_candidate_pool(run_dir: Path) -> dict[str, Any]:
    print("[tender-demo] Starting Step 02C: frame candidate pool")
    sampled_frames = _load_required_list(run_dir / "02_sampled_frames.json")
    adaptive_frames = _load_optional_adaptive_items(run_dir)
    merged: list[dict[str, Any]] = []
    deduplicated_count = 0

    def find_existing(timestamp_seconds: float) -> dict[str, Any] | None:
        for item in merged:
            if abs(_safe_float(item.get("timestamp_seconds"), -9999.0) - timestamp_seconds) <= 0.5:
                return item
        return None

    for item in sampled_frames:
        merged.append(
            {
                "frame_id": item.get("sample_id", item.get("frame_id")),
                "sample_id": item.get("sample_id"),
                "frame_idx": item.get("frame_idx"),
                "timestamp_seconds": _safe_float(item.get("timestamp_seconds")),
                "frame_path": item.get("frame_path"),
                "source_fixed_sample": True,
                "source_adaptive": False,
                "adaptive_reasons": [],
                "adaptive_motion_score": 0.0,
                "adaptive_histogram_diff": 0.0,
                "adaptive_similarity_score": 0.0,
            }
        )

    for item in adaptive_frames:
        timestamp_seconds = _safe_float(item.get("timestamp_seconds"))
        existing = find_existing(timestamp_seconds)
        if existing is not None:
            deduplicated_count += 1
            existing["source_adaptive"] = True
            existing["adaptive_reasons"] = list(item.get("keep_reasons", []) or [])
            existing["adaptive_motion_score"] = _safe_float(item.get("motion_score"))
            existing["adaptive_histogram_diff"] = _safe_float(item.get("histogram_diff"))
            existing["adaptive_similarity_score"] = _safe_float(item.get("similarity_score"))
            if not existing.get("frame_path") and item.get("frame_path"):
                existing["frame_path"] = item.get("frame_path")
            continue

        merged.append(
            {
                "frame_id": item.get("frame_id"),
                "sample_id": None,
                "frame_idx": item.get("frame_idx"),
                "timestamp_seconds": timestamp_seconds,
                "frame_path": item.get("frame_path"),
                "source_fixed_sample": False,
                "source_adaptive": True,
                "adaptive_reasons": list(item.get("keep_reasons", []) or []),
                "adaptive_motion_score": _safe_float(item.get("motion_score")),
                "adaptive_histogram_diff": _safe_float(item.get("histogram_diff")),
                "adaptive_similarity_score": _safe_float(item.get("similarity_score")),
            }
        )

    merged.sort(key=lambda item: (_safe_float(item.get("timestamp_seconds")), _safe_float(item.get("frame_idx"))))
    payload = {
        "metadata": {
            "fixed_sample_count": len(sampled_frames),
            "adaptive_count": len(adaptive_frames),
            "merged_count": len(merged),
            "deduplicated_count": deduplicated_count,
        },
        "items": merged,
    }
    output_path = run_dir / "02c_frame_candidate_pool.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[tender-demo] Frame candidate pool size: {len(merged)}")
    print(f"[tender-demo] Frame candidate pool output path: {output_path}")
    return payload

