from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from tests.final_demo.services.video_io import current_timestamp, write_json


ENV_FINAL_DEMO_CHUNK_SECONDS = "FINAL_DEMO_CHUNK_SECONDS"
ENV_FINAL_DEMO_CHUNK_OVERLAP_SECONDS = "FINAL_DEMO_CHUNK_OVERLAP_SECONDS"
DEFAULT_CHUNK_SECONDS = 300.0
DEFAULT_CHUNK_OVERLAP_SECONDS = 10.0


def read_json(input_path: Path) -> dict[str, Any]:
    return json.loads(input_path.read_text(encoding="utf-8"))


def read_positive_float_env(env_name: str, default_value: float) -> float:
    raw_value = os.environ.get(env_name, str(default_value))
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {env_name} must be a valid number. Received: {raw_value!r}"
        ) from exc

    if value <= 0:
        raise ValueError(
            f"Environment variable {env_name} must be greater than 0. Received: {value}"
        )

    return value


def read_non_negative_float_env(env_name: str, default_value: float) -> float:
    raw_value = os.environ.get(env_name, str(default_value))
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {env_name} must be a valid number. Received: {raw_value!r}"
        ) from exc

    if value < 0:
        raise ValueError(
            f"Environment variable {env_name} must be greater than or equal to 0. Received: {value}"
        )

    return value


def round_seconds(value: float) -> float:
    return round(float(value), 3)


def validate_chunk_settings(
    duration_seconds: float,
    chunk_seconds: float,
    overlap_seconds: float,
) -> None:
    if duration_seconds <= 0:
        raise ValueError(
            f"Video duration_seconds must be greater than 0. Received: {duration_seconds}"
        )
    if chunk_seconds <= 0:
        raise ValueError(f"chunk_seconds must be greater than 0. Received: {chunk_seconds}")
    if overlap_seconds < 0:
        raise ValueError(
            f"overlap_seconds must be greater than or equal to 0. Received: {overlap_seconds}"
        )
    if overlap_seconds >= chunk_seconds:
        raise ValueError(
            "overlap_seconds must be smaller than chunk_seconds. "
            f"Received overlap_seconds={overlap_seconds}, chunk_seconds={chunk_seconds}"
        )


def build_chunks(
    *,
    video_path: str,
    duration_seconds: float,
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[dict[str, Any]]:
    validate_chunk_settings(duration_seconds, chunk_seconds, overlap_seconds)

    rounded_duration = round_seconds(duration_seconds)
    rounded_chunk_seconds = round_seconds(chunk_seconds)
    rounded_overlap_seconds = round_seconds(overlap_seconds)

    if rounded_duration <= rounded_chunk_seconds:
        return [
            {
                "chunk_id": "chunk_000001",
                "chunk_index": 1,
                "start_time": 0.0,
                "end_time": rounded_duration,
                "duration_seconds": rounded_duration,
                "overlap_seconds": rounded_overlap_seconds,
                "status": "pending",
                "steps_completed": [],
                "source_video_path": video_path,
                "notes": "planned_only_not_extracted",
            }
        ]

    chunks: list[dict[str, Any]] = []
    start_time = 0.0
    chunk_index = 1
    step_size = chunk_seconds - overlap_seconds

    while start_time < duration_seconds:
        end_time = min(start_time + chunk_seconds, duration_seconds)
        rounded_start_time = round_seconds(start_time)
        rounded_end_time = round_seconds(end_time)

        chunks.append(
            {
                "chunk_id": f"chunk_{chunk_index:06d}",
                "chunk_index": chunk_index,
                "start_time": rounded_start_time,
                "end_time": rounded_end_time,
                "duration_seconds": round_seconds(rounded_end_time - rounded_start_time),
                "overlap_seconds": rounded_overlap_seconds,
                "status": "pending",
                "steps_completed": [],
                "source_video_path": video_path,
                "notes": "planned_only_not_extracted",
            }
        )

        if rounded_end_time >= rounded_duration:
            break

        start_time += step_size
        chunk_index += 1

    if chunks:
        chunks[-1]["end_time"] = rounded_duration
        chunks[-1]["duration_seconds"] = round_seconds(
            float(chunks[-1]["end_time"]) - float(chunks[-1]["start_time"])
        )

    return chunks


def build_chunk_manifest(
    *,
    video_path: str,
    duration_seconds: float,
    chunk_seconds: float,
    overlap_seconds: float,
) -> dict[str, Any]:
    chunks = build_chunks(
        video_path=video_path,
        duration_seconds=duration_seconds,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
    )
    return {
        "video_path": video_path,
        "video_duration_seconds": round_seconds(duration_seconds),
        "chunk_seconds": round_seconds(chunk_seconds),
        "overlap_seconds": round_seconds(overlap_seconds),
        "total_chunks": len(chunks),
        "created_at": current_timestamp(),
        "chunks": chunks,
    }


def update_run_manifest(run_manifest_path: Path) -> dict[str, Any]:
    run_manifest = read_json(run_manifest_path)
    completed_steps = list(run_manifest.get("completed_steps", []))
    if "02_chunk_planner" not in completed_steps:
        completed_steps.append("02_chunk_planner")

    run_manifest["completed_steps"] = completed_steps
    run_manifest["next_step"] = "03_frame_sampling"
    write_json(run_manifest_path, run_manifest)
    return run_manifest
