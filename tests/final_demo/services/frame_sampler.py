from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2

from tests.final_demo.services.chunk_planner import read_json
from tests.final_demo.services.video_io import current_timestamp, write_json


ENV_FINAL_DEMO_SAMPLE_FPS = "FINAL_DEMO_SAMPLE_FPS"
ENV_FINAL_DEMO_MAX_CHUNKS = "FINAL_DEMO_MAX_CHUNKS"
DEFAULT_SAMPLE_FPS = 2.0


def read_sample_fps() -> float:
    raw_value = os.environ.get(ENV_FINAL_DEMO_SAMPLE_FPS, str(DEFAULT_SAMPLE_FPS))
    try:
        sample_fps = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {ENV_FINAL_DEMO_SAMPLE_FPS} must be a valid number. "
            f"Received: {raw_value!r}"
        ) from exc

    if sample_fps <= 0:
        raise ValueError(
            f"Environment variable {ENV_FINAL_DEMO_SAMPLE_FPS} must be greater than 0. "
            f"Received: {sample_fps}"
        )

    return sample_fps


def read_max_chunks() -> int | None:
    raw_value = os.environ.get(ENV_FINAL_DEMO_MAX_CHUNKS)
    if raw_value is None or raw_value == "":
        return None

    try:
        max_chunks = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {ENV_FINAL_DEMO_MAX_CHUNKS} must be a valid integer. "
            f"Received: {raw_value!r}"
        ) from exc

    if max_chunks <= 0:
        raise ValueError(
            f"Environment variable {ENV_FINAL_DEMO_MAX_CHUNKS} must be greater than 0. "
            f"Received: {max_chunks}"
        )

    return max_chunks


def to_repo_relative_path(path: Path) -> str:
    repo_root = Path(__file__).resolve().parents[3]
    return path.resolve().relative_to(repo_root).as_posix()


def format_timestamp_label(timestamp_seconds: float) -> str:
    total_milliseconds = int(round(timestamp_seconds * 1000))
    whole_seconds = total_milliseconds // 1000
    milliseconds = total_milliseconds % 1000
    return f"t{whole_seconds:06d}_{milliseconds:03d}"


def compute_sample_timestamps(
    start_time: float,
    end_time: float,
    sample_fps: float,
) -> list[float]:
    step_milliseconds = int(round((1.0 / sample_fps) * 1000))
    start_milliseconds = int(round(start_time * 1000))
    end_milliseconds = int(round(end_time * 1000))

    if step_milliseconds <= 0:
        raise ValueError(f"Computed invalid step size: {step_milliseconds} ms")

    timestamps: list[float] = []
    current_milliseconds = start_milliseconds
    while current_milliseconds <= end_milliseconds:
        timestamps.append(round(current_milliseconds / 1000.0, 3))
        current_milliseconds += step_milliseconds

    if not timestamps:
        timestamps.append(round(start_milliseconds / 1000.0, 3))

    return timestamps


def seek_and_read_frame(
    capture: cv2.VideoCapture,
    *,
    timestamp_seconds: float,
    video_fps: float,
) -> tuple[bool, Any]:
    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000.0)
    success, frame = capture.read()
    if success and frame is not None:
        return True, frame

    if video_fps > 0:
        frame_index = max(0, int(timestamp_seconds * video_fps))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        if success and frame is not None:
            return True, frame

    return False, None


def sample_frames_for_chunks(
    *,
    run_dir: Path,
    video_info: dict[str, Any],
    chunk_manifest: dict[str, Any],
    sample_fps: float,
    max_chunks: int | None,
) -> dict[str, Any]:
    video_path = Path(str(video_info.get("video_path", "")))
    if not video_path.exists():
        raise FileNotFoundError(f"Input video path does not exist: {video_path}")

    chunks = list(chunk_manifest.get("chunks", []))
    if not chunks:
        raise ValueError("Chunk manifest does not contain any chunks.")

    total_chunks_in_manifest = len(chunks)
    selected_chunks = chunks[:max_chunks] if max_chunks is not None else chunks
    sampled_frames_root = run_dir / "03_sampled_frames"
    sampled_frames_root.mkdir(parents=True, exist_ok=True)

    video_fps = float(video_info.get("fps", 0.0) or 0.0)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not open input video for frame sampling: {video_path}")

    frame_records: list[dict[str, Any]] = []
    per_chunk_summary: list[dict[str, Any]] = []
    warnings: list[str] = []

    try:
        for chunk in selected_chunks:
            chunk_id = str(chunk.get("chunk_id", ""))
            chunk_index = int(chunk.get("chunk_index", 0) or 0)
            start_time = round(float(chunk.get("start_time", 0.0) or 0.0), 3)
            end_time = round(float(chunk.get("end_time", 0.0) or 0.0), 3)
            chunk_output_dir = sampled_frames_root / chunk_id
            chunk_output_dir.mkdir(parents=True, exist_ok=True)

            frames_sampled = 0
            timestamps = compute_sample_timestamps(start_time, end_time, sample_fps)

            for timestamp_seconds in timestamps:
                success, frame = seek_and_read_frame(
                    capture,
                    timestamp_seconds=timestamp_seconds,
                    video_fps=video_fps,
                )
                if not success or frame is None:
                    warnings.append(
                        f"Failed to sample frame for {chunk_id} at {timestamp_seconds:.3f} seconds."
                    )
                    continue

                frame_id = f"{chunk_id}_{format_timestamp_label(timestamp_seconds)}"
                image_name = f"{frame_id}.jpg"
                image_output_path = chunk_output_dir / image_name
                write_success = cv2.imwrite(str(image_output_path), frame)
                if not write_success:
                    warnings.append(
                        f"Failed to write sampled frame for {chunk_id} at {timestamp_seconds:.3f} seconds."
                    )
                    continue

                frame_height, frame_width = frame.shape[:2]
                relative_timestamp_seconds = round(timestamp_seconds - start_time, 3)
                frame_records.append(
                    {
                        "frame_id": frame_id,
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_index,
                        "global_timestamp_seconds": round(timestamp_seconds, 3),
                        "relative_timestamp_seconds": relative_timestamp_seconds,
                        "frame_index_estimate": max(0, int(timestamp_seconds * video_fps)),
                        "image_path": to_repo_relative_path(image_output_path),
                        "source_video_path": str(video_path),
                        "width": int(frame_width),
                        "height": int(frame_height),
                        "sample_fps": round(sample_fps, 3),
                    }
                )
                frames_sampled += 1

            chunk["status"] = "sampled"
            steps_completed = list(chunk.get("steps_completed", []))
            if "03_frame_sampling" not in steps_completed:
                steps_completed.append("03_frame_sampling")
            chunk["steps_completed"] = steps_completed
            chunk["sampled_frame_count"] = frames_sampled
            chunk["sampled_frames_dir"] = to_repo_relative_path(chunk_output_dir)

            per_chunk_summary.append(
                {
                    "chunk_id": chunk_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "frames_sampled": frames_sampled,
                    "output_dir": to_repo_relative_path(chunk_output_dir),
                    "status": "sampled",
                }
            )
    finally:
        capture.release()

    if max_chunks is not None and max_chunks < total_chunks_in_manifest:
        warnings.append(
            f"Sampling limited to first {max_chunks} chunks by {ENV_FINAL_DEMO_MAX_CHUNKS}."
        )

    return {
        "updated_chunk_manifest": chunk_manifest,
        "index_payload": {
            "video_path": str(video_path),
            "sample_fps": round(sample_fps, 3),
            "total_chunks_in_manifest": total_chunks_in_manifest,
            "chunks_processed": len(selected_chunks),
            "total_frames_sampled": len(frame_records),
            "created_at": current_timestamp(),
            "frames": frame_records,
        },
        "report_payload": {
            "sample_fps": round(sample_fps, 3),
            "chunks_processed": len(selected_chunks),
            "total_frames_sampled": len(frame_records),
            "per_chunk_summary": per_chunk_summary,
            "warnings": warnings,
        },
    }


def update_run_manifest_for_sampling(run_manifest_path: Path) -> dict[str, Any]:
    run_manifest = read_json(run_manifest_path)
    completed_steps = list(run_manifest.get("completed_steps", []))
    if "03_frame_sampling" not in completed_steps:
        completed_steps.append("03_frame_sampling")

    run_manifest["completed_steps"] = completed_steps
    run_manifest["next_step"] = "04_yolo_detection"
    write_json(run_manifest_path, run_manifest)
    return run_manifest
