from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from config import (
    DEFAULT_SAMPLE_EVERY_SECONDS,
    ENV_INPUT_VIDEO,
    ENV_OUTPUT_ROOT,
    ENV_SAMPLE_EVERY_SECONDS,
    SUPPORTED_VIDEO_EXTENSIONS,
    TDCase2Config,
    default_output_root,
)
from stage_checks import (
    build_failure_payload,
    format_seconds_text,
    read_json,
    update_stage_gate_report,
    write_json,
)


LOG_PREFIX = "[td-case2]"


def log(message: str) -> None:
    """Print a td_case2-prefixed console message."""

    print(f"{LOG_PREFIX} {message}")


def read_config() -> TDCase2Config:
    """Read and validate environment-driven configuration for td_case2."""

    raw_input_video = os.environ.get(ENV_INPUT_VIDEO, "").strip()
    if not raw_input_video:
        raise ValueError(
            f"Environment variable {ENV_INPUT_VIDEO} is required. "
            "Set it to the full local video path before running this testcase."
        )

    input_video_path = Path(raw_input_video).expanduser()
    if not input_video_path.is_absolute():
        input_video_path = input_video_path.resolve()

    if not input_video_path.exists():
        raise FileNotFoundError(f"Input video path does not exist: {input_video_path}")
    if not input_video_path.is_file():
        raise FileNotFoundError(f"Input video path is not a file: {input_video_path}")

    suffix = input_video_path.suffix.lower()
    if suffix and suffix not in SUPPORTED_VIDEO_EXTENSIONS:
        supported_list = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        raise ValueError(
            f"Unsupported video extension for td_case2: {input_video_path.suffix}. "
            f"Supported extensions include: {supported_list}"
        )

    raw_sample_seconds = os.environ.get(
        ENV_SAMPLE_EVERY_SECONDS,
        str(DEFAULT_SAMPLE_EVERY_SECONDS),
    ).strip()
    try:
        sample_every_seconds = float(raw_sample_seconds)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {ENV_SAMPLE_EVERY_SECONDS} must be a valid number. "
            f"Received: {raw_sample_seconds!r}"
        ) from exc

    if sample_every_seconds <= 0:
        raise ValueError(
            f"Environment variable {ENV_SAMPLE_EVERY_SECONDS} must be greater than 0. "
            f"Received: {sample_every_seconds}"
        )

    raw_output_root = os.environ.get(ENV_OUTPUT_ROOT, "").strip()
    output_root = Path(raw_output_root).expanduser() if raw_output_root else default_output_root()
    if not output_root.is_absolute():
        output_root = (Path.cwd() / output_root).resolve()

    return TDCase2Config(
        input_video_path=input_video_path.resolve(),
        sample_every_seconds=sample_every_seconds,
        output_root=output_root,
    )


def create_run_dir(config: TDCase2Config) -> Path:
    """Create a new isolated td_case2 run directory."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.output_root / f"{config.input_video_path.stem}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def extract_video_info(video_path: Path) -> dict[str, Any]:
    """Open a video with OpenCV and extract basic metadata for step 01."""

    log(f"Opening video: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"OpenCV could not open input video: {video_path}. "
            "Check that the file exists, is readable, and is a supported codec/container on this machine."
        )

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        try:
            backend = capture.getBackendName()
        except Exception:
            backend = "opencv"
    finally:
        capture.release()

    duration_seconds = round(frame_count / fps, 3) if fps > 0 else 0.0
    return {
        "input_video_path": str(video_path),
        "video_name": video_path.name,
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "duration_seconds": duration_seconds,
        "duration_text": format_seconds_text(duration_seconds),
        "width": width,
        "height": height,
        "backend": backend,
        "status": "success",
    }


def sample_frames(run_dir: Path, video_info: dict[str, Any], sample_every_seconds: float) -> dict[str, Any]:
    """Sample full-scene frames at a fixed interval and write a manifest."""

    input_video_path = Path(str(video_info["input_video_path"]))
    fps = float(video_info.get("fps", 0.0) or 0.0)
    frame_count = int(video_info.get("frame_count", 0) or 0)
    duration_seconds = float(video_info.get("duration_seconds", 0.0) or 0.0)

    if fps <= 0:
        raise ValueError("Cannot sample frames because video FPS is 0 or invalid.")

    sample_every_frames = max(1, int(round(fps * sample_every_seconds)))
    frame_indices = list(range(0, max(frame_count, 0), sample_every_frames))
    frames_dir = run_dir / "02_sampled_frames"
    frames_dir.mkdir(parents=True, exist_ok=False)

    log(
        f"Sampling full frames every {sample_every_seconds} seconds "
        f"({sample_every_frames} frame interval at {fps:.3f} FPS)"
    )

    capture = cv2.VideoCapture(str(input_video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not reopen input video for sampling: {input_video_path}")

    sampled_frames: list[dict[str, Any]] = []
    try:
        for sample_index, frame_idx in enumerate(frame_indices, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = capture.read()
            if not success or frame is None:
                raise RuntimeError(f"Failed to read frame index {frame_idx} from video: {input_video_path}")

            frame_id = f"frame_{frame_idx:06d}"
            image_name = f"{frame_id}.jpg"
            image_output_path = frames_dir / image_name
            write_success = cv2.imwrite(str(image_output_path), frame)
            if not write_success:
                raise RuntimeError(f"Failed to write sampled frame image: {image_output_path}")

            timestamp_seconds = round(frame_idx / fps, 3)
            sampled_frames.append(
                {
                    "sample_index": sample_index,
                    "frame_id": frame_id,
                    "frame_idx": frame_idx,
                    "timestamp_seconds": timestamp_seconds,
                    "timestamp_text": format_seconds_text(timestamp_seconds),
                    "image_path": (Path("02_sampled_frames") / image_name).as_posix(),
                    "original_video_path": str(input_video_path),
                }
            )
    finally:
        capture.release()

    expected_sample_count = len(frame_indices)
    manifest = {
        "status": "success",
        "input_video_path": str(input_video_path),
        "sample_every_seconds": sample_every_seconds,
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "duration_seconds": duration_seconds,
        "expected_sample_count": expected_sample_count,
        "actual_sample_count": len(sampled_frames),
        "sampled_frames_folder": "02_sampled_frames",
        "sampled_frames": sampled_frames,
    }
    return manifest


def main() -> None:
    """Run td_case2 step 01 and 02 end to end in an isolated debug folder."""

    config = read_config()
    run_dir = create_run_dir(config)
    log(f"Created run directory: {run_dir}")

    try:
        video_info = extract_video_info(config.input_video_path)
        write_json(run_dir / "01_video_info.json", video_info)
        update_stage_gate_report(
            run_dir,
            "01_video_info",
            {
                "status": "success",
                "fps": video_info["fps"],
                "frame_count": video_info["frame_count"],
                "duration_seconds": video_info["duration_seconds"],
            },
        )
        log("Step 01 complete: video info written")
    except Exception as exc:
        update_stage_gate_report(run_dir, "01_video_info", build_failure_payload(exc))
        log(f"Step 01 failed: {exc}")
        log(f"Run directory: {run_dir}")
        raise

    try:
        video_info = read_json(run_dir / "01_video_info.json")
        sample_manifest = sample_frames(run_dir, video_info, config.sample_every_seconds)
        write_json(run_dir / "02_sampled_frames.json", sample_manifest)
        update_stage_gate_report(
            run_dir,
            "02_base_frame_sampling",
            {
                "status": "success",
                "sample_every_seconds": config.sample_every_seconds,
                "actual_sample_count": sample_manifest["actual_sample_count"],
                "sampled_frames_folder_exists": (run_dir / "02_sampled_frames").exists(),
            },
        )
        log("Step 02 complete: sampled frames written")
    except Exception as exc:
        update_stage_gate_report(run_dir, "02_base_frame_sampling", build_failure_payload(exc))
        log(f"Step 02 failed: {exc}")
        log(f"Run directory: {run_dir}")
        raise

    log(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
