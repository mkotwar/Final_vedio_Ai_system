from __future__ import annotations

import importlib.util
import json
import os
from datetime import datetime
from pathlib import Path

import cv2


ENV_VIDEO_PATH = "TENDER_DEMO_INPUT_VIDEO"
ENV_SAMPLE_EVERY_SECONDS = "TENDER_DEMO_SAMPLE_EVERY_SECONDS"
ENV_MOTION_THRESHOLD = "TENDER_DEMO_MOTION_THRESHOLD"
ENV_MAX_GAP_SECONDS = "TENDER_DEMO_MAX_GAP_SECONDS"
ENV_MAX_CLIP_SECONDS = "TENDER_DEMO_MAX_CLIP_SECONDS"
ENV_CLIP_OVERLAP_SECONDS = "TENDER_DEMO_CLIP_OVERLAP_SECONDS"
ENV_CONTEXT_BEFORE_SECONDS = "TENDER_DEMO_CONTEXT_BEFORE_SECONDS"
ENV_CONTEXT_AFTER_SECONDS = "TENDER_DEMO_CONTEXT_AFTER_SECONDS"
ENV_MIN_EXPANDED_CLIP_SECONDS = "TENDER_DEMO_MIN_EXPANDED_CLIP_SECONDS"
DEFAULT_SAMPLE_EVERY_SECONDS = 1.0
DEFAULT_MOTION_THRESHOLD = 0.20
DEFAULT_MAX_GAP_SECONDS = 2.0
DEFAULT_MAX_CLIP_SECONDS = 12.0
DEFAULT_CLIP_OVERLAP_SECONDS = 2.0
DEFAULT_CONTEXT_BEFORE_SECONDS = 2.0
DEFAULT_CONTEXT_AFTER_SECONDS = 2.0
DEFAULT_MIN_EXPANDED_CLIP_SECONDS = 4.0


def _get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_video_path() -> Path:
    print(f"[tender-demo] Reading input video path from ${ENV_VIDEO_PATH}")
    raw_value = os.environ.get(ENV_VIDEO_PATH)
    if not raw_value:
        raise ValueError(
            f"Environment variable {ENV_VIDEO_PATH} is not set. "
            "Set it before running this script."
        )

    video_path = Path(raw_value).expanduser().resolve()
    print(f"[tender-demo] Resolved video path: {video_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video path does not exist: {video_path}")
    if not video_path.is_file():
        raise FileNotFoundError(f"Video path is not a file: {video_path}")

    return video_path


def _create_debug_run_dir(video_path: Path) -> Path:
    repo_root = _get_repo_root()
    debug_runs_root = repo_root / "tests" / "tender_demo_case" / "debug_runs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"{video_path.stem}_{timestamp}"
    run_dir = debug_runs_root / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"[tender-demo] Created debug run directory: {run_dir}")
    return run_dir


def _extract_video_info(video_path: Path) -> dict[str, object]:
    print("[tender-demo] Opening video with OpenCV")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()

    duration_seconds = round(total_frames / fps, 3) if fps > 0 else 0.0
    file_size_mb = round(video_path.stat().st_size / (1024 * 1024), 3)

    video_info = {
        "video_path": str(video_path),
        "video_name": video_path.name,
        "fps": round(fps, 3),
        "total_frames": total_frames,
        "duration_seconds": duration_seconds,
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "file_size_mb": file_size_mb,
    }
    print("[tender-demo] Video information extracted successfully")
    return video_info


def _write_video_info(run_dir: Path, video_info: dict[str, object]) -> Path:
    output_path = run_dir / "01_video_info.json"
    print(f"[tender-demo] Writing video info to: {output_path}")
    output_path.write_text(json.dumps(video_info, indent=2), encoding="utf-8")
    return output_path


def _read_sample_every_seconds() -> float:
    raw_value = os.environ.get(ENV_SAMPLE_EVERY_SECONDS, str(DEFAULT_SAMPLE_EVERY_SECONDS))
    print(
        f"[tender-demo] Reading sample interval from ${ENV_SAMPLE_EVERY_SECONDS} "
        f"(default: {DEFAULT_SAMPLE_EVERY_SECONDS})"
    )
    try:
        sample_every_seconds = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {ENV_SAMPLE_EVERY_SECONDS} must be a valid number. "
            f"Received: {raw_value!r}"
        ) from exc

    if sample_every_seconds <= 0:
        raise ValueError(
            f"Environment variable {ENV_SAMPLE_EVERY_SECONDS} must be greater than 0. "
            f"Received: {sample_every_seconds}"
        )

    print(f"[tender-demo] Using sample interval: {sample_every_seconds} seconds")
    return sample_every_seconds


def _read_motion_threshold() -> float:
    raw_value = os.environ.get(ENV_MOTION_THRESHOLD, str(DEFAULT_MOTION_THRESHOLD))
    print(
        f"[tender-demo] Reading motion threshold from ${ENV_MOTION_THRESHOLD} "
        f"(default: {DEFAULT_MOTION_THRESHOLD})"
    )
    try:
        motion_threshold = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {ENV_MOTION_THRESHOLD} must be a valid number. "
            f"Received: {raw_value!r}"
        ) from exc

    if not 0.0 <= motion_threshold <= 1.0:
        raise ValueError(
            f"Environment variable {ENV_MOTION_THRESHOLD} must be between 0.0 and 1.0. "
            f"Received: {motion_threshold}"
        )

    print(f"[tender-demo] Using motion threshold: {motion_threshold}")
    return motion_threshold


def _read_positive_float_env(
    env_name: str,
    default_value: float,
    label: str,
) -> float:
    raw_value = os.environ.get(env_name, str(default_value))
    print(f"[tender-demo] Reading {label} from ${env_name} (default: {default_value})")
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {env_name} must be a valid number. "
            f"Received: {raw_value!r}"
        ) from exc

    if value <= 0:
        raise ValueError(
            f"Environment variable {env_name} must be greater than 0. "
            f"Received: {value}"
        )

    print(f"[tender-demo] Using {label}: {value}")
    return value


def _to_repo_relative_path(path: Path) -> str:
    return path.resolve().relative_to(_get_repo_root()).as_posix()


def _sample_base_frames(
    video_path: Path,
    run_dir: Path,
    fps: float,
    total_frames: int,
    sample_every_seconds: float,
) -> tuple[Path, Path, list[dict[str, object]]]:
    if fps <= 0:
        raise ValueError("Cannot sample frames because video FPS is 0 or invalid.")

    sample_every_frames = max(1, int(round(fps * sample_every_seconds)))
    frames_dir = run_dir / "02_sampled_frames"
    frames_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "02_sampled_frames.json"

    print(f"[tender-demo] Creating sampled frames folder: {frames_dir}")
    print(
        f"[tender-demo] Sampling base frames every {sample_every_seconds} seconds "
        f"({sample_every_frames} frames at {fps:.3f} FPS)"
    )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not open video for frame sampling: {video_path}")

    manifest_entries: list[dict[str, object]] = []

    try:
        for frame_idx in range(0, total_frames, sample_every_frames):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = capture.read()
            if not success or frame is None:
                print(
                    f"[tender-demo] Warning: failed to read frame {frame_idx}; "
                    "skipping this sample"
                )
                continue

            frame_filename = f"frame_{frame_idx:06d}.jpg"
            frame_output_path = frames_dir / frame_filename
            write_success = cv2.imwrite(str(frame_output_path), frame)
            if not write_success:
                print(
                    f"[tender-demo] Warning: failed to write sampled frame {frame_idx}; "
                    "skipping this sample"
                )
                continue

            sample_id = f"sample_{len(manifest_entries) + 1:06d}"
            manifest_entries.append(
                {
                    "sample_id": sample_id,
                    "frame_idx": frame_idx,
                    "timestamp_seconds": round(frame_idx / fps, 3),
                    "frame_path": _to_repo_relative_path(frame_output_path),
                    "sample_reason": "base_interval",
                }
            )
    finally:
        capture.release()

    print(f"[tender-demo] Total sampled frames: {len(manifest_entries)}")
    print(f"[tender-demo] Sampled frames output folder: {frames_dir}")

    manifest_path.write_text(json.dumps(manifest_entries, indent=2), encoding="utf-8")
    print(f"[tender-demo] Sample manifest written to: {manifest_path}")
    return frames_dir, manifest_path, manifest_entries


def _get_motion_level(motion_score_norm: float) -> str:
    if motion_score_norm < 0.20:
        return "low"
    if motion_score_norm < 0.55:
        return "medium"
    return "high"


def _preprocess_frame_for_motion(frame_path: Path) -> cv2.typing.MatLike:
    image = cv2.imread(str(frame_path))
    if image is None:
        raise RuntimeError(f"Failed to load sampled frame image: {frame_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    return blurred


def _score_motion_on_sampled_frames(
    sampled_frames: list[dict[str, object]],
    run_dir: Path,
) -> tuple[Path, list[dict[str, object]]]:
    if not sampled_frames:
        raise ValueError("Cannot compute motion scores because no sampled frames were created.")

    print("[tender-demo] Starting Step 3: motion scoring on sampled frames")
    motion_scores_path = run_dir / "03_motion_scores.json"
    repo_root = _get_repo_root()

    raw_scores: list[float] = []
    motion_entries: list[dict[str, object]] = []
    previous_processed_frame = None

    for sample in sampled_frames:
        frame_path = repo_root / str(sample["frame_path"])
        processed_frame = _preprocess_frame_for_motion(frame_path)

        if previous_processed_frame is None:
            raw_motion_score = 0.0
        else:
            diff = cv2.absdiff(previous_processed_frame, processed_frame)
            raw_motion_score = round(float(diff.mean()), 6)

        previous_processed_frame = processed_frame
        raw_scores.append(raw_motion_score)
        motion_entries.append(
            {
                "sample_id": sample["sample_id"],
                "frame_idx": sample["frame_idx"],
                "timestamp_seconds": sample["timestamp_seconds"],
                "frame_path": sample["frame_path"],
                "raw_motion_score": raw_motion_score,
            }
        )

    min_raw_score = min(raw_scores)
    max_raw_score = max(raw_scores)
    score_range = max_raw_score - min_raw_score

    low_count = 0
    medium_count = 0
    high_count = 0

    for entry in motion_entries:
        raw_motion_score = float(entry["raw_motion_score"])
        if score_range == 0:
            motion_score_norm = 0.0
        else:
            motion_score_norm = round((raw_motion_score - min_raw_score) / score_range, 6)

        motion_level = _get_motion_level(motion_score_norm)
        entry["motion_score_norm"] = motion_score_norm
        entry["motion_level"] = motion_level

        if motion_level == "low":
            low_count += 1
        elif motion_level == "medium":
            medium_count += 1
        else:
            high_count += 1

    motion_scores_path.write_text(json.dumps(motion_entries, indent=2), encoding="utf-8")

    print(f"[tender-demo] Total frames scored: {len(motion_entries)}")
    print(f"[tender-demo] Min raw motion score: {round(min_raw_score, 6)}")
    print(f"[tender-demo] Max raw motion score: {round(max_raw_score, 6)}")
    print(f"[tender-demo] Low motion frames: {low_count}")
    print(f"[tender-demo] Medium motion frames: {medium_count}")
    print(f"[tender-demo] High motion frames: {high_count}")
    print(f"[tender-demo] Motion scores output path: {motion_scores_path}")
    return motion_scores_path, motion_entries


def _select_motion_candidates(
    motion_scores: list[dict[str, object]],
    run_dir: Path,
    motion_threshold: float,
) -> tuple[Path, list[dict[str, object]]]:
    print("[tender-demo] Starting Step 4: motion candidate selection")

    selected_candidates: list[dict[str, object]] = []
    for motion_score in motion_scores:
        if float(motion_score["motion_score_norm"]) >= motion_threshold:
            selected_candidates.append(
                {
                    "sample_id": motion_score["sample_id"],
                    "frame_idx": motion_score["frame_idx"],
                    "timestamp_seconds": motion_score["timestamp_seconds"],
                    "frame_path": motion_score["frame_path"],
                    "raw_motion_score": motion_score["raw_motion_score"],
                    "motion_score_norm": motion_score["motion_score_norm"],
                    "motion_level": motion_score["motion_level"],
                    "selection_reason": "motion_threshold",
                }
            )

    if not selected_candidates:
        print(
            "[tender-demo] No candidates met the motion threshold; "
            "falling back to top 10 frames by motion score"
        )
        fallback_candidates = sorted(
            motion_scores,
            key=lambda item: (
                float(item["motion_score_norm"]),
                float(item["timestamp_seconds"]),
            ),
            reverse=True,
        )[:10]
        selected_candidates = [
            {
                "sample_id": motion_score["sample_id"],
                "frame_idx": motion_score["frame_idx"],
                "timestamp_seconds": motion_score["timestamp_seconds"],
                "frame_path": motion_score["frame_path"],
                "raw_motion_score": motion_score["raw_motion_score"],
                "motion_score_norm": motion_score["motion_score_norm"],
                "motion_level": motion_score["motion_level"],
                "selection_reason": "fallback_top_motion",
            }
            for motion_score in fallback_candidates
        ]

    selected_candidates.sort(key=lambda item: float(item["timestamp_seconds"]))

    for index, candidate in enumerate(selected_candidates, start=1):
        candidate["candidate_id"] = f"candidate_{index:06d}"

    low_count = sum(1 for item in selected_candidates if item["motion_level"] == "low")
    medium_count = sum(1 for item in selected_candidates if item["motion_level"] == "medium")
    high_count = sum(1 for item in selected_candidates if item["motion_level"] == "high")

    output_path = run_dir / "04_motion_candidates.json"
    output_path.write_text(json.dumps(selected_candidates, indent=2), encoding="utf-8")

    print(f"[tender-demo] Motion threshold: {motion_threshold}")
    print(f"[tender-demo] Total motion scored frames: {len(motion_scores)}")
    print(f"[tender-demo] Total selected candidates: {len(selected_candidates)}")
    print(f"[tender-demo] Selected low motion candidates: {low_count}")
    print(f"[tender-demo] Selected medium motion candidates: {medium_count}")
    print(f"[tender-demo] Selected high motion candidates: {high_count}")
    print(f"[tender-demo] Motion candidates output path: {output_path}")
    return output_path, selected_candidates


def _build_clip_entry(
    clip_id: str,
    candidates: list[dict[str, object]],
    reason: str,
) -> dict[str, object]:
    start_candidate = candidates[0]
    end_candidate = candidates[-1]
    motion_scores = [float(candidate["motion_score_norm"]) for candidate in candidates]
    motion_levels_present = sorted({str(candidate["motion_level"]) for candidate in candidates})

    start_time = float(start_candidate["timestamp_seconds"])
    end_time = float(end_candidate["timestamp_seconds"])
    duration_seconds = round(end_time - start_time, 3)

    return {
        "clip_id": clip_id,
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": duration_seconds,
        "start_frame_idx": int(start_candidate["frame_idx"]),
        "end_frame_idx": int(end_candidate["frame_idx"]),
        "candidate_count": len(candidates),
        "max_motion_score_norm": round(max(motion_scores), 6),
        "avg_motion_score_norm": round(sum(motion_scores) / len(motion_scores), 6),
        "motion_levels_present": motion_levels_present,
        "reason": reason,
    }


def _group_raw_activity_segments(
    motion_candidates: list[dict[str, object]],
    max_gap_seconds: float,
) -> list[list[dict[str, object]]]:
    if not motion_candidates:
        return []

    sorted_candidates = sorted(
        motion_candidates,
        key=lambda item: float(item["timestamp_seconds"]),
    )
    raw_segments: list[list[dict[str, object]]] = [[sorted_candidates[0]]]

    for candidate in sorted_candidates[1:]:
        current_segment = raw_segments[-1]
        previous_candidate = current_segment[-1]
        gap_seconds = float(candidate["timestamp_seconds"]) - float(
            previous_candidate["timestamp_seconds"]
        )

        if gap_seconds <= max_gap_seconds:
            current_segment.append(candidate)
        else:
            raw_segments.append([candidate])

    return raw_segments


def _split_segment_into_clips(
    segment_candidates: list[dict[str, object]],
    max_clip_seconds: float,
    overlap_seconds: float,
) -> list[list[dict[str, object]]]:
    if overlap_seconds >= max_clip_seconds:
        raise ValueError(
            "TENDER_DEMO_CLIP_OVERLAP_SECONDS must be smaller than "
            "TENDER_DEMO_MAX_CLIP_SECONDS."
        )

    segment_start = float(segment_candidates[0]["timestamp_seconds"])
    segment_end = float(segment_candidates[-1]["timestamp_seconds"])
    split_start = segment_start
    split_step = max_clip_seconds - overlap_seconds
    split_clips: list[list[dict[str, object]]] = []
    last_window_end = -1.0

    while split_start <= segment_end:
        window_end = min(split_start + max_clip_seconds, segment_end)
        if window_end < split_start:
            break

        window_candidates = [
            candidate
            for candidate in segment_candidates
            if split_start <= float(candidate["timestamp_seconds"]) <= window_end
        ]
        if window_candidates:
            first_time = float(window_candidates[0]["timestamp_seconds"])
            last_time = float(window_candidates[-1]["timestamp_seconds"])
            if last_time > last_window_end or not split_clips:
                split_clips.append(window_candidates)
                last_window_end = last_time

        if window_end >= segment_end:
            break

        split_start += split_step

    return split_clips


def _create_candidate_clips(
    motion_candidates: list[dict[str, object]],
    run_dir: Path,
    max_gap_seconds: float,
    max_clip_seconds: float,
    overlap_seconds: float,
) -> tuple[Path, list[dict[str, object]]]:
    print("[tender-demo] Starting Step 5: grouping motion candidates into clips")
    raw_segments = _group_raw_activity_segments(
        motion_candidates=motion_candidates,
        max_gap_seconds=max_gap_seconds,
    )

    final_clips: list[dict[str, object]] = []
    long_segments_split = 0

    for segment_candidates in raw_segments:
        segment_start = float(segment_candidates[0]["timestamp_seconds"])
        segment_end = float(segment_candidates[-1]["timestamp_seconds"])
        segment_duration = segment_end - segment_start

        if segment_duration <= max_clip_seconds:
            clip_id = f"clip_{len(final_clips) + 1:06d}"
            final_clips.append(
                _build_clip_entry(
                    clip_id=clip_id,
                    candidates=segment_candidates,
                    reason="grouped_motion_segment",
                )
            )
            continue

        long_segments_split += 1
        split_clips = _split_segment_into_clips(
            segment_candidates=segment_candidates,
            max_clip_seconds=max_clip_seconds,
            overlap_seconds=overlap_seconds,
        )
        for split_candidates in split_clips:
            clip_id = f"clip_{len(final_clips) + 1:06d}"
            final_clips.append(
                _build_clip_entry(
                    clip_id=clip_id,
                    candidates=split_candidates,
                    reason="split_from_long_motion_segment",
                )
            )

    output_path = run_dir / "05_candidate_clips.json"
    output_path.write_text(json.dumps(final_clips, indent=2), encoding="utf-8")

    clip_durations = [float(clip["duration_seconds"]) for clip in final_clips]
    min_duration = round(min(clip_durations), 3) if clip_durations else 0.0
    max_duration = round(max(clip_durations), 3) if clip_durations else 0.0
    avg_duration = round(sum(clip_durations) / len(clip_durations), 3) if clip_durations else 0.0

    print(f"[tender-demo] Max gap seconds: {max_gap_seconds}")
    print(f"[tender-demo] Max clip seconds: {max_clip_seconds}")
    print(f"[tender-demo] Overlap seconds: {overlap_seconds}")
    print(f"[tender-demo] Total input candidates: {len(motion_candidates)}")
    print(f"[tender-demo] Total grouped raw segments: {len(raw_segments)}")
    print(f"[tender-demo] Number of long segments split: {long_segments_split}")
    print(f"[tender-demo] Final candidate clips count: {len(final_clips)}")
    print(f"[tender-demo] Min clip duration: {min_duration}")
    print(f"[tender-demo] Max clip duration: {max_duration}")
    print(f"[tender-demo] Average clip duration: {avg_duration}")
    print(f"[tender-demo] Candidate clips output path: {output_path}")
    return output_path, final_clips


def _clamp_time(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def _expand_candidate_clips(
    candidate_clips: list[dict[str, object]],
    video_info: dict[str, object],
    run_dir: Path,
    context_before_seconds: float,
    context_after_seconds: float,
    min_expanded_clip_seconds: float,
) -> tuple[Path, list[dict[str, object]]]:
    print("[tender-demo] Starting Step 6: expanding candidate clips with context")

    fps = float(video_info["fps"])
    total_frames = int(video_info["total_frames"])
    video_duration_seconds = float(video_info["duration_seconds"])
    max_frame_idx = max(total_frames - 1, 0)

    expanded_clips: list[dict[str, object]] = []
    normal_context_count = 0
    minimum_duration_count = 0

    for clip in candidate_clips:
        start_time = float(clip["start_time"])
        end_time = float(clip["end_time"])

        expanded_start_time = _clamp_time(start_time - context_before_seconds, 0.0, video_duration_seconds)
        expanded_end_time = _clamp_time(end_time + context_after_seconds, 0.0, video_duration_seconds)
        expansion_reason = "normal_context_expansion"

        expanded_duration = expanded_end_time - expanded_start_time
        if expanded_duration < min_expanded_clip_seconds:
            clip_center = (start_time + end_time) / 2.0
            half_target_duration = min_expanded_clip_seconds / 2.0
            expanded_start_time = clip_center - half_target_duration
            expanded_end_time = clip_center + half_target_duration

            if expanded_start_time < 0.0:
                expanded_end_time = min(video_duration_seconds, expanded_end_time - expanded_start_time)
                expanded_start_time = 0.0

            if expanded_end_time > video_duration_seconds:
                shift_left = expanded_end_time - video_duration_seconds
                expanded_start_time = max(0.0, expanded_start_time - shift_left)
                expanded_end_time = video_duration_seconds

            expanded_start_time = _clamp_time(expanded_start_time, 0.0, video_duration_seconds)
            expanded_end_time = _clamp_time(expanded_end_time, 0.0, video_duration_seconds)
            expansion_reason = "minimum_duration_expansion"

        expanded_duration = round(expanded_end_time - expanded_start_time, 3)
        expanded_start_frame_idx = int(round(expanded_start_time * fps))
        expanded_end_frame_idx = int(round(expanded_end_time * fps))
        expanded_start_frame_idx = max(0, expanded_start_frame_idx)
        expanded_end_frame_idx = min(max_frame_idx, expanded_end_frame_idx)

        expanded_clip = {
            **clip,
            "expanded_start_time": round(expanded_start_time, 3),
            "expanded_end_time": round(expanded_end_time, 3),
            "expanded_duration_seconds": expanded_duration,
            "expanded_start_frame_idx": expanded_start_frame_idx,
            "expanded_end_frame_idx": expanded_end_frame_idx,
            "context_before_seconds": context_before_seconds,
            "context_after_seconds": context_after_seconds,
            "expansion_reason": expansion_reason,
        }
        expanded_clips.append(expanded_clip)

        if expansion_reason == "normal_context_expansion":
            normal_context_count += 1
        else:
            minimum_duration_count += 1

    output_path = run_dir / "06_expanded_clips.json"
    output_path.write_text(json.dumps(expanded_clips, indent=2), encoding="utf-8")

    expanded_durations = [float(clip["expanded_duration_seconds"]) for clip in expanded_clips]
    min_duration = round(min(expanded_durations), 3) if expanded_durations else 0.0
    max_duration = round(max(expanded_durations), 3) if expanded_durations else 0.0
    avg_duration = (
        round(sum(expanded_durations) / len(expanded_durations), 3) if expanded_durations else 0.0
    )

    print(f"[tender-demo] Total input candidate clips: {len(candidate_clips)}")
    print(f"[tender-demo] Context before seconds: {context_before_seconds}")
    print(f"[tender-demo] Context after seconds: {context_after_seconds}")
    print(f"[tender-demo] Min expanded clip seconds: {min_expanded_clip_seconds}")
    print(f"[tender-demo] Total expanded clips: {len(expanded_clips)}")
    print(f"[tender-demo] Clips with normal context expansion: {normal_context_count}")
    print(f"[tender-demo] Clips with minimum duration expansion: {minimum_duration_count}")
    print(f"[tender-demo] Min expanded duration: {min_duration}")
    print(f"[tender-demo] Max expanded duration: {max_duration}")
    print(f"[tender-demo] Avg expanded duration: {avg_duration}")
    print(f"[tender-demo] Expanded clips output path: {output_path}")
    return output_path, expanded_clips


def _format_seconds_label(timestamp_seconds: float) -> str:
    return f"{timestamp_seconds:.2f}s"


def _read_video_frame_by_index(capture: cv2.VideoCapture, frame_idx: int) -> cv2.typing.MatLike:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = capture.read()
    if not success or frame is None:
        raise RuntimeError(f"Failed to read video frame at index {frame_idx}")
    return frame


def _render_labeled_panel(frame: cv2.typing.MatLike, label: str, timestamp_seconds: float) -> cv2.typing.MatLike:
    panel = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
    cv2.rectangle(panel, (0, 0), (640, 48), (0, 0, 0), thickness=-1)
    overlay_text = f"{label} {_format_seconds_label(timestamp_seconds)}"
    cv2.putText(
        panel,
        overlay_text,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def _create_temporal_strip_inputs(
    expanded_clips: list[dict[str, object]],
    video_path: Path,
    video_info: dict[str, object],
    run_dir: Path,
) -> tuple[Path, Path, list[dict[str, object]]]:
    print("[tender-demo] Starting Step 7: creating temporal strip images")

    output_dir = run_dir / "07_vlm_inputs"
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "07_vlm_inputs.json"

    fps = float(video_info["fps"])
    total_frames = int(video_info["total_frames"])
    video_duration_seconds = float(video_info["duration_seconds"])
    max_frame_idx = max(total_frames - 1, 0)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not open video for temporal strip generation: {video_path}")

    manifest_entries: list[dict[str, object]] = []
    failures = 0

    try:
        for index, clip in enumerate(expanded_clips, start=1):
            try:
                previous_time = _clamp_time(
                    float(clip["expanded_start_time"]),
                    0.0,
                    video_duration_seconds,
                )
                if "peak_timestamp_seconds" in clip and clip["peak_timestamp_seconds"] is not None:
                    current_time = _clamp_time(
                        float(clip["peak_timestamp_seconds"]),
                        0.0,
                        video_duration_seconds,
                    )
                else:
                    current_time = _clamp_time(
                        (float(clip["start_time"]) + float(clip["end_time"])) / 2.0,
                        0.0,
                        video_duration_seconds,
                    )
                next_time = _clamp_time(
                    float(clip["expanded_end_time"]),
                    0.0,
                    video_duration_seconds,
                )

                previous_frame_idx = min(max_frame_idx, max(0, int(round(previous_time * fps))))
                current_frame_idx = min(max_frame_idx, max(0, int(round(current_time * fps))))
                next_frame_idx = min(max_frame_idx, max(0, int(round(next_time * fps))))

                previous_frame = _read_video_frame_by_index(capture, previous_frame_idx)
                current_frame = _read_video_frame_by_index(capture, current_frame_idx)
                next_frame = _read_video_frame_by_index(capture, next_frame_idx)

                previous_panel = _render_labeled_panel(previous_frame, "PREVIOUS", previous_time)
                current_panel = _render_labeled_panel(current_frame, "CURRENT", current_time)
                next_panel = _render_labeled_panel(next_frame, "NEXT", next_time)
                strip_image = cv2.hconcat([previous_panel, current_panel, next_panel])

                vlm_input_id = f"vlm_input_{index:06d}"
                strip_path = output_dir / f"{vlm_input_id}.jpg"
                if not cv2.imwrite(str(strip_path), strip_image):
                    raise RuntimeError(f"Failed to write temporal strip image: {strip_path}")

                manifest_entries.append(
                    {
                        "vlm_input_id": vlm_input_id,
                        "clip_id": clip["clip_id"],
                        "strip_path": _to_repo_relative_path(strip_path),
                        "previous_time": round(previous_time, 3),
                        "current_time": round(current_time, 3),
                        "next_time": round(next_time, 3),
                        "previous_frame_idx": previous_frame_idx,
                        "current_frame_idx": current_frame_idx,
                        "next_frame_idx": next_frame_idx,
                        "source_start_time": float(clip["start_time"]),
                        "source_end_time": float(clip["end_time"]),
                        "expanded_start_time": float(clip["expanded_start_time"]),
                        "expanded_end_time": float(clip["expanded_end_time"]),
                        "clip_motion_score": round(float(clip.get("max_motion_score_norm", 0.0)), 6),
                        "reason": clip.get("reason", "unknown"),
                    }
                )
            except Exception as exc:
                failures += 1
                print(
                    f"[tender-demo] Warning: failed to create temporal strip for "
                    f"{clip.get('clip_id', 'unknown_clip')}: {exc}"
                )
    finally:
        capture.release()

    manifest_path.write_text(json.dumps(manifest_entries, indent=2), encoding="utf-8")

    print(f"[tender-demo] Total expanded clips received: {len(expanded_clips)}")
    print(f"[tender-demo] Total temporal strips created: {len(manifest_entries)}")
    print(f"[tender-demo] Total failures/skipped clips: {failures}")
    print(f"[tender-demo] Temporal strip output folder: {output_dir}")
    print(f"[tender-demo] Temporal strip manifest path: {manifest_path}")
    return output_dir, manifest_path, manifest_entries


def get_tender_demo_step8_prompt() -> str:
    return """Analyze this CCTV/security temporal strip image.

The image may contain 3 panels:
PREVIOUS | CURRENT | NEXT

Analyze the CURRENT panel as the main moment.
Use PREVIOUS and NEXT only as temporal context.

Return ONLY one valid JSON object.
No markdown.
No explanation.
No comments.
Do not wrap output in ```json.

Required JSON schema:

{
"scene_type": "street|entrance|parking_area|corridor|office|shop|warehouse|indoor|outdoor|unknown",
"caption": "one objective sentence",
"people_count": 0,
"objects": [
{
"id": "person_1",
"type": "person|vehicle|animal|object",
"subtype": "man|woman|child|car|truck|bus|motorcycle|bicycle|bag|backpack|box|phone|weapon|other|unknown",
"color": "brown|red|orange|yellow|green|blue|purple|pink|white|grey|black|unknown",
"condition": "standing|walking|running|sitting|lying|bending|moving|stationary|parked|unknown"
}
],
"activities": [],
"relationships": [],
"events": [
{
"event_type": "normal_activity|person_object_interaction|loitering|abandoned_object|object_removed|possible_theft|possible_robbery|weapon_visible|physical_altercation|collision|fall|medical_emergency|fire|smoke|crowd_formation",
"description": "objective visual evidence only",
"actors": [],
"severity": "low|medium|high|critical"
}
],
"keywords": []
}

Rules:

* Report only visible facts.
* Do not invent crimes, intentions, identities, or hidden actions.
* If no suspicious event is visible, use "events": [].
* Use "possible_theft" or "possible_robbery" only when clear visual evidence exists.
* Use "physical_altercation" only for visible fighting, grabbing, pushing, striking, or wrestling.
* Use "fall" only if a person is visibly falling or lying after a fall.
* Keep the JSON concise.
* All required top-level keys must always be present.
* Use stable object ids only, such as person_1, person_2, vehicle_1, bag_1.
* people_count must match the number of person objects in objects."""


def _clean_qwen_json_output(raw_output: str) -> str:
    cleaned_output = raw_output.strip()
    if cleaned_output.startswith("```"):
        first_newline = cleaned_output.find("\n")
        if first_newline != -1:
            cleaned_output = cleaned_output[first_newline + 1 :].strip()
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[:-3].strip()
    return cleaned_output


def run_qwen_on_vlm_inputs(run_dir: Path) -> list[dict[str, object]]:
    try:
        from tests.tender_demo_case.tender_demo_vlm_adapter import TenderDemoQwenVLM
    except ModuleNotFoundError:
        adapter_path = Path(__file__).resolve().parent / "tender_demo_vlm_adapter.py"
        spec = importlib.util.spec_from_file_location("tender_demo_vlm_adapter", adapter_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load tender demo VLM adapter from: {adapter_path}")
        adapter_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter_module)
        TenderDemoQwenVLM = adapter_module.TenderDemoQwenVLM

    print("[tender-demo] Starting Step 8: running isolated Qwen adapter on temporal strips")

    manifest_path = run_dir / "07_vlm_inputs.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing VLM input manifest: {manifest_path}")

    vlm_input_items = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(vlm_input_items, list):
        raise ValueError(f"Expected a list in VLM input manifest: {manifest_path}")

    repo_root = _get_repo_root()
    image_paths: list[Path] = []
    prompts: list[str] = []
    prompt = get_tender_demo_step8_prompt()

    for item in vlm_input_items:
        strip_path = item.get("strip_path")
        if not strip_path:
            raise ValueError("Each VLM input item must include strip_path.")
        image_paths.append(repo_root / str(strip_path))
        prompts.append(prompt)

    vlm = TenderDemoQwenVLM()
    model_health = vlm.health_check()
    print(f"[tender-demo] Total VLM inputs: {len(vlm_input_items)}")
    print(f"[tender-demo] Model health check: {model_health}")

    raw_outputs = vlm.generate_batch(image_paths=image_paths, prompts=prompts)
    if len(raw_outputs) != len(vlm_input_items):
        raise RuntimeError(
            "Qwen output count did not match VLM input count: "
            f"{len(raw_outputs)} vs {len(vlm_input_items)}"
        )

    results: list[dict[str, object]] = []
    parse_success_count = 0
    parse_failure_count = 0

    for item, raw_output in zip(vlm_input_items, raw_outputs):
        cleaned_output = _clean_qwen_json_output(raw_output)
        parsed_json: dict[str, object] | list[object] | None
        parse_success = False
        parse_error: str | None = None

        try:
            parsed_json = json.loads(cleaned_output)
            parse_success = True
            parse_success_count += 1
        except json.JSONDecodeError as exc:
            parsed_json = None
            parse_error = str(exc)
            parse_failure_count += 1

        result_item = {
            **item,
            "clip_id": item.get("clip_id"),
            "strip_path": item.get("strip_path"),
            "start_time": item.get("start_time", item.get("source_start_time")),
            "end_time": item.get("end_time", item.get("source_end_time")),
            "current_frame_idx": item.get("current_frame_idx"),
            "clip_score": item.get("clip_score", item.get("clip_motion_score")),
            "raw_qwen_output": raw_output,
            "parsed_json": parsed_json,
            "parse_success": parse_success,
            "parse_error": parse_error,
        }
        results.append(result_item)

    output_path = run_dir / "08_vlm_outputs.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"[tender-demo] Successful parses: {parse_success_count}")
    print(f"[tender-demo] Failed parses: {parse_failure_count}")
    print(f"[tender-demo] VLM outputs path: {output_path}")
    return results


def _load_step_09_create_final_summary():
    try:
        from tests.tender_demo_case.step_09_final_summary import create_final_summary
        return create_final_summary
    except ModuleNotFoundError:
        step_09_path = Path(__file__).resolve().parent / "step_09_final_summary.py"
        spec = importlib.util.spec_from_file_location("step_09_final_summary", step_09_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load Step 09 summary module from: {step_09_path}")
        step_09_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step_09_module)
        return step_09_module.create_final_summary


def _load_step_10_run_yolo_detection():
    try:
        from tests.tender_demo_case.step_10_yolo_detection import run_yolo_detection_on_selected_frames
        return run_yolo_detection_on_selected_frames
    except ModuleNotFoundError:
        step_10_path = Path(__file__).resolve().parent / "step_10_yolo_detection.py"
        spec = importlib.util.spec_from_file_location("step_10_yolo_detection", step_10_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load Step 10 YOLO module from: {step_10_path}")
        step_10_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step_10_module)
        return step_10_module.run_yolo_detection_on_selected_frames


def _load_step_11_run_yolo_object_scoring():
    try:
        from tests.tender_demo_case.step_11_yolo_object_scoring import run_yolo_object_scoring
        return run_yolo_object_scoring
    except ModuleNotFoundError:
        step_11_path = Path(__file__).resolve().parent / "step_11_yolo_object_scoring.py"
        spec = importlib.util.spec_from_file_location("step_11_yolo_object_scoring", step_11_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load Step 11 YOLO scoring module from: {step_11_path}")
        step_11_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step_11_module)
        return step_11_module.run_yolo_object_scoring


def _load_step_12_run_fused_clip_evidence():
    try:
        from tests.tender_demo_case.step_12_fused_clip_evidence import run_fused_clip_evidence
        return run_fused_clip_evidence
    except ModuleNotFoundError:
        step_12_path = Path(__file__).resolve().parent / "step_12_fused_clip_evidence.py"
        spec = importlib.util.spec_from_file_location("step_12_fused_clip_evidence", step_12_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load Step 12 fusion module from: {step_12_path}")
        step_12_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step_12_module)
        return step_12_module.run_fused_clip_evidence


def _load_step_13_rank_candidate_clips():
    try:
        from tests.tender_demo_case.step_13_rank_candidate_clips import rank_candidate_clips
        return rank_candidate_clips
    except ModuleNotFoundError:
        step_13_path = Path(__file__).resolve().parent / "step_13_rank_candidate_clips.py"
        spec = importlib.util.spec_from_file_location("step_13_rank_candidate_clips", step_13_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load Step 13 ranking module from: {step_13_path}")
        step_13_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step_13_module)
        return step_13_module.rank_candidate_clips


def _load_step_14_select_topk_clips():
    try:
        from tests.tender_demo_case.step_14_select_topk_clips import select_topk_clips_for_qwen
        return select_topk_clips_for_qwen
    except ModuleNotFoundError:
        step_14_path = Path(__file__).resolve().parent / "step_14_select_topk_clips.py"
        spec = importlib.util.spec_from_file_location("step_14_select_topk_clips", step_14_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load Step 14 selection module from: {step_14_path}")
        step_14_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step_14_module)
        return step_14_module.select_topk_clips_for_qwen


def _load_step_15_create_topk_vlm_inputs():
    try:
        from tests.tender_demo_case.step_15_create_topk_vlm_inputs import create_topk_vlm_inputs
        return create_topk_vlm_inputs
    except ModuleNotFoundError:
        step_15_path = Path(__file__).resolve().parent / "step_15_create_topk_vlm_inputs.py"
        spec = importlib.util.spec_from_file_location("step_15_create_topk_vlm_inputs", step_15_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load Step 15 Top-K VLM input module from: {step_15_path}")
        step_15_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step_15_module)
        return step_15_module.create_topk_vlm_inputs


def _load_step_16_run_topk_qwen():
    try:
        from tests.tender_demo_case.step_16_run_topk_qwen import run_qwen_on_topk_vlm_inputs
        return run_qwen_on_topk_vlm_inputs
    except ModuleNotFoundError:
        step_16_path = Path(__file__).resolve().parent / "step_16_run_topk_qwen.py"
        spec = importlib.util.spec_from_file_location("step_16_run_topk_qwen", step_16_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load Step 16 Top-K Qwen module from: {step_16_path}")
        step_16_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step_16_module)
        return step_16_module.run_qwen_on_topk_vlm_inputs


def _load_step_17_create_topk_final_summary():
    try:
        from tests.tender_demo_case.step_17_topk_final_summary import create_topk_final_summary
        return create_topk_final_summary
    except ModuleNotFoundError:
        step_17_path = Path(__file__).resolve().parent / "step_17_topk_final_summary.py"
        spec = importlib.util.spec_from_file_location("step_17_topk_final_summary", step_17_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load Step 17 Top-K final summary module from: {step_17_path}")
        step_17_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step_17_module)
        return step_17_module.create_topk_final_summary


def _load_step_18_export_event_clips():
    try:
        from tests.tender_demo_case.step_18_export_event_clips import export_event_clips
        return export_event_clips
    except ModuleNotFoundError:
        step_18_path = Path(__file__).resolve().parent / "step_18_export_event_clips.py"
        spec = importlib.util.spec_from_file_location("step_18_export_event_clips", step_18_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load Step 18 event clip export module from: {step_18_path}")
        step_18_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step_18_module)
        return step_18_module.export_event_clips


def _load_step_19_create_demo_report():
    try:
        from tests.tender_demo_case.step_19_create_demo_report import create_demo_report_html
        return create_demo_report_html
    except ModuleNotFoundError:
        step_19_path = Path(__file__).resolve().parent / "step_19_create_demo_report.py"
        spec = importlib.util.spec_from_file_location("step_19_create_demo_report", step_19_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load Step 19 demo report module from: {step_19_path}")
        step_19_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step_19_module)
        return step_19_module.create_demo_report_html


def main() -> None:
    print("[tender-demo] Starting tender demo pipeline")
    video_path = _read_video_path()
    run_dir = _create_debug_run_dir(video_path)
    video_info = _extract_video_info(video_path)
    _write_video_info(run_dir, video_info)
    print("[tender-demo] Step 1 complete: video info captured")

    sample_every_seconds = _read_sample_every_seconds()
    _, _, sampled_frames = _sample_base_frames(
        video_path=video_path,
        run_dir=run_dir,
        fps=float(video_info["fps"]),
        total_frames=int(video_info["total_frames"]),
        sample_every_seconds=sample_every_seconds,
    )
    print("[tender-demo] Step 2 complete: base frame sampling finished")

    _, motion_scores = _score_motion_on_sampled_frames(
        sampled_frames=sampled_frames,
        run_dir=run_dir,
    )
    print("[tender-demo] Step 3 complete: motion scoring finished")

    motion_threshold = _read_motion_threshold()
    _, motion_candidates = _select_motion_candidates(
        motion_scores=motion_scores,
        run_dir=run_dir,
        motion_threshold=motion_threshold,
    )
    print("[tender-demo] Step 4 complete: motion candidate selection finished")

    max_gap_seconds = _read_positive_float_env(
        env_name=ENV_MAX_GAP_SECONDS,
        default_value=DEFAULT_MAX_GAP_SECONDS,
        label="max gap seconds",
    )
    max_clip_seconds = _read_positive_float_env(
        env_name=ENV_MAX_CLIP_SECONDS,
        default_value=DEFAULT_MAX_CLIP_SECONDS,
        label="max clip seconds",
    )
    overlap_seconds = _read_positive_float_env(
        env_name=ENV_CLIP_OVERLAP_SECONDS,
        default_value=DEFAULT_CLIP_OVERLAP_SECONDS,
        label="clip overlap seconds",
    )
    _, candidate_clips = _create_candidate_clips(
        motion_candidates=motion_candidates,
        run_dir=run_dir,
        max_gap_seconds=max_gap_seconds,
        max_clip_seconds=max_clip_seconds,
        overlap_seconds=overlap_seconds,
    )
    print("[tender-demo] Step 5 complete: candidate clip grouping finished")

    context_before_seconds = _read_positive_float_env(
        env_name=ENV_CONTEXT_BEFORE_SECONDS,
        default_value=DEFAULT_CONTEXT_BEFORE_SECONDS,
        label="context before seconds",
    )
    context_after_seconds = _read_positive_float_env(
        env_name=ENV_CONTEXT_AFTER_SECONDS,
        default_value=DEFAULT_CONTEXT_AFTER_SECONDS,
        label="context after seconds",
    )
    min_expanded_clip_seconds = _read_positive_float_env(
        env_name=ENV_MIN_EXPANDED_CLIP_SECONDS,
        default_value=DEFAULT_MIN_EXPANDED_CLIP_SECONDS,
        label="min expanded clip seconds",
    )
    _, expanded_clips = _expand_candidate_clips(
        candidate_clips=candidate_clips,
        video_info=video_info,
        run_dir=run_dir,
        context_before_seconds=context_before_seconds,
        context_after_seconds=context_after_seconds,
        min_expanded_clip_seconds=min_expanded_clip_seconds,
    )
    print("[tender-demo] Step 6 complete: context expansion finished")

    _create_temporal_strip_inputs(
        expanded_clips=expanded_clips,
        video_path=video_path,
        video_info=video_info,
        run_dir=run_dir,
    )
    print("[tender-demo] Step 7 complete: temporal strip generation finished")

    vlm_outputs = run_qwen_on_vlm_inputs(run_dir)
    print(f"[tender-demo] Step 8 complete: VLM outputs captured ({len(vlm_outputs)} items)")

    create_final_summary = _load_step_09_create_final_summary()
    final_summary = create_final_summary(run_dir)
    timeline_count = len(final_summary.get("event_timeline", []))
    print(f"[tender-demo] Step 9 complete: final summary created ({timeline_count} timeline items)")

    run_yolo_detection_on_selected_frames = _load_step_10_run_yolo_detection()
    yolo_detections = run_yolo_detection_on_selected_frames(run_dir)
    print(f"[tender-demo] Step 10 complete: YOLO detections captured ({len(yolo_detections)} items)")

    run_yolo_object_scoring = _load_step_11_run_yolo_object_scoring()
    yolo_object_report = run_yolo_object_scoring(run_dir)
    scored_count = len(yolo_object_report.get("scored_items", []))
    print(f"[tender-demo] Step 11 complete: YOLO object scoring captured ({scored_count} items)")

    rank_candidate_clips = _load_step_13_rank_candidate_clips()
    ranked_clip_report = rank_candidate_clips(run_dir)
    ranked_count = len(ranked_clip_report.get("ranked_clips", []))
    print(f"[tender-demo] Step 13 complete: ranked candidate clips captured ({ranked_count} items)")

    select_topk_clips_for_qwen = _load_step_14_select_topk_clips()
    topk_selection_report = select_topk_clips_for_qwen(run_dir)
    selected_count = len(topk_selection_report.get("selected_clips", []))
    print(f"[tender-demo] Step 14 complete: selected Top-K clips captured ({selected_count} items)")

    create_topk_vlm_inputs = _load_step_15_create_topk_vlm_inputs()
    topk_vlm_inputs_manifest = create_topk_vlm_inputs(run_dir)
    strip_count = len(topk_vlm_inputs_manifest.get("items", []))
    print(f"[tender-demo] Step 15 complete: Top-K VLM inputs captured ({strip_count} items)")

    run_qwen_on_topk_vlm_inputs = _load_step_16_run_topk_qwen()
    topk_vlm_outputs = run_qwen_on_topk_vlm_inputs(run_dir)
    print(f"[tender-demo] Step 16 complete: Top-K VLM outputs captured ({len(topk_vlm_outputs)} items)")

    create_topk_final_summary = _load_step_17_create_topk_final_summary()
    topk_final_summary = create_topk_final_summary(run_dir)
    priority_count = len(topk_final_summary.get("priority_suspicious_events", []))
    print(f"[tender-demo] Step 17 complete: Top-K final summary created ({priority_count} priority events)")

    export_event_clips = _load_step_18_export_event_clips()
    exported_clips_manifest = export_event_clips(run_dir)
    exported_count = int(exported_clips_manifest.get("total_clips_exported", 0))
    print(f"[tender-demo] Step 18 complete: exported review clips created ({exported_count} files)")

    create_demo_report_html = _load_step_19_create_demo_report()
    demo_report = create_demo_report_html(run_dir)
    print(f"[tender-demo] Step 19 complete: local demo report created ({demo_report.get('html_report_path')})")

    run_fused_clip_evidence = _load_step_12_run_fused_clip_evidence()
    fused_evidence_report = run_fused_clip_evidence(run_dir)
    fused_count = len(fused_evidence_report.get("fused_items", []))
    print(f"[tender-demo] Step 12 complete: fused clip evidence captured ({fused_count} items)")
    print(f"[tender-demo] Debug run directory: {run_dir}")


if __name__ == "__main__":
    main()
