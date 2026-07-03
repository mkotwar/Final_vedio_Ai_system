from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_timestamp_to_seconds(value: str) -> float:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Timestamp is empty.")
    parts = text.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        minutes, seconds = parts
        return (int(minutes) * 60) + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return (int(hours) * 3600) + (int(minutes) * 60) + float(seconds)
    raise ValueError(f"Unsupported timestamp format: {value!r}")


def format_seconds_label(seconds: float) -> str:
    total_seconds = max(0.0, float(seconds))
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    remaining_seconds = total_seconds - (hours * 3600) - (minutes * 60)
    if float(remaining_seconds).is_integer():
        return f"{hours:02d}:{minutes:02d}:{int(remaining_seconds):02d}"
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:04.1f}"


def _safe_float_env(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _grayscale_motion_score(previous_gray: np.ndarray, current_gray: np.ndarray) -> float:
    diff = cv2.absdiff(previous_gray, current_gray)
    _, thresholded = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    return float(np.count_nonzero(thresholded) / thresholded.size)


def _histogram_diff(previous_gray: np.ndarray, current_gray: np.ndarray) -> float:
    hist1 = cv2.calcHist([previous_gray], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([current_gray], [0], None, [256], [0, 256])
    hist1 = hist1 / (hist1.sum() + 1e-7)
    hist2 = hist2 / (hist2.sum() + 1e-7)
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return max(0.0, 1.0 - float(correlation))


def _similarity_score(previous_gray: np.ndarray, current_gray: np.ndarray) -> float:
    previous = previous_gray.astype(np.float32)
    current = current_gray.astype(np.float32)

    c1 = 6.5025
    c2 = 58.5225

    mu1 = cv2.GaussianBlur(previous, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(current, (11, 11), 1.5)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(previous * previous, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(current * current, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(previous * current, (11, 11), 1.5) - mu1_mu2

    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / (denominator + 1e-7)
    return float(np.clip(np.mean(ssim_map), 0.0, 1.0))


def _target_coverage(
    retained_items: list[dict[str, Any]],
    target_timestamps: list[float],
    target_timestamp_labels: list[str],
    target_window_seconds: float,
) -> list[dict[str, Any]]:
    coverage: list[dict[str, Any]] = []
    for index, target_seconds in enumerate(target_timestamps):
        nearest = None
        nearest_distance = None
        for item in retained_items:
            candidate_time = float(item.get("timestamp_seconds", 0.0) or 0.0)
            distance = abs(candidate_time - target_seconds)
            if nearest_distance is None or distance < nearest_distance:
                nearest = item
                nearest_distance = distance
        coverage.append(
            {
                "timestamp": format_seconds_label(target_seconds),
                "meaning": target_timestamp_labels[index] if index < len(target_timestamp_labels) else "",
                "covered": bool(nearest is not None and nearest_distance is not None and nearest_distance <= target_window_seconds),
                "nearest_retained_time": float(nearest.get("timestamp_seconds", 0.0) or 0.0) if nearest else None,
                "distance_seconds": round(float(nearest_distance), 3) if nearest_distance is not None else None,
                "nearest_frame_path": str(nearest.get("frame_path", "")) if nearest else None,
            }
        )
    return coverage


def run_adaptive_sampler(
    video_path: Path,
    output_dir: Path,
    target_timestamps: list[float],
    target_timestamp_labels: list[str],
) -> dict[str, Any]:
    base_interval_seconds = _safe_float_env("ADAPTIVE_COMPARE_BASE_INTERVAL_SECONDS", 1.0)
    motion_threshold = _safe_float_env("ADAPTIVE_COMPARE_MOTION_THRESHOLD", 0.08)
    hist_threshold = _safe_float_env("ADAPTIVE_COMPARE_HIST_THRESHOLD", 0.12)
    similarity_threshold = _safe_float_env("ADAPTIVE_COMPARE_SIMILARITY_THRESHOLD", 0.92)
    max_frame_gap_seconds = _safe_float_env("ADAPTIVE_COMPARE_MAX_FRAME_GAP_SECONDS", 4.0)
    target_window_seconds = _safe_float_env("ADAPTIVE_COMPARE_TARGET_WINDOW_SECONDS", 3.0)

    if not video_path.exists():
        raise FileNotFoundError(f"Video path does not exist: {video_path}")

    retained_dir = output_dir / "adaptive_retained_frames"
    retained_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0:
            fps = 30.0
        duration_seconds = total_frames / fps if fps > 0 else 0.0
        sample_every_frames = max(1, int(round(base_interval_seconds * fps)))

        previous_gray: np.ndarray | None = None
        previous_kept_timestamp: float | None = None
        retained_items: list[dict[str, Any]] = []
        keep_reason_counts: dict[str, int] = {}
        total_base_candidates = 0

        for frame_idx in range(0, total_frames, sample_every_frames):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = capture.read()
            if not success or frame is None:
                continue

            total_base_candidates += 1
            timestamp_seconds = frame_idx / fps
            frame_small = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
            current_gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

            motion_score = 0.0
            histogram_diff = 0.0
            similarity_score = 1.0
            keep_reasons: list[str] = []

            if previous_gray is None:
                keep_reasons.append("first_frame")
            else:
                motion_score = _grayscale_motion_score(previous_gray, current_gray)
                histogram_diff = _histogram_diff(previous_gray, current_gray)
                similarity_score = _similarity_score(previous_gray, current_gray)
                if motion_score >= motion_threshold:
                    keep_reasons.append("motion_change")
                if histogram_diff >= hist_threshold:
                    keep_reasons.append("histogram_change")
                if similarity_score <= similarity_threshold:
                    keep_reasons.append("similarity_drop")

            if previous_kept_timestamp is None or (timestamp_seconds - previous_kept_timestamp) >= max_frame_gap_seconds:
                keep_reasons.append("coverage_safeguard")

            for target_seconds in target_timestamps:
                if abs(timestamp_seconds - target_seconds) <= target_window_seconds:
                    keep_reasons.append("target_timestamp_window")
                    break

            keep = bool(keep_reasons)
            frame_path = retained_dir / f"adaptive_frame_{frame_idx:06d}.jpg"
            if keep:
                if not cv2.imwrite(str(frame_path), frame):
                    raise RuntimeError(f"Failed to write adaptive retained frame: {frame_path}")
                previous_kept_timestamp = timestamp_seconds
                for reason in keep_reasons:
                    keep_reason_counts[reason] = keep_reason_counts.get(reason, 0) + 1
                retained_items.append(
                    {
                        "frame_id": f"adaptive_f{len(retained_items) + 1:06d}",
                        "frame_idx": frame_idx,
                        "timestamp_seconds": round(timestamp_seconds, 3),
                        "frame_path": str(frame_path),
                        "keep": True,
                        "keep_reasons": keep_reasons,
                        "motion_score": round(motion_score, 6),
                        "histogram_diff": round(histogram_diff, 6),
                        "similarity_score": round(similarity_score, 6),
                    }
                )

            previous_gray = current_gray
    finally:
        capture.release()

    retained_manifest = {
        "video_path": str(video_path),
        "retained_frames": len(retained_items),
        "items": retained_items,
    }
    retained_manifest_path = output_dir / "adaptive_retained_frames.json"
    retained_manifest_path.write_text(json.dumps(retained_manifest, indent=2), encoding="utf-8")

    report = {
        "video_path": str(video_path),
        "video_duration": round(duration_seconds, 3),
        "base_interval_seconds": base_interval_seconds,
        "total_base_candidates": total_base_candidates,
        "retained_frames": len(retained_items),
        "dropped_frames": max(0, total_base_candidates - len(retained_items)),
        "target_timestamps": [format_seconds_label(value) for value in target_timestamps],
        "target_coverage": _target_coverage(
            retained_items=retained_items,
            target_timestamps=target_timestamps,
            target_timestamp_labels=target_timestamp_labels,
            target_window_seconds=target_window_seconds,
        ),
        "keep_reason_counts": keep_reason_counts,
        "settings": {
            "motion_threshold": motion_threshold,
            "hist_threshold": hist_threshold,
            "similarity_threshold": similarity_threshold,
            "max_frame_gap_seconds": max_frame_gap_seconds,
            "target_window_seconds": target_window_seconds,
        },
    }
    report_path = output_dir / "adaptive_sampling_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return {
        "retained_dir": retained_dir,
        "retained_manifest_path": retained_manifest_path,
        "report_path": report_path,
        "report": report,
        "retained_items": retained_items,
    }
