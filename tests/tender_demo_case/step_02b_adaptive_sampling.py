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


def _load_required_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required adaptive sampling input file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in JSON file: {path}")
    return payload


def _read_env_bool(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() == "true"


def _read_env_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _mode_defaults(mode: str) -> dict[str, Any]:
    normalized = str(mode or "").strip().lower()
    if normalized == "sensitive incident review":
        return {
            "enabled": True,
            "base_interval_seconds": 1.0,
            "motion_threshold": 0.08,
            "hist_threshold": 0.12,
            "similarity_threshold": 0.92,
            "max_frame_gap_seconds": 4.0,
            "target_window_seconds": 3.0,
        }
    if normalized in {"high accuracy", "high accuracy review"}:
        return {
            "enabled": True,
            "base_interval_seconds": 0.5,
            "motion_threshold": 0.08,
            "hist_threshold": 0.12,
            "similarity_threshold": 0.92,
            "max_frame_gap_seconds": 3.0,
            "target_window_seconds": 3.0,
        }
    return {
        "enabled": False,
        "base_interval_seconds": 1.0,
        "motion_threshold": 0.08,
        "hist_threshold": 0.12,
        "similarity_threshold": 0.92,
        "max_frame_gap_seconds": 4.0,
        "target_window_seconds": 3.0,
    }


def _read_critical_timestamps() -> list[float]:
    raw_value = os.environ.get("TENDER_DEMO_CRITICAL_TIMESTAMPS", "").strip()
    if not raw_value:
        return []
    timestamps: list[float] = []
    for part in raw_value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            timestamps.append(parse_timestamp_to_seconds(part))
        except ValueError:
            print(f"[tender-demo] Warning: invalid critical timestamp ignored: {part!r}")
    return timestamps


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
    target_window_seconds: float,
) -> list[dict[str, Any]]:
    coverage: list[dict[str, Any]] = []
    for target_seconds in target_timestamps:
        nearest_item = None
        nearest_distance = None
        for item in retained_items:
            timestamp_seconds = float(item.get("timestamp_seconds", 0.0) or 0.0)
            distance = abs(timestamp_seconds - target_seconds)
            if nearest_distance is None or distance < nearest_distance:
                nearest_item = item
                nearest_distance = distance
        coverage.append(
            {
                "timestamp": format_seconds_label(target_seconds),
                "covered": bool(nearest_distance is not None and nearest_distance <= target_window_seconds),
                "nearest_retained_time": float(nearest_item.get("timestamp_seconds", 0.0) or 0.0) if nearest_item else None,
                "distance_seconds": round(float(nearest_distance), 3) if nearest_distance is not None else None,
                "nearest_frame_path": str(nearest_item.get("frame_path", "")) if nearest_item else None,
            }
        )
    return coverage


def run_adaptive_sampling(run_dir: Path) -> dict[str, Any]:
    print("[tender-demo] Starting Step 02B: adaptive sampling")
    video_info = _load_required_json(run_dir / "01_video_info.json")
    video_path = Path(str(video_info.get("video_path", "")).strip())
    if not video_path.exists():
        raise FileNotFoundError(f"Video path does not exist for adaptive sampling: {video_path}")

    mode = os.environ.get("TENDER_DEMO_ANALYSIS_SENSITIVITY_MODE", "Balanced")
    defaults = _mode_defaults(mode)
    enabled = _read_env_bool("TENDER_DEMO_ENABLE_ADAPTIVE_SAMPLING", bool(defaults["enabled"]))
    base_interval_seconds = _read_env_float("TENDER_DEMO_ADAPTIVE_BASE_INTERVAL_SECONDS", float(defaults["base_interval_seconds"]))
    motion_threshold = _read_env_float("TENDER_DEMO_ADAPTIVE_MOTION_THRESHOLD", float(defaults["motion_threshold"]))
    hist_threshold = _read_env_float("TENDER_DEMO_ADAPTIVE_HIST_THRESHOLD", float(defaults["hist_threshold"]))
    similarity_threshold = _read_env_float("TENDER_DEMO_ADAPTIVE_SIMILARITY_THRESHOLD", float(defaults["similarity_threshold"]))
    max_frame_gap_seconds = _read_env_float("TENDER_DEMO_ADAPTIVE_MAX_FRAME_GAP_SECONDS", float(defaults["max_frame_gap_seconds"]))
    target_window_seconds = _read_env_float("TENDER_DEMO_ADAPTIVE_TARGET_WINDOW_SECONDS", float(defaults["target_window_seconds"]))
    critical_timestamps = _read_critical_timestamps()

    retained_dir = run_dir / "02b_adaptive_frames"
    retained_dir.mkdir(parents=True, exist_ok=True)
    retained_manifest_path = run_dir / "02b_adaptive_frames.json"
    report_path = run_dir / "02b_adaptive_sampling_report.json"

    if not enabled:
        manifest = {
            "enabled": False,
            "video_path": str(video_path),
            "retained_frames": 0,
            "items": [],
        }
        report = {
            "enabled": False,
            "video_path": str(video_path),
            "video_duration": float(video_info.get("duration_seconds", 0.0) or 0.0),
            "base_interval_seconds": base_interval_seconds,
            "total_base_candidates": 0,
            "retained_frames": 0,
            "dropped_frames": 0,
            "target_timestamps": [format_seconds_label(value) for value in critical_timestamps],
            "target_coverage": [],
            "keep_reason_counts": {},
            "settings": {
                "motion_threshold": motion_threshold,
                "hist_threshold": hist_threshold,
                "similarity_threshold": similarity_threshold,
                "max_frame_gap_seconds": max_frame_gap_seconds,
                "target_window_seconds": target_window_seconds,
            },
        }
        retained_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("[tender-demo] Adaptive sampling disabled for this mode/run.")
        return {
            "enabled": False,
            "retained_manifest_path": retained_manifest_path,
            "report_path": report_path,
            "retained_items": [],
            "report": report,
        }

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not open video for adaptive sampling: {video_path}")

    retained_items: list[dict[str, Any]] = []
    keep_reason_counts: dict[str, int] = {}
    total_base_candidates = 0
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0:
            fps = float(video_info.get("fps", 30.0) or 30.0)
        duration_seconds = total_frames / fps if fps > 0 else float(video_info.get("duration_seconds", 0.0) or 0.0)
        sample_every_frames = max(1, int(round(base_interval_seconds * fps)))
        previous_gray: np.ndarray | None = None
        previous_kept_timestamp: float | None = None

        for frame_idx in range(0, total_frames, sample_every_frames):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = capture.read()
            if not success or frame is None:
                continue

            total_base_candidates += 1
            timestamp_seconds = frame_idx / fps if fps > 0 else 0.0
            resized = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
            current_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
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

            for target_seconds in critical_timestamps:
                if abs(timestamp_seconds - target_seconds) <= target_window_seconds:
                    keep_reasons.append("target_timestamp_window")
                    break

            if keep_reasons:
                output_path = retained_dir / f"adaptive_frame_{frame_idx:06d}.jpg"
                if not cv2.imwrite(str(output_path), frame):
                    raise RuntimeError(f"Failed to write adaptive frame: {output_path}")
                previous_kept_timestamp = timestamp_seconds
                for reason in keep_reasons:
                    keep_reason_counts[reason] = keep_reason_counts.get(reason, 0) + 1
                retained_items.append(
                    {
                        "frame_id": f"adaptive_f{len(retained_items) + 1:06d}",
                        "frame_idx": frame_idx,
                        "timestamp_seconds": round(timestamp_seconds, 3),
                        "frame_path": str(output_path),
                        "keep": True,
                        "keep_reasons": keep_reasons,
                        "motion_score": round(motion_score, 6),
                        "histogram_diff": round(histogram_diff, 6),
                        "similarity_score": round(similarity_score, 6),
                        "source": "adaptive_sampling",
                    }
                )

            previous_gray = current_gray
    finally:
        capture.release()

    manifest = {
        "enabled": True,
        "video_path": str(video_path),
        "retained_frames": len(retained_items),
        "items": retained_items,
    }
    retained_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = {
        "enabled": True,
        "video_path": str(video_path),
        "video_duration": round(float(video_info.get("duration_seconds", 0.0) or 0.0), 3),
        "base_interval_seconds": base_interval_seconds,
        "total_base_candidates": total_base_candidates,
        "retained_frames": len(retained_items),
        "dropped_frames": max(0, total_base_candidates - len(retained_items)),
        "target_timestamps": [format_seconds_label(value) for value in critical_timestamps],
        "target_coverage": _target_coverage(retained_items, critical_timestamps, target_window_seconds),
        "keep_reason_counts": keep_reason_counts,
        "settings": {
            "mode": mode,
            "motion_threshold": motion_threshold,
            "hist_threshold": hist_threshold,
            "similarity_threshold": similarity_threshold,
            "max_frame_gap_seconds": max_frame_gap_seconds,
            "target_window_seconds": target_window_seconds,
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[tender-demo] Adaptive retained frames: {len(retained_items)}")
    print(f"[tender-demo] Adaptive manifest path: {retained_manifest_path}")
    print(f"[tender-demo] Adaptive report path: {report_path}")
    return {
        "enabled": True,
        "retained_manifest_path": retained_manifest_path,
        "report_path": report_path,
        "retained_items": retained_items,
        "report": report,
    }

