from __future__ import annotations

import os
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import (
    DEFAULT_ADAPTIVE_ENABLED,
    DEFAULT_ADAPTIVE_HEARTBEAT_SECONDS,
    DEFAULT_ADAPTIVE_HISTOGRAM_CHANGE_THRESHOLD,
    DEFAULT_ADAPTIVE_MAX_BRIGHTNESS,
    DEFAULT_ADAPTIVE_MIN_BLOB_AREA_RATIO,
    DEFAULT_ADAPTIVE_MIN_BLUR_SCORE,
    DEFAULT_ADAPTIVE_MIN_BRIGHTNESS,
    DEFAULT_ADAPTIVE_MIN_PIXEL_VARIANCE,
    DEFAULT_ADAPTIVE_MIN_SELECTED_GAP_SECONDS,
    DEFAULT_ADAPTIVE_MOTION_PIXELS_RATIO_THRESHOLD,
    DEFAULT_ADAPTIVE_MOTION_SCORE_THRESHOLD,
    DEFAULT_ADAPTIVE_PREVIEW_LIMIT,
    DEFAULT_ADAPTIVE_ROAD_ROI_RECT,
    DEFAULT_ADAPTIVE_ROI_MODE,
    ENV_ADAPTIVE_ENABLED,
    ENV_ADAPTIVE_HEARTBEAT_SECONDS,
    ENV_ADAPTIVE_HISTOGRAM_CHANGE_THRESHOLD,
    ENV_ADAPTIVE_MAX_BRIGHTNESS,
    ENV_ADAPTIVE_MIN_BLOB_AREA_RATIO,
    ENV_ADAPTIVE_MIN_BLUR_SCORE,
    ENV_ADAPTIVE_MIN_BRIGHTNESS,
    ENV_ADAPTIVE_MIN_PIXEL_VARIANCE,
    ENV_ADAPTIVE_MIN_SELECTED_GAP_SECONDS,
    ENV_ADAPTIVE_MOTION_PIXELS_RATIO_THRESHOLD,
    ENV_ADAPTIVE_MOTION_SCORE_THRESHOLD,
    ENV_ADAPTIVE_PREVIEW_LIMIT,
    ENV_ADAPTIVE_ROAD_ROI_RECT,
    ENV_ADAPTIVE_ROI_MODE,
)
from stage_checks import format_seconds_text, read_json, write_json


SUPPORTED_ADAPTIVE_ROI_MODES = {"full_frame", "default_road", "custom_rect"}


@dataclass(frozen=True)
class AdaptiveSamplingConfig:
    """Configuration for Step 02A motion-based adaptive sampling."""

    enabled: bool
    roi_mode: str
    road_roi_rect_normalized: tuple[float, float, float, float]
    motion_score_threshold: float
    motion_pixels_ratio_threshold: float
    histogram_change_threshold: float
    min_blob_area_ratio: float
    heartbeat_seconds: float
    min_selected_gap_seconds: float
    preview_limit: int
    min_brightness: float
    max_brightness: float
    min_pixel_variance: float
    min_blur_score: float


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a floating point value into a fixed range."""

    return max(minimum, min(maximum, value))


def format_timestamp(total_seconds: float) -> str:
    """Return a readable mm:ss or hh:mm:ss timestamp string."""

    return format_seconds_text(total_seconds)


def _read_bool(env_name: str, default_value: bool) -> bool:
    """Read a permissive boolean value from the environment."""

    raw_value = os.environ.get(env_name)
    if raw_value is None or raw_value.strip() == "":
        return default_value
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Environment variable {env_name} must be a boolean-like value. Received: {raw_value!r}")


def _read_positive_float(env_name: str, default_value: float) -> float:
    """Read a positive float from the environment."""

    raw_value = os.environ.get(env_name, str(default_value)).strip()
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {env_name} must be a valid number. Received: {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"Environment variable {env_name} must be greater than 0. Received: {value}")
    return value


def _read_positive_int(env_name: str, default_value: int) -> int:
    """Read a positive integer from the environment."""

    raw_value = os.environ.get(env_name, str(default_value)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {env_name} must be a valid integer. Received: {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"Environment variable {env_name} must be greater than 0. Received: {value}")
    return value


def _parse_normalized_rect(raw_value: str, env_name: str) -> tuple[float, float, float, float]:
    """Parse normalized x1,y1,x2,y2 ROI coordinates."""

    parts = [item.strip() for item in raw_value.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Environment variable {env_name} must have four comma-separated numbers. Received: {raw_value!r}")
    try:
        x1, y1, x2, y2 = [float(item) for item in parts]
    except ValueError as exc:
        raise ValueError(f"Environment variable {env_name} must contain valid numbers. Received: {raw_value!r}") from exc
    values = (x1, y1, x2, y2)
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError(f"Environment variable {env_name} values must stay between 0 and 1. Received: {raw_value!r}")
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Environment variable {env_name} must satisfy x2>x1 and y2>y1. Received: {raw_value!r}")
    return values


def read_adaptive_config() -> AdaptiveSamplingConfig:
    """Read Step 02A adaptive-sampling configuration from environment variables."""

    enabled = _read_bool(ENV_ADAPTIVE_ENABLED, DEFAULT_ADAPTIVE_ENABLED)
    roi_mode = os.environ.get(ENV_ADAPTIVE_ROI_MODE, DEFAULT_ADAPTIVE_ROI_MODE).strip().lower() or DEFAULT_ADAPTIVE_ROI_MODE
    if roi_mode not in SUPPORTED_ADAPTIVE_ROI_MODES:
        raise ValueError(
            f"Environment variable {ENV_ADAPTIVE_ROI_MODE} must be one of {sorted(SUPPORTED_ADAPTIVE_ROI_MODES)}. "
            f"Received: {roi_mode!r}"
        )

    if roi_mode == "full_frame":
        road_roi_rect = (0.0, 0.0, 1.0, 1.0)
    elif roi_mode == "custom_rect":
        raw_rect = os.environ.get(ENV_ADAPTIVE_ROAD_ROI_RECT, "").strip()
        if not raw_rect:
            raise ValueError(
                f"Environment variable {ENV_ADAPTIVE_ROAD_ROI_RECT} is required when "
                f"{ENV_ADAPTIVE_ROI_MODE} is set to custom_rect."
            )
        road_roi_rect = _parse_normalized_rect(raw_rect, ENV_ADAPTIVE_ROAD_ROI_RECT)
    else:
        road_roi_rect = DEFAULT_ADAPTIVE_ROAD_ROI_RECT

    return AdaptiveSamplingConfig(
        enabled=enabled,
        roi_mode=roi_mode,
        road_roi_rect_normalized=road_roi_rect,
        motion_score_threshold=_read_positive_float(ENV_ADAPTIVE_MOTION_SCORE_THRESHOLD, DEFAULT_ADAPTIVE_MOTION_SCORE_THRESHOLD),
        motion_pixels_ratio_threshold=_read_positive_float(
            ENV_ADAPTIVE_MOTION_PIXELS_RATIO_THRESHOLD,
            DEFAULT_ADAPTIVE_MOTION_PIXELS_RATIO_THRESHOLD,
        ),
        histogram_change_threshold=_read_positive_float(
            ENV_ADAPTIVE_HISTOGRAM_CHANGE_THRESHOLD,
            DEFAULT_ADAPTIVE_HISTOGRAM_CHANGE_THRESHOLD,
        ),
        min_blob_area_ratio=_read_positive_float(ENV_ADAPTIVE_MIN_BLOB_AREA_RATIO, DEFAULT_ADAPTIVE_MIN_BLOB_AREA_RATIO),
        heartbeat_seconds=_read_positive_float(ENV_ADAPTIVE_HEARTBEAT_SECONDS, DEFAULT_ADAPTIVE_HEARTBEAT_SECONDS),
        min_selected_gap_seconds=_read_positive_float(
            ENV_ADAPTIVE_MIN_SELECTED_GAP_SECONDS,
            DEFAULT_ADAPTIVE_MIN_SELECTED_GAP_SECONDS,
        ),
        preview_limit=_read_positive_int(ENV_ADAPTIVE_PREVIEW_LIMIT, DEFAULT_ADAPTIVE_PREVIEW_LIMIT),
        min_brightness=_read_positive_float(ENV_ADAPTIVE_MIN_BRIGHTNESS, DEFAULT_ADAPTIVE_MIN_BRIGHTNESS),
        max_brightness=_read_positive_float(ENV_ADAPTIVE_MAX_BRIGHTNESS, DEFAULT_ADAPTIVE_MAX_BRIGHTNESS),
        min_pixel_variance=_read_positive_float(ENV_ADAPTIVE_MIN_PIXEL_VARIANCE, DEFAULT_ADAPTIVE_MIN_PIXEL_VARIANCE),
        min_blur_score=_read_positive_float(ENV_ADAPTIVE_MIN_BLUR_SCORE, DEFAULT_ADAPTIVE_MIN_BLUR_SCORE),
    )


def _normalized_rect_to_pixels(
    normalized_rect: tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
) -> list[int]:
    """Convert normalized ROI coordinates into pixel coordinates."""

    x1 = int(round(normalized_rect[0] * frame_width))
    y1 = int(round(normalized_rect[1] * frame_height))
    x2 = int(round(normalized_rect[2] * frame_width))
    y2 = int(round(normalized_rect[3] * frame_height))
    x1 = max(0, min(x1, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))
    x2 = max(x1 + 1, min(x2, frame_width))
    y2 = max(y1 + 1, min(y2, frame_height))
    return [x1, y1, x2, y2]


def _crop_roi(image: np.ndarray, roi_pixels: list[int]) -> np.ndarray:
    """Crop the configured road ROI from an image."""

    x1, y1, x2, y2 = roi_pixels
    return image[y1:y2, x1:x2]


def compute_quality_metrics(frame: np.ndarray | None, config: AdaptiveSamplingConfig) -> dict[str, Any]:
    """Compute brightness, variance, blur, and conservative validity flags."""

    if frame is None or frame.size == 0:
        return {
            "brightness_mean": 0.0,
            "pixel_variance": 0.0,
            "blur_score": 0.0,
            "valid_frame": False,
            "quality_failure_reasons": ["unreadable_image"],
        }

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness_mean = float(np.mean(gray_frame))
    pixel_variance = float(np.var(gray_frame))
    blur_score = float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())

    quality_failure_reasons: list[str] = []
    if brightness_mean < config.min_brightness:
        quality_failure_reasons.append("too_dark")
    if brightness_mean > config.max_brightness:
        quality_failure_reasons.append("too_bright")
    if pixel_variance < config.min_pixel_variance:
        quality_failure_reasons.append("very_low_pixel_variance")
    if blur_score < config.min_blur_score:
        quality_failure_reasons.append("extremely_blurry")

    return {
        "brightness_mean": round(brightness_mean, 3),
        "pixel_variance": round(pixel_variance, 3),
        "blur_score": round(blur_score, 3),
        "valid_frame": not quality_failure_reasons,
        "quality_failure_reasons": quality_failure_reasons,
    }


def compute_motion_score(current_roi_gray: np.ndarray, previous_roi_gray: np.ndarray | None) -> tuple[float, np.ndarray | None]:
    """Use grayscale differencing to estimate scene motion between sampled frames."""

    if previous_roi_gray is None:
        return 0.0, None

    diff = cv2.absdiff(current_roi_gray, previous_roi_gray)
    motion_score = float(np.mean(diff) / 255.0)
    return round(clamp(motion_score, 0.0, 1.0), 6), diff


def compute_motion_pixels_and_blobs(
    diff_image: np.ndarray | None,
    roi_pixel_count: int,
    min_blob_area_ratio: float,
) -> dict[str, Any]:
    """Use thresholded diff images to estimate changed pixels and motion blobs."""

    if diff_image is None or roi_pixel_count <= 0:
        return {
            "motion_pixels_ratio": 0.0,
            "motion_blob_count": 0,
            "largest_motion_blob_area": 0.0,
            "largest_motion_blob_area_ratio": 0.0,
            "largest_motion_blob_bbox_xyxy": None,
        }

    _, thresholded = cv2.threshold(diff_image, 25, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), dtype=np.uint8)
    cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.dilate(cleaned, kernel, iterations=2)

    motion_pixels_ratio = round(clamp(float(np.count_nonzero(cleaned)) / float(roi_pixel_count), 0.0, 1.0), 6)
    contour_min_area = max(25.0, float(roi_pixel_count) * min_blob_area_ratio * 0.25)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_motion_blob_area = 0.0
    largest_motion_blob_bbox_xyxy: list[int] | None = None
    kept_contours = 0
    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < contour_min_area:
            continue
        kept_contours += 1
        if contour_area > largest_motion_blob_area:
            largest_motion_blob_area = contour_area
            x, y, width, height = cv2.boundingRect(contour)
            largest_motion_blob_bbox_xyxy = [int(x), int(y), int(x + width), int(y + height)]

    largest_motion_blob_area_ratio = round(
        clamp(largest_motion_blob_area / float(roi_pixel_count), 0.0, 1.0),
        6,
    )
    return {
        "motion_pixels_ratio": motion_pixels_ratio,
        "motion_blob_count": kept_contours,
        "largest_motion_blob_area": round(largest_motion_blob_area, 3),
        "largest_motion_blob_area_ratio": largest_motion_blob_area_ratio,
        "largest_motion_blob_bbox_xyxy": largest_motion_blob_bbox_xyxy,
    }


def compute_histogram_change_score(
    current_roi_bgr: np.ndarray,
    previous_selected_roi_histogram: np.ndarray | None,
) -> tuple[float, np.ndarray]:
    """Use histogram comparison to distinguish near-duplicates from changed frames."""

    hsv_frame = cv2.cvtColor(current_roi_bgr, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv_frame], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram)

    if previous_selected_roi_histogram is None:
        return 0.0, histogram

    raw_distance = float(cv2.compareHist(previous_selected_roi_histogram, histogram, cv2.HISTCMP_BHATTACHARYYA))
    return round(clamp(raw_distance, 0.0, 1.0), 6), histogram


def _safe_stats(values: list[float]) -> dict[str, float]:
    """Return min/max/avg stats with zero defaults."""

    if not values:
        return {"min": 0.0, "max": 0.0, "avg": 0.0}
    return {
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "avg": round(sum(values) / len(values), 6),
    }


def _safe_gap_stats(values: list[float]) -> dict[str, float]:
    """Return gap stats using the expected field names."""

    if not values:
        return {
            "min_seconds_since_previous_selected": 0.0,
            "max_seconds_since_previous_selected": 0.0,
            "avg_seconds_since_previous_selected": 0.0,
        }
    return {
        "min_seconds_since_previous_selected": round(min(values), 6),
        "max_seconds_since_previous_selected": round(max(values), 6),
        "avg_seconds_since_previous_selected": round(sum(values) / len(values), 6),
    }


def run_motion_adaptive_sampling(run_dir: Path, adaptive_config: AdaptiveSamplingConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run Step 02A using only step-01/02 outputs and sampled-frame images."""

    video_info = read_json(run_dir / "01_video_info.json")
    sampled_manifest = read_json(run_dir / "02_sampled_frames.json")
    sampled_frames = list(sampled_manifest.get("sampled_frames", []))
    if not sampled_frames:
        raise ValueError("02_sampled_frames.json does not contain any sampled frames.")

    input_video_path = str(video_info.get("input_video_path", ""))
    frame_width = int(video_info.get("width", 0) or 0)
    frame_height = int(video_info.get("height", 0) or 0)
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("01_video_info.json does not contain valid width/height for adaptive filtering.")

    road_roi_rect_pixels = _normalized_rect_to_pixels(
        adaptive_config.road_roi_rect_normalized,
        frame_width,
        frame_height,
    )
    roi_pixel_count = max(1, (road_roi_rect_pixels[2] - road_roi_rect_pixels[0]) * (road_roi_rect_pixels[3] - road_roi_rect_pixels[1]))

    preview_dir = run_dir / "02A_adaptive_preview_frames"
    preview_dir.mkdir(parents=True, exist_ok=False)

    selected_frames: list[dict[str, Any]] = []
    keep_reason_counts: Counter[str] = Counter()
    skip_reason_counts: Counter[str] = Counter()
    quality_failure_counts: Counter[str] = Counter()
    selected_keep_reason_primary: list[str] = []
    motion_score_values: list[float] = []
    motion_pixels_ratio_values: list[float] = []
    histogram_change_score_values: list[float] = []
    largest_motion_blob_area_values: list[float] = []
    largest_motion_blob_area_ratio_values: list[float] = []
    previous_sampled_roi_gray: np.ndarray | None = None
    previous_selected_roi_histogram: np.ndarray | None = None
    previous_sampled_frame_id: str | None = None
    previous_sampled_timestamp_seconds: float | None = None
    previous_selected_frame_id: str | None = None
    previous_selected_timestamp_seconds: float | None = None

    scanned_records: list[dict[str, Any]] = []

    for item in sampled_frames:
        image_relative_path = str(item.get("image_path", "")).replace("/", "\\")
        image_path = run_dir / Path(image_relative_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Sampled frame image not found: {image_path}")

        frame = cv2.imread(str(image_path))
        if frame is None:
            frame_bgr = None
        else:
            frame_bgr = frame

        quality_metrics = compute_quality_metrics(frame_bgr, adaptive_config)
        timestamp_seconds = float(item.get("timestamp_seconds", 0.0) or 0.0)
        if frame_bgr is None:
            roi_bgr = None
            roi_gray = None
            motion_score = 0.0
            diff_image = None
            blob_metrics = {
                "motion_pixels_ratio": 0.0,
                "motion_blob_count": 0,
                "largest_motion_blob_area": 0.0,
                "largest_motion_blob_area_ratio": 0.0,
                "largest_motion_blob_bbox_xyxy": None,
            }
            histogram_change_score = 0.0
            current_histogram = None
        else:
            roi_bgr = _crop_roi(frame_bgr, road_roi_rect_pixels)
            roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            motion_score, diff_image = compute_motion_score(roi_gray, previous_sampled_roi_gray)
            blob_metrics = compute_motion_pixels_and_blobs(diff_image, roi_pixel_count, adaptive_config.min_blob_area_ratio)
            histogram_change_score, current_histogram = compute_histogram_change_score(roi_bgr, previous_selected_roi_histogram)

        motion_score_values.append(float(motion_score))
        motion_pixels_ratio_values.append(float(blob_metrics["motion_pixels_ratio"]))
        histogram_change_score_values.append(float(histogram_change_score))
        largest_motion_blob_area_values.append(float(blob_metrics["largest_motion_blob_area"]))
        largest_motion_blob_area_ratio_values.append(float(blob_metrics["largest_motion_blob_area_ratio"]))

        if previous_sampled_timestamp_seconds is None:
            seconds_since_previous_sampled = None
        else:
            seconds_since_previous_sampled = round(timestamp_seconds - previous_sampled_timestamp_seconds, 6)

        if previous_selected_timestamp_seconds is None:
            seconds_since_previous_selected = None
        else:
            seconds_since_previous_selected = round(timestamp_seconds - previous_selected_timestamp_seconds, 6)

        keep_reason: list[str] = []
        skip_reason: list[str] = []

        if not quality_metrics["valid_frame"]:
            skip_reason.append("invalid_frame")
            for reason in quality_metrics["quality_failure_reasons"]:
                quality_failure_counts[reason] += 1
        else:
            if not adaptive_config.enabled:
                keep_reason.append("adaptive_disabled_pass_through")
            else:
                if previous_selected_timestamp_seconds is None:
                    keep_reason.append("first_valid_frame")
                if motion_score >= adaptive_config.motion_score_threshold:
                    keep_reason.append("motion_score")
                if float(blob_metrics["motion_pixels_ratio"]) >= adaptive_config.motion_pixels_ratio_threshold:
                    keep_reason.append("motion_pixels")
                if float(blob_metrics["largest_motion_blob_area_ratio"]) >= adaptive_config.min_blob_area_ratio:
                    keep_reason.append("motion_blob")
                if histogram_change_score >= adaptive_config.histogram_change_threshold:
                    keep_reason.append("histogram_change")
                # Heartbeat keeps periodic frames during idle periods so later YOLO does not miss long quiet windows.
                if (
                    previous_selected_timestamp_seconds is not None
                    and seconds_since_previous_selected is not None
                    and seconds_since_previous_selected >= adaptive_config.heartbeat_seconds
                ):
                    keep_reason.append("heartbeat")

                if (
                    previous_selected_timestamp_seconds is not None
                    and seconds_since_previous_selected is not None
                    and seconds_since_previous_selected < adaptive_config.min_selected_gap_seconds
                ):
                    skip_reason.append("too_close_to_previous_selected")

                if not keep_reason:
                    if histogram_change_score < adaptive_config.histogram_change_threshold:
                        skip_reason.append("near_duplicate")
                    else:
                        skip_reason.append("low_motion_no_heartbeat")

        should_keep = bool(
            quality_metrics["valid_frame"]
            and keep_reason
            and (
                not adaptive_config.enabled
                or seconds_since_previous_selected is None
                or seconds_since_previous_selected >= adaptive_config.min_selected_gap_seconds
            )
        )

        if should_keep:
            selected_index = len(selected_frames) + 1
            selected_payload = {
                "selected_index": selected_index,
                "source_sample_index": int(item.get("sample_index", selected_index) or selected_index),
                "frame_id": str(item.get("frame_id", "")),
                "frame_idx": int(item.get("frame_idx", 0) or 0),
                "timestamp_seconds": round(timestamp_seconds, 3),
                "timestamp_text": format_timestamp(timestamp_seconds),
                "image_path": str(item.get("image_path", "")),
                "original_video_path": str(item.get("original_video_path", input_video_path)),
                "adaptive_selected": True,
                "keep_reason": keep_reason,
                "motion_score": round(motion_score, 6),
                "motion_pixels_ratio": round(float(blob_metrics["motion_pixels_ratio"]), 6),
                "histogram_change_score": round(histogram_change_score, 6),
                "motion_blob_count": int(blob_metrics["motion_blob_count"]),
                "largest_motion_blob_area": round(float(blob_metrics["largest_motion_blob_area"]), 3),
                "largest_motion_blob_area_ratio": round(float(blob_metrics["largest_motion_blob_area_ratio"]), 6),
                "largest_motion_blob_bbox_xyxy": blob_metrics["largest_motion_blob_bbox_xyxy"],
                "brightness_mean": quality_metrics["brightness_mean"],
                "pixel_variance": quality_metrics["pixel_variance"],
                "blur_score": quality_metrics["blur_score"],
                "valid_frame": quality_metrics["valid_frame"],
                "previous_selected_frame_id": previous_selected_frame_id,
                "previous_selected_timestamp_seconds": previous_selected_timestamp_seconds,
                # Time gaps are recorded so later tracking stages know how sparse the pre-YOLO inputs were.
                "seconds_since_previous_selected": seconds_since_previous_selected,
                "previous_sampled_frame_id": previous_sampled_frame_id,
                "seconds_since_previous_sampled": seconds_since_previous_sampled,
                "roi_mode": adaptive_config.roi_mode,
                "road_roi_rect_pixels": road_roi_rect_pixels,
            }
            selected_frames.append(selected_payload)
            primary_reason = keep_reason[0]
            selected_keep_reason_primary.append(primary_reason)
            for reason in keep_reason:
                keep_reason_counts[reason] += 1

            if selected_index <= adaptive_config.preview_limit:
                preview_output_path = preview_dir / image_path.name
                shutil.copy2(image_path, preview_output_path)

            previous_selected_frame_id = str(item.get("frame_id", ""))
            previous_selected_timestamp_seconds = timestamp_seconds
            if current_histogram is not None:
                previous_selected_roi_histogram = current_histogram
        else:
            if not skip_reason:
                skip_reason.append("low_motion_no_heartbeat")
            for reason in skip_reason:
                skip_reason_counts[reason] += 1

        scanned_records.append(
            {
                "timestamp_seconds": timestamp_seconds,
                "selected": should_keep,
                "keep_reason": keep_reason[0] if keep_reason else None,
                "motion_score": round(motion_score, 6),
                "motion_pixels_ratio": round(float(blob_metrics["motion_pixels_ratio"]), 6),
                "histogram_change_score": round(histogram_change_score, 6),
            }
        )
        previous_sampled_frame_id = str(item.get("frame_id", ""))
        previous_sampled_timestamp_seconds = timestamp_seconds
        if roi_gray is not None:
            # Grayscale differencing is the core pre-YOLO motion filter between base-sampled frames.
            previous_sampled_roi_gray = roi_gray

    input_sampled_frames = len(sampled_frames)
    selected_for_yolo = len(selected_frames)
    skipped_frames = max(0, input_sampled_frames - selected_for_yolo)
    selection_ratio = round((selected_for_yolo / input_sampled_frames), 6) if input_sampled_frames > 0 else 0.0

    timeline_buckets_map: dict[int, dict[str, Any]] = {}
    for record in scanned_records:
        bucket_start = int(record["timestamp_seconds"] // 10) * 10
        bucket = timeline_buckets_map.setdefault(
            bucket_start,
            {
                "start_seconds": bucket_start,
                "end_seconds": bucket_start + 10,
                "sampled_frames": 0,
                "selected_frames": 0,
                "keep_reason_counter": Counter(),
                "motion_score_values": [],
                "motion_pixels_ratio_values": [],
                "histogram_change_score_values": [],
            },
        )
        bucket["sampled_frames"] += 1
        if record["selected"]:
            bucket["selected_frames"] += 1
            if record["keep_reason"]:
                bucket["keep_reason_counter"][str(record["keep_reason"])] += 1
        bucket["motion_score_values"].append(float(record["motion_score"]))
        bucket["motion_pixels_ratio_values"].append(float(record["motion_pixels_ratio"]))
        bucket["histogram_change_score_values"].append(float(record["histogram_change_score"]))

    timeline_buckets: list[dict[str, Any]] = []
    for bucket_start in sorted(timeline_buckets_map):
        bucket = timeline_buckets_map[bucket_start]
        dominant_keep_reason = bucket["keep_reason_counter"].most_common(1)[0][0] if bucket["keep_reason_counter"] else None
        timeline_buckets.append(
            {
                "start_seconds": bucket["start_seconds"],
                "end_seconds": bucket["end_seconds"],
                "sampled_frames": bucket["sampled_frames"],
                "selected_frames": bucket["selected_frames"],
                "dominant_keep_reason": dominant_keep_reason,
                "avg_motion_score": round(sum(bucket["motion_score_values"]) / len(bucket["motion_score_values"]), 6),
                "avg_motion_pixels_ratio": round(
                    sum(bucket["motion_pixels_ratio_values"]) / len(bucket["motion_pixels_ratio_values"]),
                    6,
                ),
                "avg_histogram_change_score": round(
                    sum(bucket["histogram_change_score_values"]) / len(bucket["histogram_change_score_values"]),
                    6,
                ),
            }
        )

    selected_gap_values = [
        float(item["seconds_since_previous_selected"])
        for item in selected_frames
        if item["seconds_since_previous_selected"] is not None
    ]

    manifest = {
        "status": "success",
        "input_video_path": input_video_path,
        "source_manifest": "02_sampled_frames.json",
        "input_sampled_frames": input_sampled_frames,
        "selected_for_yolo": selected_for_yolo,
        "skipped_frames": skipped_frames,
        "selection_ratio": selection_ratio,
        "adaptive_config": {
            "enabled": adaptive_config.enabled,
            "motion_score_threshold": adaptive_config.motion_score_threshold,
            "motion_pixels_ratio_threshold": adaptive_config.motion_pixels_ratio_threshold,
            "histogram_change_threshold": adaptive_config.histogram_change_threshold,
            "min_blob_area_ratio": adaptive_config.min_blob_area_ratio,
            "heartbeat_seconds": adaptive_config.heartbeat_seconds,
            "min_selected_gap_seconds": adaptive_config.min_selected_gap_seconds,
            "preview_limit": adaptive_config.preview_limit,
            "min_brightness": adaptive_config.min_brightness,
            "max_brightness": adaptive_config.max_brightness,
            "min_pixel_variance": adaptive_config.min_pixel_variance,
            "min_blur_score": adaptive_config.min_blur_score,
        },
        "roi_mode": adaptive_config.roi_mode,
        "road_roi_rect_normalized": [round(value, 4) for value in adaptive_config.road_roi_rect_normalized],
        "road_roi_rect_pixels": road_roi_rect_pixels,
        "selected_frames": selected_frames,
    }

    report = {
        "status": "success",
        "input_video_path": input_video_path,
        "input_sampled_frames": input_sampled_frames,
        "selected_for_yolo": selected_for_yolo,
        "skipped_frames": skipped_frames,
        "selection_ratio": selection_ratio,
        "roi_mode": adaptive_config.roi_mode,
        "road_roi_rect_normalized": [round(value, 4) for value in adaptive_config.road_roi_rect_normalized],
        "road_roi_rect_pixels": road_roi_rect_pixels,
        "keep_reason_counts": dict(sorted(keep_reason_counts.items())),
        "skip_reason_counts": dict(sorted(skip_reason_counts.items())),
        "quality_failure_counts": dict(sorted(quality_failure_counts.items())),
        "selected_gap_stats": _safe_gap_stats(selected_gap_values),
        "motion_score_stats": _safe_stats(motion_score_values),
        "motion_pixels_ratio_stats": _safe_stats(motion_pixels_ratio_values),
        "histogram_change_score_stats": _safe_stats(histogram_change_score_values),
        "motion_blob_stats": {
            "max_largest_motion_blob_area": round(max(largest_motion_blob_area_values), 3) if largest_motion_blob_area_values else 0.0,
            "avg_largest_motion_blob_area": round(sum(largest_motion_blob_area_values) / len(largest_motion_blob_area_values), 3)
            if largest_motion_blob_area_values
            else 0.0,
            "max_largest_motion_blob_area_ratio": round(max(largest_motion_blob_area_ratio_values), 6)
            if largest_motion_blob_area_ratio_values
            else 0.0,
            "avg_largest_motion_blob_area_ratio": round(
                sum(largest_motion_blob_area_ratio_values) / len(largest_motion_blob_area_ratio_values),
                6,
            )
            if largest_motion_blob_area_ratio_values
            else 0.0,
        },
        "timeline_buckets": timeline_buckets,
    }

    write_json(run_dir / "02A_adaptive_frames.json", manifest)
    write_json(run_dir / "02A_adaptive_filter_report.json", report)
    return manifest, report
