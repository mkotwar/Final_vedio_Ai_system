from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from .representative_frame_validator import RepresentativeFrameValidationConfig, clipping_ratio


@dataclass(frozen=True)
class FrameRequest:
    source_frame_index: int


def load_requested_frames(video_path: Path, source_frame_indexes: list[int]) -> dict[int, Any]:
    unique_indexes = sorted(set(int(index) for index in source_frame_indexes if int(index) >= 0))
    if not unique_indexes:
        return {}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames: dict[int, Any] = {}
    current_target_pointer = 0
    current_target = unique_indexes[current_target_pointer]
    frame_index = 0
    while current_target_pointer < len(unique_indexes):
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index == current_target:
            frames[current_target] = frame.copy()
            current_target_pointer += 1
            if current_target_pointer >= len(unique_indexes):
                break
            current_target = unique_indexes[current_target_pointer]
        frame_index += 1
    capture.release()
    return frames


def padded_crop_bbox(bbox_xyxy: list[float], *, frame_width: int, frame_height: int, padding_ratio: float) -> list[int]:
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    pad_x = width * float(padding_ratio)
    pad_y = height * float(padding_ratio)
    return [
        max(0, int(round(x1 - pad_x))),
        max(0, int(round(y1 - pad_y))),
        min(frame_width, int(round(x2 + pad_x))),
        min(frame_height, int(round(y2 + pad_y))),
    ]


def save_frame_and_crop(
    *,
    frame,
    bbox_xyxy: list[float],
    frame_output_path: Path,
    crop_output_path: Path,
    padding_ratio: float,
) -> dict[str, Any]:
    frame_output_path.parent.mkdir(parents=True, exist_ok=True)
    crop_output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_height, frame_width = frame.shape[:2]
    crop_bbox = padded_crop_bbox(
        bbox_xyxy,
        frame_width=frame_width,
        frame_height=frame_height,
        padding_ratio=padding_ratio,
    )
    x1, y1, x2, y2 = crop_bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        raise RuntimeError(f"Empty crop for bbox {bbox_xyxy}")
    if not cv2.imwrite(str(frame_output_path), frame):
        raise RuntimeError(f"Failed to write frame: {frame_output_path}")
    if not cv2.imwrite(str(crop_output_path), crop):
        raise RuntimeError(f"Failed to write crop: {crop_output_path}")
    return {
        "frame_output_path": str(frame_output_path),
        "crop_output_path": str(crop_output_path),
        "crop_bbox_xyxy": [int(value) for value in crop_bbox],
        "crop_width": int(crop.shape[1]),
        "crop_height": int(crop.shape[0]),
    }


def save_validated_frame_and_crop(
    *,
    frame,
    bbox_xyxy: list[float],
    frame_output_path: Path,
    crop_output_path: Path,
    padding_ratio: float,
    identity_crop_eligible: bool,
    frame_width: int,
    frame_height: int,
    config: RepresentativeFrameValidationConfig,
) -> dict[str, Any]:
    result = save_frame_and_crop(
        frame=frame,
        bbox_xyxy=bbox_xyxy,
        frame_output_path=frame_output_path,
        crop_output_path=crop_output_path,
        padding_ratio=padding_ratio,
    )
    crop_status = "invalid"
    crop_content_valid = False
    reasons: list[str] = []
    if not identity_crop_eligible:
        reasons.append("identity_crop_ineligible")
    clip_ratio = clipping_ratio(bbox_xyxy, frame_width, frame_height)
    if result["crop_width"] < config.minimum_crop_width or result["crop_height"] < config.minimum_crop_height:
        reasons.append("crop_too_small")
    if clip_ratio > config.maximum_fallback_crop_clipping_ratio:
        reasons.append("heavily_clipped")
    if not reasons:
        crop_content_valid = True
        crop_status = "ready" if clip_ratio <= config.maximum_ready_crop_clipping_ratio else "fallback"
    return {
        **result,
        "crop_file_written": True,
        "crop_content_valid": crop_content_valid,
        "crop_status": crop_status,
        "validation_reasons": sorted(set(reasons)),
        "clipping_ratio": round(clip_ratio, 6),
    }
