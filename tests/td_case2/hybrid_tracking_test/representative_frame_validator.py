from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RepresentativeFrameValidationConfig:
    confidence_decay_seconds: float = 0.3
    maximum_ready_crop_clipping_ratio: float = 0.15
    maximum_fallback_crop_clipping_ratio: float = 0.35
    minimum_crop_width: int = 24
    minimum_crop_height: int = 24
    minimum_visible_area_ratio: float = 0.0002
    minimum_plate_candidate_score: float = 0.60


def _bbox_area(bbox_xyxy: list[float]) -> float:
    return max(0.0, float(bbox_xyxy[2]) - float(bbox_xyxy[0])) * max(0.0, float(bbox_xyxy[3]) - float(bbox_xyxy[1]))


def clipping_ratio(bbox_xyxy: list[float], frame_width: int, frame_height: int) -> float:
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    clipped_x1 = max(0.0, min(float(frame_width), x1))
    clipped_y1 = max(0.0, min(float(frame_height), y1))
    clipped_x2 = max(0.0, min(float(frame_width), x2))
    clipped_y2 = max(0.0, min(float(frame_height), y2))
    visible = max(0.0, clipped_x2 - clipped_x1) * max(0.0, clipped_y2 - clipped_y1)
    total = _bbox_area(bbox_xyxy)
    if total <= 0.0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - (visible / total)))


def effective_detector_support_score(last_real_yolo_confidence: float | None, seconds_since_last_yolo: float, *, config: RepresentativeFrameValidationConfig) -> float:
    if last_real_yolo_confidence is None:
        return 0.0
    return round(max(0.0, float(last_real_yolo_confidence) * math.exp(-(max(0.0, seconds_since_last_yolo) / max(config.confidence_decay_seconds, 1e-6)))), 6)


def validate_representative_observation(
    observation: dict[str, Any],
    *,
    frame_width: int,
    frame_height: int,
    config: RepresentativeFrameValidationConfig,
) -> dict[str, Any]:
    bbox = [float(value) for value in list(observation.get("bbox_xyxy", []))]
    width = max(0.0, bbox[2] - bbox[0])
    height = max(0.0, bbox[3] - bbox[1])
    visible_area_ratio = _bbox_area([
        max(0.0, min(float(frame_width), bbox[0])),
        max(0.0, min(float(frame_height), bbox[1])),
        max(0.0, min(float(frame_width), bbox[2])),
        max(0.0, min(float(frame_height), bbox[3])),
    ]) / max(float(frame_width * frame_height), 1.0)
    clip_ratio = clipping_ratio(bbox, frame_width, frame_height)
    validity = str(observation.get("observation_validity", "invalid"))
    reasons: list[str] = []
    if width < config.minimum_crop_width or height < config.minimum_crop_height:
        reasons.append("crop_too_small")
    if visible_area_ratio < config.minimum_visible_area_ratio:
        reasons.append("visible_area_too_small")
    if clip_ratio > config.maximum_fallback_crop_clipping_ratio:
        reasons.append("heavily_clipped")
    if validity == "invalid":
        reasons.append("invalid_observation")
    if "frozen_kcf_box" in list(observation.get("drift_segment_flags", [])):
        reasons.append("frozen_kcf_box")
    if "boundary_stuck_box" in list(observation.get("drift_segment_flags", [])):
        reasons.append("boundary_stuck_box")
    seconds_since_last_yolo = float(observation.get("seconds_since_detection", 0.0) or 0.0)
    last_real_confidence = observation.get("last_detection_confidence")
    effective_support = effective_detector_support_score(last_real_confidence, seconds_since_last_yolo, config=config)
    identity_crop_eligible = not reasons and (
        validity == "valid"
        or (validity == "supported" and clip_ratio <= config.maximum_ready_crop_clipping_ratio)
        or (validity == "fallback" and clip_ratio <= config.maximum_fallback_crop_clipping_ratio)
    )
    plate_candidate_eligible = (
        identity_crop_eligible
        and str(observation.get("object_family")) == "vehicle"
        and validity == "valid"
        and clip_ratio <= config.maximum_ready_crop_clipping_ratio
        and visible_area_ratio >= 0.005
        and (float(bbox[3]) / max(float(frame_height), 1.0)) >= 0.25
    )
    return {
        "observation_validity": validity,
        "eligibility_reasons": sorted(set(reasons)),
        "seconds_since_last_yolo": round(seconds_since_last_yolo, 6),
        "last_real_yolo_confidence": None if last_real_confidence is None else round(float(last_real_confidence), 6),
        "effective_detector_support_score": effective_support,
        "clipping_ratio": round(clip_ratio, 6),
        "visible_area_ratio": round(visible_area_ratio, 6),
        "identity_crop_eligible": bool(identity_crop_eligible),
        "plate_crop_eligible": bool(plate_candidate_eligible),
    }


__all__ = [
    "RepresentativeFrameValidationConfig",
    "clipping_ratio",
    "effective_detector_support_score",
    "validate_representative_observation",
]
