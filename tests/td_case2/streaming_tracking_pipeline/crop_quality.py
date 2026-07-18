from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .config import CropCollectionConfig
from .schemas import BoundingBox, CropQualityMetrics


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _cv2() -> Any:
    try:
        import cv2  # type: ignore

        return cv2
    except Exception:
        return None


@dataclass(frozen=True)
class ExtractedCrop:
    crop: Any
    crop_bbox: BoundingBox
    requested_bbox: BoundingBox
    crop_width: int
    crop_height: int
    crop_completeness: float
    padding_clipped: bool


def padded_crop_bbox(bbox: BoundingBox, *, frame_width: int, frame_height: int, padding_ratio: float) -> tuple[BoundingBox, BoundingBox, bool]:
    width = bbox.x2 - bbox.x1
    height = bbox.y2 - bbox.y1
    pad_x = width * padding_ratio
    pad_y = height * padding_ratio
    requested = BoundingBox(
        math.floor(bbox.x1 - pad_x),
        math.floor(bbox.y1 - pad_y),
        math.ceil(bbox.x2 + pad_x),
        math.ceil(bbox.y2 + pad_y),
    )
    clipped = requested.clip(frame_width, frame_height)
    padding_clipped = clipped.to_xyxy() != requested.to_xyxy()
    return requested, clipped, padding_clipped


def extract_crop(frame: Any, bbox: BoundingBox, *, frame_width: int, frame_height: int, config: CropCollectionConfig) -> ExtractedCrop | None:
    if frame is None:
        return None
    requested, clipped, padding_clipped = padded_crop_bbox(
        bbox,
        frame_width=frame_width,
        frame_height=frame_height,
        padding_ratio=config.padding_ratio,
    )
    x1, y1, x2, y2 = (int(value) for value in clipped.to_xyxy())
    crop_width = x2 - x1
    crop_height = y2 - y1
    if crop_width < config.minimum_crop_width or crop_height < config.minimum_crop_height:
        return None
    crop = frame[y1:y2, x1:x2].copy()
    if getattr(crop, "size", 0) == 0:
        return None
    requested_area = max(requested.area, 1.0)
    completeness = _clip01(clipped.area / requested_area)
    return ExtractedCrop(
        crop=crop,
        crop_bbox=clipped,
        requested_bbox=requested,
        crop_width=crop_width,
        crop_height=crop_height,
        crop_completeness=completeness,
        padding_clipped=padding_clipped,
    )


def _to_grayscale_array(image: Any) -> Any:
    cv2 = _cv2()
    if cv2 is not None and getattr(image, "ndim", 0) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if getattr(image, "ndim", 0) == 3:
        return image.mean(axis=2)
    return image


def _sharpness(gray: Any) -> float:
    cv2 = _cv2()
    if cv2 is None:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _brightness(gray: Any) -> float:
    return _clip01(float(gray.mean()) / 255.0)


def _contrast(gray: Any) -> float:
    return float(gray.std())


def score_preliminary_quality(
    *,
    detection_confidence: float,
    bbox_area_ratio: float,
    sharpness: float,
    brightness: float,
    contrast: float,
    edge_touching: bool,
    crop_completeness: float,
    config: CropCollectionConfig,
) -> float:
    sharpness_term = _clip01(sharpness / config.sharpness_normalization_cap)
    contrast_term = _clip01(contrast / config.contrast_normalization_cap)
    brightness_distance = abs(brightness - config.target_brightness)
    brightness_term = _clip01(1.0 - (brightness_distance / max(config.target_brightness, 1.0 - config.target_brightness, 1e-6)))
    positive = (
        config.preliminary_confidence_weight * detection_confidence
        + config.preliminary_area_weight * math.sqrt(_clip01(bbox_area_ratio))
        + config.preliminary_sharpness_weight * sharpness_term
        + config.preliminary_brightness_weight * brightness_term
        + config.preliminary_contrast_weight * contrast_term
        + config.preliminary_completeness_weight * crop_completeness
    )
    positive_weights = (
        config.preliminary_confidence_weight
        + config.preliminary_area_weight
        + config.preliminary_sharpness_weight
        + config.preliminary_brightness_weight
        + config.preliminary_contrast_weight
        + config.preliminary_completeness_weight
    )
    score = positive / max(positive_weights, 1e-9)
    if edge_touching:
        score -= config.preliminary_edge_penalty_weight
    return round(_clip01(score), 6)


def compute_crop_quality(
    *,
    crop: Any,
    source_bbox: BoundingBox,
    extracted: ExtractedCrop,
    frame_width: int,
    frame_height: int,
    detection_confidence: float,
    config: CropCollectionConfig,
) -> CropQualityMetrics:
    gray = _to_grayscale_array(crop)
    sharpness = _sharpness(gray)
    brightness = _brightness(gray)
    contrast = _contrast(gray)
    bbox_area_ratio = _clip01(source_bbox.area / max(float(frame_width * frame_height), 1.0))
    edge_touching = source_bbox.touches_frame_edge(frame_width, frame_height, margin_ratio=config.edge_margin_ratio)
    preliminary = score_preliminary_quality(
        detection_confidence=detection_confidence,
        bbox_area_ratio=bbox_area_ratio,
        sharpness=sharpness,
        brightness=brightness,
        contrast=contrast,
        edge_touching=edge_touching,
        crop_completeness=extracted.crop_completeness,
        config=config,
    )
    return CropQualityMetrics(
        detection_confidence=detection_confidence,
        bbox_area_ratio=bbox_area_ratio,
        sharpness=round(sharpness, 6),
        brightness=round(brightness, 6),
        edge_touching=edge_touching,
        crop_width=extracted.crop_width,
        crop_height=extracted.crop_height,
        contrast=round(contrast, 6),
        crop_completeness=round(extracted.crop_completeness, 6),
        padding_clipped=extracted.padding_clipped,
        preliminary_score=preliminary,
        combined_score=preliminary,
    )
