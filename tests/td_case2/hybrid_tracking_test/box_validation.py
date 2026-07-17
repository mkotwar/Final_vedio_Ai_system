from __future__ import annotations

import math
from typing import Any

from .data_models import ValidationResult


def xyxy_to_xywh(bbox_xyxy: list[float]) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def xywh_to_xyxy(bbox_xywh: list[float]) -> list[float]:
    x, y, width, height = [float(value) for value in bbox_xywh]
    return [x, y, x + max(0.0, width), y + max(0.0, height)]


def bbox_area(bbox_xyxy: list[float]) -> float:
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_aspect_ratio(bbox_xyxy: list[float]) -> float:
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return width / max(height, 1e-6)


def bbox_center(bbox_xyxy: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    union_area = bbox_area(box_a) + bbox_area(box_b) - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def clip_bbox_to_frame(bbox_xyxy: list[float], frame_width: int, frame_height: int) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    clipped = [
        max(0.0, min(x1, float(frame_width))),
        max(0.0, min(y1, float(frame_height))),
        max(0.0, min(x2, float(frame_width))),
        max(0.0, min(y2, float(frame_height))),
    ]
    if clipped[2] < clipped[0]:
        clipped[0], clipped[2] = clipped[2], clipped[0]
    if clipped[3] < clipped[1]:
        clipped[1], clipped[3] = clipped[3], clipped[1]
    return clipped


def normalized_center_jump(current_bbox_xyxy: list[float], previous_bbox_xyxy: list[float]) -> float:
    current_center = bbox_center(current_bbox_xyxy)
    previous_center = bbox_center(previous_bbox_xyxy)
    center_distance = math.dist(current_center, previous_center)
    previous_diagonal = math.sqrt(max(1.0, bbox_area(previous_bbox_xyxy)))
    return center_distance / max(previous_diagonal, 1.0)


def validate_propagated_bbox(
    *,
    current_bbox_xyxy: list[float],
    previous_bbox_xyxy: list[float] | None,
    frame_width: int,
    frame_height: int,
    minimum_area_ratio_change: float,
    maximum_area_ratio_change: float,
    minimum_aspect_ratio_change: float,
    maximum_aspect_ratio_change: float,
    maximum_center_jump_diagonals: float,
    minimum_visible_area_ratio: float,
) -> ValidationResult:
    bbox = clip_bbox_to_frame(current_bbox_xyxy, frame_width, frame_height)
    reasons: list[str] = []
    metrics: dict[str, Any] = {}

    if any(not math.isfinite(float(value)) for value in bbox):
        reasons.append("non_finite_coordinates")
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if width <= 0 or height <= 0:
        reasons.append("non_positive_box_size")
    if bbox[0] >= frame_width or bbox[1] >= frame_height or bbox[2] <= 0 or bbox[3] <= 0:
        reasons.append("box_outside_frame")

    visible_area_ratio = bbox_area(bbox) / max(1.0, float(frame_width * frame_height))
    metrics["visible_area_ratio"] = round(visible_area_ratio, 6)
    if visible_area_ratio < float(minimum_visible_area_ratio):
        reasons.append("visible_area_too_small")

    if previous_bbox_xyxy is not None:
        previous_area = bbox_area(previous_bbox_xyxy)
        current_area = bbox_area(bbox)
        area_ratio = current_area / max(previous_area, 1.0)
        aspect_ratio_change = bbox_aspect_ratio(bbox) / max(bbox_aspect_ratio(previous_bbox_xyxy), 1e-6)
        center_jump = normalized_center_jump(bbox, previous_bbox_xyxy)
        metrics.update(
            {
                "area_ratio": round(area_ratio, 6),
                "aspect_ratio_change": round(aspect_ratio_change, 6),
                "normalized_center_jump": round(center_jump, 6),
            }
        )
        if area_ratio < float(minimum_area_ratio_change):
            reasons.append("area_ratio_too_small")
        if area_ratio > float(maximum_area_ratio_change):
            reasons.append("area_ratio_too_large")
        if aspect_ratio_change < float(minimum_aspect_ratio_change):
            reasons.append("aspect_ratio_change_too_small")
        if aspect_ratio_change > float(maximum_aspect_ratio_change):
            reasons.append("aspect_ratio_change_too_large")
        if center_jump > float(maximum_center_jump_diagonals):
            reasons.append("center_jump_too_large")

    return ValidationResult(valid=not reasons, reasons=reasons, metrics=metrics)

