from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class MotionGateConfig:
    roi_top_ratio: float = 0.20
    roi_bottom_ratio: float = 0.95
    roi_left_ratio: float = 0.05
    roi_right_ratio: float = 0.95
    changed_pixel_threshold: int = 24
    changed_ratio_threshold: float = 0.0035
    min_blob_area_ratio: float = 0.0009
    histogram_delta_threshold: float = 0.18
    blur_kernel_size: int = 5
    morph_kernel_size: int = 5


@dataclass(frozen=True)
class MotionGateResult:
    motion_score: float
    changed_ratio: float
    largest_blob_area_ratio: float
    histogram_delta: float
    meaningful_motion: bool
    roi_bounds: list[int]


class CheapMotionGate:
    def __init__(self, config: MotionGateConfig | None = None):
        self.config = config or MotionGateConfig()
        self._previous_gray_roi: np.ndarray | None = None
        self._previous_histogram: np.ndarray | None = None

    def evaluate(self, frame: np.ndarray) -> MotionGateResult:
        height, width = frame.shape[:2]
        y1 = int(round(height * self.config.roi_top_ratio))
        y2 = int(round(height * self.config.roi_bottom_ratio))
        x1 = int(round(width * self.config.roi_left_ratio))
        x2 = int(round(width * self.config.roi_right_ratio))
        y1 = max(0, min(y1, height))
        y2 = max(y1 + 1, min(y2, height))
        x1 = max(0, min(x1, width))
        x2 = max(x1 + 1, min(x2, width))
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if self.config.blur_kernel_size > 1:
            gray = cv2.GaussianBlur(gray, (self.config.blur_kernel_size, self.config.blur_kernel_size), 0)

        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if self._previous_gray_roi is None or self._previous_histogram is None:
            self._previous_gray_roi = gray
            self._previous_histogram = hist
            return MotionGateResult(
                motion_score=0.0,
                changed_ratio=0.0,
                largest_blob_area_ratio=0.0,
                histogram_delta=0.0,
                meaningful_motion=False,
                roi_bounds=[x1, y1, x2, y2],
            )

        delta = cv2.absdiff(gray, self._previous_gray_roi)
        _, binary = cv2.threshold(delta, self.config.changed_pixel_threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((self.config.morph_kernel_size, self.config.morph_kernel_size), dtype=np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        changed_ratio = float(np.count_nonzero(cleaned)) / float(cleaned.size) if cleaned.size else 0.0

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_area = float(roi.shape[0] * roi.shape[1]) if roi.size else 1.0
        largest_blob_area = max((cv2.contourArea(contour) for contour in contours), default=0.0)
        largest_blob_area_ratio = largest_blob_area / roi_area if roi_area > 0 else 0.0

        histogram_delta = 1.0 - float(cv2.compareHist(self._previous_histogram.astype(np.float32), hist.astype(np.float32), cv2.HISTCMP_CORREL))
        histogram_delta = max(0.0, histogram_delta)
        motion_score = max(changed_ratio, largest_blob_area_ratio, histogram_delta * 0.5)
        meaningful_motion = bool(
            changed_ratio >= self.config.changed_ratio_threshold
            or largest_blob_area_ratio >= self.config.min_blob_area_ratio
            or histogram_delta >= self.config.histogram_delta_threshold
        )

        self._previous_gray_roi = gray
        self._previous_histogram = hist
        return MotionGateResult(
            motion_score=round(motion_score, 6),
            changed_ratio=round(changed_ratio, 6),
            largest_blob_area_ratio=round(largest_blob_area_ratio, 6),
            histogram_delta=round(histogram_delta, 6),
            meaningful_motion=meaningful_motion,
            roi_bounds=[x1, y1, x2, y2],
        )


__all__ = [
    "CheapMotionGate",
    "MotionGateConfig",
    "MotionGateResult",
]
