from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class MotionEstimate:
    motion_score: float
    changed_pixels_ratio: float


def estimate_scene_motion(previous_frame: Any | None, current_frame: Any) -> MotionEstimate:
    if previous_frame is None:
        return MotionEstimate(motion_score=0.0, changed_pixels_ratio=0.0)
    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    delta = cv2.absdiff(previous_gray, current_gray)
    _, thresholded = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)
    changed_pixels = int(np.count_nonzero(thresholded))
    total_pixels = int(thresholded.shape[0] * thresholded.shape[1]) or 1
    changed_pixels_ratio = changed_pixels / total_pixels
    motion_score = min(1.0, float(delta.mean()) / 32.0)
    return MotionEstimate(
        motion_score=round(motion_score, 6),
        changed_pixels_ratio=round(changed_pixels_ratio, 6),
    )

