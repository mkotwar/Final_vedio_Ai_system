from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass
class MotionTriggerState:
    persistence_counter: int = 0
    overlap_cooldown_remaining: int = 0


@dataclass
class MotionTrigger:
    motion_min_area_ratio: float
    motion_persistence_frames: int
    motion_track_region_expansion: float
    subtractor: Any = field(default_factory=lambda: cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=32, detectShadows=False))
    recent_uncovered_ratios: deque[float] = field(default_factory=lambda: deque(maxlen=8))
    state: MotionTriggerState = field(default_factory=MotionTriggerState)

    def _expand_box(self, bbox_xyxy: list[float], frame_width: int, frame_height: int) -> list[int]:
        x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        expand_x = width * self.motion_track_region_expansion
        expand_y = height * self.motion_track_region_expansion
        return [
            max(0, int(round(x1 - expand_x))),
            max(0, int(round(y1 - expand_y))),
            min(frame_width, int(round(x2 + expand_x))),
            min(frame_height, int(round(y2 + expand_y))),
        ]

    def evaluate(
        self,
        *,
        frame,
        active_boxes: list[list[float]],
        roi_masks: list[dict[str, float]] | None,
    ) -> dict[str, Any]:
        frame_height, frame_width = frame.shape[:2]
        mask = self.subtractor.apply(frame)
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        for bbox in active_boxes:
            x1, y1, x2, y2 = self._expand_box(bbox, frame_width, frame_height)
            mask[y1:y2, x1:x2] = 0

        roi_mask = np.ones_like(mask, dtype=np.uint8) * 255
        if roi_masks:
            roi_mask[:] = 0
            for zone in roi_masks:
                x1 = max(0, min(frame_width, int(round(float(zone["x1"]) * frame_width))))
                y1 = max(0, min(frame_height, int(round(float(zone["y1"]) * frame_height))))
                x2 = max(0, min(frame_width, int(round(float(zone["x2"]) * frame_width))))
                y2 = max(0, min(frame_height, int(round(float(zone["y2"]) * frame_height))))
                roi_mask[y1:y2, x1:x2] = 255
        mask = cv2.bitwise_and(mask, roi_mask)

        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        uncovered_regions: list[dict[str, Any]] = []
        total_motion_area = 0.0
        largest_uncovered_region = 0.0
        frame_area = max(1.0, float(frame_width * frame_height))
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = float(width * height)
            area_ratio = area / frame_area
            if area_ratio < self.motion_min_area_ratio:
                continue
            uncovered_regions.append(
                {
                    "bbox_xyxy": [float(x), float(y), float(x + width), float(y + height)],
                    "area_pixels": int(area),
                    "area_ratio": round(area_ratio, 6),
                }
            )
            total_motion_area += area
            largest_uncovered_region = max(largest_uncovered_region, area)

        motion_area_ratio = total_motion_area / frame_area
        self.recent_uncovered_ratios.append(motion_area_ratio)
        if uncovered_regions:
            self.state.persistence_counter += 1
        else:
            self.state.persistence_counter = 0
        return {
            "enabled": True,
            "triggered": bool(uncovered_regions and self.state.persistence_counter >= self.motion_persistence_frames),
            "motion_area_pixels": int(total_motion_area),
            "motion_area_ratio": round(motion_area_ratio, 6),
            "uncovered_region_count": len(uncovered_regions),
            "largest_uncovered_region": int(largest_uncovered_region),
            "persistence_count": int(self.state.persistence_counter),
            "uncovered_regions": uncovered_regions,
            "mask": mask,
        }


def detect_entry_zone_motion(
    *,
    motion_regions: list[dict[str, Any]],
    entry_zones: list[dict[str, float]],
    frame_width: int,
    frame_height: int,
) -> tuple[bool, list[str]]:
    if not entry_zones:
        return False, []
    triggered_names: list[str] = []
    for region in motion_regions:
        rx1, ry1, rx2, ry2 = [float(value) for value in region["bbox_xyxy"]]
        for zone in entry_zones:
            zx1 = float(zone["x1"]) * frame_width
            zy1 = float(zone["y1"]) * frame_height
            zx2 = float(zone["x2"]) * frame_width
            zy2 = float(zone["y2"]) * frame_height
            if rx2 <= zx1 or rx1 >= zx2 or ry2 <= zy1 or ry1 >= zy2:
                continue
            zone_name = str(zone.get("name", "entry_zone"))
            if zone_name not in triggered_names:
                triggered_names.append(zone_name)
    return bool(triggered_names), triggered_names

