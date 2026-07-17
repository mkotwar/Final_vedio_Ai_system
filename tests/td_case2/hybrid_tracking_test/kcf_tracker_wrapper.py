from __future__ import annotations

from dataclasses import dataclass

import cv2

from .box_validation import clip_bbox_to_frame, xywh_to_xyxy, xyxy_to_xywh


def create_kcf_tracker():
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        return cv2.legacy.TrackerKCF_create()
    raise RuntimeError(
        "OpenCV KCF tracker is unavailable. Install a compatible opencv-contrib-python package."
    )


def kcf_api_name() -> str:
    if hasattr(cv2, "TrackerKCF_create"):
        return "cv2.TrackerKCF_create"
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        return "cv2.legacy.TrackerKCF_create"
    return "unavailable"


def _opencv_kcf_bbox_xywh(bbox_xyxy: list[float]) -> tuple[int, int, int, int]:
    x, y, width, height = xyxy_to_xywh(bbox_xyxy)
    return (
        int(round(x)),
        int(round(y)),
        max(1, int(round(width))),
        max(1, int(round(height))),
    )


def _opencv_kcf_bbox_fallbacks(bbox_xyxy: list[float]) -> list[tuple[object, str]]:
    x, y, width, height = xyxy_to_xywh(bbox_xyxy)
    int_bbox = _opencv_kcf_bbox_xywh(bbox_xyxy)
    float_bbox = (
        float(x),
        float(y),
        float(max(1.0, width)),
        float(max(1.0, height)),
    )
    return [
        (int_bbox, "tuple[int,int,int,int]"),
        (float_bbox, "tuple[float,float,float,float]"),
        (list(int_bbox), "list[int]"),
        (list(float_bbox), "list[float]"),
    ]


@dataclass
class KcfTrackerWrapper:
    tracker: object | None = None
    initialized: bool = False
    last_bbox_xyxy: list[float] | None = None

    def initialize(self, frame, bbox_xyxy: list[float]) -> None:
        frame_height, frame_width = frame.shape[:2]
        clipped = clip_bbox_to_frame(bbox_xyxy, frame_width, frame_height)
        tracker = create_kcf_tracker()
        last_error: Exception | None = None
        ok = False
        for bbox_xywh, _label in _opencv_kcf_bbox_fallbacks(clipped):
            try:
                ok = tracker.init(frame, bbox_xywh)
                last_error = None
                break
            except cv2.error as exc:
                last_error = exc
                tracker = create_kcf_tracker()
        if last_error is not None:
            raise RuntimeError(
                "KCF tracker initialization failed for all bbox formats. "
                f"bbox_xyxy={clipped!r}; attempted_xywh={_opencv_kcf_bbox_fallbacks(clipped)!r}; "
                f"frame_shape={tuple(frame.shape)!r}; last_error={last_error}"
            ) from last_error
        if ok is False:
            raise RuntimeError(
                "KCF tracker initialization returned False. "
                f"bbox_xyxy={clipped!r}; attempted_xywh={_opencv_kcf_bbox_fallbacks(clipped)!r}; "
                f"frame_shape={tuple(frame.shape)!r}"
            )
        self.tracker = tracker
        self.initialized = True
        self.last_bbox_xyxy = clipped

    def update(self, frame) -> tuple[bool, list[float] | None]:
        if not self.initialized or self.tracker is None:
            return False, None
        success, bbox_xywh = self.tracker.update(frame)
        if not success:
            return False, None
        frame_height, frame_width = frame.shape[:2]
        bbox_xyxy = clip_bbox_to_frame(xywh_to_xyxy(list(bbox_xywh)), frame_width, frame_height)
        self.last_bbox_xyxy = bbox_xyxy
        return True, bbox_xyxy

    def reset(self, frame, bbox_xyxy: list[float]) -> None:
        self.initialize(frame, bbox_xyxy)

    def is_initialized(self) -> bool:
        return bool(self.initialized and self.tracker is not None)
