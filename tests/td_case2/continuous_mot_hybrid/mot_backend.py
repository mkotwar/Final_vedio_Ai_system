from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class BackendDetection:
    detection_id: str
    class_id: int
    class_name: str
    family: str
    confidence: float
    bbox_xyxy: list[float]


@dataclass(frozen=True)
class BackendTrack:
    track_id: str
    family: str
    class_name: str
    bbox_xyxy: list[float]
    confirmed: bool
    age_frames: int
    hits: int
    time_since_update_frames: int
    backend_state: str
    matched_detection_id: str | None
    matched_detection_confidence: float | None
    association_cost: float | None


class ResultsLike:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = xyxy.astype(np.float32)
        self.conf = conf.astype(np.float32)
        self.cls = cls.astype(np.float32)

    @property
    def xywh(self) -> np.ndarray:
        xywh = self.xyxy.copy()
        xywh[:, 2] = xywh[:, 2] - xywh[:, 0]
        xywh[:, 3] = xywh[:, 3] - xywh[:, 1]
        xywh[:, 0] = xywh[:, 0] + (xywh[:, 2] / 2.0)
        xywh[:, 1] = xywh[:, 1] + (xywh[:, 3] / 2.0)
        return xywh

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])

    def __getitem__(self, mask):
        return ResultsLike(self.xyxy[mask], self.conf[mask], self.cls[mask])


class MotBackend(Protocol):
    def update(self, *, detections: list[BackendDetection], frame=None) -> list[BackendTrack]:
        ...

    def handle_detector_skipped(self) -> list[BackendTrack]:
        ...

    def active_tracks(self) -> list[BackendTrack]:
        ...
