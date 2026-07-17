from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def clip_bbox_to_image(bbox_xyxy: list[float], *, image_width: int, image_height: int) -> list[int]:
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox_xyxy]
    x1 = max(0, min(image_width - 1, x1))
    y1 = max(0, min(image_height - 1, y1))
    x2 = max(0, min(image_width, x2))
    y2 = max(0, min(image_height, y2))
    return [x1, y1, x2, y2]


def extract_valid_crop(frame: np.ndarray, bbox_xyxy: list[float], *, min_size: int = 8) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = clip_bbox_to_image(bbox_xyxy, image_width=width, image_height=height)
    if x2 <= x1 or y2 <= y1:
        return None
    if (x2 - x1) < min_size or (y2 - y1) < min_size:
        return None
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else None


def l2_normalize_embedding(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        return array
    return array / norm


@dataclass(frozen=True)
class ExternalEncoderConfig:
    model_path: Path
    device: str


class ExternalReidEncoder:
    def __init__(self, config: ExternalEncoderConfig):
        self.config = config
        self._encoder = None

    def initialize(self) -> None:
        if self._encoder is not None:
            return
        from ultralytics.trackers.utils.reid import ReID

        self._encoder = ReID(str(self.config.model_path), device=self.config.device)

    def encode_crops(self, crops: list[np.ndarray]) -> list[np.ndarray]:
        if self._encoder is None:
            self.initialize()
        embeddings: list[np.ndarray] = []
        for crop in crops:
            if crop is None or not crop.size:
                continue
            resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
            result = self._encoder.model.predictor([resized])[0]
            embeddings.append(l2_normalize_embedding(result.cpu().numpy()))
        return embeddings
