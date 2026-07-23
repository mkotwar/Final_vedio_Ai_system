from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class VehicleDetection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(slots=True)
class DetectionPacket:
    camera_code: str
    camera_name: str
    source_path: Path
    frame_number: int
    video_time_seconds: float
    camera_timestamp: datetime | None
    frame_width: int
    frame_height: int
    detections: list[VehicleDetection]
    inference_time_ms: float
    detector_model: str
    detector_device: str
    source_fps: float | None = None
    frame: Any | None = None
