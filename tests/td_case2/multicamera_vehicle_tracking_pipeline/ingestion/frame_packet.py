from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class FramePacket:
    camera_code: str
    camera_name: str
    source_path: Path
    frame_number: int
    source_fps: float
    source_frame_count: int | None
    video_time_seconds: float
    camera_timestamp: datetime | None
    frame: Any
