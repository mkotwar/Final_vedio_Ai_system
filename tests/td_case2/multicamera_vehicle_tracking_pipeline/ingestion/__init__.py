"""Input layer for multi-camera local video ingestion."""

from .camera_config import CameraConfig, CameraConfigError, load_camera_configs
from .camera_source import CameraSource, CameraSourceError
from .frame_packet import FramePacket
from .multi_camera_reader import MultiCameraReader, MultiCameraReaderError

__all__ = [
    "CameraConfig",
    "CameraConfigError",
    "CameraSource",
    "CameraSourceError",
    "FramePacket",
    "MultiCameraReader",
    "MultiCameraReaderError",
    "load_camera_configs",
]
