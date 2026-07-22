"""Shared vehicle detection for multi-camera frame packets."""

from .annotation import annotate_detection_frame
from .detection_config import DetectionConfig, DetectionConfigError, load_detection_config
from .detection_models import DetectionPacket, VehicleDetection
from .vehicle_detector import SharedVehicleDetector, VehicleDetectorError

__all__ = [
    "annotate_detection_frame",
    "DetectionConfig",
    "DetectionConfigError",
    "DetectionPacket",
    "VehicleDetection",
    "SharedVehicleDetector",
    "VehicleDetectorError",
    "load_detection_config",
]
