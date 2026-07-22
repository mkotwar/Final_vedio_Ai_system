"""Independent per-camera tracking for the experimental multi-camera pipeline."""

from .annotation import annotate_tracking_frame
from .camera_detection_router import CameraDetectionRouter
from .camera_tracker import CameraTracker, CameraTrackingResult
from .track_lifecycle import LocalTrackLifecycle, build_track_uuid
from .tracker_factory import TrackerFactory
from .tracking_config import TrackingConfig, TrackingConfigError, load_tracking_config
from .tracking_models import LocalVehicleTrack, TrackObservation

__all__ = [
    "annotate_tracking_frame",
    "CameraDetectionRouter",
    "CameraTracker",
    "CameraTrackingResult",
    "LocalTrackLifecycle",
    "LocalVehicleTrack",
    "TrackObservation",
    "TrackerFactory",
    "TrackingConfig",
    "TrackingConfigError",
    "build_track_uuid",
    "load_tracking_config",
]
