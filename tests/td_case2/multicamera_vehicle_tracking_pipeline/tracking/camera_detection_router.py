from __future__ import annotations

from collections.abc import Iterable

from ..detection.detection_models import DetectionPacket
from .camera_tracker import CameraTracker, CameraTrackingResult
from .track_lifecycle import LocalTrackLifecycle
from .tracker_factory import TrackerFactory
from .tracking_config import TrackingConfig


class CameraDetectionRouter:
    def __init__(
        self,
        tracking_config: TrackingConfig,
        *,
        run_id: str | None = None,
        tracker_factory: TrackerFactory | None = None,
        allowed_camera_codes: Iterable[str] | None = None,
    ) -> None:
        self.tracking_config = tracking_config
        self.tracker_factory = tracker_factory or TrackerFactory(tracking_config)
        self.lifecycle = LocalTrackLifecycle(tracking_config, run_id=run_id)
        self._camera_trackers: dict[str, CameraTracker] = {}
        self._allowed_camera_codes = tuple(allowed_camera_codes or ())

    def route(self, packet: DetectionPacket) -> CameraTrackingResult:
        if self._allowed_camera_codes and packet.camera_code not in self._allowed_camera_codes:
            raise ValueError(f"Detection packet received for unknown or disabled camera: {packet.camera_code}")
        tracker = self._camera_trackers.get(packet.camera_code)
        if tracker is None:
            tracker = CameraTracker(packet.camera_code, self.tracking_config, self.tracker_factory, self.lifecycle)
            self._camera_trackers[packet.camera_code] = tracker
        return tracker.update(packet)

    def flush_camera(self, camera_code: str) -> CameraTrackingResult:
        if self._allowed_camera_codes and camera_code not in self._allowed_camera_codes:
            raise ValueError(f"Flush requested for unknown or disabled camera: {camera_code}")
        tracker = self._camera_trackers.get(camera_code)
        if tracker is None:
            result = self.lifecycle.flush_camera(camera_code)
            return CameraTrackingResult(observations=[], completed_tracks=result.completed_tracks, active_tracks=result.active_tracks)
        return tracker.flush()

    def flush_all(self) -> CameraTrackingResult:
        result = self.lifecycle.flush_all()
        return CameraTrackingResult(observations=[], completed_tracks=result.completed_tracks, active_tracks=result.active_tracks)

    def configured_camera_codes(self) -> tuple[str, ...]:
        return tuple(sorted(self._camera_trackers))

    def diagnostics_by_camera(self) -> dict[str, dict[str, int | float | None]]:
        return {camera_code: tracker.diagnostics() for camera_code, tracker in sorted(self._camera_trackers.items())}

    @property
    def allowed_camera_codes(self) -> tuple[str, ...]:
        return self._allowed_camera_codes
