from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any, Callable

from .tracking_config import TrackingConfig


def _install_lap_shim() -> None:
    if "lap" in sys.modules:
        return
    import numpy as np

    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:  # pragma: no cover
        from ultralytics.utils.ops import linear_sum_assignment  # type: ignore

    lap_module = types.ModuleType("lap")
    lap_module.__version__ = "0.5.12"

    def lapjv(cost_matrix: Any, extend_cost: bool = True, cost_limit: float = np.inf):
        rows, cols = linear_sum_assignment(cost_matrix)
        x = np.full(cost_matrix.shape[0], -1, dtype=int)
        y = np.full(cost_matrix.shape[1], -1, dtype=int)
        total_cost = 0.0
        for row_index, col_index in zip(rows, cols):
            value = float(cost_matrix[row_index, col_index])
            if value <= float(cost_limit):
                x[row_index] = col_index
                y[col_index] = row_index
                total_cost += value
        return total_cost, x, y

    lap_module.lapjv = lapjv
    sys.modules["lap"] = lap_module


class TrackerFactory:
    """Create one native tracker instance per camera code."""

    def __init__(self, config: TrackingConfig, *, tracker_creator: Callable[[TrackingConfig], Any] | None = None) -> None:
        self.config = config
        self._tracker_creator = tracker_creator or self._default_tracker_creator
        self._trackers: dict[str, Any] = {}

    def get_or_create(self, camera_code: str) -> Any:
        tracker = self._trackers.get(camera_code)
        if tracker is None:
            tracker = self._tracker_creator(self.config)
            self._trackers[camera_code] = tracker
        return tracker

    def configured_camera_codes(self) -> tuple[str, ...]:
        return tuple(sorted(self._trackers))

    def reset(self) -> None:
        self._trackers.clear()

    @staticmethod
    def _default_tracker_creator(config: TrackingConfig) -> Any:
        if config.backend == "supervision_bytetrack":
            import supervision as sv  # type: ignore

            return sv.ByteTrack(
                lost_track_buffer=max(1, int(config.track_buffer)),
                track_activation_threshold=float(config.track_high_thresh),
                minimum_matching_threshold=float(config.match_thresh),
                minimum_consecutive_frames=max(1, int(config.min_confirmed_observations)),
            )
        if config.backend == "ultralytics_bytetrack":
            _install_lap_shim()
            from ultralytics.trackers.byte_tracker import BYTETracker  # type: ignore

            args = SimpleNamespace(
                track_high_thresh=float(config.track_high_thresh),
                track_low_thresh=float(config.track_low_thresh),
                new_track_thresh=float(config.new_track_thresh),
                match_thresh=float(config.match_thresh),
                track_buffer=max(1, int(config.track_buffer)),
                fuse_score=True,
                model="manual",
            )
            return BYTETracker(args)
        raise ValueError(f"Unsupported tracking backend: {config.backend}")
