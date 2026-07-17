from __future__ import annotations

import importlib.util
import sys
import types
from types import SimpleNamespace

import numpy as np
from ultralytics.utils.ops import linear_sum_assignment

from .mot_backend import BackendDetection, BackendTrack, MotBackend, ResultsLike


def _install_lap_shim() -> None:
    if "lap" in sys.modules:
        return
    lap_module = types.ModuleType("lap")
    lap_module.__version__ = "0.5.12"

    def lapjv(cost_matrix: np.ndarray, extend_cost: bool = True, cost_limit: float = np.inf):
        rows, cols = linear_sum_assignment(cost_matrix)
        x = np.full(cost_matrix.shape[0], -1, dtype=int)
        y = np.full(cost_matrix.shape[1], -1, dtype=int)
        total_cost = 0.0
        for row_index, col_index in zip(rows, cols):
            cost = float(cost_matrix[row_index, col_index])
            if cost <= float(cost_limit):
                x[row_index] = col_index
                y[col_index] = row_index
                total_cost += cost
        return total_cost, x, y

    lap_module.lapjv = lapjv
    sys.modules["lap"] = lap_module


def load_botsort_type():
    _install_lap_shim()
    import ultralytics

    trackers_dir = importlib.util.find_spec("ultralytics.trackers")
    if trackers_dir is None or trackers_dir.submodule_search_locations is None:
        raise ImportError("Unable to resolve ultralytics.trackers package for BOTSORT.")
    from ultralytics.trackers.bot_sort import BOTSORT  # type: ignore

    return BOTSORT


class BotSortBackend(MotBackend):
    def __init__(self, *, track_high_thresh: float, track_low_thresh: float, match_thresh: float, track_buffer_frames: int):
        BOTSORT = load_botsort_type()
        args = SimpleNamespace(
            track_high_thresh=float(track_high_thresh),
            track_low_thresh=float(track_low_thresh),
            new_track_thresh=float(track_high_thresh),
            match_thresh=float(match_thresh),
            track_buffer=max(1, int(track_buffer_frames)),
            fuse_score=False,
            model="manual",
            gmc_method="none",
            proximity_thresh=0.5,
            appearance_thresh=0.25,
            with_reid=False,
            device=None,
        )
        self.trackers = {
            "person": BOTSORT(args),
            "vehicle": BOTSORT(args),
        }
        self._active_tracks: list[BackendTrack] = []

    def _rows_to_results(self, detections: list[BackendDetection]) -> ResultsLike:
        if not detections:
            return ResultsLike(np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32))
        return ResultsLike(
            np.asarray([item.bbox_xyxy for item in detections], dtype=np.float32),
            np.asarray([item.confidence for item in detections], dtype=np.float32),
            np.asarray([item.class_id for item in detections], dtype=np.float32),
        )

    def update(self, *, detections: list[BackendDetection], frame=None) -> list[BackendTrack]:
        grouped = {
            "person": [item for item in detections if item.family == "person"],
            "vehicle": [item for item in detections if item.family == "vehicle"],
        }
        active_tracks: list[BackendTrack] = []
        for family, tracker in self.trackers.items():
            tracker.update(self._rows_to_results(grouped[family]), img=frame)
            for track in list(tracker.tracked_stracks) + list(tracker.lost_stracks):
                bbox_xyxy = track.xyxy.tolist() if getattr(track, "xyxy", None) is not None else [0.0, 0.0, 0.0, 0.0]
                active_tracks.append(
                    BackendTrack(
                        track_id=f"{family}_track_{int(track.track_id):04d}",
                        family=family,
                        class_name=str(getattr(track, "cls", family)),
                        bbox_xyxy=[float(value) for value in bbox_xyxy],
                        confirmed=bool(getattr(track, "is_activated", False)),
                        age_frames=int(max(0, getattr(track, "frame_id", 0) - getattr(track, "start_frame", 0) + 1)),
                        hits=int(max(1, getattr(track, "tracklet_len", 0) + 1)),
                        time_since_update_frames=int(max(0, tracker.frame_id - getattr(track, "frame_id", tracker.frame_id))),
                        backend_state="tracked" if track in tracker.tracked_stracks else "lost",
                        matched_detection_id=None,
                        matched_detection_confidence=None,
                        association_cost=None,
                    )
                )
        self._active_tracks = sorted(active_tracks, key=lambda item: (item.track_id, item.backend_state))
        return list(self._active_tracks)

    def handle_detector_skipped(self) -> list[BackendTrack]:
        active_tracks: list[BackendTrack] = []
        for family, tracker in self.trackers.items():
            tracker.multi_predict(list(tracker.tracked_stracks))
            for track in list(tracker.tracked_stracks) + list(tracker.lost_stracks):
                bbox_xyxy = track.xyxy.tolist() if getattr(track, "xyxy", None) is not None else [0.0, 0.0, 0.0, 0.0]
                active_tracks.append(
                    BackendTrack(
                        track_id=f"{family}_track_{int(track.track_id):04d}",
                        family=family,
                        class_name=str(getattr(track, "cls", family)),
                        bbox_xyxy=[float(value) for value in bbox_xyxy],
                        confirmed=bool(getattr(track, "is_activated", False)),
                        age_frames=int(max(0, getattr(track, "frame_id", 0) - getattr(track, "start_frame", 0) + 1)),
                        hits=int(max(1, getattr(track, "tracklet_len", 0) + 1)),
                        time_since_update_frames=int(max(0, tracker.frame_id - getattr(track, "frame_id", tracker.frame_id))),
                        backend_state="tracked" if track in tracker.tracked_stracks else "lost",
                        matched_detection_id=None,
                        matched_detection_confidence=None,
                        association_cost=None,
                    )
                )
        self._active_tracks = sorted(active_tracks, key=lambda item: (item.track_id, item.backend_state))
        return list(self._active_tracks)

    def active_tracks(self) -> list[BackendTrack]:
        return list(self._active_tracks)
