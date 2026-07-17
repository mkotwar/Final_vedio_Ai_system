from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - exercised in the isolated td_case2 venv
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


def load_tracker_types():
    _install_lap_shim()
    from ultralytics.trackers.byte_tracker import BYTETracker, TrackState  # type: ignore

    return BYTETracker, TrackState


class ByteTrackBackend(MotBackend):
    def __init__(self, *, track_high_thresh: float, track_low_thresh: float, match_thresh: float, track_buffer_frames: int):
        BYTETracker, TrackState = load_tracker_types()
        self.track_state_enum = TrackState
        args = SimpleNamespace(
            track_high_thresh=float(track_high_thresh),
            track_low_thresh=float(track_low_thresh),
            new_track_thresh=float(track_high_thresh),
            match_thresh=float(match_thresh),
            track_buffer=max(1, int(track_buffer_frames)),
            fuse_score=False,
            model="manual",
        )
        self.trackers = {
            "person": BYTETracker(args),
            "vehicle": BYTETracker(args),
        }
        self._active_tracks: list[BackendTrack] = []
        self._track_labels: dict[str, dict[str, str]] = {}

    def _rows_to_results(self, detections: list[BackendDetection]) -> ResultsLike:
        if not detections:
            return ResultsLike(np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32))
        return ResultsLike(
            np.asarray([item.bbox_xyxy for item in detections], dtype=np.float32),
            np.asarray([item.confidence for item in detections], dtype=np.float32),
            np.asarray([item.class_id for item in detections], dtype=np.float32),
        )

    def _collect_backend_tracks(self, *, family: str, detections: list[BackendDetection], tracker) -> list[BackendTrack]:
        detection_by_index = {index: row for index, row in enumerate(detections)}
        active: list[BackendTrack] = []
        track_groups = [("tracked", list(tracker.tracked_stracks)), ("lost", list(tracker.lost_stracks))]
        for backend_state, track_list in track_groups:
            for track in track_list:
                bbox_xyxy = track.xyxy.tolist() if getattr(track, "xyxy", None) is not None else [0.0, 0.0, 0.0, 0.0]
                detection_row = detection_by_index.get(int(getattr(track, "idx", -1)))
                track_id = f"{family}_track_{int(track.track_id):04d}"
                if detection_row is not None:
                    self._track_labels[track_id] = {
                        "class_name": str(detection_row.class_name),
                        "family": str(detection_row.family),
                    }
                label = self._track_labels.get(track_id, {"class_name": str(getattr(track, "cls", family)), "family": family})
                active.append(
                    BackendTrack(
                        track_id=track_id,
                        family=str(label["family"]),
                        class_name=str(label["class_name"]),
                        bbox_xyxy=[float(value) for value in bbox_xyxy],
                        confirmed=bool(getattr(track, "is_activated", False)),
                        age_frames=int(max(0, getattr(track, "frame_id", 0) - getattr(track, "start_frame", 0) + 1)),
                        hits=int(max(1, getattr(track, "tracklet_len", 0) + 1)),
                        time_since_update_frames=int(max(0, tracker.frame_id - getattr(track, "frame_id", tracker.frame_id))),
                        backend_state=backend_state,
                        matched_detection_id=detection_row.detection_id if detection_row is not None else None,
                        matched_detection_confidence=detection_row.confidence if detection_row is not None else None,
                        association_cost=None,
                    )
                )
        return active

    def update(self, *, detections: list[BackendDetection], frame=None) -> list[BackendTrack]:
        grouped = {
            "person": [item for item in detections if item.family == "person"],
            "vehicle": [item for item in detections if item.family == "vehicle"],
        }
        active_tracks: list[BackendTrack] = []
        for family, tracker in self.trackers.items():
            tracker.update(self._rows_to_results(grouped[family]), img=frame)
            active_tracks.extend(self._collect_backend_tracks(family=family, detections=grouped[family], tracker=tracker))
        self._active_tracks = sorted(active_tracks, key=lambda item: (item.track_id, item.time_since_update_frames, item.backend_state))
        return list(self._active_tracks)

    def handle_detector_skipped(self) -> list[BackendTrack]:
        active_tracks: list[BackendTrack] = []
        for family, tracker in self.trackers.items():
            tracker.multi_predict(list(tracker.tracked_stracks))
            active_tracks.extend(self._collect_backend_tracks(family=family, detections=[], tracker=tracker))
        self._active_tracks = sorted(active_tracks, key=lambda item: (item.track_id, item.time_since_update_frames, item.backend_state))
        return list(self._active_tracks)

    def active_tracks(self) -> list[BackendTrack]:
        return list(self._active_tracks)
