from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
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


def _load_botsort_type():
    _install_lap_shim()
    from ultralytics.trackers.bot_sort import BOTSORT  # type: ignore

    return BOTSORT


def installed_ultralytics_info(site_packages_root: Path) -> dict[str, Any]:
    init_path = site_packages_root / "ultralytics" / "__init__.py"
    version = "unknown"
    for raw_line in init_path.read_text(encoding="utf-8").splitlines():
        if "__version__" in raw_line and "=" in raw_line:
            version = raw_line.split("=", 1)[1].strip().strip("\"'")
            break
    return {
        "ultralytics_version": version,
        "botsort_yaml_path": str(site_packages_root / "ultralytics" / "cfg" / "trackers" / "botsort.yaml"),
        "bytetrack_yaml_path": str(site_packages_root / "ultralytics" / "cfg" / "trackers" / "bytetrack.yaml"),
    }


class _EncoderProbe:
    def __init__(self, encoder: Any, verification: dict[str, Any]):
        self.encoder = encoder
        self.verification = verification

    def __call__(self, source: Any, dets: Any):
        features = self.encoder(source, dets)
        valid = [item for item in features if item is not None]
        self.verification["encoder_initialized"] = True
        self.verification["feature_vector_count"] += len(valid)
        if valid and self.verification.get("feature_dimension") is None:
            first = np.asarray(valid[0])
            self.verification["feature_dimension"] = int(first.shape[-1]) if first.ndim else 1
        return features


class UltralyticsBotSortBackend(MotBackend):
    def __init__(
        self,
        *,
        track_high_thresh: float,
        track_low_thresh: float,
        match_thresh: float,
        track_buffer_frames: int,
        gmc_method: str,
        with_reid: bool,
        model: str,
        device: str | None,
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.8,
        fuse_score: bool = True,
    ):
        BOTSORT = _load_botsort_type()
        self.verification: dict[str, Any] = {
            "requested_with_reid": with_reid,
            "actual_with_reid": False,
            "requested_model": model,
            "resolved_model": model,
            "detector_native_features_used": False,
            "separate_classification_model_used": False,
            "encoder_initialized": False,
            "feature_vector_count": 0,
            "feature_dimension": None,
            "fallback_reason": None,
            "runtime_warnings": [],
            "gpu_device": device,
            "peak_allocated_vram_mb": "not_available",
            "appearance_comparisons": "not_available",
            "accepted_appearance_matches": "not_available",
            "rejected_appearance_matches": "not_available",
            "average_appearance_similarity": "not_available",
            "configured_gmc_method": gmc_method,
            "meaningful_camera_motion_detected": "not_available",
            "gmc_failures": 0,
        }
        args = SimpleNamespace(
            track_high_thresh=float(track_high_thresh),
            track_low_thresh=float(track_low_thresh),
            new_track_thresh=float(track_high_thresh),
            match_thresh=float(match_thresh),
            track_buffer=max(1, int(track_buffer_frames)),
            fuse_score=bool(fuse_score),
            model=str(model),
            gmc_method=str(gmc_method),
            proximity_thresh=float(proximity_thresh),
            appearance_thresh=float(appearance_thresh),
            with_reid=bool(with_reid),
            device=device,
        )
        self.trackers = {
            "person": BOTSORT(args),
            "vehicle": BOTSORT(args),
        }
        self._active_tracks: list[BackendTrack] = []
        self._track_labels: dict[str, dict[str, str]] = {}
        if with_reid:
            self._install_encoder_probes(model=model)

    def _install_encoder_probes(self, *, model: str) -> None:
        any_encoder = False
        for tracker in self.trackers.values():
            encoder = getattr(tracker, "encoder", None)
            if encoder is None:
                continue
            any_encoder = True
            if model == "auto":
                self.verification["detector_native_features_used"] = True
                self.verification["fallback_reason"] = "cached_detection_replay_has_no_native_detector_features_for_model_auto"
            else:
                self.verification["separate_classification_model_used"] = True
                tracker.encoder = _EncoderProbe(encoder, self.verification)
        if not any_encoder:
            self.verification["fallback_reason"] = "reid_encoder_not_initialized"
            return
        if model == "auto":
            self.verification["runtime_warnings"].append(
                "Requested model=auto cannot prove active ReID during cached replay because detector-native features are unavailable."
            )

    def _rows_to_results(self, detections: list[BackendDetection]) -> ResultsLike:
        if not detections:
            return ResultsLike(np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32))
        return ResultsLike(
            np.asarray([item.bbox_xyxy for item in detections], dtype=np.float32),
            np.asarray([item.confidence for item in detections], dtype=np.float32),
            np.asarray([item.class_id for item in detections], dtype=np.float32),
        )

    def update(self, *, detections: list[BackendDetection], frame: Any | None = None) -> list[BackendTrack]:
        grouped = {
            "person": [item for item in detections if item.family == "person"],
            "vehicle": [item for item in detections if item.family == "vehicle"],
        }
        active_tracks: list[BackendTrack] = []
        for family, tracker in self.trackers.items():
            tracker.update(self._rows_to_results(grouped[family]), img=frame, feats=None)
            for backend_state, track_list in (("tracked", list(tracker.tracked_stracks)), ("lost", list(tracker.lost_stracks))):
                for track in track_list:
                    bbox_xyxy = track.xyxy.tolist() if getattr(track, "xyxy", None) is not None else [0.0, 0.0, 0.0, 0.0]
                    detection_row = None
                    idx = int(getattr(track, "idx", -1))
                    if 0 <= idx < len(grouped[family]):
                        detection_row = grouped[family][idx]
                    track_id = f"{family}_track_{int(track.track_id):04d}"
                    if detection_row is not None:
                        self._track_labels[track_id] = {"class_name": detection_row.class_name, "family": detection_row.family}
                    label = self._track_labels.get(track_id, {"class_name": family, "family": family})
                    curr_feat = getattr(track, "curr_feat", None)
                    if curr_feat is not None:
                        self.verification["actual_with_reid"] = True
                    active_tracks.append(
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
        self._active_tracks = sorted(active_tracks, key=lambda item: (item.track_id, item.time_since_update_frames, item.backend_state))
        return list(self._active_tracks)

    def handle_detector_skipped(self) -> list[BackendTrack]:
        active_tracks: list[BackendTrack] = []
        for family, tracker in self.trackers.items():
            tracker.multi_predict(list(tracker.tracked_stracks))
            for backend_state, track_list in (("tracked", list(tracker.tracked_stracks)), ("lost", list(tracker.lost_stracks))):
                for track in track_list:
                    bbox_xyxy = track.xyxy.tolist() if getattr(track, "xyxy", None) is not None else [0.0, 0.0, 0.0, 0.0]
                    track_id = f"{family}_track_{int(track.track_id):04d}"
                    label = self._track_labels.get(track_id, {"class_name": family, "family": family})
                    active_tracks.append(
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
                            matched_detection_id=None,
                            matched_detection_confidence=None,
                            association_cost=None,
                        )
                    )
        self._active_tracks = sorted(active_tracks, key=lambda item: (item.track_id, item.time_since_update_frames, item.backend_state))
        return list(self._active_tracks)

    def active_tracks(self) -> list[BackendTrack]:
        return list(self._active_tracks)

    def write_verification(self, output_path: Path) -> dict[str, Any]:
        payload = dict(self.verification)
        if payload["requested_with_reid"] and not payload["actual_with_reid"] and payload["fallback_reason"] is None:
            payload["fallback_reason"] = "requested_reid_but_no_runtime_features_observed"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload
