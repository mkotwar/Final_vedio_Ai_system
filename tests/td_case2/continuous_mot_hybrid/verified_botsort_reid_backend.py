from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    from ultralytics.utils.ops import linear_sum_assignment

from .mot_backend import BackendDetection, BackendTrack, ResultsLike


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


class VerifiedBotSortReidBackend:
    def __init__(
        self,
        *,
        track_high_thresh: float,
        track_low_thresh: float,
        match_thresh: float,
        track_buffer_frames: int,
        gmc_method: str,
        model: str,
        device: str | None,
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.8,
        fuse_score: bool = True,
    ):
        BOTSORT = _load_botsort_type()
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
            with_reid=True,
            device=device,
        )
        self.verification: dict[str, Any] = {
            "status": "success",
            "requested_with_reid": True,
            "actual_with_reid": False,
            "requested_model": model,
            "resolved_model": model,
            "native_detector_features_used": model == "auto",
            "external_encoder_used": model != "auto",
            "encoder_initialized": False,
            "feature_vector_count": 0,
            "feature_dimension": 0,
            "appearance_comparison_count": 0,
            "appearance_assisted_accepted_matches": 0,
            "appearance_rejected_matches": 0,
            "ambiguous_appearance_matches": 0,
            "crops_processed": 0,
            "embeddings_generated": 0,
            "invalid_crops_skipped": 0,
            "average_embedding_time_ms": "not_available",
            "total_reid_runtime_seconds": 0.0,
            "gpu_device": device,
            "peak_gpu_vram_mb": "not_available",
            "warnings": [],
            "fallback_reason": None,
        }
        self.trackers = {
            "person": BOTSORT(args),
            "vehicle": BOTSORT(args),
        }
        self._active_tracks: list[BackendTrack] = []
        self._track_labels: dict[str, dict[str, str]] = {}
        self._install_probes()

    def _install_probes(self) -> None:
        from ultralytics.trackers.utils import matching

        for tracker in self.trackers.values():
            if getattr(tracker, "encoder", None) is not None:
                self.verification["encoder_initialized"] = True
            original_get_dists = tracker.get_dists

            def _wrapped_get_dists(this, tracks, detections, _original=original_get_dists):
                if getattr(this.args, "with_reid", False) and getattr(this, "encoder", None) is not None:
                    track_feats = [getattr(track, "smooth_feat", None) for track in tracks]
                    det_feats = [getattr(det, "curr_feat", None) for det in detections]
                    valid_track_indices = [index for index, feat in enumerate(track_feats) if feat is not None]
                    valid_det_indices = [index for index, feat in enumerate(det_feats) if feat is not None]
                    if valid_track_indices and valid_det_indices:
                        this_outer = self.verification
                        this_outer["appearance_comparison_count"] += len(valid_track_indices) * len(valid_det_indices)
                        emb = matching.embedding_distance(tracks, detections) / 2.0
                        accepted = int(np.sum(emb < (1 - float(this.args.appearance_thresh))))
                        rejected = int(np.sum(emb >= (1 - float(this.args.appearance_thresh))))
                        this_outer["appearance_assisted_accepted_matches"] += accepted
                        this_outer["appearance_rejected_matches"] += rejected
                return _original(this._tracks_proxy if hasattr(this, "_tracks_proxy") else tracks, detections)  # pragma: no cover

            # Simpler safe wrapper without altering signature binding quirks.
            def _bound_get_dists(tracks, detections, _tracker=tracker, _original=original_get_dists):
                if getattr(_tracker.args, "with_reid", False) and getattr(_tracker, "encoder", None) is not None:
                    track_feats = [getattr(track, "smooth_feat", None) for track in tracks]
                    det_feats = [getattr(det, "curr_feat", None) for det in detections]
                    valid_track_indices = [index for index, feat in enumerate(track_feats) if feat is not None]
                    valid_det_indices = [index for index, feat in enumerate(det_feats) if feat is not None]
                    if valid_track_indices and valid_det_indices:
                        self.verification["appearance_comparison_count"] += len(valid_track_indices) * len(valid_det_indices)
                        emb = matching.embedding_distance(tracks, detections) / 2.0
                        accepted = int(np.sum(emb < (1 - float(_tracker.args.appearance_thresh))))
                        rejected = int(np.sum(emb >= (1 - float(_tracker.args.appearance_thresh))))
                        self.verification["appearance_assisted_accepted_matches"] += accepted
                        self.verification["appearance_rejected_matches"] += rejected
                return _original(tracks, detections)

            tracker.get_dists = _bound_get_dists  # type: ignore[assignment]

    def _rows_to_results(self, detections: list[BackendDetection]) -> ResultsLike:
        if not detections:
            return ResultsLike(np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32))
        return ResultsLike(
            np.asarray([item.bbox_xyxy for item in detections], dtype=np.float32),
            np.asarray([item.confidence for item in detections], dtype=np.float32),
            np.asarray([item.class_id for item in detections], dtype=np.float32),
        )

    def update(self, *, detections: list[BackendDetection], frame: Any | None = None, feature_vectors: np.ndarray | None = None) -> list[BackendTrack]:
        grouped = {
            "person": [(index, item) for index, item in enumerate(detections) if item.family == "person"],
            "vehicle": [(index, item) for index, item in enumerate(detections) if item.family == "vehicle"],
        }
        active_tracks: list[BackendTrack] = []
        for family, tracker in self.trackers.items():
            family_indexes = [index for index, _ in grouped[family]]
            family_detections = [item for _, item in grouped[family]]
            feats = None
            if feature_vectors is not None and len(family_indexes):
                feats = np.asarray(feature_vectors[family_indexes], dtype=np.float32)
                if feats.ndim == 2 and feats.shape[0] > 0:
                    self.verification["feature_vector_count"] += int(feats.shape[0])
                    self.verification["embeddings_generated"] += int(feats.shape[0])
                    self.verification["feature_dimension"] = max(int(self.verification["feature_dimension"]), int(feats.shape[1]))
            tracker.update(self._rows_to_results(family_detections), img=frame, feats=feats)
            for backend_state, track_list in (("tracked", list(tracker.tracked_stracks)), ("lost", list(tracker.lost_stracks))):
                for track in track_list:
                    bbox_xyxy = track.xyxy.tolist() if getattr(track, "xyxy", None) is not None else [0.0, 0.0, 0.0, 0.0]
                    detection_row = None
                    idx = int(getattr(track, "idx", -1))
                    if 0 <= idx < len(family_detections):
                        detection_row = family_detections[idx]
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

    def write_verification(self, output_path: Path) -> dict[str, Any]:
        if self.verification["requested_with_reid"] and not self.verification["encoder_initialized"]:
            self.verification["status"] = "reid_not_active"
            self.verification["fallback_reason"] = self.verification["fallback_reason"] or "encoder_not_initialized"
        elif self.verification["feature_vector_count"] <= 0 or self.verification["feature_dimension"] <= 0 or self.verification["appearance_comparison_count"] <= 0:
            self.verification["status"] = "reid_not_active"
            self.verification["actual_with_reid"] = False
            self.verification["fallback_reason"] = self.verification["fallback_reason"] or "missing_runtime_reid_evidence"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.verification, indent=2, ensure_ascii=False), encoding="utf-8")
        return dict(self.verification)
