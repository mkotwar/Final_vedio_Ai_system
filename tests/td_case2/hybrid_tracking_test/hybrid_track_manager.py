from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .box_validation import bbox_area, bbox_aspect_ratio, bbox_center, bbox_iou
from .data_models import DetectionObservation, EventRecord, HybridTrack, ValidationResult
from .kcf_tracker_wrapper import KcfTrackerWrapper

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
except Exception:  # pragma: no cover - optional
    linear_sum_assignment = None


VEHICLE_GROUP = {"car", "truck", "bus", "motorcycle", "vehicle", "van", "auto", "bicycle"}


@dataclass
class AssociationDecision:
    detection_index: int
    track_id: int | None
    result: str
    matching_stage: str
    iou: float | None = None
    center_distance_ratio: float | None = None
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection_index": int(self.detection_index),
            "track_id": self.track_id,
            "result": str(self.result),
            "matching_stage": str(self.matching_stage),
            "iou": None if self.iou is None else round(float(self.iou), 6),
            "center_distance_ratio": None if self.center_distance_ratio is None else round(float(self.center_distance_ratio), 6),
            "details": dict(self.details or {}),
        }


def _object_family(class_name: str) -> str:
    lowered = str(class_name).lower()
    if lowered == "person":
        return "person"
    if lowered in VEHICLE_GROUP:
        return "vehicle"
    return "other"


def _class_compatible(track_class: str, detection_class: str, *, vehicle_compatibility_enabled: bool) -> bool:
    left_family = _object_family(track_class)
    right_family = _object_family(detection_class)
    if left_family != right_family:
        return False
    if track_class == detection_class:
        return True
    if left_family == "vehicle":
        return bool(vehicle_compatibility_enabled)
    return False


def _normalized_center_distance(track_box: list[float], detection_box: list[float]) -> float:
    track_center = bbox_center(track_box)
    detection_center = bbox_center(detection_box)
    previous_diagonal = math.sqrt(max(bbox_area(track_box), 1.0))
    return math.dist(track_center, detection_center) / max(previous_diagonal, 1.0)


def _normalized_center_distance_to_point(track_box: list[float], center_xy: tuple[float, float]) -> float:
    track_center = bbox_center(track_box)
    previous_diagonal = math.sqrt(max(bbox_area(track_box), 1.0))
    return math.dist(track_center, center_xy) / max(previous_diagonal, 1.0)


def _area_ratio(track_box: list[float], detection_box: list[float]) -> float:
    return bbox_area(detection_box) / max(bbox_area(track_box), 1.0)


def _aspect_ratio_change(track_box: list[float], detection_box: list[float]) -> float:
    return bbox_aspect_ratio(detection_box) / max(bbox_aspect_ratio(track_box), 1e-6)


def _compute_histogram(frame, bbox_xyxy: list[float]) -> list[float] | None:
    x1, y1, x2, y2 = [int(round(value)) for value in bbox_xyxy]
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [12, 12], [0, 180, 0, 256])
    histogram = cv2.normalize(histogram, histogram).flatten()
    return [float(value) for value in histogram.tolist()]


def _blend_histogram(previous_histogram: list[float] | None, new_histogram: list[float] | None, alpha: float) -> list[float] | None:
    if new_histogram is None:
        return previous_histogram
    if previous_histogram is None:
        return list(new_histogram)
    return [
        ((float(alpha) * float(new_value)) + ((1.0 - float(alpha)) * float(old_value)))
        for old_value, new_value in zip(previous_histogram, new_histogram)
    ]


def _appearance_similarity(left_histogram: list[float] | None, right_histogram: list[float] | None) -> float | None:
    if not left_histogram or not right_histogram:
        return None
    if len(left_histogram) != len(right_histogram):
        return None
    left_array = np.asarray(left_histogram, dtype=np.float32).reshape(-1, 1)
    right_array = np.asarray(right_histogram, dtype=np.float32).reshape(-1, 1)
    score = cv2.compareHist(left_array, right_array, cv2.HISTCMP_CORREL)
    return float(score)


def greedy_assignment(cost_matrix: list[list[float]], max_cost: float) -> list[tuple[int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for row_index, row in enumerate(cost_matrix):
        for col_index, cost in enumerate(row):
            if cost <= max_cost:
                candidates.append((float(cost), row_index, col_index))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    assignments: list[tuple[int, int]] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    for _cost, row_index, col_index in candidates:
        if row_index in used_rows or col_index in used_cols:
            continue
        assignments.append((row_index, col_index))
        used_rows.add(row_index)
        used_cols.add(col_index)
    return assignments


class HybridTrackManager:
    def __init__(self, config):
        self.config = config
        self.active_tracks: dict[int, HybridTrack] = {}
        self.lost_tracks: dict[int, HybridTrack] = {}
        self.completed_tracks: dict[int, HybridTrack] = {}
        self.events: list[EventRecord] = []
        self.next_track_id = 1
        self.counters: Counter[str] = Counter()

    def active_track_list(self) -> list[HybridTrack]:
        return [track for track in self.active_tracks.values() if track.is_active]

    def _emit_event(self, *, timestamp_seconds: float, source_frame_index: int, event_type: str, track_id: int | None = None, details: dict[str, Any] | None = None) -> None:
        self.events.append(
            EventRecord(
                timestamp_seconds=timestamp_seconds,
                source_frame_index=source_frame_index,
                event_type=event_type,
                track_id=track_id,
                details=details or {},
            )
        )

    def _initialize_kcf(self, track: HybridTrack, frame, bbox_xyxy: list[float]) -> None:
        wrapper = KcfTrackerWrapper()
        wrapper.initialize(frame, bbox_xyxy)
        track.kcf_instance = wrapper
        track.kcf_initialized = True

    def _make_track_payload(self, track: HybridTrack, *, processed_frame_index: int, timestamp_seconds: float, validation: ValidationResult | None = None) -> dict[str, Any]:
        payload = {
            "track_id": track.track_id,
            "class_id": track.class_id,
            "class_name": track.class_name,
            "object_family": track.object_family,
            "bbox_xyxy": [round(float(value), 3) for value in track.bbox_xyxy],
            "bbox_source": track.bbox_source,
            "status": track.status,
            "kcf_success": track.kcf_success,
            "frames_since_detection": track.frames_since_detection(processed_frame_index),
            "seconds_since_detection": round(track.seconds_since_detection(timestamp_seconds), 6),
            "last_detection_confidence": None if track.last_detection_confidence is None else round(float(track.last_detection_confidence), 6),
            "reactivation_count": int(track.reactivation_count),
            "validation": validation.to_dict() if validation is not None else {"valid": True, "reasons": [], "metrics": {}},
        }
        return payload

    def _create_track(self, detection: DetectionObservation, frame) -> HybridTrack:
        track = HybridTrack(
            track_id=self.next_track_id,
            class_id=detection.class_id,
            class_name=detection.class_name,
            bbox_xyxy=list(detection.bbox_xyxy),
            previous_bbox_xyxy=None,
            bbox_source="yolo",
            last_update_source="yolo",
            created_frame_index=detection.source_frame_index,
            created_timestamp_seconds=detection.timestamp_seconds,
            last_update_frame_index=detection.source_frame_index,
            last_update_timestamp_seconds=detection.timestamp_seconds,
            last_detection_frame_index=detection.processed_frame_index,
            last_detection_timestamp_seconds=detection.timestamp_seconds,
            last_detection_confidence=detection.confidence,
            object_family=_object_family(detection.class_name),
            age_frames=1,
            detection_hits=1,
            status="tentative",
            is_confirmed=False,
            is_active=True,
            creation_reason="unmatched_yolo_detection",
            last_valid_kcf_timestamp_seconds=detection.timestamp_seconds,
            class_votes={str(detection.class_name).lower(): float(detection.confidence)},
        )
        track.append_trajectory(
            source_frame_index=detection.source_frame_index,
            processed_frame_index=detection.processed_frame_index,
            timestamp_seconds=detection.timestamp_seconds,
            bbox_xyxy=detection.bbox_xyxy,
            bbox_source="yolo",
            limit=self.config.trajectory_history_limit,
        )
        histogram = _compute_histogram(frame, detection.bbox_xyxy)
        track.appearance_histogram = histogram
        self._initialize_kcf(track, frame, detection.bbox_xyxy)
        self.next_track_id += 1
        self.counters["tracks_created"] += 1
        self.counters["kcf_initialization_count"] += 1
        self._emit_event(
            timestamp_seconds=detection.timestamp_seconds,
            source_frame_index=detection.source_frame_index,
            event_type="track_created",
            track_id=track.track_id,
            details={"class_name": detection.class_name, "object_family": track.object_family},
        )
        self._emit_event(
            timestamp_seconds=detection.timestamp_seconds,
            source_frame_index=detection.source_frame_index,
            event_type="kcf_initialized",
            track_id=track.track_id,
            details={},
        )
        return track

    def _promote_if_confirmed(self, track: HybridTrack, detection: DetectionObservation) -> None:
        if not track.is_confirmed and track.detection_hits >= self.config.minimum_track_hits:
            track.is_confirmed = True
            track.status = "confirmed"
            self.counters["tracks_confirmed"] += 1
            self._emit_event(
                timestamp_seconds=detection.timestamp_seconds,
                source_frame_index=detection.source_frame_index,
                event_type="track_confirmed",
                track_id=track.track_id,
                details={"detection_hits": track.detection_hits},
            )

    def _move_to_lost(self, track: HybridTrack, *, timestamp_seconds: float, source_frame_index: int, reason: str) -> None:
        track.is_active = False
        track.status = "temporarily_lost"
        track.lost_timestamp_seconds = timestamp_seconds
        track.lost_reason = reason
        self.active_tracks.pop(track.track_id, None)
        self.lost_tracks[track.track_id] = track
        self.counters["tracks_temporarily_lost"] += 1
        self._emit_event(
            timestamp_seconds=timestamp_seconds,
            source_frame_index=source_frame_index,
            event_type="track_temporarily_lost",
            track_id=track.track_id,
            details={"reason": reason, "missed_detection_refreshes": track.missed_detection_refreshes},
        )

    def _complete_track(self, track: HybridTrack, *, timestamp_seconds: float, source_frame_index: int, reason: str) -> None:
        track.is_active = False
        track.status = "completed"
        track.termination_reason = reason
        self.active_tracks.pop(track.track_id, None)
        self.lost_tracks.pop(track.track_id, None)
        self.completed_tracks[track.track_id] = track
        self.counters["tracks_removed"] += 1
        self._emit_event(
            timestamp_seconds=timestamp_seconds,
            source_frame_index=source_frame_index,
            event_type="track_completed",
            track_id=track.track_id,
            details={"reason": reason},
        )

    def _flush_expired_lost_tracks(self, *, timestamp_seconds: float, source_frame_index: int) -> None:
        expired_ids = [
            track_id
            for track_id, track in self.lost_tracks.items()
            if track.lost_timestamp_seconds is not None
            and (float(timestamp_seconds) - float(track.lost_timestamp_seconds)) > float(self.config.lost_track_recovery_seconds)
        ]
        for track_id in expired_ids:
            self._complete_track(self.lost_tracks[track_id], timestamp_seconds=timestamp_seconds, source_frame_index=source_frame_index, reason="lost_recovery_expired")

    def active_track_boxes(self) -> list[list[float]]:
        return [list(track.bbox_xyxy) for track in self.active_track_list()]

    def update_kcf_tracks(
        self,
        *,
        frame,
        source_frame_index: int,
        processed_frame_index: int,
        timestamp_seconds: float,
        validate_fn,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        track_payloads: list[dict[str, Any]] = []
        refresh_reasons: list[str] = []
        for track in self.active_track_list():
            wrapper = track.kcf_instance
            success = bool(wrapper and wrapper.is_initialized())
            bbox_xyxy: list[float] | None = None
            if success and wrapper is not None:
                success, bbox_xyxy = wrapper.update(frame)
            track.kcf_success = success
            if not success or bbox_xyxy is None:
                track.consecutive_kcf_failures += 1
                track.consecutive_unreliable_updates += 1
                track.last_propagated_box_valid = False
                self.counters["kcf_failure_count"] += 1
                refresh_reasons.append("kcf_tracker_failed")
                validation = ValidationResult(valid=False, reasons=["kcf_update_failed"], metrics={})
                self._emit_event(
                    timestamp_seconds=timestamp_seconds,
                    source_frame_index=source_frame_index,
                    event_type="kcf_failed",
                    track_id=track.track_id,
                    details={"consecutive_kcf_failures": track.consecutive_kcf_failures},
                )
            else:
                validation = validate_fn(current_bbox_xyxy=bbox_xyxy, previous_bbox_xyxy=track.bbox_xyxy)
                if validation.valid:
                    track.previous_bbox_xyxy = list(track.bbox_xyxy)
                    track.bbox_xyxy = list(bbox_xyxy)
                    track.bbox_source = "kcf"
                    track.last_update_source = "kcf"
                    track.last_update_frame_index = source_frame_index
                    track.last_update_timestamp_seconds = timestamp_seconds
                    track.last_valid_kcf_timestamp_seconds = timestamp_seconds
                    track.propagation_hits += 1
                    track.age_frames += 1
                    track.consecutive_kcf_failures = 0
                    track.consecutive_unreliable_updates = 0
                    track.last_propagated_box_valid = True
                    track.status = "propagated" if track.is_confirmed else "propagated_unconfirmed"
                    track.append_trajectory(
                        source_frame_index=source_frame_index,
                        processed_frame_index=processed_frame_index,
                        timestamp_seconds=timestamp_seconds,
                        bbox_xyxy=track.bbox_xyxy,
                        bbox_source="kcf",
                        limit=self.config.trajectory_history_limit,
                    )
                else:
                    track.consecutive_kcf_failures += 1
                    track.consecutive_unreliable_updates += 1
                    track.last_propagated_box_valid = False
                    refresh_reasons.append("invalid_kcf_box_detected")
                    self.counters["invalid_box_count"] += 1
                    self._emit_event(
                        timestamp_seconds=timestamp_seconds,
                        source_frame_index=source_frame_index,
                        event_type="invalid_propagated_box",
                        track_id=track.track_id,
                        details=validation.to_dict(),
                    )
            seconds_since_detection = track.seconds_since_detection(timestamp_seconds)
            track.max_seconds_without_detection = max(track.max_seconds_without_detection, seconds_since_detection)
            if track.seconds_since_valid_update(timestamp_seconds) > self.config.maximum_track_idle_seconds and not track.last_propagated_box_valid:
                refresh_reasons.append("stale_track_requires_refresh")
            track_payloads.append(
                self._make_track_payload(
                    track,
                    processed_frame_index=processed_frame_index,
                    timestamp_seconds=timestamp_seconds,
                    validation=validation,
                )
            )
        return track_payloads, list(dict.fromkeys(refresh_reasons))

    def _assignment_for_pairs(
        self,
        *,
        track_ids: list[int],
        detections: list[DetectionObservation],
        allow_pair,
    ) -> tuple[list[tuple[int, int]], dict[tuple[int, int], dict[str, float]]]:
        cost_matrix: list[list[float]] = []
        valid_pairs: dict[tuple[int, int], dict[str, float]] = {}
        for track_id in track_ids:
            track = self.active_tracks[track_id]
            row: list[float] = []
            for detection_index, detection in enumerate(detections):
                pair_metrics = allow_pair(track, detection)
                if pair_metrics is None:
                    row.append(1e6)
                    continue
                row.append(float(pair_metrics["cost"]))
                valid_pairs[(track_id, detection_index)] = pair_metrics
            cost_matrix.append(row)
        assignments: list[tuple[int, int]] = []
        if linear_sum_assignment is not None and cost_matrix:
            import numpy as np

            matrix = np.asarray(cost_matrix, dtype=float)
            rows, cols = linear_sum_assignment(matrix)
            for row_index, col_index in zip(rows.tolist(), cols.tolist()):
                if matrix[row_index, col_index] >= 1e5:
                    continue
                assignments.append((track_ids[row_index], col_index))
        else:
            assignments = greedy_assignment(cost_matrix, max_cost=0.9999)
            assignments = [(track_ids[row_index], col_index) for row_index, col_index in assignments]
        return assignments, valid_pairs

    def _assign_detections(
        self,
        *,
        detections: list[DetectionObservation],
    ) -> tuple[list[tuple[int, int, str, float, float]], list[int], list[int], list[AssociationDecision]]:
        track_ids = [track.track_id for track in self.active_track_list()]
        if not track_ids or not detections:
            return [], track_ids, list(range(len(detections))), []

        matches: list[tuple[int, int, str, float, float]] = []
        decisions: list[AssociationDecision] = []
        matched_track_ids: set[int] = set()
        matched_detection_indexes: set[int] = set()

        def run_stage(stage_name: str, candidate_track_ids: list[int], allow_pair) -> None:
            nonlocal matches, decisions, matched_track_ids, matched_detection_indexes
            stage_assignments, valid_pairs = self._assignment_for_pairs(
                track_ids=candidate_track_ids,
                detections=detections,
                allow_pair=allow_pair,
            )
            for track_id, detection_index in stage_assignments:
                if track_id in matched_track_ids or detection_index in matched_detection_indexes:
                    continue
                pair_metrics = valid_pairs[(track_id, detection_index)]
                matched_track_ids.add(track_id)
                matched_detection_indexes.add(detection_index)
                matches.append(
                    (
                        track_id,
                        detection_index,
                        stage_name,
                        float(pair_metrics.get("iou", 0.0)),
                        float(pair_metrics.get("center_distance_ratio", 0.0)),
                    )
                )
                decisions.append(
                    AssociationDecision(
                        detection_index=detection_index,
                        track_id=track_id,
                        result="matched_existing",
                        matching_stage=stage_name,
                        iou=float(pair_metrics.get("iou", 0.0)),
                        center_distance_ratio=float(pair_metrics.get("center_distance_ratio", 0.0)),
                        details={key: round(float(value), 6) for key, value in pair_metrics.items() if key not in {"cost", "iou", "center_distance_ratio"}},
                    )
                )

        run_stage(
            "same_class_iou",
            track_ids,
            lambda track, detection: (
                None
                if str(track.class_name).lower() != str(detection.class_name).lower()
                else (
                    None
                    if bbox_iou(track.bbox_xyxy, detection.bbox_xyxy) < self.config.minimum_iou_match
                    else {
                        "cost": 1.0 - bbox_iou(track.bbox_xyxy, detection.bbox_xyxy),
                        "iou": bbox_iou(track.bbox_xyxy, detection.bbox_xyxy),
                        "center_distance_ratio": _normalized_center_distance(track.bbox_xyxy, detection.bbox_xyxy),
                    }
                )
            ),
        )

        run_stage(
            "compatible_class_iou",
            [track_id for track_id in track_ids if track_id not in matched_track_ids],
            lambda track, detection: (
                None
                if str(track.class_name).lower() == str(detection.class_name).lower()
                or not _class_compatible(track.class_name, detection.class_name, vehicle_compatibility_enabled=self.config.vehicle_class_compatibility_enabled)
                else (
                    None
                    if bbox_iou(track.bbox_xyxy, detection.bbox_xyxy) < self.config.minimum_iou_match
                    else {
                        "cost": 1.0 - bbox_iou(track.bbox_xyxy, detection.bbox_xyxy),
                        "iou": bbox_iou(track.bbox_xyxy, detection.bbox_xyxy),
                        "center_distance_ratio": _normalized_center_distance(track.bbox_xyxy, detection.bbox_xyxy),
                    }
                )
            ),
        )

        remaining_track_ids = [track_id for track_id in track_ids if track_id not in matched_track_ids]
        remaining_detection_indexes = [index for index in range(len(detections)) if index not in matched_detection_indexes]
        stage3_pairs: list[tuple[float, int, int, float, float, float, float | None]] = []
        for track_id in remaining_track_ids:
            track = self.active_tracks[track_id]
            predicted_center = track.predicted_center(track.last_update_timestamp_seconds + (1.0 / max(self.config.processing_fps, 1.0)))
            velocity_x, velocity_y = track.estimated_velocity()
            for detection_index in remaining_detection_indexes:
                detection = detections[detection_index]
                if not _class_compatible(track.class_name, detection.class_name, vehicle_compatibility_enabled=self.config.vehicle_class_compatibility_enabled):
                    continue
                center_distance = _normalized_center_distance_to_point(track.bbox_xyxy, bbox_center(detection.bbox_xyxy))
                predicted_distance = _normalized_center_distance_to_point(track.bbox_xyxy, predicted_center)
                area_ratio = _area_ratio(track.bbox_xyxy, detection.bbox_xyxy)
                aspect_ratio = _aspect_ratio_change(track.bbox_xyxy, detection.bbox_xyxy)
                if center_distance > self.config.maximum_center_jump_diagonals:
                    continue
                if area_ratio < self.config.minimum_area_ratio_change or area_ratio > self.config.maximum_area_ratio_change:
                    continue
                if aspect_ratio < self.config.minimum_aspect_ratio_change or aspect_ratio > self.config.maximum_aspect_ratio_change:
                    continue
                current_center = bbox_center(track.bbox_xyxy)
                detection_center = bbox_center(detection.bbox_xyxy)
                direction_penalty = 0.0
                displacement_x = detection_center[0] - current_center[0]
                displacement_y = detection_center[1] - current_center[1]
                if abs(velocity_x) > 1.0 and (velocity_x * displacement_x) < -5.0:
                    direction_penalty += 0.25
                if abs(velocity_y) > 1.0 and (velocity_y * displacement_y) < -5.0:
                    direction_penalty += 0.25
                new_histogram = _compute_histogram(getattr(detection, "_frame", None) or detection, detection.bbox_xyxy) if False else None
                appearance_similarity = None
                score = 1.0 - min(center_distance / max(self.config.maximum_center_jump_diagonals, 1e-6), 1.0)
                score += min(1.0, bbox_iou(track.bbox_xyxy, detection.bbox_xyxy) + 0.10)
                score += min(1.0, float(detection.confidence))
                score -= direction_penalty
                stage3_pairs.append((score, track_id, detection_index, center_distance, area_ratio, aspect_ratio, appearance_similarity))
        stage3_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
        for score, track_id, detection_index, center_distance, area_ratio, aspect_ratio, appearance_similarity in stage3_pairs:
            if track_id in matched_track_ids or detection_index in matched_detection_indexes:
                continue
            if score < 0.90:
                continue
            matched_track_ids.add(track_id)
            matched_detection_indexes.add(detection_index)
            iou_value = bbox_iou(self.active_tracks[track_id].bbox_xyxy, detections[detection_index].bbox_xyxy)
            matches.append((track_id, detection_index, "geometry", iou_value, center_distance))
            decisions.append(
                AssociationDecision(
                    detection_index=detection_index,
                    track_id=track_id,
                    result="matched_existing",
                    matching_stage="geometry",
                    iou=iou_value,
                    center_distance_ratio=center_distance,
                    details={
                        "area_ratio": round(area_ratio, 6),
                        "aspect_ratio_change": round(aspect_ratio, 6),
                        "appearance_similarity": None if appearance_similarity is None else round(appearance_similarity, 6),
                    },
                )
            )

        unmatched_track_ids = [track_id for track_id in track_ids if track_id not in matched_track_ids]
        unmatched_detection_indexes = [index for index in range(len(detections)) if index not in matched_detection_indexes]
        return matches, unmatched_track_ids, unmatched_detection_indexes, decisions

    def _apply_detection_update(
        self,
        *,
        track: HybridTrack,
        detection: DetectionObservation,
        frame,
        source_frame_index: int,
        processed_frame_index: int,
        timestamp_seconds: float,
        matching_stage: str,
        reactivated: bool = False,
        center_distance_ratio: float | None = None,
        area_ratio: float | None = None,
        appearance_similarity: float | None = None,
    ) -> None:
        previous_bbox = list(track.bbox_xyxy)
        track.previous_bbox_xyxy = previous_bbox
        track.bbox_xyxy = list(detection.bbox_xyxy)
        track.class_id = detection.class_id
        track.object_family = _object_family(detection.class_name)
        if self.config.class_vote_history_enabled:
            track.update_class_vote(detection.class_name, detection.confidence)
        else:
            track.class_name = detection.class_name
            track.class_votes = {str(detection.class_name).lower(): float(detection.confidence)}
        track.bbox_source = "yolo"
        track.last_update_source = "yolo"
        track.last_update_frame_index = source_frame_index
        track.last_update_timestamp_seconds = timestamp_seconds
        track.last_detection_frame_index = processed_frame_index
        track.last_detection_timestamp_seconds = timestamp_seconds
        track.last_detection_confidence = detection.confidence
        track.last_valid_kcf_timestamp_seconds = timestamp_seconds
        track.detection_hits += 1
        track.age_frames += 1
        track.missed_detection_refreshes = 0
        track.consecutive_unreliable_updates = 0
        track.kcf_success = True
        track.last_propagated_box_valid = True
        track.status = "reactivated" if reactivated else ("confirmed" if track.is_confirmed else "tentative")
        track.append_trajectory(
            source_frame_index=source_frame_index,
            processed_frame_index=processed_frame_index,
            timestamp_seconds=timestamp_seconds,
            bbox_xyxy=track.bbox_xyxy,
            bbox_source="yolo",
            limit=self.config.trajectory_history_limit,
        )
        histogram = _compute_histogram(frame, detection.bbox_xyxy)
        track.appearance_histogram = _blend_histogram(track.appearance_histogram, histogram, self.config.appearance_alpha)
        if track.kcf_instance is None:
            track.kcf_instance = KcfTrackerWrapper()
        track.kcf_instance.reset(frame, detection.bbox_xyxy)
        track.kcf_initialized = True
        self.counters["kcf_reinitialization_count"] += 1
        self._promote_if_confirmed(track, detection)
        if reactivated:
            track.reactivation_count += 1
            self.counters["tracks_reactivated"] += 1
            self._emit_event(
                timestamp_seconds=timestamp_seconds,
                source_frame_index=source_frame_index,
                event_type="track_reactivated",
                track_id=track.track_id,
                details={
                    "lost_duration_seconds": round(max(0.0, timestamp_seconds - float(track.lost_timestamp_seconds or timestamp_seconds)), 6),
                    "matching_stage": matching_stage,
                    "normalized_center_distance": None if center_distance_ratio is None else round(center_distance_ratio, 6),
                    "area_ratio": None if area_ratio is None else round(area_ratio, 6),
                    "appearance_similarity": None if appearance_similarity is None else round(appearance_similarity, 6),
                },
            )
        self._emit_event(
            timestamp_seconds=timestamp_seconds,
            source_frame_index=source_frame_index,
            event_type="track_corrected_by_yolo",
            track_id=track.track_id,
            details={
                "previous_bbox_source": track.last_update_source,
                "association_iou": round(float(bbox_iou(previous_bbox, track.bbox_xyxy)), 6),
                "center_correction_pixels": round(math.dist(bbox_center(previous_bbox), bbox_center(track.bbox_xyxy)), 6),
                "matching_stage": matching_stage,
            },
        )

    def _recover_lost_track(
        self,
        *,
        detection: DetectionObservation,
        frame,
        source_frame_index: int,
        processed_frame_index: int,
        timestamp_seconds: float,
    ) -> tuple[HybridTrack | None, dict[str, Any] | None]:
        best_match: tuple[float, HybridTrack, dict[str, Any]] | None = None
        detection_histogram = _compute_histogram(frame, detection.bbox_xyxy)
        for track in self.lost_tracks.values():
            if track.lost_timestamp_seconds is None:
                continue
            lost_duration = float(timestamp_seconds) - float(track.lost_timestamp_seconds)
            if lost_duration < 0 or lost_duration > self.config.lost_track_recovery_seconds:
                continue
            if not _class_compatible(track.class_name, detection.class_name, vehicle_compatibility_enabled=self.config.vehicle_class_compatibility_enabled):
                continue
            predicted_center = track.predicted_center(timestamp_seconds)
            detection_center = bbox_center(detection.bbox_xyxy)
            normalized_center_distance = _normalized_center_distance_to_point(track.bbox_xyxy, detection_center)
            predicted_distance = math.dist(predicted_center, detection_center) / max(math.sqrt(max(bbox_area(track.bbox_xyxy), 1.0)), 1.0)
            area_ratio = _area_ratio(track.bbox_xyxy, detection.bbox_xyxy)
            aspect_ratio = _aspect_ratio_change(track.bbox_xyxy, detection.bbox_xyxy)
            if predicted_distance > self.config.lost_track_max_center_distance_ratio:
                continue
            if area_ratio < self.config.lost_track_min_area_ratio or area_ratio > self.config.lost_track_max_area_ratio:
                continue
            if aspect_ratio < self.config.minimum_aspect_ratio_change or aspect_ratio > self.config.maximum_aspect_ratio_change:
                continue
            appearance_similarity = _appearance_similarity(track.appearance_histogram, detection_histogram)
            score = 1.0 - min(predicted_distance / max(self.config.lost_track_max_center_distance_ratio, 1e-6), 1.0)
            score += min(1.0, detection.confidence)
            score += 0.5 * min(1.0, bbox_iou(track.bbox_xyxy, detection.bbox_xyxy) + 0.1)
            if appearance_similarity is not None:
                score += 0.5 * max(0.0, appearance_similarity)
            candidate_details = {
                "lost_duration_seconds": lost_duration,
                "normalized_center_distance": normalized_center_distance,
                "predicted_distance": predicted_distance,
                "area_ratio": area_ratio,
                "aspect_ratio_change": aspect_ratio,
                "appearance_similarity": appearance_similarity,
            }
            if best_match is None or score > best_match[0]:
                best_match = (score, track, candidate_details)
        if best_match is None or best_match[0] < 1.10:
            return None, None
        track = best_match[1]
        details = best_match[2]
        self.lost_tracks.pop(track.track_id, None)
        track.is_active = True
        self.active_tracks[track.track_id] = track
        self._apply_detection_update(
            track=track,
            detection=detection,
            frame=frame,
            source_frame_index=source_frame_index,
            processed_frame_index=processed_frame_index,
            timestamp_seconds=timestamp_seconds,
            matching_stage="lost_track_recovery",
            reactivated=True,
            center_distance_ratio=float(details["predicted_distance"]),
            area_ratio=float(details["area_ratio"]),
            appearance_similarity=details["appearance_similarity"],
        )
        return track, details

    def refresh_with_detections(
        self,
        *,
        detections: list[DetectionObservation],
        frame,
        source_frame_index: int,
        processed_frame_index: int,
        timestamp_seconds: float,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        self._flush_expired_lost_tracks(timestamp_seconds=timestamp_seconds, source_frame_index=source_frame_index)
        associations: list[dict[str, Any]] = []
        detection_dicts = [detection.to_dict() for detection in detections]
        matches, unmatched_track_ids, unmatched_detection_indexes, decisions = self._assign_detections(detections=detections)
        for decision in decisions:
            associations.append(decision.to_dict())
            self._emit_event(
                timestamp_seconds=timestamp_seconds,
                source_frame_index=source_frame_index,
                event_type="association_match",
                track_id=decision.track_id,
                details=decision.to_dict(),
            )

        for track_id, detection_index, matching_stage, iou_value, center_distance in matches:
            track = self.active_tracks[track_id]
            detection = detections[detection_index]
            self._apply_detection_update(
                track=track,
                detection=detection,
                frame=frame,
                source_frame_index=source_frame_index,
                processed_frame_index=processed_frame_index,
                timestamp_seconds=timestamp_seconds,
                matching_stage=matching_stage,
            )

        for track_id in unmatched_track_ids:
            track = self.active_tracks[track_id]
            track.missed_detection_refreshes += 1
            has_reliable_kcf = bool(track.kcf_success and track.last_propagated_box_valid)
            should_move_to_lost = (
                track.missed_detection_refreshes >= self.config.maximum_missed_yolo_refreshes
                and (
                    not has_reliable_kcf
                    or track.seconds_since_valid_update(timestamp_seconds) >= self.config.maximum_track_idle_seconds
                )
            )
            if should_move_to_lost:
                self._move_to_lost(track, timestamp_seconds=timestamp_seconds, source_frame_index=source_frame_index, reason="missed_refresh_limit")
            else:
                track.status = "propagated_unconfirmed" if has_reliable_kcf else "temporarily_lost"
                self._emit_event(
                    timestamp_seconds=timestamp_seconds,
                    source_frame_index=source_frame_index,
                    event_type="track_unmatched_detector",
                    track_id=track.track_id,
                    details={
                        "missed_detection_refreshes": track.missed_detection_refreshes,
                        "kcf_success": bool(track.kcf_success),
                        "last_propagated_box_valid": bool(track.last_propagated_box_valid),
                    },
                )

        remaining_unmatched_detections: list[int] = []
        for detection_index in unmatched_detection_indexes:
            detection = detections[detection_index]
            recovered_track, recovery_details = self._recover_lost_track(
                detection=detection,
                frame=frame,
                source_frame_index=source_frame_index,
                processed_frame_index=processed_frame_index,
                timestamp_seconds=timestamp_seconds,
            )
            if recovered_track is None:
                remaining_unmatched_detections.append(detection_index)
                continue
            associations.append(
                AssociationDecision(
                    detection_index=detection_index,
                    track_id=recovered_track.track_id,
                    result="reactivated_track",
                    matching_stage="lost_track_recovery",
                    center_distance_ratio=None if recovery_details is None else float(recovery_details["predicted_distance"]),
                    details={
                        "lost_duration_seconds": None if recovery_details is None else round(float(recovery_details["lost_duration_seconds"]), 6),
                    },
                ).to_dict()
            )

        for detection_index in remaining_unmatched_detections:
            detection = detections[detection_index]
            track = self._create_track(detection, frame)
            self.active_tracks[track.track_id] = track
            associations.append(
                AssociationDecision(
                    detection_index=detection_index,
                    track_id=track.track_id,
                    result="new_track",
                    matching_stage="unmatched_yolo_detection",
                ).to_dict()
            )

        track_payloads = [
            self._make_track_payload(
                track,
                processed_frame_index=processed_frame_index,
                timestamp_seconds=timestamp_seconds,
            )
            for track in self.active_track_list()
        ]
        return track_payloads, detection_dicts, associations

    def flush_at_video_end(self, *, timestamp_seconds: float, source_frame_index: int) -> None:
        for track in list(self.active_track_list()):
            self._complete_track(track, timestamp_seconds=timestamp_seconds, source_frame_index=source_frame_index, reason="video_end")
        for track in list(self.lost_tracks.values()):
            self._complete_track(track, timestamp_seconds=timestamp_seconds, source_frame_index=source_frame_index, reason="video_end")

    def all_tracks(self) -> list[HybridTrack]:
        return list(self.completed_tracks.values()) + list(self.active_tracks.values()) + list(self.lost_tracks.values())
