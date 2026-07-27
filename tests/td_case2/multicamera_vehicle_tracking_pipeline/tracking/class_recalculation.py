from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from .class_stabilization import build_class_diagnostics, normalize_track_class_name
from .tracking_config import TrackingConfig
from .tracking_models import ClassObservation, LocalVehicleTrack, TrackClassDiagnostics, TrackObservation


@dataclass(frozen=True, slots=True)
class FragmentLinkEvaluation:
    previous_track_uuid: str
    next_track_uuid: str
    eligible: bool
    reasons: list[str]
    time_gap_seconds: float | None
    spatial_score: float | None
    class_compatibility: float | None


@dataclass(frozen=True, slots=True)
class IdentityContinuityEvaluation:
    eligible: bool
    reasons: list[str]
    spatial_score: float
    class_compatibility: float
    area_ratio: float


def recalculate_track_class(
    observations: list[TrackObservation],
    config: TrackingConfig,
) -> TrackClassDiagnostics:
    class_scores: dict[str, float] = {}
    class_counts: dict[str, int] = {}
    class_max_confidences: dict[str, float] = {}
    normalized_observations: list[TrackObservation] = []
    for observation in observations:
        normalized_class_name = normalize_track_class_name(observation.class_name, config)
        normalized = TrackObservation(
            camera_code=observation.camera_code,
            local_track_id=observation.local_track_id,
            frame_number=observation.frame_number,
            video_time_seconds=observation.video_time_seconds,
            camera_timestamp=observation.camera_timestamp,
            class_name=normalized_class_name,
            confidence=observation.confidence,
            bbox_xyxy=observation.bbox_xyxy,
            track_uuid=observation.track_uuid,
            state=observation.state,
            raw_class_name=observation.raw_class_name or observation.class_name,
        )
        normalized_observations.append(normalized)
        class_scores[normalized_class_name] = class_scores.get(normalized_class_name, 0.0) + float(normalized.confidence)
        class_counts[normalized_class_name] = class_counts.get(normalized_class_name, 0) + 1
        class_max_confidences[normalized_class_name] = max(
            class_max_confidences.get(normalized_class_name, 0.0),
            float(normalized.confidence),
        )
    history = [
        ClassObservation(
            frame_number=item.frame_number,
            video_time_seconds=item.video_time_seconds,
            camera_timestamp=item.camera_timestamp,
            class_name=item.class_name,
            confidence=float(item.confidence),
            bbox_xyxy=item.bbox_xyxy,
            raw_class_name=item.raw_class_name,
        )
        for item in normalized_observations
    ]
    return build_class_diagnostics(
        history=history,
        class_scores=class_scores,
        class_counts=class_counts,
        class_max_confidences=class_max_confidences,
        config=config,
    )


def evaluate_fragment_link(
    previous_track: LocalVehicleTrack,
    next_track: LocalVehicleTrack,
    config: TrackingConfig,
) -> FragmentLinkEvaluation:
    reasons: list[str] = []
    if previous_track.camera_code != next_track.camera_code:
        reasons.append("different_camera")
    if not previous_track.observations or not next_track.observations:
        reasons.append("missing_observations")

    time_gap_seconds = None
    spatial_score = None
    class_compatibility = None
    if previous_track.observations and next_track.observations:
        previous_last = previous_track.observations[-1]
        next_first = next_track.observations[0]
        time_gap_seconds = float(next_first.video_time_seconds) - float(previous_last.video_time_seconds)
        if config.fragment_linking.require_no_time_overlap and time_gap_seconds < 0.0:
            reasons.append("time_overlap")
        if time_gap_seconds > float(config.fragment_linking.maximum_gap_seconds):
            reasons.append("gap_too_large")
        spatial_score = _spatial_score(previous_last.bbox_xyxy, next_first.bbox_xyxy)
        if spatial_score < float(config.fragment_linking.minimum_spatial_score):
            reasons.append("spatial_score_too_low")
        class_compatibility = _class_compatibility(
            previous_track.stable_class_name or previous_track.class_name,
            next_track.stable_class_name or next_track.class_name,
            config,
        )
        if class_compatibility < float(config.fragment_linking.minimum_class_compatibility):
            reasons.append("class_incompatible")
    eligible = not reasons
    return FragmentLinkEvaluation(
        previous_track_uuid=previous_track.track_uuid,
        next_track_uuid=next_track.track_uuid,
        eligible=eligible,
        reasons=reasons,
        time_gap_seconds=time_gap_seconds,
        spatial_score=spatial_score,
        class_compatibility=class_compatibility,
    )


def evaluate_identity_continuity(
    previous_track: LocalVehicleTrack,
    next_track: LocalVehicleTrack,
    config: TrackingConfig,
) -> IdentityContinuityEvaluation:
    reasons: list[str] = []
    previous_last = previous_track.observations[-1]
    next_first = next_track.observations[0]
    spatial_score = _spatial_score(previous_last.bbox_xyxy, next_first.bbox_xyxy)
    class_compatibility = _class_compatibility(
        previous_track.stable_class_name or previous_track.class_name,
        next_track.stable_class_name or next_track.class_name,
        config,
    )
    area_ratio = _area_ratio(previous_last.bbox_xyxy, next_first.bbox_xyxy)
    continuity = config.identity_continuity
    if spatial_score < float(continuity.hard_split_spatial_score):
        reasons.append("hard_spatial_break")
    elif spatial_score < float(continuity.minimum_spatial_score) and class_compatibility < float(continuity.minimum_class_compatibility):
        reasons.append("spatial_and_class_break")
    elif area_ratio > float(continuity.maximum_area_ratio) and spatial_score < max(0.35, float(continuity.minimum_spatial_score)):
        reasons.append("scale_break")
    return IdentityContinuityEvaluation(
        eligible=not reasons,
        reasons=reasons,
        spatial_score=spatial_score,
        class_compatibility=class_compatibility,
        area_ratio=area_ratio,
    )


def _bbox_center(bbox_xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    return ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)


def _spatial_score(
    previous_bbox: tuple[float, float, float, float],
    next_bbox: tuple[float, float, float, float],
) -> float:
    prev_center = _bbox_center(previous_bbox)
    next_center = _bbox_center(next_bbox)
    distance = hypot(next_center[0] - prev_center[0], next_center[1] - prev_center[1])
    previous_width = max(float(previous_bbox[2]) - float(previous_bbox[0]), 1.0)
    previous_height = max(float(previous_bbox[3]) - float(previous_bbox[1]), 1.0)
    next_width = max(float(next_bbox[2]) - float(next_bbox[0]), 1.0)
    next_height = max(float(next_bbox[3]) - float(next_bbox[1]), 1.0)
    scale = max((previous_width + next_width) / 2.0, (previous_height + next_height) / 2.0, 1.0)
    return max(0.0, 1.0 - (distance / (scale * 2.5)))


def _class_compatibility(previous_class_name: str, next_class_name: str, config: TrackingConfig) -> float:
    left = normalize_track_class_name(previous_class_name, config)
    right = normalize_track_class_name(next_class_name, config)
    if left == right:
        return 1.0
    for family_members in config.class_families.values():
        if left in family_members and right in family_members:
            return 0.5
    return 0.0


def _area_ratio(
    previous_bbox: tuple[float, float, float, float],
    next_bbox: tuple[float, float, float, float],
) -> float:
    previous_area = max((float(previous_bbox[2]) - float(previous_bbox[0])) * (float(previous_bbox[3]) - float(previous_bbox[1])), 1.0)
    next_area = max((float(next_bbox[2]) - float(next_bbox[0])) * (float(next_bbox[3]) - float(next_bbox[1])), 1.0)
    return max(previous_area, next_area) / min(previous_area, next_area)
