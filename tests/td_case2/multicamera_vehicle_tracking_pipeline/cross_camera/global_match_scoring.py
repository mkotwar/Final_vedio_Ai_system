from __future__ import annotations

from dataclasses import dataclass

from .global_match_config import CameraRouteRule, GlobalMatchConfig
from .global_match_models import CrossCameraMatchResult, TrackIdentityFeatures


UNKNOWN_COLOURS = {"", "UNKNOWN", "UNSPECIFIED", "NONE", "NULL"}


@dataclass(frozen=True, slots=True)
class MatchEvaluation:
    result: CrossCameraMatchResult
    impossible: bool = False


def _canonical_pair(left: TrackIdentityFeatures, right: TrackIdentityFeatures) -> tuple[TrackIdentityFeatures, TrackIdentityFeatures]:
    return (left, right) if left.vehicle_track_id <= right.vehicle_track_id else (right, left)


def _normalized_colour(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().upper()
    return None if normalized in UNKNOWN_COLOURS else normalized


def _normalized_class(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().upper()
    return normalized or None


def _route_score(route_rule: CameraRouteRule | None) -> tuple[float, bool, tuple[str, ...]]:
    if route_rule is None:
        return 0.5, False, ("camera-route-unknown",)
    if not route_rule.allowed:
        return 0.0, True, ("camera-route-disallowed",)
    return 1.0, False, ("camera-route-allowed",)


def _time_score(
    left: TrackIdentityFeatures,
    right: TrackIdentityFeatures,
    config: GlobalMatchConfig,
    route_rule: CameraRouteRule | None,
) -> tuple[float, bool, tuple[str, ...]]:
    mode = config.time_matching.mode
    if mode == "disabled":
        return 0.5, False, ("time-disabled",)
    if mode == "recording_timestamp":
        if left.last_seen_at is None or right.first_seen_at is None:
            return 0.5, False, ("timestamp-missing",)
        gap_seconds = abs((right.first_seen_at - left.last_seen_at).total_seconds())
    else:
        if left.last_video_time_seconds is None or right.first_video_time_seconds is None:
            return 0.5, False, ("relative-video-time-missing",)
        gap_seconds = abs(float(right.first_video_time_seconds) - float(left.last_video_time_seconds))
    if route_rule is not None:
        if gap_seconds < float(route_rule.minimum_travel_seconds):
            return 0.0, True, ("travel-time-too-short",)
        if route_rule.maximum_travel_seconds is not None and gap_seconds > float(route_rule.maximum_travel_seconds):
            return 0.0, True, ("travel-time-too-long",)
    return 1.0, False, ("time-compatible",)


def evaluate_track_pair(
    left: TrackIdentityFeatures,
    right: TrackIdentityFeatures,
    config: GlobalMatchConfig,
    route_rule: CameraRouteRule | None = None,
) -> MatchEvaluation:
    left, right = _canonical_pair(left, right)
    reasons: list[str] = []
    if left.vehicle_track_id == right.vehicle_track_id:
        reasons.append("same-track")
        return MatchEvaluation(
            result=CrossCameraMatchResult(
                left_track_uuid=left.track_uuid,
                right_track_uuid=right.track_uuid,
                left_vehicle_track_id=left.vehicle_track_id,
                right_vehicle_track_id=right.vehicle_track_id,
                decision="REJECTED",
                score=0.0,
                plate_score=0.0,
                time_score=0.0,
                camera_route_score=0.0,
                class_score=0.0,
                colour_score=0.0,
                visual_score=0.0,
                reasons=tuple(reasons),
                rule_version=config.rule_version,
            ),
            impossible=True,
        )
    if not config.same_camera_matching and left.camera_id == right.camera_id:
        reasons.append("same-camera-disallowed")
        return MatchEvaluation(
            result=CrossCameraMatchResult(
                left_track_uuid=left.track_uuid,
                right_track_uuid=right.track_uuid,
                left_vehicle_track_id=left.vehicle_track_id,
                right_vehicle_track_id=right.vehicle_track_id,
                decision="REJECTED",
                score=0.0,
                plate_score=0.0,
                time_score=0.0,
                camera_route_score=0.0,
                class_score=0.0,
                colour_score=0.0,
                visual_score=0.0,
                reasons=tuple(reasons),
                rule_version=config.rule_version,
            ),
            impossible=True,
        )

    plate_score = 0.0
    verified_plate_match = False
    left_plate_status = str(left.plate_status or "").upper()
    right_plate_status = str(right.plate_status or "").upper()
    left_plate = str(left.normalized_plate or "").strip().upper() or None
    right_plate = str(right.normalized_plate or "").strip().upper() or None

    if left_plate_status == "VERIFIED" and right_plate_status == "VERIFIED":
        if left_plate and right_plate and left_plate == right_plate:
            verified_plate_match = True
            plate_score = 1.0
            reasons.append("verified-plate-match")
        elif left_plate and right_plate and left_plate != right_plate:
            reasons.append("verified-plate-mismatch")
            return MatchEvaluation(
                result=CrossCameraMatchResult(
                    left_track_uuid=left.track_uuid,
                    right_track_uuid=right.track_uuid,
                    left_vehicle_track_id=left.vehicle_track_id,
                    right_vehicle_track_id=right.vehicle_track_id,
                    decision="REJECTED",
                    score=0.0,
                    plate_score=0.0,
                    time_score=0.0,
                    camera_route_score=0.0,
                    class_score=0.0,
                    colour_score=0.0,
                    visual_score=0.0,
                    reasons=tuple(reasons),
                    rule_version=config.rule_version,
                ),
                impossible=True,
            )
    elif (left_plate_status == "VERIFIED" and right_plate is None) or (right_plate_status == "VERIFIED" and left_plate is None):
        plate_score = 0.35
        reasons.append("one-verified-one-missing")
    elif left_plate and right_plate and left_plate == right_plate:
        plate_score = 0.20
        reasons.append("unverified-plate-text-match")
    else:
        reasons.append("plate-not-decisive")

    route_score, route_impossible, route_reasons = _route_score(route_rule)
    reasons.extend(route_reasons)
    time_score, time_impossible, time_reasons = _time_score(left, right, config, route_rule)
    reasons.extend(time_reasons)
    if route_impossible or time_impossible:
        return MatchEvaluation(
            result=CrossCameraMatchResult(
                left_track_uuid=left.track_uuid,
                right_track_uuid=right.track_uuid,
                left_vehicle_track_id=left.vehicle_track_id,
                right_vehicle_track_id=right.vehicle_track_id,
                decision="REJECTED",
                score=0.0,
                plate_score=plate_score,
                time_score=time_score,
                camera_route_score=route_score,
                class_score=0.0,
                colour_score=0.0,
                visual_score=0.0,
                reasons=tuple(reasons),
                rule_version=config.rule_version,
            ),
            impossible=True,
        )

    left_class = _normalized_class(left.canonical_class)
    right_class = _normalized_class(right.canonical_class)
    if left_class and right_class and left_class == right_class:
        class_score = 1.0
        reasons.append("class-match")
    elif left_class and right_class and left_class != right_class:
        class_score = 0.0
        reasons.append("class-mismatch")
    else:
        class_score = 0.5
        reasons.append("class-unknown")

    left_colour = _normalized_colour(left.canonical_colour)
    right_colour = _normalized_colour(right.canonical_colour)
    if left_colour and right_colour and left_colour == right_colour:
        colour_score = 1.0
        reasons.append("colour-match")
    elif left_colour and right_colour and left_colour != right_colour:
        colour_score = 0.0
        reasons.append("colour-mismatch")
    else:
        colour_score = 0.0
        reasons.append("colour-unknown")

    weights = config.weights
    score = (
        plate_score * weights.verified_plate
        + time_score * weights.time
        + route_score * weights.camera_route
        + class_score * weights.vehicle_class
        + colour_score * weights.vehicle_colour
    )
    score = round(min(max(score, 0.0), 1.0), 6)

    if verified_plate_match:
        decision = "CONFIRMED"
    elif plate_score < 1.0 and score >= config.thresholds.confirmed:
        decision = "POSSIBLE"
        reasons.append("confirmed-threshold-downgraded-without-verified-plate")
    elif score >= config.thresholds.possible:
        decision = "POSSIBLE"
    else:
        decision = "INSUFFICIENT_EVIDENCE"
    return MatchEvaluation(
        result=CrossCameraMatchResult(
            left_track_uuid=left.track_uuid,
            right_track_uuid=right.track_uuid,
            left_vehicle_track_id=left.vehicle_track_id,
            right_vehicle_track_id=right.vehicle_track_id,
            decision=decision,
            score=score,
            plate_score=plate_score,
            time_score=time_score,
            camera_route_score=route_score,
            class_score=class_score,
            colour_score=colour_score,
            visual_score=0.0,
            reasons=tuple(reasons),
            rule_version=config.rule_version,
        ),
        impossible=False,
    )
