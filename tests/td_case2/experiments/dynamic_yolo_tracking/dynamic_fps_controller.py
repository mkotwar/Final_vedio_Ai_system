from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


STATE_IDLE = "EMPTY"
STATE_LOW = "LOW"
STATE_NORMAL = "NORMAL"
STATE_BURST = "BURST"
TRACKING_STATES = (STATE_IDLE, STATE_LOW, STATE_NORMAL, STATE_BURST)


@dataclass(frozen=True)
class DynamicFpsConfig:
    """Configurable dynamic FPS-controller thresholds."""

    idle_fps: float = 1.0
    low_fps: float = 2.0
    normal_fps: float = 5.0
    burst_fps: float = 10.0
    idle_after_empty_seconds: float = 3.0
    low_after_stationary_seconds: float = 3.0
    burst_cooldown_seconds: float = 2.0
    min_state_hold_seconds: float = 0.6
    strong_displacement_threshold: float = 0.12
    medium_displacement_threshold: float = 0.06
    strong_area_change_threshold: float = 0.60
    medium_area_change_threshold: float = 0.30
    abrupt_direction_change_threshold: float = 0.75
    fast_speed_threshold: float = 0.14
    medium_speed_threshold: float = 0.08
    proximity_burst_threshold: float = 0.10
    vehicle_person_burst_threshold: float = 0.12
    scene_motion_burst_threshold: float = 0.20
    tracker_instability_threshold: float = 0.30
    stationary_speed_threshold: float = 0.015
    stationary_displacement_threshold: float = 0.015
    new_track_burst_threshold: int = 2
    lost_track_burst_threshold: int = 2

    def target_fps_for_state(self, state: str) -> float:
        return {
            STATE_IDLE: self.idle_fps,
            STATE_LOW: self.low_fps,
            STATE_NORMAL: self.normal_fps,
            STATE_BURST: self.burst_fps,
        }[state]


@dataclass(frozen=True)
class ControllerObservation:
    """Causal observation built from previously processed frames only."""

    timestamp_seconds: float
    detection_count: int
    active_track_count: int
    avg_center_displacement: float = 0.0
    avg_bbox_area_change: float = 0.0
    avg_direction_change: float = 0.0
    max_track_speed: float = 0.0
    stationary_track_ratio: float = 0.0
    new_track_count: int = 0
    lost_track_count: int = 0
    vehicle_vehicle_proximity: float | None = None
    vehicle_person_proximity: float | None = None
    scene_motion_score: float = 0.0
    consecutive_empty_detections: int = 0
    tracker_confidence_instability: float = 0.0


@dataclass(frozen=True)
class ControllerDecision:
    """Controller output for the next scheduling interval."""

    timestamp_seconds: float
    previous_state: str
    state: str
    target_fps: float
    transition: bool
    selection_reason: list[str]
    burst_signals: list[str]
    metrics: dict[str, Any]


@dataclass
class _ControllerMemory:
    current_state: str = STATE_IDLE
    state_started_at: float = 0.0
    last_observation_at: float | None = None
    empty_started_at: float | None = None
    stationary_started_at: float | None = None
    burst_cooldown_started_at: float | None = None
    last_transition_at: float = 0.0
    transition_log: list[dict[str, Any]] = field(default_factory=list)


class DynamicFpsController:
    """Causal state-machine controller for dynamic tracking FPS."""

    def __init__(self, config: DynamicFpsConfig | None = None):
        self.config = config or DynamicFpsConfig()
        self.memory = _ControllerMemory()

    @property
    def current_state(self) -> str:
        return self.memory.current_state

    def _state_duration(self, timestamp_seconds: float) -> float:
        return max(0.0, timestamp_seconds - self.memory.state_started_at)

    def _signals(self, observation: ControllerObservation) -> tuple[list[str], list[str]]:
        strong_signals: list[str] = []
        medium_signals: list[str] = []
        if observation.avg_center_displacement >= self.config.strong_displacement_threshold:
            strong_signals.append("rapid_center_displacement")
        elif observation.avg_center_displacement >= self.config.medium_displacement_threshold:
            medium_signals.append("elevated_center_displacement")

        if observation.max_track_speed >= self.config.fast_speed_threshold:
            strong_signals.append("high_track_speed")
        elif observation.max_track_speed >= self.config.medium_speed_threshold:
            medium_signals.append("medium_track_speed")

        if observation.avg_bbox_area_change >= self.config.strong_area_change_threshold:
            strong_signals.append("rapid_bbox_area_change")
        elif observation.avg_bbox_area_change >= self.config.medium_area_change_threshold:
            medium_signals.append("moderate_bbox_area_change")

        if observation.avg_direction_change >= self.config.abrupt_direction_change_threshold:
            strong_signals.append("abrupt_direction_change")

        if observation.vehicle_vehicle_proximity is not None and observation.vehicle_vehicle_proximity <= self.config.proximity_burst_threshold:
            medium_signals.append("vehicle_proximity_reduction")

        if observation.vehicle_person_proximity is not None and observation.vehicle_person_proximity <= self.config.vehicle_person_burst_threshold:
            medium_signals.append("vehicle_person_proximity_reduction")

        if observation.scene_motion_score >= self.config.scene_motion_burst_threshold:
            medium_signals.append("scene_motion_increase")

        if observation.new_track_count >= self.config.new_track_burst_threshold:
            medium_signals.append("multiple_new_tracks")

        if observation.lost_track_count >= self.config.lost_track_burst_threshold:
            medium_signals.append("multiple_lost_tracks")

        if observation.tracker_confidence_instability >= self.config.tracker_instability_threshold:
            medium_signals.append("tracker_confidence_instability")

        return strong_signals, medium_signals

    def observe(self, observation: ControllerObservation) -> ControllerDecision:
        previous_state = self.memory.current_state
        state_duration = self._state_duration(observation.timestamp_seconds)

        if observation.detection_count <= 0:
            if self.memory.empty_started_at is None:
                self.memory.empty_started_at = observation.timestamp_seconds
        else:
            self.memory.empty_started_at = None

        low_motion = (
            observation.detection_count > 0
            and observation.active_track_count > 0
            and observation.max_track_speed <= self.config.stationary_speed_threshold
            and observation.avg_center_displacement <= self.config.stationary_displacement_threshold
        )
        if low_motion:
            if self.memory.stationary_started_at is None:
                self.memory.stationary_started_at = observation.timestamp_seconds
        else:
            self.memory.stationary_started_at = None

        strong_signals, medium_signals = self._signals(observation)
        burst_triggered = bool(strong_signals or len(medium_signals) >= 2)
        state = previous_state
        reasons: list[str] = []
        burst_signals = [*strong_signals, *medium_signals]

        empty_duration = (
            max(0.0, observation.timestamp_seconds - self.memory.empty_started_at)
            if self.memory.empty_started_at is not None
            else 0.0
        )
        stationary_duration = (
            max(0.0, observation.timestamp_seconds - self.memory.stationary_started_at)
            if self.memory.stationary_started_at is not None
            else 0.0
        )

        can_switch = state_duration >= self.config.min_state_hold_seconds

        if observation.detection_count > 0 and previous_state == STATE_IDLE and can_switch:
            state = STATE_NORMAL
            reasons.append("object_detected_resume_normal")
        elif empty_duration >= self.config.idle_after_empty_seconds and can_switch:
            state = STATE_IDLE
            reasons.append("empty_road_idle")
        elif burst_triggered and previous_state != STATE_BURST and can_switch:
            state = STATE_BURST
            reasons.extend(burst_signals[:])
            self.memory.burst_cooldown_started_at = observation.timestamp_seconds
        elif previous_state == STATE_BURST:
            if burst_triggered:
                self.memory.burst_cooldown_started_at = observation.timestamp_seconds
                reasons.extend(burst_signals[:])
            else:
                burst_quiet_for = (
                    max(0.0, observation.timestamp_seconds - self.memory.burst_cooldown_started_at)
                    if self.memory.burst_cooldown_started_at is not None
                    else 0.0
                )
                if burst_quiet_for >= self.config.burst_cooldown_seconds and can_switch:
                    state = STATE_NORMAL if observation.detection_count > 0 else STATE_IDLE
                    reasons.append("burst_cooldown_complete")
        elif stationary_duration >= self.config.low_after_stationary_seconds and can_switch:
            state = STATE_LOW
            reasons.append("stationary_tracks_low")
        elif previous_state == STATE_LOW and observation.detection_count > 0 and not low_motion and can_switch:
            state = STATE_NORMAL
            reasons.append("movement_resume_normal")
        elif previous_state not in {STATE_NORMAL, STATE_BURST} and observation.detection_count > 0 and can_switch:
            state = STATE_NORMAL
            reasons.append("object_present_normal")

        transition = state != previous_state
        if transition:
            self.memory.current_state = state
            self.memory.state_started_at = observation.timestamp_seconds
            self.memory.last_transition_at = observation.timestamp_seconds
            transition_entry = {
                "timestamp_seconds": round(observation.timestamp_seconds, 6),
                "from_state": previous_state,
                "to_state": state,
                "signals": reasons or ["state_change"],
                "metrics": {
                    "detection_count": observation.detection_count,
                    "active_track_count": observation.active_track_count,
                    "avg_center_displacement": round(observation.avg_center_displacement, 6),
                    "avg_bbox_area_change": round(observation.avg_bbox_area_change, 6),
                    "avg_direction_change": round(observation.avg_direction_change, 6),
                    "max_track_speed": round(observation.max_track_speed, 6),
                    "scene_motion_score": round(observation.scene_motion_score, 6),
                },
            }
            self.memory.transition_log.append(transition_entry)

        self.memory.last_observation_at = observation.timestamp_seconds
        state = self.memory.current_state
        target_fps = self.config.target_fps_for_state(state)
        return ControllerDecision(
            timestamp_seconds=observation.timestamp_seconds,
            previous_state=previous_state,
            state=state,
            target_fps=target_fps,
            transition=transition,
            selection_reason=reasons or ["state_hold"],
            burst_signals=burst_signals,
            metrics={
                "state_duration_seconds": round(self._state_duration(observation.timestamp_seconds), 6),
                "empty_duration_seconds": round(empty_duration, 6),
                "stationary_duration_seconds": round(stationary_duration, 6),
                "detection_count": observation.detection_count,
                "active_track_count": observation.active_track_count,
            },
        )

    def transition_log(self) -> list[dict[str, Any]]:
        return list(self.memory.transition_log)
