from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


STATE_DENSE = "dense"
STATE_NORMAL = "normal"
STATE_SPARSE = "sparse"
STATE_IDLE = "idle"
STATE_EMERGENCY = "emergency"


@dataclass(frozen=True)
class SchedulerObservation:
    timestamp_seconds: float
    active_tracks: int
    low_confidence_tracks: int
    recent_track_losses: int
    recent_unmatched_detections: int
    average_assignment_cost: float
    scene_motion_change: float
    overlap_count: int
    entry_zone_activity: bool
    recent_visual_tracker_failure: bool
    last_detector_call_timestamp: float | None


@dataclass(frozen=True)
class SchedulerDecision:
    state: str
    detector_interval_seconds: float
    should_run_detector: bool
    reasons: list[str]
    time_since_last_detector_call: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "detector_interval_seconds": round(self.detector_interval_seconds, 6),
            "should_run_detector": self.should_run_detector,
            "reasons": list(self.reasons),
            "time_since_last_detector_call": round(self.time_since_last_detector_call, 6),
        }


@dataclass
class AdaptiveDetectorScheduler:
    normal_interval_seconds: float
    sparse_interval_seconds: float
    idle_interval_seconds: float
    maximum_gap_seconds: float
    call_history: list[dict[str, Any]] = field(default_factory=list)

    def _state_for_observation(self, observation: SchedulerObservation) -> tuple[str, list[str]]:
        reasons: list[str] = []
        if observation.active_tracks <= 0 and observation.scene_motion_change < 0.05 and not observation.entry_zone_activity:
            reasons.append("idle_scene")
            return STATE_IDLE, reasons
        if (
            observation.recent_unmatched_detections >= 2
            or observation.recent_track_losses >= 2
            or observation.low_confidence_tracks >= 2
            or observation.overlap_count >= 2
            or observation.recent_visual_tracker_failure
            or observation.entry_zone_activity
        ):
            if observation.recent_visual_tracker_failure:
                reasons.append("visual_tracker_failure")
            if observation.entry_zone_activity:
                reasons.append("entry_zone_activity")
            if observation.recent_unmatched_detections >= 2:
                reasons.append("multiple_unmatched_detections")
            if observation.recent_track_losses >= 2:
                reasons.append("multiple_track_losses")
            return STATE_DENSE, reasons
        if observation.average_assignment_cost <= 0.20 and observation.scene_motion_change < 0.12 and observation.low_confidence_tracks == 0:
            reasons.append("stable_tracks")
            return STATE_SPARSE, reasons
        reasons.append("ordinary_tracking")
        return STATE_NORMAL, reasons

    def decide(self, observation: SchedulerObservation) -> SchedulerDecision:
        time_since_last = (
            max(0.0, observation.timestamp_seconds - observation.last_detector_call_timestamp)
            if observation.last_detector_call_timestamp is not None
            else float("inf")
        )
        state, reasons = self._state_for_observation(observation)
        interval = {
            STATE_DENSE: 0.1,
            STATE_NORMAL: self.normal_interval_seconds,
            STATE_SPARSE: self.sparse_interval_seconds,
            STATE_IDLE: self.idle_interval_seconds,
            STATE_EMERGENCY: 0.0,
        }[state]
        should_run = observation.last_detector_call_timestamp is None or time_since_last >= interval
        if observation.last_detector_call_timestamp is not None and time_since_last >= self.maximum_gap_seconds:
            state = STATE_EMERGENCY
            interval = 0.0
            should_run = True
            reasons = ["maximum_gap_enforced", *reasons]
        return SchedulerDecision(
            state=state,
            detector_interval_seconds=interval,
            should_run_detector=should_run,
            reasons=reasons,
            time_since_last_detector_call=0.0 if time_since_last == float("inf") else time_since_last,
        )

    def record_detector_call(self, *, frame_record: dict[str, Any], decision: SchedulerDecision, reason: str) -> None:
        self.call_history.append(
            {
                "processed_frame_index": int(frame_record["processed_frame_index"]),
                "timestamp_seconds": round(float(frame_record["timestamp_seconds"]), 6),
                "scheduler_state": decision.state,
                "detector_interval_seconds": round(decision.detector_interval_seconds, 6),
                "reasons": list(decision.reasons),
                "detector_call_reason": reason,
            }
        )

    def build_report(self) -> dict[str, Any]:
        counts = Counter(str(item["scheduler_state"]) for item in self.call_history)
        reason_counts = Counter()
        for item in self.call_history:
            for reason in item["reasons"]:
                reason_counts[str(reason)] += 1
        return {
            "status": "success",
            "total_detector_calls": len(self.call_history),
            "calls_by_state": dict(sorted(counts.items())),
            "reasons": dict(sorted(reason_counts.items())),
            "calls": list(self.call_history),
        }
