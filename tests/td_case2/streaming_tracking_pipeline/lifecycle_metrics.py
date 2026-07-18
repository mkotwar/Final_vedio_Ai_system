"""Metrics for application-level track lifecycle output."""

from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any

from .lifecycle import LifecycleUpdateResult
from .schemas import TrackCompletionReason, TrackLifecycleEventType, TrackRecord, TrackStatus


class LifecycleMetricsAccumulator:
    """Accumulate descriptive lifecycle metrics without physical-ID claims."""

    def __init__(self) -> None:
        self._events = []
        self._completed: list[TrackRecord] = []
        self._active_peak = 0
        self._active_end = 0
        self._max_generation_by_track_id: dict[int, int] = {}

    def update(self, result: LifecycleUpdateResult) -> None:
        self._events.extend(result.events)
        self._completed.extend(result.newly_completed_tracks)
        self._active_peak = max(self._active_peak, len(result.active_tracks))
        self._active_end = len(result.active_tracks)
        for record in result.active_tracks + result.newly_completed_tracks:
            self._max_generation_by_track_id[record.track_id] = max(
                self._max_generation_by_track_id.get(record.track_id, 0),
                record.track_generation,
            )

    def to_dict(self) -> dict[str, Any]:
        event_counts = Counter(event.event_type for event in self._events)
        reason_counts = Counter(record.completion_reason for record in self._completed if record.completion_reason is not None)
        completed_observations = [record.observation_count for record in self._completed]
        confirmed_durations = [
            record.duration_sec
            for record in self._completed
            if record.status == TrackStatus.COMPLETED and record.observation_count >= 1 and record.completion_reason is not None and record.duration_sec >= 0
            if record.observation_count >= 3
        ]
        transition_counts = Counter(
            f"{event.previous_status.value if event.previous_status else 'none'}->{event.new_status.value}"
            for event in self._events
        )
        recovery_attempts = event_counts[TrackLifecycleEventType.RECOVERED] + reason_counts[TrackCompletionReason.LOST_BUFFER_EXPIRED]
        return {
            "tracks_created": event_counts[TrackLifecycleEventType.CREATED],
            "tracks_confirmed": event_counts[TrackLifecycleEventType.CONFIRMED],
            "tracks_completed": event_counts[TrackLifecycleEventType.COMPLETED],
            "tentative_tracks_completed": len(
                [
                    record
                    for record in self._completed
                    if record.completion_reason == TrackCompletionReason.INVALID_TRACK
                ]
            ),
            "confirmed_tracks_completed": len(
                [
                    record
                    for record in self._completed
                    if record.completion_reason in {TrackCompletionReason.VIDEO_ENDED, TrackCompletionReason.STREAM_ENDED, TrackCompletionReason.LOST_BUFFER_EXPIRED}
                    and record.observation_count >= 3
                ]
            ),
            "tracks_temporarily_lost": event_counts[TrackLifecycleEventType.TEMPORARILY_LOST],
            "tracks_recovered": event_counts[TrackLifecycleEventType.RECOVERED],
            "tracks_completed_lost_buffer": reason_counts[TrackCompletionReason.LOST_BUFFER_EXPIRED],
            "tracks_completed_video_end": reason_counts[TrackCompletionReason.VIDEO_ENDED],
            "tracks_completed_invalid": reason_counts[TrackCompletionReason.INVALID_TRACK],
            "active_tracks_at_peak": self._active_peak,
            "active_tracks_at_end": self._active_end,
            "completed_tracks_at_end": len(self._completed),
            "average_observations_per_completed_track": round(sum(completed_observations) / len(completed_observations), 6)
            if completed_observations
            else 0.0,
            "median_observations_per_completed_track": median(completed_observations) if completed_observations else 0.0,
            "tracks_with_one_observation": len([record for record in self._completed if record.observation_count == 1]),
            "tracks_with_less_than_three_observations": len([record for record in self._completed if record.observation_count < 3]),
            "average_confirmed_duration_sec": round(sum(confirmed_durations) / len(confirmed_durations), 6)
            if confirmed_durations
            else 0.0,
            "completion_reason_counts": {reason.value: reason_counts[reason] for reason in sorted(reason_counts, key=lambda item: item.value)},
            "status_transition_counts": dict(sorted(transition_counts.items())),
            "generation_count": sum(value + 1 for value in self._max_generation_by_track_id.values()),
            "reused_completed_track_ids": len([track_id for track_id, generation in self._max_generation_by_track_id.items() if generation > 0]),
            "recovery_attempts": recovery_attempts,
            "successful_same_id_recoveries": event_counts[TrackLifecycleEventType.RECOVERED],
            "expired_lost_tracks": reason_counts[TrackCompletionReason.LOST_BUFFER_EXPIRED],
            "event_counts": {event_type.value: event_counts[event_type] for event_type in TrackLifecycleEventType},
        }
