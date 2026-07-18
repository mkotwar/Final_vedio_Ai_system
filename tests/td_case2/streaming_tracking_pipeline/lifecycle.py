"""Application-level track lifecycle manager for sequential tracked packets."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .config import TrackLifecycleConfig
from .schemas import (
    BoundingBox,
    TrackCompletionReason,
    TrackLifecycleEvent,
    TrackLifecycleEventType,
    TrackRecord,
    TrackStatus,
    TrackedFramePacket,
    TrackedObject,
)


@dataclass(frozen=True)
class LifecycleUpdateResult:
    """Lifecycle update output for one packet or flush operation."""

    frame_index: int | None
    timestamp_sec: float | None
    events: list[TrackLifecycleEvent] = field(default_factory=list)
    active_tracks: list[TrackRecord] = field(default_factory=list)
    newly_completed_tracks: list[TrackRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_sec": self.timestamp_sec,
            "events": [item.to_dict() for item in self.events],
            "active_tracks": [item.to_dict() for item in self.active_tracks],
            "newly_completed_tracks": [item.to_dict() for item in self.newly_completed_tracks],
        }


@dataclass
class _TrackState:
    source_id: str
    track_id: int
    source_track_id: str | int | None
    generation: int
    status: TrackStatus
    first_seen_frame: int
    last_seen_frame: int
    first_seen_sec: float
    last_seen_sec: float
    observation_count: int
    missed_processed_frames: int
    class_votes: Counter[str]
    last_bbox: BoundingBox
    last_confidence: float
    last_class_id: int
    last_class_name: str
    object_group: str | None
    creation_order: int
    completion_reason: TrackCompletionReason | None = None

    @property
    def dominant_class(self) -> str | None:
        if not self.class_votes:
            return None
        return sorted(self.class_votes.items(), key=lambda item: (-item[1], item[0]))[0][0]


class TrackLifecycleManager:
    """Maintain application lifecycle state around sequential tracker output IDs."""

    def __init__(self, config: TrackLifecycleConfig) -> None:
        self.config = config
        self._active: dict[tuple[int, int], _TrackState] = {}
        self._active_generation_by_track_id: dict[int, int] = {}
        self._completed: list[TrackRecord] = []
        self._next_generation_by_track_id: dict[int, int] = {}
        self._last_frame_index: int | None = None
        self._last_timestamp_sec: float | None = None
        self._source_id: str | None = None
        self._creation_counter = 0

    def update(self, packet: TrackedFramePacket) -> LifecycleUpdateResult:
        self._validate_packet(packet)
        events: list[TrackLifecycleEvent] = []
        completed: list[TrackRecord] = []
        visible_track_ids = {track.track_id for track in packet.tracks}

        for track in sorted(packet.tracks, key=lambda item: (item.track_id, item.class_id, item.bbox.x1)):
            completed.extend(self._observe_track(track, packet, events))

        for state in self._ordered_active_states():
            if state.track_id in visible_track_ids:
                continue
            completed_track = self._mark_missing(state, packet.frame_index, packet.timestamp_sec, events)
            if completed_track is not None:
                completed.append(completed_track)

        self._last_frame_index = packet.frame_index
        self._last_timestamp_sec = packet.timestamp_sec
        return LifecycleUpdateResult(
            frame_index=packet.frame_index,
            timestamp_sec=packet.timestamp_sec,
            events=events,
            active_tracks=list(self.get_active_tracks()),
            newly_completed_tracks=completed,
        )

    def flush(
        self,
        frame_index: int | None = None,
        timestamp_sec: float | None = None,
        reason: TrackCompletionReason = TrackCompletionReason.VIDEO_ENDED,
    ) -> LifecycleUpdateResult:
        if not self.config.flush_on_end_of_stream:
            return LifecycleUpdateResult(frame_index=frame_index, timestamp_sec=timestamp_sec, active_tracks=list(self.get_active_tracks()))
        events: list[TrackLifecycleEvent] = []
        completed: list[TrackRecord] = []
        flush_frame = frame_index if frame_index is not None else self._last_frame_index
        flush_time = timestamp_sec if timestamp_sec is not None else self._last_timestamp_sec
        if flush_frame is None:
            flush_frame = 0
        if flush_time is None:
            flush_time = 0.0
        for state in self._ordered_active_states():
            completed.append(self._complete_state(state, flush_frame, flush_time, reason, events))
            events.append(self._event(TrackLifecycleEventType.FLUSHED, state, flush_frame, flush_time, TrackStatus.COMPLETED, TrackStatus.COMPLETED, reason))
        return LifecycleUpdateResult(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            events=events,
            active_tracks=list(self.get_active_tracks()),
            newly_completed_tracks=completed,
        )

    def reset(self) -> None:
        self._active.clear()
        self._active_generation_by_track_id.clear()
        self._completed.clear()
        self._next_generation_by_track_id.clear()
        self._last_frame_index = None
        self._last_timestamp_sec = None
        self._source_id = None
        self._creation_counter = 0

    def get_active_tracks(self) -> tuple[TrackRecord, ...]:
        return tuple(self._to_record(state) for state in self._ordered_active_states())

    def get_completed_tracks(self) -> tuple[TrackRecord, ...]:
        return tuple(self._completed)

    def get_track(self, track_id: int, generation: int | None = None) -> TrackRecord | None:
        if generation is None:
            generation = self._active_generation_by_track_id.get(track_id)
            if generation is None:
                candidates = [item for item in self._completed if item.track_id == track_id]
                return candidates[-1] if candidates else None
        state = self._active.get((track_id, generation))
        if state is not None:
            return self._to_record(state)
        for record in reversed(self._completed):
            if record.track_id == track_id and record.track_generation == generation:
                return record
        return None

    def _observe_track(
        self,
        track: TrackedObject,
        packet: TrackedFramePacket,
        events: list[TrackLifecycleEvent],
    ) -> list[TrackRecord]:
        completed: list[TrackRecord] = []
        generation = self._active_generation_by_track_id.get(track.track_id)
        state = self._active.get((track.track_id, generation)) if generation is not None else None
        if state is None:
            state = self._create_state(track, packet)
            events.append(self._event(TrackLifecycleEventType.CREATED, state, packet.frame_index, packet.timestamp_sec, None, TrackStatus.TENTATIVE, None))
            events.append(self._event(TrackLifecycleEventType.OBSERVED, state, packet.frame_index, packet.timestamp_sec, TrackStatus.TENTATIVE, TrackStatus.TENTATIVE, None))
            if state.observation_count >= self.config.minimum_confirmation_observations:
                previous = state.status
                state.status = TrackStatus.CONFIRMED
                events.append(self._event(TrackLifecycleEventType.CONFIRMED, state, packet.frame_index, packet.timestamp_sec, previous, state.status, None))
            return completed

        if state.status == TrackStatus.TEMPORARILY_LOST and not self.config.allow_recovery:
            completed.append(self._complete_state(state, packet.frame_index, packet.timestamp_sec, TrackCompletionReason.TRACKER_REMOVED, events))
            state = self._create_state(track, packet)
            events.append(self._event(TrackLifecycleEventType.CREATED, state, packet.frame_index, packet.timestamp_sec, None, TrackStatus.TENTATIVE, None))
            events.append(self._event(TrackLifecycleEventType.OBSERVED, state, packet.frame_index, packet.timestamp_sec, TrackStatus.TENTATIVE, TrackStatus.TENTATIVE, None))
            return completed

        previous_status = state.status
        previous_dominant = state.dominant_class
        self._apply_observation(state, track)
        events.append(self._event(TrackLifecycleEventType.OBSERVED, state, packet.frame_index, packet.timestamp_sec, previous_status, state.status, None))
        if state.dominant_class != previous_dominant:
            events.append(
                self._event(
                    TrackLifecycleEventType.CLASS_UPDATED,
                    state,
                    packet.frame_index,
                    packet.timestamp_sec,
                    state.status,
                    state.status,
                    None,
                    metadata={"previous_dominant_class": previous_dominant, "new_dominant_class": state.dominant_class},
                )
            )
        if previous_status == TrackStatus.TEMPORARILY_LOST:
            state.status = TrackStatus.CONFIRMED
            events.append(self._event(TrackLifecycleEventType.RECOVERED, state, packet.frame_index, packet.timestamp_sec, previous_status, state.status, None))
        elif state.status == TrackStatus.TENTATIVE and state.observation_count >= self.config.minimum_confirmation_observations:
            state.status = TrackStatus.CONFIRMED
            events.append(self._event(TrackLifecycleEventType.CONFIRMED, state, packet.frame_index, packet.timestamp_sec, previous_status, state.status, None))
        return completed

    def _mark_missing(
        self,
        state: _TrackState,
        frame_index: int,
        timestamp_sec: float,
        events: list[TrackLifecycleEvent],
    ) -> TrackRecord | None:
        state.missed_processed_frames += 1
        if state.status == TrackStatus.TENTATIVE:
            if state.missed_processed_frames > self.config.maximum_tentative_missed_frames and self.config.complete_tentative_tracks:
                if self.config.emit_tentative_completion:
                    return self._complete_state(state, frame_index, timestamp_sec, TrackCompletionReason.INVALID_TRACK, events)
                self._remove_active(state)
            return None
        if state.status == TrackStatus.CONFIRMED:
            previous = state.status
            state.status = TrackStatus.TEMPORARILY_LOST
            events.append(self._event(TrackLifecycleEventType.TEMPORARILY_LOST, state, frame_index, timestamp_sec, previous, state.status, None))
            if self._expired(state, timestamp_sec):
                return self._complete_state(state, frame_index, timestamp_sec, TrackCompletionReason.LOST_BUFFER_EXPIRED, events)
            return None
        if state.status == TrackStatus.TEMPORARILY_LOST and self._expired(state, timestamp_sec):
            return self._complete_state(state, frame_index, timestamp_sec, TrackCompletionReason.LOST_BUFFER_EXPIRED, events)
        return None

    def _expired(self, state: _TrackState, current_timestamp_sec: float) -> bool:
        if state.missed_processed_frames > self.config.maximum_lost_processed_frames:
            return True
        if self.config.maximum_lost_seconds is not None:
            return (current_timestamp_sec - state.last_seen_sec) > self.config.maximum_lost_seconds
        return False

    def _create_state(self, track: TrackedObject, packet: TrackedFramePacket) -> _TrackState:
        generation = self._next_generation_by_track_id.get(track.track_id, 0)
        self._next_generation_by_track_id[track.track_id] = generation + 1
        self._creation_counter += 1
        state = _TrackState(
            source_id=packet.source_id,
            track_id=track.track_id,
            source_track_id=track.source_track_id,
            generation=generation,
            status=TrackStatus.TENTATIVE,
            first_seen_frame=packet.frame_index,
            last_seen_frame=packet.frame_index,
            first_seen_sec=packet.timestamp_sec,
            last_seen_sec=packet.timestamp_sec,
            observation_count=1,
            missed_processed_frames=0,
            class_votes=Counter({track.class_name: 1}),
            last_bbox=track.bbox,
            last_confidence=track.confidence,
            last_class_id=track.class_id,
            last_class_name=track.class_name,
            object_group=track.object_group,
            creation_order=self._creation_counter,
        )
        self._active[(track.track_id, generation)] = state
        self._active_generation_by_track_id[track.track_id] = generation
        return state

    def _apply_observation(self, state: _TrackState, track: TrackedObject) -> None:
        state.last_seen_frame = track.frame_index
        state.last_seen_sec = track.timestamp_sec
        state.observation_count += 1
        state.missed_processed_frames = 0
        state.class_votes[track.class_name] += 1
        state.last_bbox = track.bbox
        state.last_confidence = track.confidence
        state.last_class_id = track.class_id
        state.last_class_name = track.class_name
        if state.object_group is None and track.object_group is not None:
            state.object_group = track.object_group
        if state.source_track_id is None and track.source_track_id is not None:
            state.source_track_id = track.source_track_id

    def _complete_state(
        self,
        state: _TrackState,
        frame_index: int,
        timestamp_sec: float,
        reason: TrackCompletionReason,
        events: list[TrackLifecycleEvent],
    ) -> TrackRecord:
        previous = state.status
        state.status = TrackStatus.COMPLETED
        state.completion_reason = reason
        record = self._to_record(state)
        self._completed.append(record)
        self._remove_active(state)
        events.append(self._event(TrackLifecycleEventType.COMPLETED, state, frame_index, timestamp_sec, previous, TrackStatus.COMPLETED, reason))
        return record

    def _remove_active(self, state: _TrackState) -> None:
        self._active.pop((state.track_id, state.generation), None)
        if self._active_generation_by_track_id.get(state.track_id) == state.generation:
            self._active_generation_by_track_id.pop(state.track_id, None)

    def _to_record(self, state: _TrackState) -> TrackRecord:
        return TrackRecord(
            source_id=state.source_id,
            track_id=state.track_id,
            source_track_id=state.source_track_id,
            track_generation=state.generation,
            status=state.status,
            first_seen_frame=state.first_seen_frame,
            last_seen_frame=state.last_seen_frame,
            first_seen_sec=state.first_seen_sec,
            last_seen_sec=state.last_seen_sec,
            observation_count=state.observation_count,
            missed_frame_count=state.missed_processed_frames,
            class_votes=dict(state.class_votes),
            completion_reason=state.completion_reason,
            last_bbox=state.last_bbox,
            last_confidence=state.last_confidence,
            last_class_id=state.last_class_id,
            last_class_name=state.last_class_name,
            object_group=state.object_group,
        )

    def _event(
        self,
        event_type: TrackLifecycleEventType,
        state: _TrackState,
        frame_index: int,
        timestamp_sec: float,
        previous_status: TrackStatus | None,
        new_status: TrackStatus,
        reason: TrackCompletionReason | None,
        metadata: dict[str, Any] | None = None,
    ) -> TrackLifecycleEvent:
        payload = {
            "last_seen_frame": state.last_seen_frame,
            "last_seen_sec": round(state.last_seen_sec, 6),
            "dominant_class": state.dominant_class,
            "last_class_name": state.last_class_name,
            "object_group": state.object_group,
            "last_confidence": round(float(state.last_confidence), 6),
        }
        if metadata:
            payload.update(metadata)
        return TrackLifecycleEvent(
            event_type=event_type,
            source_id=state.source_id,
            track_id=state.track_id,
            source_track_id=state.source_track_id,
            track_generation=state.generation,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            previous_status=previous_status,
            new_status=new_status,
            observation_count=state.observation_count,
            missed_processed_frames=state.missed_processed_frames,
            reason=reason,
            metadata=payload,
        )

    def _validate_packet(self, packet: TrackedFramePacket) -> None:
        if self._source_id is None:
            self._source_id = packet.source_id
        elif packet.source_id != self._source_id:
            raise ValueError("TrackLifecycleManager received a second source without reset().")
        if self._last_frame_index is not None and packet.frame_index <= self._last_frame_index:
            raise ValueError("TrackLifecycleManager rejected frame-index regression or duplicate frame.")
        if self._last_timestamp_sec is not None and packet.timestamp_sec < self._last_timestamp_sec:
            raise ValueError("TrackLifecycleManager rejected timestamp regression.")
        seen: set[int] = set()
        for track in packet.tracks:
            if track.track_id in seen:
                raise ValueError(f"Duplicate track_id in TrackedFramePacket: {track.track_id}")
            seen.add(track.track_id)
            mismatches = []
            if track.frame_index != packet.frame_index:
                mismatches.append("frame_index")
            if track.timestamp_sec != packet.timestamp_sec:
                mismatches.append("timestamp_sec")
            if mismatches:
                raise ValueError(f"TrackedObject metadata mismatch: {', '.join(mismatches)}.")

    def _ordered_active_states(self) -> list[_TrackState]:
        return sorted(self._active.values(), key=lambda item: (item.first_seen_frame, item.creation_order, item.track_id, item.generation))
