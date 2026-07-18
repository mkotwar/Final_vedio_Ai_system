from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import CropCollectionConfig
from .lifecycle import LifecycleUpdateResult
from .schemas import BoundingBox, TrackLifecycleEventType, TrackRecord, TrackStatus, TrackedFramePacket
from .serialization import dataclass_to_dict
from .validation import (
    validate_enum_value,
    validate_finite_float,
    validate_non_empty_string,
    validate_non_negative_int,
    validate_probability,
)


@dataclass(frozen=True, order=True)
class TrackIdentity:
    """Application-level track identity; track_id alone is not unique enough."""

    source_id: str
    track_id: int
    track_generation: int

    def __post_init__(self) -> None:
        validate_non_empty_string(self.source_id, "source_id")
        validate_non_negative_int(self.track_id, "track_id")
        validate_non_negative_int(self.track_generation, "track_generation")

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass(frozen=True)
class TrackObservation:
    """JSON-safe observation for one visible track in one processed frame."""

    source_id: str
    track_id: int
    track_generation: int
    source_track_id: str | int | None
    frame_index: int
    timestamp_sec: float
    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str
    lifecycle_status: TrackStatus
    full_frame_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_non_empty_string(self.source_id, "source_id")
        validate_non_negative_int(self.track_id, "track_id")
        validate_non_negative_int(self.track_generation, "track_generation")
        if isinstance(self.source_track_id, str):
            validate_non_empty_string(self.source_track_id, "source_track_id")
        validate_non_negative_int(self.frame_index, "frame_index")
        timestamp = validate_finite_float(self.timestamp_sec, "timestamp_sec")
        if timestamp < 0.0:
            raise ValueError("timestamp_sec must be non-negative.")
        validate_probability(self.confidence, "confidence")
        validate_non_negative_int(self.class_id, "class_id")
        validate_non_empty_string(self.class_name, "class_name")
        object.__setattr__(self, "lifecycle_status", validate_enum_value(self.lifecycle_status, TrackStatus, "lifecycle_status"))
        if self.full_frame_path is not None:
            validate_non_empty_string(self.full_frame_path, "full_frame_path")

    @property
    def identity(self) -> TrackIdentity:
        return TrackIdentity(self.source_id, self.track_id, self.track_generation)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass(frozen=True)
class RuntimeTrackObservation:
    """Runtime wrapper that carries the frame array without serializing it."""

    observation: TrackObservation
    frame: Any

    @property
    def identity(self) -> TrackIdentity:
        return self.observation.identity

    def to_dict(self) -> dict[str, Any]:
        return self.observation.to_dict()


@dataclass(frozen=True)
class ObservationCollectionResult:
    observations: list[RuntimeTrackObservation]
    dropped_count: int
    drop_reasons: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "observations": [item.to_dict() for item in self.observations],
            "dropped_count": self.dropped_count,
            "drop_reasons": dict(sorted(self.drop_reasons.items())),
        }


class TrackObservationCollector:
    """Collect deterministic per-track observations after lifecycle normalization."""

    def __init__(self, config: CropCollectionConfig) -> None:
        self.config = config
        self._observations_by_identity: dict[TrackIdentity, list[TrackObservation]] = {}
        self._last_key_by_identity: dict[TrackIdentity, tuple[int, float]] = {}
        self.drop_reasons: dict[str, int] = {}

    def collect(self, packet: TrackedFramePacket, lifecycle_result: LifecycleUpdateResult) -> ObservationCollectionResult:
        if packet.source_id != self._source_id_from_result(lifecycle_result, packet.source_id):
            return ObservationCollectionResult([], 1, {"source_mismatch": 1})
        if not self.config.enabled:
            return ObservationCollectionResult([], len(packet.tracks), {"disabled": len(packet.tracks)})

        records = self._record_lookup(lifecycle_result)
        events = self._event_lookup(lifecycle_result)
        observations: list[RuntimeTrackObservation] = []
        local_drops: dict[str, int] = {}

        for track in sorted(packet.tracks, key=lambda item: (item.track_id, item.class_id, item.bbox.x1, item.bbox.y1)):
            record = records.get((packet.source_id, track.track_id))
            event = events.get((packet.source_id, track.track_id))
            if record is None and event is None:
                self._count_drop(local_drops, "missing_lifecycle_identity")
                continue
            generation = record.track_generation if record is not None else event.track_generation
            status = record.status if record is not None else event.new_status
            if status == TrackStatus.COMPLETED:
                self._count_drop(local_drops, "completed_status")
                continue
            identity = TrackIdentity(packet.source_id, track.track_id, generation)
            current_key = (packet.frame_index, packet.timestamp_sec)
            previous_key = self._last_key_by_identity.get(identity)
            if previous_key is not None and current_key <= previous_key:
                self._count_drop(local_drops, "duplicate_or_regressed_observation")
                continue
            observation = TrackObservation(
                source_id=packet.source_id,
                track_id=track.track_id,
                track_generation=generation,
                source_track_id=record.source_track_id if record is not None else track.source_track_id,
                frame_index=packet.frame_index,
                timestamp_sec=packet.timestamp_sec,
                bbox=track.bbox,
                confidence=track.confidence,
                class_id=track.class_id,
                class_name=track.class_name,
                lifecycle_status=status,
                metadata={"frame_width": packet.frame_width, "frame_height": packet.frame_height},
            )
            self._last_key_by_identity[identity] = current_key
            retained = self._observations_by_identity.setdefault(identity, [])
            retained.append(observation)
            if len(retained) > self.config.max_observations_per_track:
                del retained[0 : len(retained) - self.config.max_observations_per_track]
                self._count_drop(local_drops, "observation_history_trimmed")
            observations.append(RuntimeTrackObservation(observation=observation, frame=packet.frame))

        for reason, count in local_drops.items():
            self.drop_reasons[reason] = self.drop_reasons.get(reason, 0) + count
        return ObservationCollectionResult(observations, sum(local_drops.values()), local_drops)

    def observations_for(self, identity: TrackIdentity) -> tuple[TrackObservation, ...]:
        return tuple(self._observations_by_identity.get(identity, ()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_count": len(self._observations_by_identity),
            "observation_count": sum(len(items) for items in self._observations_by_identity.values()),
            "drop_reasons": dict(sorted(self.drop_reasons.items())),
        }

    def _count_drop(self, local_drops: dict[str, int], reason: str) -> None:
        local_drops[reason] = local_drops.get(reason, 0) + 1

    def _record_lookup(self, lifecycle_result: LifecycleUpdateResult) -> dict[tuple[str, int], TrackRecord]:
        records: dict[tuple[str, int], TrackRecord] = {}
        for record in list(lifecycle_result.active_tracks) + list(lifecycle_result.newly_completed_tracks):
            records[(record.source_id, record.track_id)] = record
        return records

    def _event_lookup(self, lifecycle_result: LifecycleUpdateResult) -> dict[tuple[str, int], Any]:
        visible_event_types = {
            TrackLifecycleEventType.CREATED,
            TrackLifecycleEventType.OBSERVED,
            TrackLifecycleEventType.CONFIRMED,
            TrackLifecycleEventType.RECOVERED,
            TrackLifecycleEventType.CLASS_UPDATED,
        }
        events: dict[tuple[str, int], Any] = {}
        for event in lifecycle_result.events:
            if event.event_type in visible_event_types:
                events[(event.source_id, event.track_id)] = event
        return events

    def _source_id_from_result(self, lifecycle_result: LifecycleUpdateResult, fallback: str) -> str:
        if lifecycle_result.active_tracks:
            return lifecycle_result.active_tracks[0].source_id
        if lifecycle_result.newly_completed_tracks:
            return lifecycle_result.newly_completed_tracks[0].source_id
        if lifecycle_result.events:
            return lifecycle_result.events[0].source_id
        return fallback
