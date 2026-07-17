from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LocalObjectIdentity:
    local_object_id: int
    object_family: str
    stable_class: str
    class_votes: Counter[str]
    created_timestamp_seconds: float
    entry_zone: str
    confirmed: bool = False
    detector_hit_timestamps: list[float] = field(default_factory=list)
    tracker_id_history: list[str] = field(default_factory=list)
    current_tracker_id: str | None = None
    observations: list[dict[str, Any]] = field(default_factory=list)
    gap_events: list[dict[str, Any]] = field(default_factory=list)
    possible_recovery_links: list[dict[str, Any]] = field(default_factory=list)
    termination_reason: str | None = None
    terminated: bool = False


class StableIdentityManager:
    def __init__(self) -> None:
        self._next_local_object_id = 1
        self.active_tracker_id_to_local_object_id: dict[str, int] = {}
        self.local_object_id_to_current_tracker_id: dict[int, str | None] = {}
        self.identities: dict[int, LocalObjectIdentity] = {}
        self.tracker_id_remap_events: list[dict[str, Any]] = []
        self.local_id_reuse_guard: set[int] = set()

    def _new_local_object_id(self) -> int:
        local_object_id = self._next_local_object_id
        self._next_local_object_id += 1
        if local_object_id in self.local_id_reuse_guard:
            raise RuntimeError(f"Local object ID reuse detected for {local_object_id}.")
        self.local_id_reuse_guard.add(local_object_id)
        return local_object_id

    def create_identity(
        self,
        *,
        tracker_id: str,
        object_family: str,
        class_name: str,
        timestamp_seconds: float,
        zone: str,
    ) -> int:
        local_object_id = self._new_local_object_id()
        identity = LocalObjectIdentity(
            local_object_id=local_object_id,
            object_family=object_family,
            stable_class=class_name,
            class_votes=Counter({class_name: 1}),
            created_timestamp_seconds=timestamp_seconds,
            entry_zone=zone,
            tracker_id_history=[tracker_id],
            current_tracker_id=tracker_id,
        )
        self.identities[local_object_id] = identity
        self.active_tracker_id_to_local_object_id[tracker_id] = local_object_id
        self.local_object_id_to_current_tracker_id[local_object_id] = tracker_id
        return local_object_id

    def bind_tracker(self, *, tracker_id: str, local_object_id: int) -> None:
        current_owner = self.active_tracker_id_to_local_object_id.get(tracker_id)
        if current_owner is not None and current_owner != local_object_id:
            raise RuntimeError(f"Tracker {tracker_id} already mapped to local object {current_owner}.")
        existing_tracker = self.local_object_id_to_current_tracker_id.get(local_object_id)
        if existing_tracker is not None and existing_tracker != tracker_id:
            self.active_tracker_id_to_local_object_id.pop(existing_tracker, None)
        self.active_tracker_id_to_local_object_id[tracker_id] = local_object_id
        self.local_object_id_to_current_tracker_id[local_object_id] = tracker_id
        identity = self.identities[local_object_id]
        identity.current_tracker_id = tracker_id
        if tracker_id not in identity.tracker_id_history:
            identity.tracker_id_history.append(tracker_id)

    def remap_tracker(
        self,
        *,
        previous_tracker_id: str,
        new_tracker_id: str,
        local_object_id: int,
        timestamp_seconds: float,
        recovery_score: float,
    ) -> None:
        self.bind_tracker(tracker_id=new_tracker_id, local_object_id=local_object_id)
        self.active_tracker_id_to_local_object_id.pop(previous_tracker_id, None)
        self.tracker_id_remap_events.append(
            {
                "timestamp_seconds": round(timestamp_seconds, 6),
                "local_object_id": local_object_id,
                "previous_tracker_id": previous_tracker_id,
                "new_tracker_id": new_tracker_id,
                "recovery_score": round(recovery_score, 6),
            }
        )

    def release_tracker(self, tracker_id: str) -> int | None:
        local_object_id = self.active_tracker_id_to_local_object_id.pop(tracker_id, None)
        if local_object_id is None:
            return None
        self.local_object_id_to_current_tracker_id[local_object_id] = None
        self.identities[local_object_id].current_tracker_id = None
        return local_object_id

    def add_observation(
        self,
        *,
        local_object_id: int,
        tracker_id: str,
        class_name: str,
        timestamp_seconds: float,
        source_frame_index: int,
        processed_frame_index: int,
        bbox_xyxy: list[float],
        state: str,
        detector_ran: bool,
        detector_confidence: float | None,
    ) -> None:
        identity = self.identities[local_object_id]
        identity.class_votes[class_name] += 1
        identity.stable_class = max(sorted(identity.class_votes), key=lambda item: (identity.class_votes[item], item))
        if detector_ran:
            identity.detector_hit_timestamps.append(timestamp_seconds)
        identity.observations.append(
            {
                "local_object_id": local_object_id,
                "tracker_id": tracker_id,
                "timestamp_seconds": round(timestamp_seconds, 6),
                "source_frame_index": source_frame_index,
                "processed_frame_index": processed_frame_index,
                "bbox_xyxy": [round(float(value), 3) for value in bbox_xyxy],
                "state": state,
                "detector_ran": detector_ran,
                "class_name": class_name,
                "detector_confidence": None if detector_confidence is None else round(float(detector_confidence), 6),
            }
        )

    def add_gap_event(
        self,
        *,
        local_object_id: int,
        gap_start_seconds: float,
        gap_end_seconds: float,
        recovery_score: float,
    ) -> None:
        identity = self.identities[local_object_id]
        identity.gap_events.append(
            {
                "gap_start_seconds": round(gap_start_seconds, 6),
                "gap_end_seconds": round(gap_end_seconds, 6),
                "gap_duration_seconds": round(max(0.0, gap_end_seconds - gap_start_seconds), 6),
                "recovery_score": round(recovery_score, 6),
            }
        )

    def add_possible_recovery_link(self, *, local_object_id: int, payload: dict[str, Any]) -> None:
        self.identities[local_object_id].possible_recovery_links.append(payload)

    def mark_confirmed(self, *, local_object_id: int) -> None:
        self.identities[local_object_id].confirmed = True

    def mark_terminated(self, *, local_object_id: int, reason: str) -> None:
        identity = self.identities[local_object_id]
        identity.terminated = True
        identity.termination_reason = reason
        current_tracker_id = self.local_object_id_to_current_tracker_id.get(local_object_id)
        if current_tracker_id is not None:
            self.active_tracker_id_to_local_object_id.pop(current_tracker_id, None)
        self.local_object_id_to_current_tracker_id[local_object_id] = None
        identity.current_tracker_id = None

    def local_object_id_for_tracker(self, tracker_id: str) -> int | None:
        return self.active_tracker_id_to_local_object_id.get(tracker_id)

    def build_mappings_payload(self) -> dict[str, Any]:
        return {
            "status": "success",
            "active_tracker_id_to_local_object_id": dict(sorted(self.active_tracker_id_to_local_object_id.items())),
            "local_object_id_to_current_tracker_id": {
                str(key): value for key, value in sorted(self.local_object_id_to_current_tracker_id.items())
            },
            "tracker_id_history_by_local_object_id": {
                str(local_object_id): list(identity.tracker_id_history)
                for local_object_id, identity in sorted(self.identities.items())
            },
        }

    def build_timelines_payload(self) -> dict[str, Any]:
        return {
            "status": "success",
            "local_objects": [
                {
                    "local_object_id": identity.local_object_id,
                    "object_family": identity.object_family,
                    "stable_class": identity.stable_class,
                    "class_votes": dict(identity.class_votes),
                    "created_timestamp_seconds": round(identity.created_timestamp_seconds, 6),
                    "entry_zone": identity.entry_zone,
                    "confirmed": identity.confirmed,
                    "tracker_id_history": list(identity.tracker_id_history),
                    "current_tracker_id": identity.current_tracker_id,
                    "terminated": identity.terminated,
                    "termination_reason": identity.termination_reason,
                    "observations": list(identity.observations),
                }
                for identity in self.identities.values()
            ],
        }

    def build_gap_payload(self) -> dict[str, Any]:
        return {
            "status": "success",
            "gap_events": [
                {
                    "local_object_id": identity.local_object_id,
                    **gap_event,
                }
                for identity in self.identities.values()
                for gap_event in identity.gap_events
            ],
        }

