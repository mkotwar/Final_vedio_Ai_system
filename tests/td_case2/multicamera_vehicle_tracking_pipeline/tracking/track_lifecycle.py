from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..detection.detection_models import DetectionPacket
from .class_recalculation import evaluate_fragment_link, evaluate_identity_continuity
from .class_stabilization import build_class_diagnostics, normalize_track_class_name
from .tracking_config import TrackingConfig
from .tracking_models import ClassObservation, LocalVehicleTrack, TrackClassDiagnostics, TrackObservation


@dataclass(slots=True)
class LifecycleUpdateResult:
    observations: list[TrackObservation]
    completed_tracks: list[LocalVehicleTrack]
    active_tracks: list[LocalVehicleTrack]


def build_track_uuid(camera_code: str, local_track_id: int, run_id: str | None = None) -> str:
    prefix = f"{run_id}:" if run_id else ""
    return f"{prefix}{camera_code}:TRACK_{local_track_id}"


class _TrackState:
    def __init__(
        self,
        *,
        track_uuid: str,
        camera_code: str,
        camera_name: str,
        source_path: Path,
        local_track_id: int,
        first_observation: TrackObservation,
    ) -> None:
        self.track_uuid = track_uuid
        self.camera_code = camera_code
        self.camera_name = camera_name
        self.source_path = source_path
        self.local_track_id = local_track_id
        self.class_name = first_observation.class_name
        self.first_frame_number = first_observation.frame_number
        self.last_frame_number = first_observation.frame_number
        self.first_seen_at = first_observation.camera_timestamp
        self.last_seen_at = first_observation.camera_timestamp
        self.first_video_time_seconds = first_observation.video_time_seconds
        self.last_video_time_seconds = first_observation.video_time_seconds
        self.observation_count = 1
        self.best_confidence = float(first_observation.confidence)
        self.state = "tentative"
        self.observations = [first_observation]
        self.lost_frame_count = 0
        self.class_scores: dict[str, float] = {}
        self.class_observation_counts: dict[str, int] = {}
        self.class_max_confidences: dict[str, float] = {}
        self.raw_class_history: list[ClassObservation] = []
        self.class_diagnostics = TrackClassDiagnostics()
        self.linked_track_group_id = track_uuid
        self.fragment_candidate_track_uuids: list[str] = []

    def to_track(self) -> LocalVehicleTrack:
        return LocalVehicleTrack(
            track_uuid=self.track_uuid,
            camera_code=self.camera_code,
            local_track_id=self.local_track_id,
            class_name=self.class_name,
            first_frame_number=self.first_frame_number,
            last_frame_number=self.last_frame_number,
            first_seen_at=self.first_seen_at,
            last_seen_at=self.last_seen_at,
            first_video_time_seconds=self.first_video_time_seconds,
            last_video_time_seconds=self.last_video_time_seconds,
            observation_count=self.observation_count,
            best_confidence=self.best_confidence,
            state=self.state,
            observations=list(self.observations),
            camera_name=self.camera_name,
            source_path=self.source_path,
            lost_frame_count=self.lost_frame_count,
            provisional_class_name=self.class_diagnostics.provisional_class_name,
            stable_class_name=self.class_diagnostics.stable_class_name,
            class_is_locked=self.class_diagnostics.class_is_locked,
            class_confidence=self.class_diagnostics.class_confidence,
            class_winner_margin=self.class_diagnostics.class_winner_margin,
            class_observation_count=self.class_diagnostics.class_observation_count,
            class_conflict_count=self.class_diagnostics.class_conflict_count,
            class_scores=dict(self.class_diagnostics.class_scores),
            class_observation_counts=dict(self.class_diagnostics.class_observation_counts),
            class_max_confidences=dict(self.class_diagnostics.class_max_confidences),
            raw_class_history=list(self.class_diagnostics.raw_class_history),
            latest_observation_class_name=self.class_diagnostics.latest_observation_class_name,
            linked_track_group_id=self.linked_track_group_id,
            fragment_candidate_track_uuids=list(self.fragment_candidate_track_uuids),
        )

    def _record_class_observation(self, observation: TrackObservation, config: TrackingConfig | None) -> None:
        normalized_class = observation.class_name
        self.class_scores[normalized_class] = self.class_scores.get(normalized_class, 0.0) + float(observation.confidence)
        self.class_observation_counts[normalized_class] = self.class_observation_counts.get(normalized_class, 0) + 1
        self.class_max_confidences[normalized_class] = max(self.class_max_confidences.get(normalized_class, 0.0), float(observation.confidence))
        self.raw_class_history.append(
            ClassObservation(
                frame_number=observation.frame_number,
                video_time_seconds=observation.video_time_seconds,
                camera_timestamp=observation.camera_timestamp,
                class_name=normalized_class,
                confidence=float(observation.confidence),
                bbox_xyxy=observation.bbox_xyxy,
                raw_class_name=observation.raw_class_name,
            )
        )
        if config is not None:
            self.class_diagnostics = build_class_diagnostics(
                history=self.raw_class_history,
                class_scores=self.class_scores,
                class_counts=self.class_observation_counts,
                class_max_confidences=self.class_max_confidences,
                config=config,
                previous_stable_class_name=self.class_diagnostics.stable_class_name,
                previous_class_is_locked=self.class_diagnostics.class_is_locked,
            )
            self.class_name = self.class_diagnostics.stable_class_name or self.class_diagnostics.provisional_class_name or normalized_class

    def record_fragment_candidate(self, track_uuid: str) -> None:
        normalized = str(track_uuid).strip()
        if not normalized or normalized == self.track_uuid:
            return
        if normalized not in self.fragment_candidate_track_uuids:
            self.fragment_candidate_track_uuids.append(normalized)


class LocalTrackLifecycle:
    def __init__(self, config: TrackingConfig, *, run_id: str | None = None) -> None:
        self.config = config
        self.run_id = run_id
        self._states_by_camera: dict[str, dict[int, _TrackState]] = {}
        self._metadata_by_camera: dict[str, tuple[str, Path]] = {}
        self._last_frame_by_camera: dict[str, int] = {}
        self._completed_by_camera: dict[str, list[LocalVehicleTrack]] = {}
        self._used_logical_track_ids_by_camera: dict[str, set[int]] = {}
        self._next_logical_track_id_by_camera: dict[str, int] = {}

    def update(self, packet: DetectionPacket, observations: list[TrackObservation]) -> LifecycleUpdateResult:
        camera_code = packet.camera_code
        self._validate_packet_order(packet)
        self._metadata_by_camera[camera_code] = (packet.camera_name, packet.source_path)
        states = self._states_by_camera.setdefault(camera_code, {})
        completed_store = self._completed_by_camera.setdefault(camera_code, [])
        visible_ids: set[int] = set()
        adjusted_observations: list[TrackObservation] = []
        newly_completed: list[LocalVehicleTrack] = []

        for observation in observations:
            if observation.camera_code != camera_code:
                raise ValueError("Observation camera code does not match lifecycle packet camera.")
            tracker_local_track_id = observation.local_track_id
            state = states.get(tracker_local_track_id)
            if state is None:
                state, previous_tracker_local_track_id = self._try_link_fragment(
                    camera_code=camera_code,
                    tracker_local_track_id=tracker_local_track_id,
                    observation=observation,
                    states=states,
                )
                if state is not None and previous_tracker_local_track_id is not None:
                    del states[previous_tracker_local_track_id]
                    states[tracker_local_track_id] = state
            elif self._should_split_identity(state, observation):
                finalized = self._finalize_identity_break(states, tracker_local_track_id, state)
                completed_store.append(finalized)
                newly_completed.append(finalized)
                state = None
            if state is None:
                logical_track_id = self._allocate_logical_track_id(camera_code, tracker_local_track_id)
                track_uuid = build_track_uuid(camera_code, logical_track_id, self.run_id)
                provisional = TrackObservation(
                    camera_code=observation.camera_code,
                    local_track_id=logical_track_id,
                    frame_number=observation.frame_number,
                    video_time_seconds=observation.video_time_seconds,
                    camera_timestamp=observation.camera_timestamp,
                    class_name=normalize_track_class_name(observation.class_name, self.config),
                    confidence=observation.confidence,
                    bbox_xyxy=observation.bbox_xyxy,
                    track_uuid=track_uuid,
                    state="tentative",
                    raw_class_name=observation.raw_class_name or observation.class_name,
                )
                state = _TrackState(
                    track_uuid=track_uuid,
                    camera_code=camera_code,
                    camera_name=packet.camera_name,
                    source_path=packet.source_path,
                    local_track_id=logical_track_id,
                    first_observation=provisional,
                )
                states[tracker_local_track_id] = state
                state._record_class_observation(provisional, self.config)
            else:
                state.last_frame_number = observation.frame_number
                state.last_seen_at = observation.camera_timestamp
                state.last_video_time_seconds = observation.video_time_seconds
                state.observation_count += 1
                state.best_confidence = max(state.best_confidence, float(observation.confidence))
                state.lost_frame_count = 0
                normalized_class = normalize_track_class_name(observation.class_name, self.config)
                stabilized_observation = TrackObservation(
                    camera_code=observation.camera_code,
                    local_track_id=state.local_track_id,
                    frame_number=observation.frame_number,
                    video_time_seconds=observation.video_time_seconds,
                    camera_timestamp=observation.camera_timestamp,
                    class_name=normalized_class,
                    confidence=observation.confidence,
                    bbox_xyxy=observation.bbox_xyxy,
                    track_uuid=state.track_uuid,
                    state=state.state,
                    raw_class_name=observation.raw_class_name or observation.class_name,
                )
                state._record_class_observation(stabilized_observation, self.config)
            if state.state == "temporarily_lost":
                state.state = "active"
            elif state.observation_count >= self.config.min_confirmed_observations:
                state.state = "active"
            visible_ids.add(tracker_local_track_id)
            base_observation = TrackObservation(
                camera_code=observation.camera_code,
                local_track_id=state.local_track_id,
                frame_number=observation.frame_number,
                video_time_seconds=observation.video_time_seconds,
                camera_timestamp=observation.camera_timestamp,
                class_name=state.class_diagnostics.stable_class_name or state.class_diagnostics.provisional_class_name or normalize_track_class_name(observation.class_name, self.config),
                confidence=observation.confidence,
                bbox_xyxy=observation.bbox_xyxy,
                track_uuid=state.track_uuid,
                state=state.state,
                raw_class_name=observation.raw_class_name or observation.class_name,
            )
            if state is not None and state.observations:
                if state.observations[-1].frame_number != base_observation.frame_number:
                    state.observations.append(base_observation)
                elif state.observations[-1].track_uuid != base_observation.track_uuid:
                    state.observations[-1] = base_observation
                else:
                    state.observations[-1] = base_observation
            adjusted_observations.append(base_observation)

        for local_track_id, state in list(states.items()):
            if local_track_id in visible_ids:
                continue
            state.lost_frame_count += 1
            if state.state == "tentative":
                if state.lost_frame_count > self.config.max_lost_frames:
                    state.state = "discarded"
                    track = state.to_track()
                    completed_store.append(track)
                    newly_completed.append(track)
                    del states[local_track_id]
                continue
            if state.state in ("active", "temporarily_lost"):
                state.state = "temporarily_lost"
                if state.lost_frame_count > self.config.max_lost_frames:
                    state.state = "completed"
                    track = state.to_track()
                    completed_store.append(track)
                    newly_completed.append(track)
                    del states[local_track_id]

        active_tracks = [state.to_track() for state in states.values()]
        return LifecycleUpdateResult(
            observations=adjusted_observations,
            completed_tracks=newly_completed,
            active_tracks=sorted(active_tracks, key=lambda item: (item.camera_code, item.first_frame_number, item.local_track_id)),
        )

    def flush_camera(self, camera_code: str) -> LifecycleUpdateResult:
        states = self._states_by_camera.get(camera_code, {})
        completed = self._completed_by_camera.setdefault(camera_code, [])
        newly_completed: list[LocalVehicleTrack] = []
        for local_track_id, state in list(states.items()):
            state.state = "discarded" if state.observation_count < self.config.min_confirmed_observations else "completed"
            track = state.to_track()
            completed.append(track)
            newly_completed.append(track)
            del states[local_track_id]
        return LifecycleUpdateResult(observations=[], completed_tracks=newly_completed, active_tracks=[])

    def flush_all(self) -> LifecycleUpdateResult:
        completed: list[LocalVehicleTrack] = []
        for camera_code in list(self._states_by_camera):
            completed.extend(self.flush_camera(camera_code).completed_tracks)
        return LifecycleUpdateResult(observations=[], completed_tracks=completed, active_tracks=[])

    def get_active_tracks(self, camera_code: str) -> list[LocalVehicleTrack]:
        return [state.to_track() for state in self._states_by_camera.get(camera_code, {}).values()]

    def get_completed_tracks(self, camera_code: str | None = None) -> list[LocalVehicleTrack]:
        if camera_code is not None:
            return list(self._completed_by_camera.get(camera_code, []))
        completed: list[LocalVehicleTrack] = []
        for items in self._completed_by_camera.values():
            completed.extend(items)
        return completed

    def _validate_packet_order(self, packet: DetectionPacket) -> None:
        previous = self._last_frame_by_camera.get(packet.camera_code)
        if previous is not None and packet.frame_number <= previous:
            raise ValueError(f"Frame numbers must be strictly increasing per camera: {packet.camera_code}")
        self._last_frame_by_camera[packet.camera_code] = packet.frame_number

    def _try_link_fragment(
        self,
        *,
        camera_code: str,
        tracker_local_track_id: int,
        observation: TrackObservation,
        states: dict[int, _TrackState],
    ) -> tuple[_TrackState | None, int | None]:
        if not self.config.fragment_linking.enabled:
            return None, None
        next_track_uuid = build_track_uuid(camera_code, tracker_local_track_id, self.run_id)
        next_track = LocalVehicleTrack(
            track_uuid=next_track_uuid,
            camera_code=camera_code,
            local_track_id=tracker_local_track_id,
            class_name=normalize_track_class_name(observation.class_name, self.config),
            first_frame_number=observation.frame_number,
            last_frame_number=observation.frame_number,
            first_seen_at=observation.camera_timestamp,
            last_seen_at=observation.camera_timestamp,
            first_video_time_seconds=observation.video_time_seconds,
            last_video_time_seconds=observation.video_time_seconds,
            observation_count=1,
            best_confidence=float(observation.confidence),
            state="tentative",
            observations=[
                TrackObservation(
                    camera_code=observation.camera_code,
                    local_track_id=tracker_local_track_id,
                    frame_number=observation.frame_number,
                    video_time_seconds=observation.video_time_seconds,
                    camera_timestamp=observation.camera_timestamp,
                    class_name=normalize_track_class_name(observation.class_name, self.config),
                    confidence=observation.confidence,
                    bbox_xyxy=observation.bbox_xyxy,
                    track_uuid=next_track_uuid,
                    state="tentative",
                    raw_class_name=observation.raw_class_name or observation.class_name,
                )
            ],
            camera_name=self._metadata_by_camera.get(camera_code, ("", Path()))[0] or None,
            source_path=self._metadata_by_camera.get(camera_code, ("", Path()))[1] if camera_code in self._metadata_by_camera else None,
        )
        best_match: tuple[_TrackState, int, float, float] | None = None
        for previous_tracker_local_track_id, candidate_state in states.items():
            if candidate_state.state != "temporarily_lost":
                continue
            evaluation = evaluate_fragment_link(candidate_state.to_track(), next_track, self.config)
            if not evaluation.eligible:
                continue
            score = float(evaluation.spatial_score or 0.0)
            class_score = float(evaluation.class_compatibility or 0.0)
            if best_match is None or (score, class_score) > (best_match[2], best_match[3]):
                best_match = (candidate_state, previous_tracker_local_track_id, score, class_score)
        if best_match is None:
            return None, None
        state, previous_tracker_local_track_id, _, _ = best_match
        state.record_fragment_candidate(next_track_uuid)
        state.lost_frame_count = 0
        state.state = "active" if state.observation_count >= self.config.min_confirmed_observations else "tentative"
        return state, previous_tracker_local_track_id

    def _allocate_logical_track_id(self, camera_code: str, preferred_track_id: int) -> int:
        used = self._used_logical_track_ids_by_camera.setdefault(camera_code, set())
        next_track_id = self._next_logical_track_id_by_camera.get(camera_code, 1)
        preferred = int(preferred_track_id)
        if preferred >= 0 and preferred not in used:
            used.add(preferred)
            self._next_logical_track_id_by_camera[camera_code] = max(next_track_id, preferred + 1)
            return preferred
        candidate = max(next_track_id, 1)
        while candidate in used:
            candidate += 1
        used.add(candidate)
        self._next_logical_track_id_by_camera[camera_code] = candidate + 1
        return candidate

    def _should_split_identity(self, state: _TrackState, observation: TrackObservation) -> bool:
        if not self.config.identity_continuity.enabled or not state.observations:
            return False
        proposed_track = LocalVehicleTrack(
            track_uuid=state.track_uuid,
            camera_code=state.camera_code,
            local_track_id=state.local_track_id,
            class_name=normalize_track_class_name(observation.class_name, self.config),
            first_frame_number=observation.frame_number,
            last_frame_number=observation.frame_number,
            first_seen_at=observation.camera_timestamp,
            last_seen_at=observation.camera_timestamp,
            first_video_time_seconds=observation.video_time_seconds,
            last_video_time_seconds=observation.video_time_seconds,
            observation_count=1,
            best_confidence=float(observation.confidence),
            state=state.state,
            observations=[
                TrackObservation(
                    camera_code=observation.camera_code,
                    local_track_id=state.local_track_id,
                    frame_number=observation.frame_number,
                    video_time_seconds=observation.video_time_seconds,
                    camera_timestamp=observation.camera_timestamp,
                    class_name=normalize_track_class_name(observation.class_name, self.config),
                    confidence=observation.confidence,
                    bbox_xyxy=observation.bbox_xyxy,
                    track_uuid=state.track_uuid,
                    state=state.state,
                    raw_class_name=observation.raw_class_name or observation.class_name,
                )
            ],
            camera_name=state.camera_name,
            source_path=state.source_path,
            stable_class_name=state.class_diagnostics.stable_class_name,
        )
        evaluation = evaluate_identity_continuity(state.to_track(), proposed_track, self.config)
        return not evaluation.eligible

    def _finalize_identity_break(
        self,
        states: dict[int, _TrackState],
        tracker_local_track_id: int,
        state: _TrackState,
    ) -> LocalVehicleTrack:
        state.state = "discarded" if state.observation_count < self.config.min_confirmed_observations else "completed"
        track = state.to_track()
        del states[tracker_local_track_id]
        return track
