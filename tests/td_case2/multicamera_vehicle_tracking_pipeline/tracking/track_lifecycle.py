from __future__ import annotations

from dataclasses import dataclass
from math import hypot
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


@dataclass(frozen=True, slots=True)
class _ConflictSplitDecision:
    conflict_start_index: int
    split_frame: int
    conflicting_class: str
    consecutive_conflict_count: int
    average_conflict_confidence: float
    reason_codes: list[str]
    bbox_iou: float
    center_distance: float
    normalized_center_distance: float
    area_ratio: float
    width_ratio: float
    height_ratio: float
    spatial_score: float
    stable_class_before_split: str | None


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
        self.native_tracker_ids_seen: list[int] = []
        self.reactivation_count = 0
        self.fragment_relink_count = 0
        self.maximum_consecutive_missing_frames = 0
        self.split_from_track_uuid: str | None = None
        self.completion_reason: str | None = None
        self.split_executed = False
        self.split_frame: int | None = None
        self.split_reason_codes: list[str] = []
        self.source_logical_track_id: int | None = None
        self.new_logical_track_id: int | None = None
        self.split_native_tracker_id: int | None = None
        self.pending_conflict_observation_count = 0
        self.stable_class_before_split: str | None = None
        self.conflicting_class: str | None = None
        self.average_conflict_confidence: float | None = None
        self.bbox_iou_at_split: float | None = None
        self.center_distance_at_split: float | None = None
        self.normalized_center_distance_at_split: float | None = None
        self.area_ratio_at_split: float | None = None
        self.width_ratio_at_split: float | None = None
        self.height_ratio_at_split: float | None = None
        self.spatial_score_at_split: float | None = None
        self._register_native_tracker_id(first_observation.native_tracker_id or local_track_id)

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
            class_status=self.class_diagnostics.class_status,
            final_class_reason=self.class_diagnostics.final_class_reason,
            class_is_locked=self.class_diagnostics.class_is_locked,
            class_confidence=self.class_diagnostics.class_confidence,
            class_winner_margin=self.class_diagnostics.class_winner_margin,
            class_observation_count=self.class_diagnostics.class_observation_count,
            class_conflict_count=self.class_diagnostics.class_conflict_count,
            winning_class_name=self.class_diagnostics.winning_class_name,
            winning_class_count=self.class_diagnostics.winning_class_count,
            winning_class_ratio=self.class_diagnostics.winning_class_ratio,
            runner_up_class_name=self.class_diagnostics.runner_up_class_name,
            runner_up_class_count=self.class_diagnostics.runner_up_class_count,
            runner_up_ratio=self.class_diagnostics.runner_up_ratio,
            winner_count_margin=self.class_diagnostics.winner_count_margin,
            count_winner_class_name=self.class_diagnostics.count_winner_class_name,
            score_winner_class_name=self.class_diagnostics.score_winner_class_name,
            winners_agree=self.class_diagnostics.winners_agree,
            maximum_consecutive_winner_count=self.class_diagnostics.maximum_consecutive_winner_count,
            recent_consecutive_winner_count=self.class_diagnostics.recent_consecutive_winner_count,
            class_transition_count=self.class_diagnostics.class_transition_count,
            incompatible_class_transition_count=self.class_diagnostics.incompatible_class_transition_count,
            recent_class_counts=dict(self.class_diagnostics.recent_class_counts),
            recent_winning_class_name=self.class_diagnostics.recent_winning_class_name,
            recent_winning_ratio=self.class_diagnostics.recent_winning_ratio,
            recent_observation_count=self.class_diagnostics.recent_observation_count,
            strong_conflict_detected=self.class_diagnostics.strong_conflict_detected,
            split_recommended=self.class_diagnostics.split_recommended,
            mixed_identity_detected=self.class_diagnostics.mixed_identity_detected,
            mixed_identity_classes=tuple(self.class_diagnostics.mixed_identity_classes),
            mixed_identity_start_frame=self.class_diagnostics.mixed_identity_start_frame,
            mixed_identity_confidence=self.class_diagnostics.mixed_identity_confidence,
            final_class_blocked_due_to_mixed_identity=self.class_diagnostics.final_class_blocked_due_to_mixed_identity,
            class_scores=dict(self.class_diagnostics.class_scores),
            class_ratios=dict(self.class_diagnostics.class_ratios),
            class_observation_counts=dict(self.class_diagnostics.class_observation_counts),
            class_max_confidences=dict(self.class_diagnostics.class_max_confidences),
            winner_confidence_sum=self.class_diagnostics.winner_confidence_sum,
            raw_class_history=list(self.class_diagnostics.raw_class_history),
            latest_observation_class_name=self.class_diagnostics.latest_observation_class_name,
            linked_track_group_id=self.linked_track_group_id,
            fragment_candidate_track_uuids=list(self.fragment_candidate_track_uuids),
            possible_identity_switch=self.class_diagnostics.possible_identity_switch,
            native_tracker_ids_seen=list(self.native_tracker_ids_seen),
            reactivation_count=self.reactivation_count,
            fragment_relink_count=self.fragment_relink_count,
            maximum_consecutive_missing_frames=self.maximum_consecutive_missing_frames,
            split_from_track_uuid=self.split_from_track_uuid,
            completion_reason=self.completion_reason,
            split_executed=self.split_executed,
            split_frame=self.split_frame,
            split_reason_codes=list(self.split_reason_codes),
            source_logical_track_id=self.source_logical_track_id,
            new_logical_track_id=self.new_logical_track_id,
            split_native_tracker_id=self.split_native_tracker_id,
            pending_conflict_observation_count=self.pending_conflict_observation_count,
            stable_class_before_split=self.stable_class_before_split,
            conflicting_class=self.conflicting_class,
            average_conflict_confidence=self.average_conflict_confidence,
            bbox_iou_at_split=self.bbox_iou_at_split,
            center_distance_at_split=self.center_distance_at_split,
            normalized_center_distance_at_split=self.normalized_center_distance_at_split,
            area_ratio_at_split=self.area_ratio_at_split,
            width_ratio_at_split=self.width_ratio_at_split,
            height_ratio_at_split=self.height_ratio_at_split,
            spatial_score_at_split=self.spatial_score_at_split,
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

    def _register_native_tracker_id(self, tracker_id: int) -> None:
        normalized = int(tracker_id)
        if normalized not in self.native_tracker_ids_seen:
            self.native_tracker_ids_seen.append(normalized)

    def _append_observation(self, observation: TrackObservation, config: TrackingConfig) -> None:
        self.last_frame_number = observation.frame_number
        self.last_seen_at = observation.camera_timestamp
        self.last_video_time_seconds = observation.video_time_seconds
        self.observation_count += 1
        self.best_confidence = max(self.best_confidence, float(observation.confidence))
        self.lost_frame_count = 0
        self._record_class_observation(observation, config)
        if self.state == "temporarily_lost":
            self.reactivation_count += 1
            self.state = "active"
        elif self.observation_count >= config.min_confirmed_observations:
            self.state = "active"
        observation_state = TrackObservation(
            camera_code=observation.camera_code,
            local_track_id=self.local_track_id,
            native_tracker_id=observation.native_tracker_id,
            frame_number=observation.frame_number,
            video_time_seconds=observation.video_time_seconds,
            camera_timestamp=observation.camera_timestamp,
            class_name=observation.class_name if config.is_standard_bytetrack else (self.class_diagnostics.stable_class_name or self.class_diagnostics.provisional_class_name or observation.class_name),
            confidence=observation.confidence,
            bbox_xyxy=observation.bbox_xyxy,
            track_uuid=self.track_uuid,
            state=self.state,
            raw_class_name=observation.raw_class_name or observation.class_name,
        )
        self.observations.append(observation_state)

    def _rebuild_from_observations(self, observations: list[TrackObservation], config: TrackingConfig) -> None:
        if not observations:
            raise ValueError("Cannot rebuild track state without observations.")
        first = observations[0]
        self.class_name = first.class_name
        self.first_frame_number = first.frame_number
        self.last_frame_number = first.frame_number
        self.first_seen_at = first.camera_timestamp
        self.last_seen_at = first.camera_timestamp
        self.first_video_time_seconds = first.video_time_seconds
        self.last_video_time_seconds = first.video_time_seconds
        self.observation_count = 1
        self.best_confidence = float(first.confidence)
        self.state = "tentative"
        self.observations = [
            TrackObservation(
                camera_code=first.camera_code,
                local_track_id=self.local_track_id,
                native_tracker_id=first.native_tracker_id,
                frame_number=first.frame_number,
                video_time_seconds=first.video_time_seconds,
                camera_timestamp=first.camera_timestamp,
                class_name=first.class_name,
                confidence=first.confidence,
                bbox_xyxy=first.bbox_xyxy,
                track_uuid=self.track_uuid,
                state="tentative",
                raw_class_name=first.raw_class_name or first.class_name,
            )
        ]
        self.lost_frame_count = 0
        self.class_scores = {}
        self.class_observation_counts = {}
        self.class_max_confidences = {}
        self.raw_class_history = []
        self.class_diagnostics = TrackClassDiagnostics()
        self._record_class_observation(first, config)
        for item in observations[1:]:
            normalized = TrackObservation(
                camera_code=item.camera_code,
                local_track_id=self.local_track_id,
                native_tracker_id=item.native_tracker_id,
                frame_number=item.frame_number,
                video_time_seconds=item.video_time_seconds,
                camera_timestamp=item.camera_timestamp,
                class_name=item.class_name,
                confidence=item.confidence,
                bbox_xyxy=item.bbox_xyxy,
                track_uuid=self.track_uuid,
                state=self.state,
                raw_class_name=item.raw_class_name or item.class_name,
            )
            self._append_observation(normalized, config)


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

    def _logical_track_id_for_new_state(self, camera_code: str, native_tracker_id: int) -> int:
        if self.config.is_standard_bytetrack:
            return int(native_tracker_id)
        return self._allocate_logical_track_id(camera_code, native_tracker_id)

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
            split_from_track_uuid: str | None = None
            if state is None:
                if not self.config.is_standard_bytetrack:
                    state, previous_tracker_local_track_id = self._try_link_fragment(
                        camera_code=camera_code,
                        tracker_local_track_id=tracker_local_track_id,
                        observation=observation,
                        states=states,
                    )
                    if state is not None and previous_tracker_local_track_id is not None:
                        del states[previous_tracker_local_track_id]
                        states[tracker_local_track_id] = state
            elif state is not None:
                if not self.config.is_standard_bytetrack:
                    conflict_split = self._evaluate_class_conflict_split(state, observation)
                    if conflict_split is not None:
                        finalized, state = self._execute_class_conflict_split(
                            camera_code=camera_code,
                            state=state,
                            incoming_tracker_local_track_id=tracker_local_track_id,
                            decision=conflict_split,
                        )
                        split_from_track_uuid = finalized.track_uuid
                        completed_store.append(finalized)
                        newly_completed.append(finalized)
                        states[tracker_local_track_id] = state
                    elif self._should_split_identity(state, observation):
                        finalized = self._finalize_identity_break(states, tracker_local_track_id, state)
                        split_from_track_uuid = finalized.track_uuid
                        completed_store.append(finalized)
                        newly_completed.append(finalized)
                        state = None
            if state is None:
                logical_track_id = self._logical_track_id_for_new_state(camera_code, tracker_local_track_id)
                track_uuid = build_track_uuid(camera_code, logical_track_id, self.run_id)
                provisional = TrackObservation(
                    camera_code=observation.camera_code,
                    local_track_id=logical_track_id,
                    native_tracker_id=observation.native_tracker_id or tracker_local_track_id,
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
                state.split_from_track_uuid = split_from_track_uuid
                states[tracker_local_track_id] = state
                state._record_class_observation(provisional, self.config)
                if state.observation_count >= self.config.min_confirmed_observations:
                    state.state = "active"
            else:
                normalized_class = normalize_track_class_name(observation.class_name, self.config)
                stabilized_observation = TrackObservation(
                    camera_code=observation.camera_code,
                    local_track_id=state.local_track_id,
                    native_tracker_id=observation.native_tracker_id or tracker_local_track_id,
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
                state._append_observation(stabilized_observation, self.config)
            state._register_native_tracker_id(observation.native_tracker_id or tracker_local_track_id)
            visible_ids.add(tracker_local_track_id)
            base_observation = TrackObservation(
                camera_code=observation.camera_code,
                local_track_id=state.local_track_id,
                native_tracker_id=observation.native_tracker_id or tracker_local_track_id,
                frame_number=observation.frame_number,
                video_time_seconds=observation.video_time_seconds,
                camera_timestamp=observation.camera_timestamp,
                class_name=normalize_track_class_name(observation.class_name, self.config) if self.config.is_standard_bytetrack else (state.class_diagnostics.stable_class_name or state.class_diagnostics.provisional_class_name or normalize_track_class_name(observation.class_name, self.config)),
                confidence=observation.confidence,
                bbox_xyxy=observation.bbox_xyxy,
                track_uuid=state.track_uuid,
                state=state.state,
                raw_class_name=observation.raw_class_name or observation.class_name,
            )
            if state is not None and state.observations:
                state.observations[-1] = base_observation
            adjusted_observations.append(base_observation)

        for local_track_id, state in list(states.items()):
            if local_track_id in visible_ids:
                continue
            state.lost_frame_count += 1
            state.maximum_consecutive_missing_frames = max(state.maximum_consecutive_missing_frames, state.lost_frame_count)
            if state.state == "tentative":
                if state.lost_frame_count > self.config.max_lost_frames:
                    state.state = "discarded"
                    state.completion_reason = "tentative_timeout"
                    track = state.to_track()
                    completed_store.append(track)
                    newly_completed.append(track)
                    del states[local_track_id]
                continue
            if state.state in ("active", "temporarily_lost"):
                state.state = "temporarily_lost"
                if state.lost_frame_count > self.config.max_lost_frames:
                    state.state = "completed"
                    state.completion_reason = "lost_timeout"
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
            state.completion_reason = "flush"
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
                    native_tracker_id=observation.native_tracker_id or tracker_local_track_id,
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
        state.fragment_relink_count += 1
        state._register_native_tracker_id(tracker_local_track_id)
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
        normalized_class = normalize_track_class_name(observation.class_name, self.config)
        proposed_history = list(state.raw_class_history)
        proposed_history.append(
            ClassObservation(
                frame_number=observation.frame_number,
                video_time_seconds=observation.video_time_seconds,
                camera_timestamp=observation.camera_timestamp,
                class_name=normalized_class,
                confidence=float(observation.confidence),
                bbox_xyxy=observation.bbox_xyxy,
                raw_class_name=observation.raw_class_name or observation.class_name,
            )
        )
        proposed_scores = dict(state.class_scores)
        proposed_scores[normalized_class] = proposed_scores.get(normalized_class, 0.0) + float(observation.confidence)
        proposed_counts = dict(state.class_observation_counts)
        proposed_counts[normalized_class] = proposed_counts.get(normalized_class, 0) + 1
        proposed_max_confidences = dict(state.class_max_confidences)
        proposed_max_confidences[normalized_class] = max(
            proposed_max_confidences.get(normalized_class, 0.0),
            float(observation.confidence),
        )
        proposed_diagnostics = build_class_diagnostics(
            history=proposed_history,
            class_scores=proposed_scores,
            class_counts=proposed_counts,
            class_max_confidences=proposed_max_confidences,
            config=self.config,
            previous_stable_class_name=state.class_diagnostics.stable_class_name,
            previous_class_is_locked=state.class_diagnostics.class_is_locked,
        )
        proposed_track = LocalVehicleTrack(
            track_uuid=state.track_uuid,
            camera_code=state.camera_code,
            local_track_id=state.local_track_id,
            class_name=normalized_class,
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
                    native_tracker_id=observation.native_tracker_id or state.local_track_id,
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
        if not evaluation.eligible:
            return True
        if not self.config.class_conflict_split.enabled:
            return False
        if not proposed_diagnostics.strong_conflict_detected:
            return False
        stable_class_name = state.class_diagnostics.stable_class_name or state.class_diagnostics.provisional_class_name
        if self._class_compatibility(stable_class_name, normalized_class) >= float(self.config.identity_continuity.minimum_class_compatibility):
            return False
        if not self.config.class_conflict_split.require_spatial_discontinuity:
            return True
        return (
            float(evaluation.spatial_score) < float(self.config.identity_continuity.minimum_spatial_score)
            or float(evaluation.area_ratio) > float(self.config.identity_continuity.maximum_area_ratio)
        )

    def _evaluate_class_conflict_split(self, state: _TrackState, observation: TrackObservation) -> _ConflictSplitDecision | None:
        if not self.config.class_conflict_split.enabled or not state.observations:
            return None
        normalized_class = normalize_track_class_name(observation.class_name, self.config)
        proposed_history = list(state.raw_class_history)
        proposed_history.append(
            ClassObservation(
                frame_number=observation.frame_number,
                video_time_seconds=observation.video_time_seconds,
                camera_timestamp=observation.camera_timestamp,
                class_name=normalized_class,
                confidence=float(observation.confidence),
                bbox_xyxy=observation.bbox_xyxy,
                raw_class_name=observation.raw_class_name or observation.class_name,
            )
        )
        proposed_scores = dict(state.class_scores)
        proposed_scores[normalized_class] = proposed_scores.get(normalized_class, 0.0) + float(observation.confidence)
        proposed_counts = dict(state.class_observation_counts)
        proposed_counts[normalized_class] = proposed_counts.get(normalized_class, 0) + 1
        proposed_max_confidences = dict(state.class_max_confidences)
        proposed_max_confidences[normalized_class] = max(proposed_max_confidences.get(normalized_class, 0.0), float(observation.confidence))
        proposed_diagnostics = build_class_diagnostics(
            history=proposed_history,
            class_scores=proposed_scores,
            class_counts=proposed_counts,
            class_max_confidences=proposed_max_confidences,
            config=self.config,
            previous_stable_class_name=state.class_diagnostics.stable_class_name,
            previous_class_is_locked=state.class_diagnostics.class_is_locked,
        )
        if not proposed_diagnostics.split_recommended:
            return None
        if not (proposed_diagnostics.strong_conflict_detected or proposed_diagnostics.mixed_identity_detected):
            return None
        stable_class_name = state.class_diagnostics.stable_class_name or (
            state.class_diagnostics.provisional_class_name
            if state.class_diagnostics.class_observation_count >= self.config.class_stabilization.minimum_observations
            else None
        )
        conflicting_class = proposed_diagnostics.recent_winning_class_name
        if not stable_class_name or not conflicting_class:
            return None
        if self._class_compatibility(stable_class_name, conflicting_class) >= float(self.config.identity_continuity.minimum_class_compatibility):
            return None
        consecutive_count = self._trailing_class_count(proposed_history, conflicting_class)
        if consecutive_count < int(self.config.class_conflict_split.minimum_consecutive_conflicting_observations):
            return None
        conflict_history = proposed_history[-consecutive_count:]
        average_conflict_confidence = sum(float(item.confidence) for item in conflict_history) / max(len(conflict_history), 1)
        conflict_start_index = len(proposed_history) - consecutive_count
        if conflict_start_index <= 0:
            return None
        previous_observation = state.observations[conflict_start_index - 1]
        latest_conflict_observation = TrackObservation(
            camera_code=observation.camera_code,
            local_track_id=state.local_track_id,
            native_tracker_id=observation.native_tracker_id or state.local_track_id,
            frame_number=observation.frame_number,
            video_time_seconds=observation.video_time_seconds,
            camera_timestamp=observation.camera_timestamp,
            class_name=conflicting_class,
            confidence=observation.confidence,
            bbox_xyxy=observation.bbox_xyxy,
            track_uuid=state.track_uuid,
            state=state.state,
            raw_class_name=observation.raw_class_name or observation.class_name,
        )
        metrics = _discontinuity_metrics(previous_observation.bbox_xyxy, latest_conflict_observation.bbox_xyxy)
        reasons: list[str] = []
        if float(metrics["spatial_score"]) < float(self.config.identity_continuity.minimum_spatial_score):
            reasons.append("CLASS_CONFLICT_AND_LOW_SPATIAL_SCORE")
        if float(metrics["bbox_iou"]) <= float(self.config.class_conflict_split.maximum_iou_for_split):
            reasons.append("CLASS_CONFLICT_AND_LOW_IOU")
        if float(metrics["normalized_center_distance"]) >= float(self.config.class_conflict_split.minimum_normalized_center_distance_for_split):
            reasons.append("CLASS_CONFLICT_AND_CENTER_JUMP")
        if float(metrics["area_ratio"]) > float(self.config.identity_continuity.maximum_area_ratio):
            reasons.append("CLASS_CONFLICT_AND_AREA_CHANGE")
        if float(metrics["width_ratio"]) > float(self.config.class_conflict_split.maximum_width_ratio_for_split):
            reasons.append("CLASS_CONFLICT_AND_WIDTH_CHANGE")
        if float(metrics["height_ratio"]) > float(self.config.class_conflict_split.maximum_height_ratio_for_split):
            reasons.append("CLASS_CONFLICT_AND_HEIGHT_CHANGE")
        if self.config.class_conflict_split.require_spatial_discontinuity and not reasons:
            return None
        return _ConflictSplitDecision(
            conflict_start_index=conflict_start_index,
            split_frame=proposed_history[conflict_start_index].frame_number,
            conflicting_class=conflicting_class,
            consecutive_conflict_count=consecutive_count,
            average_conflict_confidence=average_conflict_confidence,
            reason_codes=reasons,
            bbox_iou=float(metrics["bbox_iou"]),
            center_distance=float(metrics["center_distance"]),
            normalized_center_distance=float(metrics["normalized_center_distance"]),
            area_ratio=float(metrics["area_ratio"]),
            width_ratio=float(metrics["width_ratio"]),
            height_ratio=float(metrics["height_ratio"]),
            spatial_score=float(metrics["spatial_score"]),
            stable_class_before_split=stable_class_name,
        )

    def _execute_class_conflict_split(
        self,
        *,
        camera_code: str,
        state: _TrackState,
        incoming_tracker_local_track_id: int,
        decision: _ConflictSplitDecision,
    ) -> tuple[LocalVehicleTrack, _TrackState]:
        original_observations = list(state.observations)
        old_observations = original_observations[: decision.conflict_start_index]
        new_observations = original_observations[decision.conflict_start_index :]
        if not old_observations or not new_observations:
            raise ValueError("Class-conflict split requires both old and new observation sequences.")
        state._rebuild_from_observations(old_observations, self.config)
        new_logical_track_id = self._allocate_logical_track_id(camera_code, incoming_tracker_local_track_id)
        new_track_uuid = build_track_uuid(camera_code, new_logical_track_id, self.run_id)
        new_state = _TrackState(
            track_uuid=new_track_uuid,
            camera_code=state.camera_code,
            camera_name=state.camera_name,
            source_path=state.source_path,
            local_track_id=new_logical_track_id,
            first_observation=TrackObservation(
                camera_code=new_observations[0].camera_code,
                local_track_id=new_logical_track_id,
                native_tracker_id=new_observations[0].native_tracker_id,
                frame_number=new_observations[0].frame_number,
                video_time_seconds=new_observations[0].video_time_seconds,
                camera_timestamp=new_observations[0].camera_timestamp,
                class_name=new_observations[0].class_name,
                confidence=new_observations[0].confidence,
                bbox_xyxy=new_observations[0].bbox_xyxy,
                track_uuid=new_track_uuid,
                state="tentative",
                raw_class_name=new_observations[0].raw_class_name or new_observations[0].class_name,
            ),
        )
        new_state.split_from_track_uuid = state.track_uuid
        new_state.source_logical_track_id = state.local_track_id
        new_state.linked_track_group_id = state.linked_track_group_id
        new_state._rebuild_from_observations(
            [
                TrackObservation(
                    camera_code=item.camera_code,
                    local_track_id=new_logical_track_id,
                    native_tracker_id=item.native_tracker_id,
                    frame_number=item.frame_number,
                    video_time_seconds=item.video_time_seconds,
                    camera_timestamp=item.camera_timestamp,
                    class_name=item.class_name,
                    confidence=item.confidence,
                    bbox_xyxy=item.bbox_xyxy,
                    track_uuid=new_track_uuid,
                    state=item.state,
                    raw_class_name=item.raw_class_name or item.class_name,
                )
                for item in new_observations
            ],
            self.config,
        )
        new_state.split_executed = True
        new_state.split_frame = decision.split_frame
        new_state.split_reason_codes = list(decision.reason_codes)
        new_state.source_logical_track_id = state.local_track_id
        new_state.new_logical_track_id = new_logical_track_id
        new_state.split_native_tracker_id = new_observations[0].native_tracker_id or incoming_tracker_local_track_id
        new_state.pending_conflict_observation_count = len(new_observations)
        new_state.stable_class_before_split = decision.stable_class_before_split
        new_state.conflicting_class = decision.conflicting_class
        new_state.average_conflict_confidence = decision.average_conflict_confidence
        new_state.bbox_iou_at_split = decision.bbox_iou
        new_state.center_distance_at_split = decision.center_distance
        new_state.normalized_center_distance_at_split = decision.normalized_center_distance
        new_state.area_ratio_at_split = decision.area_ratio
        new_state.width_ratio_at_split = decision.width_ratio
        new_state.height_ratio_at_split = decision.height_ratio
        new_state.spatial_score_at_split = decision.spatial_score
        for item in state.native_tracker_ids_seen:
            new_state._register_native_tracker_id(item)
        state.state = "completed"
        state.completion_reason = "identity_split"
        state.split_executed = True
        state.split_frame = decision.split_frame
        state.split_reason_codes = list(decision.reason_codes)
        state.source_logical_track_id = state.local_track_id
        state.new_logical_track_id = new_logical_track_id
        state.split_native_tracker_id = new_observations[0].native_tracker_id or incoming_tracker_local_track_id
        state.pending_conflict_observation_count = len(new_observations)
        state.stable_class_before_split = decision.stable_class_before_split
        state.conflicting_class = decision.conflicting_class
        state.average_conflict_confidence = decision.average_conflict_confidence
        state.bbox_iou_at_split = decision.bbox_iou
        state.center_distance_at_split = decision.center_distance
        state.normalized_center_distance_at_split = decision.normalized_center_distance
        state.area_ratio_at_split = decision.area_ratio
        state.width_ratio_at_split = decision.width_ratio
        state.height_ratio_at_split = decision.height_ratio
        state.spatial_score_at_split = decision.spatial_score
        return state.to_track(), new_state

    def _class_compatibility(self, left: str | None, right: str | None) -> float:
        normalized_left = normalize_track_class_name(left, self.config)
        normalized_right = normalize_track_class_name(right, self.config)
        if normalized_left == normalized_right:
            return 1.0
        for family_members in self.config.class_families.values():
            if normalized_left in family_members and normalized_right in family_members:
                return 0.5
        return 0.0

    def _finalize_identity_break(
        self,
        states: dict[int, _TrackState],
        tracker_local_track_id: int,
        state: _TrackState,
    ) -> LocalVehicleTrack:
        state.state = "discarded" if state.observation_count < self.config.min_confirmed_observations else "completed"
        state.completion_reason = "identity_split"
        track = state.to_track()
        del states[tracker_local_track_id]
        return track

    def _trailing_class_count(self, history: list[ClassObservation], target_class: str) -> int:
        count = 0
        for item in reversed(history):
            if item.class_name != target_class:
                break
            count += 1
        return count


def _bbox_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    left_x1, left_y1, left_x2, left_y2 = [float(value) for value in left]
    right_x1, right_y1, right_x2, right_y2 = [float(value) for value in right]
    intersection_x1 = max(left_x1, right_x1)
    intersection_y1 = max(left_y1, right_y1)
    intersection_x2 = min(left_x2, right_x2)
    intersection_y2 = min(left_y2, right_y2)
    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height
    left_area = max(0.0, (left_x2 - left_x1) * (left_y2 - left_y1))
    right_area = max(0.0, (right_x2 - right_x1) * (right_y2 - right_y1))
    denominator = max((left_area + right_area - intersection_area), 1e-6)
    return intersection_area / denominator


def _discontinuity_metrics(previous_bbox: tuple[float, float, float, float], next_bbox: tuple[float, float, float, float]) -> dict[str, float]:
    previous_width = max(float(previous_bbox[2]) - float(previous_bbox[0]), 1.0)
    previous_height = max(float(previous_bbox[3]) - float(previous_bbox[1]), 1.0)
    next_width = max(float(next_bbox[2]) - float(next_bbox[0]), 1.0)
    next_height = max(float(next_bbox[3]) - float(next_bbox[1]), 1.0)
    previous_center = ((float(previous_bbox[0]) + float(previous_bbox[2])) / 2.0, (float(previous_bbox[1]) + float(previous_bbox[3])) / 2.0)
    next_center = ((float(next_bbox[0]) + float(next_bbox[2])) / 2.0, (float(next_bbox[1]) + float(next_bbox[3])) / 2.0)
    center_distance = hypot(next_center[0] - previous_center[0], next_center[1] - previous_center[1])
    scale = max((previous_width + next_width) / 2.0, (previous_height + next_height) / 2.0, 1.0)
    spatial_score = max(0.0, 1.0 - (center_distance / (scale * 2.5)))
    return {
        "bbox_iou": _bbox_iou(previous_bbox, next_bbox),
        "center_distance": center_distance,
        "normalized_center_distance": center_distance / scale,
        "area_ratio": max(previous_width * previous_height, next_width * next_height) / max(min(previous_width * previous_height, next_width * next_height), 1.0),
        "width_ratio": max(previous_width, next_width) / max(min(previous_width, next_width), 1.0),
        "height_ratio": max(previous_height, next_height) / max(min(previous_height, next_height), 1.0),
        "spatial_score": spatial_score,
    }
