from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..detection.detection_models import DetectionPacket
from .tracking_config import TrackingConfig
from .tracking_models import LocalVehicleTrack, TrackObservation


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
        )


class LocalTrackLifecycle:
    def __init__(self, config: TrackingConfig, *, run_id: str | None = None) -> None:
        self.config = config
        self.run_id = run_id
        self._states_by_camera: dict[str, dict[int, _TrackState]] = {}
        self._metadata_by_camera: dict[str, tuple[str, Path]] = {}
        self._last_frame_by_camera: dict[str, int] = {}
        self._completed_by_camera: dict[str, list[LocalVehicleTrack]] = {}

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
            visible_ids.add(observation.local_track_id)
            state = states.get(observation.local_track_id)
            if state is None:
                track_uuid = build_track_uuid(camera_code, observation.local_track_id, self.run_id)
                provisional = TrackObservation(
                    camera_code=observation.camera_code,
                    local_track_id=observation.local_track_id,
                    frame_number=observation.frame_number,
                    video_time_seconds=observation.video_time_seconds,
                    camera_timestamp=observation.camera_timestamp,
                    class_name=observation.class_name,
                    confidence=observation.confidence,
                    bbox_xyxy=observation.bbox_xyxy,
                    track_uuid=track_uuid,
                    state="tentative",
                )
                state = _TrackState(
                    track_uuid=track_uuid,
                    camera_code=camera_code,
                    camera_name=packet.camera_name,
                    source_path=packet.source_path,
                    local_track_id=observation.local_track_id,
                    first_observation=provisional,
                )
                states[observation.local_track_id] = state
            else:
                state.class_name = observation.class_name
                state.last_frame_number = observation.frame_number
                state.last_seen_at = observation.camera_timestamp
                state.last_video_time_seconds = observation.video_time_seconds
                state.observation_count += 1
                state.best_confidence = max(state.best_confidence, float(observation.confidence))
                state.lost_frame_count = 0
            if state.state == "temporarily_lost":
                state.state = "active"
            elif state.observation_count >= self.config.min_confirmed_observations:
                state.state = "active"
            base_observation = TrackObservation(
                camera_code=observation.camera_code,
                local_track_id=observation.local_track_id,
                frame_number=observation.frame_number,
                video_time_seconds=observation.video_time_seconds,
                camera_timestamp=observation.camera_timestamp,
                class_name=observation.class_name,
                confidence=observation.confidence,
                bbox_xyxy=observation.bbox_xyxy,
                track_uuid=state.track_uuid,
                state=state.state,
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
