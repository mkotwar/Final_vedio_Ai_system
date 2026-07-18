"""Descriptive tracking metrics from emitted packets, without ground-truth claims."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .schemas import DetectionPacket, TrackedFramePacket


class TrackingMetricsAccumulator:
    """Accumulate detection/tracking packet metrics for one sequential run."""

    def __init__(self, *, expected_physical_objects: int | None = None) -> None:
        self.expected_physical_objects = expected_physical_objects
        self.total_frames_processed = 0
        self.frames_with_detections = 0
        self.frames_with_tracks = 0
        self.total_detections = 0
        self.total_track_observations = 0
        self.maximum_simultaneous_tracks = 0
        self._track_observation_counts: Counter[int] = Counter()
        self._source_track_ids: dict[int, str | int | None] = {}
        self._first_seen_frame: dict[int, int] = {}
        self._last_seen_frame: dict[int, int] = {}
        self._first_seen_sec: dict[int, float] = {}
        self._last_seen_sec: dict[int, float] = {}
        self._seen_processed_positions: dict[int, list[int]] = defaultdict(list)
        self._processed_position = 0

    def update(self, detection_packet: DetectionPacket, tracked_packet: TrackedFramePacket) -> None:
        self.total_frames_processed += 1
        if detection_packet.detections:
            self.frames_with_detections += 1
        if tracked_packet.tracks:
            self.frames_with_tracks += 1
        self.total_detections += len(detection_packet.detections)
        self.total_track_observations += len(tracked_packet.tracks)
        self.maximum_simultaneous_tracks = max(self.maximum_simultaneous_tracks, len(tracked_packet.tracks))
        for track in tracked_packet.tracks:
            self._track_observation_counts[track.track_id] += 1
            self._source_track_ids[track.track_id] = track.source_track_id
            self._first_seen_frame.setdefault(track.track_id, track.frame_index)
            self._first_seen_sec.setdefault(track.track_id, track.timestamp_sec)
            self._last_seen_frame[track.track_id] = track.frame_index
            self._last_seen_sec[track.track_id] = track.timestamp_sec
            self._seen_processed_positions[track.track_id].append(self._processed_position)
        self._processed_position += 1

    def to_dict(self) -> dict[str, Any]:
        duration_by_track = {
            str(track_id): round(self._last_seen_sec[track_id] - self._first_seen_sec[track_id], 6)
            for track_id in sorted(self._track_observation_counts)
        }
        gaps = {}
        for track_id, positions in self._seen_processed_positions.items():
            gap_count = 0
            for left, right in zip(positions, positions[1:]):
                if right - left > 1:
                    gap_count += right - left - 1
            gaps[str(track_id)] = gap_count
        return {
            "total_frames_processed": self.total_frames_processed,
            "frames_with_detections": self.frames_with_detections,
            "frames_with_tracks": self.frames_with_tracks,
            "total_detections": self.total_detections,
            "total_track_observations": self.total_track_observations,
            "unique_track_ids": len(self._track_observation_counts),
            "unique_source_track_ids": len({value for value in self._source_track_ids.values() if value is not None}),
            "track_observation_counts": {str(key): self._track_observation_counts[key] for key in sorted(self._track_observation_counts)},
            "first_seen_frame_by_track": {str(key): self._first_seen_frame[key] for key in sorted(self._first_seen_frame)},
            "last_seen_frame_by_track": {str(key): self._last_seen_frame[key] for key in sorted(self._last_seen_frame)},
            "duration_sec_by_track": duration_by_track,
            "tracks_under_0_5_sec": sum(1 for value in duration_by_track.values() if value < 0.5),
            "tracks_under_1_0_sec": sum(1 for value in duration_by_track.values() if value < 1.0),
            "maximum_simultaneous_tracks": self.maximum_simultaneous_tracks,
            "average_tracks_per_processed_frame": round(
                self.total_track_observations / self.total_frames_processed,
                6,
            )
            if self.total_frames_processed
            else 0.0,
            "track_gaps_by_id": gaps,
            "gap_note": "A gap is not automatically an ID switch and is not automatically successful ReID.",
            "expected_physical_objects": self.expected_physical_objects,
            "unique_tracks_minus_expected_physical_objects": (
                len(self._track_observation_counts) - self.expected_physical_objects
                if self.expected_physical_objects is not None
                else None
            ),
        }
