from __future__ import annotations

from dataclasses import dataclass, field


PERSISTENCE_STATUSES = ("inserted", "already_exists", "skipped_discarded", "skipped_invalid_state", "dry_run", "failed")


@dataclass(slots=True)
class TrackPersistenceResult:
    track_uuid: str
    status: str
    database_track_id: str | None
    observations_written: int
    error: str | None = None


@dataclass(slots=True)
class PersistenceRunMetrics:
    cameras_synced: int = 0
    tracks_considered: int = 0
    tracks_inserted: int = 0
    tracks_already_existing: int = 0
    tracks_skipped_discarded: int = 0
    tracks_skipped_invalid_state: int = 0
    tracks_failed: int = 0
    observations_written: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "cameras_synced": self.cameras_synced,
            "tracks_considered": self.tracks_considered,
            "tracks_inserted": self.tracks_inserted,
            "tracks_already_existing": self.tracks_already_existing,
            "tracks_skipped_discarded": self.tracks_skipped_discarded,
            "tracks_skipped_invalid_state": self.tracks_skipped_invalid_state,
            "tracks_failed": self.tracks_failed,
            "observations_written": self.observations_written,
            "errors": list(self.errors),
        }
