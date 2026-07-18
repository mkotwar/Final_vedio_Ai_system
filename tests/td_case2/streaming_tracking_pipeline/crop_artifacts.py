from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .observations import TrackObservation
from .schemas import CropCandidate, TrackRecord
from .serialization import dataclass_to_dict, to_json_safe, write_json


def _cv2() -> Any:
    try:
        import cv2  # type: ignore

        return cv2
    except Exception:
        return None


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return slug or "source"


@dataclass(frozen=True)
class CompletedTrackCropBundle:
    """JSON-safe crop candidate bundle emitted when a lifecycle track completes."""

    source_id: str
    track_id: int
    track_generation: int
    source_track_id: str | int | None
    completion_reason: str | None
    lifecycle_status: str
    observation_count: int
    retained_candidate_count: int
    candidates: list[CropCandidate] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_track(cls, track: TrackRecord, candidates: list[CropCandidate]) -> "CompletedTrackCropBundle":
        return cls(
            source_id=track.source_id,
            track_id=track.track_id,
            track_generation=track.track_generation,
            source_track_id=track.source_track_id,
            completion_reason=track.completion_reason.value if track.completion_reason is not None else None,
            lifecycle_status=track.status.value,
            observation_count=track.observation_count,
            retained_candidate_count=len(candidates),
            candidates=list(candidates),
            metadata={"dominant_class": track.dominant_class},
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


class CropImageWriter:
    """Optional deterministic image writer for Step 5 crop candidates."""

    def __init__(self, root_dir: str | Path, *, enabled: bool = True, overwrite: bool = False) -> None:
        self.root_dir = Path(root_dir)
        self.enabled = enabled
        self.overwrite = overwrite
        self.images_dir = self.root_dir / "05_crops" / "images"
        if self.enabled:
            self.images_dir.mkdir(parents=True, exist_ok=True)

    def write_crop(self, *, observation: TrackObservation, crop: Any) -> str | None:
        if not self.enabled:
            return None
        cv2 = _cv2()
        if cv2 is None:
            raise RuntimeError("cv2 is required to save crop images.")
        track_dir = (
            self.images_dir
            / _safe_slug(observation.source_id)
            / f"track_{observation.track_id:06d}_gen_{observation.track_generation:03d}"
        )
        track_dir.mkdir(parents=True, exist_ok=True)
        path = track_dir / f"frame_{observation.frame_index:08d}_crop.jpg"
        if path.exists() and not self.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing crop image: {path}")
        success = bool(cv2.imwrite(str(path), crop))
        if not success:
            raise RuntimeError(f"cv2.imwrite failed for crop image: {path}")
        return str(path)

    def write_full_frame(self, *, observation: TrackObservation, frame: Any) -> str | None:
        if not self.enabled:
            return None
        cv2 = _cv2()
        if cv2 is None:
            raise RuntimeError("cv2 is required to save full-frame images.")
        frame_dir = self.images_dir / _safe_slug(observation.source_id) / "full_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        path = frame_dir / f"frame_{observation.frame_index:08d}.jpg"
        if path.exists():
            return str(path)
        success = bool(cv2.imwrite(str(path), frame))
        if not success:
            raise RuntimeError(f"cv2.imwrite failed for full-frame image: {path}")
        return str(path)


class CropArtifactSink:
    """Write Step 5 observation, candidate, and completed-bundle artifacts."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.crop_dir = self.run_dir / "05_crops"
        self.report_dir = self.run_dir / "reports"
        self.crop_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self._handles = {
            "observations": (self.crop_dir / "track_observations.jsonl").open("w", encoding="utf-8", newline="\n"),
            "candidates": (self.crop_dir / "crop_candidates.jsonl").open("w", encoding="utf-8", newline="\n"),
            "completed": (self.crop_dir / "completed_track_crop_bundles.jsonl").open("w", encoding="utf-8", newline="\n"),
        }
        self.counts = {"observations": 0, "candidates": 0, "completed": 0}
        self.closed = False

    def write_observation(self, observation: TrackObservation) -> None:
        self._write("observations", observation)

    def write_candidate(self, candidate: CropCandidate) -> None:
        self._write("candidates", candidate)

    def write_completed_bundle(self, bundle: CompletedTrackCropBundle) -> None:
        self._write("completed", bundle)

    def write_summary(self, payload: dict[str, Any]) -> None:
        write_json(self.crop_dir / "crop_collection_summary.json", payload)
        write_json(self.report_dir / "step5_crop_collection_report.json", payload)

    def close(self) -> None:
        if self.closed:
            return
        for handle in self._handles.values():
            handle.flush()
            handle.close()
        self.closed = True

    def _write(self, name: str, value: Any) -> None:
        if self.closed:
            raise RuntimeError("Cannot write to closed CropArtifactSink.")
        self._handles[name].write(json.dumps(to_json_safe(value), ensure_ascii=False, sort_keys=True))
        self._handles[name].write("\n")
        self._handles[name].flush()
        self.counts[name] += 1
