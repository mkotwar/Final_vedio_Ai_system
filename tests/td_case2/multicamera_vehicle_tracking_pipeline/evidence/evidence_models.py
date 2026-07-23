from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EvidenceCandidate:
    candidate_type: str
    frame_number: int
    video_time_seconds: float
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    crop_width: int
    crop_height: int
    area: int
    sharpness_score: float
    edge_penalty: float
    overall_score: float
    encoded_jpeg: bytes
    file_path: str | None = None


@dataclass(slots=True)
class TrackEvidencePackage:
    run_id: str
    camera_code: str
    local_track_id: int
    track_uuid: str
    class_name: str
    candidates: dict[str, EvidenceCandidate] = field(default_factory=dict)
    output_directory: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "camera_code": self.camera_code,
            "local_track_id": self.local_track_id,
            "track_uuid": self.track_uuid,
            "class_name": self.class_name,
            "output_directory": self.output_directory,
            "candidate_count": len(self.candidates),
            "candidates": {
                name: {
                    "candidate_type": candidate.candidate_type,
                    "frame_number": candidate.frame_number,
                    "video_time_seconds": candidate.video_time_seconds,
                    "confidence": candidate.confidence,
                    "bbox_xyxy": list(candidate.bbox_xyxy),
                    "crop_width": candidate.crop_width,
                    "crop_height": candidate.crop_height,
                    "area": candidate.area,
                    "sharpness_score": candidate.sharpness_score,
                    "edge_penalty": candidate.edge_penalty,
                    "overall_score": candidate.overall_score,
                    "file_path": candidate.file_path,
                }
                for name, candidate in sorted(self.candidates.items())
            },
        }
