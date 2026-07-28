from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EvidenceCandidate:
    candidate_type: str
    frame_number: int
    video_time_seconds: float
    confidence: float
    original_bbox_xyxy: tuple[float, float, float, float]
    expanded_bbox_xyxy: tuple[float, float, float, float]
    bbox_xyxy: tuple[float, float, float, float]
    crop_width: int
    crop_height: int
    area: int
    sharpness_score: float
    visibility_score: float
    centeredness_score: float
    visible_bbox_ratio: float
    edge_penalty: float
    overall_score: float
    crop_clipped: bool
    touches_left_edge: bool
    touches_right_edge: bool
    touches_top_edge: bool
    touches_bottom_edge: bool
    encoded_jpeg: bytes
    file_path: str | None = None
    source_frame_path: str | None = None
    annotated_frame_path: str | None = None
    source_frame_jpeg: bytes | None = None
    source_frame_width: int | None = None
    source_frame_height: int | None = None
    frame_observations: tuple[dict[str, object], ...] = ()


@dataclass(slots=True)
class TrackEvidencePackage:
    run_id: str
    camera_code: str
    local_track_id: int
    track_uuid: str
    class_name: str
    candidates: dict[str, EvidenceCandidate] = field(default_factory=dict)
    output_directory: str | None = None
    manifest_path: str | None = None
    full_frame_path: str | None = None
    annotated_full_frame_path: str | None = None
    full_frame_frame_number: int | None = None
    full_frame_video_time_seconds: float | None = None
    full_frame_bbox_xyxy: tuple[float, float, float, float] | None = None
    full_frame_width: int | None = None
    full_frame_height: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "camera_code": self.camera_code,
            "local_track_id": self.local_track_id,
            "track_uuid": self.track_uuid,
            "class_name": self.class_name,
            "output_directory": self.output_directory,
            "manifest_path": self.manifest_path,
            "full_frame_path": self.full_frame_path,
            "annotated_full_frame_path": self.annotated_full_frame_path,
            "full_frame_frame_number": self.full_frame_frame_number,
            "full_frame_video_time_seconds": self.full_frame_video_time_seconds,
            "full_frame_bbox_xyxy": list(self.full_frame_bbox_xyxy) if self.full_frame_bbox_xyxy is not None else None,
            "full_frame_width": self.full_frame_width,
            "full_frame_height": self.full_frame_height,
            "candidate_count": len(self.candidates),
            "candidates": {
                name: {
                    "candidate_type": candidate.candidate_type,
                    "frame_number": candidate.frame_number,
                    "video_time_seconds": candidate.video_time_seconds,
                    "confidence": candidate.confidence,
                    "original_bbox_xyxy": list(candidate.original_bbox_xyxy),
                    "expanded_bbox_xyxy": list(candidate.expanded_bbox_xyxy),
                    "bbox_xyxy": list(candidate.bbox_xyxy),
                    "crop_width": candidate.crop_width,
                    "crop_height": candidate.crop_height,
                    "area": candidate.area,
                    "sharpness_score": candidate.sharpness_score,
                    "visibility_score": candidate.visibility_score,
                    "centeredness_score": candidate.centeredness_score,
                    "visible_bbox_ratio": candidate.visible_bbox_ratio,
                    "edge_penalty": candidate.edge_penalty,
                    "overall_score": candidate.overall_score,
                    "crop_clipped": candidate.crop_clipped,
                    "touches_left_edge": candidate.touches_left_edge,
                    "touches_right_edge": candidate.touches_right_edge,
                    "touches_top_edge": candidate.touches_top_edge,
                    "touches_bottom_edge": candidate.touches_bottom_edge,
                    "file_path": candidate.file_path,
                    "source_frame_path": candidate.source_frame_path,
                    "annotated_frame_path": candidate.annotated_frame_path,
                    "source_frame_width": candidate.source_frame_width,
                    "source_frame_height": candidate.source_frame_height,
                }
                for name, candidate in sorted(self.candidates.items())
            },
        }
