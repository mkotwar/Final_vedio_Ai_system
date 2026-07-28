from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..evidence.evidence_models import TrackEvidencePackage
from .persistence_models import TrackMediaRecord
from .track_media_types import TRACK_MEDIA_TYPE_ANNOTATED_FULL_FRAME, TRACK_MEDIA_TYPE_FULL_FRAME
from .track_media_mapping import ROLE_TO_SELECTION_RANK, normalize_track_media_role, normalize_track_media_type


def _to_portable_relative_path(path: Path, artifact_root: Path) -> str:
    resolved_root = artifact_root.resolve()
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except Exception as exc:
        raise ValueError(f"Evidence file is outside artifact root: {resolved_path}") from exc
    relative_str = relative.as_posix()
    if relative_str.startswith("/") or ".." in relative.parts:
        raise ValueError(f"Unsafe relative evidence path: {relative_str}")
    return relative_str


def build_track_media_records(
    *,
    evidence_package: TrackEvidencePackage,
    vehicle_track_id: str,
    camera_id: str,
    artifact_root: Path,
    persist_roles: Sequence[str],
) -> list[TrackMediaRecord]:
    records: list[TrackMediaRecord] = []
    persisted_source_frames: set[tuple[str, int | None]] = set()
    persisted_annotated_frames: set[tuple[str, int | None]] = set()
    for role in persist_roles:
        canonical_role = normalize_track_media_role(role)
        candidate_key = canonical_role.lower()
        candidate = evidence_package.candidates.get(candidate_key)
        if candidate is None:
            continue
        if candidate.file_path is None:
            raise FileNotFoundError(f"Evidence candidate '{candidate_key}' has no saved file path.")
        file_path = Path(candidate.file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Evidence file does not exist: {file_path}")
        relative_path = _to_portable_relative_path(file_path, artifact_root)
        records.append(
            TrackMediaRecord(
                vehicle_track_id=vehicle_track_id,
                media_type=normalize_track_media_type(canonical_role),
                storage_uri=relative_path,
                storage_provider="LOCAL",
                mime_type="image/jpeg",
                file_size_bytes=file_path.stat().st_size,
                frame_number=candidate.frame_number,
                captured_at=None,
                video_time_seconds=candidate.video_time_seconds,
                bbox={
                    "bbox_xyxy": list(candidate.bbox_xyxy),
                },
                width=candidate.crop_width,
                height=candidate.crop_height,
                quality_score=min(max(float(candidate.confidence), 0.0), 1.0),
                sharpness_score=None,
                visibility_score=max(0.0, min(1.0, 1.0 - float(candidate.edge_penalty))),
                occlusion_score=None,
                selection_rank=ROLE_TO_SELECTION_RANK[canonical_role],
                is_primary=canonical_role == "BEST_OVERALL",
                metadata={
                    "camera_id": camera_id,
                    "track_uuid": evidence_package.track_uuid,
                    "local_track_id": evidence_package.local_track_id,
                    "candidate_type": candidate_key,
                    "confidence": candidate.confidence,
                    "bbox_xyxy": list(candidate.bbox_xyxy),
                    "sharpness_score_raw": candidate.sharpness_score,
                    "edge_penalty": candidate.edge_penalty,
                    "overall_score": candidate.overall_score,
                    "crop_area": candidate.area,
                },
            )
        )
        source_key = (candidate.source_frame_path or "", candidate.frame_number)
        if candidate.source_frame_path and source_key not in persisted_source_frames:
            source_frame_path = Path(candidate.source_frame_path)
            if not source_frame_path.exists():
                raise FileNotFoundError(f"Full-frame evidence file does not exist: {source_frame_path}")
            relative_path = _to_portable_relative_path(source_frame_path, artifact_root)
            records.append(
                TrackMediaRecord(
                    vehicle_track_id=vehicle_track_id,
                    media_type=TRACK_MEDIA_TYPE_FULL_FRAME,
                    storage_uri=relative_path,
                    storage_provider="LOCAL",
                    mime_type="image/jpeg",
                    file_size_bytes=source_frame_path.stat().st_size,
                    frame_number=candidate.frame_number,
                    captured_at=None,
                    video_time_seconds=candidate.video_time_seconds,
                    bbox={"bbox_xyxy": list(candidate.original_bbox_xyxy)},
                    width=candidate.source_frame_width,
                    height=candidate.source_frame_height,
                    quality_score=None,
                    sharpness_score=None,
                    visibility_score=candidate.visibility_score,
                    occlusion_score=None,
                    selection_rank=ROLE_TO_SELECTION_RANK[canonical_role],
                    is_primary=canonical_role == "BEST_OVERALL",
                    metadata={
                        "camera_id": camera_id,
                        "track_uuid": evidence_package.track_uuid,
                        "local_track_id": evidence_package.local_track_id,
                        "source_role": canonical_role,
                        "bbox_xyxy": list(candidate.original_bbox_xyxy),
                    },
                )
            )
            persisted_source_frames.add(source_key)
        annotated_key = (candidate.annotated_frame_path or "", candidate.frame_number)
        if candidate.annotated_frame_path and annotated_key not in persisted_annotated_frames:
            annotated_frame_path = Path(candidate.annotated_frame_path)
            if not annotated_frame_path.exists():
                raise FileNotFoundError(f"Annotated full-frame evidence file does not exist: {annotated_frame_path}")
            relative_path = _to_portable_relative_path(annotated_frame_path, artifact_root)
            records.append(
                TrackMediaRecord(
                    vehicle_track_id=vehicle_track_id,
                    media_type=TRACK_MEDIA_TYPE_ANNOTATED_FULL_FRAME,
                    storage_uri=relative_path,
                    storage_provider="LOCAL",
                    mime_type="image/jpeg",
                    file_size_bytes=annotated_frame_path.stat().st_size,
                    frame_number=candidate.frame_number,
                    captured_at=None,
                    video_time_seconds=candidate.video_time_seconds,
                    bbox={"bbox_xyxy": list(candidate.original_bbox_xyxy)},
                    width=candidate.source_frame_width,
                    height=candidate.source_frame_height,
                    quality_score=None,
                    sharpness_score=None,
                    visibility_score=candidate.visibility_score,
                    occlusion_score=None,
                    selection_rank=ROLE_TO_SELECTION_RANK[canonical_role],
                    is_primary=canonical_role == "BEST_OVERALL",
                    metadata={
                        "camera_id": camera_id,
                        "track_uuid": evidence_package.track_uuid,
                        "local_track_id": evidence_package.local_track_id,
                        "source_role": canonical_role,
                        "bbox_xyxy": list(candidate.original_bbox_xyxy),
                    },
                )
            )
            persisted_annotated_frames.add(annotated_key)
    legacy_source_key = (evidence_package.full_frame_path or "", evidence_package.full_frame_frame_number)
    if evidence_package.full_frame_path and legacy_source_key not in persisted_source_frames:
        full_frame_path = Path(evidence_package.full_frame_path)
        if full_frame_path.exists():
            relative_path = _to_portable_relative_path(full_frame_path, artifact_root)
            bbox_metadata = list(evidence_package.full_frame_bbox_xyxy) if evidence_package.full_frame_bbox_xyxy is not None else None
            records.append(
                TrackMediaRecord(
                    vehicle_track_id=vehicle_track_id,
                    media_type=TRACK_MEDIA_TYPE_FULL_FRAME,
                    storage_uri=relative_path,
                    storage_provider="LOCAL",
                    mime_type="image/jpeg",
                    file_size_bytes=full_frame_path.stat().st_size,
                    frame_number=evidence_package.full_frame_frame_number,
                    captured_at=None,
                    video_time_seconds=evidence_package.full_frame_video_time_seconds,
                    bbox={"bbox_xyxy": bbox_metadata} if bbox_metadata is not None else None,
                    width=evidence_package.full_frame_width,
                    height=evidence_package.full_frame_height,
                    quality_score=None,
                    sharpness_score=None,
                    visibility_score=None,
                    occlusion_score=None,
                    selection_rank=5,
                    is_primary=False,
                    metadata={
                        "camera_id": camera_id,
                        "track_uuid": evidence_package.track_uuid,
                        "local_track_id": evidence_package.local_track_id,
                        "source_role": "BEST_OVERALL",
                        "bbox_xyxy": bbox_metadata,
                    },
                )
            )
    return records
