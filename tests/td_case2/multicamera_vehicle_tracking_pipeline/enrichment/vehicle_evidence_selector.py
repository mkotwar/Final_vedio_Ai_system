from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..tracking.tracking_models import LocalVehicleTrack
from .plate_models import VehicleEvidenceInput


def select_vehicle_evidence_candidates(
    *,
    completed_track: LocalVehicleTrack,
    configured_roles: Sequence[str],
    maximum_candidates: int,
    artifact_root: Path,
) -> list[VehicleEvidenceInput]:
    evidence = completed_track.evidence_package
    if evidence is None:
        return []
    selected: list[VehicleEvidenceInput] = []
    seen_paths: set[Path] = set()
    for role in configured_roles:
        if len(selected) >= maximum_candidates:
            break
        candidate = evidence.candidates.get(str(role).lower())
        if candidate is None:
            continue
        if not candidate.file_path:
            continue
        file_path = Path(candidate.file_path).expanduser().resolve()
        if file_path in seen_paths:
            continue
        if not file_path.exists():
            continue
        relative_storage_uri = file_path.relative_to(artifact_root.resolve()).as_posix()
        selected.append(
            VehicleEvidenceInput(
                track_uuid=completed_track.track_uuid,
                camera_code=completed_track.camera_code,
                source_vehicle_role=str(role).upper(),
                source_vehicle_storage_uri=relative_storage_uri,
                local_file_path=file_path,
                frame_number=candidate.frame_number,
                video_time_seconds=candidate.video_time_seconds,
                confidence=candidate.confidence,
                bbox_xyxy=candidate.bbox_xyxy,
                crop_width=candidate.crop_width,
                crop_height=candidate.crop_height,
                sharpness_score=candidate.sharpness_score,
                edge_penalty=candidate.edge_penalty,
                overall_score=candidate.overall_score,
                source_image_kind="VEHICLE_CROP",
                metadata={
                    "candidate_type": candidate.candidate_type,
                    "track_output_directory": evidence.output_directory,
                    "track_class_name": completed_track.class_name,
                    "source_path": str(completed_track.source_path) if completed_track.source_path is not None else None,
                },
            )
        )
        seen_paths.add(file_path)
    return selected
