from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ..enrichment.anpr_config import load_anpr_config
from ..enrichment.anpr_enrichment_service import AnprEnrichmentService
from ..enrichment.florence_plate_ocr_extractor import FlorencePlateOcrExtractor
from ..evidence.evidence_models import EvidenceCandidate, TrackEvidencePackage
from ..models.florence_runtime_factory import FlorenceRuntimeFactory
from ..models.plate_detector_runtime_factory import PlateDetectorRuntimeFactory
from ..persistence.persistence_config import PersistenceConfig
from ..tracking.tracking_models import LocalVehicleTrack, TrackObservation


KNOWN_EVIDENCE_ROLES = (
    "best_overall",
    "highest_confidence",
    "largest",
    "sharpest",
    "first",
    "middle",
    "last",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate enhanced ANPR on an existing artifact run.")
    parser.add_argument("--run-code", dest="run_code", default=None)
    parser.add_argument("--run-id", dest="run_id", default=None)
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--camera-code", default=None)
    parser.add_argument("--track-uuid", default=None)
    parser.add_argument("--max-tracks", type=int, default=None)
    parser.add_argument("--anpr-config", required=True)
    parser.add_argument("--florence-config", required=True)
    parser.add_argument("--plate-detector-model-path", default=None)
    parser.add_argument("--florence-model-path", default=None)
    parser.add_argument("--florence-adapter-path", default=None)
    parser.add_argument("--florence-processor-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--persist", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--visual-output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_code = args.run_code or args.run_id
    if not run_code:
        raise ValueError("Either --run-code or --run-id is required.")
    artifact_root = Path(args.artifact_root).expanduser().resolve()
    run_root = artifact_root / run_code
    anpr_config = load_anpr_config(args.anpr_config, overrides={"enabled": True, "persist_result": bool(args.persist)})
    from ..enrichment.florence_config import load_florence_config

    florence_config = load_florence_config(args.florence_config, overrides={"enabled": True})
    project_root = Path.cwd()
    plate_factory = PlateDetectorRuntimeFactory(project_root=project_root)
    florence_factory = FlorenceRuntimeFactory(project_root=project_root)
    plate_runtime = plate_factory.get_runtime(
        config=anpr_config,
        model_path_cli=args.plate_detector_model_path,
        device_override=args.device,
    )
    florence_runtime = florence_factory.get_runtime(
        config=florence_config,
        model_path_cli=args.florence_model_path,
        adapter_path_cli=args.florence_adapter_path,
        processor_path_cli=args.florence_processor_path,
        device_override=args.device,
    )
    if plate_runtime is None or florence_runtime is None:
        raise RuntimeError("Both plate detector and Florence runtime are required for standalone ANPR validation.")
    from ..enrichment.plate_candidate_collector import PlateCandidateCollector

    candidate_collector = PlateCandidateCollector(detector_runtime=plate_runtime, config=anpr_config, artifact_root=artifact_root)
    ocr_extractor = FlorencePlateOcrExtractor(runtime=florence_runtime, ocr_config=anpr_config.ocr, validation_config=anpr_config.validation)
    persistence_config = PersistenceConfig(backend="analytics_supabase" if args.persist and not args.dry_run else "dry_run", dry_run=not args.persist or args.dry_run)
    service = AnprEnrichmentService(
        config=anpr_config,
        persistence_config=persistence_config,
        artifact_root=artifact_root,
        candidate_collector=candidate_collector,
        ocr_extractor=ocr_extractor,
    )
    track_directories = _discover_track_directories(run_root=run_root, camera_code=args.camera_code, track_uuid=args.track_uuid)
    if args.max_tracks is not None:
        track_directories = track_directories[: int(args.max_tracks)]
    visual_output_dir = Path(args.visual_output_dir).expanduser().resolve() if args.visual_output_dir else None
    results: list[dict[str, object]] = []
    for track_dir in track_directories:
        track = _build_track_from_artifacts(track_dir=track_dir, artifact_root=artifact_root, run_code=run_code)
        result = service.enrich_track(completed_track=track, persisted_vehicle_track_id=f"DRYRUN:TRACK:{track.track_uuid}")
        payload = {
            "track_uuid": track.track_uuid,
            "camera_code": track.camera_code,
            "evidence_roles_examined": sorted(track.evidence_package.candidates.keys()) if track.evidence_package is not None else [],
            "vehicle_crop_directory": str(track_dir.resolve().relative_to(artifact_root)),
            "anpr": result.to_report_payload(),
        }
        if visual_output_dir is not None:
            visual_record_dir = visual_output_dir / track.camera_code / str(track.local_track_id)
            visual_record_dir.mkdir(parents=True, exist_ok=True)
            for role_name, candidate in (track.evidence_package.candidates if track.evidence_package is not None else {}).items():
                if candidate.file_path:
                    source_image = Path(candidate.file_path)
                    if source_image.exists():
                        shutil.copy2(source_image, visual_record_dir / f"{role_name}.jpg")
            if result.plate_storage_uri:
                selected_plate = artifact_root / result.plate_storage_uri
                if selected_plate.exists():
                    shutil.copy2(selected_plate, visual_record_dir / "selected_plate_crop.jpg")
        results.append(payload)
    report = {
        "run_code": run_code,
        "artifact_root": str(artifact_root),
        "tracks_considered": len(track_directories),
        "results": results,
        "metrics": service.get_metrics().to_dict(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "persist" if args.persist and not args.dry_run else "dry_run",
    }
    output_path = Path(args.output_report).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _discover_track_directories(*, run_root: Path, camera_code: str | None, track_uuid: str | None) -> list[Path]:
    if not run_root.exists():
        raise FileNotFoundError(f"Run artifact directory does not exist: {run_root}")
    if track_uuid:
        return [_track_uuid_to_directory(run_root, track_uuid)]
    directories = sorted(path for path in run_root.glob("CAM_*\\track_*\\*") if path.is_dir())
    if camera_code:
        directories = [path for path in directories if path.parent.parent.name == camera_code]
    return directories


def _track_uuid_to_directory(run_root: Path, track_uuid: str) -> Path:
    parts = track_uuid.split(":")
    if len(parts) < 3:
        raise ValueError(f"Track UUID is not in the expected format: {track_uuid}")
    camera_code = parts[-2]
    track_token = parts[-1]
    track_number = "".join(char for char in track_token if char.isdigit())
    candidate = run_root / camera_code / f"track_{int(track_number):06d}" / track_uuid.replace(":", "_")
    if not candidate.exists():
        raise FileNotFoundError(f"Track artifact directory does not exist: {candidate}")
    return candidate


def _build_track_from_artifacts(*, track_dir: Path, artifact_root: Path, run_code: str) -> LocalVehicleTrack:
    camera_code = track_dir.parent.parent.name
    local_track_id = int("".join(char for char in track_dir.name if char.isdigit()) or "0")
    observation = TrackObservation(
        camera_code=camera_code,
        local_track_id=local_track_id,
        frame_number=0,
        video_time_seconds=0.0,
        camera_timestamp=datetime.now(timezone.utc),
        class_name="UNKNOWN",
        confidence=1.0,
        bbox_xyxy=(0.0, 0.0, 1.0, 1.0),
        track_uuid=track_dir.name.replace("_TRACK_", ":TRACK_").replace("_CAM_", ":CAM_").replace(run_code + "_", run_code + ":"),
        state="active",
    )
    track = LocalVehicleTrack(
        track_uuid=observation.track_uuid,
        camera_code=camera_code,
        local_track_id=local_track_id,
        class_name="3WHEELER" if camera_code == "CAM_003" else "CAR",
        first_frame_number=0,
        last_frame_number=0,
        first_seen_at=observation.camera_timestamp,
        last_seen_at=observation.camera_timestamp,
        first_video_time_seconds=0.0,
        last_video_time_seconds=0.0,
        observation_count=1,
        best_confidence=1.0,
        state="completed",
        observations=[observation],
        camera_name=camera_code,
        source_path=Path("unknown.mp4"),
    )
    candidates: dict[str, EvidenceCandidate] = {}
    for role_index, role in enumerate(KNOWN_EVIDENCE_ROLES, start=1):
        image_path = track_dir / f"{role}.jpg"
        if not image_path.exists():
            continue
        candidates[role] = EvidenceCandidate(
            candidate_type=role,
            frame_number=role_index,
            video_time_seconds=float(role_index) / 10.0,
            confidence=1.0,
            bbox_xyxy=(0.0, 0.0, 1.0, 1.0),
            crop_width=1,
            crop_height=1,
            area=1,
            sharpness_score=1.0,
            edge_penalty=0.0,
            overall_score=1.0,
            encoded_jpeg=b"",
            file_path=str(image_path),
        )
    track.evidence_package = TrackEvidencePackage(
        run_id=run_code,
        camera_code=camera_code,
        local_track_id=local_track_id,
        track_uuid=track.track_uuid,
        class_name=track.class_name,
        candidates=candidates,
        output_directory=str(track_dir),
    )
    return track


if __name__ == "__main__":
    main()
