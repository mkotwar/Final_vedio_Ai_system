from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ..enrichment.anpr_config import load_anpr_config
from ..enrichment.anpr_enrichment_service import AnprEnrichmentService
from ..enrichment.florence_plate_ocr_extractor import FlorencePlateOcrExtractor
from ..enrichment.plate_candidate_collector import PlateCandidateCollector
from ..evidence.evidence_models import EvidenceCandidate, TrackEvidencePackage
from ..models.florence_runtime_factory import FlorenceRuntimeFactory
from ..models.plate_detector_runtime_factory import PlateDetectorRuntimeFactory
from ..persistence.persistence_config import PersistenceConfig
from ..tracking.tracking_models import LocalVehicleTrack, TrackObservation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ANPR on an existing completed run without database writes.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--camera-code", default=None)
    parser.add_argument("--max-tracks", type=int, default=None)
    parser.add_argument("--anpr-config", required=True)
    parser.add_argument("--florence-config", required=True)
    parser.add_argument("--plate-detector-model-path", default=None)
    parser.add_argument("--florence-model-path", default=None)
    parser.add_argument("--florence-adapter-path", default=None)
    parser.add_argument("--florence-processor-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--visual-output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_root = Path(args.artifact_root).expanduser().resolve()
    run_root = artifact_root / args.run_id
    anpr_config = load_anpr_config(args.anpr_config, overrides={"enabled": True})
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
    candidate_collector = PlateCandidateCollector(detector_runtime=plate_runtime, config=anpr_config, artifact_root=artifact_root)
    ocr_extractor = FlorencePlateOcrExtractor(runtime=florence_runtime, ocr_config=anpr_config.ocr, validation_config=anpr_config.validation)
    service = AnprEnrichmentService(
        config=anpr_config,
        persistence_config=PersistenceConfig(backend="dry_run", dry_run=True),
        artifact_root=artifact_root,
        candidate_collector=candidate_collector,
        ocr_extractor=ocr_extractor,
    )
    best_overall_images = sorted(run_root.glob("**/best_overall.jpg"))
    if args.camera_code:
        best_overall_images = [path for path in best_overall_images if args.camera_code in path.parts]
    if args.max_tracks is not None:
        best_overall_images = best_overall_images[: int(args.max_tracks)]
    visual_output_dir = Path(args.visual_output_dir).expanduser().resolve() if args.visual_output_dir else None
    results: list[dict[str, object]] = []
    for image_path in best_overall_images:
        track = _build_track_from_vehicle_crop(image_path=image_path, artifact_root=artifact_root, run_id=args.run_id)
        result = service.enrich_track(completed_track=track, persisted_vehicle_track_id=f"DRYRUN:TRACK:{track.track_uuid}")
        payload = {
            "track_uuid": track.track_uuid,
            "camera_code": track.camera_code,
            "vehicle_crop": image_path.resolve().relative_to(artifact_root).as_posix(),
            "anpr": result.to_report_payload(),
        }
        if visual_output_dir is not None:
            visual_record_dir = visual_output_dir / track.camera_code / str(track.local_track_id)
            visual_record_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, visual_record_dir / "source_vehicle_crop.jpg")
            if result.plate_storage_uri:
                selected_plate = artifact_root / result.plate_storage_uri
                if selected_plate.exists():
                    shutil.copy2(selected_plate, visual_record_dir / "selected_plate_crop.jpg")
        results.append(payload)
    report = {
        "run_id": args.run_id,
        "artifact_root": str(artifact_root),
        "tracks_considered": len(best_overall_images),
        "results": results,
        "metrics": service.get_metrics().to_dict(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    output_path = Path(args.output_report).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _build_track_from_vehicle_crop(*, image_path: Path, artifact_root: Path, run_id: str) -> LocalVehicleTrack:
    relative = image_path.resolve().relative_to(artifact_root)
    parts = list(relative.parts)
    camera_code = next((part for part in parts if part.startswith("CAM_")), "UNKNOWN_CAMERA")
    track_dir = image_path.parent
    local_track_id = _infer_local_track_id(track_dir.name)
    observation = TrackObservation(
        camera_code=camera_code,
        local_track_id=local_track_id,
        frame_number=0,
        video_time_seconds=0.0,
        camera_timestamp=datetime.now(timezone.utc),
        class_name="UNKNOWN",
        confidence=1.0,
        bbox_xyxy=(0.0, 0.0, 1.0, 1.0),
        track_uuid=f"{run_id}:{camera_code}:TRACK_{local_track_id}",
        state="active",
    )
    track = LocalVehicleTrack(
        track_uuid=observation.track_uuid,
        camera_code=camera_code,
        local_track_id=local_track_id,
        class_name="car",
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
        source_path=Path(relative.parts[0]),
    )
    track.evidence_package = TrackEvidencePackage(
        run_id=run_id,
        camera_code=camera_code,
        local_track_id=local_track_id,
        track_uuid=track.track_uuid,
        class_name=track.class_name,
        candidates={
            "best_overall": EvidenceCandidate(
                candidate_type="best_overall",
                frame_number=0,
                video_time_seconds=0.0,
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
        },
        output_directory=str(track_dir),
    )
    return track


def _infer_local_track_id(track_dir_name: str) -> int:
    digits = "".join(char for char in track_dir_name if char.isdigit())
    return int(digits or "0")


if __name__ == "__main__":
    main()
