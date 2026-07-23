from __future__ import annotations

import argparse
import json
import logging

from ..orchestration.worker_multicamera_tracking_orchestrator import WorkerMultiCameraTrackingOrchestrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the worker-based multi-camera tracking pipeline.")
    parser.add_argument("--camera-config", required=True)
    parser.add_argument("--detection-config", required=True)
    parser.add_argument("--tracking-config", required=True)
    parser.add_argument("--worker-config", required=True)
    parser.add_argument("--persistence-config", default=None)
    parser.add_argument("--evidence-config", default=None)
    parser.add_argument("--florence-config", default=None)
    parser.add_argument("--vehicle-colour-config", default=None)
    parser.add_argument("--anpr-config", default=None)
    parser.add_argument("--florence-model-path", default=None)
    parser.add_argument("--florence-adapter-path", default=None)
    parser.add_argument("--florence-processor-path", default=None)
    parser.add_argument("--florence-device", default=None)
    parser.add_argument("--plate-detector-model-path", default=None)
    parser.add_argument("--plate-detector-device", default=None)
    parser.add_argument("--camera-code", default=None)
    parser.add_argument("--camera-codes", nargs="+", default=None)
    parser.add_argument("--camera-limit", type=int, default=None)
    parser.add_argument("--max-frames-per-camera", type=int, default=None)
    parser.add_argument("--persist-to-supabase", action="store_true")
    parser.add_argument("--dry-run-persistence", action="store_true")
    parser.add_argument("--save-sample-frames", action="store_true")
    parser.add_argument("--sample-frame-limit-per-camera", type=int, default=1)
    parser.add_argument("--output-report", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="%(levelname)s %(name)s %(message)s")
    persistence_backend = "disabled"
    if args.dry_run_persistence:
        persistence_backend = "dry_run"
    elif args.persist_to_supabase:
        persistence_backend = "analytics_supabase"
    worker_overrides = {
        "enabled": True,
        "enable_persistence_worker": args.persist_to_supabase or args.dry_run_persistence,
        "enable_anpr_worker": bool(args.anpr_config),
    }
    persistence_overrides = {
        "backend": persistence_backend,
        "enabled": args.persist_to_supabase or args.dry_run_persistence,
        "dry_run": args.dry_run_persistence,
    }
    orchestrator = WorkerMultiCameraTrackingOrchestrator(
        args.camera_config,
        args.detection_config,
        args.tracking_config,
        args.worker_config,
        args.persistence_config,
        args.evidence_config,
        args.florence_config,
        args.vehicle_colour_config,
        args.anpr_config,
        max_frames_per_camera=args.max_frames_per_camera,
        worker_overrides=worker_overrides,
        persistence_overrides=persistence_overrides,
        florence_model_path=args.florence_model_path,
        florence_adapter_path=args.florence_adapter_path,
        florence_processor_path=args.florence_processor_path,
        florence_device=args.florence_device,
        plate_detector_model_path=args.plate_detector_model_path,
        plate_detector_device=args.plate_detector_device,
        camera_code=args.camera_code,
        camera_codes=args.camera_codes,
        camera_limit=args.camera_limit,
    )
    result = orchestrator.run(
        save_sample_frames=args.save_sample_frames,
        sample_frame_limit_per_camera=args.sample_frame_limit_per_camera,
        output_report=args.output_report,
    )
    print(json.dumps(result.report, indent=2))


if __name__ == "__main__":
    main()
