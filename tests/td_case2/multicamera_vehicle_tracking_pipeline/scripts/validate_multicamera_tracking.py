from __future__ import annotations

import argparse
import json
import logging

from ..orchestration.multicamera_tracking_orchestrator import MultiCameraTrackingOrchestrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate independent ByteTrack instances for multi-camera local video inputs.")
    parser.add_argument("--camera-config", required=True)
    parser.add_argument("--detection-config", required=True)
    parser.add_argument("--tracking-config", required=True)
    parser.add_argument("--persistence-config", default=None)
    parser.add_argument("--mode", choices=("sequential", "round_robin"), default="round_robin")
    parser.add_argument("--max-frames-per-camera", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--fallback-model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--track-high-thresh", type=float, default=None)
    parser.add_argument("--track-low-thresh", type=float, default=None)
    parser.add_argument("--new-track-thresh", type=float, default=None)
    parser.add_argument("--match-thresh", type=float, default=None)
    parser.add_argument("--track-buffer", type=int, default=None)
    parser.add_argument("--min-confirmed-observations", type=int, default=None)
    parser.add_argument("--max-lost-frames", type=int, default=None)
    parser.add_argument("--persist-to-supabase", action="store_true")
    parser.add_argument("--dry-run-persistence", action="store_true")
    parser.add_argument("--include-discarded-tracks", action="store_true")
    parser.add_argument("--observation-mode", choices=("all", "sampled", "none"), default=None)
    parser.add_argument("--observation-sample-every-n", type=int, default=None)
    parser.add_argument("--fail-on-database-error", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-scale", type=float, default=1.0)
    parser.add_argument("--save-sample-frames", action="store_true")
    parser.add_argument("--sample-frame-limit-per-camera", type=int, default=1)
    parser.add_argument("--output-report", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="%(levelname)s %(name)s %(message)s")
    detection_overrides = {
        key: value
        for key, value in {
            "model_path": args.model,
            "fallback_model_path": args.fallback_model,
            "device": args.device,
            "confidence_threshold": args.confidence,
            "iou_threshold": args.iou,
            "image_size": args.image_size,
        }.items()
        if value is not None
    }
    tracking_overrides = {
        key: value
        for key, value in {
            "track_high_thresh": args.track_high_thresh,
            "track_low_thresh": args.track_low_thresh,
            "new_track_thresh": args.new_track_thresh,
            "match_thresh": args.match_thresh,
            "track_buffer": args.track_buffer,
            "min_confirmed_observations": args.min_confirmed_observations,
            "max_lost_frames": args.max_lost_frames,
        }.items()
        if value is not None
    }
    persistence_overrides = {
        key: value
        for key, value in {
            "enabled": args.persist_to_supabase or args.dry_run_persistence,
            "dry_run": args.dry_run_persistence,
            "include_discarded_tracks": args.include_discarded_tracks,
            "observation_mode": args.observation_mode,
            "observation_sample_every_n": args.observation_sample_every_n,
            "fail_on_database_error": True if args.fail_on_database_error else None,
        }.items()
        if value is not None
    }
    orchestrator = MultiCameraTrackingOrchestrator(
        args.camera_config,
        args.detection_config,
        args.tracking_config,
        args.persistence_config,
        mode=args.mode,
        max_frames_per_camera=args.max_frames_per_camera,
        detection_overrides=detection_overrides,
        tracking_overrides=tracking_overrides,
        persistence_overrides=persistence_overrides,
    )
    result = orchestrator.run(
        preview=args.preview,
        preview_scale=args.preview_scale,
        save_sample_frames=args.save_sample_frames,
        sample_frame_limit_per_camera=args.sample_frame_limit_per_camera,
        output_report=args.output_report,
    )
    print(json.dumps(result.report, indent=2))


if __name__ == "__main__":
    main()
