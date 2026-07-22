from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ..orchestration.multicamera_detection_orchestrator import MultiCameraDetectionOrchestrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate shared vehicle detection on multi-camera local video inputs.")
    parser.add_argument("--camera-config", required=True)
    parser.add_argument("--detection-config", required=True)
    parser.add_argument("--mode", choices=("sequential", "round_robin"), default="round_robin")
    parser.add_argument("--max-frames-per-camera", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--fallback-model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=None)
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
    overrides = {
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
    orchestrator = MultiCameraDetectionOrchestrator(
        args.camera_config,
        args.detection_config,
        mode=args.mode,
        max_frames_per_camera=args.max_frames_per_camera,
        detection_overrides=overrides,
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
