from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from ..ingestion.frame_packet import FramePacket
from ..orchestration.multi_camera_orchestrator import MultiCameraOrchestrator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate multi-camera local video input handling.")
    parser.add_argument("--config", required=True, help="Path to cameras.yaml")
    parser.add_argument("--mode", choices=("sequential", "round_robin"), default="round_robin")
    parser.add_argument("--max-frames-per-camera", type=int, default=None)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-scale", type=float, default=1.0)
    parser.add_argument("--output-report", default=None)
    parser.add_argument("--sync-cameras-to-db", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _default_report_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("debug_runs") / "multicamera_vehicle_tracking_pipeline" / f"input_validation_{timestamp}" / "report.json"


def _preview_handler(scale: float):
    import cv2

    def handler(packet: FramePacket) -> bool:
        frame = packet.frame.copy()
        overlay_lines = [
            f"camera={packet.camera_code}",
            f"frame={packet.frame_number}",
            f"time={packet.video_time_seconds:.2f}s",
        ]
        if packet.camera_timestamp is not None:
            overlay_lines.append(packet.camera_timestamp.isoformat())
        y = 24
        for line in overlay_lines:
            cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            y += 24
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        cv2.imshow("multicamera_input_preview", frame)
        key = cv2.waitKey(1) & 0xFF
        return key != ord("q")

    return handler


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="%(levelname)s %(name)s %(message)s")
    orchestrator = MultiCameraOrchestrator(args.config, mode=args.mode, max_frames_per_camera=args.max_frames_per_camera)
    preview_callback = _preview_handler(args.preview_scale) if args.preview else None
    result = orchestrator.run(preview_callback=preview_callback, sync_cameras_to_db=args.sync_cameras_to_db)

    report_path = Path(args.output_report) if args.output_report else _default_report_path()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result.report, indent=2), encoding="utf-8")
    print(json.dumps(result.report, indent=2))

    if args.preview:
        import cv2

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
