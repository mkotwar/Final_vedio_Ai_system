from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare sequential and worker report files.")
    parser.add_argument("--sequential-report", required=True)
    parser.add_argument("--worker-report", required=True)
    return parser.parse_args()


def _load(path: str) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    sequential = _load(args.sequential_report)
    worker = _load(args.worker_report)
    comparison = {
        "frames_processed": {
            "sequential": sequential.get("total_frames_processed"),
            "workers": worker.get("total_frames_read"),
        },
        "detections": {
            "sequential": sequential.get("total_detections", sequential.get("total_vehicle_detections")),
            "workers": worker.get("total_detections"),
        },
        "track_observations": {
            "sequential": sequential.get("total_track_observations"),
            "workers": worker.get("total_track_observations"),
        },
        "completed_tracks": {
            "sequential": sum(1 for item in sequential.get("completed_tracks", []) if item.get("state") == "completed"),
            "workers": worker.get("total_completed_tracks"),
        },
        "discarded_tracks": {
            "sequential": sum(1 for item in sequential.get("completed_tracks", []) if item.get("state") == "discarded"),
            "workers": worker.get("total_discarded_tracks"),
        },
        "wall_clock_runtime_seconds": {
            "sequential": sequential.get("wall_clock_runtime_seconds"),
            "workers": worker.get("wall_clock_runtime_seconds"),
        },
        "processing_fps": {
            "sequential": sequential.get("processing_fps"),
            "workers": worker.get("processing_fps"),
        },
        "errors": {
            "sequential": sequential.get("persistence", {}).get("errors", []) if isinstance(sequential.get("persistence"), dict) else [],
            "workers": worker.get("errors", []),
        },
        "per_camera": {
            "sequential": sequential.get("cameras", {}),
            "workers": worker.get("cameras", {}),
        },
    }
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
