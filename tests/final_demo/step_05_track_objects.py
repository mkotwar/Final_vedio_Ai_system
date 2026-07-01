from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.final_demo.services.chunk_planner import read_json
from tests.final_demo.services.tracking_focus import resolve_tracking_focus
from tests.final_demo.services.tracker_adapter import (
    build_tracking_outputs,
    update_run_manifest_for_tracking,
)
from tests.final_demo.services.video_io import write_json


def run_step_05_track_objects(run_dir: Path) -> dict[str, Any]:
    print("[final-demo] Starting Step 5: object tracking per chunk")

    detections_path = run_dir / "04_yolo_detections.json"
    chunk_manifest_path = run_dir / "02_chunk_manifest.json"
    focus_path = run_dir / "05_tracking_focus.json"

    if not detections_path.exists():
        raise FileNotFoundError(f"Missing Step 4 detections file: {detections_path}")
    if not chunk_manifest_path.exists():
        raise FileNotFoundError(f"Missing Step 2 chunk manifest file: {chunk_manifest_path}")

    detections_payload = read_json(detections_path)
    chunk_manifest = read_json(chunk_manifest_path)
    tracking_focus = resolve_tracking_focus(
        run_dir=run_dir,
        detections_payload=detections_payload,
    )

    tracking_result = build_tracking_outputs(
        run_dir=run_dir,
        detections_payload=detections_payload,
        chunk_manifest=chunk_manifest,
        tracking_focus=tracking_focus,
    )

    tracks_path = run_dir / "05_tracks.json"
    report_path = run_dir / "05_tracking_report.json"
    diagnostics_path = run_dir / "05_tracking_diagnostics.json"

    write_json(chunk_manifest_path, tracking_result["updated_chunk_manifest"])
    write_json(focus_path, tracking_focus)
    write_json(tracks_path, tracking_result["tracks_payload"])
    write_json(report_path, tracking_result["report_payload"])
    write_json(diagnostics_path, tracking_result["diagnostics_payload"])
    update_run_manifest_for_tracking(run_dir / "run_manifest.json")

    tracks_payload = tracking_result["tracks_payload"]
    report_payload = tracking_result["report_payload"]
    print(f"[final-demo] Tracking focus mode: {tracking_focus['focus_mode']}")
    print(f"[final-demo] Selected focus profile: {tracking_focus['selected_focus_profile']}")
    print(
        f"[final-demo] Selected track classes: "
        f"{', '.join(tracking_focus['selected_track_classes'])}"
    )
    print(f"[final-demo] Focus confidence: {tracking_focus['focus_confidence']}")
    for warning in list(tracking_focus.get("warnings") or []):
        print(f"[final-demo] Focus warning: {warning}")
    print(f"[final-demo] Tracker name: {tracks_payload['tracker_name']}")
    print(f"[final-demo] Input detections: {tracks_payload['total_input_detections']}")
    print(f"[final-demo] Tracks created: {tracks_payload['total_tracks_created']}")
    print(f"[final-demo] Tracks kept: {tracks_payload['total_tracks_kept']}")
    print(
        f"[final-demo] Cross-chunk candidates: {report_payload['cross_chunk_candidates_count']}"
    )
    print(f"[final-demo] Output path: {tracks_path}")

    return {
        "run_dir": run_dir,
        "tracks_path": tracks_path,
        "report_path": report_path,
        "diagnostics_path": diagnostics_path,
        "focus_path": focus_path,
        "tracking_result": tracking_result,
        "tracking_focus": tracking_focus,
    }


def main() -> None:
    raise RuntimeError(
        "Run Step 5 from the final demo pipeline or call run_step_05_track_objects(run_dir)."
    )


if __name__ == "__main__":
    main()
