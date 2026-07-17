from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    case_root = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[3]
    for import_root in (case_root, repo_root):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))
    from continuous_mot_hybrid.local_identity_package import build_local_identity_packages
    from continuous_mot_hybrid.metrics import build_crop_report, build_runtime_report
    from continuous_mot_hybrid.report_writer import write_final_report, write_json
    from continuous_mot_hybrid.representative_frame_selector import build_representative_frames
    from continuous_mot_hybrid.track_integrity import sanitize_tracks
    from continuous_mot_hybrid.track_reconciliation import reconcile_track_fragments
else:
    from .local_identity_package import build_local_identity_packages
    from .metrics import build_crop_report, build_runtime_report
    from .report_writer import write_final_report, write_json
    from .representative_frame_selector import build_representative_frames
    from .track_integrity import sanitize_tracks
    from .track_reconciliation import reconcile_track_fragments


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cleanup for the continuous MOT hybrid experiment.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()

    resolved_config = _read_json(run_dir / "01_video" / "resolved_config.json")
    video_info = _read_json(run_dir / "01_video" / "video_info.json")
    raw_tracks_payload = _read_json(run_dir / "04_tracking" / "raw_tracks.json")
    runtime_report = _read_json(run_dir / "09_reports" / "runtime_report.json")
    cleanup_started = time.perf_counter()

    sanitized_tracks, integrity_report, integrity_events = sanitize_tracks(
        list(raw_tracks_payload.get("tracks", [])),
        frame_width=int(video_info["width"]),
        frame_height=int(video_info["height"]),
        maximum_active_detector_gap_seconds=float(resolved_config["detector_max_gap_seconds"]),
        maximum_visual_bridge_seconds=float(resolved_config["visual_bridge_max_seconds"]),
        frozen_window_seconds=0.5,
    )
    write_json(run_dir / "05_integrity" / "track_integrity_report.json", integrity_report)
    write_json(run_dir / "05_integrity" / "track_integrity_events.json", {"status": "success", "events": integrity_events})
    write_json(run_dir / "05_integrity" / "sanitized_tracks.json", {"status": "success", "tracks": sanitized_tracks})

    candidate_payload, accepted_payload, possible_payload, rejected_payload, reconciled_payload = reconcile_track_fragments(
        sanitized_tracks,
        camera_id=str(resolved_config["camera_id"]),
        camera_group=str(resolved_config["camera_group"]),
        camera_timezone=str(resolved_config["camera_timezone"]),
        maximum_gap_seconds=1.5,
        duplicate_overlap_seconds=float(resolved_config["duplicate_overlap_seconds"]),
    )
    write_json(run_dir / "06_reconciliation" / "reconciliation_candidates.json", candidate_payload)
    write_json(run_dir / "06_reconciliation" / "accepted_merges.json", accepted_payload)
    write_json(run_dir / "06_reconciliation" / "possible_merges.json", possible_payload)
    write_json(run_dir / "06_reconciliation" / "rejected_merges.json", rejected_payload)
    write_json(run_dir / "06_reconciliation" / "reconciled_tracks.json", reconciled_payload)
    write_json(run_dir / "06_reconciliation" / "reconciliation_report.json", reconciled_payload["report"])

    crop_started = time.perf_counter()
    representative_frames, representative_report, crop_failures, invalid_crop_candidates = build_representative_frames(
        video_path=Path(str(resolved_config["video_path"])),
        local_objects=list(reconciled_payload["tracks"]),
        frame_width=int(video_info["width"]),
        frame_height=int(video_info["height"]),
        run_dir=run_dir,
    )
    crop_runtime_seconds = time.perf_counter() - crop_started
    write_json(run_dir / "07_representative_frames" / "representative_frames.json", {"status": "success", "objects": representative_frames})
    write_json(run_dir / "07_representative_frames" / "representative_frames_report.json", representative_report)

    packages, flat_rows, package_report = build_local_identity_packages(
        local_objects=list(reconciled_payload["tracks"]),
        representative_frames=representative_frames,
    )
    write_json(run_dir / "08_identity_packages" / "local_identity_packages.json", {"status": "success", "packages": packages})
    write_json(run_dir / "08_identity_packages" / "local_identity_packages_flat.json", {"status": "success", "rows": flat_rows})
    write_json(run_dir / "08_identity_packages" / "local_identity_package_report.json", package_report)

    cleanup_runtime_seconds = time.perf_counter() - cleanup_started
    final_runtime_report = build_runtime_report(
        video_duration_seconds=float(video_info["duration_seconds"]),
        processed_frames=int(_read_json(run_dir / "02_frames" / "frame_stream_metrics.json")["processed_frame_count"]),
        tracking_runtime_seconds=float(runtime_report["tracking_runtime_seconds"]),
        cleanup_runtime_seconds=cleanup_runtime_seconds,
        crop_runtime_seconds=crop_runtime_seconds,
    )
    crop_report = build_crop_report(representative_report=representative_report, identity_report=package_report)
    write_json(run_dir / "09_reports" / "runtime_report.json", final_runtime_report)
    write_json(run_dir / "09_reports" / "reconciliation_report.json", reconciled_payload["report"])
    write_json(run_dir / "09_reports" / "crop_report.json", crop_report)
    markdown_text = write_final_report(
        report_path=run_dir / "09_reports" / "final_report",
        summary={
            "run_dir": str(run_dir),
            "mot_backend": resolved_config["mot_backend"],
            "processed_frames": _read_json(run_dir / "02_frames" / "frame_stream_metrics.json")["processed_frame_count"],
            "detector_calls": len(_read_json(run_dir / "03_detections" / "detector_schedule.json")["calls"]),
            "reconciled_objects": reconciled_payload["report"]["reconciled_objects"],
            "ready_packages": package_report["ready_packages"],
        },
        sections={
            "runtime": final_runtime_report,
            "integrity": integrity_report,
            "reconciliation": reconciled_payload["report"],
            "crops": crop_report,
            "identity_packages": package_report,
        },
    )
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
