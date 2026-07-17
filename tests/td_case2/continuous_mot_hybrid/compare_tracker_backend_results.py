from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .report_writer import write_html_from_markdown, write_json, write_markdown
from .tracker_backend_metrics import build_reid_metric_block, winner_for_metric


def build_identity_switch_candidates(*, backend_tracks: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    bytetrack_tracks = backend_tracks.get("bytetrack", [])
    for left in bytetrack_tracks:
        for right in bytetrack_tracks:
            if left["tracker_id"] == right["tracker_id"]:
                continue
            gap = float(right["start_timestamp"]) - float(left["end_timestamp"])
            if gap < 0 or gap > 1.0:
                continue
            if left["object_family"] != right["object_family"]:
                continue
            candidates.append(
                {
                    "label": "review_required",
                    "backend_anchor": "bytetrack",
                    "primary_track_id": left["tracker_id"],
                    "candidate_successor_track_id": right["tracker_id"],
                    "gap_seconds": round(gap, 6),
                    "all_backend_matches": {
                        backend_name: [
                            track["tracker_id"]
                            for track in tracks
                            if track["object_family"] == left["object_family"]
                            and abs(float(track["start_timestamp"]) - float(right["start_timestamp"])) <= 1.0
                        ][:3]
                        for backend_name, tracks in backend_tracks.items()
                    },
                }
            )
            if len(candidates) >= 50:
                return candidates
    return candidates


def compare_results(*, run_dir: Path, shared_checksum: str, backend_payloads: dict[str, dict[str, Any]], reid_verification: dict[str, Any] | None) -> dict[str, Any]:
    comparison_dir = run_dir / "04_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    reports = {name: payload["report"] for name, payload in backend_payloads.items()}
    backend_tracks = {name: payload["raw_tracks"] for name, payload in backend_payloads.items()}
    runtime = {
        name: {
            "tracker_runtime_seconds": report["tracker_runtime_seconds"],
            "end_to_end_replay_runtime_seconds": report["end_to_end_replay_runtime_seconds"],
            "average_ms_per_processed_frame": report["average_ms_per_processed_frame"],
            "realtime_factor": report["realtime_factor"],
            "peak_gpu_memory_mb": report["peak_gpu_memory_mb"],
            "peak_system_memory_mb": report["peak_system_memory_mb"],
        }
        for name, report in reports.items()
    }
    track_counts = {
        name: {
            "raw_tracker_ids": report["raw_track_ids"],
            "confirmed_tracks": report["confirmed_tracks"],
            "tentative_tracks": report["tentative_tracks"],
            "active_tracks": report["active_tracks_final"],
            "lost_tracks": report["lost_tracks_final"],
            "removed_tracks": report["removed_tracks_final"],
            "reactivated_tracks": report["reactivated_tracks"],
        }
        for name, report in reports.items()
    }
    fragmentation = {
        name: {
            "tracks_under_0_5_seconds": report["tracks_under_0_5_seconds"],
            "tracks_under_1_0_seconds": report["tracks_under_1_0_seconds"],
            "average_duration_seconds": report["average_duration_seconds"],
            "median_duration_seconds": report["median_duration_seconds"],
            "maximum_duration_seconds": report["maximum_duration_seconds"],
            "ids_created_per_detector_frame": report["ids_created_per_detector_frame"],
            "interior_new_ids": report["interior_new_ids"],
            "boundary_new_ids": report["boundary_new_ids"],
        }
        for name, report in reports.items()
    }
    reactivation = {
        name: {
            "reactivation_attempts": report["reactivation_attempts"],
            "successful_reactivations": report["successful_reactivations"],
            "failed_reactivations": report["failed_reactivations"],
            "reactivation_success_rate": report["reactivation_success_rate"],
            "tracker_removals": report["removed_tracks_final"],
            "tracks_lost_due_to_skipped_frames": report["tracks_lost_due_to_skipped_detector_frame"],
        }
        for name, report in reports.items()
    }
    reid_runtime = build_reid_metric_block(reid_verification or {})
    config_diff = {
        "status": "success",
        "shared_detection_cache_checksum": shared_checksum,
        "differences": [
            "ByteTrack uses tracker_type=bytetrack while the BoT-SORT runs use tracker_type=botsort.",
            "BoT-SORT default GMC method remains enabled for the first official-default run.",
            "BoT-SORT + ReID requested with_reid=true, but actual runtime activation depends on encoder initialization and observed feature vectors.",
        ],
    }
    identity_switch_candidates = build_identity_switch_candidates(backend_tracks=backend_tracks)
    final = {
        "status": "success",
        "decision_categories": {
            "lowest_runtime": winner_for_metric({name: report["tracker_runtime_seconds"] for name, report in reports.items()}, lower_is_better=True),
            "lowest_short_track_fragmentation_proxy": winner_for_metric({name: report["tracks_under_0_5_seconds"] for name, report in reports.items()}, lower_is_better=True),
            "highest_confirmed_track_ratio": winner_for_metric({name: report["confirmed_track_ratio"] for name, report in reports.items()}, lower_is_better=False),
            "best_short_occlusion_recovery": winner_for_metric({name: report["successful_reactivations"] for name, report in reports.items()}, lower_is_better=False),
            "fewest_interior_new_ids": winner_for_metric({name: report["interior_new_ids"] for name, report in reports.items()}, lower_is_better=True),
            "lowest_removal_count": winner_for_metric({name: report["removed_tracks_final"] for name, report in reports.items()}, lower_is_better=True),
            "best_visual_identity_continuity": "review_required",
            "best_traffic_tracking_backend": winner_for_metric({name: report["confirmed_tracks"] for name, report in reports.items()}, lower_is_better=False),
            "best_speed_quality_balance": "inconclusive",
        },
        "reid_status": (
            "reid_not_available"
            if reid_verification and reid_verification.get("requested_with_reid") and not reid_verification.get("actual_with_reid")
            else "available_or_not_requested"
        ),
    }
    write_json(comparison_dir / "01_runtime_comparison.json", {"status": "success", "backends": runtime})
    write_json(comparison_dir / "02_track_count_comparison.json", {"status": "success", "backends": track_counts})
    write_json(comparison_dir / "03_fragmentation_proxy_comparison.json", {"status": "success", "backends": fragmentation})
    write_json(comparison_dir / "04_reactivation_comparison.json", {"status": "success", "backends": reactivation})
    write_json(comparison_dir / "05_reid_runtime_comparison.json", {"status": "success", "reid": reid_runtime})
    write_json(comparison_dir / "06_config_comparison.json", config_diff)
    write_json(comparison_dir / "08_identity_switch_candidates.json", {"status": "success", "candidates": identity_switch_candidates})
    write_json(comparison_dir / "09_final_tracker_comparison.json", {"status": "success", "final": final, "reports": reports})
    lines = [
        "# Final Tracker Comparison",
        "",
        "| Metric | ByteTrack | BoT-SORT no ReID | BoT-SORT + ReID |",
        "|---|---:|---:|---:|",
        f"| Tracker runtime | {reports.get('bytetrack', {}).get('tracker_runtime_seconds', 'N/A')} | {reports.get('botsort_no_reid', {}).get('tracker_runtime_seconds', 'N/A')} | {reports.get('botsort_reid', {}).get('tracker_runtime_seconds', 'N/A')} |",
        f"| Raw tracker IDs | {reports.get('bytetrack', {}).get('raw_track_ids', 'N/A')} | {reports.get('botsort_no_reid', {}).get('raw_track_ids', 'N/A')} | {reports.get('botsort_reid', {}).get('raw_track_ids', 'N/A')} |",
        f"| Confirmed tracks | {reports.get('bytetrack', {}).get('confirmed_tracks', 'N/A')} | {reports.get('botsort_no_reid', {}).get('confirmed_tracks', 'N/A')} | {reports.get('botsort_reid', {}).get('confirmed_tracks', 'N/A')} |",
        f"| Tentative tracks | {reports.get('bytetrack', {}).get('tentative_tracks', 'N/A')} | {reports.get('botsort_no_reid', {}).get('tentative_tracks', 'N/A')} | {reports.get('botsort_reid', {}).get('tentative_tracks', 'N/A')} |",
        f"| Tracks under 0.5s | {reports.get('bytetrack', {}).get('tracks_under_0_5_seconds', 'N/A')} | {reports.get('botsort_no_reid', {}).get('tracks_under_0_5_seconds', 'N/A')} | {reports.get('botsort_reid', {}).get('tracks_under_0_5_seconds', 'N/A')} |",
        f"| Tracks under 1.0s | {reports.get('bytetrack', {}).get('tracks_under_1_0_seconds', 'N/A')} | {reports.get('botsort_no_reid', {}).get('tracks_under_1_0_seconds', 'N/A')} | {reports.get('botsort_reid', {}).get('tracks_under_1_0_seconds', 'N/A')} |",
        f"| Reactivations | {reports.get('bytetrack', {}).get('successful_reactivations', 'N/A')} | {reports.get('botsort_no_reid', {}).get('successful_reactivations', 'N/A')} | {reports.get('botsort_reid', {}).get('successful_reactivations', 'N/A')} |",
        f"| Tracker removals | {reports.get('bytetrack', {}).get('removed_tracks_final', 'N/A')} | {reports.get('botsort_no_reid', {}).get('removed_tracks_final', 'N/A')} | {reports.get('botsort_reid', {}).get('removed_tracks_final', 'N/A')} |",
        f"| Interior new IDs | {reports.get('bytetrack', {}).get('interior_new_ids', 'N/A')} | {reports.get('botsort_no_reid', {}).get('interior_new_ids', 'N/A')} | {reports.get('botsort_reid', {}).get('interior_new_ids', 'N/A')} |",
        f"| Boundary new IDs | {reports.get('bytetrack', {}).get('boundary_new_ids', 'N/A')} | {reports.get('botsort_no_reid', {}).get('boundary_new_ids', 'N/A')} | {reports.get('botsort_reid', {}).get('boundary_new_ids', 'N/A')} |",
        f"| Tracks lost due to skipped frames | {reports.get('bytetrack', {}).get('tracks_lost_due_to_skipped_detector_frame', 'N/A')} | {reports.get('botsort_no_reid', {}).get('tracks_lost_due_to_skipped_detector_frame', 'N/A')} | {reports.get('botsort_reid', {}).get('tracks_lost_due_to_skipped_detector_frame', 'N/A')} |",
        f"| Appearance-assisted matches | N/A | N/A | {(reid_verification or {}).get('accepted_appearance_matches', 'N/A')} |",
        f"| ReID runtime overhead | N/A | N/A | {(reid_verification or {}).get('reid_runtime_overhead_seconds', 'N/A')} |",
        f"| Peak GPU memory | {reports.get('bytetrack', {}).get('peak_gpu_memory_mb', 'N/A')} | {reports.get('botsort_no_reid', {}).get('peak_gpu_memory_mb', 'N/A')} | {reports.get('botsort_reid', {}).get('peak_gpu_memory_mb', 'N/A')} |",
        "",
        "## Decisions",
    ]
    for key, value in final["decision_categories"].items():
        lines.append(f"- {key.replace('_', ' ')}: {value}")
    markdown_text = "\n".join(lines) + "\n"
    write_markdown(comparison_dir / "09_final_tracker_comparison.md", lines)
    write_html_from_markdown(comparison_dir / "09_final_tracker_comparison.html", markdown_text)
    return {
        "runtime": runtime,
        "track_counts": track_counts,
        "fragmentation": fragmentation,
        "reactivation": reactivation,
        "reid_runtime": reid_runtime,
        "config_diff": config_diff,
        "identity_switch_candidates": identity_switch_candidates,
        "final": final,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare already-generated tracker backend results.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    shared_checksum = json.loads((run_dir / "00_shared" / "detection_cache_checksum.json").read_text(encoding="utf-8"))["sha256"]
    backend_payloads: dict[str, dict[str, Any]] = {}
    for name, folder in (("bytetrack", "01_bytetrack"), ("botsort_no_reid", "02_botsort_no_reid"), ("botsort_reid", "03_botsort_reid")):
        track_path = run_dir / folder / "raw_tracks.json"
        report_path = run_dir / folder / "tracking_metrics.json"
        if track_path.exists() and report_path.exists():
            backend_payloads[name] = {
                "raw_tracks": json.loads(track_path.read_text(encoding="utf-8")).get("tracks", []),
                "report": json.loads(report_path.read_text(encoding="utf-8")),
            }
    reid_verification = None
    reid_path = run_dir / "03_botsort_reid" / "reid_runtime_verification.json"
    if reid_path.exists():
        reid_verification = json.loads(reid_path.read_text(encoding="utf-8"))
    compare_results(run_dir=run_dir, shared_checksum=shared_checksum, backend_payloads=backend_payloads, reid_verification=reid_verification)


if __name__ == "__main__":
    main()
