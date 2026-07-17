from __future__ import annotations

from pathlib import Path
from typing import Any

from .report_writer import write_html_from_markdown, write_json, write_markdown


def compare_verified_results(
    *,
    output_dir: Path,
    bytetrack_report: dict[str, Any],
    botsort_report: dict[str, Any],
    reid_verification: dict[str, Any],
    visual_manifest: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = {
        "status": "success",
        "bytetrack": {
            "tracker_runtime_seconds": bytetrack_report["tracker_runtime_seconds"],
            "processed_frames": bytetrack_report["processed_frames"],
            "detector_frames": bytetrack_report["detector_frames"],
            "yolo_calls": bytetrack_report["detector_frames"],
            "peak_gpu_memory_mb": bytetrack_report["peak_gpu_memory_mb"],
        },
        "botsort_reid": {
            "tracker_runtime_seconds": botsort_report["tracker_runtime_seconds"],
            "processed_frames": botsort_report["processed_frames"],
            "detector_frames": botsort_report["detector_frames"],
            "yolo_calls": botsort_report["detector_frames"],
            "peak_gpu_memory_mb": botsort_report["peak_gpu_memory_mb"],
            "reid_runtime_seconds": reid_verification.get("total_reid_runtime_seconds", "not_available"),
        },
    }
    track_comparison = {"status": "success", "bytetrack": bytetrack_report, "botsort_reid": botsort_report}
    fragmentation = {
        "status": "success",
        "bytetrack": {key: bytetrack_report[key] for key in ("raw_track_ids", "confirmed_tracks", "tentative_tracks", "tracks_under_0_5_seconds", "tracks_under_1_0_seconds", "interior_new_ids", "boundary_new_ids", "removed_tracks_final")},
        "botsort_reid": {key: botsort_report[key] for key in ("raw_track_ids", "confirmed_tracks", "tentative_tracks", "tracks_under_0_5_seconds", "tracks_under_1_0_seconds", "interior_new_ids", "boundary_new_ids", "removed_tracks_final")},
    }
    reactivation = {
        "status": "success",
        "bytetrack": {key: bytetrack_report[key] for key in ("successful_reactivations", "failed_reactivations", "reactivation_attempts", "tracks_lost_due_to_skipped_detector_frame")},
        "botsort_reid": {key: botsort_report[key] for key in ("successful_reactivations", "failed_reactivations", "reactivation_attempts", "tracks_lost_due_to_skipped_detector_frame")},
    }
    write_json(output_dir / "01_runtime_comparison.json", runtime)
    write_json(output_dir / "02_track_comparison.json", track_comparison)
    write_json(output_dir / "03_fragmentation_comparison.json", fragmentation)
    write_json(output_dir / "04_reactivation_comparison.json", reactivation)
    write_json(output_dir / "05_reid_verification.json", reid_verification)
    write_json(output_dir / "06_visual_review_manifest.json", visual_manifest)
    final = {
        "status": "success",
        "actual_with_reid": bool(reid_verification.get("actual_with_reid")),
        "winner_runtime": "bytetrack" if bytetrack_report["tracker_runtime_seconds"] < botsort_report["tracker_runtime_seconds"] else "botsort_reid",
        "winner_fragmentation": "bytetrack" if bytetrack_report["tracks_under_0_5_seconds"] < botsort_report["tracks_under_0_5_seconds"] else "botsort_reid",
        "winner_confirmed_tracks": "bytetrack" if bytetrack_report["confirmed_tracks"] > botsort_report["confirmed_tracks"] else "botsort_reid",
        "practical_targets_improved": {
            "raw_ids_below_151": botsort_report["raw_track_ids"] < 151,
            "confirmed_tracks_above_51": botsort_report["confirmed_tracks"] > 51,
            "tentative_tracks_below_100": botsort_report["tentative_tracks"] < 100,
            "tracks_under_0_5_seconds_below_98": botsort_report["tracks_under_0_5_seconds"] < 98,
            "successful_reactivations_above_3": botsort_report["successful_reactivations"] > 3,
            "removals_below_98": botsort_report["removed_tracks_final"] < 98,
            "interior_new_ids_below_84": botsort_report["interior_new_ids"] < 84,
        },
    }
    write_json(output_dir / "07_final_verified_reid_comparison.json", final)
    lines = [
        "# Final Verified ReID Comparison",
        "",
        f"- actual_with_reid: {final['actual_with_reid']}",
        f"- ByteTrack runtime: {bytetrack_report['tracker_runtime_seconds']}",
        f"- BoT-SORT ReID runtime: {botsort_report['tracker_runtime_seconds']}",
        f"- ByteTrack raw IDs: {bytetrack_report['raw_track_ids']}",
        f"- BoT-SORT ReID raw IDs: {botsort_report['raw_track_ids']}",
        f"- ByteTrack confirmed: {bytetrack_report['confirmed_tracks']}",
        f"- BoT-SORT ReID confirmed: {botsort_report['confirmed_tracks']}",
        f"- ByteTrack fragments <0.5s: {bytetrack_report['tracks_under_0_5_seconds']}",
        f"- BoT-SORT ReID fragments <0.5s: {botsort_report['tracks_under_0_5_seconds']}",
        f"- Appearance comparison count: {reid_verification.get('appearance_comparison_count', 'not_available')}",
    ]
    markdown_text = "\n".join(lines) + "\n"
    write_markdown(output_dir / "07_final_verified_reid_comparison.md", lines)
    write_html_from_markdown(output_dir / "07_final_verified_reid_comparison.html", markdown_text)
    return final
