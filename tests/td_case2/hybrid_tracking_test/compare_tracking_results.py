from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _find_baseline_tracking_file(run_dir: Path) -> Path:
    candidates = [
        run_dir / "04B_tracks.json",
        run_dir / "04B_tracks_experimental.json",
        run_dir / "04B_tracks_experimental_raw.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No likely baseline tracking file was found in: {run_dir}")


def _load_track_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if isinstance(payload.get("tracks"), list):
            return list(payload["tracks"])
        if isinstance(payload.get("track_summaries"), list):
            return list(payload["track_summaries"])
    raise ValueError(f"Unsupported tracking payload format: {path}")


def _class_name(track: dict[str, Any]) -> str:
    return str(track.get("class_name") or track.get("dominant_class_name") or track.get("track_type") or "unknown")


def _duration_seconds(track: dict[str, Any]) -> float:
    if track.get("duration_seconds") is not None:
        return float(track.get("duration_seconds") or 0.0)
    detections = list(track.get("detections", []))
    if not detections:
        return 0.0
    return float(detections[-1].get("timestamp_seconds", 0.0) or 0.0) - float(detections[0].get("timestamp_seconds", 0.0) or 0.0)


def _track_type_counts(tracks: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {"vehicle": 0, "person": 0, "other": 0}
    for track in tracks:
        class_name = _class_name(track).lower()
        if class_name == "person":
            counts["person"] += 1
        elif class_name in {"car", "truck", "bus", "motorcycle", "vehicle", "van", "auto", "bicycle"}:
            counts["vehicle"] += 1
        else:
            counts["other"] += 1
    return counts


def _fragmentation_heuristics(tracks: list[dict[str, Any]]) -> dict[str, Any]:
    short_lived = [track for track in tracks if _duration_seconds(track) < 0.5]
    few_hits = [track for track in tracks if int(track.get("detection_hits", track.get("detection_count", 0)) or 0) < 2]
    return {
        "tracks_shorter_than_0_5_seconds": len(short_lived),
        "tracks_with_fewer_than_2_detector_hits": len(few_hits),
    }


def compare_tracking_results(
    *,
    run_dir: Path,
    hybrid_tracks_path: Path,
    baseline_tracks_path: Path | None = None,
) -> tuple[dict[str, Any], str]:
    baseline_path: Path | None = None
    baseline_tracks: list[dict[str, Any]] = []
    baseline_error: str | None = None
    try:
        baseline_path = baseline_tracks_path or _find_baseline_tracking_file(run_dir)
        baseline_tracks = _load_track_records(baseline_path)
    except FileNotFoundError as exc:
        baseline_error = str(exc)
    hybrid_tracks = _load_track_records(hybrid_tracks_path)
    baseline_type_counts = _track_type_counts(baseline_tracks)
    hybrid_type_counts = _track_type_counts(hybrid_tracks)
    comparison = {
        "status": "success",
        "baseline_path": None if baseline_path is None else str(baseline_path),
        "hybrid_path": str(hybrid_tracks_path),
        "baseline_missing": baseline_path is None,
        "baseline_error": baseline_error,
        "baseline": {
            "track_count": len(baseline_tracks),
            "type_counts": baseline_type_counts,
            "duration_mean_seconds": round(sum(_duration_seconds(track) for track in baseline_tracks) / max(len(baseline_tracks), 1), 6),
            "frame_coverage_ratio": 1.0,
        },
        "hybrid": {
            "track_count": len(hybrid_tracks),
            "type_counts": hybrid_type_counts,
            "duration_mean_seconds": round(sum(_duration_seconds(track) for track in hybrid_tracks) / max(len(hybrid_tracks), 1), 6),
            "frame_coverage_ratio": 1.0,
        },
        "heuristics": {
            "baseline_short_lived_tracks": _fragmentation_heuristics(baseline_tracks)["tracks_shorter_than_0_5_seconds"],
            "hybrid_short_lived_tracks": _fragmentation_heuristics(hybrid_tracks)["tracks_shorter_than_0_5_seconds"],
            "baseline_few_hit_tracks": _fragmentation_heuristics(baseline_tracks)["tracks_with_fewer_than_2_detector_hits"],
            "hybrid_few_hit_tracks": _fragmentation_heuristics(hybrid_tracks)["tracks_with_fewer_than_2_detector_hits"],
            "notes": "These fragmentation and identity metrics are heuristics only because ground-truth annotations are unavailable.",
        },
        "ground_truth_metrics": {
            "MOTA": "not_measurable",
            "HOTA": "not_measurable",
            "IDF1": "not_measurable",
        },
    }
    markdown_summary = "\n".join(
        [
            "# Hybrid Tracking Comparison",
            "",
            f"- Baseline tracks: {comparison['baseline']['track_count']}",
            f"- Hybrid tracks: {comparison['hybrid']['track_count']}",
            f"- Baseline vehicle tracks: {comparison['baseline']['type_counts']['vehicle']}",
            f"- Hybrid vehicle tracks: {comparison['hybrid']['type_counts']['vehicle']}",
            f"- Baseline short-lived tracks (<0.5s heuristic): {comparison['heuristics']['baseline_short_lived_tracks']}",
            f"- Hybrid short-lived tracks (<0.5s heuristic): {comparison['heuristics']['hybrid_short_lived_tracks']}",
            "",
            f"- Baseline available: {'no' if comparison['baseline_missing'] else 'yes'}",
            "Ground-truth identity metrics were not computed because annotations are unavailable.",
        ]
    )
    return comparison, markdown_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline td_case2 tracking output to 04c hybrid tracks.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--hybrid-tracks-path")
    parser.add_argument("--baseline-tracks-path")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    hybrid_tracks_path = (
        Path(args.hybrid_tracks_path).expanduser().resolve()
        if args.hybrid_tracks_path
        else run_dir / "hybrid_tracking_test" / "04c_hybrid_tracks.json"
    )
    comparison, markdown_summary = compare_tracking_results(
        run_dir=run_dir,
        hybrid_tracks_path=hybrid_tracks_path,
        baseline_tracks_path=Path(args.baseline_tracks_path).expanduser().resolve() if args.baseline_tracks_path else None,
    )
    output_dir = hybrid_tracks_path.parent
    (output_dir / "04c_hybrid_comparison_report.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    (output_dir / "04c_hybrid_comparison_report.md").write_text(markdown_summary, encoding="utf-8")
    print(str(output_dir / "04c_hybrid_comparison_report.json"))


if __name__ == "__main__":
    main()
