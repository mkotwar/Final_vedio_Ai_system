from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_get(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def _report_summary(report: dict[str, Any], reconciliation: dict[str, Any] | None) -> dict[str, Any]:
    duplication = dict(report.get("duplication_diagnostics", {}))
    cars = dict(duplication.get("cars", {}))
    return {
        "raw_track_ids": int(report.get("raw_track_id_count", report.get("tracks_created", 0)) or 0),
        "confirmed_raw_tracks": int(report.get("confirmed_raw_track_count", report.get("tracks_confirmed", 0)) or 0),
        "raw_car_track_ids": int(cars.get("raw_car_track_ids", 0) or 0),
        "reconciled_estimated_cars": int(_safe_get(reconciliation or {}, "reconciled_objects_by_class", "car", default=0) or 0),
        "average_track_duration_seconds": float(_safe_get(report, "track_durations", "mean", default=0.0) or 0.0),
        "median_track_duration_seconds": float(_safe_get(report, "track_durations", "median", default=0.0) or 0.0),
        "tracks_shorter_than_1_second": int(duplication.get("tracks_shorter_than_1_0_second", 0) or 0),
        "missed_refresh_removals": int(duplication.get("tracks_ending_because_of_missed_refresh_limit", 0) or 0),
        "track_reactivations": int(report.get("reactivated_track_count", 0) or 0),
        "yolo_calls": int(report.get("yolo_call_count", 0) or 0),
        "yolo_execution_ratio": float(report.get("yolo_execution_ratio", 0.0) or 0.0),
        "kcf_failures": int(report.get("kcf_failure_count", 0) or 0),
        "maximum_simultaneous_tracks": int(report.get("maximum_simultaneous_tracks", 0) or 0),
        "runtime_seconds": float(_safe_get(report, "processing_speed", "total_runtime_seconds", default=0.0) or 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare duplication diagnostics between two hybrid td_case2 runs.")
    parser.add_argument("--before-run-dir", required=True)
    parser.add_argument("--after-run-dir", required=True)
    args = parser.parse_args()

    after_dir = Path(args.after_run_dir).expanduser().resolve() / "hybrid_tracking_test"
    before_dir = Path(args.before_run_dir).expanduser().resolve() / "hybrid_tracking_test"
    before_report = _load_json(before_dir / "04c_hybrid_tracking_report.json")
    after_report = _load_json(after_dir / "04c_hybrid_tracking_report.json")
    before_reconciliation = _load_json(before_dir / "04d_track_reconciliation_report.json") if (before_dir / "04d_track_reconciliation_report.json").exists() else None
    after_reconciliation = _load_json(after_dir / "04d_track_reconciliation_report.json") if (after_dir / "04d_track_reconciliation_report.json").exists() else None

    before_summary = _report_summary(before_report, before_reconciliation)
    after_summary = _report_summary(after_report, after_reconciliation)
    comparison = {
        "status": "success",
        "before_run_dir": str(Path(args.before_run_dir).expanduser().resolve()),
        "after_run_dir": str(Path(args.after_run_dir).expanduser().resolve()),
        "before": before_summary,
        "after": after_summary,
    }
    comparison["delta"] = {
        key: round(float(after_summary[key]) - float(before_summary[key]), 6)
        for key in before_summary
        if isinstance(before_summary[key], (int, float)) and isinstance(after_summary[key], (int, float))
    }
    markdown = "\n".join(
        [
            "# Before/After Duplication Comparison",
            "",
            f"- Before raw track IDs: {before_summary['raw_track_ids']}",
            f"- After raw track IDs: {after_summary['raw_track_ids']}",
            f"- Before raw car track IDs: {before_summary['raw_car_track_ids']}",
            f"- After raw car track IDs: {after_summary['raw_car_track_ids']}",
            f"- Before track reactivations: {before_summary['track_reactivations']}",
            f"- After track reactivations: {after_summary['track_reactivations']}",
            f"- Before YOLO calls: {before_summary['yolo_calls']}",
            f"- After YOLO calls: {after_summary['yolo_calls']}",
        ]
    )
    (after_dir / "04e_before_after_duplication_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    (after_dir / "04e_before_after_duplication_comparison.md").write_text(markdown, encoding="utf-8")
    print(str(after_dir / "04e_before_after_duplication_comparison.json"))


if __name__ == "__main__":
    main()
