from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    case_root = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[3]
    for import_root in (case_root, repo_root):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))
    from continuous_mot_hybrid.report_writer import write_html_from_markdown, write_json, write_markdown
else:
    from .report_writer import write_html_from_markdown, write_json, write_markdown


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _label_winner(values: dict[str, float | int | None], *, lower_is_better: bool, labels: dict[str, str]) -> str:
    usable = {key: float(value) for key, value in values.items() if value is not None}
    if len(usable) < 2:
        return "inconclusive"
    best_key = min(usable, key=usable.get) if lower_is_better else max(usable, key=usable.get)
    return labels.get(best_key, "inconclusive")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare td_case2, KCF hybrid, and continuous MOT hybrid.")
    parser.add_argument("--continuous-run-dir", required=True)
    parser.add_argument("--previous-comparison-dir", required=True)
    args = parser.parse_args()

    continuous_dir = Path(args.continuous_run_dir).expanduser().resolve()
    previous_dir = Path(args.previous_comparison_dir).expanduser().resolve()

    td_vs_kcf_report = (previous_dir / "comparison" / "10_final_comparison_report.md").read_text(encoding="utf-8")
    runtime = _read_json(continuous_dir / "09_reports" / "runtime_report.json")
    tracking = _read_json(continuous_dir / "09_reports" / "tracking_report.json")
    reconciliation = _read_json(continuous_dir / "09_reports" / "reconciliation_report.json")
    crop_report = _read_json(continuous_dir / "09_reports" / "crop_report.json")
    detector_report = _read_json(continuous_dir / "09_reports" / "detector_report.json")
    identity_report = _read_json(continuous_dir / "08_identity_packages" / "local_identity_package_report.json")

    comparison = {
        "status": "success",
        "pipelines": {
            "td_case2": {
                "runtime_seconds": 38.819512,
                "processed_frames": 43,
                "yolo_calls": None,
                "raw_tracks": 54,
                "confirmed_tracks": None,
                "reconciled_objects": None,
                "tracks_under_0_5_seconds": 52,
                "tracks_under_1_second": 52,
                "frozen_tracks": 0,
                "boundary_stuck_tracks": 0,
                "accepted_merges": 0,
                "primary_crops": 41,
                "three_crop_sets": 0,
                "plate_candidates": 0,
                "ready_packages": None,
                "manual_review_packages": None,
                "rejected_packages": None,
            },
            "kcf_hybrid": {
                "runtime_seconds": 95.814283,
                "processed_frames": 1587,
                "yolo_calls": 903,
                "raw_tracks": 87,
                "confirmed_tracks": 68,
                "reconciled_objects": 83,
                "tracks_under_0_5_seconds": 41,
                "tracks_under_1_second": 72,
                "frozen_tracks": 2,
                "boundary_stuck_tracks": 2,
                "accepted_merges": 4,
                "primary_crops": 83,
                "three_crop_sets": 66,
                "plate_candidates": 47,
                "ready_packages": 10,
                "manual_review_packages": 73,
                "rejected_packages": 0,
            },
            "continuous_mot": {
                "runtime_seconds": float(runtime["total_runtime_seconds"]),
                "processed_frames": int(_read_json(continuous_dir / "02_frames" / "frame_stream_metrics.json")["processed_frame_count"]),
                "yolo_calls": int(detector_report["total_yolo_calls"]),
                "raw_tracks": int(tracking["raw_track_ids"]),
                "confirmed_tracks": int(tracking["confirmed_tracks"]),
                "reconciled_objects": int(reconciliation["reconciled_objects"]),
                "tracks_under_0_5_seconds": int(tracking["tracks_under_0_5_seconds"]),
                "tracks_under_1_second": int(tracking["tracks_under_1_second"]),
                "frozen_tracks": int(_read_json(continuous_dir / "05_integrity" / "track_integrity_report.json")["frozen_tracks"]),
                "boundary_stuck_tracks": int(_read_json(continuous_dir / "05_integrity" / "track_integrity_report.json")["boundary_stuck_tracks"]),
                "accepted_merges": int(reconciliation["accepted_merges"]),
                "primary_crops": int(crop_report["primary_crops"]),
                "three_crop_sets": int(representative_count := _read_json(continuous_dir / "07_representative_frames" / "representative_frames.json")["status"] == "success" and len([item for item in _read_json(continuous_dir / "07_representative_frames" / "representative_frames.json")["objects"] if len(list(item.get("representative_frames", {}).get("alternatives", []))) >= 2])),
                "plate_candidates": int(crop_report["plate_candidates"]),
                "ready_packages": int(identity_report["ready_packages"]),
                "manual_review_packages": int(identity_report["manual_review_packages"]),
                "rejected_packages": int(identity_report["rejected_packages"]),
            },
        },
    }
    winners = {
        "runtime": _label_winner(
            {key: value["runtime_seconds"] for key, value in comparison["pipelines"].items()},
            lower_is_better=True,
            labels={"td_case2": "td_case2_better", "kcf_hybrid": "kcf_hybrid_better", "continuous_mot": "continuous_mot_better"},
        ),
        "continuity": _label_winner(
            {key: value["tracks_under_1_second"] for key, value in comparison["pipelines"].items()},
            lower_is_better=True,
            labels={"td_case2": "td_case2_better", "kcf_hybrid": "kcf_hybrid_better", "continuous_mot": "continuous_mot_better"},
        ),
        "crop_base": _label_winner(
            {key: value["primary_crops"] for key, value in comparison["pipelines"].items()},
            lower_is_better=False,
            labels={"td_case2": "td_case2_better", "kcf_hybrid": "kcf_hybrid_better", "continuous_mot": "continuous_mot_better"},
        ),
    }
    comparison["winner_summary"] = winners
    comparison["limitations"] = [
        "No manual ground truth was used, so this report does not claim tracking accuracy or object-count accuracy.",
        "The three pipelines use different tracking architectures and different detector schedules.",
        "The td_case2 baseline does not expose detector-call or confirmed-track metrics in the same way as the hybrids.",
    ]
    write_json(continuous_dir / "09_reports" / "three_pipeline_comparison.json", comparison)

    lines = [
        "# Three Pipeline Comparison",
        "",
        f"- Previous comparison directory: {previous_dir}",
        f"- Continuous run directory: {continuous_dir}",
        "",
        "## Winner Summary",
    ]
    for key, value in winners.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Continuous MOT Summary"])
    for key, value in comparison["pipelines"]["continuous_mot"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Previous td_case2 vs KCF Hybrid Report Excerpt", ""])
    lines.extend(td_vs_kcf_report.splitlines()[:30])
    markdown_text = "\n".join(lines) + "\n"
    write_markdown(continuous_dir / "09_reports" / "three_pipeline_comparison.md", lines)
    write_html_from_markdown(continuous_dir / "09_reports" / "three_pipeline_comparison.html", markdown_text)
    print(f"continuous_run_dir={continuous_dir}")


if __name__ == "__main__":
    main()
