from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def build_final_report_markdown(summary: dict[str, Any]) -> str:
    table_rows = summary["final_table_rows"]
    lines = [
        "# td_case2 vs Hybrid Comparison Report",
        "",
        f"- Comparison run directory: `{summary['comparison_run_directory']}`",
        f"- Video: `{summary['video_path']}`",
        "",
        "## Decision Summary",
    ]
    for key, value in summary["decisions"].items():
        lines.append(f"- {key}: {value['decision']}")
    lines.extend(
        [
            "",
            "## Final Table",
            "",
            "| Metric | Existing td_case2 | Hybrid | Better | Notes |",
            "|---|---:|---:|---|---|",
        ]
    )
    for row in table_rows:
        lines.append(
            f"| {row['metric']} | {row['td_case2']} | {row['hybrid']} | {row['better']} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## Fairness Warnings",
        ]
    )
    for warning in summary["config_differences"].get("fairness_warnings", []):
        lines.append(f"- {warning}")
    lines.extend(
        [
            "",
            "## Manual Review",
            f"- Status: {summary['manual_review'].get('status')}",
        ]
    )
    if summary["manual_review"].get("status") == "found":
        lines.append(f"- Reviewed object count: {summary['manual_review'].get('reviewed_object_count')}")
        lines.append(f"- Unreviewed object count: {summary['manual_review'].get('unreviewed_object_count')}")
    return "\n".join(lines) + "\n"


def build_final_report_html(markdown: str) -> str:
    escaped = (
        markdown.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return f"<html><body><pre>{escaped}</pre></body></html>\n"


def build_final_table_rows(runtime: dict[str, Any], detector: dict[str, Any], tracking: dict[str, Any], crop: dict[str, Any], decisions: dict[str, Any]) -> list[dict[str, str]]:
    return [
        _row("Total runtime", runtime["td_case2"]["end_to_end_tracking_plus_crop_runtime_seconds"], runtime["hybrid"]["end_to_end_tracking_plus_crop_runtime_seconds"], decisions["speed_winner"]["decision"], "End-to-end tracking plus crop stages."),
        _row("Realtime factor", runtime["td_case2"]["realtime_factor"], runtime["hybrid"]["realtime_factor"], decisions["speed_winner"]["decision"], "Lower is faster than source-video time."),
        _row("Processed frames", detector["td_case2"]["processed_frame_count"], detector["hybrid"]["processed_frame_count"], "approximately_equal", "Different frame sampling may apply."),
        _row("YOLO calls", detector["td_case2"]["yolo_calls"], detector["hybrid"]["yolo_calls"], decisions["detector_efficiency_winner"]["decision"], "td_case2 may expose only derived detector-call counts."),
        _row("Raw track IDs", tracking["td_case2"]["raw_track_ids"], tracking["hybrid"]["raw_track_ids"], decisions["fragmentation_risk_winner"]["decision"], "Lower is not ground truth, only fragmentation risk proxy."),
        _row("Confirmed tracks", tracking["td_case2"]["confirmed_raw_tracks"], tracking["hybrid"]["confirmed_raw_tracks"], "approximately_equal", "Pipeline-specific tracker outputs."),
        _row("Reconciled objects", tracking["td_case2"]["reconciled_local_objects"], tracking["hybrid"]["reconciled_local_objects"], "inconclusive", "td_case2 baseline does not expose reconciled local objects in the same way."),
        _row("Tracks under 0.5 s", tracking["td_case2"]["tracks_shorter_than_0_5_seconds"], tracking["hybrid"]["tracks_shorter_than_0_5_seconds"], decisions["track_continuity_winner"]["decision"], "Lower suggests fewer tiny fragments."),
        _row("Tracks under 1 s", tracking["td_case2"]["tracks_shorter_than_1_second"], tracking["hybrid"]["tracks_shorter_than_1_second"], decisions["track_continuity_winner"]["decision"], "Lower suggests better continuity."),
        _row("Accepted merges", tracking["td_case2"]["accepted_merges"], tracking["hybrid"]["accepted_merges"], "hybrid_better", "Only hybrid performs explicit fragment reconciliation here."),
        _row("Valid primary crops", crop["td_case2"]["objects_with_primary_crop"], crop["hybrid"]["objects_with_primary_crop"], decisions["crop_quality_winner"]["decision"], "A written crop is not automatically visually correct."),
        _row("Plate candidates", crop["td_case2"]["plate_candidate_count"], crop["hybrid"]["plate_candidate_count"], decisions["better_base_for_ocr_and_colour"]["decision"], "Higher is only a candidate count, not verified accuracy."),
        _row("Crop failures", crop["td_case2"]["crop_failures"], crop["hybrid"]["crop_failures"], decisions["crop_quality_winner"]["decision"], "Lower is operationally safer."),
        _row("Frozen/stale tracks", tracking["td_case2"]["frozen_or_stale_tracks"], tracking["hybrid"]["frozen_or_stale_tracks"], decisions["operational_safety_winner"]["decision"], "Hybrid explicitly detects these issues."),
    ]


def _row(metric: str, td_value: Any, hy_value: Any, better: str, notes: str) -> dict[str, str]:
    return {
        "metric": metric,
        "td_case2": _format_cell(td_value),
        "hybrid": _format_cell(hy_value),
        "better": better,
        "notes": notes,
    }
