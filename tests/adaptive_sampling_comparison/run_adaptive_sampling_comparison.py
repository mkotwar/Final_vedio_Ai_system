from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

try:
    from tests.adaptive_sampling_comparison.adaptive_sampler_prototype import (
        format_seconds_label,
        parse_timestamp_to_seconds,
        run_adaptive_sampler,
    )
    from tests.adaptive_sampling_comparison.audit_production_adaptive_sampling import (
        run_production_adaptive_sampling_audit,
    )
    from tests.adaptive_sampling_comparison.compare_vlm_frame_selection import (
        compare_vlm_frame_selection,
    )
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    from adaptive_sampler_prototype import (
        format_seconds_label,
        parse_timestamp_to_seconds,
        run_adaptive_sampler,
    )
    from audit_production_adaptive_sampling import (
        run_production_adaptive_sampling_audit,
    )
    from compare_vlm_frame_selection import (
        compare_vlm_frame_selection,
    )


def _format_optional_seconds_label(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        return format_seconds_label(float(value))
    except (TypeError, ValueError):
        return str(value)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_required_env_path(name: str) -> Path:
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        raise ValueError(f"Environment variable {name} is required.")
    return Path(raw_value).expanduser()


def _read_target_timestamps() -> list[float]:
    raw_value = os.environ.get("ADAPTIVE_COMPARE_TARGET_TIMESTAMPS", "").strip()
    if not raw_value:
        raise ValueError("Environment variable ADAPTIVE_COMPARE_TARGET_TIMESTAMPS is required.")
    return [parse_timestamp_to_seconds(part.strip()) for part in raw_value.split(",") if part.strip()]


def _read_target_labels() -> list[str]:
    raw_value = os.environ.get("ADAPTIVE_COMPARE_TARGET_LABELS", "").strip()
    if not raw_value:
        return []
    return [part.strip() for part in raw_value.split(",")]


def _safe_float_env(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _write_summary(
    output_dir: Path,
    video_path: Path,
    tender_run_dir: Path,
    audit_result: dict,
    adaptive_result: dict,
    comparison_result: dict,
) -> Path:
    comparison_summary = comparison_result["comparison"]["summary"]
    comparison_rows = comparison_result["comparison"]["comparison"]

    likely_drop_stages: list[str] = []
    for row in comparison_rows:
        if row["verdict"] == "adaptive_better_current_focus":
            likely_drop_stages.append(
                f"{row['timestamp']} ({row.get('meaning', '')}): tender only offered {row.get('tender_coverage_type')} "
                f"while adaptive provided a focal frame at {_format_optional_seconds_label(row.get('nearest_adaptive_time'))} "
                f"({row.get('adaptive_distance_seconds')}s away)."
            )

    if comparison_summary["adaptive_better_current_focus_count"] > comparison_summary["tender_better_count"]:
        recommendation = (
            "Yes. The adaptive prototype more often provides the target as the focal/current frame, "
            "while tender sometimes only places it in PREVIOUS/NEXT context."
        )
    elif (
        comparison_summary["adaptive_better_current_focus_count"] == 0
        and comparison_summary["tender_better_count"] == 0
    ):
        recommendation = (
            "Maybe. Both approaches show similar current-panel coverage on the configured targets, "
            "so the next question is visual quality rather than simple timestamp reach."
        )
    else:
        recommendation = (
            "Not clearly. Current tender-demo focal coverage was comparable or better on the configured targets."
        )

    lines = [
        "# Adaptive Sampling Comparison Summary",
        "",
        f"- Video: `{video_path}`",
        f"- Tender run: `{tender_run_dir}`",
        f"- Production audit JSON: `{audit_result['audit_json_path']}`",
        f"- Adaptive sampling report: `{adaptive_result['report_path']}`",
        f"- Comparison JSON: `{comparison_result['comparison_json_path']}`",
        f"- Contact sheet: `{comparison_result['contact_sheet_path']}`",
        "",
        "## What did tender-demo send to Qwen?",
        "",
        f"- Tender-demo provided {len(comparison_rows)} target comparisons against existing Step 15 strip inputs.",
        f"- Tender current-panel covered count: {comparison_summary['tender_current_panel_covered_count']}",
        f"- Tender context-only covered count: {comparison_summary['tender_context_only_covered_count']}",
        f"- Tender missing count: {comparison_summary['tender_missing_count']}",
        "- Context-only coverage is weaker because Qwen mainly analyzes the CURRENT panel and uses PREVIOUS/NEXT as supporting context.",
        "",
        "## What did the adaptive prototype retain?",
        "",
        f"- Adaptive retained frames: {adaptive_result['report']['retained_frames']}",
        f"- Adaptive covered count: {comparison_summary['adaptive_covered_count']}",
        f"- Adaptive better current-focus count: {comparison_summary['adaptive_better_current_focus_count']}",
        "",
        "## Which target timestamps were only context-covered or missed by tender-demo?",
        "",
    ]

    weak_tender_rows = [
        row
        for row in comparison_rows
        if row["tender_coverage_type"] in {"context_panel_covered", "missing"}
    ]
    if weak_tender_rows:
        for row in weak_tender_rows:
            lines.append(
                f"- {row['timestamp']} ({row.get('meaning', '')}): tender was `{row.get('tender_coverage_type')}`, "
                f"nearest CURRENT was {_format_optional_seconds_label(row.get('nearest_tender_current_time'))} "
                f"({row.get('nearest_tender_current_distance')}s), nearest PREVIOUS was {_format_optional_seconds_label(row.get('nearest_tender_previous_time'))} "
                f"({row.get('nearest_tender_previous_distance')}s), nearest NEXT was {_format_optional_seconds_label(row.get('nearest_tender_next_time'))} "
                f"({row.get('nearest_tender_next_distance')}s), and nearest ANY was "
                f"{str(row.get('nearest_tender_any_panel_name', 'n/a')).upper()} {_format_optional_seconds_label(row.get('nearest_tender_any_panel_time'))} "
                f"({row.get('nearest_tender_any_panel_distance')}s)."
            )
    else:
        lines.append("- None. All configured targets appeared in tender-demo as the CURRENT/focal panel within the threshold.")

    lines.extend(["", "## Which target timestamps were captured by the adaptive prototype?", ""])
    captured_by_adaptive = [row for row in comparison_rows if row["adaptive_covered"]]
    if captured_by_adaptive:
        for row in captured_by_adaptive:
            lines.append(
                f"- {row['timestamp']} ({row.get('meaning', '')}): adaptive nearest frame at {_format_optional_seconds_label(row.get('nearest_adaptive_time'))} "
                f"with distance {row.get('adaptive_distance_seconds')}s."
            )
    else:
        lines.append("- None of the configured targets were captured by the adaptive prototype within the configured coverage window.")

    lines.extend(["", "## At which stage does tender-demo likely drop important frames?", ""])
    if likely_drop_stages:
        for item in likely_drop_stages:
            lines.append(f"- {item}")
    else:
        lines.append("- The configured targets do not show a clear tender-demo focal-frame drop point from this comparison alone.")

    lines.extend(
        [
            "",
            "## Should adaptive logic be ported into tender-demo later?",
            "",
            f"- Recommendation: {recommendation}",
            "",
            "## Verdict Table",
            "",
            "| Timestamp | Meaning | Tender coverage type | Nearest current | Nearest context | Nearest any | Adaptive focal | Verdict |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in comparison_rows:
        context_label = (
            f"PREV {_format_optional_seconds_label(row.get('nearest_tender_previous_time'))} ({row.get('nearest_tender_previous_distance')}s) / "
            f"NEXT {_format_optional_seconds_label(row.get('nearest_tender_next_time'))} ({row.get('nearest_tender_next_distance')}s)"
        )
        lines.append(
            f"| {row['timestamp']} | {row.get('meaning', '')} | {row['tender_coverage_type']} | "
            f"{_format_optional_seconds_label(row.get('nearest_tender_current_time'))} ({row.get('nearest_tender_current_distance')}s) | "
            f"{context_label} | "
            f"{str(row.get('nearest_tender_any_panel_name', 'n/a')).upper()} {_format_optional_seconds_label(row.get('nearest_tender_any_panel_time'))} ({row.get('nearest_tender_any_panel_distance')}s) | "
            f"{_format_optional_seconds_label(row.get('nearest_adaptive_time'))} ({row.get('adaptive_distance_seconds')}s) | {row['verdict']} |"
        )

    summary_path = output_dir / "comparison_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def main() -> None:
    repo_root = _repo_root()
    video_path = _read_required_env_path("ADAPTIVE_COMPARE_VIDEO_PATH")
    tender_run_dir = _read_required_env_path("ADAPTIVE_COMPARE_TENDER_RUN_DIR")
    target_timestamps = _read_target_timestamps()
    target_labels = _read_target_labels()
    coverage_window_seconds = _safe_float_env("ADAPTIVE_COMPARE_COVERAGE_THRESHOLD_SECONDS", 3.0)

    if not video_path.exists():
        raise FileNotFoundError(f"Comparison video path does not exist: {video_path}")
    if not tender_run_dir.exists():
        raise FileNotFoundError(f"Tender run directory does not exist: {tender_run_dir}")

    run_root = repo_root / "tests" / "adaptive_sampling_comparison" / "debug_runs"
    run_root.mkdir(parents=True, exist_ok=True)
    run_dir = run_root / f"{video_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"[adaptive-compare] Run directory: {run_dir}")

    audit_result = run_production_adaptive_sampling_audit(repo_root, run_dir)
    print(f"[adaptive-compare] Production audit written to: {audit_result['audit_json_path']}")

    adaptive_result = run_adaptive_sampler(
        video_path=video_path,
        output_dir=run_dir,
        target_timestamps=target_timestamps,
        target_timestamp_labels=target_labels,
    )
    print(f"[adaptive-compare] Adaptive retained frames: {adaptive_result['report']['retained_frames']}")

    comparison_result = compare_vlm_frame_selection(
        tender_run_dir=tender_run_dir,
        adaptive_output_dir=run_dir,
        output_dir=run_dir,
        target_timestamps=target_timestamps,
        target_labels=target_labels,
        coverage_window_seconds=coverage_window_seconds,
    )
    print(f"[adaptive-compare] Comparison JSON written to: {comparison_result['comparison_json_path']}")

    summary_path = _write_summary(
        output_dir=run_dir,
        video_path=video_path,
        tender_run_dir=tender_run_dir,
        audit_result=audit_result,
        adaptive_result=adaptive_result,
        comparison_result=comparison_result,
    )
    print(f"[adaptive-compare] Summary written to: {summary_path}")

    run_manifest = {
        "video_path": str(video_path),
        "tender_run_dir": str(tender_run_dir),
        "target_timestamps": [format_seconds_label(value) for value in target_timestamps],
        "target_labels": target_labels,
        "coverage_threshold_seconds": coverage_window_seconds,
        "output_dir": str(run_dir),
        "production_audit_json": str(audit_result["audit_json_path"]),
        "adaptive_sampling_report": str(adaptive_result["report_path"]),
        "comparison_json": str(comparison_result["comparison_json_path"]),
        "comparison_summary": str(summary_path),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
