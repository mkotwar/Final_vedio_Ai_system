from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from comparison_metrics import (
        build_class_count_comparison,
        build_config_differences,
        build_crop_quality_comparison,
        build_decisions,
        build_detector_usage_comparison,
        build_approximate_cross_pipeline_matches,
        build_failure_comparison,
        build_manual_review_summary,
        build_normalized_metrics,
        build_runtime_comparison,
        build_tracking_count_comparison,
    )
    from comparison_report import build_final_report_html, build_final_report_markdown, build_final_table_rows, write_json
    from comparison_visual_review import build_visual_review_manifest
    from hybrid_runner_adapter import run_hybrid_pipeline
    from td_case2_runner_adapter import run_td_case2_pipeline
else:
    from .comparison_metrics import (
        build_class_count_comparison,
        build_config_differences,
        build_crop_quality_comparison,
        build_decisions,
        build_detector_usage_comparison,
        build_approximate_cross_pipeline_matches,
        build_failure_comparison,
        build_manual_review_summary,
        build_normalized_metrics,
        build_runtime_comparison,
        build_tracking_count_comparison,
    )
    from .comparison_report import build_final_report_html, build_final_report_markdown, build_final_table_rows, write_json
    from .comparison_visual_review import build_visual_review_manifest
    from .hybrid_runner_adapter import run_hybrid_pipeline
    from .td_case2_runner_adapter import run_td_case2_pipeline


def _sanitize_stem(name: str) -> str:
    return "_".join(name.replace(".", " ").split())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run td_case2 and hybrid tracking on the same video and compare outputs.")
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--camera-id", default="test_cam_01")
    parser.add_argument("--camera-group", default="single_camera_comparison")
    parser.add_argument("--camera-timezone", default="Asia/Kolkata")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    video_path = Path(args.video_path).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = repo_root / "debug_runs" / f"td_case2_vs_hybrid_{_sanitize_stem(video_path.stem)}_{timestamp}"
    td_case2_root = run_root / "td_case2"
    hybrid_root = run_root / "hybrid"
    comparison_dir = run_root / "comparison"
    logs_dir = run_root / "logs"
    td_case2_root.mkdir(parents=True, exist_ok=True)
    hybrid_root.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    td_case2_artifacts = run_td_case2_pipeline(
        repo_root=repo_root,
        video_path=video_path,
        output_root=td_case2_root,
        logs_dir=logs_dir,
    )
    hybrid_artifacts = run_hybrid_pipeline(
        repo_root=repo_root,
        run_dir=hybrid_root,
        video_path=video_path,
        camera_id=args.camera_id,
        camera_group=args.camera_group,
        camera_timezone=args.camera_timezone,
        logs_dir=logs_dir,
    )

    config_differences = build_config_differences(
        td_case2_artifacts.to_dict(),
        hybrid_artifacts.to_dict(),
        args.camera_id,
        args.camera_group,
        args.camera_timezone,
    )
    runtime = build_runtime_comparison(td_case2_artifacts.metrics, hybrid_artifacts.metrics)
    detector = build_detector_usage_comparison(td_case2_artifacts.metrics, hybrid_artifacts.metrics)
    tracking = build_tracking_count_comparison(td_case2_artifacts.metrics, hybrid_artifacts.metrics)
    crop = build_crop_quality_comparison(td_case2_artifacts.metrics, hybrid_artifacts.metrics)
    class_counts = build_class_count_comparison(td_case2_artifacts.metrics, hybrid_artifacts.metrics)
    failures = build_failure_comparison(td_case2_artifacts.metrics, hybrid_artifacts.metrics)
    manual_review = build_manual_review_summary(hybrid_artifacts.metrics)
    normalized = build_normalized_metrics(td_case2_artifacts.metrics, hybrid_artifacts.metrics, runtime)
    decisions = build_decisions(runtime, detector, tracking, crop)
    approximate_matches = build_approximate_cross_pipeline_matches(td_case2_artifacts.metrics, hybrid_artifacts.metrics)
    visual_manifest = build_visual_review_manifest(
        comparison_dir=comparison_dir,
        video_path=video_path,
        td_case2_metrics=td_case2_artifacts.metrics,
        hybrid_metrics=hybrid_artifacts.metrics,
        count=20,
    )

    write_json(comparison_dir / "01_comparison_config.json", {
        "status": "success",
        "video_path": str(video_path),
        "camera_id": args.camera_id,
        "camera_group": args.camera_group,
        "camera_timezone": args.camera_timezone,
        "td_case2_run_dir": td_case2_artifacts.run_dir,
        "hybrid_run_dir": hybrid_artifacts.run_dir,
        "td_case2_command": td_case2_artifacts.command,
        "hybrid_command": hybrid_artifacts.command,
    })
    write_json(comparison_dir / "config_differences.json", config_differences)
    write_json(comparison_dir / "02_runtime_comparison.json", runtime)
    write_json(comparison_dir / "03_detector_usage_comparison.json", detector)
    write_json(comparison_dir / "04_tracking_count_comparison.json", tracking)
    write_json(comparison_dir / "05_track_quality_comparison.json", {"td_case2": td_case2_artifacts.metrics.get("track_quality_counts"), "hybrid": hybrid_artifacts.metrics["quality_report"]})
    write_json(comparison_dir / "06_crop_quality_comparison.json", crop)
    write_json(comparison_dir / "07_class_count_comparison.json", class_counts)
    write_json(comparison_dir / "08_failure_comparison.json", failures)
    write_json(comparison_dir / "approximate_cross_pipeline_matches.json", approximate_matches)

    final_summary = {
        "status": "success",
        "comparison_run_directory": str(run_root),
        "video_path": str(video_path),
        "td_case2_run_dir": td_case2_artifacts.run_dir,
        "hybrid_run_dir": hybrid_artifacts.run_dir,
        "runtime_comparison": runtime,
        "detector_usage_comparison": detector,
        "tracking_count_comparison": tracking,
        "crop_quality_comparison": crop,
        "class_count_comparison": class_counts,
        "failure_comparison": failures,
        "normalized_metrics": normalized,
        "manual_review": manual_review,
        "config_differences": config_differences,
        "decisions": decisions,
    }
    final_summary["final_table_rows"] = build_final_table_rows(runtime, detector, tracking, crop, decisions)
    markdown = build_final_report_markdown(final_summary)
    html = build_final_report_html(markdown)
    write_json(comparison_dir / "10_final_comparison_report.json", final_summary)
    (comparison_dir / "10_final_comparison_report.md").write_text(markdown, encoding="utf-8")
    (comparison_dir / "10_final_comparison_report.html").write_text(html, encoding="utf-8")
    print(f"Comparison run directory: {run_root}")


if __name__ == "__main__":
    main()
