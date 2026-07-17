from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    case_root = Path(__file__).resolve().parents[1]
    if str(case_root) not in sys.path:
        sys.path.insert(0, str(case_root))
    from hybrid_tracking_test.crop_extractor import save_validated_frame_and_crop
    from hybrid_tracking_test.kcf_drift_detector import DriftDetectionConfig, detect_kcf_drift_segments
    from hybrid_tracking_test.local_identity_package import build_local_identity_packages
    from hybrid_tracking_test.reconciliation_candidate_index import CandidateIndexConfig, generate_reconciliation_candidates
    from hybrid_tracking_test.representative_frame_selector import build_representative_frames_v2
    from hybrid_tracking_test.track_fragment_reconciliation import (
        MergeScoringConfig,
        load_hybrid_inputs,
        reconcile_track_fragments,
    )
    from hybrid_tracking_test.track_integrity_validator import build_track_integrity_report
    from hybrid_tracking_test.track_quality import build_track_quality_markdown, build_track_quality_report
    from hybrid_tracking_test.track_timeline_rebuilder import rebuild_track_timelines
    from hybrid_tracking_test.trajectory_sanitizer import SanitizationConfig, sanitize_track_timeline
else:
    from .kcf_drift_detector import DriftDetectionConfig, detect_kcf_drift_segments
    from .local_identity_package import build_local_identity_packages
    from .reconciliation_candidate_index import CandidateIndexConfig, generate_reconciliation_candidates
    from .representative_frame_selector import build_representative_frames_v2
    from .track_fragment_reconciliation import MergeScoringConfig, load_hybrid_inputs, reconcile_track_fragments
    from .track_integrity_validator import build_track_integrity_report
    from .track_quality import build_track_quality_markdown, build_track_quality_report
    from .track_timeline_rebuilder import rebuild_track_timelines
    from .trajectory_sanitizer import SanitizationConfig, sanitize_track_timeline


ENV_RUN_DIR = "TD_CASE2_RUN_DIR"
ENV_CAMERA_ID = "TD_CASE2_CAMERA_ID"
ENV_CAMERA_GROUP = "TD_CASE2_CAMERA_GROUP"
ENV_CAMERA_TIMEZONE = "TD_CASE2_CAMERA_TIMEZONE"
ENV_TIMELINE_TOLERANCE = "TD_CASE2_TIMELINE_TOLERANCE_SECONDS"
ENV_FROZEN_WINDOW_SECONDS = "TD_CASE2_FROZEN_WINDOW_SECONDS"
ENV_FROZEN_MIN_OBSERVATIONS = "TD_CASE2_FROZEN_MIN_OBSERVATIONS"
ENV_FROZEN_MAX_CENTER_MOTION_DIAGONALS = "TD_CASE2_FROZEN_MAX_CENTER_MOTION_DIAGONALS"
ENV_MAX_KCF_ONLY_READY_GAP_SECONDS = "TD_CASE2_MAX_KCF_ONLY_READY_GAP_SECONDS"
ENV_RECONCILE_MAX_GAP_SECONDS = "TD_CASE2_RECONCILE_MAX_GAP_SECONDS"
ENV_RECONCILE_AUTO_MERGE_SCORE = "TD_CASE2_RECONCILE_AUTO_MERGE_SCORE"
ENV_RECONCILE_POSSIBLE_MERGE_SCORE = "TD_CASE2_RECONCILE_POSSIBLE_MERGE_SCORE"
ENV_MAX_READY_CROP_CLIPPING_RATIO = "TD_CASE2_MAX_READY_CROP_CLIPPING_RATIO"
ENV_MAX_FALLBACK_CROP_CLIPPING_RATIO = "TD_CASE2_MAX_FALLBACK_CROP_CLIPPING_RATIO"
ENV_MIN_PLATE_CANDIDATE_SCORE = "TD_CASE2_MIN_PLATE_CANDIDATE_SCORE"


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_markdown(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _arg_env(cli_value: str | None, env_name: str, default_value: str | None = None) -> str | None:
    if cli_value is not None and str(cli_value).strip():
        return str(cli_value).strip()
    if env_name:
        raw_env = os.environ.get(env_name)
        if raw_env is not None and str(raw_env).strip():
            return str(raw_env).strip()
    return default_value


def _read_float(raw_value: str | None, default_value: float) -> float:
    if raw_value is None or not str(raw_value).strip():
        return default_value
    return float(str(raw_value).strip())


def _read_int(raw_value: str | None, default_value: int) -> int:
    if raw_value is None or not str(raw_value).strip():
        return default_value
    return int(str(raw_value).strip())


def _comparison_markdown(payload: dict[str, Any]) -> str:
    lines = ["# Before/After Post-Tracking Comparison", ""]
    for key, value in payload.items():
        if key in {"status", "notes"}:
            continue
        label = key.replace("_", " ")
        lines.append(f"- {label}: {value}")
    if payload.get("notes"):
        lines.append("")
        for note in list(payload["notes"]):
            lines.append(f"- {note}")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repair isolated td_case2 hybrid post-tracking outputs using existing frame-level observations.")
    parser.add_argument("--run-dir")
    parser.add_argument("--hybrid-output-dir")
    parser.add_argument("--previous-post-tracking-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--camera-id")
    parser.add_argument("--camera-group")
    parser.add_argument("--camera-timezone")
    parser.add_argument("--max-merge-gap-seconds", type=float)
    parser.add_argument("--auto-merge-score", type=float)
    parser.add_argument("--possible-merge-score", type=float)
    parser.add_argument("--frozen-window-seconds", type=float)
    parser.add_argument("--max-kcf-only-ready-gap-seconds", type=float)
    return parser


def main() -> None:
    started = time.perf_counter()
    args = build_arg_parser().parse_args()
    raw_run_dir = _arg_env(args.run_dir, ENV_RUN_DIR)
    if not raw_run_dir:
        raise ValueError(f"{ENV_RUN_DIR} or --run-dir is required.")
    run_dir = Path(raw_run_dir).expanduser().resolve()
    hybrid_output_dir = Path(
        _arg_env(args.hybrid_output_dir, "", str(run_dir / "hybrid_tracking_test")) or str(run_dir / "hybrid_tracking_test")
    ).expanduser().resolve()
    previous_post_tracking_dir = Path(
        _arg_env(args.previous_post_tracking_dir, "", str(hybrid_output_dir / "post_tracking")) or str(hybrid_output_dir / "post_tracking")
    ).expanduser().resolve()
    output_dir = Path(
        _arg_env(args.output_dir, "", str(hybrid_output_dir / "post_tracking_v2")) or str(hybrid_output_dir / "post_tracking_v2")
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "best_crops").mkdir(parents=True, exist_ok=True)
    (output_dir / "full_frames").mkdir(parents=True, exist_ok=True)
    (output_dir / "invalid_crop_debug").mkdir(parents=True, exist_ok=True)
    (output_dir / "debug_track_timelines").mkdir(parents=True, exist_ok=True)
    hybrid_inputs = load_hybrid_inputs(hybrid_output_dir)
    raw_tracks = list(hybrid_inputs["tracks"].get("track_summaries", []))
    report_payload = dict(hybrid_inputs["report"])
    config_payload = dict(hybrid_inputs["config"])
    video_path = Path(str(config_payload.get("video_path") or report_payload["video_metadata"]["input_video_path"])).expanduser().resolve()
    camera_id = _arg_env(args.camera_id, ENV_CAMERA_ID, "test_cam_01") or "test_cam_01"
    camera_group = _arg_env(args.camera_group, ENV_CAMERA_GROUP, "single_camera_test") or "single_camera_test"
    camera_timezone = _arg_env(args.camera_timezone, ENV_CAMERA_TIMEZONE, "Asia/Kolkata") or "Asia/Kolkata"
    frame_width = int(report_payload["video_metadata"]["width"])
    frame_height = int(report_payload["video_metadata"]["height"])

    rebuilt_tracks, rebuild_report, rebuild_markdown = rebuild_track_timelines(
        raw_tracks,
        hybrid_inputs["frame_metrics"],
        report_payload,
        timeline_timestamp_tolerance_seconds=_read_float(os.environ.get(ENV_TIMELINE_TOLERANCE), 0.15),
    )
    _write_json(output_dir / "04d2_track_timeline_rebuild.json", {"status": "success", "tracks": rebuilt_tracks})
    drift_reports = [
        detect_kcf_drift_segments(
            track,
            frame_width=frame_width,
            frame_height=frame_height,
            config=DriftDetectionConfig(
                frozen_window_seconds=_read_float(_arg_env(str(args.frozen_window_seconds) if args.frozen_window_seconds is not None else None, ENV_FROZEN_WINDOW_SECONDS), 0.8),
                frozen_min_observations=_read_int(os.environ.get(ENV_FROZEN_MIN_OBSERVATIONS), 5),
                frozen_max_center_motion_diagonals=_read_float(os.environ.get(ENV_FROZEN_MAX_CENTER_MOTION_DIAGONALS), 0.05),
                maximum_kcf_only_ready_gap_seconds=_read_float(
                    _arg_env(
                        str(args.max_kcf_only_ready_gap_seconds) if args.max_kcf_only_ready_gap_seconds is not None else None,
                        ENV_MAX_KCF_ONLY_READY_GAP_SECONDS,
                    ),
                    0.6,
                ),
            ),
        )
        for track in rebuilt_tracks
    ]
    _write_json(output_dir / "04d2_track_drift_segments.json", {"status": "success", "tracks": drift_reports})
    sanitized_tracks: list[dict[str, Any]] = []
    sanitization_events: list[dict[str, Any]] = []
    for rebuilt, drift_report in zip(rebuilt_tracks, drift_reports):
        sanitized_track, track_events = sanitize_track_timeline(
            rebuilt,
            drift_report,
            config=SanitizationConfig(
                maximum_supported_kcf_gap_seconds=0.3,
                maximum_fallback_kcf_gap_seconds=_read_float(os.environ.get(ENV_MAX_KCF_ONLY_READY_GAP_SECONDS), 0.6),
            ),
        )
        sanitized_tracks.append(sanitized_track)
        sanitization_events.extend(track_events)
    _write_json(output_dir / "04d2_track_sanitization_events.json", {"status": "success", "events": sanitization_events})
    _write_json(output_dir / "04d2_sanitized_tracks.json", {"status": "success", "tracks": sanitized_tracks})
    integrity_report, integrity_markdown = build_track_integrity_report(rebuilt_tracks, drift_reports, sanitized_tracks)
    _write_json(output_dir / "04d2_track_timeline_integrity_report.json", integrity_report)
    _write_markdown(output_dir / "04d2_track_timeline_integrity_report.md", integrity_markdown)

    quality_report = build_track_quality_report(sanitized_tracks, frame_width=frame_width, frame_height=frame_height)
    _write_json(output_dir / "04d2_track_quality_report.json", quality_report)
    _write_markdown(output_dir / "04d2_track_quality_report.md", build_track_quality_markdown(quality_report))

    candidates, candidate_report = generate_reconciliation_candidates(
        sanitized_tracks,
        config=CandidateIndexConfig(
            maximum_merge_gap_seconds=_read_float(
                _arg_env(str(args.max_merge_gap_seconds) if args.max_merge_gap_seconds is not None else None, ENV_RECONCILE_MAX_GAP_SECONDS),
                2.0,
            )
        ),
    )
    _write_json(output_dir / "04d2_reconciliation_candidates.json", {"status": "success", "report": candidate_report, "candidates": candidates})
    merge_config = MergeScoringConfig(
        maximum_merge_gap_seconds=_read_float(
            _arg_env(str(args.max_merge_gap_seconds) if args.max_merge_gap_seconds is not None else None, ENV_RECONCILE_MAX_GAP_SECONDS),
            2.0,
        ),
        automatic_merge_score=_read_float(
            _arg_env(str(args.auto_merge_score) if args.auto_merge_score is not None else None, ENV_RECONCILE_AUTO_MERGE_SCORE),
            0.75,
        ),
        possible_merge_score=_read_float(
            _arg_env(str(args.possible_merge_score) if args.possible_merge_score is not None else None, ENV_RECONCILE_POSSIBLE_MERGE_SCORE),
            0.62,
        ),
    )
    reconciled_tracks, merge_events, rejected_candidates, reconciliation_report = reconcile_track_fragments(
        sanitized_tracks,
        quality_report,
        camera_id=camera_id,
        camera_group=camera_group,
        camera_timezone=camera_timezone,
        scoring_config=merge_config,
        candidate_records=candidates,
    )
    _write_json(output_dir / "04d2_rejected_merge_candidates.json", {"status": "success", "candidates": rejected_candidates})
    _write_json(output_dir / "04d2_track_merge_events.json", {"status": "success", "events": merge_events})
    _write_json(output_dir / "04d2_reconciled_tracks.json", {"status": "success", "tracks": reconciled_tracks})
    _write_json(output_dir / "04d2_track_reconciliation_report.json", reconciliation_report)
    reconciliation_markdown = "\n".join(
        [
            "# Track Reconciliation Report",
            "",
            f"- Raw track IDs: {reconciliation_report['raw_track_id_count']}",
            f"- Reconciliation candidates: {candidate_report['candidate_count']}",
            f"- Accepted fragment merges: {reconciliation_report['accepted_merge_count']}",
            f"- Possible merges: {reconciliation_report['possible_merge_count']}",
            f"- Reconciled local objects: {reconciliation_report['reconciled_local_object_count']}",
            f"- Fragment reduction percent: {reconciliation_report['fragment_reduction_percent']:.3f}%",
            "",
            "Reconciled local-object counts remain estimated single-camera counts after timeline repair and drift validation, not ground truth.",
        ]
    )
    _write_markdown(output_dir / "04d2_track_reconciliation_report.md", reconciliation_markdown)

    representative_frames, representative_report, crop_failures, invalid_crop_candidates = build_representative_frames_v2(
        video_path=video_path,
        local_objects=reconciled_tracks,
        frame_width=frame_width,
        frame_height=frame_height,
        post_tracking_dir=output_dir,
        maximum_ready_crop_clipping_ratio=_read_float(os.environ.get(ENV_MAX_READY_CROP_CLIPPING_RATIO), 0.15),
        maximum_fallback_crop_clipping_ratio=_read_float(os.environ.get(ENV_MAX_FALLBACK_CROP_CLIPPING_RATIO), 0.35),
        minimum_plate_candidate_score=_read_float(os.environ.get(ENV_MIN_PLATE_CANDIDATE_SCORE), 0.60),
    )
    _write_json(output_dir / "05v2_representative_frames.json", {"status": "success", "objects": representative_frames})
    _write_json(output_dir / "05v2_representative_frames_report.json", representative_report)
    _write_json(output_dir / "05v2_best_track_crops.json", {"status": "success", "objects": representative_frames})
    _write_json(output_dir / "05v2_best_track_crops_report.json", representative_report)
    _write_json(output_dir / "05v2_crop_failures.json", crop_failures)
    _write_json(output_dir / "05v2_invalid_crop_candidates.json", {"status": "success", "candidates": invalid_crop_candidates})

    packages, package_rows, package_report = build_local_identity_packages(
        local_objects=reconciled_tracks,
        representative_frames=representative_frames,
    )
    _write_json(output_dir / "05v2_local_identity_packages.json", {"status": "success", "packages": packages})
    _write_json(output_dir / "05v2_local_identity_packages_flat.json", {"status": "success", "rows": package_rows})
    _write_json(output_dir / "05v2_local_identity_package_report.json", package_report)

    previous_quality = json.loads((previous_post_tracking_dir / "04d_track_quality_report.json").read_text(encoding="utf-8")) if (previous_post_tracking_dir / "04d_track_quality_report.json").exists() else {}
    previous_reconciliation = json.loads((previous_post_tracking_dir / "04d_track_reconciliation_report.json").read_text(encoding="utf-8")) if (previous_post_tracking_dir / "04d_track_reconciliation_report.json").exists() else {}
    previous_package_report = json.loads((previous_post_tracking_dir / "05_local_identity_package_report.json").read_text(encoding="utf-8")) if (previous_post_tracking_dir / "05_local_identity_package_report.json").exists() else {}
    comparison = {
        "status": "success",
        "previous_raw_track_count": int(previous_reconciliation.get("raw_track_id_count", len(raw_tracks)) or len(raw_tracks)),
        "current_raw_track_count": len(raw_tracks),
        "corrected_usable_track_count": len([item for item in sanitized_tracks if item.get("track_integrity_status") == "usable"]),
        "invalid_track_count": len([item for item in sanitized_tracks if item.get("track_integrity_status") == "invalid"]),
        "tracks_with_timeline_corrections": int(rebuild_report.get("timeline_corrections", 0) or 0),
        "frozen_kcf_tracks_detected": len([item for item in sanitized_tracks if bool(item.get("frozen_kcf_detected"))]),
        "boundary_stuck_tracks_detected": len([item for item in sanitized_tracks if bool(item.get("boundary_stuck_detected"))]),
        "trimmed_kcf_segments": len([item for item in sanitized_tracks if float(item.get("trimmed_kcf_duration_seconds", 0.0) or 0.0) > 0.0]),
        "average_original_duration": round(sum(float(item.get("duration_seconds", 0.0) or 0.0) for item in raw_tracks) / max(len(raw_tracks), 1), 6),
        "average_sanitized_duration": round(sum(float(item.get("sanitized_duration_seconds", 0.0) or 0.0) for item in sanitized_tracks) / max(len(sanitized_tracks), 1), 6),
        "accepted_merges_previous": int(previous_reconciliation.get("accepted_merge_count", 0) or 0),
        "accepted_merges_current": int(reconciliation_report.get("accepted_merge_count", 0) or 0),
        "possible_merges_current": int(reconciliation_report.get("possible_merge_count", 0) or 0),
        "reconciled_local_object_count_previous": int(previous_reconciliation.get("reconciled_local_object_count", 0) or 0),
        "reconciled_local_object_count_current": int(reconciliation_report.get("reconciled_local_object_count", 0) or 0),
        "fragment_reduction_percent_current": round(float(reconciliation_report.get("fragment_reduction_percent", 0.0) or 0.0), 6),
        "ready_packages_previous": int(previous_package_report.get("ready_packages", 0) or 0),
        "ready_packages_current": int(package_report.get("ready_packages", 0) or 0),
        "fallback_packages_current": int(package_report.get("fallback_packages", 0) or 0),
        "manual_review_packages_current": int(package_report.get("manual_review_packages", 0) or 0),
        "rejected_packages_current": int(package_report.get("rejected_packages", 0) or 0),
        "primary_crops_previous": int(previous_package_report.get("packages_with_primary_crop", 0) or 0),
        "valid_primary_crops_current": int(representative_report.get("valid_primary_crops", 0) or 0),
        "invalid_crop_candidates_current": len(invalid_crop_candidates),
        "plate_candidates_previous": int(previous_package_report.get("packages_with_plate_candidate", 0) or 0),
        "plate_candidates_current": int(representative_report.get("objects_with_plate_candidate", 0) or 0),
        "tracks_shorter_than_0_5_seconds_current": len([item for item in sanitized_tracks if float(item.get("sanitized_duration_seconds", 0.0) or 0.0) < 0.5]),
        "tracks_shorter_than_1_0_second_current": len([item for item in sanitized_tracks if float(item.get("sanitized_duration_seconds", 0.0) or 0.0) < 1.0]),
        "notes": [
            "Corrected reconciled object counts are estimated single-camera local-object counts after timeline repair and drift validation.",
            "Previous post_tracking outputs were preserved unchanged.",
        ],
    }
    _write_json(output_dir / "05v2_before_after_comparison.json", comparison)
    _write_markdown(output_dir / "05v2_before_after_comparison.md", _comparison_markdown(comparison))
    runtime_seconds = time.perf_counter() - started
    print("Post-tracking integrity correction completed")
    print(f"Raw track IDs: {len(raw_tracks)}")
    print(f"Timelines rebuilt: {rebuild_report['timelines_rebuilt']}")
    print(f"Timeline corrections: {rebuild_report['timeline_corrections']}")
    print(f"Frozen KCF tracks: {comparison['frozen_kcf_tracks_detected']}")
    print(f"Boundary-stuck tracks: {comparison['boundary_stuck_tracks_detected']}")
    print(f"Sanitized usable tracks: {comparison['corrected_usable_track_count']}")
    print(f"Invalid tracks: {comparison['invalid_track_count']}")
    print(f"Reconciliation candidates: {candidate_report['candidate_count']}")
    print(f"Accepted merges: {reconciliation_report['accepted_merge_count']}")
    print(f"Possible merges: {reconciliation_report['possible_merge_count']}")
    print(f"Reconciled local objects: {reconciliation_report['reconciled_local_object_count']}")
    print(f"Fragment reduction: {reconciliation_report['fragment_reduction_percent']:.3f}%")
    print(f"Valid primary crops: {representative_report['valid_primary_crops']}")
    print(f"Fallback crops: {representative_report['fallback_crops']}")
    print(f"Invalid crop candidates: {len(invalid_crop_candidates)}")
    print(f"Plate candidates: {representative_report['objects_with_plate_candidate']}")
    print(f"Ready packages: {package_report['ready_packages']}")
    print(f"Fallback packages: {package_report['fallback_packages']}")
    print(f"Manual review packages: {package_report['manual_review_packages']}")
    print(f"Rejected packages: {package_report['rejected_packages']}")
    print(f"Runtime: {runtime_seconds:.3f}s")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
