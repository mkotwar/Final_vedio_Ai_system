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
    from hybrid_tracking_test.local_identity_package import build_local_identity_packages
    from hybrid_tracking_test.representative_frame_selector import build_representative_frames
    from hybrid_tracking_test.track_fragment_reconciliation import (
        MergeScoringConfig,
        build_track_quality_report,
        load_hybrid_inputs,
        reconcile_track_fragments,
        write_reconciliation_outputs,
        write_track_quality_outputs,
    )
else:
    from .local_identity_package import build_local_identity_packages
    from .representative_frame_selector import build_representative_frames
    from .track_fragment_reconciliation import (
        MergeScoringConfig,
        build_track_quality_report,
        load_hybrid_inputs,
        reconcile_track_fragments,
        write_reconciliation_outputs,
        write_track_quality_outputs,
    )


ENV_RUN_DIR = "TD_CASE2_RUN_DIR"
ENV_VIDEO_PATH = "TD_CASE2_VIDEO_PATH"
ENV_CAMERA_ID = "TD_CASE2_CAMERA_ID"
ENV_CAMERA_GROUP = "TD_CASE2_CAMERA_GROUP"
ENV_CAMERA_TIMEZONE = "TD_CASE2_CAMERA_TIMEZONE"
ENV_RECONCILE_MAX_GAP_SECONDS = "TD_CASE2_RECONCILE_MAX_GAP_SECONDS"
ENV_RECONCILE_AUTO_MERGE_SCORE = "TD_CASE2_RECONCILE_AUTO_MERGE_SCORE"
ENV_RECONCILE_POSSIBLE_MERGE_SCORE = "TD_CASE2_RECONCILE_POSSIBLE_MERGE_SCORE"
ENV_BEST_FRAME_TOP_K = "TD_CASE2_BEST_FRAME_TOP_K"
ENV_BEST_FRAME_MIN_GAP_SECONDS = "TD_CASE2_BEST_FRAME_MIN_GAP_SECONDS"
ENV_CROP_VEHICLE_PADDING = "TD_CASE2_CROP_VEHICLE_PADDING"
ENV_CROP_PERSON_PADDING = "TD_CASE2_CROP_PERSON_PADDING"


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_float(raw_value: str | None, default_value: float) -> float:
    if raw_value is None or not str(raw_value).strip():
        return default_value
    return float(str(raw_value).strip())


def _read_int(raw_value: str | None, default_value: int) -> int:
    if raw_value is None or not str(raw_value).strip():
        return default_value
    return int(str(raw_value).strip())


def _arg_env(cli_value: str | None, env_name: str, default_value: str | None = None) -> str | None:
    if cli_value is not None and str(cli_value).strip():
        return str(cli_value).strip()
    raw_env = os.environ.get(env_name)
    if raw_env is not None and str(raw_env).strip():
        return str(raw_env).strip()
    return default_value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline post-tracking reconciliation and best-frame selection for isolated td_case2 hybrid outputs.")
    parser.add_argument("--run-dir")
    parser.add_argument("--hybrid-output-dir")
    parser.add_argument("--video-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--camera-id")
    parser.add_argument("--camera-group")
    parser.add_argument("--camera-timezone")
    return parser


def main() -> None:
    started = time.perf_counter()
    args = build_arg_parser().parse_args()
    run_dir_value = _arg_env(args.run_dir, ENV_RUN_DIR)
    if not run_dir_value:
        raise ValueError(f"{ENV_RUN_DIR} or --run-dir is required.")
    run_dir = Path(run_dir_value).expanduser().resolve()
    hybrid_output_dir = Path(_arg_env(args.hybrid_output_dir, "", str(run_dir / "hybrid_tracking_test")) or str(run_dir / "hybrid_tracking_test")).expanduser().resolve()
    hybrid_inputs = load_hybrid_inputs(hybrid_output_dir)
    report_payload = dict(hybrid_inputs["report"])
    config_payload = dict(hybrid_inputs["config"])
    video_path = Path(_arg_env(args.video_path, ENV_VIDEO_PATH, str(config_payload.get("video_path") or report_payload["video_metadata"]["input_video_path"])) or "").expanduser().resolve()
    output_dir = Path(_arg_env(args.output_dir, "", str(hybrid_output_dir / "post_tracking")) or str(hybrid_output_dir / "post_tracking")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_id = _arg_env(args.camera_id, ENV_CAMERA_ID, "test_cam_01") or "test_cam_01"
    camera_group = _arg_env(args.camera_group, ENV_CAMERA_GROUP, "single_camera_test") or "single_camera_test"
    camera_timezone = _arg_env(args.camera_timezone, ENV_CAMERA_TIMEZONE, "Asia/Kolkata") or "Asia/Kolkata"

    raw_tracks = list(hybrid_inputs["tracks"].get("track_summaries", []))
    frame_width = int(report_payload["video_metadata"]["width"])
    frame_height = int(report_payload["video_metadata"]["height"])
    quality_report = build_track_quality_report(raw_tracks, frame_width=frame_width, frame_height=frame_height)
    write_track_quality_outputs(output_dir, quality_report)

    merge_config = MergeScoringConfig(
        maximum_merge_gap_seconds=_read_float(os.environ.get(ENV_RECONCILE_MAX_GAP_SECONDS), 2.0),
        automatic_merge_score=_read_float(os.environ.get(ENV_RECONCILE_AUTO_MERGE_SCORE), 0.78),
        possible_merge_score=_read_float(os.environ.get(ENV_RECONCILE_POSSIBLE_MERGE_SCORE), 0.62),
    )
    reconciled_local_objects, accepted_merges, rejected_candidates, reconciliation_report = reconcile_track_fragments(
        raw_tracks,
        quality_report,
        camera_id=camera_id,
        camera_group=camera_group,
        camera_timezone=camera_timezone,
        scoring_config=merge_config,
    )
    write_reconciliation_outputs(
        output_dir,
        reconciled_tracks=reconciled_local_objects,
        merge_events=accepted_merges,
        rejected_candidates=rejected_candidates,
        report=reconciliation_report,
    )

    representative_frames, representative_report, crop_failures = build_representative_frames(
        video_path=video_path,
        local_objects=reconciled_local_objects,
        frame_width=frame_width,
        frame_height=frame_height,
        post_tracking_dir=output_dir,
        top_k=_read_int(os.environ.get(ENV_BEST_FRAME_TOP_K), 3),
        minimum_gap_seconds=_read_float(os.environ.get(ENV_BEST_FRAME_MIN_GAP_SECONDS), 0.3),
        vehicle_padding=_read_float(os.environ.get(ENV_CROP_VEHICLE_PADDING), 0.10),
        person_padding=_read_float(os.environ.get(ENV_CROP_PERSON_PADDING), 0.12),
    )
    _write_json(output_dir / "05_representative_frames.json", {"status": "success", "objects": representative_frames})
    _write_json(output_dir / "05_representative_frames_report.json", representative_report)
    _write_json(output_dir / "05_best_track_crops.json", {"status": "success", "objects": representative_frames})
    _write_json(output_dir / "05_best_track_crops_report.json", representative_report)
    _write_json(output_dir / "05_crop_failures.json", crop_failures)

    local_packages, flat_packages, package_report = build_local_identity_packages(
        local_objects=reconciled_local_objects,
        representative_frames=representative_frames,
    )
    _write_json(output_dir / "05_local_identity_packages.json", {"status": "success", "packages": local_packages})
    _write_json(output_dir / "05_local_identity_packages_flat.json", {"status": "success", "rows": flat_packages})
    _write_json(output_dir / "05_local_identity_package_report.json", package_report)

    runtime_seconds = time.perf_counter() - started
    print("Post-tracking processing completed")
    print(f"Raw track IDs: {reconciliation_report['raw_track_id_count']}")
    print(f"Confirmed raw track segments: {reconciliation_report['confirmed_raw_track_segment_count']}")
    print(f"Reconciled local objects: {reconciliation_report['reconciled_local_object_count']}")
    print(f"Accepted fragment merges: {reconciliation_report['accepted_merge_count']}")
    print(f"Possible merges: {reconciliation_report['possible_merge_count']}")
    print(f"Rejected merge candidates: {reconciliation_report['rejected_merge_candidate_count']}")
    print(f"Objects with primary crop: {representative_report['objects_with_primary_crop']}")
    print(f"Objects with plate candidate: {representative_report['objects_with_plate_candidate']}")
    print(f"Fallback objects: {representative_report['fallback_objects']}")
    print(f"Crop failures: {representative_report['crop_failures']}")
    print(f"Runtime: {runtime_seconds:.3f}s")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
