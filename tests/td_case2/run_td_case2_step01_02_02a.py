from __future__ import annotations

from run_td_case2_step01_02 import (
    create_run_dir,
    extract_video_info,
    log,
    read_config,
    sample_frames,
)
from stage_checks import build_failure_payload, update_stage_gate_report, write_json
from step_02a_motion_adaptive_sampling import read_adaptive_config, run_motion_adaptive_sampling


def main() -> None:
    """Run the isolated td_case2 pipeline through step 02A adaptive sampling."""

    config = read_config()
    adaptive_config = read_adaptive_config()
    run_dir = create_run_dir(config)

    log(f"Input video path: {config.input_video_path}")
    log(f"Sample interval: {config.sample_every_seconds} seconds")
    log(
        "Adaptive ROI: "
        f"{adaptive_config.roi_mode} "
        f"{[round(value, 4) for value in adaptive_config.road_roi_rect_normalized]}"
    )
    log(f"Created run directory: {run_dir}")

    try:
        video_info = extract_video_info(config.input_video_path)
        write_json(run_dir / "01_video_info.json", video_info)
        update_stage_gate_report(
            run_dir,
            "01_video_info",
            {
                "status": "success",
                "fps": video_info["fps"],
                "frame_count": video_info["frame_count"],
                "duration_seconds": video_info["duration_seconds"],
            },
        )
        log("Step 01 complete: video info written")
    except Exception as exc:
        update_stage_gate_report(run_dir, "01_video_info", build_failure_payload(exc))
        log(f"Step 01 failed: {exc}")
        log(f"Run directory: {run_dir}")
        raise

    try:
        sample_manifest = sample_frames(run_dir, video_info, config.sample_every_seconds)
        write_json(run_dir / "02_sampled_frames.json", sample_manifest)
        update_stage_gate_report(
            run_dir,
            "02_base_frame_sampling",
            {
                "status": "success",
                "sample_every_seconds": config.sample_every_seconds,
                "actual_sample_count": sample_manifest["actual_sample_count"],
                "sampled_frames_folder_exists": (run_dir / "02_sampled_frames").exists(),
            },
        )
        log("Step 02 complete: sampled frames written")
    except Exception as exc:
        update_stage_gate_report(run_dir, "02_base_frame_sampling", build_failure_payload(exc))
        log(f"Step 02 failed: {exc}")
        log(f"Run directory: {run_dir}")
        raise

    try:
        adaptive_manifest, _adaptive_report = run_motion_adaptive_sampling(run_dir, adaptive_config)
        update_stage_gate_report(
            run_dir,
            "02A_motion_adaptive_sampling",
            {
                "status": "success",
                "input_sampled_frames": adaptive_manifest["input_sampled_frames"],
                "selected_for_yolo": adaptive_manifest["selected_for_yolo"],
                "selection_ratio": adaptive_manifest["selection_ratio"],
                "adaptive_preview_exists": (run_dir / "02A_adaptive_preview_frames").exists(),
            },
        )
        log("Step 02A complete: motion adaptive sampling written")
        log(f"Input sampled frames: {adaptive_manifest['input_sampled_frames']}")
        log(f"Selected for YOLO: {adaptive_manifest['selected_for_yolo']}")
        log(f"Selection ratio: {adaptive_manifest['selection_ratio']}")
    except Exception as exc:
        update_stage_gate_report(run_dir, "02A_motion_adaptive_sampling", build_failure_payload(exc))
        log(f"Step 02A failed: {exc}")
        log(f"Run directory: {run_dir}")
        raise

    log(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
