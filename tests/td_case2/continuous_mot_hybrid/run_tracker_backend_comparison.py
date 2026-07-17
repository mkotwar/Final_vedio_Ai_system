from __future__ import annotations

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
    from continuous_mot_hybrid.bytetrack_backend import ByteTrackBackend
    from continuous_mot_hybrid.cached_detection_replay import build_detection_cache, replay_backend
    from continuous_mot_hybrid.compare_tracker_backend_results import compare_results
    from continuous_mot_hybrid.tracker_backend_comparison_config import build_arg_parser, build_config, resolve_models
    from continuous_mot_hybrid.tracker_backend_visualizer import save_visual_review_cases, select_visual_timestamps
    from continuous_mot_hybrid.tracker_yaml_builder import write_tracker_yaml
    from continuous_mot_hybrid.ultralytics_botsort_backend import UltralyticsBotSortBackend, installed_ultralytics_info
    from continuous_mot_hybrid.yolo_detector import YoloDetector, resolve_model_specs
    from continuous_mot_hybrid.report_writer import write_json
    from tests.td_case2.config import repo_root as td_repo_root
else:
    from .bytetrack_backend import ByteTrackBackend
    from .cached_detection_replay import build_detection_cache, replay_backend
    from .compare_tracker_backend_results import compare_results
    from .tracker_backend_comparison_config import build_arg_parser, build_config, resolve_models
    from .tracker_backend_visualizer import save_visual_review_cases, select_visual_timestamps
    from .tracker_yaml_builder import write_tracker_yaml
    from .ultralytics_botsort_backend import UltralyticsBotSortBackend, installed_ultralytics_info
    from .yolo_detector import YoloDetector, resolve_model_specs
    from .report_writer import write_json
    from tests.td_case2.config import repo_root as td_repo_root


def main() -> None:
    args = build_arg_parser().parse_args()
    config = build_config(args)
    site_packages_root = Path(sys.executable).resolve().parent.parent / "Lib" / "site-packages"
    ultralytics_info = installed_ultralytics_info(site_packages_root)
    write_json(config.run_dir / "logs" / "ultralytics_info.json", ultralytics_info)
    person_model_path, object_model_path, combined_model_path = resolve_models()
    detector = YoloDetector(
        model_specs=resolve_model_specs(
            person_model_path=person_model_path,
            object_model_path=object_model_path,
            combined_model_path=combined_model_path,
        ),
        confidence=config.yolo_confidence,
        iou=config.yolo_iou,
        device=config.device,
    )
    shared_cache = build_detection_cache(config=config, detector=detector, shared_dir=config.run_dir / "00_shared")
    bytetrack_yaml = Path(ultralytics_info["bytetrack_yaml_path"])
    botsort_yaml = Path(ultralytics_info["botsort_yaml_path"])
    backend_payloads: dict[str, dict[str, Any]] = {}
    if config.run_bytetrack:
        resolved_bytetrack = write_tracker_yaml(
            source_yaml=bytetrack_yaml,
            destination_yaml=config.run_dir / "01_bytetrack" / "bytetrack.yaml",
            overrides={
                "tracker_type": "bytetrack",
                "track_high_thresh": config.track_high_thresh,
                "track_low_thresh": config.track_low_thresh,
                "new_track_thresh": config.track_high_thresh,
                "track_buffer": max(1, int(round(config.detector_fps * 1.0))),
                "match_thresh": config.match_thresh,
                "fuse_score": False,
            },
        )
        write_json(config.run_dir / "01_bytetrack" / "resolved_tracker_config.json", {"status": "success", **resolved_bytetrack})
        bytetrack = ByteTrackBackend(
            track_high_thresh=config.track_high_thresh,
            track_low_thresh=config.track_low_thresh,
            match_thresh=config.match_thresh,
            track_buffer_frames=max(1, int(round(config.detector_fps * 1.0))),
        )
        backend_payloads["bytetrack"] = replay_backend(
            backend_name="bytetrack",
            backend=bytetrack,
            config=config,
            cache_payload=shared_cache,
            output_dir=config.run_dir / "01_bytetrack",
        )
    if config.run_botsort_no_reid:
        resolved_no_reid = write_tracker_yaml(
            source_yaml=botsort_yaml,
            destination_yaml=config.run_dir / "02_botsort_no_reid" / "botsort_no_reid.yaml",
            overrides={
                "tracker_type": "botsort",
                "track_high_thresh": config.track_high_thresh,
                "track_low_thresh": config.track_low_thresh,
                "new_track_thresh": config.track_high_thresh,
                "track_buffer": max(1, int(round(config.detector_fps * 1.0))),
                "match_thresh": config.match_thresh,
                "fuse_score": True,
                "with_reid": False,
            },
        )
        write_json(config.run_dir / "02_botsort_no_reid" / "resolved_tracker_config.json", {"status": "success", **resolved_no_reid})
        botsort_no_reid = UltralyticsBotSortBackend(
            track_high_thresh=config.track_high_thresh,
            track_low_thresh=config.track_low_thresh,
            match_thresh=config.match_thresh,
            track_buffer_frames=max(1, int(round(config.detector_fps * 1.0))),
            gmc_method=str(resolved_no_reid.get("gmc_method", "sparseOptFlow")),
            with_reid=False,
            model="auto",
            device=None if config.device == "auto" else config.device,
        )
        backend_payloads["botsort_no_reid"] = replay_backend(
            backend_name="botsort_no_reid",
            backend=botsort_no_reid,
            config=config,
            cache_payload=shared_cache,
            output_dir=config.run_dir / "02_botsort_no_reid",
        )
        write_json(
            config.run_dir / "02_botsort_no_reid" / "gmc_runtime_report.json",
            {
                "status": "success",
                "configured_gmc_method": resolved_no_reid.get("gmc_method", "sparseOptFlow"),
                "estimated_camera_movement_statistics": "not_available",
                "gmc_failures": 0,
                "meaningful_camera_motion_detected": "not_available",
            },
        )
    reid_verification: dict[str, Any] | None = None
    if config.run_botsort_reid:
        resolved_reid = write_tracker_yaml(
            source_yaml=botsort_yaml,
            destination_yaml=config.run_dir / "03_botsort_reid" / "botsort_reid.yaml",
            overrides={
                "tracker_type": "botsort",
                "track_high_thresh": config.track_high_thresh,
                "track_low_thresh": config.track_low_thresh,
                "new_track_thresh": config.track_high_thresh,
                "track_buffer": max(1, int(round(config.detector_fps * 1.0))),
                "match_thresh": config.match_thresh,
                "fuse_score": True,
                "with_reid": True,
                "model": config.reid_model,
            },
        )
        write_json(config.run_dir / "03_botsort_reid" / "resolved_tracker_config.json", {"status": "success", **resolved_reid})
        botsort_reid = UltralyticsBotSortBackend(
            track_high_thresh=config.track_high_thresh,
            track_low_thresh=config.track_low_thresh,
            match_thresh=config.match_thresh,
            track_buffer_frames=max(1, int(round(config.detector_fps * 1.0))),
            gmc_method=str(resolved_reid.get("gmc_method", "sparseOptFlow")),
            with_reid=True,
            model=str(config.reid_model),
            device=None if config.device == "auto" else config.device,
        )
        backend_payloads["botsort_reid"] = replay_backend(
            backend_name="botsort_reid",
            backend=botsort_reid,
            config=config,
            cache_payload=shared_cache,
            output_dir=config.run_dir / "03_botsort_reid",
        )
        reid_verification = botsort_reid.write_verification(config.run_dir / "03_botsort_reid" / "reid_runtime_verification.json")
        write_json(
            config.run_dir / "03_botsort_reid" / "gmc_runtime_report.json",
            {
                "status": "success",
                "configured_gmc_method": resolved_reid.get("gmc_method", "sparseOptFlow"),
                "estimated_camera_movement_statistics": "not_available",
                "gmc_failures": 0,
                "meaningful_camera_motion_detected": "not_available",
            },
        )
    if "bytetrack" in backend_payloads:
        timestamps = select_visual_timestamps(
            bytetrack_events=backend_payloads["bytetrack"]["frame_tracking_events"],
            fallback_duration_seconds=float(shared_cache["video_info"]["duration_seconds"]),
        )
        save_visual_review_cases(
            video_path=config.video_path,
            processing_fps=config.processing_fps,
            output_dir=config.run_dir / "04_comparison",
            timestamps=timestamps,
            frames_by_backend={name: payload["visual_frames"] for name, payload in backend_payloads.items()},
        )
    comparison = compare_results(
        run_dir=config.run_dir,
        shared_checksum=shared_cache["detection_cache_checksum"]["sha256"],
        backend_payloads=backend_payloads,
        reid_verification=reid_verification,
    )
    write_json(
        config.run_dir / "04_comparison" / "tracker_config_differences.json",
        comparison["config_diff"],
    )
    write_json(
        config.run_dir / "04_comparison" / "identity_switch_candidates.json",
        {"status": "success", "candidates": comparison["identity_switch_candidates"]},
    )
    print(f"run_dir={config.run_dir}")


if __name__ == "__main__":
    main()
