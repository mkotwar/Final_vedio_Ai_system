from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .run_tracking_confidence_experiment import (
    AspectRatioClassRange,
    AspectRatioValidationConfig,
    ExperimentRunConfig,
    _build_runtime_baseline,
    _json_dump,
    _run_single_configuration,
    resolve_experiment_video,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare experimental custom tracking against clean standard ByteTrack mode.")
    parser.add_argument("--camera-config", default="tests\\td_case2\\multicamera_vehicle_tracking_pipeline\\config\\cameras.yaml")
    parser.add_argument("--detection-config", default="tests\\td_case2\\multicamera_vehicle_tracking_pipeline\\config\\detection.yaml")
    parser.add_argument("--tracking-config", default="tests\\td_case2\\multicamera_vehicle_tracking_pipeline\\config\\tracking.yaml")
    parser.add_argument("--worker-config", default="tests\\td_case2\\multicamera_vehicle_tracking_pipeline\\config\\workers.yaml")
    parser.add_argument("--persistence-config", default="tests\\td_case2\\multicamera_vehicle_tracking_pipeline\\config\\persistence.yaml")
    parser.add_argument("--evidence-config", default="tests\\td_case2\\multicamera_vehicle_tracking_pipeline\\config\\evidence.yaml")
    parser.add_argument("--anpr-config", default="tests\\td_case2\\multicamera_vehicle_tracking_pipeline\\config\\anpr.yaml")
    parser.add_argument("--florence-config", default="tests\\td_case2\\multicamera_vehicle_tracking_pipeline\\config\\florence.yaml")
    parser.add_argument("--camera-code", default="CAM_002")
    parser.add_argument("--video-path", default="tests\\td_case2\\multicamera_vehicle_tracking_pipeline\\data\\testv\\2test_20.mp4")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=599)
    parser.add_argument("--confidence-threshold", type=float, default=0.38)
    parser.add_argument("--match-threshold", type=float, default=0.80)
    parser.add_argument("--minimum-consecutive-frames", type=int, default=1)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--output-root", default="debug_runs\\multicamera_vehicle_tracking_pipeline\\standard_bytetrack_comparison")
    parser.add_argument("--skip-anpr", action="store_true")
    return parser.parse_args()


def _mode_summary(result: dict[str, Any]) -> dict[str, Any]:
    tracks = result["tracks"]
    summary = result["summary"]
    return {
        "run_label": result["run_report"]["run_label"],
        "behavior_mode": result["run_report"]["tracking_config"]["behavior_mode"],
        "native_tracks": summary["native_tracks_created"],
        "logical_tracks": summary["logical_tracks"],
        "mixed_tracks": summary["mixed_tracks"],
        "mixed_car_truck_tracks": summary.get("tracks_with_car_and_truck", 0),
        "mixed_car_bus_tracks": summary.get("tracks_with_car_and_bus", 0),
        "fragment_relinks": summary["fragment_relinks"],
        "executed_custom_splits": summary["tracks_split_by_identity_continuity"],
        "unknown_final_classes": sum(1 for track in tracks if str(track.get("final_class", "unknown")).lower() == "unknown"),
        "possible_identity_switch_tracks": sum(1 for track in tracks if bool(track.get("possible_identity_switch"))),
        "artifact_path": result["run_report"]["artifact_paths"]["run_report"],
    }


def _comparison_markdown(payload: dict[str, Any]) -> str:
    custom = payload["custom"]
    standard = payload["standard"]
    rows = [
        ("Native tracks", custom["native_tracks"], standard["native_tracks"]),
        ("Logical tracks", custom["logical_tracks"], standard["logical_tracks"]),
        ("Mixed tracks", custom["mixed_tracks"], standard["mixed_tracks"]),
        ("Mixed CAR/TRUCK tracks", custom["mixed_car_truck_tracks"], standard["mixed_car_truck_tracks"]),
        ("Mixed CAR/BUS tracks", custom["mixed_car_bus_tracks"], standard["mixed_car_bus_tracks"]),
        ("Fragment relinks", custom["fragment_relinks"], standard["fragment_relinks"]),
        ("Executed custom splits", custom["executed_custom_splits"], standard["executed_custom_splits"]),
        ("Unknown final classes", custom["unknown_final_classes"], standard["unknown_final_classes"]),
        ("Possible identity switch tracks", custom["possible_identity_switch_tracks"], standard["possible_identity_switch_tracks"]),
    ]
    lines = [
        "| Metric | Custom | Standard ByteTrack |",
        "| --- | ---: | ---: |",
    ]
    for label, custom_value, standard_value in rows:
        lines.append(f"| {label} | {custom_value} | {standard_value} |")
    return "\n".join(lines) + "\n"


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    camera_config_path = Path(args.camera_config).expanduser().resolve()
    detection_config_path = Path(args.detection_config).expanduser().resolve()
    tracking_config_path = Path(args.tracking_config).expanduser().resolve()
    worker_config_path = Path(args.worker_config).expanduser().resolve()
    persistence_config_path = Path(args.persistence_config).expanduser().resolve()
    evidence_config_path = Path(args.evidence_config).expanduser().resolve()
    anpr_config_path = Path(args.anpr_config).expanduser().resolve()
    florence_config_path = Path(args.florence_config).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    video_path, camera_code = resolve_experiment_video(
        camera_config_path=camera_config_path,
        camera_code=args.camera_code,
        video_path=args.video_path,
    )
    baseline = _build_runtime_baseline(
        camera_config_path=camera_config_path,
        detection_config_path=detection_config_path,
        tracking_config_path=tracking_config_path,
        worker_config_path=worker_config_path,
        persistence_config_path=persistence_config_path,
        evidence_config_path=evidence_config_path,
        video_path=video_path,
        camera_code=camera_code,
    )
    _json_dump(output_root / "baseline_runtime_configuration.json", baseline)
    base_run_id = f"STANDARD_BYTETRACK_COMPARE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    aspect_ratio_config = AspectRatioValidationConfig(
        enabled=True,
        classes={name: AspectRatioClassRange() for name in ("car", "bus", "truck", "motorcycle", "3wheeler")},
        action="report_only",
    )
    common_detection_overrides = {"iou_threshold": float(args.nms_iou)}
    custom_run_dir = output_root / "custom"
    standard_run_dir = output_root / "standard"
    custom_run_dir.mkdir(parents=True, exist_ok=True)
    standard_run_dir.mkdir(parents=True, exist_ok=True)
    custom_result = _run_single_configuration(
        run_config=ExperimentRunConfig(
            label="custom",
            confidence_threshold=float(args.confidence_threshold),
            match_threshold=float(args.match_threshold),
            minimum_consecutive_frames=int(args.minimum_consecutive_frames),
        ),
        camera_code=camera_code,
        camera_name=camera_code,
        video_path=video_path,
        start_frame=int(args.start_frame),
        end_frame=int(args.end_frame),
        detection_config_path=detection_config_path,
        tracking_config_path=tracking_config_path,
        persistence_config_path=persistence_config_path,
        evidence_config_path=evidence_config_path,
        run_dir=custom_run_dir,
        base_run_id=base_run_id,
        aspect_ratio_config=aspect_ratio_config,
        skip_anpr=bool(args.skip_anpr),
        anpr_config_path=anpr_config_path,
        florence_config_path=florence_config_path,
        tracking_overrides={"behavior_mode": "experimental_custom"},
        detection_overrides=common_detection_overrides,
    )
    standard_result = _run_single_configuration(
        run_config=ExperimentRunConfig(
            label="standard",
            confidence_threshold=float(args.confidence_threshold),
            match_threshold=float(args.match_threshold),
            minimum_consecutive_frames=int(args.minimum_consecutive_frames),
        ),
        camera_code=camera_code,
        camera_name=camera_code,
        video_path=video_path,
        start_frame=int(args.start_frame),
        end_frame=int(args.end_frame),
        detection_config_path=detection_config_path,
        tracking_config_path=tracking_config_path,
        persistence_config_path=persistence_config_path,
        evidence_config_path=evidence_config_path,
        run_dir=standard_run_dir,
        base_run_id=base_run_id,
        aspect_ratio_config=aspect_ratio_config,
        skip_anpr=bool(args.skip_anpr),
        anpr_config_path=anpr_config_path,
        florence_config_path=florence_config_path,
        tracking_overrides={"behavior_mode": "standard_bytetrack"},
        detection_overrides=common_detection_overrides,
    )
    payload = {
        "camera_code": camera_code,
        "video_path": str(video_path),
        "frame_range": {"start_frame": int(args.start_frame), "end_frame": int(args.end_frame)},
        "confidence_threshold": float(args.confidence_threshold),
        "nms_iou": float(args.nms_iou),
        "custom": _mode_summary(custom_result),
        "standard": _mode_summary(standard_result),
        "baseline_runtime_configuration": str((output_root / "baseline_runtime_configuration.json").resolve()),
    }
    _json_dump(output_root / "comparison.json", payload)
    (output_root / "comparison.md").write_text(_comparison_markdown(payload), encoding="utf-8")
    return payload


def main() -> None:
    print(json.dumps(run_comparison(parse_args()), indent=2))


if __name__ == "__main__":
    main()
