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
    ensure_unique_run_directory,
    resolve_experiment_video,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CAM_002 class-consistency experiment with fixed detector confidence.")
    parser.add_argument("--camera-config", default="tests\\td_case2\\multicamera_vehicle_tracking_pipeline\\config\\cameras.yaml")
    parser.add_argument("--detection-config", default="tests\\td_case2\\multicamera_vehicle_tracking_pipeline\\config\\detection.yaml")
    parser.add_argument("--tracking-config", default="tests\\td_case2\\multicamera_vehicle_tracking_pipeline\\config\\tracking.yaml")
    parser.add_argument("--worker-config", default="tests\\td_case2\\multicamera_vehicle_tracking_pipeline\\config\\workers.yaml")
    parser.add_argument("--persistence-config", default="tests\\td_case2\\multicamera_vehicle_tracking_pipeline\\config\\persistence.yaml")
    parser.add_argument("--evidence-config", default="tests\\td_case2\\multicamera_vehicle_tracking_pipeline\\config\\evidence.yaml")
    parser.add_argument("--anpr-config", default="tests\\td_case2\\multicamera_vehicle_tracking_pipeline\\config\\anpr.yaml")
    parser.add_argument("--florence-config", default="tests\\td_case2\\multicamera_vehicle_tracking_pipeline\\config\\florence.yaml")
    parser.add_argument("--camera-code", default="CAM_002")
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=599)
    parser.add_argument("--confidence-threshold", type=float, default=0.38)
    parser.add_argument("--output-root", default="debug_runs\\multicamera_vehicle_tracking_pipeline\\class_consistency_experiment")
    parser.add_argument("--skip-anpr", action="store_true")
    return parser.parse_args()


def _class_stabilization_override(
    *,
    minimum_observations: int,
    minimum_consistency_ratio: float,
    minimum_consecutive_winner_observations: int,
) -> dict[str, Any]:
    return {
        "class_stabilization": {
            "enabled": True,
            "strategy": "confidence_weighted_vote",
            "minimum_observations": minimum_observations,
            "minimum_consistency_ratio": minimum_consistency_ratio,
            "minimum_consecutive_winner_observations": minimum_consecutive_winner_observations,
            "recent_window_size": 5,
            "recent_conflict_minimum_ratio": 0.60,
            "recent_conflict_minimum_observations": 3,
        }
    }


def _comparison_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        summary = result["summary"]
        rows.append(
            {
                "label": result["run_report"]["run_label"],
                "minimum_observations": result["run_report"]["tracking_config"]["class_stabilization"]["minimum_observations"],
                "minimum_consistency_ratio": result["run_report"]["tracking_config"]["class_stabilization"]["minimum_consistency_ratio"],
                "minimum_consecutive_winner_observations": result["run_report"]["tracking_config"]["class_stabilization"]["minimum_consecutive_winner_observations"],
                "stable_classes": sum(1 for track in result["tracks"] if track.get("stable_class")),
                "unknown_or_unstable": sum(1 for track in result["tracks"] if not track.get("stable_class")),
                "mixed_car_truck_tracks": summary.get("tracks_with_car_and_truck", 0),
                "mixed_car_bus_tracks": summary.get("tracks_with_car_and_bus", 0),
                "strong_conflicts": sum(1 for track in result["tracks"] if track.get("strong_conflict_detected")),
                "split_recommended": sum(1 for track in result["tracks"] if track.get("split_recommended")),
                "split_executed": sum(1 for track in result["tracks"] if track.get("split_executed")),
                "artifact_path": result["run_report"]["artifact_paths"]["run_report"],
            }
        )
    return rows


def _comparison_markdown(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Test",
        "Min obs",
        "Min ratio",
        "Min consecutive",
        "Stable classes",
        "Unknown/unstable",
        "Mixed car/truck",
        "Mixed car/bus",
        "Strong conflicts",
        "Splits recommended",
        "Splits executed",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join([" --- " for _ in headers]) + "|"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    str(row["minimum_observations"]),
                    str(row["minimum_consistency_ratio"]),
                    str(row["minimum_consecutive_winner_observations"]),
                    str(row["stable_classes"]),
                    str(row["unknown_or_unstable"]),
                    str(row["mixed_car_truck_tracks"]),
                    str(row["mixed_car_bus_tracks"]),
                    str(row["strong_conflicts"]),
                    str(row["split_recommended"]),
                    str(row["split_executed"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    camera_config_path = Path(args.camera_config).expanduser().resolve()
    detection_config_path = Path(args.detection_config).expanduser().resolve()
    tracking_config_path = Path(args.tracking_config).expanduser().resolve()
    worker_config_path = Path(args.worker_config).expanduser().resolve()
    persistence_config_path = Path(args.persistence_config).expanduser().resolve()
    evidence_config_path = Path(args.evidence_config).expanduser().resolve()
    anpr_config_path = Path(args.anpr_config).expanduser().resolve()
    florence_config_path = Path(args.florence_config).expanduser().resolve()
    video_path, camera_code = resolve_experiment_video(camera_config_path=camera_config_path, camera_code=args.camera_code, video_path=args.video_path)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
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
    base_run_id = f"CLASS_CONSISTENCY_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    aspect_ratio_config = AspectRatioValidationConfig(
        enabled=True,
        classes={name: AspectRatioClassRange() for name in ("car", "bus", "truck", "motorcycle", "3wheeler")},
        action="report_only",
    )
    configs = [
        ("test_A", _class_stabilization_override(minimum_observations=3, minimum_consistency_ratio=0.60, minimum_consecutive_winner_observations=2)),
        ("test_B", _class_stabilization_override(minimum_observations=4, minimum_consistency_ratio=0.70, minimum_consecutive_winner_observations=3)),
        ("test_C", _class_stabilization_override(minimum_observations=5, minimum_consistency_ratio=0.80, minimum_consecutive_winner_observations=3)),
    ]
    results: list[dict[str, Any]] = []
    for label, overrides in configs:
        run_dir = ensure_unique_run_directory(output_root, label)
        results.append(
            _run_single_configuration(
                run_config=ExperimentRunConfig(
                    label=label,
                    confidence_threshold=float(args.confidence_threshold),
                    match_threshold=0.80,
                    minimum_consecutive_frames=1,
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
                run_dir=run_dir,
                base_run_id=base_run_id,
                aspect_ratio_config=aspect_ratio_config,
                skip_anpr=bool(args.skip_anpr),
                anpr_config_path=anpr_config_path,
                florence_config_path=florence_config_path,
                tracking_overrides=overrides,
            )
        )
    rows = _comparison_rows(results)
    payload = {
        "camera_code": camera_code,
        "video_path": str(video_path),
        "confidence_threshold": float(args.confidence_threshold),
        "frame_range": {"start_frame": int(args.start_frame), "end_frame": int(args.end_frame)},
        "baseline_runtime_configuration": str((output_root / "baseline_runtime_configuration.json").resolve()),
        "results": rows,
    }
    _json_dump(output_root / "comparison.json", payload)
    (output_root / "comparison.md").write_text(_comparison_markdown(rows), encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    print(json.dumps(run_experiment(args), indent=2))


if __name__ == "__main__":
    main()
