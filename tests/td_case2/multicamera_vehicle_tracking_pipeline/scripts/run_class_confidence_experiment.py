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
    parser = argparse.ArgumentParser(description="Run the CAM_002 class-specific confidence experiment.")
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
    parser.add_argument("--output-root", default="debug_runs\\multicamera_vehicle_tracking_pipeline\\class_confidence_experiment")
    parser.add_argument("--skip-anpr", action="store_true")
    return parser.parse_args()


def _class_threshold_override(*, default_threshold: float, classes: dict[str, float], inference_floor: float) -> dict[str, Any]:
    return {
        "confidence_threshold": float(inference_floor),
        "class_confidence_thresholds": {
            "enabled": True,
            "default": float(default_threshold),
            "classes": {key: float(value) for key, value in classes.items()},
        },
    }


def _per_class_rows(result: dict[str, Any]) -> dict[str, dict[str, int]]:
    raw = result["summary"].get("class_detection_counts", {})
    accepted = result["summary"].get("accepted_by_class", {})
    rejected = result["summary"].get("rejected_by_class", {})
    tracks = result["tracks"]
    rows: dict[str, dict[str, int]] = {}
    for class_name in ("car", "truck", "bus", "motorcycle", "3wheeler"):
        rows[class_name] = {
            "raw_detections": int(raw.get(class_name, 0)),
            "accepted": int(accepted.get(class_name, 0)),
            "rejected": int(rejected.get(class_name, 0)),
            "logical_tracks": sum(1 for track in tracks if (track.get("stable_class") or track.get("provisional_class") or track.get("class_name")) == class_name),
            "short_tracks": sum(
                1
                for track in tracks
                if (track.get("stable_class") or track.get("provisional_class") or track.get("class_name")) == class_name
                and int(track.get("observation_count") or 0) < 3
            ),
            "mixed_tracks": sum(
                1
                for track in tracks
                if class_name in (track.get("class_observation_counts") or {})
                and len(track.get("class_observation_counts") or {}) > 1
            ),
        }
    return rows


def _comparison_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Test | Inference floor | Motorcycle accepted | Car accepted | Truck accepted | Bus accepted | 3wheeler accepted | Mixed tracks | Executed splits | Unknown/unstable |",
        "|" + "|".join([" --- " for _ in range(10)]) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    str(row["inference_floor"]),
                    str(row["per_class"]["motorcycle"]["accepted"]),
                    str(row["per_class"]["car"]["accepted"]),
                    str(row["per_class"]["truck"]["accepted"]),
                    str(row["per_class"]["bus"]["accepted"]),
                    str(row["per_class"]["3wheeler"]["accepted"]),
                    str(row["mixed_tracks"]),
                    str(row["executed_logical_splits"]),
                    str(row["unknown_or_unstable"]),
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
    base_run_id = f"CLASS_CONFIDENCE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    aspect_ratio_config = AspectRatioValidationConfig(
        enabled=True,
        classes={name: AspectRatioClassRange() for name in ("car", "bus", "truck", "motorcycle", "3wheeler")},
        action="report_only",
    )
    configs = [
        ("baseline_global_038", None, 0.38),
        (
            "class_thresholds_A",
            _class_threshold_override(
                default_threshold=0.38,
                classes={"car": 0.38, "truck": 0.42, "bus": 0.42, "motorcycle": 0.32, "3wheeler": 0.36},
                inference_floor=0.32,
            ),
            0.32,
        ),
        (
            "class_thresholds_B",
            _class_threshold_override(
                default_threshold=0.38,
                classes={"car": 0.38, "truck": 0.50, "bus": 0.50, "motorcycle": 0.32, "3wheeler": 0.38},
                inference_floor=0.32,
            ),
            0.32,
        ),
        (
            "class_thresholds_C",
            _class_threshold_override(
                default_threshold=0.40,
                classes={"car": 0.40, "truck": 0.55, "bus": 0.55, "motorcycle": 0.30, "3wheeler": 0.40},
                inference_floor=0.30,
            ),
            0.30,
        ),
    ]
    results: list[dict[str, Any]] = []
    for label, detection_overrides, confidence_threshold in configs:
        run_dir = ensure_unique_run_directory(output_root, label)
        result = _run_single_configuration(
            run_config=ExperimentRunConfig(
                label=label,
                confidence_threshold=float(confidence_threshold),
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
            detection_overrides=detection_overrides,
        )
        results.append(result)
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "label": result["run_report"]["run_label"],
                "inference_floor": result["run_report"]["detection_config"]["effective_inference_confidence_floor"],
                "class_thresholds": result["run_report"]["detection_config"]["class_confidence_thresholds"],
                "per_class": _per_class_rows(result),
                "mixed_tracks": result["summary"].get("mixed_tracks", 0),
                "executed_logical_splits": sum(1 for track in result["tracks"] if track.get("split_executed")),
                "unknown_or_unstable": sum(1 for track in result["tracks"] if not track.get("stable_class")),
                "artifact_path": result["run_report"]["artifact_paths"]["run_report"],
            }
        )
    payload = {
        "camera_code": camera_code,
        "video_path": str(video_path),
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
