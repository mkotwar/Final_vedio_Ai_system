from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

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
    parser = argparse.ArgumentParser(description="Run the staged 19-run final-class parameter matrix on CAM_002.")
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
    parser.add_argument("--output-root", default="debug_runs\\multicamera_vehicle_tracking_pipeline\\final_class_parameter_matrix")
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


def _class_stabilization_override(
    *,
    minimum_observations: int,
    minimum_consistency_ratio: float,
    minimum_consecutive_winner_observations: int,
    minimum_winner_margin: float,
) -> dict[str, Any]:
    return {
        "class_stabilization": {
            "enabled": True,
            "strategy": "confidence_weighted_vote",
            "minimum_observations": minimum_observations,
            "minimum_consistency_ratio": minimum_consistency_ratio,
            "minimum_consecutive_winner_observations": minimum_consecutive_winner_observations,
            "minimum_winner_margin": minimum_winner_margin,
            "lock_after_observations": 5,
            "strong_conflict_min_observations": 3,
            "recent_window_size": 5,
            "recent_conflict_minimum_ratio": 0.60,
            "recent_conflict_minimum_observations": 3,
        }
    }


def _split_override(
    *,
    enabled: bool,
    minimum_consecutive_conflicting_observations: int = 3,
    minimum_average_conflict_confidence: float = 0.50,
    maximum_iou_for_split: float = 0.10,
    minimum_normalized_center_distance_for_split: float = 0.50,
    maximum_width_ratio_for_split: float = 2.50,
    maximum_height_ratio_for_split: float = 2.50,
) -> dict[str, Any]:
    return {
        "class_conflict_split": {
            "enabled": bool(enabled),
            "minimum_consecutive_conflicting_observations": int(minimum_consecutive_conflicting_observations),
            "minimum_conflict_confidence": 0.50,
            "minimum_average_conflict_confidence": float(minimum_average_conflict_confidence),
            "require_spatial_discontinuity": True,
            "maximum_iou_for_split": float(maximum_iou_for_split),
            "minimum_normalized_center_distance_for_split": float(minimum_normalized_center_distance_for_split),
            "maximum_width_ratio_for_split": float(maximum_width_ratio_for_split),
            "maximum_height_ratio_for_split": float(maximum_height_ratio_for_split),
        }
    }


def _final_class_name(track: dict[str, Any]) -> str | None:
    for key in ("stable_class", "provisional_class", "class_name"):
        value = str(track.get(key) or "").strip().lower()
        if value:
            return value
    return None


def _wrong_final_classes(tracks: Sequence[dict[str, Any]]) -> int:
    total = 0
    for track in tracks:
        final_class = _final_class_name(track)
        if not final_class or bool(track.get("final_class_blocked_due_to_mixed_identity")):
            continue
        if not bool(track.get("winner_agreement")) and final_class == str(track.get("score_winner") or "").strip().lower():
            total += 1
    return total


def _false_splits(tracks: Sequence[dict[str, Any]]) -> int:
    by_source: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for track in tracks:
        source_logical_id = track.get("source_logical_track_id")
        if source_logical_id is None:
            continue
        key = (str(track.get("linked_track_group_id") or track.get("track_uuid") or ""), int(source_logical_id))
        by_source.setdefault(key, []).append(track)
    false_splits = 0
    for children in by_source.values():
        for child in children:
            if not child.get("split_from_track_uuid"):
                continue
            child_class = _final_class_name(child)
            if not child_class:
                continue
            if child_class == str(child.get("stable_class_before_split") or "").strip().lower():
                false_splits += 1
    return false_splits


def _duplicate_observation_count(tracks: Sequence[dict[str, Any]]) -> int:
    seen: set[tuple[str, int, tuple[float, float, float, float]]] = set()
    duplicates = 0
    for track in tracks:
        track_uuid = str(track.get("track_uuid") or "")
        for item in track.get("raw_class_history", []):
            bbox = tuple(float(value) for value in item.get("bbox_xyxy", [])) if item.get("bbox_xyxy") else ()
            key = (track_uuid, int(item.get("frame_number") or 0), bbox)
            if key in seen:
                duplicates += 1
            else:
                seen.add(key)
    return duplicates


def _matrix_row(result: dict[str, Any], *, stage: str, parameters: dict[str, Any]) -> dict[str, Any]:
    tracks = result["tracks"]
    summary = result["summary"]
    executed_splits = sum(1 for track in tracks if track.get("split_executed"))
    mixed_identity_tracks = sum(1 for track in tracks if track.get("mixed_identity_detected"))
    return {
        "run": result["run_report"]["run_label"],
        "stage": stage,
        "car_conf": parameters.get("car_conf"),
        "truck_conf": parameters.get("truck_conf"),
        "bus_conf": parameters.get("bus_conf"),
        "motorcycle_conf": parameters.get("motorcycle_conf"),
        "consistency_ratio": parameters.get("consistency_ratio"),
        "conflict_observations": parameters.get("conflict_observations"),
        "split_iou": parameters.get("split_iou"),
        "nms_iou": parameters.get("nms_iou"),
        "mixed_car_truck": int(summary.get("tracks_with_car_and_truck", 0)),
        "executed_splits": executed_splits,
        "false_splits": _false_splits(tracks),
        "motorcycle_tracks": sum(1 for track in tracks if _final_class_name(track) == "motorcycle"),
        "blue_vehicle_tracks": sum(1 for track in tracks if _final_class_name(track) in {"bus", "truck"}),
        "wrong_final_classes": _wrong_final_classes(tracks),
        "count_score_disagreement": sum(1 for track in tracks if not bool(track.get("winner_agreement"))),
        "mixed_identity_tracks": mixed_identity_tracks,
        "split_recommendations": sum(1 for track in tracks if bool(track.get("split_recommended"))),
        "duplicate_overlapping_detections": int(summary.get("duplicate_overlapping_detections", 0)),
        "accepted_detections": int(summary.get("accepted_detections", 0)),
        "logical_tracks": int(summary.get("logical_tracks", 0)),
        "mixed_tracks": int(summary.get("mixed_tracks", 0)),
        "lost_observations": 0,
        "duplicate_observations": _duplicate_observation_count(tracks),
        "artifact_path": result["run_report"]["artifact_paths"]["run_report"],
        "selected": False,
    }


def _rank_detection(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["wrong_final_classes"]),
        int(row["mixed_car_truck"]),
        int(row["count_score_disagreement"]),
        -int(row["motorcycle_tracks"]),
        int(row["mixed_tracks"]),
        int(row["duplicate_overlapping_detections"]),
    )


def _rank_consistency(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["wrong_final_classes"]),
        int(row["mixed_identity_tracks"]),
        int(row["mixed_car_truck"]),
        int(row["count_score_disagreement"]),
        int(row["mixed_tracks"]),
    )


def _rank_split(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["wrong_final_classes"]),
        int(row["mixed_car_truck"]),
        int(row["false_splits"]),
        int(row["lost_observations"]),
        int(row["duplicate_observations"]),
        -int(row["executed_splits"]),
    )


def _rank_nms(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["duplicate_overlapping_detections"]),
        int(row["mixed_car_truck"]),
        int(row["wrong_final_classes"]),
        -int(row["motorcycle_tracks"]),
        int(row["mixed_tracks"]),
    )


def _mark_selected(rows: list[dict[str, Any]], selected_run: str) -> None:
    for row in rows:
        if row["run"] == selected_run:
            row["selected"] = True


def _comparison_markdown(rows: Sequence[dict[str, Any]]) -> str:
    headers = [
        "Run",
        "Car conf",
        "Truck conf",
        "Bus conf",
        "Motorcycle conf",
        "Consistency ratio",
        "Conflict observations",
        "Split IoU",
        "NMS IoU",
        "Mixed CAR/TRUCK",
        "Executed splits",
        "False splits",
        "Motorcycle tracks",
        "Blue vehicle tracks",
        "Wrong final classes",
        "Selected",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join([" --- " for _ in headers]) + "|"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["run"]),
                    str(row["car_conf"] or ""),
                    str(row["truck_conf"] or ""),
                    str(row["bus_conf"] or ""),
                    str(row["motorcycle_conf"] or ""),
                    str(row["consistency_ratio"] or ""),
                    str(row["conflict_observations"] or ""),
                    str(row["split_iou"] or ""),
                    str(row["nms_iou"] or ""),
                    str(row["mixed_car_truck"]),
                    str(row["executed_splits"]),
                    str(row["false_splits"]),
                    str(row["motorcycle_tracks"]),
                    str(row["blue_vehicle_tracks"]),
                    str(row["wrong_final_classes"]),
                    "yes" if row["selected"] else "",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _run_matrix_entry(
    *,
    label: str,
    output_root: Path,
    base_run_id: str,
    camera_code: str,
    video_path: Path,
    start_frame: int,
    end_frame: int,
    detection_config_path: Path,
    tracking_config_path: Path,
    persistence_config_path: Path,
    evidence_config_path: Path,
    anpr_config_path: Path,
    florence_config_path: Path,
    aspect_ratio_config: AspectRatioValidationConfig,
    skip_anpr: bool,
    detection_overrides: dict[str, Any] | None,
    tracking_overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    run_dir = ensure_unique_run_directory(output_root, label)
    return _run_single_configuration(
        run_config=ExperimentRunConfig(
            label=label,
            confidence_threshold=0.38,
            match_threshold=0.80,
            minimum_consecutive_frames=1,
        ),
        camera_code=camera_code,
        camera_name=camera_code,
        video_path=video_path,
        start_frame=start_frame,
        end_frame=end_frame,
        detection_config_path=detection_config_path,
        tracking_config_path=tracking_config_path,
        persistence_config_path=persistence_config_path,
        evidence_config_path=evidence_config_path,
        run_dir=run_dir,
        base_run_id=base_run_id,
        aspect_ratio_config=aspect_ratio_config,
        skip_anpr=skip_anpr,
        anpr_config_path=anpr_config_path,
        florence_config_path=florence_config_path,
        detection_overrides=detection_overrides,
        tracking_overrides=tracking_overrides,
    )


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
    base_run_id = f"FINAL_CLASS_MATRIX_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    aspect_ratio_config = AspectRatioValidationConfig(
        enabled=True,
        classes={name: AspectRatioClassRange() for name in ("car", "bus", "truck", "motorcycle", "3wheeler")},
        action="report_only",
    )
    class_consistency_b = _class_stabilization_override(
        minimum_observations=4,
        minimum_consistency_ratio=0.70,
        minimum_consecutive_winner_observations=3,
        minimum_winner_margin=0.20,
    )

    stage_rows: list[dict[str, Any]] = []

    detection_configs = [
        ("D0", None, class_consistency_b, {"car_conf": 0.38, "truck_conf": 0.38, "bus_conf": 0.38, "motorcycle_conf": 0.38, "nms_iou": 0.45}),
        ("D1", _class_threshold_override(default_threshold=0.38, classes={"car": 0.38, "truck": 0.65, "bus": 0.65, "motorcycle": 0.32, "3wheeler": 0.38}, inference_floor=0.32), class_consistency_b, {"car_conf": 0.38, "truck_conf": 0.65, "bus_conf": 0.65, "motorcycle_conf": 0.32, "nms_iou": 0.45}),
        ("D2", _class_threshold_override(default_threshold=0.40, classes={"car": 0.40, "truck": 0.75, "bus": 0.75, "motorcycle": 0.32, "3wheeler": 0.40}, inference_floor=0.32), class_consistency_b, {"car_conf": 0.40, "truck_conf": 0.75, "bus_conf": 0.75, "motorcycle_conf": 0.32, "nms_iou": 0.45}),
        ("D3", _class_threshold_override(default_threshold=0.40, classes={"car": 0.40, "truck": 0.80, "bus": 0.80, "motorcycle": 0.30, "3wheeler": 0.40}, inference_floor=0.30), class_consistency_b, {"car_conf": 0.40, "truck_conf": 0.80, "bus_conf": 0.80, "motorcycle_conf": 0.30, "nms_iou": 0.45}),
        ("D4", _class_threshold_override(default_threshold=0.36, classes={"car": 0.36, "truck": 0.70, "bus": 0.70, "motorcycle": 0.30, "3wheeler": 0.36}, inference_floor=0.30), class_consistency_b, {"car_conf": 0.36, "truck_conf": 0.70, "bus_conf": 0.70, "motorcycle_conf": 0.30, "nms_iou": 0.45}),
    ]
    detection_results: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]] = []
    for label, detection_override, tracking_override, parameters in detection_configs:
        result = _run_matrix_entry(
            label=label,
            output_root=output_root,
            base_run_id=base_run_id,
            camera_code=camera_code,
            video_path=video_path,
            start_frame=int(args.start_frame),
            end_frame=int(args.end_frame),
            detection_config_path=detection_config_path,
            tracking_config_path=tracking_config_path,
            persistence_config_path=persistence_config_path,
            evidence_config_path=evidence_config_path,
            anpr_config_path=anpr_config_path,
            florence_config_path=florence_config_path,
            aspect_ratio_config=aspect_ratio_config,
            skip_anpr=bool(args.skip_anpr),
            detection_overrides=detection_override,
            tracking_overrides=tracking_override,
        )
        row = _matrix_row(result, stage="detection", parameters=parameters)
        stage_rows.append(row)
        detection_results.append((result, row, detection_override))
    selected_detection_result, selected_detection_row, selected_detection_override = min(detection_results, key=lambda item: _rank_detection(item[1]))
    _mark_selected(stage_rows, selected_detection_row["run"])

    consistency_configs = [
        ("C0", _class_stabilization_override(minimum_observations=2, minimum_consistency_ratio=0.00, minimum_consecutive_winner_observations=1, minimum_winner_margin=0.20), {"consistency_ratio": 0.00}),
        ("C1", _class_stabilization_override(minimum_observations=3, minimum_consistency_ratio=0.60, minimum_consecutive_winner_observations=2, minimum_winner_margin=0.20), {"consistency_ratio": 0.60}),
        ("C2", _class_stabilization_override(minimum_observations=4, minimum_consistency_ratio=0.70, minimum_consecutive_winner_observations=3, minimum_winner_margin=0.20), {"consistency_ratio": 0.70}),
        ("C3", _class_stabilization_override(minimum_observations=5, minimum_consistency_ratio=0.80, minimum_consecutive_winner_observations=3, minimum_winner_margin=0.25), {"consistency_ratio": 0.80}),
    ]
    consistency_results: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for label, tracking_override, parameters in consistency_configs:
        result = _run_matrix_entry(
            label=label,
            output_root=output_root,
            base_run_id=base_run_id,
            camera_code=camera_code,
            video_path=video_path,
            start_frame=int(args.start_frame),
            end_frame=int(args.end_frame),
            detection_config_path=detection_config_path,
            tracking_config_path=tracking_config_path,
            persistence_config_path=persistence_config_path,
            evidence_config_path=evidence_config_path,
            anpr_config_path=anpr_config_path,
            florence_config_path=florence_config_path,
            aspect_ratio_config=aspect_ratio_config,
            skip_anpr=bool(args.skip_anpr),
            detection_overrides=selected_detection_override,
            tracking_overrides=tracking_override,
        )
        row = _matrix_row(
            result,
            stage="consistency",
            parameters={
                **parameters,
                "car_conf": selected_detection_row["car_conf"],
                "truck_conf": selected_detection_row["truck_conf"],
                "bus_conf": selected_detection_row["bus_conf"],
                "motorcycle_conf": selected_detection_row["motorcycle_conf"],
                "nms_iou": 0.45,
            },
        )
        stage_rows.append(row)
        consistency_results.append((result, row, tracking_override))
    selected_consistency_result, selected_consistency_row, selected_consistency_override = min(consistency_results, key=lambda item: _rank_consistency(item[1]))
    _mark_selected(stage_rows, selected_consistency_row["run"])

    split_configs = [
        ("S0", _split_override(enabled=False), {"conflict_observations": 0, "split_iou": None}),
        ("S1", _split_override(enabled=True, minimum_consecutive_conflicting_observations=3, minimum_average_conflict_confidence=0.50, maximum_iou_for_split=0.10, minimum_normalized_center_distance_for_split=0.50, maximum_width_ratio_for_split=2.50, maximum_height_ratio_for_split=2.50), {"conflict_observations": 3, "split_iou": 0.10}),
        ("S2", _split_override(enabled=True, minimum_consecutive_conflicting_observations=3, minimum_average_conflict_confidence=0.50, maximum_iou_for_split=0.20, minimum_normalized_center_distance_for_split=0.35, maximum_width_ratio_for_split=2.00, maximum_height_ratio_for_split=2.00), {"conflict_observations": 3, "split_iou": 0.20}),
        ("S3", _split_override(enabled=True, minimum_consecutive_conflicting_observations=2, minimum_average_conflict_confidence=0.60, maximum_iou_for_split=0.15, minimum_normalized_center_distance_for_split=0.40, maximum_width_ratio_for_split=2.25, maximum_height_ratio_for_split=2.25), {"conflict_observations": 2, "split_iou": 0.15}),
        ("S4", _split_override(enabled=True, minimum_consecutive_conflicting_observations=3, minimum_average_conflict_confidence=0.75, maximum_iou_for_split=0.20, minimum_normalized_center_distance_for_split=0.35, maximum_width_ratio_for_split=2.00, maximum_height_ratio_for_split=2.00), {"conflict_observations": 3, "split_iou": 0.20}),
    ]
    split_results: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for label, split_override, parameters in split_configs:
        tracking_override = dict(selected_consistency_override)
        tracking_override.update(split_override)
        result = _run_matrix_entry(
            label=label,
            output_root=output_root,
            base_run_id=base_run_id,
            camera_code=camera_code,
            video_path=video_path,
            start_frame=int(args.start_frame),
            end_frame=int(args.end_frame),
            detection_config_path=detection_config_path,
            tracking_config_path=tracking_config_path,
            persistence_config_path=persistence_config_path,
            evidence_config_path=evidence_config_path,
            anpr_config_path=anpr_config_path,
            florence_config_path=florence_config_path,
            aspect_ratio_config=aspect_ratio_config,
            skip_anpr=bool(args.skip_anpr),
            detection_overrides=selected_detection_override,
            tracking_overrides=tracking_override,
        )
        row = _matrix_row(
            result,
            stage="split",
            parameters={
                **parameters,
                "car_conf": selected_detection_row["car_conf"],
                "truck_conf": selected_detection_row["truck_conf"],
                "bus_conf": selected_detection_row["bus_conf"],
                "motorcycle_conf": selected_detection_row["motorcycle_conf"],
                "consistency_ratio": selected_consistency_row["consistency_ratio"],
                "nms_iou": 0.45,
            },
        )
        stage_rows.append(row)
        split_results.append((result, row, split_override))
    selected_split_result, selected_split_row, selected_split_override = min(split_results, key=lambda item: _rank_split(item[1]))
    _mark_selected(stage_rows, selected_split_row["run"])

    nms_configs = [
        ("N0", 0.35),
        ("N1", 0.40),
        ("N2", 0.45),
        ("N3", 0.50),
    ]
    nms_results: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    for label, nms_iou in nms_configs:
        detection_override = dict(selected_detection_override or {})
        detection_override["iou_threshold"] = float(nms_iou)
        tracking_override = dict(selected_consistency_override)
        tracking_override.update(selected_split_override)
        result = _run_matrix_entry(
            label=label,
            output_root=output_root,
            base_run_id=base_run_id,
            camera_code=camera_code,
            video_path=video_path,
            start_frame=int(args.start_frame),
            end_frame=int(args.end_frame),
            detection_config_path=detection_config_path,
            tracking_config_path=tracking_config_path,
            persistence_config_path=persistence_config_path,
            evidence_config_path=evidence_config_path,
            anpr_config_path=anpr_config_path,
            florence_config_path=florence_config_path,
            aspect_ratio_config=aspect_ratio_config,
            skip_anpr=bool(args.skip_anpr),
            detection_overrides=detection_override,
            tracking_overrides=tracking_override,
        )
        row = _matrix_row(
            result,
            stage="nms",
            parameters={
                "car_conf": selected_detection_row["car_conf"],
                "truck_conf": selected_detection_row["truck_conf"],
                "bus_conf": selected_detection_row["bus_conf"],
                "motorcycle_conf": selected_detection_row["motorcycle_conf"],
                "consistency_ratio": selected_consistency_row["consistency_ratio"],
                "conflict_observations": selected_split_row["conflict_observations"],
                "split_iou": selected_split_row["split_iou"],
                "nms_iou": nms_iou,
            },
        )
        stage_rows.append(row)
        nms_results.append((result, row, nms_iou))
    selected_nms_result, selected_nms_row, selected_nms_iou = min(nms_results, key=lambda item: _rank_nms(item[1]))
    _mark_selected(stage_rows, selected_nms_row["run"])

    final_detection_override = dict(selected_detection_override or {})
    final_detection_override["iou_threshold"] = float(selected_nms_iou)
    final_tracking_override = dict(selected_consistency_override)
    final_tracking_override.update(selected_split_override)
    final_result = _run_matrix_entry(
        label="final_selected",
        output_root=output_root,
        base_run_id=base_run_id,
        camera_code=camera_code,
        video_path=video_path,
        start_frame=int(args.start_frame),
        end_frame=int(args.end_frame),
        detection_config_path=detection_config_path,
        tracking_config_path=tracking_config_path,
        persistence_config_path=persistence_config_path,
        evidence_config_path=evidence_config_path,
        anpr_config_path=anpr_config_path,
        florence_config_path=florence_config_path,
        aspect_ratio_config=aspect_ratio_config,
        skip_anpr=bool(args.skip_anpr),
        detection_overrides=final_detection_override,
        tracking_overrides=final_tracking_override,
    )
    final_row = _matrix_row(
        final_result,
        stage="final",
        parameters={
            "car_conf": selected_detection_row["car_conf"],
            "truck_conf": selected_detection_row["truck_conf"],
            "bus_conf": selected_detection_row["bus_conf"],
            "motorcycle_conf": selected_detection_row["motorcycle_conf"],
            "consistency_ratio": selected_consistency_row["consistency_ratio"],
            "conflict_observations": selected_split_row["conflict_observations"],
            "split_iou": selected_split_row["split_iou"],
            "nms_iou": selected_nms_iou,
        },
    )
    final_row["selected"] = True
    stage_rows.append(final_row)

    payload = {
        "camera_code": camera_code,
        "video_path": str(video_path),
        "frame_range": {"start_frame": int(args.start_frame), "end_frame": int(args.end_frame)},
        "baseline_runtime_configuration": str((output_root / "baseline_runtime_configuration.json").resolve()),
        "selected_detection_run": selected_detection_row["run"],
        "selected_consistency_run": selected_consistency_row["run"],
        "selected_split_run": selected_split_row["run"],
        "selected_nms_run": selected_nms_row["run"],
        "rows": stage_rows,
    }
    _json_dump(output_root / "comparison.json", payload)
    (output_root / "comparison.md").write_text(_comparison_markdown(stage_rows), encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    print(json.dumps(run_experiment(args), indent=2))


if __name__ == "__main__":
    main()
