from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2

if __package__ in {None, ""}:
    case_root = Path(__file__).resolve().parents[2]
    if str(case_root) not in sys.path:
        sys.path.insert(0, str(case_root))
    from experiments.dynamic_yolo_tracking.dynamic_fps_controller import (
        DynamicFpsConfig,
        DynamicFpsController,
        ControllerObservation,
        STATE_BURST,
        STATE_IDLE,
        STATE_LOW,
        STATE_NORMAL,
    )
    from experiments.dynamic_yolo_tracking.dynamic_frame_decoder import (
        frame_interval_for_fps,
        next_frame_index,
        read_video_metadata,
        validate_chronological_frame_records,
    )
    from experiments.dynamic_yolo_tracking.motion_gate import CheapMotionGate, MotionGateConfig
    from experiments.dynamic_yolo_tracking.tracking_metrics import (
        build_step05_compatible_tracks,
        compare_mode_summaries,
        summarize_tracks,
        validate_step05_compatibility,
    )
    from experiments.dynamic_yolo_tracking.yolo_tracking_pipeline import (
        build_preview_video,
        class_group,
        detect_on_selected_frames,
        merge_track_fragments,
        resolve_pipeline_config,
        run_variable_gap_bytetrack,
        YoloFrameProcessor,
    )
else:
    from .dynamic_fps_controller import (
        DynamicFpsConfig,
        DynamicFpsController,
        ControllerObservation,
        STATE_BURST,
        STATE_IDLE,
        STATE_LOW,
        STATE_NORMAL,
    )
    from .dynamic_frame_decoder import frame_interval_for_fps, next_frame_index, read_video_metadata, validate_chronological_frame_records
    from .motion_gate import CheapMotionGate, MotionGateConfig
    from .tracking_metrics import build_step05_compatible_tracks, compare_mode_summaries, summarize_tracks, validate_step05_compatibility
    from .yolo_tracking_pipeline import build_preview_video, class_group, detect_on_selected_frames, merge_track_fragments, resolve_pipeline_config, run_variable_gap_bytetrack, YoloFrameProcessor

from stage_checks import read_json, write_json


ENV_DYNAMIC_VIDEO_PATH = "TD_CASE2_DYNAMIC_VIDEO_PATH"
ENV_DYNAMIC_OUTPUT_ROOT = "TD_CASE2_DYNAMIC_OUTPUT_ROOT"
ENV_DYNAMIC_TRACK_BUFFER_SECONDS = "TD_CASE2_DYNAMIC_TRACK_BUFFER_SECONDS"
ENV_DYNAMIC_TRACK_HIGH_CONFIDENCE = "TD_CASE2_DYNAMIC_TRACK_HIGH_CONFIDENCE"
ENV_DYNAMIC_TRACK_LOW_CONFIDENCE = "TD_CASE2_DYNAMIC_TRACK_LOW_CONFIDENCE"
ENV_DYNAMIC_TRACK_MATCH_THRESHOLD = "TD_CASE2_DYNAMIC_TRACK_MATCH_THRESHOLD"
ENV_DYNAMIC_TRACK_MIN_LENGTH = "TD_CASE2_DYNAMIC_TRACK_MIN_LENGTH"
ENV_DYNAMIC_SAVE_ANNOTATED = "TD_CASE2_DYNAMIC_SAVE_ANNOTATED"
ENV_DYNAMIC_SAVE_CROPS = "TD_CASE2_DYNAMIC_SAVE_CROPS"
ENV_DYNAMIC_EMPTY_HEARTBEAT_SECONDS = "TD_CASE2_EXP_EMPTY_HEARTBEAT_SECONDS"
ENV_DYNAMIC_MERGE_MAX_CANDIDATES_PER_TRACK = "TD_CASE2_EXP_MERGE_MAX_CANDIDATES_PER_TRACK"


def _read_bool(env_name: str, default_value: bool) -> bool:
    raw_value = os.environ.get(env_name, "").strip().lower()
    if not raw_value:
        return default_value
    return raw_value in {"1", "true", "yes", "on", "y"}


def _read_float(env_name: str, default_value: float) -> float:
    return float(os.environ.get(env_name, str(default_value)).strip())


def _read_int(env_name: str, default_value: int) -> int:
    return int(os.environ.get(env_name, str(default_value)).strip())


def _output_root(repo_root: Path) -> Path:
    raw_value = os.environ.get(ENV_DYNAMIC_OUTPUT_ROOT, "").strip()
    if raw_value:
        path = Path(raw_value).expanduser()
        return path if path.is_absolute() else (repo_root / path).resolve()
    return repo_root / "debug_runs"


def _create_run_dir(repo_root: Path, video_path: Path) -> tuple[Path, Path]:
    output_root = _output_root(repo_root)
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{video_path.stem}_{timestamp}"
    experiment_dir = run_dir / "dynamic_yolo_tracking_experiment"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, experiment_dir


def _write_video_info(run_dir: Path, metadata: Any) -> None:
    write_json(
        run_dir / "01_video_info.json",
        {
            "input_video_path": str(metadata.video_path),
            "video_name": metadata.video_path.name,
            "fps": round(metadata.fps, 6),
            "frame_count": metadata.frame_count,
            "duration_seconds": round(metadata.duration_seconds, 6),
            "width": metadata.width,
            "height": metadata.height,
            "status": "success",
        },
    )


def _save_frame(image: Any, image_path: Path) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_path), image):
        raise RuntimeError(f"Failed to write decoded frame: {image_path}")


def _selection_record(
    *,
    experiment_dir: Path,
    frame_idx: int,
    timestamp_seconds: float,
    tracking_state: str,
    target_fps: float,
    selection_reason: list[str],
    previous_detection_count: int,
    active_track_count: int,
) -> dict[str, Any]:
    frame_id = f"dynamic_frame_{frame_idx:06d}"
    image_name = f"{frame_id}.jpg"
    return {
        "frame_id": frame_id,
        "frame_idx": frame_idx,
        "timestamp_seconds": round(timestamp_seconds, 6),
        "timestamp_text": f"{timestamp_seconds:.3f}s",
        "image_path": str(Path("decoded_frames") / image_name).replace("\\", "/"),
        "tracking_state": tracking_state,
        "target_fps": round(target_fps, 3),
        "selection_reason": selection_reason,
        "previous_detection_count": int(previous_detection_count),
        "active_track_count": int(active_track_count),
    }


def _build_fixed_schedule(*, metadata: Any, experiment_dir: Path, target_fps: float, state_name: str) -> list[dict[str, Any]]:
    capture = cv2.VideoCapture(str(metadata.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {metadata.video_path}")
    selected_indices: set[int] = set()
    records: list[dict[str, Any]] = []
    target_index = 0
    step = frame_interval_for_fps(metadata.fps, target_fps)
    frame_idx = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_idx == target_index and frame_idx not in selected_indices:
            timestamp_seconds = frame_idx / metadata.fps if metadata.fps > 0 else 0.0
            record = _selection_record(
                experiment_dir=experiment_dir,
                frame_idx=frame_idx,
                timestamp_seconds=timestamp_seconds,
                tracking_state=state_name,
                target_fps=target_fps,
                selection_reason=[f"fixed_{target_fps:.1f}_fps"],
                previous_detection_count=0,
                active_track_count=0,
            )
            _save_frame(frame, experiment_dir / record["image_path"])
            records.append(record)
            selected_indices.add(frame_idx)
            target_index += step
        frame_idx += 1
    capture.release()
    return records


def _pairwise_distances(boxes: list[list[float]], diagonal: float) -> list[float]:
    distances: list[float] = []
    for left_index in range(len(boxes)):
        for right_index in range(left_index + 1, len(boxes)):
            ax = (boxes[left_index][0] + boxes[left_index][2]) / 2.0
            ay = (boxes[left_index][1] + boxes[left_index][3]) / 2.0
            bx = (boxes[right_index][0] + boxes[right_index][2]) / 2.0
            by = (boxes[right_index][1] + boxes[right_index][3]) / 2.0
            distance = (((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5) / diagonal if diagonal > 0 else 0.0
            distances.append(distance)
    return distances


def _build_observation(
    *,
    detection_rows: list[dict[str, Any]],
    previous_rows: list[dict[str, Any]],
    timestamp_seconds: float,
    previous_detection_count: int,
    active_track_count: int,
    image_width: int,
    image_height: int,
) -> ControllerObservation:
    image_diagonal = (image_width**2 + image_height**2) ** 0.5
    vehicle_rows = [row for row in detection_rows if class_group(str(row["class_name"])) == "vehicle"]
    person_rows = [row for row in detection_rows if class_group(str(row["class_name"])) == "person"]
    previous_by_class = defaultdict(list)
    for row in previous_rows:
        previous_by_class[str(row["class_name"])].append(row)

    displacements: list[float] = []
    area_changes: list[float] = []
    confidences = [float(row["confidence"]) for row in detection_rows]
    for row in vehicle_rows:
        previous_candidates = previous_by_class.get(str(row["class_name"]), [])
        if not previous_candidates:
            continue
        cx = (row["bbox_xyxy"][0] + row["bbox_xyxy"][2]) / 2.0
        cy = (row["bbox_xyxy"][1] + row["bbox_xyxy"][3]) / 2.0
        previous = min(
            previous_candidates,
            key=lambda candidate: (
                (
                    ((candidate["bbox_xyxy"][0] + candidate["bbox_xyxy"][2]) / 2.0) - cx
                ) ** 2
                + (
                    ((candidate["bbox_xyxy"][1] + candidate["bbox_xyxy"][3]) / 2.0) - cy
                ) ** 2
            ),
        )
        px = (previous["bbox_xyxy"][0] + previous["bbox_xyxy"][2]) / 2.0
        py = (previous["bbox_xyxy"][1] + previous["bbox_xyxy"][3]) / 2.0
        displacements.append((((px - cx) ** 2 + (py - cy) ** 2) ** 0.5) / image_diagonal if image_diagonal > 0 else 0.0)
        prev_area = max(0.0, previous["bbox_xyxy"][2] - previous["bbox_xyxy"][0]) * max(0.0, previous["bbox_xyxy"][3] - previous["bbox_xyxy"][1])
        curr_area = max(0.0, row["bbox_xyxy"][2] - row["bbox_xyxy"][0]) * max(0.0, row["bbox_xyxy"][3] - row["bbox_xyxy"][1])
        if prev_area > 0 and curr_area > 0:
            area_changes.append(abs(curr_area - prev_area) / max(prev_area, curr_area))

    vehicle_distances = _pairwise_distances([row["bbox_xyxy"] for row in vehicle_rows], image_diagonal)
    vehicle_person_distances: list[float] = []
    for vehicle in vehicle_rows:
        for person in person_rows:
            vehicle_person_distances.extend(_pairwise_distances([vehicle["bbox_xyxy"], person["bbox_xyxy"]], image_diagonal))

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return ControllerObservation(
        timestamp_seconds=timestamp_seconds,
        detection_count=len(detection_rows),
        active_track_count=active_track_count,
        avg_center_displacement=(sum(displacements) / len(displacements)) if displacements else 0.0,
        avg_bbox_area_change=(sum(area_changes) / len(area_changes)) if area_changes else 0.0,
        avg_direction_change=(sum(displacements) / len(displacements)) if displacements else 0.0,
        max_track_speed=max(displacements) if displacements else 0.0,
        stationary_track_ratio=1.0 if displacements and max(displacements) <= 0.015 else 0.0,
        new_track_count=max(0, len(detection_rows) - previous_detection_count),
        lost_track_count=max(0, previous_detection_count - len(detection_rows)),
        vehicle_vehicle_proximity=min(vehicle_distances) if vehicle_distances else None,
        vehicle_person_proximity=min(vehicle_person_distances) if vehicle_person_distances else None,
        scene_motion_score=(sum(area_changes) / len(area_changes)) if area_changes else 0.0,
        consecutive_empty_detections=0 if detection_rows else max(1, previous_detection_count + 1),
        tracker_confidence_instability=(1.0 - avg_conf) if confidences else 0.0,
    )


def _build_yolo_config() -> Any:
    return resolve_pipeline_config(
        track_buffer_seconds=_read_float(ENV_DYNAMIC_TRACK_BUFFER_SECONDS, 2.0),
        high_confidence=_read_float(ENV_DYNAMIC_TRACK_HIGH_CONFIDENCE, 0.25),
        low_confidence=_read_float(ENV_DYNAMIC_TRACK_LOW_CONFIDENCE, 0.10),
        match_threshold=_read_float(ENV_DYNAMIC_TRACK_MATCH_THRESHOLD, 0.80),
        min_track_length=_read_int(ENV_DYNAMIC_TRACK_MIN_LENGTH, 2),
        save_annotated=_read_bool(ENV_DYNAMIC_SAVE_ANNOTATED, True),
        save_crops=_read_bool(ENV_DYNAMIC_SAVE_CROPS, True),
    )


def _build_tracking_runtime_config() -> dict[str, Any]:
    return {
        "track_buffer_seconds": _read_float(ENV_DYNAMIC_TRACK_BUFFER_SECONDS, 2.0),
        "high_confidence": _read_float(ENV_DYNAMIC_TRACK_HIGH_CONFIDENCE, 0.25),
        "low_confidence": _read_float(ENV_DYNAMIC_TRACK_LOW_CONFIDENCE, 0.10),
        "match_threshold": _read_float(ENV_DYNAMIC_TRACK_MATCH_THRESHOLD, 0.80),
        "min_track_length": _read_int(ENV_DYNAMIC_TRACK_MIN_LENGTH, 2),
        "merge_max_candidates_per_track": _read_int(ENV_DYNAMIC_MERGE_MAX_CANDIDATES_PER_TRACK, 5),
    }


def _dedupe_reason_items(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _single_pass_report_seed() -> dict[str, Any]:
    return {
        "video_frames_decoded": 0,
        "unique_frames_decoded": 0,
        "duplicate_decoded_frames": 0,
        "frames_sent_to_yolo": 0,
        "duplicate_yolo_frames": 0,
        "idle_frames_skipped": 0,
        "idle_frames_saved": 0,
        "motion_triggered_yolo_calls": 0,
        "heartbeat_yolo_calls": 0,
        "low_state_yolo_calls": 0,
        "normal_state_yolo_calls": 0,
        "burst_state_yolo_calls": 0,
        "yolo_processing_time_seconds": 0.0,
        "tracking_time_seconds": 0.0,
        "raw_track_count": 0,
        "merged_track_count": 0,
        "single_frame_track_count": 0,
        "fragmented_track_count": 0,
        "good_track_count": 0,
    }


def _run_single_pass_dynamic(
    *,
    metadata: Any,
    experiment_dir: Path,
    controller: DynamicFpsController,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    config = _build_yolo_config()
    detector = YoloFrameProcessor(
        experiment_dir=experiment_dir,
        config=config,
        annotated_dir_name="annotated_frames",
        crop_dir_name="dynamic_object_crops",
    )
    heartbeat_seconds = max(0.0, _read_float(ENV_DYNAMIC_EMPTY_HEARTBEAT_SECONDS, 3.0))
    motion_gate = CheapMotionGate(MotionGateConfig())
    capture = cv2.VideoCapture(str(metadata.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {metadata.video_path}")

    selected_indices: set[int] = set()
    yolo_frame_indices: set[int] = set()
    decoded_indices: set[int] = set()
    frame_records: list[dict[str, Any]] = []
    yolo_frame_payloads: list[dict[str, Any]] = []
    detection_rows: list[dict[str, Any]] = []
    transition_log: list[dict[str, Any]] = []
    report = _single_pass_report_seed()
    class_counts: dict[str, int] = {}
    state_counts: dict[str, int] = {}
    previous_rows: list[dict[str, Any]] = []
    previous_detection_count = 0
    active_track_count = 0
    current_state = controller.current_state
    current_target_fps = controller.config.target_fps_for_state(current_state)
    next_yolo_due_timestamp = 0.0
    last_yolo_timestamp: float | None = None
    last_detection_timestamp: float | None = None

    frame_idx = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        report["video_frames_decoded"] += 1
        if frame_idx in decoded_indices:
            report["duplicate_decoded_frames"] += 1
        else:
            decoded_indices.add(frame_idx)

        timestamp_seconds = frame_idx / metadata.fps if metadata.fps > 0 else 0.0
        motion = motion_gate.evaluate(frame)
        recent_detection = last_detection_timestamp is not None and (timestamp_seconds - last_detection_timestamp) <= max(1.5, 2.0 / max(current_target_fps, 0.1))
        heartbeat_due = heartbeat_seconds > 0.0 and (last_yolo_timestamp is None or (timestamp_seconds - last_yolo_timestamp) >= heartbeat_seconds)
        scheduled_due = current_state != STATE_IDLE and (last_yolo_timestamp is None or timestamp_seconds >= next_yolo_due_timestamp)
        motion_trigger = bool(motion.meaningful_motion and (current_state == STATE_IDLE or not recent_detection))
        should_run_yolo = motion_trigger or scheduled_due or (current_state == STATE_IDLE and heartbeat_due)
        if not should_run_yolo:
            report["idle_frames_skipped"] += 1
            frame_idx += 1
            continue

        if frame_idx in yolo_frame_indices:
            report["duplicate_yolo_frames"] += 1
            frame_idx += 1
            continue

        selection_reason: list[str] = []
        if motion_trigger:
            selection_reason.append("motion_trigger")
            report["motion_triggered_yolo_calls"] += 1
        if current_state == STATE_IDLE and heartbeat_due and not motion_trigger:
            selection_reason.append("empty_heartbeat")
            report["heartbeat_yolo_calls"] += 1
        if scheduled_due:
            selection_reason.append(f"scheduled_{current_state.lower()}")
        if current_state == STATE_LOW:
            report["low_state_yolo_calls"] += 1
        elif current_state == STATE_NORMAL:
            report["normal_state_yolo_calls"] += 1
        elif current_state == STATE_BURST:
            report["burst_state_yolo_calls"] += 1

        record = _selection_record(
            experiment_dir=experiment_dir,
            frame_idx=frame_idx,
            timestamp_seconds=timestamp_seconds,
            tracking_state=current_state,
            target_fps=current_target_fps,
            selection_reason=selection_reason or [f"scheduled_{current_state.lower()}"],
            previous_detection_count=previous_detection_count,
            active_track_count=active_track_count,
        )
        _save_frame(frame, experiment_dir / record["image_path"])
        frame_records.append(record)
        selected_indices.add(frame_idx)
        yolo_frame_indices.add(frame_idx)
        report["frames_sent_to_yolo"] += 1

        frame_payload, current_rows, inference_seconds = detector.process_frame(frame_item=record, original_image=frame)
        report["yolo_processing_time_seconds"] += inference_seconds
        yolo_frame_payloads.append(frame_payload)
        detection_rows.extend(current_rows)
        for row in current_rows:
            class_name = str(row["class_name"])
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        observation = _build_observation(
            detection_rows=current_rows,
            previous_rows=previous_rows,
            timestamp_seconds=timestamp_seconds,
            previous_detection_count=previous_detection_count,
            active_track_count=active_track_count,
            image_width=metadata.width,
            image_height=metadata.height,
        )
        decision = controller.observe(observation)
        current_state = decision.state
        current_target_fps = decision.target_fps
        record["tracking_state"] = current_state
        record["target_fps"] = current_target_fps
        record["selection_reason"] = _dedupe_reason_items(list(record["selection_reason"]) + list(decision.selection_reason))
        state_counts[current_state] = state_counts.get(current_state, 0) + 1

        previous_rows = current_rows
        previous_detection_count = observation.detection_count
        active_track_count = observation.detection_count
        last_yolo_timestamp = timestamp_seconds
        if current_rows:
            last_detection_timestamp = timestamp_seconds
        next_yolo_due_timestamp = timestamp_seconds + (1.0 / current_target_fps if current_target_fps > 0 else 0.0)
        if decision.transition:
            transition_log.extend(controller.transition_log()[-1:])
        frame_idx += 1

    capture.release()
    report["unique_frames_decoded"] = len(decoded_indices)
    yolo_payload = {
        "status": "success",
        "models_used": config.model_specs,
        "yolo_conf_threshold": config.conf_threshold,
        "yolo_iou_threshold": config.iou_threshold,
        "device_used": config.device,
        "frames_processed": len(yolo_frame_payloads),
        "total_detections": len(detection_rows),
        "class_counts": dict(sorted(class_counts.items())),
        "detections": yolo_frame_payloads,
    }
    yolo_report = {
        "status": "success",
        "total_video_frames": metadata.frame_count,
        "frames_processed_by_yolo": len(yolo_frame_payloads),
        "processed_frame_percentage": round((len(yolo_frame_payloads) / metadata.frame_count) * 100.0, 3) if metadata.frame_count > 0 else 0.0,
        "detections_by_class": dict(sorted(class_counts.items())),
        "yolo_inference_time_seconds": round(report["yolo_processing_time_seconds"], 3),
        "average_yolo_time_per_processed_frame": round(report["yolo_processing_time_seconds"] / len(yolo_frame_payloads), 6) if yolo_frame_payloads else 0.0,
        "frame_counts_per_controller_state": dict(sorted(state_counts.items())),
    }
    return frame_records, transition_log, yolo_payload, yolo_report, detection_rows, report


def _run_dynamic_schedule(*, metadata: Any, experiment_dir: Path, controller: DynamicFpsController) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config = resolve_pipeline_config()
    capture = cv2.VideoCapture(str(metadata.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {metadata.video_path}")
    selected_indices: set[int] = set()
    frame_records: list[dict[str, Any]] = []
    transition_log: list[dict[str, Any]] = []
    previous_rows: list[dict[str, Any]] = []
    previous_detection_count = 0
    active_track_count = 0
    current_target_fps = controller.config.idle_fps
    current_state = controller.current_state
    next_target_idx = 0
    frame_idx = 0

    YOLO = cv2  # placeholder to keep local scope explicit for linting parity
    del YOLO
    yolo_config = resolve_pipeline_config()
    detector_payload, _, _, _ = detect_on_selected_frames(
        experiment_dir=experiment_dir,
        frame_records=[],
        config=yolo_config,
        annotated_dir_name="__unused__",
        crop_dir_name="__unused__",
        report_name="__unused__",
        output_name="__unused__",
    ) if False else ({}, {}, [], 0.0)  # pragma: no cover - keeps type shape without executing

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_idx == next_target_idx and frame_idx not in selected_indices:
            timestamp_seconds = frame_idx / metadata.fps if metadata.fps > 0 else 0.0
            selection_reason = ["initial_idle_probe"] if not frame_records else [f"controller_state_{current_state.lower()}"]
            record = _selection_record(
                experiment_dir=experiment_dir,
                frame_idx=frame_idx,
                timestamp_seconds=timestamp_seconds,
                tracking_state=current_state,
                target_fps=current_target_fps,
                selection_reason=selection_reason,
                previous_detection_count=previous_detection_count,
                active_track_count=active_track_count,
            )
            _save_frame(frame, experiment_dir / record["image_path"])
            frame_records.append(record)
            selected_indices.add(frame_idx)
            # Use one-frame YOLO inference causally to decide the next sampling rate.
            output_payload, _, detection_rows, _ = detect_on_selected_frames(
                experiment_dir=experiment_dir,
                frame_records=[record],
                config=config,
                annotated_dir_name="__dynamic_probe_annotated__",
                crop_dir_name="__dynamic_probe_crops__",
                report_name="__dynamic_probe_report__",
                output_name="__dynamic_probe_output__",
            )
            current_rows = list(output_payload["detections"][0]["detections"]) if output_payload.get("detections") else []
            observation = _build_observation(
                detection_rows=current_rows,
                previous_rows=previous_rows,
                timestamp_seconds=timestamp_seconds,
                previous_detection_count=previous_detection_count,
                active_track_count=active_track_count,
                image_width=metadata.width,
                image_height=metadata.height,
            )
            decision = controller.observe(observation)
            current_state = decision.state
            current_target_fps = decision.target_fps
            frame_records[-1]["tracking_state"] = current_state
            frame_records[-1]["target_fps"] = current_target_fps
            frame_records[-1]["selection_reason"] = decision.selection_reason
            previous_detection_count = observation.detection_count
            active_track_count = observation.detection_count
            previous_rows = current_rows
            if decision.transition:
                transition_log.extend(controller.transition_log()[-1:])
            next_target_idx = next_frame_index(
                current_frame_idx=frame_idx,
                video_fps=metadata.fps,
                target_fps=current_target_fps,
                total_frames=metadata.frame_count,
            )
        frame_idx += 1

    capture.release()
    return frame_records, transition_log


def _run_mode(
    *,
    mode_name: str,
    metadata: Any,
    experiment_dir: Path,
    frame_records: list[dict[str, Any]],
    preview_name: str,
    detections_name: str,
    report_name: str,
    raw_tracks_name: str,
    merged_tracks_name: str,
    merge_audit_name: str,
) -> dict[str, Any]:
    tracking_runtime = _build_tracking_runtime_config()
    config = resolve_pipeline_config(
        track_buffer_seconds=tracking_runtime["track_buffer_seconds"],
        high_confidence=tracking_runtime["high_confidence"],
        low_confidence=tracking_runtime["low_confidence"],
        match_threshold=tracking_runtime["match_threshold"],
        min_track_length=tracking_runtime["min_track_length"],
        save_annotated=_read_bool(ENV_DYNAMIC_SAVE_ANNOTATED, True),
        save_crops=_read_bool(ENV_DYNAMIC_SAVE_CROPS, True),
    )
    annotated_dir_name = "annotated_frames"
    crop_dir_name = "dynamic_object_crops"
    yolo_payload, yolo_report, detection_rows, yolo_seconds = detect_on_selected_frames(
        experiment_dir=experiment_dir,
        frame_records=frame_records,
        config=config,
        annotated_dir_name=annotated_dir_name,
        crop_dir_name=crop_dir_name,
        report_name=report_name,
        output_name=detections_name,
    )
    yolo_report["total_video_frames"] = metadata.frame_count
    yolo_report["processed_frame_percentage"] = round((len(frame_records) / metadata.frame_count) * 100.0, 3) if metadata.frame_count > 0 else 0.0
    write_json(experiment_dir / detections_name, yolo_payload)
    write_json(experiment_dir / report_name, yolo_report)

    tracker_started = time.perf_counter()
    raw_tracks, bytetrack_meta, _tracker_state = run_variable_gap_bytetrack(
        frame_records=frame_records,
        detection_rows=detection_rows,
        image_width=metadata.width,
        image_height=metadata.height,
        reference_fps=max(10.0, config.match_threshold * 12.5),
        track_buffer_seconds=tracking_runtime["track_buffer_seconds"],
        high_confidence=tracking_runtime["high_confidence"],
        low_confidence=tracking_runtime["low_confidence"],
        match_threshold=tracking_runtime["match_threshold"],
    )
    tracker_seconds = time.perf_counter() - tracker_started

    raw_tracks_payload, _ = build_step05_compatible_tracks(
        run_dir=experiment_dir,
        tracks=raw_tracks,
        image_width=metadata.width,
        image_height=metadata.height,
        min_track_length=tracking_runtime["min_track_length"],
    )
    write_json(experiment_dir / raw_tracks_name, raw_tracks_payload)

    image_diagonal = (metadata.width**2 + metadata.height**2) ** 0.5
    merge_started = time.perf_counter()
    merged_tracks, merge_audit, _merge_meta = merge_track_fragments(
        run_dir=experiment_dir,
        raw_tracks=raw_tracks,
        image_diagonal=image_diagonal,
        max_candidates_per_track=tracking_runtime["merge_max_candidates_per_track"],
    )
    merge_seconds = time.perf_counter() - merge_started
    write_json(experiment_dir / merge_audit_name, merge_audit)
    merged_tracks_payload, _ = build_step05_compatible_tracks(
        run_dir=experiment_dir,
        tracks=merged_tracks,
        image_width=metadata.width,
        image_height=metadata.height,
        min_track_length=tracking_runtime["min_track_length"],
    )
    compatibility = validate_step05_compatibility(merged_tracks_payload)
    if not compatibility["compatible"]:
        raise RuntimeError(f"{mode_name} output is not Step 05 compatible: {compatibility['errors']}")
    write_json(experiment_dir / merged_tracks_name, merged_tracks_payload)

    burst_timestamps = {
        round(float(item["timestamp_seconds"]), 6)
        for item in frame_records
        if str(item.get("tracking_state")) == STATE_BURST
    }
    build_preview_video(
        experiment_dir=experiment_dir,
        preview_name=preview_name,
        frame_records=frame_records,
        merged_tracks_payload=merged_tracks_payload,
        burst_timestamps=burst_timestamps,
    )
    return summarize_tracks(
        mode_name=mode_name,
        total_video_frames=metadata.frame_count,
        frames_processed=len(frame_records),
        yolo_seconds=yolo_seconds,
        tracker_seconds=tracker_seconds,
        merge_seconds=merge_seconds,
        yolo_payload=yolo_payload,
        raw_tracks_payload=raw_tracks_payload,
        merged_tracks_payload=merged_tracks_payload,
        bytetrack_meta=bytetrack_meta,
    )


def _run_single_pass_tracking_outputs(
    *,
    metadata: Any,
    experiment_dir: Path,
    frame_records: list[dict[str, Any]],
    yolo_payload: dict[str, Any],
    yolo_report: dict[str, Any],
    detection_rows: list[dict[str, Any]],
    single_pass_report: dict[str, Any],
) -> dict[str, Any]:
    tracking_runtime = _build_tracking_runtime_config()
    write_json(experiment_dir / "single_pass_dynamic_yolo_detections.json", yolo_payload)

    tracker_started = time.perf_counter()
    raw_tracks, bytetrack_meta, _tracker_state = run_variable_gap_bytetrack(
        frame_records=frame_records,
        detection_rows=detection_rows,
        image_width=metadata.width,
        image_height=metadata.height,
        reference_fps=max(10.0, tracking_runtime["match_threshold"] * 12.5),
        track_buffer_seconds=tracking_runtime["track_buffer_seconds"],
        high_confidence=tracking_runtime["high_confidence"],
        low_confidence=tracking_runtime["low_confidence"],
        match_threshold=tracking_runtime["match_threshold"],
    )
    single_pass_report["tracking_time_seconds"] = round(time.perf_counter() - tracker_started, 3)

    raw_tracks_payload, _ = build_step05_compatible_tracks(
        run_dir=experiment_dir,
        tracks=raw_tracks,
        image_width=metadata.width,
        image_height=metadata.height,
        min_track_length=tracking_runtime["min_track_length"],
    )
    write_json(experiment_dir / "single_pass_dynamic_tracks_raw.json", raw_tracks_payload)

    return _finalize_single_pass_merge_outputs(
        metadata=metadata,
        experiment_dir=experiment_dir,
        frame_records=frame_records,
        yolo_payload=yolo_payload,
        yolo_report=yolo_report,
        raw_tracks_payload=raw_tracks_payload,
        raw_tracks=raw_tracks,
        bytetrack_meta=bytetrack_meta,
        yolo_seconds=float(single_pass_report["yolo_processing_time_seconds"]),
        tracker_seconds=float(single_pass_report["tracking_time_seconds"]),
        report_seed=single_pass_report,
        resumed_from_existing_raw_tracks=False,
    )


def _finalize_single_pass_merge_outputs(
    *,
    metadata: Any,
    experiment_dir: Path,
    frame_records: list[dict[str, Any]],
    yolo_payload: dict[str, Any],
    yolo_report: dict[str, Any],
    raw_tracks_payload: dict[str, Any],
    raw_tracks: list[dict[str, Any]],
    bytetrack_meta: dict[str, Any],
    yolo_seconds: float,
    tracker_seconds: float,
    report_seed: dict[str, Any],
    resumed_from_existing_raw_tracks: bool,
) -> dict[str, Any]:
    tracking_runtime = _build_tracking_runtime_config()
    image_diagonal = (metadata.width**2 + metadata.height**2) ** 0.5
    merged_tracks, merge_audit, merge_meta = merge_track_fragments(
        run_dir=experiment_dir,
        raw_tracks=raw_tracks,
        image_diagonal=image_diagonal,
        max_candidates_per_track=tracking_runtime["merge_max_candidates_per_track"],
    )
    write_json(experiment_dir / "single_pass_dynamic_merge_audit.json", merge_audit)
    merged_tracks_payload, _ = build_step05_compatible_tracks(
        run_dir=experiment_dir,
        tracks=merged_tracks,
        image_width=metadata.width,
        image_height=metadata.height,
        min_track_length=tracking_runtime["min_track_length"],
    )
    compatibility = validate_step05_compatibility(merged_tracks_payload)
    if not compatibility["compatible"]:
        raise RuntimeError(f"single_pass_dynamic output is not Step 05 compatible: {compatibility['errors']}")
    write_json(experiment_dir / "single_pass_dynamic_tracks.json", merged_tracks_payload)

    burst_timestamps = {
        round(float(item["timestamp_seconds"]), 6)
        for item in frame_records
        if str(item.get("tracking_state")) == STATE_BURST
    }
    build_preview_video(
        experiment_dir=experiment_dir,
        preview_name="single_pass_dynamic_preview.mp4",
        frame_records=frame_records,
        merged_tracks_payload=merged_tracks_payload,
        burst_timestamps=burst_timestamps,
    )

    merge_seconds = float(merge_meta.get("merge_time_seconds", 0.0))
    summary = summarize_tracks(
        mode_name="single_pass_dynamic",
        total_video_frames=metadata.frame_count,
        frames_processed=len(frame_records),
        yolo_seconds=yolo_seconds,
        tracker_seconds=tracker_seconds,
        merge_seconds=merge_seconds,
        yolo_payload=yolo_payload,
        raw_tracks_payload=raw_tracks_payload,
        merged_tracks_payload=merged_tracks_payload,
        bytetrack_meta=bytetrack_meta,
    )
    raw_tracks_list = list(raw_tracks_payload.get("tracks", []))
    merged_tracks_list = list(merged_tracks_payload.get("tracks", []))
    raw_vehicle_tracks = [item for item in raw_tracks_list if item.get("track_type") == "vehicle"]
    merged_vehicle_tracks = [item for item in merged_tracks_list if item.get("track_type") == "vehicle"]
    raw_person_tracks = [item for item in raw_tracks_list if item.get("track_type") == "person"]
    merged_person_tracks = [item for item in merged_tracks_list if item.get("track_type") == "person"]
    final_report = dict(report_seed)
    final_report["raw_track_count"] = len(raw_tracks_list)
    final_report["merged_track_count"] = len(merged_tracks_list)
    final_report["raw_vehicle_track_count"] = len(raw_vehicle_tracks)
    final_report["merged_vehicle_track_count"] = len(merged_vehicle_tracks)
    final_report["raw_person_track_count"] = len(raw_person_tracks)
    final_report["merged_person_track_count"] = len(merged_person_tracks)
    final_report["single_frame_track_count"] = int(summary.get("single_frame_tracks", 0))
    final_report["fragmented_track_count"] = int(summary.get("fragmented_tracks", 0))
    final_report["good_track_count"] = int(summary.get("good_tracks", 0))
    final_report["merge_meta"] = merge_meta
    final_report["lost_track_recoveries"] = int(bytetrack_meta.get("vehicle_refind_count", 0))
    final_report["step05_compatibility"] = compatibility
    final_report["frames_processed_by_yolo"] = len(frame_records)
    final_report["processed_frame_percentage"] = yolo_report.get("processed_frame_percentage", 0.0)
    final_report["mode_name"] = "single_pass_dynamic"
    final_report["resumed_from_existing_raw_tracks"] = resumed_from_existing_raw_tracks
    return final_report


def _resolve_resume_paths(path_value: Path, repo_root: Path) -> tuple[Path, Path]:
    resolved = path_value.expanduser()
    if not resolved.is_absolute():
        resolved = (repo_root / resolved).resolve()
    if resolved.name == "dynamic_yolo_tracking_experiment":
        experiment_dir = resolved
        run_dir = resolved.parent
    else:
        run_dir = resolved
        experiment_dir = run_dir / "dynamic_yolo_tracking_experiment"
    return run_dir, experiment_dir


def _resume_single_pass_merge_outputs(*, repo_root: Path, resume_run_dir: Path) -> tuple[Path, Path]:
    run_dir, experiment_dir = _resolve_resume_paths(resume_run_dir, repo_root)
    required_paths = [
        run_dir / "01_video_info.json",
        experiment_dir / "single_pass_dynamic_frames.json",
        experiment_dir / "single_pass_dynamic_yolo_detections.json",
        experiment_dir / "single_pass_dynamic_tracks_raw.json",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Resume merge mode is missing required files: {missing}")

    video_info = read_json(run_dir / "01_video_info.json")
    metadata = SimpleNamespace(
        video_path=Path(str(video_info.get("input_video_path", ""))),
        fps=float(video_info.get("fps", 0.0) or 0.0),
        frame_count=int(video_info.get("frame_count", 0) or 0),
        width=int(video_info.get("width", 0) or 0),
        height=int(video_info.get("height", 0) or 0),
        duration_seconds=float(video_info.get("duration_seconds", 0.0) or 0.0),
    )
    frame_records = list(read_json(experiment_dir / "single_pass_dynamic_frames.json").get("selected_frames", []))
    yolo_payload = read_json(experiment_dir / "single_pass_dynamic_yolo_detections.json")
    raw_tracks_payload = read_json(experiment_dir / "single_pass_dynamic_tracks_raw.json")
    yolo_report = {
        "processed_frame_percentage": round((len(frame_records) / metadata.frame_count) * 100.0, 3) if metadata.frame_count > 0 else 0.0,
    }
    final_report = _finalize_single_pass_merge_outputs(
        metadata=metadata,
        experiment_dir=experiment_dir,
        frame_records=frame_records,
        yolo_payload=yolo_payload,
        yolo_report=yolo_report,
        raw_tracks_payload=raw_tracks_payload,
        raw_tracks=list(raw_tracks_payload.get("tracks", [])),
        bytetrack_meta={},
        yolo_seconds=0.0,
        tracker_seconds=0.0,
        report_seed={
            "tracking_time_seconds": 0.0,
            "yolo_processing_time_seconds": 0.0,
        },
        resumed_from_existing_raw_tracks=True,
    )
    write_json(experiment_dir / "single_pass_dynamic_report.json", final_report)
    return run_dir, experiment_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the isolated single-pass dynamic YOLO tracking experiment.")
    parser.add_argument("--with-fixed-comparisons", action="store_true", help="Also run the legacy fixed 3 FPS and fixed 5 FPS comparison passes.")
    parser.add_argument("--resume-merge-run-dir", type=Path, help="Resume only the fragment-merge/finalization stage from an existing dynamic experiment run directory.")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[4]
    if args.resume_merge_run_dir is not None:
        run_dir, experiment_dir = _resume_single_pass_merge_outputs(repo_root=repo_root, resume_run_dir=args.resume_merge_run_dir)
        print(f"run_dir={run_dir}")
        print(f"experiment_dir={experiment_dir}")
        return
    raw_video_path = os.environ.get(ENV_DYNAMIC_VIDEO_PATH, "").strip()
    if not raw_video_path:
        raise RuntimeError(
            f"Environment variable {ENV_DYNAMIC_VIDEO_PATH} is required before running the real dynamic tracking experiment."
        )
    video_path = Path(raw_video_path).expanduser()
    if not video_path.is_absolute():
        video_path = (repo_root / video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Dynamic tracking experiment video path does not exist: {video_path}")

    metadata = read_video_metadata(video_path)
    run_dir, experiment_dir = _create_run_dir(repo_root, video_path)
    _write_video_info(run_dir, metadata)

    dynamic_controller = DynamicFpsController(DynamicFpsConfig())
    dynamic_frames, transition_log, yolo_payload, yolo_report, detection_rows, single_pass_report = _run_single_pass_dynamic(
        metadata=metadata,
        experiment_dir=experiment_dir,
        controller=dynamic_controller,
    )

    dynamic_validation = validate_chronological_frame_records(dynamic_frames)
    if not dynamic_validation["chronological"] or dynamic_validation["duplicate_frame_indexes"]:
        raise RuntimeError(f"Dynamic frame schedule failed validation: {dynamic_validation}")

    write_json(experiment_dir / "single_pass_dynamic_frames.json", {"status": "success", "selected_frames": dynamic_frames})
    write_json(experiment_dir / "single_pass_dynamic_state_transitions.json", transition_log)
    final_report = _run_single_pass_tracking_outputs(
        metadata=metadata,
        experiment_dir=experiment_dir,
        frame_records=dynamic_frames,
        yolo_payload=yolo_payload,
        yolo_report=yolo_report,
        detection_rows=detection_rows,
        single_pass_report=single_pass_report,
    )
    write_json(experiment_dir / "single_pass_dynamic_report.json", final_report)

    if args.with_fixed_comparisons:
        fixed_3fps_frames = _build_fixed_schedule(metadata=metadata, experiment_dir=experiment_dir, target_fps=3.0, state_name="FIXED_3FPS")
        fixed_5fps_frames = _build_fixed_schedule(metadata=metadata, experiment_dir=experiment_dir, target_fps=5.0, state_name="FIXED_5FPS")
        fixed_3fps_summary = _run_mode(
            mode_name="fixed_3fps",
            metadata=metadata,
            experiment_dir=experiment_dir,
            frame_records=fixed_3fps_frames,
            preview_name="tracking_preview_fixed_3fps.mp4",
            detections_name="fixed_3fps_detections.json",
            report_name="fixed_3fps_detection_report.json",
            raw_tracks_name="fixed_3fps_tracks_raw.json",
            merged_tracks_name="fixed_3fps_tracks.json",
            merge_audit_name="fixed_3fps_merge_audit.json",
        )
        fixed_5fps_summary = _run_mode(
            mode_name="fixed_5fps",
            metadata=metadata,
            experiment_dir=experiment_dir,
            frame_records=fixed_5fps_frames,
            preview_name="tracking_preview_fixed_5fps.mp4",
            detections_name="fixed_5fps_detections.json",
            report_name="fixed_5fps_detection_report.json",
            raw_tracks_name="fixed_5fps_tracks_raw.json",
            merged_tracks_name="fixed_5fps_tracks.json",
            merge_audit_name="fixed_5fps_merge_audit.json",
        )
        write_json(experiment_dir / "fixed_3fps_results.json", fixed_3fps_summary)
        write_json(experiment_dir / "fixed_5fps_results.json", fixed_5fps_summary)
        write_json(
            experiment_dir / "dynamic_tracking_comparison.json",
            compare_mode_summaries(fixed_3fps_summary, fixed_5fps_summary, final_report),
        )

    print(f"run_dir={run_dir}")
    print(f"experiment_dir={experiment_dir}")


if __name__ == "__main__":
    main()
