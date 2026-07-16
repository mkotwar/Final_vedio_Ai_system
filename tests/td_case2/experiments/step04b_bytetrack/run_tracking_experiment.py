from __future__ import annotations

import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

if __package__ in {None, ""}:
    case_root = Path(__file__).resolve().parents[2]
    if str(case_root) not in sys.path:
        sys.path.insert(0, str(case_root))
    from experiments.step04b_bytetrack.bytetrack_adapter import run_bytetrack_tracking
    from experiments.step04b_bytetrack.dense_tracking_frames import build_dense_tracking_frames
    from experiments.step04b_bytetrack.fragment_merger import merge_track_fragments
    from experiments.step04b_bytetrack.tracking_metrics import (
        build_step05_compatible_tracks,
        validate_step05_compatibility,
    )
else:
    from .bytetrack_adapter import run_bytetrack_tracking
    from .dense_tracking_frames import build_dense_tracking_frames
    from .fragment_merger import merge_track_fragments
    from .tracking_metrics import build_step05_compatible_tracks, validate_step05_compatibility

from config import (
    DEFAULT_TRACKING_MIN_TRACK_LENGTH,
    DEFAULT_YOLO_CONF_THRESHOLD,
    DEFAULT_YOLO_IOU_THRESHOLD,
)
from stage_checks import read_json, write_json
from step_03b_yolo_detection import _load_yolo_class


ENV_TRACKING_FPS = "TD_CASE2_EXP_TRACKING_FPS"
ENV_TRACK_BUFFER_SECONDS = "TD_CASE2_EXP_TRACK_BUFFER_SECONDS"
ENV_TRACK_HIGH_CONFIDENCE = "TD_CASE2_EXP_TRACK_HIGH_CONFIDENCE"
ENV_TRACK_LOW_CONFIDENCE = "TD_CASE2_EXP_TRACK_LOW_CONFIDENCE"
ENV_TRACK_MATCH_THRESHOLD = "TD_CASE2_EXP_TRACK_MATCH_THRESHOLD"
ENV_TRACK_MIN_LENGTH = "TD_CASE2_EXP_TRACK_MIN_LENGTH"
ENV_EXPERIMENT_RUN_DIR = "TD_CASE2_EXP_RUN_DIR"


def _read_float(env_name: str, default_value: float) -> float:
    raw_value = os.environ.get(env_name, str(default_value)).strip()
    return float(raw_value)


def _read_int(env_name: str, default_value: int) -> int:
    raw_value = os.environ.get(env_name, str(default_value)).strip()
    return int(raw_value)


def _latest_crash_run(repo_root: Path) -> Path:
    debug_dir = repo_root / "debug_runs"
    candidates = sorted(
        [
            path
            for path in debug_dir.iterdir()
            if path.is_dir()
            and path.name.startswith("vidssave.com Woman crashes into lamppost, flips car during driving test 720P_")
            and (path / "01_video_info.json").exists()
            and (path / "03_yolo_detections.json").exists()
            and (path / "04B_tracks.json").exists()
        ],
        key=lambda item: item.name,
    )
    if not candidates:
        raise FileNotFoundError("No complete crash-video run was found for the isolated tracking experiment.")
    return candidates[-1]


def _resolve_target_run(repo_root: Path) -> Path:
    raw_run_dir = os.environ.get(ENV_EXPERIMENT_RUN_DIR, "").strip()
    if raw_run_dir:
        run_dir = Path(raw_run_dir).expanduser()
        if not run_dir.is_absolute():
            run_dir = (repo_root / run_dir).resolve()
        required = ("01_video_info.json", "03_yolo_detections.json", "04B_tracks.json")
        missing = [name for name in required if not (run_dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"Experiment run override is missing required files {missing}: {run_dir}")
        return run_dir
    return _latest_crash_run(repo_root)


def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def _center_distance_ratio(box_a: list[float], box_b: list[float], image_diagonal: float) -> float:
    acx = (box_a[0] + box_a[2]) / 2.0
    acy = (box_a[1] + box_a[3]) / 2.0
    bcx = (box_b[0] + box_b[2]) / 2.0
    bcy = (box_b[1] + box_b[3]) / 2.0
    distance = (((acx - bcx) ** 2) + ((acy - bcy) ** 2)) ** 0.5
    return distance / image_diagonal if image_diagonal > 0 else 0.0


def _area_ratio(box_a: list[float], box_b: list[float]) -> float:
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    if area_a <= 0.0 or area_b <= 0.0:
        return float("inf")
    return max(area_a, area_b) / min(area_a, area_b)


def _baseline_analysis(run_dir: Path, image_diagonal: float) -> dict[str, Any]:
    adaptive_payload = read_json(run_dir / "02A_adaptive_frames.json")
    yolo_payload = read_json(run_dir / "03_yolo_detections.json")
    tracks_payload = read_json(run_dir / "04B_tracks.json")
    report_payload = read_json(run_dir / "04B_tracking_report.json")

    selected_frames = list(adaptive_payload.get("selected_frames", []))
    gaps = [
        round(float(selected_frames[index]["timestamp_seconds"]) - float(selected_frames[index - 1]["timestamp_seconds"]), 6)
        for index in range(1, len(selected_frames))
    ]
    vehicle_tracks = [track for track in list(tracks_payload.get("tracks", [])) if track.get("track_type") == "vehicle"]
    vehicle_tracks.sort(key=lambda item: float(item.get("start_timestamp_seconds", 0.0)))
    fragment_links: list[dict[str, Any]] = []
    likely_same_car_track_ids: list[str] = []
    for index, track in enumerate(vehicle_tracks):
        if index == 0:
            likely_same_car_track_ids.append(track["track_id"])
            continue
        previous = vehicle_tracks[index - 1]
        prev_box = previous["detections"][-1]["bbox_xyxy"]
        curr_box = track["detections"][0]["bbox_xyxy"]
        gap = round(float(track["start_timestamp_seconds"]) - float(previous["end_timestamp_seconds"]), 6)
        iou = round(_bbox_iou(prev_box, curr_box), 6)
        center_ratio = round(_center_distance_ratio(prev_box, curr_box, image_diagonal), 6)
        bbox_ratio = _area_ratio(prev_box, curr_box)
        reasons: list[str] = []
        if gap > 2.5:
            reasons.append("large_timestamp_gap")
        if center_ratio > 0.25:
            reasons.append("large_centre_movement")
        if bbox_ratio > 3.0:
            reasons.append("bbox_size_change")
        if iou < 0.25 and center_ratio > 0.0875:
            reasons.append("low_iou_plus_distance_gate")
        likely_same_car = bool(track.get("dominant_class_name") == "car" and previous.get("dominant_class_name") == "car")
        if likely_same_car:
            likely_same_car_track_ids.append(track["track_id"])
        fragment_links.append(
            {
                "previous_track_id": previous["track_id"],
                "next_track_id": track["track_id"],
                "time_gap_seconds": gap,
                "iou": iou,
                "center_distance_ratio": center_ratio,
                "bbox_area_ratio": None if bbox_ratio == float("inf") else round(bbox_ratio, 6),
                "likely_same_physical_car": likely_same_car,
                "failure_reasons": reasons or ["new_track_created_despite_same_class"],
            }
        )

    return {
        "tracking_input_frames": len(selected_frames),
        "average_time_gap_seconds": round(sum(gaps) / len(gaps), 6) if gaps else 0.0,
        "max_time_gap_seconds": max(gaps) if gaps else 0.0,
        "vehicle_detections": int(yolo_payload.get("class_counts", {}).get("car", 0)),
        "vehicle_tracks": len(vehicle_tracks),
        "track_quality_counts": report_payload.get("track_quality_counts", {}),
        "likely_same_physical_car_track_ids": likely_same_car_track_ids,
        "fragment_link_analysis": fragment_links,
    }


def _run_dense_yolo(
    *,
    experiment_dir: Path,
    dense_manifest: dict[str, Any],
    model_specs: list[dict[str, Any]],
    conf_threshold: float,
    iou_threshold: float,
    device: str,
) -> tuple[dict[str, Any], float]:
    YOLO = _load_yolo_class()
    loaded_models = {item["model_role"]: YOLO(str(item["model_path"])) for item in model_specs}
    class_lookups: dict[str, dict[str, str]] = {}
    for role, model in loaded_models.items():
        names = getattr(model, "names", {})
        if isinstance(names, dict):
            class_lookups[role] = {str(key): str(value) for key, value in names.items()}
        else:
            class_lookups[role] = {str(index): str(value) for index, value in enumerate(names)}

    annotated_dir = experiment_dir / "experimental_dense_yolo_annotated"
    crops_dir = experiment_dir / "experimental_dense_yolo_object_crops"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    frame_results: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    started = time.perf_counter()
    for frame_item in list(dense_manifest.get("selected_frames", [])):
        relative_image_path = Path(str(frame_item["image_path"]))
        absolute_image_path = (experiment_dir / relative_image_path).resolve()
        original_image = cv2.imread(str(absolute_image_path))
        if original_image is None:
            raise RuntimeError(f"Failed to load dense tracking frame: {absolute_image_path}")
        annotated_image = original_image.copy()
        frame_height, frame_width = original_image.shape[:2]
        frame_payload = {
            "frame_id": frame_item["frame_id"],
            "frame_idx": frame_item["frame_idx"],
            "timestamp_seconds": frame_item["timestamp_seconds"],
            "timestamp_text": frame_item["timestamp_text"],
            "image_path": frame_item["image_path"],
            "detections": [],
        }
        for model_spec in model_specs:
            role = str(model_spec["model_role"])
            results = loaded_models[role].predict(
                source=str(absolute_image_path),
                conf=conf_threshold,
                iou=iou_threshold,
                device=device,
                verbose=False,
            )
            if not results:
                continue
            result = results[0]
            annotated_image = result.plot()
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy_list = boxes.xyxy.tolist()
            xywh_list = boxes.xywh.tolist() if getattr(boxes, "xywh", None) is not None else []
            cls_list = boxes.cls.tolist() if getattr(boxes, "cls", None) is not None else []
            conf_list = boxes.conf.tolist() if getattr(boxes, "conf", None) is not None else []
            for index, bbox_xyxy in enumerate(xyxy_list):
                x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
                class_id = int(cls_list[index]) if index < len(cls_list) else -1
                class_name = class_lookups[role].get(str(class_id), str(class_id))
                if class_name.lower() not in {"person", "car", "motorcycle", "bus", "truck", "bicycle", "auto", "van", "vehicle"}:
                    continue
                confidence = float(conf_list[index]) if index < len(conf_list) else 0.0
                bbox_width = max(0.0, x2 - x1)
                bbox_height = max(0.0, y2 - y1)
                crop_padding_ratio = 0.05 if class_name.lower() != "person" else 0.0
                pad_x = bbox_width * crop_padding_ratio
                pad_y = bbox_height * crop_padding_ratio
                ix1 = max(0, int(round(x1 - pad_x)))
                iy1 = max(0, int(round(y1 - pad_y)))
                ix2 = min(frame_width, int(round(x2 + pad_x)))
                iy2 = min(frame_height, int(round(y2 + pad_y)))
                crop_path = ""
                if ix2 > ix1 and iy2 > iy1:
                    crop = original_image[iy1:iy2, ix1:ix2]
                    crop_name = f"{frame_item['frame_id']}_{role}_{index + 1:03d}_{class_name}.jpg"
                    crop_output_path = crops_dir / crop_name
                    if cv2.imwrite(str(crop_output_path), crop):
                        crop_path = str(Path("experimental_dense_yolo_object_crops") / crop_name).replace("\\", "/")
                bbox_area = bbox_width * bbox_height
                bbox_area_ratio = bbox_area / float(frame_width * frame_height) if frame_width > 0 and frame_height > 0 else 0.0
                bbox_xywh = [float(value) for value in xywh_list[index]] if index < len(xywh_list) else [0.0, 0.0, 0.0, 0.0]
                frame_payload["detections"].append(
                    {
                        "detection_id": f"{frame_item['frame_id']}_{role}_{index + 1:03d}",
                        "model_role": role,
                        "class_id": class_id,
                        "class_name": class_name.lower(),
                        "confidence": round(confidence, 6),
                        "bbox_xyxy": [round(value, 3) for value in [x1, y1, x2, y2]],
                        "bbox_xywh": [round(float(value), 3) for value in bbox_xywh],
                        "bbox_area_ratio": round(bbox_area_ratio, 6),
                        "frame_id": frame_item["frame_id"],
                        "frame_idx": frame_item["frame_idx"],
                        "timestamp_seconds": frame_item["timestamp_seconds"],
                        "timestamp_text": frame_item["timestamp_text"],
                        "image_path": frame_item["image_path"],
                        "crop_path": crop_path,
                        "crop_exists": bool(crop_path),
                    }
                )
                class_counts[class_name.lower()] += 1
        annotated_name = f"{frame_item['frame_id']}.jpg"
        cv2.imwrite(str(annotated_dir / annotated_name), annotated_image)
        frame_results.append(frame_payload)

    elapsed = time.perf_counter() - started
    return {
        "status": "success",
        "models_used": model_specs,
        "yolo_conf_threshold": conf_threshold,
        "yolo_iou_threshold": iou_threshold,
        "device_used": device,
        "frames_processed": len(frame_results),
        "total_detections": sum(len(item["detections"]) for item in frame_results),
        "class_counts": dict(sorted(class_counts.items())),
        "detections": frame_results,
    }, elapsed


def _build_contact_sheets(experiment_dir: Path, tracks_payload: dict[str, Any]) -> None:
    import numpy as np

    contact_dir = experiment_dir / "track_contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)
    for track in list(tracks_payload.get("tracks", [])):
        detections = list(track.get("detections", []))[:4]
        tiles = []
        for detection in detections:
            crop_path_value = str(detection.get("crop_path", ""))
            if not crop_path_value:
                continue
            crop_path = (experiment_dir / crop_path_value).resolve()
            image = cv2.imread(str(crop_path))
            if image is None:
                continue
            resized = cv2.resize(image, (320, 220), interpolation=cv2.INTER_AREA)
            footer = np.full((60, 320, 3), 245, dtype=np.uint8)
            cv2.putText(
                footer,
                f"{track['track_id']} t={float(detection['timestamp_seconds']):.2f}s",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
            tiles.append(np.vstack([resized, footer]))
        if tiles:
            cv2.imwrite(str(contact_dir / f"{track['track_id']}_contact_sheet.jpg"), cv2.hconcat(tiles))


def _build_preview_video(experiment_dir: Path, detections_payload: dict[str, Any], tracks_payload: dict[str, Any]) -> None:
    track_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for track in list(tracks_payload.get("tracks", [])):
        for detection in list(track.get("detections", [])):
            track_lookup[(str(detection["frame_id"]), str(detection["detection_id"]))] = {
                "track_id": track["track_id"],
                "class_name": detection["class_name"],
                "confidence": detection["confidence"],
                "merged": bool(track.get("source_information", {}).get("merged_from_track_ids", [])),
                "bbox_xyxy": detection["bbox_xyxy"],
                "timestamp_seconds": detection["timestamp_seconds"],
            }

    frames = list(detections_payload.get("detections", []))
    if not frames:
        return
    first_frame_path = (experiment_dir / frames[0]["image_path"]).resolve()
    first_image = cv2.imread(str(first_frame_path))
    if first_image is None:
        return
    height, width = first_image.shape[:2]
    writer = cv2.VideoWriter(
        str(experiment_dir / "tracking_preview.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (width, height),
    )
    for frame in frames:
        image_path = (experiment_dir / frame["image_path"]).resolve()
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        for detection in list(frame.get("detections", [])):
            track_item = track_lookup.get((str(frame["frame_id"]), str(detection["detection_id"])))
            if track_item is None:
                continue
            x1, y1, x2, y2 = [int(round(float(value))) for value in track_item["bbox_xyxy"]]
            color = (40, 200, 40) if track_item["merged"] else (0, 180, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = (
                f"{track_item['track_id']} | {track_item['class_name']} | "
                f"{float(track_item['confidence']):.2f} | t={float(track_item['timestamp_seconds']):.2f}s | "
                f"{'post-merged' if track_item['merged'] else 'raw'}"
            )
            cv2.putText(image, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)
        writer.write(image)
    writer.release()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    run_dir = _resolve_target_run(repo_root)
    experiment_dir = run_dir / "tracking_experiment"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    tracking_fps = _read_float(ENV_TRACKING_FPS, 5.0)
    track_buffer_seconds = _read_float(ENV_TRACK_BUFFER_SECONDS, 2.0)
    high_confidence = _read_float(ENV_TRACK_HIGH_CONFIDENCE, 0.25)
    low_confidence = _read_float(ENV_TRACK_LOW_CONFIDENCE, 0.10)
    match_threshold = _read_float(ENV_TRACK_MATCH_THRESHOLD, 0.80)
    min_track_length = _read_int(ENV_TRACK_MIN_LENGTH, DEFAULT_TRACKING_MIN_TRACK_LENGTH)

    video_info = read_json(run_dir / "01_video_info.json")
    image_width = int(video_info.get("width", 0) or 0)
    image_height = int(video_info.get("height", 0) or 0)
    image_diagonal = (image_width**2 + image_height**2) ** 0.5
    baseline = _baseline_analysis(run_dir, image_diagonal)
    write_json(experiment_dir / "tracking_baseline_analysis.json", baseline)

    dense_manifest = build_dense_tracking_frames(
        video_path=Path(str(video_info["input_video_path"])),
        output_dir=experiment_dir,
        tracking_fps=tracking_fps,
    )
    write_json(experiment_dir / "experimental_dense_frames.json", dense_manifest)

    active_yolo_payload = read_json(run_dir / "03_yolo_detections.json")
    model_specs = [
        {
            "model_role": str(item["model_role"]),
            "model_path": str(item["model_path"]),
        }
        for item in list(active_yolo_payload.get("models_used", []))
    ]
    if not model_specs:
        raise RuntimeError("The active crash run does not record Step 03 model paths, so the experiment cannot reuse them.")
    dense_yolo_payload, dense_yolo_seconds = _run_dense_yolo(
        experiment_dir=experiment_dir,
        dense_manifest=dense_manifest,
        model_specs=model_specs,
        conf_threshold=float(active_yolo_payload.get("yolo_conf_threshold", DEFAULT_YOLO_CONF_THRESHOLD)),
        iou_threshold=float(active_yolo_payload.get("yolo_iou_threshold", DEFAULT_YOLO_IOU_THRESHOLD)),
        device=str(active_yolo_payload.get("device_used", "auto")),
    )
    write_json(experiment_dir / "experimental_dense_yolo_detections.json", dense_yolo_payload)

    detections_by_frame_id = {
        str(item["frame_id"]): list(item.get("detections", []))
        for item in list(dense_yolo_payload.get("detections", []))
    }
    tracking_started = time.perf_counter()
    raw_tracks, bytetrack_meta = run_bytetrack_tracking(
        frame_items=list(dense_manifest.get("selected_frames", [])),
        detections_by_frame_id=detections_by_frame_id,
        image_width=image_width,
        image_height=image_height,
        tracking_fps=tracking_fps,
        track_buffer_seconds=track_buffer_seconds,
        high_confidence=high_confidence,
        low_confidence=low_confidence,
        match_threshold=match_threshold,
    )
    tracker_seconds = time.perf_counter() - tracking_started

    raw_tracks_payload, raw_tracks_report = build_step05_compatible_tracks(
        run_dir=experiment_dir,
        tracks=raw_tracks,
        image_width=image_width,
        image_height=image_height,
        min_track_length=min_track_length,
    )
    write_json(experiment_dir / "04B_tracks_experimental_raw.json", raw_tracks_payload)

    merge_started = time.perf_counter()
    merged_tracks, merge_audit, merge_meta = merge_track_fragments(
        run_dir=experiment_dir,
        raw_tracks=raw_tracks,
        image_diagonal=image_diagonal,
    )
    merge_seconds = time.perf_counter() - merge_started
    write_json(experiment_dir / "tracking_fragment_merge_audit.json", merge_audit)

    merged_tracks_payload, merged_tracks_report = build_step05_compatible_tracks(
        run_dir=experiment_dir,
        tracks=merged_tracks,
        image_width=image_width,
        image_height=image_height,
        min_track_length=min_track_length,
    )
    compatibility = validate_step05_compatibility(merged_tracks_payload)
    if not compatibility["compatible"]:
        raise RuntimeError(f"Experimental Step 04B output failed Step 05 compatibility checks: {compatibility['errors']}")
    write_json(experiment_dir / "04B_tracks_experimental.json", merged_tracks_payload)
    write_json(experiment_dir / "04B_tracks_experimental_report.json", merged_tracks_report)
    write_json(experiment_dir / "04B_tracks_experimental_compatibility.json", compatibility)

    _build_contact_sheets(experiment_dir, merged_tracks_payload)
    _build_preview_video(experiment_dir, dense_yolo_payload, merged_tracks_payload)

    baseline_track_count = int(baseline["vehicle_tracks"])
    baseline_single_frame = int(baseline["track_quality_counts"].get("single_frame", 0))
    experimental_vehicle_tracks = [
        track for track in list(merged_tracks_payload.get("tracks", [])) if track.get("track_type") == "vehicle"
    ]
    raw_vehicle_tracks = [
        track for track in list(raw_tracks_payload.get("tracks", [])) if track.get("track_type") == "vehicle"
    ]
    longest_vehicle_track = max(experimental_vehicle_tracks, key=lambda item: item["detection_count"], default=None)
    dominant_coverage = (
        round((float(longest_vehicle_track["detection_count"]) / float(dense_yolo_payload["class_counts"].get("car", 0))) * 100.0, 3)
        if longest_vehicle_track and float(dense_yolo_payload["class_counts"].get("car", 0)) > 0
        else 0.0
    )
    comparison = {
        "video": video_info["video_name"],
        "baseline": {
            "tracking_input_frames": baseline["tracking_input_frames"],
            "vehicle_detections": baseline["vehicle_detections"],
            "raw_track_count": baseline_track_count,
            "good_track_count": int(baseline["track_quality_counts"].get("good", 0)),
            "fragmented_track_count": int(baseline["track_quality_counts"].get("fragmented", 0)),
            "single_frame_track_count": baseline_single_frame,
            "longest_vehicle_track_detections": max((int(track["detection_count"]) for track in read_json(run_dir / "04B_tracks.json")["tracks"] if track["track_type"] == "vehicle"), default=0),
            "longest_vehicle_track_duration_seconds": max((float(track["duration_seconds"]) for track in read_json(run_dir / "04B_tracks.json")["tracks"] if track["track_type"] == "vehicle"), default=0.0),
        },
        "experimental": {
            "tracking_fps": tracking_fps,
            "tracking_input_frames": int(dense_manifest["selected_frame_count"]),
            "vehicle_detections": int(dense_yolo_payload["class_counts"].get("car", 0)),
            "bytetrack_raw_track_count": len(raw_vehicle_tracks),
            "post_merge_track_count": len(experimental_vehicle_tracks),
            "good_track_count": int(merged_tracks_report["track_quality_counts"].get("good", 0)),
            "fragmented_track_count": int(merged_tracks_report["track_quality_counts"].get("fragmented", 0)),
            "single_frame_track_count": int(merged_tracks_report["track_quality_counts"].get("single_frame", 0)),
            "longest_vehicle_track_detections": int(longest_vehicle_track["detection_count"]) if longest_vehicle_track else 0,
            "longest_vehicle_track_duration_seconds": float(longest_vehicle_track["duration_seconds"]) if longest_vehicle_track else 0.0,
        },
        "improvement": {
            "track_count_reduction_percent": round(((baseline_track_count - len(experimental_vehicle_tracks)) / baseline_track_count) * 100.0, 3) if baseline_track_count else 0.0,
            "single_frame_reduction_percent": round(((baseline_single_frame - int(merged_tracks_report["track_quality_counts"].get("single_frame", 0))) / baseline_single_frame) * 100.0, 3) if baseline_single_frame else 0.0,
            "dominant_track_detection_coverage": dominant_coverage,
            "estimated_id_switch_reduction": max(0, baseline_track_count - len(experimental_vehicle_tracks)),
        },
    }
    write_json(experiment_dir / "tracking_baseline_comparison.json", comparison)

    experiment_report = {
        "status": "success",
        "run_directory": str(run_dir),
        "tracking_fps": tracking_fps,
        "baseline_analysis": baseline,
        "dense_frame_count": dense_manifest["selected_frame_count"],
        "dense_yolo_total_detections": dense_yolo_payload["total_detections"],
        "dense_yolo_class_counts": dense_yolo_payload["class_counts"],
        "bytetrack_meta": bytetrack_meta,
        "raw_tracking_report": raw_tracks_report,
        "merged_tracking_report": merged_tracks_report,
        "post_merge_meta": merge_meta,
        "timing_seconds": {
            "dense_yolo_inference": round(dense_yolo_seconds, 3),
            "tracker": round(tracker_seconds, 3),
            "fragment_merging": round(merge_seconds, 3),
            "total_added_processing": round(dense_yolo_seconds + tracker_seconds + merge_seconds, 3),
        },
        "step05_compatibility": compatibility,
        "comparison": comparison,
    }
    write_json(experiment_dir / "tracking_experiment_report.json", experiment_report)
    print(f"tracking_experiment_dir={experiment_dir}")
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
