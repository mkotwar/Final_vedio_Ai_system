from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from .crop_extractor import load_requested_frames, save_frame_and_crop, save_validated_frame_and_crop
from .representative_frame_validator import RepresentativeFrameValidationConfig, validate_representative_observation


def _bbox_area(bbox_xyxy: list[float]) -> float:
    return max(0.0, float(bbox_xyxy[2]) - float(bbox_xyxy[0])) * max(0.0, float(bbox_xyxy[3]) - float(bbox_xyxy[1]))


def _boundary_margin_score(bbox_xyxy: list[float], frame_width: int, frame_height: int) -> float:
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    margins = [
        x1 / max(float(frame_width), 1.0),
        y1 / max(float(frame_height), 1.0),
        (float(frame_width) - x2) / max(float(frame_width), 1.0),
        (float(frame_height) - y2) / max(float(frame_height), 1.0),
    ]
    return max(0.0, min(1.0, min(margins) / 0.08))


def _sharpness_score(frame, bbox_xyxy: list[float]) -> float:
    x1, y1, x2, y2 = [int(round(value)) for value in bbox_xyxy]
    crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return max(0.0, min(1.0, float(variance) / 400.0))


def _brightness_score(frame, bbox_xyxy: list[float]) -> float:
    x1, y1, x2, y2 = [int(round(value)) for value in bbox_xyxy]
    crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    brightness = float(hsv[:, :, 2].mean()) / 255.0
    return max(0.0, 1.0 - abs(brightness - 0.6))


def _plate_candidate_score(frame, bbox_xyxy: list[float], frame_width: int, frame_height: int) -> float:
    area_score = min(1.0, _bbox_area(bbox_xyxy) / max(float(frame_width * frame_height) * 0.08, 1.0))
    lower_visibility = max(0.0, min(1.0, float(bbox_xyxy[3]) / max(float(frame_height), 1.0)))
    return round((0.55 * area_score) + (0.45 * lower_visibility), 6)


def _candidate_score(
    *,
    frame,
    bbox_xyxy: list[float],
    bbox_source: str,
    frame_width: int,
    frame_height: int,
    detector_confidence_hint: float,
    bbox_stability_hint: float,
    is_vehicle: bool,
) -> dict[str, float]:
    object_size_score = min(1.0, _bbox_area(bbox_xyxy) / max((frame_width * frame_height) * 0.10, 1.0))
    boundary_margin = _boundary_margin_score(bbox_xyxy, frame_width, frame_height)
    sharpness = _sharpness_score(frame, bbox_xyxy)
    brightness = _brightness_score(frame, bbox_xyxy)
    source_reliability = 1.0 if str(bbox_source) == "yolo" else 0.65
    occlusion_score = boundary_margin
    crop_score = (
        0.25 * sharpness
        + 0.20 * object_size_score
        + 0.15 * boundary_margin
        + 0.15 * detector_confidence_hint
        + 0.10 * occlusion_score
        + 0.05 * brightness
        + 0.05 * bbox_stability_hint
        + 0.05 * source_reliability
    )
    plate_score = _plate_candidate_score(frame, bbox_xyxy, frame_width, frame_height) if is_vehicle else 0.0
    return {
        "sharpness_score": round(sharpness, 6),
        "object_size_score": round(object_size_score, 6),
        "boundary_margin_score": round(boundary_margin, 6),
        "detector_confidence_score": round(detector_confidence_hint, 6),
        "occlusion_score": round(occlusion_score, 6),
        "brightness_score": round(brightness, 6),
        "bbox_stability_score": round(bbox_stability_hint, 6),
        "source_reliability_score": round(source_reliability, 6),
        "crop_score": round(crop_score, 6),
        "plate_candidate_score": round(plate_score, 6),
    }


def _select_diverse_candidates(candidates: list[dict[str, Any]], *, top_k: int, minimum_gap_seconds: float) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: (-float(item["crop_score"]), float(item["timestamp_seconds"]), int(item["source_frame_index"]))):
        if any(abs(float(candidate["timestamp_seconds"]) - float(existing["timestamp_seconds"])) < minimum_gap_seconds for existing in selected):
            continue
        selected.append(candidate)
        if len(selected) >= top_k:
            break
    return selected


def build_representative_frames(
    *,
    video_path: Path,
    local_objects: list[dict[str, Any]],
    frame_width: int,
    frame_height: int,
    post_tracking_dir: Path,
    top_k: int,
    minimum_gap_seconds: float,
    vehicle_padding: float,
    person_padding: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    all_frame_indexes: list[int] = []
    for local_object in local_objects:
        for item in list(local_object.get("combined_trajectory", [])):
            all_frame_indexes.append(int(item.get("source_frame_index", 0) or 0))
    frames_by_index = load_requested_frames(video_path, all_frame_indexes)
    representative_results: list[dict[str, Any]] = []
    crop_failures: list[dict[str, Any]] = []
    selected_yolo_frames = 0
    selected_kcf_frames = 0

    representative_dir = post_tracking_dir / "representative_frames"
    crop_dir = post_tracking_dir / "best_crops"
    full_frames_dir = post_tracking_dir / "full_frames"

    for local_object in local_objects:
        candidates: list[dict[str, Any]] = []
        trajectory = list(local_object.get("combined_trajectory", []))
        if not trajectory:
            representative_results.append(
                {
                    "camera_id": local_object["camera_id"],
                    "local_object_id": local_object["local_object_id"],
                    "local_object_key": local_object["local_object_key"],
                    "object_family": local_object["object_family"],
                    "final_class": local_object["final_class"],
                    "source_raw_track_ids": list(local_object["source_raw_track_ids"]),
                    "quality_level": local_object["quality_level"],
                    "quality_score": local_object["quality_score"],
                    "representative_frames": {},
                    "downstream_status": "rejected",
                    "warnings": [*list(local_object.get("warnings", [])), "missing_trajectory"],
                }
            )
            continue
        detector_confidence_hint = min(1.0, max(0.25, float(sum(local_object.get("class_votes", {}).values()) / max(len(local_object.get("class_votes", {})), 1)) / 4.0))
        bbox_stability_hint = min(1.0, max(0.0, float(local_object.get("quality_score", 0.0))))
        is_vehicle = str(local_object["object_family"]) == "vehicle"
        for item in trajectory:
            frame_index = int(item.get("source_frame_index", 0) or 0)
            frame = frames_by_index.get(frame_index)
            if frame is None:
                crop_failures.append(
                    {
                        "local_object_key": local_object["local_object_key"],
                        "source_frame_index": frame_index,
                        "reason": "frame_not_decoded",
                    }
                )
                continue
            scores = _candidate_score(
                frame=frame,
                bbox_xyxy=list(item["bbox_xyxy"]),
                bbox_source=str(item.get("bbox_source", "")),
                frame_width=frame_width,
                frame_height=frame_height,
                detector_confidence_hint=detector_confidence_hint if str(item.get("bbox_source")) == "yolo" else detector_confidence_hint * 0.8,
                bbox_stability_hint=bbox_stability_hint,
                is_vehicle=is_vehicle,
            )
            candidates.append(
                {
                    "source_frame_index": frame_index,
                    "timestamp_seconds": round(float(item.get("timestamp_seconds", 0.0) or 0.0), 6),
                    "bbox_xyxy": [float(value) for value in item["bbox_xyxy"]],
                    "bbox_source": str(item.get("bbox_source", "")),
                    **scores,
                }
            )
        selected = _select_diverse_candidates(candidates, top_k=max(1, top_k), minimum_gap_seconds=minimum_gap_seconds)
        representative_frames: dict[str, Any] = {}
        alternatives: list[dict[str, Any]] = []
        plate_candidate_payload: dict[str, Any] | None = None
        for selected_index, candidate in enumerate(selected):
            frame = frames_by_index[int(candidate["source_frame_index"])]
            frame_prefix = f"camera_{local_object['camera_id']}_object_{int(local_object['local_object_id']):06d}"
            full_frame_path = full_frames_dir / f"{frame_prefix}_full_frame_{int(candidate['source_frame_index']):06d}.jpg"
            label_name = "primary" if selected_index == 0 else f"alt{selected_index:02d}"
            crop_path = crop_dir / f"{frame_prefix}_{label_name}_frame_{int(candidate['source_frame_index']):06d}.jpg"
            try:
                saved = save_frame_and_crop(
                    frame=frame,
                    bbox_xyxy=list(candidate["bbox_xyxy"]),
                    frame_output_path=full_frame_path,
                    crop_output_path=crop_path,
                    padding_ratio=vehicle_padding if is_vehicle else person_padding,
                )
            except Exception as exc:
                crop_failures.append(
                    {
                        "local_object_key": local_object["local_object_key"],
                        "source_frame_index": int(candidate["source_frame_index"]),
                        "reason": str(exc),
                    }
                )
                continue
            payload = {
                **candidate,
                "score": candidate["crop_score"],
                "crop_path": str(crop_path),
                "full_frame_path": str(full_frame_path),
                "crop_bbox_xyxy": saved["crop_bbox_xyxy"],
            }
            if candidate["bbox_source"] == "yolo":
                selected_yolo_frames += 1
            else:
                selected_kcf_frames += 1
            if selected_index == 0:
                representative_frames["primary"] = payload
            else:
                alternatives.append(payload)
            if is_vehicle and (plate_candidate_payload is None or float(candidate["plate_candidate_score"]) > float(plate_candidate_payload["plate_candidate_score"])):
                plate_candidate_payload = payload
        if alternatives:
            representative_frames["alternatives"] = alternatives[:2]
        if plate_candidate_payload is not None:
            representative_frames["plate_candidate"] = plate_candidate_payload
        entry_candidate = min(candidates, key=lambda item: (float(item["timestamp_seconds"]), int(item["source_frame_index"])), default=None)
        exit_candidate = max(candidates, key=lambda item: (float(item["timestamp_seconds"]), int(item["source_frame_index"])), default=None)
        if entry_candidate is not None:
            representative_frames["entry_frame"] = entry_candidate
        if exit_candidate is not None:
            representative_frames["exit_frame"] = exit_candidate
        downstream_status = "ready" if "primary" in representative_frames and (bool(local_object.get("confirmed")) or len(list(local_object.get("source_raw_track_ids", []))) > 1) else "fallback"
        if "primary" not in representative_frames:
            downstream_status = "rejected"
        representative_results.append(
            {
                "camera_id": local_object["camera_id"],
                "local_object_id": local_object["local_object_id"],
                "local_object_key": local_object["local_object_key"],
                "object_family": local_object["object_family"],
                "final_class": local_object["final_class"],
                "source_raw_track_ids": list(local_object["source_raw_track_ids"]),
                "quality_level": local_object["quality_level"],
                "quality_score": local_object["quality_score"],
                "representative_frames": representative_frames,
                "downstream_status": downstream_status,
                "warnings": list(local_object.get("warnings", [])),
            }
        )
    report = {
        "status": "success",
        "total_reconciled_objects": len(local_objects),
        "objects_with_primary_crop": len([item for item in representative_results if item.get("representative_frames", {}).get("primary")]),
        "objects_with_three_crops": len([item for item in representative_results if len(list(item.get("representative_frames", {}).get("alternatives", []))) >= 2]),
        "objects_with_plate_candidate": len([item for item in representative_results if item.get("representative_frames", {}).get("plate_candidate")]),
        "objects_with_full_scene_frame": len([item for item in representative_results if item.get("representative_frames", {}).get("primary", {}).get("full_frame_path")]),
        "fallback_objects": len([item for item in representative_results if item.get("downstream_status") == "fallback"]),
        "crop_failures": len(crop_failures),
        "selected_yolo_frames": selected_yolo_frames,
        "selected_kcf_frames": selected_kcf_frames,
        "average_selected_frames_per_object": round(float(sum(1 + len(list(item.get("representative_frames", {}).get("alternatives", []))) for item in representative_results if item.get("representative_frames", {}).get("primary"))) / max(len(representative_results), 1), 6),
        "quality_distribution": dict(sorted(Counter(str(item.get("quality_level", "low")) for item in representative_results).items())),
    }
    return representative_results, report, {"status": "success", "failures": crop_failures}


def build_representative_frames_v2(
    *,
    video_path: Path,
    local_objects: list[dict[str, Any]],
    frame_width: int,
    frame_height: int,
    post_tracking_dir: Path,
    maximum_ready_crop_clipping_ratio: float,
    maximum_fallback_crop_clipping_ratio: float,
    minimum_plate_candidate_score: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    all_frame_indexes: list[int] = []
    for local_object in local_objects:
        for item in list(local_object.get("combined_trajectory", [])):
            all_frame_indexes.append(int(item.get("source_frame_index", 0) or 0))
    frames_by_index = load_requested_frames(video_path, all_frame_indexes)
    config = RepresentativeFrameValidationConfig(
        maximum_ready_crop_clipping_ratio=maximum_ready_crop_clipping_ratio,
        maximum_fallback_crop_clipping_ratio=maximum_fallback_crop_clipping_ratio,
        minimum_plate_candidate_score=minimum_plate_candidate_score,
    )
    representative_results: list[dict[str, Any]] = []
    crop_failures: list[dict[str, Any]] = []
    invalid_crop_candidates: list[dict[str, Any]] = []
    for local_object in local_objects:
        combined = list(local_object.get("combined_trajectory", []))
        candidates: list[dict[str, Any]] = []
        for observation in combined:
            validation = validate_representative_observation(
                observation,
                frame_width=frame_width,
                frame_height=frame_height,
                config=config,
            )
            frame_index = int(observation.get("source_frame_index", 0) or 0)
            frame = frames_by_index.get(frame_index)
            if frame is None:
                crop_failures.append({"local_object_key": local_object["local_object_key"], "source_frame_index": frame_index, "reason": "frame_not_decoded"})
                continue
            if not validation["identity_crop_eligible"]:
                invalid_crop_candidates.append(
                    {
                        "local_object_key": local_object["local_object_key"],
                        "source_frame_index": frame_index,
                        "bbox_xyxy": list(observation.get("bbox_xyxy", [])),
                        **validation,
                    }
                )
                continue
            scores = _candidate_score(
                frame=frame,
                bbox_xyxy=list(observation["bbox_xyxy"]),
                bbox_source=str(observation.get("bbox_source", "")),
                frame_width=frame_width,
                frame_height=frame_height,
                detector_confidence_hint=float(validation["effective_detector_support_score"]),
                bbox_stability_hint=min(1.0, max(0.0, float(local_object.get("quality_score", 0.0) or 0.0))),
                is_vehicle=str(local_object.get("object_family")) == "vehicle",
            )
            priority = 0
            if str(observation.get("bbox_source")) == "yolo" and str(observation.get("observation_validity")) == "valid":
                priority = 3
            elif str(observation.get("observation_validity")) == "supported":
                priority = 2
            else:
                priority = 1
            candidates.append(
                {
                    **dict(observation),
                    **validation,
                    **scores,
                    "priority": priority,
                }
            )
        ordered = sorted(
            candidates,
            key=lambda item: (
                -int(item["priority"]),
                -float(item["crop_score"]),
                float(item["seconds_since_last_yolo"]),
                float(item["timestamp_seconds"]),
            ),
        )
        representative_frames: dict[str, Any] = {}
        alternatives: list[dict[str, Any]] = []
        plate_candidate_payload: dict[str, Any] | None = None
        for index, candidate in enumerate(ordered[:3]):
            frame_index = int(candidate["source_frame_index"])
            frame = frames_by_index[frame_index]
            frame_prefix = f"camera_{local_object['camera_id']}_object_{int(local_object['local_object_id']):06d}"
            full_frame_path = post_tracking_dir / "full_frames" / f"{frame_prefix}_full_frame_{frame_index:06d}.jpg"
            crop_path = post_tracking_dir / "best_crops" / f"{frame_prefix}_{'primary' if index == 0 else f'alt{index:02d}'}_frame_{frame_index:06d}.jpg"
            try:
                saved = save_validated_frame_and_crop(
                    frame=frame,
                    bbox_xyxy=list(candidate["bbox_xyxy"]),
                    frame_output_path=full_frame_path,
                    crop_output_path=crop_path,
                    padding_ratio=0.10 if str(local_object.get("object_family")) == "vehicle" else 0.12,
                    identity_crop_eligible=bool(candidate["identity_crop_eligible"]),
                    frame_width=frame_width,
                    frame_height=frame_height,
                    config=config,
                )
            except Exception as exc:
                crop_failures.append({"local_object_key": local_object["local_object_key"], "source_frame_index": frame_index, "reason": str(exc)})
                continue
            payload = {
                "source_frame_index": frame_index,
                "timestamp_seconds": round(float(candidate["timestamp_seconds"]), 6),
                "bbox_xyxy": [float(value) for value in candidate["bbox_xyxy"]],
                "bbox_source": str(candidate.get("bbox_source", "")),
                "observation_validity": str(candidate.get("observation_validity", "")),
                "eligibility_reasons": list(candidate.get("eligibility_reasons", [])),
                "seconds_since_last_yolo": round(float(candidate.get("seconds_since_last_yolo", 0.0) or 0.0), 6),
                "effective_detector_support_score": round(float(candidate.get("effective_detector_support_score", 0.0) or 0.0), 6),
                "clipping_ratio": round(float(candidate.get("clipping_ratio", 0.0) or 0.0), 6),
                "identity_crop_eligible": bool(candidate.get("identity_crop_eligible", False)),
                "plate_crop_eligible": bool(candidate.get("plate_crop_eligible", False)),
                "frozen_segment": "frozen_kcf_box" in list(candidate.get("drift_segment_flags", [])),
                "boundary_stuck_segment": "boundary_stuck_box" in list(candidate.get("drift_segment_flags", [])),
                "score": round(float(candidate["crop_score"]), 6),
                "plate_candidate_score": round(float(candidate["plate_candidate_score"]), 6),
                "crop_path": str(crop_path),
                "full_frame_path": str(full_frame_path),
                "crop_bbox_xyxy": list(saved["crop_bbox_xyxy"]),
                "crop_status": saved["crop_status"],
                "crop_content_valid": bool(saved["crop_content_valid"]),
            }
            if index == 0:
                representative_frames["primary"] = payload
            else:
                alternatives.append(payload)
            if payload["plate_crop_eligible"] and payload["plate_candidate_score"] >= minimum_plate_candidate_score:
                if plate_candidate_payload is None or float(payload["plate_candidate_score"]) > float(plate_candidate_payload["plate_candidate_score"]):
                    plate_candidate_payload = payload
        if alternatives:
            representative_frames["alternatives"] = alternatives
        if plate_candidate_payload is not None:
            representative_frames["plate_candidate"] = plate_candidate_payload
        downstream_status = str(local_object.get("downstream_status", "manual_review"))
        if "primary" not in representative_frames:
            downstream_status = "rejected"
        representative_results.append(
            {
                "camera_id": local_object["camera_id"],
                "local_object_id": local_object["local_object_id"],
                "local_object_key": local_object["local_object_key"],
                "object_family": local_object["object_family"],
                "final_class": local_object["final_class"],
                "source_raw_track_ids": list(local_object["source_raw_track_ids"]),
                "quality_level": local_object["quality_level"],
                "quality_score": local_object["quality_score"],
                "representative_frames": representative_frames,
                "downstream_status": downstream_status,
                "warnings": list(local_object.get("warnings", [])),
            }
        )
    report = {
        "status": "success",
        "total_reconciled_objects": len(local_objects),
        "valid_primary_crops": len([item for item in representative_results if item.get("representative_frames", {}).get("primary", {}).get("crop_content_valid")]),
        "fallback_crops": len([item for item in representative_results if item.get("representative_frames", {}).get("primary", {}).get("crop_status") == "fallback"]),
        "objects_with_plate_candidate": len([item for item in representative_results if item.get("representative_frames", {}).get("plate_candidate")]),
        "invalid_crop_candidate_count": len(invalid_crop_candidates),
        "crop_failures": len(crop_failures),
    }
    return representative_results, report, {"status": "success", "failures": crop_failures}, invalid_crop_candidates
