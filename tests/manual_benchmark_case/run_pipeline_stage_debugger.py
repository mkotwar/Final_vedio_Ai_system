from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

# Keep config loading predictable for local debug runs.
os.environ.setdefault("DEBUG", "false")
os.environ.setdefault("ENV", "testing")

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT_PATH = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from app.core.config import PROJECT_ROOT, settings
from app.core.utils import format_timestamp_human
from app.schemas.frame import FrameRichMetadata
from app.services.event_aggregation import EventAggregationService
from app.services.frame import FrameExtractionService
from app.services.mock_vlm import MockVLMService
from app.services.motion_window_service import MotionWindowService
from app.services.object_detection.detector import ObjectDetector
from app.services.object_detection.schemas import Detection, FrameDetection
from app.services.event_candidate_selector import EventCandidateSelector
from app.services.object_tracker import ObjectTrackerService
from app.services.ocr import OCRService
from app.services.qwen_vlm_hf import NativeQwenTransformersService
from app.services.vlm_utils import clean_json_response, finalize_frame_metadata
from app.services import vlm_prompt as vlm_prompt_module


BASE_INPUT_VIDEO = Path(
    os.getenv("BENCHMARK_INPUT_VIDEO", r"C:\Mukul K\test_video\robbery_5mins.mp4")
)

VLM_FRAME_METADATA_PROMPT = getattr(vlm_prompt_module, "SHARED_VLM_FRAME_METADATA_PROMPT", None)
if VLM_FRAME_METADATA_PROMPT is None:
    VLM_FRAME_METADATA_PROMPT = getattr(vlm_prompt_module, "VLM_FRAME_METADATA_PROMPT", None)
if VLM_FRAME_METADATA_PROMPT is None:
    raise ImportError(
        "No supported VLM prompt symbol found in app.services.vlm_prompt. "
        "Expected SHARED_VLM_FRAME_METADATA_PROMPT or VLM_FRAME_METADATA_PROMPT."
    )


@dataclass
class SampledFrame:
    frame_id: str
    video_id: str
    timestamp_seconds: float
    frame_path: Path


def _sanitize_name(name: str) -> str:
    cleaned = []
    for ch in name:
        cleaned.append(ch if ch.isalnum() or ch in ("-", "_") else "_")
    return "".join(cleaned).strip("_") or "video"


def _timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_input_video_path(video_path: Path) -> Path:
    if video_path.exists():
        return video_path

    if not video_path.is_absolute():
        input_root = SCRIPT_PATH.parent / "debug_runs"
        candidate = input_root / video_path
        if candidate.exists():
            return candidate

    return video_path


def _build_run_directories(video_path: Path) -> Dict[str, Path]:
    run_name = f"{_sanitize_name(video_path.stem)}_{_timestamp_slug()}"
    run_root = SCRIPT_PATH.parent / "debug_runs" / run_name

    stage_names = [
        "01_input_video",
        "02_sampled_frames",
        "03_motion_windows",
        "04_adaptive_sampling",
        "05_object_detection",
        "06_tracking",
        "06_ocr",
        "07_candidate_selection",
        "08_vlm_inputs",
        "09_vlm_raw_metadata",
        "10_metadata_postprocessed",
        "11_event_aggregation",
        "12_final_reports",
        "logs",
    ]

    paths = {"run_root": run_root}
    for stage_name in stage_names:
        paths[stage_name] = run_root / stage_name
        paths[stage_name].mkdir(parents=True, exist_ok=True)

    for subdir in (
        paths["04_adaptive_sampling"] / "selected_frames",
        paths["04_adaptive_sampling"] / "rejected_frames",
        paths["07_candidate_selection"] / "selected_candidates",
        paths["07_candidate_selection"] / "rejected_candidates",
    ):
        subdir.mkdir(parents=True, exist_ok=True)

    return paths


def _configure_logging(log_dir: Path) -> logging.Logger:
    log_file = log_dir / "pipeline_debug.log"
    logger = logging.getLogger("pipeline_stage_debugger")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def _copy_input_video(video_path: Path, stage_dir: Path) -> Path:
    copied = stage_dir / video_path.name
    shutil.copy2(video_path, copied)
    return copied


def _get_video_stats(video_path: Path) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_seconds = (total_frames / fps) if fps > 0 else 0.0
    cap.release()
    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration_seconds": duration_seconds,
    }


def _extract_sampled_frames(video_id: str, video_path: Path, out_dir: Path, logger: logging.Logger) -> List[SampledFrame]:
    sampled_root = out_dir
    sampled_root.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0.0:
        fps = 30.0

    frame_interval = max(1, int(round(fps)))
    current_raw_frame = 0
    frame_idx = 1
    sampled_frames: List[SampledFrame] = []

    try:
        while True:
            if current_raw_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_raw_frame)

            success, frame = cap.read()
            if not success:
                break

            timestamp_seconds = current_raw_frame / fps
            frame_id = f"{video_id}_f{frame_idx:04d}"
            frame_path = sampled_root / f"f{frame_idx:04d}.jpg"

            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"Failed to save sampled frame: {frame_id}")

            sampled_frames.append(
                SampledFrame(
                    frame_id=frame_id,
                    video_id=video_id,
                    timestamp_seconds=timestamp_seconds,
                    frame_path=frame_path,
                )
            )
            logger.info(f"Sampled frame {frame_id} at {timestamp_seconds:.2f}s -> {frame_path.name}")
            frame_idx += 1
            current_raw_frame += frame_interval
    finally:
        cap.release()

    return sampled_frames


def _save_sampling_summary(stage_dir: Path, sampled_frames: List[SampledFrame], video_stats: Dict[str, Any]) -> None:
    summary = {
        "total_video_frames": video_stats["total_frames"],
        "source_fps": video_stats["fps"],
        "video_duration_seconds": video_stats["duration_seconds"],
        "sampled_frames": len(sampled_frames),
        "sampling_rate": "1 fps",
    }
    with open(stage_dir / "sampling_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)


def _detect_motion_windows(video_path: Path, stage_dir: Path, video_stats: Dict[str, Any]) -> List[Dict[str, float]]:
    windows = MotionWindowService.detect_motion_windows(video_path)
    payload = [{"start": start, "end": end} for start, end in windows]
    with open(stage_dir / "motion_windows.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    _render_motion_window_visualization(stage_dir / "motion_window_visualization.jpg", payload, video_stats)
    return payload


def _render_motion_window_visualization(out_path: Path, motion_windows: List[Dict[str, float]], video_stats: Dict[str, Any]) -> None:
    width, height = 1600, 240
    image = None
    try:
        import numpy as np

        image = np.full((height, width, 3), 245, dtype=np.uint8)
        cv2.rectangle(image, (40, 60), (width - 40, 150), (220, 220, 220), thickness=-1)
        cv2.rectangle(image, (40, 60), (width - 40, 150), (180, 180, 180), thickness=1)

        total_duration = max(1.0, float(video_stats.get("duration_seconds", 0.0) or 0.0))
        for item in motion_windows:
            start = float(item["start"])
            end = float(item["end"])
            x1 = int(40 + ((width - 80) * (start / total_duration)))
            x2 = int(40 + ((width - 80) * (end / total_duration)))
            cv2.rectangle(image, (x1, 60), (x2, 150), (80, 170, 80), thickness=-1)

        cv2.putText(image, "Motion windows", (40, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(
            image,
            f"Duration: {total_duration:.1f}s | Windows: {len(motion_windows)}",
            (40, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (40, 40, 40),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(out_path), image)
    finally:
        if image is None:
            out_path.touch()


def _frame_in_windows(timestamp_seconds: float, windows: List[Dict[str, float]], pad_seconds: float = 0.0) -> bool:
    return any((window["start"] - pad_seconds) <= timestamp_seconds <= (window["end"] + pad_seconds) for window in windows)


def _compute_adaptive_sampling(
    sampled_frames: List[SampledFrame],
    motion_windows: List[Dict[str, float]],
    stage_dir: Path,
    logger: logging.Logger,
) -> Tuple[List[SampledFrame], List[Dict[str, Any]]]:
    selected_frames: List[SampledFrame] = []
    decisions: List[Dict[str, Any]] = []

    last_sent_frame: Optional[Any] = None
    last_retained_timestamp = 0.0
    last_out_of_window_timestamp: Optional[float] = None

    for frame in sampled_frames:
        should_send = True
        reasons: List[str] = []
        hist_diff = 0.0
        ssim_diff = 0.0
        motion_score = 0.0
        passed_ssim = False
        passed_histogram = False
        passed_motion = False
        passed_temporal = False
        passed_baseline = False
        passed_coverage = False

        in_motion_window = _frame_in_windows(frame.timestamp_seconds, motion_windows, pad_seconds=0.0)
        if not in_motion_window:
            baseline_interval = max(1.0, float(settings.OUT_OF_WINDOW_BASELINE_SECONDS))
            if last_out_of_window_timestamp is None or (frame.timestamp_seconds - last_out_of_window_timestamp) >= baseline_interval:
                passed_baseline = True
                last_out_of_window_timestamp = frame.timestamp_seconds
                reasons.append("baseline_outside_motion_window")
                logger.info(
                    f"[adaptive] keep {frame.frame_id}: outside motion windows but baseline slot opened"
                )
            else:
                should_send = False
                reasons.append("outside_motion_window_no_baseline_slot")

        if (
            should_send
            and settings.ENABLE_ADAPTIVE_SAMPLING
            and last_sent_frame is not None
            and not in_motion_window
            and not passed_baseline
        ):
            hist_diff, ssim_diff, motion_score = FrameExtractionService.compute_similarity_metrics(
                cv2.imread(str(frame.frame_path)),
                last_sent_frame,
            )
            ssim_score = 1.0 - ssim_diff
            passed_ssim = ssim_score < settings.SSIM_THRESHOLD
            passed_histogram = hist_diff > settings.HISTOGRAM_THRESHOLD
            passed_motion = motion_score > settings.MOTION_THRESHOLD

            if passed_ssim or passed_histogram or passed_motion:
                reasons.append("scene_change_or_motion")
                if passed_ssim:
                    reasons.append("ssim_below_threshold")
                if passed_histogram:
                    reasons.append("histogram_changed")
                if passed_motion:
                    reasons.append("motion_score_above_threshold")
            else:
                if (frame.timestamp_seconds - last_retained_timestamp) >= settings.TEMPORAL_INTERVAL_SECONDS:
                    passed_temporal = True
                    reasons.append("temporal_safeguard")
                else:
                    should_send = False
                    reasons.append("duplicate_frame")

        if not should_send and (frame.timestamp_seconds - last_retained_timestamp) >= settings.MAX_FRAME_GAP_SECONDS:
            should_send = True
            passed_coverage = True
            reasons.append("coverage_safeguard")

        if should_send:
            last_retained_timestamp = frame.timestamp_seconds
            last_sent_frame = cv2.imread(str(frame.frame_path))
            selected_frames.append(frame)
            logger.info(
                f"[adaptive] keep {frame.frame_id}: {', '.join(reasons) if reasons else 'initial_frame'}"
            )
        else:
            logger.info(
                f"[adaptive] drop {frame.frame_id}: {', '.join(reasons) if reasons else 'duplicate_frame'}"
            )

        decisions.append(
            {
                "frame_id": frame.frame_id,
                "timestamp_seconds": frame.timestamp_seconds,
                "selected": should_send,
                "reason": ", ".join(reasons) if reasons else "initial_frame",
                "in_motion_window": in_motion_window,
                "metrics": {
                    "hist_diff": hist_diff,
                    "ssim_diff": ssim_diff,
                    "motion_score": motion_score,
                },
                "thresholds": {
                    "ssim": settings.SSIM_THRESHOLD,
                    "histogram": settings.HISTOGRAM_THRESHOLD,
                    "motion": settings.MOTION_THRESHOLD,
                    "temporal_interval_seconds": settings.TEMPORAL_INTERVAL_SECONDS,
                    "max_frame_gap_seconds": settings.MAX_FRAME_GAP_SECONDS,
                    "out_of_window_baseline_seconds": settings.OUT_OF_WINDOW_BASELINE_SECONDS,
                },
                "passed_ssim": passed_ssim,
                "passed_histogram": passed_histogram,
                "passed_motion": passed_motion,
                "passed_temporal": passed_temporal,
                "passed_baseline": passed_baseline,
                "passed_coverage_safeguard": passed_coverage,
            }
        )

        target_dir = stage_dir / "selected_frames" if should_send else stage_dir / "rejected_frames"
        shutil.copy2(frame.frame_path, target_dir / frame.frame_path.name)

    with open(stage_dir / "adaptive_sampling_decisions.json", "w", encoding="utf-8") as f:
        json.dump(decisions, f, indent=4)

    summary = {
        "sampled_frames": len(sampled_frames),
        "selected_frames": len(selected_frames),
        "rejected_frames": len(sampled_frames) - len(selected_frames),
        "selection_rate": round((len(selected_frames) / max(1, len(sampled_frames))) * 100.0, 2),
    }
    with open(stage_dir / "adaptive_sampling_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    return selected_frames, decisions


def _detect_objects(
    selected_frames: List[SampledFrame],
    stage_dir: Path,
    logger: logging.Logger,
) -> Tuple[List[FrameDetection], List[Dict[str, Any]]]:
    detector = ObjectDetector()
    detector._initialize_model()

    frame_detections: List[FrameDetection] = []
    results: List[Dict[str, Any]] = []

    for frame in selected_frames:
        image = cv2.imread(str(frame.frame_path))
        if image is None:
            logger.warning(f"[object_detection] could not read {frame.frame_path}")
            continue

        raw_results = detector._model(str(frame.frame_path))
        detection_list: List[Detection] = []
        frame_height, frame_width = image.shape[:2]
        if raw_results and len(raw_results) > 0:
            result = raw_results[0]
            boxes = result.boxes
            names = result.names
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    xyxy = detector._to_list(box.xyxy[0])
                    conf = float(detector._to_list(box.conf)[0])
                    cls_id = int(detector._to_list(box.cls)[0])
                    class_name = names.get(cls_id, str(cls_id))
                    x1, y1, x2, y2 = xyxy
                    detection_list.append(
                        Detection(
                            class_id=cls_id,
                            class_name=class_name,
                            confidence=conf,
                            bbox=xyxy,
                            center_x=x1 + (x2 - x1) / 2.0,
                            center_y=y1 + (y2 - y1) / 2.0,
                            width=x2 - x1,
                            height=y2 - y1,
                        )
                    )

        frame_detection = FrameDetection(
            frame_id=frame.frame_id,
            video_id=frame.video_id,
            timestamp_seconds=frame.timestamp_seconds,
            frame_width=frame_width,
            frame_height=frame_height,
            frame_path=str(frame.frame_path),
            detections=detection_list,
        )
        frame_detections.append(frame_detection)

        annotated = image.copy()
        for idx, detection in enumerate(detection_list, start=1):
            x1, y1, x2, y2 = map(int, detection.bbox)
            color = (0, 200, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                f"{idx}:{detection.class_name} {detection.confidence:.2f}",
                (x1, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

        annotated_path = stage_dir / f"{frame.frame_path.stem}_detected.jpg"
        cv2.imwrite(str(annotated_path), annotated)

        result_row = {
            "frame_id": frame.frame_id,
            "timestamp_seconds": frame.timestamp_seconds,
            "frame_path": str(frame.frame_path),
            "annotated_path": str(annotated_path),
            "detection_count": len(detection_list),
            "detections": [
                {
                    "class_id": det.class_id,
                    "class_name": det.class_name,
                    "confidence": det.confidence,
                    "bbox": det.bbox,
                    "center_x": det.center_x,
                    "center_y": det.center_y,
                    "width": det.width,
                    "height": det.height,
                }
                for det in detection_list
            ],
        }
        results.append(result_row)
        logger.info(
            f"[object_detection] {frame.frame_id}: {len(detection_list)} detections -> {annotated_path.name}"
        )

    with open(stage_dir / "object_detection_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    return frame_detections, results


def _track_objects(frame_detections: List[FrameDetection], stage_dir: Path, logger: logging.Logger) -> Dict[str, Dict[str, Any]]:
    tracking_map = ObjectTrackerService.track_frames(frame_detections, debug_output_dir=stage_dir)
    serializable = []
    for frame_detection in frame_detections:
        tracked = tracking_map.get(frame_detection.frame_id, {})
        serializable.append(
            {
                "frame_id": frame_detection.frame_id,
                "timestamp_seconds": frame_detection.timestamp_seconds,
                "tracked_entities": tracked.get("tracked_entities", []),
                "track_ids": tracked.get("track_ids", []),
                "class_counts": tracked.get("class_counts", {}),
                "new_track_count": tracked.get("new_track_count", 0),
                "ended_track_count": tracked.get("ended_track_count", 0),
            }
        )
        logger.info(
            f"[tracking] {frame_detection.frame_id}: new={tracked.get('new_track_count', 0)} "
            f"ended={tracked.get('ended_track_count', 0)} tracks={tracked.get('track_ids', [])}"
        )

    with open(stage_dir / "tracking_results.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=4)

    return tracking_map


def _run_ocr(selected_frames: List[SampledFrame], stage_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    ocr_results: Dict[str, Any] = {}
    for frame in selected_frames:
        result = OCRService.extract_text(frame.frame_path)
        ocr_results[frame.frame_id] = result
        logger.info(
            f"[ocr] {frame.frame_id}: {len(result.get('detected_text', []))} text items"
        )

    with open(stage_dir / "ocr_results.json", "w", encoding="utf-8") as f:
        json.dump(ocr_results, f, indent=4)

    return ocr_results


def _build_selection_manifest(
    selected_frames: List[SampledFrame],
    selected_frame_detections: List[FrameDetection],
    tracking_map: Dict[str, Dict[str, Any]],
    motion_windows: List[Dict[str, float]],
    stage_dir: Path,
    logger: logging.Logger,
) -> Tuple[Dict[str, Dict[str, Any]], List[SampledFrame], List[SampledFrame]]:
    selection_map = EventCandidateSelector.select(
        extracted_tuples=[
            (frame.frame_id, frame.video_id, frame.timestamp_seconds, frame.frame_path)
            for frame in selected_frames
        ],
        frame_detections=selected_frame_detections,
        tracking_map=tracking_map,
        motion_windows=[(window["start"], window["end"]) for window in motion_windows],
    )

    candidate_rows: List[Dict[str, Any]] = []
    vlm_selected: List[SampledFrame] = []
    rejected: List[SampledFrame] = []

    selected_dir = stage_dir / "selected_candidates"
    rejected_dir = stage_dir / "rejected_candidates"

    for frame in selected_frames:
        selection = dict(selection_map.get(frame.frame_id, {}))
        selection["video_id"] = frame.video_id
        selection["frame_path"] = str(frame.frame_path)
        selection["timestamp_seconds"] = frame.timestamp_seconds
        selection_map[frame.frame_id] = selection
        selected_for_vlm = bool(selection.get("selected"))
        candidate_reasons = selection.get("candidate_reasons", [])
        reason = ", ".join(candidate_reasons) if candidate_reasons else "no_significant_change"
        row = {
            "frame_id": frame.frame_id,
            "timestamp_seconds": frame.timestamp_seconds,
            "selected_for_vlm": selected_for_vlm,
            "reason": reason,
            "candidate_reasons": candidate_reasons,
            "detection_count": selection.get("detection_count", 0),
            "track_ids": selection.get("track_ids", []),
            "object_counts": selection.get("object_counts", {}),
            "detected_objects": selection.get("detected_objects", []),
            "tracked_entities": selection.get("tracked_entities", []),
            "selection_payload": selection,
        }
        candidate_rows.append(row)

        dst_dir = selected_dir if selected_for_vlm else rejected_dir
        shutil.copy2(frame.frame_path, dst_dir / frame.frame_path.name)

        if selected_for_vlm:
            vlm_selected.append(frame)
            logger.info(f"[candidate_selection] keep {frame.frame_id}: {reason}")
        else:
            rejected.append(frame)
            logger.info(f"[candidate_selection] drop {frame.frame_id}: {reason}")

    with open(stage_dir / "candidate_selection.json", "w", encoding="utf-8") as f:
        json.dump(candidate_rows, f, indent=4)

    summary = {
        "total_selected_frames": len(selected_frames),
        "selected_for_vlm": len(vlm_selected),
        "rejected_for_vlm": len(rejected),
        "selection_rate": round((len(vlm_selected) / max(1, len(selected_frames))) * 100.0, 2),
    }
    with open(stage_dir / "candidate_selection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    return selection_map, vlm_selected, rejected


def _build_prompt(frame: SampledFrame, selection: Dict[str, Any], ocr_result: Dict[str, Any]) -> str:
    ocr_text = ", ".join(ocr_result.get("detected_text", [])[:8]) if ocr_result else ""
    detected_objects = selection.get("detected_objects", [])
    object_names = ", ".join(det.get("class_name", "unknown") for det in detected_objects[:10])
    reasons = ", ".join(selection.get("candidate_reasons", []))
    parts = [
        VLM_FRAME_METADATA_PROMPT,
        "",
        "Debug context:",
        f"- frame_id: {frame.frame_id}",
        f"- timestamp_seconds: {frame.timestamp_seconds:.2f}",
        f"- selection_reasons: {reasons or 'none'}",
        f"- detector_hints: {object_names or 'none'}",
        f"- ocr_text: {ocr_text or 'none'}",
        "Return strict JSON only.",
    ]
    return "\n".join(parts)


async def _run_vlm(
    selected_frames: List[SampledFrame],
    selection_map: Dict[str, Dict[str, Any]],
    ocr_results: Dict[str, Any],
    stage_dir: Path,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    if not selected_frames:
        with open(stage_dir / "vlm_batch_manifest.json", "w", encoding="utf-8") as f:
            json.dump([], f, indent=4)
        return []

    manifest: List[Dict[str, Any]] = []
    raw_results: List[Dict[str, Any]] = []

    use_mock = settings.MOCK_MODEL or settings.VLM_ENGINE_TYPE == "mock" or os.getenv("BENCHMARK_USE_MOCK_VLM", "").lower() in {"1", "true", "yes"}
    if not use_mock:
        NativeQwenTransformersService.get_runtime()

    batch_size = max(1, int(settings.BATCH_SIZE))

    for index in range(0, len(selected_frames), batch_size):
        batch = selected_frames[index:index + batch_size]
        image_paths = [frame.frame_path for frame in batch]
        prompts = [
            _build_prompt(frame, selection_map.get(frame.frame_id, {}), ocr_results.get(frame.frame_id, {}))
            for frame in batch
        ]

        if use_mock:
            batch_results = await MockVLMService.generate_metadata_batch(
                [(frame.frame_id, frame.video_id, frame.timestamp_seconds, frame.frame_path) for frame in batch]
            )
            for frame, prompt, (rich_meta, timings) in zip(batch, prompts, batch_results):
                raw_json = rich_meta.model_dump_json(indent=4)
                raw_results.append(
                    {
                        "frame_id": frame.frame_id,
                        "timestamp_seconds": frame.timestamp_seconds,
                        "input_image_path": str(frame.frame_path),
                        "prompt": prompt,
                        "raw_response": raw_json,
                        "source": "mock",
                    }
                )
                manifest.append(
                    {
                        "frame_id": frame.frame_id,
                        "input_image_path": str(frame.frame_path),
                        "vlm_input_path": str(frame.frame_path),
                        "prompt": prompt,
                        "batch_index": index,
                    }
                )
        else:
            raw_outputs = await NativeQwenTransformersService._async_hf_generate(image_paths, prompts)
            for frame, prompt, raw_output in zip(batch, prompts, raw_outputs):
                raw_results.append(
                    {
                        "frame_id": frame.frame_id,
                        "timestamp_seconds": frame.timestamp_seconds,
                        "input_image_path": str(frame.frame_path),
                        "prompt": prompt,
                        "raw_response": raw_output,
                        "source": "native_hf",
                    }
                )
                manifest.append(
                    {
                        "frame_id": frame.frame_id,
                        "input_image_path": str(frame.frame_path),
                        "vlm_input_path": str(frame.frame_path),
                        "prompt": prompt,
                        "batch_index": index,
                    }
                )

        logger.info(
            f"[vlm] processed batch {index // batch_size + 1}: {len(batch)} frame(s)"
        )

    with open(stage_dir / "vlm_batch_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)
    with open(stage_dir / "vlm_raw_metadata.json", "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=4)

    return raw_results


def _postprocess_metadata(
    raw_results: List[Dict[str, Any]],
    selection_map: Dict[str, Dict[str, Any]],
    ocr_results: Dict[str, Any],
    stage_dir: Path,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    final_results: List[Dict[str, Any]] = []

    for item in raw_results:
        frame_id = item["frame_id"]
        raw_response = item["raw_response"]
        cleaned = clean_json_response(raw_response)
        parsed_raw = json.loads(cleaned)

        selection = selection_map.get(frame_id, {})
        frame_path = Path(item["input_image_path"])
        timestamp_seconds = float(item["timestamp_seconds"])
        video_id = selection.get("video_id", "debug_video")

        rich_meta = finalize_frame_metadata(
            parsed_raw=parsed_raw,
            frame_id=frame_id,
            video_id=video_id,
            timestamp_seconds=timestamp_seconds,
            frame_path=frame_path,
            ocr_result=ocr_results.get(frame_id, {}),
            project_root=PROJECT_ROOT,
            detection_context=selection,
        )
        final_dict = rich_meta.model_dump()
        final_results.append(
            {
                "frame_id": frame_id,
                "raw_response": raw_response,
                "cleaned_response": cleaned,
                "parsed_response": parsed_raw,
                "final_metadata": final_dict,
            }
        )
        with open(stage_dir / f"{frame_id}_final.json", "w", encoding="utf-8") as f:
            json.dump(final_dict, f, indent=4)
        logger.info(f"[postprocess] {frame_id}: schema repaired and finalized")

    with open(stage_dir / "postprocessed_metadata.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4)

    return final_results


def _aggregate_events(
    final_results: List[Dict[str, Any]],
    stage_dir: Path,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    if not final_results:
        with open(stage_dir / "event_catalog.json", "w", encoding="utf-8") as f:
            json.dump([], f, indent=4)
        with open(stage_dir / "event_grouping_debug.json", "w", encoding="utf-8") as f:
            json.dump([], f, indent=4)
        return []

    frames_metadata = [item["final_metadata"] for item in final_results]

    import app.services.event_aggregation as event_aggregation_module

    original_event_dir = event_aggregation_module.event_dir
    original_write_event_catalog = event_aggregation_module.write_event_catalog
    original_job_status_update = event_aggregation_module.JobStatusService.update

    debug_events_dir = stage_dir / "events"
    debug_events_dir.mkdir(parents=True, exist_ok=True)
    debug_catalog_path = stage_dir / "event_catalog.json"

    def _debug_event_dir(video_id: str) -> Path:
        return debug_events_dir

    def _debug_write_event_catalog(video_id: str, events: List[Dict[str, Any]]) -> Path:
        with open(debug_catalog_path, "w", encoding="utf-8") as f:
            json.dump(events, f, indent=4)
        return debug_catalog_path

    def _noop_status_update(*args: Any, **kwargs: Any) -> None:
        return None

    event_aggregation_module.event_dir = _debug_event_dir
    event_aggregation_module.write_event_catalog = _debug_write_event_catalog
    event_aggregation_module.JobStatusService.update = _noop_status_update

    try:
        video_id = str(frames_metadata[0].get("video_id", "debug_video"))
        events = EventAggregationService.process_events(video_id, frames_metadata)
    finally:
        event_aggregation_module.event_dir = original_event_dir
        event_aggregation_module.write_event_catalog = original_write_event_catalog
        event_aggregation_module.JobStatusService.update = original_job_status_update

    debug_rows: List[Dict[str, Any]] = []
    frame_lookup = {item["frame_id"]: item["final_metadata"] for item in final_results}
    for event in events:
        source_frames = event.get("source_frames", [])
        activities = event.get("activities", [])
        objects = event.get("objects", [])
        common_activity = activities[0] if activities else "unknown"
        common_object = objects[0].get("subtype") if objects else "unknown"
        debug_rows.append(
            {
                "event_id": event.get("event_id"),
                "frames": source_frames,
                "reason_grouped": f"shared {common_object} + {common_activity} continuity",
                "event_type": event.get("event_type"),
                "duration_seconds": event.get("duration_seconds"),
                "confidence": event.get("confidence"),
                "behavioral_flags": event.get("behavioral_flags", []),
                "frame_summaries": [
                    {
                        "frame_id": frame_id,
                        "timestamp_seconds": frame_lookup.get(frame_id, {}).get("timestamp_seconds"),
                        "activities": frame_lookup.get(frame_id, {}).get("activities", []),
                        "scene_type": frame_lookup.get(frame_id, {}).get("scene_type"),
                        "candidate_reasons": frame_lookup.get(frame_id, {}).get("candidate_reasons", []),
                    }
                    for frame_id in source_frames
                ],
            }
        )

    with open(stage_dir / "event_grouping_debug.json", "w", encoding="utf-8") as f:
        json.dump(debug_rows, f, indent=4)

    logger.info(f"[event_aggregation] produced {len(events)} event(s)")
    return events


def _write_final_reports(
    run_root: Path,
    video_path: Path,
    video_stats: Dict[str, Any],
    sampled_frames: List[SampledFrame],
    adaptive_selected_frames: List[SampledFrame],
    candidate_rows: List[Dict[str, Any]],
    vlm_results: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    stage_dirs: Dict[str, Path],
) -> Dict[str, Any]:
    selected_for_vlm = [row for row in candidate_rows if row["selected_for_vlm"]]
    stats = {
        "input_video": str(video_path),
        "run_root": str(run_root),
        "total_video_frames": video_stats["total_frames"],
        "video_duration_seconds": video_stats["duration_seconds"],
        "sampled_frames": len(sampled_frames),
        "motion_windows": len(json.loads((stage_dirs["03_motion_windows"] / "motion_windows.json").read_text(encoding="utf-8"))),
        "adaptive_selected_frames": len(adaptive_selected_frames),
        "candidate_frames_selected_for_vlm": len(selected_for_vlm),
        "vlm_frames": len(vlm_results),
        "events_generated": len(events),
        "sampling_reduction_pct": round((1.0 - (len(sampled_frames) / max(1, video_stats["total_frames"]))) * 100.0, 2),
        "adaptive_reduction_pct": round((1.0 - (len(adaptive_selected_frames) / max(1, len(sampled_frames)))) * 100.0, 2),
        "candidate_reduction_pct": round((1.0 - (len(selected_for_vlm) / max(1, len(adaptive_selected_frames)))) * 100.0, 2),
        "vlm_reduction_pct": round((1.0 - (len(vlm_results) / max(1, len(selected_for_vlm)))) * 100.0, 2),
    }

    with open(stage_dirs["12_final_reports"] / "statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    summary_lines = [
        "# Pipeline Stage Debugger Summary",
        "",
        f"- Run folder: `{run_root}`",
        f"- Input video: `{video_path}`",
        f"- Total video frames: `{stats['total_video_frames']}`",
        f"- Sampled frames: `{stats['sampled_frames']}`",
        f"- Motion windows: `{stats['motion_windows']}`",
        f"- Adaptive selected frames: `{stats['adaptive_selected_frames']}`",
        f"- Candidate frames selected for VLM: `{stats['candidate_frames_selected_for_vlm']}`",
        f"- VLM frames processed: `{stats['vlm_frames']}`",
        f"- Events generated: `{stats['events_generated']}`",
        "",
        "## Reduction",
        f"- Sampling reduction: `{stats['sampling_reduction_pct']}%`",
        f"- Adaptive reduction: `{stats['adaptive_reduction_pct']}%`",
        f"- Candidate reduction: `{stats['candidate_reduction_pct']}%`",
        f"- VLM reduction: `{stats['vlm_reduction_pct']}%`",
    ]
    with open(stage_dirs["12_final_reports"] / "pipeline_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    return stats


def _save_run_manifest(run_root: Path, stage_dirs: Dict[str, Path], input_video: Path) -> None:
    manifest = {
        "input_video": str(input_video),
        "run_root": str(run_root),
        "stages": {key: str(path) for key, path in stage_dirs.items() if key != "run_root"},
    }
    with open(run_root / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)


async def main(input_path: Optional[Path] = None) -> None:
    video_path = _resolve_input_video_path(input_path or BASE_INPUT_VIDEO)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    stage_dirs = _build_run_directories(video_path)
    logger = _configure_logging(stage_dirs["logs"])
    _save_run_manifest(stage_dirs["run_root"], stage_dirs, video_path)

    logger.info("Starting isolated pipeline stage debugger")
    logger.info(f"Input video: {video_path}")
    logger.info(f"Output run root: {stage_dirs['run_root']}")

    copied_input = _copy_input_video(video_path, stage_dirs["01_input_video"])
    video_stats = _get_video_stats(video_path)

    with open(stage_dirs["01_input_video"] / "input_video_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "original_video": str(video_path),
                "copied_video": str(copied_input),
                "video_duration_seconds": video_stats["duration_seconds"],
                "total_video_frames": video_stats["total_frames"],
                "source_fps": video_stats["fps"],
            },
            f,
            indent=4,
        )

    sampled_frames = _extract_sampled_frames(
        video_id=_sanitize_name(video_path.stem),
        video_path=video_path,
        out_dir=stage_dirs["02_sampled_frames"],
        logger=logger,
    )
    _save_sampling_summary(stage_dirs["02_sampled_frames"], sampled_frames, video_stats)

    motion_windows = _detect_motion_windows(video_path, stage_dirs["03_motion_windows"], video_stats)

    adaptive_selected_frames, adaptive_decisions = _compute_adaptive_sampling(
        sampled_frames=sampled_frames,
        motion_windows=motion_windows,
        stage_dir=stage_dirs["04_adaptive_sampling"],
        logger=logger,
    )

    frame_detections, detection_results = _detect_objects(
        selected_frames=adaptive_selected_frames,
        stage_dir=stage_dirs["05_object_detection"],
        logger=logger,
    )

    tracking_map = _track_objects(
        frame_detections=frame_detections,
        stage_dir=stage_dirs["06_tracking"],
        logger=logger,
    )

    ocr_results = _run_ocr(
        selected_frames=adaptive_selected_frames,
        stage_dir=stage_dirs["06_ocr"],
        logger=logger,
    )

    selection_map, vlm_selected_frames, rejected_frames = _build_selection_manifest(
        selected_frames=adaptive_selected_frames,
        selected_frame_detections=frame_detections,
        tracking_map=tracking_map,
        motion_windows=motion_windows,
        stage_dir=stage_dirs["07_candidate_selection"],
        logger=logger,
    )

    with open(stage_dirs["08_vlm_inputs"] / "vlm_batch_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "frame_id": frame.frame_id,
                    "source_image_path": str(frame.frame_path),
                    "vlm_input_path": str(stage_dirs["08_vlm_inputs"] / f"vlm_input_{index:03d}.jpg"),
                    "reason": ", ".join(selection_map.get(frame.frame_id, {}).get("candidate_reasons", []))
                    or "no_significant_change",
                }
                for index, frame in enumerate(vlm_selected_frames, start=1)
            ],
            f,
            indent=4,
        )

    for index, frame in enumerate(vlm_selected_frames, start=1):
        shutil.copy2(frame.frame_path, stage_dirs["08_vlm_inputs"] / f"vlm_input_{index:03d}.jpg")

    raw_results = await _run_vlm(
        selected_frames=vlm_selected_frames,
        selection_map=selection_map,
        ocr_results=ocr_results,
        stage_dir=stage_dirs["09_vlm_raw_metadata"],
        logger=logger,
    )

    postprocessed_results = _postprocess_metadata(
        raw_results=raw_results,
        selection_map=selection_map,
        ocr_results=ocr_results,
        stage_dir=stage_dirs["10_metadata_postprocessed"],
        logger=logger,
    )

    events = _aggregate_events(
        final_results=postprocessed_results,
        stage_dir=stage_dirs["11_event_aggregation"],
        logger=logger,
    )

    stats = _write_final_reports(
        run_root=stage_dirs["run_root"],
        video_path=video_path,
        video_stats=video_stats,
        sampled_frames=sampled_frames,
        adaptive_selected_frames=adaptive_selected_frames,
        candidate_rows=json.loads((stage_dirs["07_candidate_selection"] / "candidate_selection.json").read_text(encoding="utf-8")),
        vlm_results=raw_results,
        events=events,
        stage_dirs=stage_dirs,
    )

    logger.info("Pipeline stage debugger finished successfully")
    logger.info(json.dumps(stats, indent=4))
    print("PIPELINE_STAGE_DEBUGGER_COMPLETE")
    print(json.dumps(stats, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Isolated benchmark pipeline stage debugger")
    parser.add_argument(
        "--input",
        type=str,
        default=str(BASE_INPUT_VIDEO),
        help="Path to the input video",
    )
    args = parser.parse_args()
    asyncio.run(main(Path(args.input)))
