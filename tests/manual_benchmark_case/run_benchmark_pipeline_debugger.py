from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT_PATH = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from tests.manual_benchmark_case import run_event_candidate_reasoning_benchmark as bench

from app.core.config import PROJECT_ROOT, settings
from app.core.utils import format_timestamp_human
from app.services.event_aggregation import EventAggregationService
from app.services.qwen_vlm_hf import NativeQwenTransformersService
from app.services.vlm_utils import clean_json_response, finalize_frame_metadata


BASE_INPUT_VIDEO = Path(
    os.getenv("BENCHMARK_INPUT_VIDEO", r"C:\Mukul K\test_video\robbery_5mins.mp4")
)
RUNS_ROOT = SCRIPT_PATH.parent / "benchmark_debug_runs"


def _sanitize_name(name: str) -> str:
    cleaned = []
    for ch in name:
        cleaned.append(ch if ch.isalnum() or ch in ("-", "_") else "_")
    return "".join(cleaned).strip("_") or "video"


def _timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return [_jsonable(item) for item in sorted(value, key=lambda item: str(item))]
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _dump_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(data), f, indent=4)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def _resolve_input_video_path(video_path: Path) -> Path:
    if video_path.exists():
        return video_path
    if not video_path.is_absolute():
        candidate = PROJECT_ROOT / video_path
        if candidate.exists():
            return candidate
    return video_path


def _build_run_directories(video_path: Path) -> Dict[str, Path]:
    run_root = RUNS_ROOT / f"{_sanitize_name(video_path.stem)}_{_timestamp_slug()}"
    stage_names = [
        "01_input_video",
        "02_sampled_frames",
        "03_event_candidate_layer",
        "04_candidate_event_clustering",
        "05_keyframe_selection",
        "06_temporal_strip_builder",
        "07_vlm_inputs",
        "08_vlm_raw_metadata",
        "09_metadata_cleanup",
        "10_event_aggregation_inputs",
        "11_benchmark_outputs",
        "12_timing_analysis",
        "logs",
    ]

    dirs = {"run_root": run_root}
    for stage_name in stage_names:
        stage_dir = run_root / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        dirs[stage_name] = stage_dir

    tracking_dir = run_root / "tracking"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    dirs["tracking"] = tracking_dir

    for subdir in (
        dirs["03_event_candidate_layer"] / "selected_frames",
        dirs["03_event_candidate_layer"] / "rejected_frames",
        dirs["05_keyframe_selection"] / "selected_keyframes",
        dirs["05_keyframe_selection"] / "rejected_keyframes",
        dirs["06_temporal_strip_builder"] / "raw_strips",
        dirs["07_vlm_inputs"] / "selected_inputs",
        dirs["08_vlm_raw_metadata"] / "raw_text",
        dirs["11_benchmark_outputs"] / "events",
    ):
        subdir.mkdir(parents=True, exist_ok=True)

    return dirs


def _configure_logging(log_dir: Path) -> logging.Logger:
    logger = logging.getLogger("benchmark_pipeline_debugger")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_dir / "pipeline_debug.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _write_stage_summary(stage_dir: Path, stage_name: str, execution_time_seconds: float, summary: Dict[str, Any]) -> None:
    payload = {
        "stage": stage_name,
        "execution_time_seconds": execution_time_seconds,
        "summary": _jsonable(summary),
    }
    _dump_json(stage_dir / "stage_summary.json", payload)

    lines = [
        f"# {stage_name}",
        "",
        f"- Execution time: `{execution_time_seconds:.3f}s`",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- {key}: `{value}`")
    (stage_dir / "stage_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _copy_video(video_path: Path, stage_dir: Path) -> Path:
    copied = stage_dir / video_path.name
    shutil.copy2(video_path, copied)
    return copied


def _save_run_manifest(run_root: Path, stage_dirs: Dict[str, Path], input_video: Path) -> None:
    manifest = {
        "input_video": str(input_video),
        "run_root": str(run_root),
        "stages": {key: str(path) for key, path in stage_dirs.items() if key != "run_root"},
    }
    _dump_json(run_root / "run_manifest.json", manifest)


def _frame_tuple_manifest(extracted_tuples: List[Tuple[str, str, float, Path]]) -> List[Dict[str, Any]]:
    return [
        {
            "frame_id": frame_id,
            "video_id": video_id,
            "timestamp_seconds": ts,
            "frame_path": str(path),
        }
        for frame_id, video_id, ts, path in extracted_tuples
    ]


def _detection_to_dict(frame_detection: Any) -> Dict[str, Any]:
    return {
        "frame_id": frame_detection.frame_id,
        "video_id": frame_detection.video_id,
        "timestamp_seconds": frame_detection.timestamp_seconds,
        "frame_width": frame_detection.frame_width,
        "frame_height": frame_detection.frame_height,
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
            for det in frame_detection.detections
        ],
    }


def _cluster_frame_lookup(signals: List[Any]) -> Dict[str, Any]:
    return {signal.frame_id: signal for signal in signals}


def _frame_counts_from_signal(signal: Any) -> Dict[str, int]:
    return {str(key): int(value) for key, value in signal.class_counts.items()}


def _reason_flags(signal: Any) -> str:
    return ", ".join(signal.reason_flags) if getattr(signal, "reason_flags", None) else "no_significant_change"


def _reconstruct_keyframe_reasons(cluster: List[Any], selected_ids: Sequence[str]) -> List[Dict[str, Any]]:
    selected_set = set(selected_ids)
    if not cluster:
        return []

    best_index = max(range(len(cluster)), key=lambda idx: cluster[idx].score)
    evenly_spaced = set()
    if len(selected_set) > 0:
        step = (len(cluster) - 1) / max(1, len(selected_set) - 1)
        for slot in range(len(selected_set)):
            evenly_spaced.add(round(slot * step))

    rows: List[Dict[str, Any]] = []
    previous_counts: Dict[str, int] = {}
    previous_tracks: List[int] = []
    for idx, signal in enumerate(cluster):
        reasons: List[str] = []
        if idx == 0:
            reasons.append("first_frame")
        if idx == len(cluster) - 1:
            reasons.append("last_frame")
        if idx == best_index:
            reasons.append("highest_score")
        if signal.class_counts != previous_counts and signal.class_counts:
            reasons.append("state_change")
        if signal.track_ids != previous_tracks and signal.track_ids:
            reasons.append("track_change")
        if signal.class_counts.get("person", 0) > 0 and any(
            signal.class_counts.get(cls, 0) > 0 for cls in {"backpack", "handbag", "suitcase"}
        ):
            reasons.append("interaction_frame")
        if idx in evenly_spaced:
            reasons.append("interval_sampling")
        previous_counts = signal.class_counts
        previous_tracks = signal.track_ids

        rows.append(
            {
                "frame_id": signal.frame_id,
                "timestamp_seconds": signal.timestamp_seconds,
                "selected": signal.frame_id in selected_set,
                "reason": ", ".join(dict.fromkeys(reasons)) if signal.frame_id in selected_set else "lower_priority_than_selected_frames",
                "score": signal.score,
                "track_ids": signal.track_ids,
                "object_counts": _frame_counts_from_signal(signal),
                "candidate_reasons": signal.reason_flags,
            }
        )

    return rows


async def _stage_01_sampled_frames(
    video_path: Path,
    stage_dir: Path,
    logger: logging.Logger,
) -> Tuple[List[Tuple[str, str, float, Path]], Dict[str, Any]]:
    start = time.perf_counter()
    original_frame_root = bench.FRAME_ROOT
    bench.FRAME_ROOT = stage_dir
    try:
        extracted_tuples = bench._extract_one_fps_frames(_sanitize_name(video_path.stem), video_path)
    finally:
        bench.FRAME_ROOT = original_frame_root

    stats = _get_video_stats(video_path)
    manifest = {
        "input_video": str(video_path),
        "total_video_frames": stats["total_frames"],
        "source_fps": stats["fps"],
        "video_duration_seconds": stats["duration_seconds"],
        "sampled_frames": len(extracted_tuples),
        "sampling_rate": "1 fps",
        "frames": _frame_tuple_manifest(extracted_tuples),
    }
    _dump_json(stage_dir / "sampled_frames_manifest.json", manifest)
    _write_stage_summary(
        stage_dir,
        "Stage 01 - Sampled Frames",
        time.perf_counter() - start,
        {
            "input_video": str(video_path),
            "total_video_frames": stats["total_frames"],
            "sampled_frames": len(extracted_tuples),
            "sampling_rate": "1 fps",
            "output_directory": str(stage_dir),
        },
    )

    logger.info(
        f"[stage01] sampled {len(extracted_tuples)} frames from {video_path.name} into {stage_dir}"
    )
    return extracted_tuples, manifest


async def _stage_02_event_candidate_layer(
    extracted_tuples: List[Tuple[str, str, float, Path]],
    stage_dir: Path,
    logger: logging.Logger,
) -> Tuple[List[Any], Dict[str, Any], Dict[str, Any]]:
    start = time.perf_counter()
    frame_detections, tracking_map = bench._run_detection_and_tracking(extracted_tuples)
    signals = bench._build_frame_signals(extracted_tuples, tracking_map)

    candidate_layer_input = {
        "sampled_frames": _frame_tuple_manifest(extracted_tuples),
        "frame_detections": [_detection_to_dict(item) for item in frame_detections],
        "tracking_map": _jsonable(tracking_map),
    }
    _dump_json(stage_dir / "candidate_layer_input.json", candidate_layer_input)

    rows: List[Dict[str, Any]] = []
    selected_dir = stage_dir / "selected_frames"
    rejected_dir = stage_dir / "rejected_frames"

    for frame_id, _video_id, _ts, frame_path in extracted_tuples:
        signal = next(item for item in signals if item.frame_id == frame_id)
        selected = bool(signal.reason_flags)
        reason = _reason_flags(signal)
        selected_payload = {
            "frame_id": signal.frame_id,
            "video_id": signal.video_id,
            "timestamp_seconds": signal.timestamp_seconds,
            "selected": selected,
            "reason": reason,
            "score": signal.score,
            "objects": [
                {
                    "class_name": entity.get("class_name"),
                    "track_id": entity.get("track_id"),
                }
                for entity in signal.tracked_entities
            ],
            "tracks": signal.track_ids,
            "tracked_entities": signal.tracked_entities,
            "object_counts": signal.class_counts,
            "frame_path": str(frame_path),
            "candidate_reasons": signal.reason_flags,
        }
        rows.append(selected_payload)
        dst_dir = selected_dir if selected else rejected_dir
        shutil.copy2(frame_path, dst_dir / frame_path.name)
        logger.info(
            f"[stage02] {frame_id}: {'selected' if selected else 'rejected'} -> {reason}"
        )

    _dump_json(stage_dir / "candidate_frame_signals.json", rows)
    summary = {
        "total_sampled_frames": len(extracted_tuples),
        "selected_frames": sum(1 for row in rows if row["selected"]),
        "rejected_frames": sum(1 for row in rows if not row["selected"]),
        "selection_rate": round((sum(1 for row in rows if row["selected"]) / max(1, len(rows))) * 100.0, 2),
    }
    _write_stage_summary(
        stage_dir,
        "Stage 02 - Event Candidate Layer",
        time.perf_counter() - start,
        summary,
    )
    return signals, candidate_layer_input, summary


async def _stage_03_candidate_event_clustering(
    signals: List[Any],
    stage_dir: Path,
    logger: logging.Logger,
) -> Tuple[List[Any], List[List[Any]]]:
    start = time.perf_counter()
    clusters = bench._cluster_active_frames(signals)
    budgets = bench._allocate_keyframe_budgets(clusters)

    candidate_events = []
    cluster_rows = []
    for idx, (cluster, budget) in enumerate(zip(clusters, budgets), start=1):
        event_type, reason = bench._guess_event_type(cluster)
        selected_frame_ids = bench._select_keyframes(cluster, budget)
        ocr_text = bench._collect_event_ocr(_cluster_frame_lookup(signals), selected_frame_ids)
        event = bench.CandidateEvent(
            event_id=f"cand_evt_{idx:03d}",
            event_type=event_type,
            reason=reason,
            start_seconds=cluster[0].timestamp_seconds,
            end_seconds=cluster[-1].timestamp_seconds,
            frame_ids=[signal.frame_id for signal in cluster],
            selected_frame_ids=selected_frame_ids,
            persons_max=max((signal.class_counts.get("person", 0) for signal in cluster), default=0),
            bags_max=max(
                (
                    max(
                        (signal.class_counts.get(cls, 0) for cls in {"backpack", "handbag", "suitcase"}),
                        default=0,
                    )
                    for signal in cluster
                ),
                default=0,
            ),
            dwell_seconds=cluster[-1].timestamp_seconds - cluster[0].timestamp_seconds,
            track_ids=sorted({track_id for signal in cluster for track_id in signal.track_ids}),
            ocr_text=ocr_text,
        )
        candidate_events.append(event)
        cluster_rows.append(
            {
                "candidate_event_id": event.event_id,
                "event_type": event.event_type,
                "reason": event.reason,
                "start_frame": cluster[0].frame_id,
                "end_frame": cluster[-1].frame_id,
                "start_seconds": event.start_seconds,
                "end_seconds": event.end_seconds,
                "dwell_seconds": event.dwell_seconds,
                "frame_ids": event.frame_ids,
                "selected_frame_ids": event.selected_frame_ids,
                "track_ids": event.track_ids,
                "persons_max": event.persons_max,
                "bags_max": event.bags_max,
                "ocr_text": event.ocr_text,
                "cluster_size": len(cluster),
                "keyframe_budget": budget,
            }
        )

    _dump_json(stage_dir / "candidate_event_catalog.json", cluster_rows)
    _write_stage_summary(
        stage_dir,
        "Stage 03 - Candidate Event Clustering",
        time.perf_counter() - start,
        {
            "cluster_count": len(candidate_events),
            "cluster_sizes": [len(cluster) for cluster in clusters],
            "output_file": str(stage_dir / "candidate_event_catalog.json"),
        },
    )
    logger.info(f"[stage03] built {len(candidate_events)} candidate event cluster(s)")
    return candidate_events, clusters


async def _stage_04_keyframe_selection(
    candidate_events: List[Any],
    clusters: List[List[Any]],
    stage_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    start = time.perf_counter()
    selected_dir = stage_dir / "selected_keyframes"
    rejected_dir = stage_dir / "rejected_keyframes"
    rows: List[Dict[str, Any]] = []

    for event, cluster in zip(candidate_events, clusters):
        frame_rows = _reconstruct_keyframe_reasons(cluster, event.selected_frame_ids)
        rows.append(
            {
                "event": event.event_id,
                "selected_frames": event.selected_frame_ids,
                "rejected_frames": [row["frame_id"] for row in frame_rows if not row["selected"]],
                "selection_reasons": [row["reason"] for row in frame_rows if row["selected"]],
                "frame_decisions": frame_rows,
            }
        )

        selected_set = set(event.selected_frame_ids)
        for signal in cluster:
            source_path = signal.frame_path
            dst_dir = selected_dir if signal.frame_id in selected_set else rejected_dir
            shutil.copy2(source_path, dst_dir / source_path.name)

    _dump_json(stage_dir / "keyframe_selection_debug.json", rows)
    _write_stage_summary(
        stage_dir,
        "Stage 04 - Keyframe Selection",
        time.perf_counter() - start,
        {
            "candidate_events": len(candidate_events),
            "selected_keyframes": sum(len(event.selected_frame_ids) for event in candidate_events),
            "output_file": str(stage_dir / "keyframe_selection_debug.json"),
        },
    )
    logger.info(f"[stage04] selected keyframes for {len(candidate_events)} candidate event(s)")
    return {"keyframe_selection_debug": rows}


async def _stage_05_temporal_strip_builder(
    candidate_events: List[Any],
    signals: List[Any],
    stage_dir: Path,
    logger: logging.Logger,
) -> Tuple[List[Dict[str, Any]], Dict[str, Path]]:
    start = time.perf_counter()
    previous_root = bench.REASONING_INPUT_ROOT
    bench.REASONING_INPUT_ROOT = stage_dir / "raw_strips"
    bench.REASONING_INPUT_ROOT.mkdir(parents=True, exist_ok=True)

    frame_lookup = _cluster_frame_lookup(signals)
    strip_manifest: List[Dict[str, Any]] = []
    strip_map: Dict[str, Path] = {}

    try:
        for event in candidate_events:
            selected_ids = event.selected_frame_ids
            frame_paths = [Path(frame_lookup[frame_id].frame_path) for frame_id in selected_ids]
            labels = []
            for idx, frame_id in enumerate(selected_ids):
                if idx == 0:
                    labels.append("START")
                elif idx == len(selected_ids) - 1:
                    labels.append("END")
                else:
                    labels.append("KEY")
            generated = bench._render_reasoning_strip(
                event.event_id,
                frame_paths,
                labels,
                "debug",
            )
            final_path = stage_dir / f"{event.event_id}_strip.jpg"
            shutil.copy2(generated, final_path)
            strip_map[event.event_id] = final_path
            strip_manifest.append(
                {
                    "event_id": event.event_id,
                    "strip_path": str(final_path),
                    "generated_strip_path": str(generated),
                    "source_frame_ids": selected_ids,
                    "source_frame_paths": [str(path) for path in frame_paths],
                    "labels": labels,
                }
            )
            logger.info(
                f"[stage05] {event.event_id}: strip built from {len(selected_ids)} frame(s) -> {final_path.name}"
            )
    finally:
        bench.REASONING_INPUT_ROOT = previous_root

    _dump_json(stage_dir / "temporal_strip_manifest.json", strip_manifest)
    _write_stage_summary(
        stage_dir,
        "Stage 05 - Temporal Strip Builder",
        time.perf_counter() - start,
        {
            "strip_count": len(strip_manifest),
            "output_directory": str(stage_dir),
        },
    )
    return strip_manifest, strip_map


async def _stage_06_vlm_inputs(
    strip_manifest: List[Dict[str, Any]],
    strip_map: Dict[str, Path],
    stage_dir: Path,
    logger: logging.Logger,
) -> Tuple[List[Dict[str, Any]], Dict[str, Path]]:
    start = time.perf_counter()
    manifest: List[Dict[str, Any]] = []
    input_map: Dict[str, Path] = {}

    for index, row in enumerate(strip_manifest, start=1):
        event_id = row["event_id"]
        source_path = strip_map[event_id]
        vlm_path = stage_dir / f"vlm_input_{index:03d}.jpg"
        shutil.copy2(source_path, vlm_path)
        input_map[event_id] = vlm_path
        manifest.append(
            {
                "event_id": event_id,
                "source_strip_path": str(source_path),
                "vlm_input_path": str(vlm_path),
                "source_frame_ids": row["source_frame_ids"],
            }
        )

    _dump_json(stage_dir / "vlm_input_manifest.json", manifest)
    _write_stage_summary(
        stage_dir,
        "Stage 06 - VLM Inputs",
        time.perf_counter() - start,
        {
            "vlm_input_count": len(manifest),
            "output_directory": str(stage_dir),
        },
    )
    logger.info(f"[stage06] prepared {len(manifest)} VLM input image(s)")
    return manifest, input_map


async def _stage_07_vlm_raw_metadata(
    candidate_events: List[Any],
    vlm_inputs: Dict[str, Path],
    signals: List[Any],
    stage_dir: Path,
    logger: logging.Logger,
) -> Tuple[List[Dict[str, Any]], float]:
    start = time.perf_counter()
    frame_lookup = _cluster_frame_lookup(signals)
    results: List[Dict[str, Any]] = []
    raw_text_parts: List[str] = []
    total_inference_seconds = 0.0

    old_batch_size = settings.BATCH_SIZE
    old_max_new_tokens = settings.QWEN_MAX_NEW_TOKENS
    settings.BATCH_SIZE = 1
    settings.QWEN_MAX_NEW_TOKENS = max(128, old_max_new_tokens)

    try:
        NativeQwenTransformersService.get_runtime()
        for event in candidate_events:
            strip_path = vlm_inputs[event.event_id]
            representative_frame_id = event.selected_frame_ids[len(event.selected_frame_ids) // 2]
            representative_signal = frame_lookup[representative_frame_id]
            context = bench._build_structured_context(event)
            prompt = bench._build_reasoning_prompt(context)

            request_start = time.perf_counter()
            raw_outputs = await NativeQwenTransformersService._async_hf_generate(
                [strip_path],
                [prompt],
            )
            request_seconds = time.perf_counter() - request_start
            total_inference_seconds += request_seconds
            raw_output = raw_outputs[0] if raw_outputs else ""
            raw_row = {
                "event_id": event.event_id,
                "candidate_event_id": event.event_id,
                "frame_id": representative_frame_id,
                "timestamp_seconds": representative_signal.timestamp_seconds,
                "input_image_path": str(strip_path),
                "source_frame_ids": event.selected_frame_ids,
                "prompt": prompt,
                "raw_response": raw_output,
                "per_request_inference_seconds": request_seconds,
                "source": "native_hf",
            }
            results.append(raw_row)

            with open(stage_dir / f"{event.event_id}_raw.json", "w", encoding="utf-8") as f:
                json.dump(_jsonable(raw_row), f, indent=4)
            (stage_dir / "raw_text" / f"{event.event_id}_raw.txt").write_text(raw_output or "", encoding="utf-8")
            raw_text_parts.append(f"### {event.event_id}\n{raw_output}\n")
            logger.info(
                f"[stage07] {event.event_id}: raw VLM response captured in {request_seconds:.2f}s"
            )
    finally:
        settings.BATCH_SIZE = old_batch_size
        settings.QWEN_MAX_NEW_TOKENS = old_max_new_tokens

    (stage_dir / "raw_vlm_response.txt").write_text("\n".join(raw_text_parts), encoding="utf-8")
    _dump_json(stage_dir / "vlm_raw_metadata.json", results)
    _write_stage_summary(
        stage_dir,
        "Stage 07 - VLM Raw Metadata",
        time.perf_counter() - start,
        {
            "raw_responses": len(results),
            "total_inference_seconds": round(total_inference_seconds, 3),
            "output_directory": str(stage_dir),
        },
    )
    return results, total_inference_seconds


async def _stage_08_metadata_cleanup(
    raw_results: List[Dict[str, Any]],
    candidate_events: List[Any],
    signals: List[Any],
    stage_dir: Path,
    logger: logging.Logger,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    start = time.perf_counter()
    frame_lookup = _cluster_frame_lookup(signals)
    cleaned_rows: List[Dict[str, Any]] = []
    diffs: List[Dict[str, Any]] = []

    for raw_item in raw_results:
        event_id = raw_item["event_id"]
        event = next(item for item in candidate_events if item.event_id == event_id)
        representative_frame_id = raw_item["frame_id"]
        representative_signal = frame_lookup[representative_frame_id]

        cleaned_response = ""
        parsed_response: Dict[str, Any] = {}
        try:
            cleaned_response = clean_json_response(raw_item["raw_response"])
            parsed_response = json.loads(cleaned_response) if cleaned_response.strip() else {}
        except Exception as exc:
            logger.warning(f"[stage08] {event_id}: failed to parse cleaned VLM output ({exc})")
            parsed_response = {"raw_response": raw_item["raw_response"]}

        detection_context = {
            "candidate_reasons": event.reason.split(", ") if event.reason else [],
            "detected_objects": [
                {
                    "class_name": entity.get("class_name"),
                    "confidence": entity.get("confidence"),
                    "bbox": entity.get("bbox"),
                }
                for entity in representative_signal.tracked_entities
            ],
            "tracked_entities": representative_signal.tracked_entities,
            "track_ids": representative_signal.track_ids,
            "object_counts": representative_signal.class_counts,
        }

        final_meta = finalize_frame_metadata(
            parsed_raw=parsed_response,
            frame_id=representative_frame_id,
            video_id=str(representative_signal.video_id),
            timestamp_seconds=raw_item["timestamp_seconds"],
            frame_path=Path(representative_signal.frame_path),
            ocr_result={"detected_text": event.ocr_text},
            project_root=PROJECT_ROOT,
            detection_context=detection_context,
        )
        final_dict = final_meta.model_dump()
        final_dict["candidate_event_id"] = event_id
        final_dict["source_frame_ids"] = event.selected_frame_ids
        final_dict["strip_path"] = raw_item["input_image_path"]
        cleaned_rows.append(final_dict)

        diff_row = {
            "event_id": event_id,
            "raw_response": raw_item["raw_response"],
            "cleaned_response": cleaned_response,
            "parsed_response": parsed_response,
            "final_metadata": final_dict,
        }
        diffs.append(diff_row)

        with open(stage_dir / f"{event_id}_final.json", "w", encoding="utf-8") as f:
            json.dump(_jsonable(final_dict), f, indent=4)
        with open(stage_dir / f"{event_id}_raw_vs_cleaned_diff.json", "w", encoding="utf-8") as f:
            json.dump(_jsonable(diff_row), f, indent=4)

        logger.info(f"[stage08] {event_id}: metadata cleaned and finalized")

    _dump_json(stage_dir / "cleaned_metadata.json", cleaned_rows)
    _dump_json(stage_dir / "raw_vs_cleaned_diff.json", diffs)
    _write_stage_summary(
        stage_dir,
        "Stage 08 - Metadata Cleanup",
        time.perf_counter() - start,
        {
            "cleaned_items": len(cleaned_rows),
            "output_directory": str(stage_dir),
        },
    )
    return cleaned_rows, diffs


async def _stage_09_event_aggregation_inputs(
    cleaned_metadata: List[Dict[str, Any]],
    stage_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    start = time.perf_counter()
    payload = {
        "input_count": len(cleaned_metadata),
        "frames_metadata": cleaned_metadata,
    }
    _dump_json(stage_dir / "event_aggregation_input.json", payload)
    _write_stage_summary(
        stage_dir,
        "Stage 09 - Event Aggregation Inputs",
        time.perf_counter() - start,
        {
            "input_items": len(cleaned_metadata),
            "output_file": str(stage_dir / "event_aggregation_input.json"),
        },
    )
    logger.info(f"[stage09] prepared {len(cleaned_metadata)} event aggregation input item(s)")
    return payload


def _aggregate_events_in_debug_space(
    cleaned_metadata: List[Dict[str, Any]],
    stage_dir: Path,
    logger: logging.Logger,
) -> Tuple[List[Dict[str, Any]], Path]:
    if not cleaned_metadata:
        event_catalog_path = stage_dir / "event_catalog.json"
        _dump_json(event_catalog_path, [])
        return [], event_catalog_path

    import app.services.event_aggregation as event_aggregation_module

    original_event_dir = event_aggregation_module.event_dir
    original_write_event_catalog = event_aggregation_module.write_event_catalog
    original_job_status_update = event_aggregation_module.JobStatusService.update

    debug_events_dir = stage_dir / "events"
    debug_events_dir.mkdir(parents=True, exist_ok=True)
    event_catalog_path = stage_dir / "event_catalog.json"

    def _debug_event_dir(video_id: str) -> Path:
        return debug_events_dir / video_id

    def _debug_write_event_catalog(video_id: str, events: List[Dict[str, Any]]) -> Path:
        _dump_json(event_catalog_path, events)
        return event_catalog_path

    def _noop_status_update(*args: Any, **kwargs: Any) -> None:
        return None

    event_aggregation_module.event_dir = _debug_event_dir
    event_aggregation_module.write_event_catalog = _debug_write_event_catalog
    event_aggregation_module.JobStatusService.update = _noop_status_update

    try:
        video_id = str(cleaned_metadata[0].get("video_id", "debug_video"))
        events = EventAggregationService.process_events(video_id, cleaned_metadata)
    finally:
        event_aggregation_module.event_dir = original_event_dir
        event_aggregation_module.write_event_catalog = original_write_event_catalog
        event_aggregation_module.JobStatusService.update = original_job_status_update

    logger.info(f"[event_aggregation] produced {len(events)} event(s)")
    return events, event_catalog_path


async def _stage_10_benchmark_outputs(
    candidate_events: List[Any],
    cleaned_metadata: List[Dict[str, Any]],
    aggregation_events: List[Dict[str, Any]],
    stage_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    start = time.perf_counter()
    summary = {
        "candidate_event_count": len(candidate_events),
        "frames_sent_to_qwen": len(cleaned_metadata),
        "cleaned_metadata_items": len(cleaned_metadata),
        "events_generated": len(aggregation_events),
        "candidate_event_ids": [event.event_id for event in candidate_events],
        "event_ids": [event.get("event_id") for event in aggregation_events],
    }
    _dump_json(stage_dir / "candidate_event_catalog.json", [asdict(event) for event in candidate_events])
    _dump_json(stage_dir / "event_candidate_benchmark_summary.json", summary)

    lines = [
        "# Event Candidate Benchmark Summary",
        "",
        f"- Candidate events: `{summary['candidate_event_count']}`",
        f"- Frames sent to Qwen: `{summary['frames_sent_to_qwen']}`",
        f"- Cleaned metadata rows: `{summary['cleaned_metadata_items']}`",
        f"- Events generated: `{summary['events_generated']}`",
        "",
        "## Event IDs",
        "",
    ]
    for event_id in summary["event_ids"]:
        lines.append(f"- {event_id}")
    (stage_dir / "event_candidate_benchmark_summary.md").write_text("\n".join(lines), encoding="utf-8")
    _write_stage_summary(
        stage_dir,
        "Stage 10 - Benchmark Outputs",
        time.perf_counter() - start,
        summary,
    )
    logger.info(f"[stage10] benchmark outputs written to {stage_dir}")
    return summary


async def _stage_12_timing_analysis(
    timing_map: Dict[str, float],
    stage_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    start = time.perf_counter()
    total = sum(timing_map.values())
    bottleneck_stage = max(timing_map.items(), key=lambda item: item[1])[0] if timing_map else None
    payload = {
        "stages": timing_map,
        "total_pipeline_time": total,
        "bottleneck_stage": bottleneck_stage,
        "stage_percentages": {
            stage: round((seconds / max(total, 1e-9)) * 100.0, 2)
            for stage, seconds in timing_map.items()
        },
    }
    _dump_json(stage_dir / "pipeline_timing.json", payload)

    lines = [
        "# Pipeline Timing Report",
        "",
        f"- Total runtime: `{total:.3f}s`",
        f"- Bottleneck stage: `{bottleneck_stage}`",
        "",
        "## Per-stage timing",
        "",
    ]
    for stage, seconds in timing_map.items():
        pct = payload["stage_percentages"][stage]
        lines.append(f"- {stage}: `{seconds:.3f}s` ({pct:.2f}%)")
    (stage_dir / "pipeline_timing_report.md").write_text("\n".join(lines), encoding="utf-8")
    _write_stage_summary(
        stage_dir,
        "Stage 12 - Timing Analysis",
        time.perf_counter() - start,
        payload,
    )
    logger.info(f"[stage12] timing analysis written to {stage_dir}")
    return payload


async def main(input_path: Optional[Path] = None) -> None:
    video_path = _resolve_input_video_path(input_path or BASE_INPUT_VIDEO)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    stage_dirs = _build_run_directories(video_path)
    logger = _configure_logging(stage_dirs["logs"])
    _save_run_manifest(stage_dirs["run_root"], stage_dirs, video_path)

    old_tracking_debug_dir = os.environ.get("TRACKING_DEBUG_DIR")
    os.environ["TRACKING_DEBUG_DIR"] = str(stage_dirs["tracking"])

    logger.info("Starting isolated benchmark pipeline debugger")
    logger.info(f"Input video: {video_path}")
    logger.info(f"Run root: {stage_dirs['run_root']}")

    try:
        stage_timings: Dict[str, float] = {}
        overall_start = time.perf_counter()

        copied_input = _copy_video(video_path, stage_dirs["01_input_video"])
        video_stats = _get_video_stats(video_path)
        _dump_json(
            stage_dirs["01_input_video"] / "input_video_summary.json",
            {
                "original_video": str(video_path),
                "copied_video": str(copied_input),
                "video_duration_seconds": video_stats["duration_seconds"],
                "total_video_frames": video_stats["total_frames"],
                "source_fps": video_stats["fps"],
            },
        )

        t0 = time.perf_counter()
        extracted_tuples, stage1_manifest = await _stage_01_sampled_frames(
            video_path=video_path,
            stage_dir=stage_dirs["02_sampled_frames"],
            logger=logger,
        )
        stage_timings["frame_extraction"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        signals, candidate_layer_input, stage2_summary = await _stage_02_event_candidate_layer(
            extracted_tuples=extracted_tuples,
            stage_dir=stage_dirs["03_event_candidate_layer"],
            logger=logger,
        )
        stage_timings["event_candidate_layer"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        candidate_events, clusters = await _stage_03_candidate_event_clustering(
            signals=signals,
            stage_dir=stage_dirs["04_candidate_event_clustering"],
            logger=logger,
        )
        stage_timings["candidate_event_clustering"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        keyframe_debug = await _stage_04_keyframe_selection(
            candidate_events=candidate_events,
            clusters=clusters,
            stage_dir=stage_dirs["05_keyframe_selection"],
            logger=logger,
        )
        stage_timings["keyframe_selection"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        strip_manifest, strip_map = await _stage_05_temporal_strip_builder(
            candidate_events=candidate_events,
            signals=signals,
            stage_dir=stage_dirs["06_temporal_strip_builder"],
            logger=logger,
        )
        stage_timings["temporal_strip_builder"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        vlm_manifest, vlm_input_map = await _stage_06_vlm_inputs(
            strip_manifest=strip_manifest,
            strip_map=strip_map,
            stage_dir=stage_dirs["07_vlm_inputs"],
            logger=logger,
        )
        stage_timings["vlm_inputs"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        raw_results, total_vlm_seconds = await _stage_07_vlm_raw_metadata(
            candidate_events=candidate_events,
            vlm_inputs=vlm_input_map,
            signals=signals,
            stage_dir=stage_dirs["08_vlm_raw_metadata"],
            logger=logger,
        )
        stage_timings["vlm_inference"] = max(time.perf_counter() - t0, total_vlm_seconds)

        t0 = time.perf_counter()
        cleaned_metadata, diffs = await _stage_08_metadata_cleanup(
            raw_results=raw_results,
            candidate_events=candidate_events,
            signals=signals,
            stage_dir=stage_dirs["09_metadata_cleanup"],
            logger=logger,
        )
        stage_timings["metadata_cleanup"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        event_agg_payload = await _stage_09_event_aggregation_inputs(
            cleaned_metadata=cleaned_metadata,
            stage_dir=stage_dirs["10_event_aggregation_inputs"],
            logger=logger,
        )
        stage_timings["event_aggregation_inputs"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        aggregation_events, event_catalog_path = _aggregate_events_in_debug_space(
            cleaned_metadata=event_agg_payload["frames_metadata"],
            stage_dir=stage_dirs["11_benchmark_outputs"],
            logger=logger,
        )
        stage10_summary = await _stage_10_benchmark_outputs(
            candidate_events=candidate_events,
            cleaned_metadata=cleaned_metadata,
            aggregation_events=aggregation_events,
            stage_dir=stage_dirs["11_benchmark_outputs"],
            logger=logger,
        )
        stage_timings["benchmark_outputs"] = time.perf_counter() - t0

        total_runtime = time.perf_counter() - overall_start
        stage_timings["total_pipeline_time"] = total_runtime

        timing_payload = await _stage_12_timing_analysis(
            timing_map={k: v for k, v in stage_timings.items() if k != "total_pipeline_time"},
            stage_dir=stage_dirs["12_timing_analysis"],
            logger=logger,
        )

        final_summary = {
            "input_video": str(video_path),
            "run_root": str(stage_dirs["run_root"]),
            "sampled_frames": len(extracted_tuples),
            "candidate_frames": stage2_summary["selected_frames"],
            "candidate_events": len(candidate_events),
            "vlm_inputs": len(vlm_manifest),
            "raw_vlm_rows": len(raw_results),
            "cleaned_metadata_rows": len(cleaned_metadata),
            "events_generated": len(aggregation_events),
            "total_runtime_seconds": total_runtime,
            "stage_timings": stage_timings,
            "timing_report": str(stage_dirs["12_timing_analysis"] / "pipeline_timing_report.md"),
            "event_catalog_path": str(event_catalog_path),
            "tracking_debug_dir": str(stage_dirs["tracking"]),
        }
        _dump_json(stage_dirs["11_benchmark_outputs"] / "event_candidate_benchmark_summary.json", final_summary)

        summary_lines = [
            "# Benchmark Pipeline Debugger Summary",
            "",
            f"- Input video: `{video_path}`",
            f"- Run root: `{stage_dirs['run_root']}`",
            f"- Sampled frames: `{len(extracted_tuples)}`",
            f"- Candidate events: `{len(candidate_events)}`",
            f"- VLM inputs: `{len(vlm_manifest)}`",
            f"- Raw VLM rows: `{len(raw_results)}`",
            f"- Cleaned metadata rows: `{len(cleaned_metadata)}`",
            f"- Events generated: `{len(aggregation_events)}`",
            f"- Total runtime: `{total_runtime:.3f}s`",
            "",
            "## Key Files",
            "",
            f"- Stage 01 manifest: `{stage_dirs['02_sampled_frames'] / 'sampled_frames_manifest.json'}`",
            f"- Stage 03 catalog: `{stage_dirs['04_candidate_event_clustering'] / 'candidate_event_catalog.json'}`",
            f"- Stage 05 strips: `{stage_dirs['06_temporal_strip_builder']}`",
            f"- Stage 07 raw metadata: `{stage_dirs['08_vlm_raw_metadata']}`",
            f"- Stage 08 cleaned metadata: `{stage_dirs['09_metadata_cleanup'] / 'cleaned_metadata.json'}`",
            f"- Stage 09 event aggregation input: `{stage_dirs['10_event_aggregation_inputs'] / 'event_aggregation_input.json'}`",
            f"- Stage 11 benchmark summary: `{stage_dirs['11_benchmark_outputs'] / 'event_candidate_benchmark_summary.json'}`",
            f"- Stage 12 timing report: `{stage_dirs['12_timing_analysis'] / 'pipeline_timing_report.md'}`",
            f"- Tracking debug dir: `{stage_dirs['tracking']}`",
        ]
        (stage_dirs["11_benchmark_outputs"] / "event_candidate_benchmark_summary.md").write_text(
            "\n".join(summary_lines),
            encoding="utf-8",
        )

        logger.info("Benchmark pipeline debugger finished successfully")
        logger.info(json.dumps(final_summary, indent=4))
        print("BENCHMARK_PIPELINE_DEBUGGER_COMPLETE")
        print(json.dumps(final_summary, indent=4))
    finally:
        if old_tracking_debug_dir is None:
            os.environ.pop("TRACKING_DEBUG_DIR", None)
        else:
            os.environ["TRACKING_DEBUG_DIR"] = old_tracking_debug_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark-only pipeline debugger")
    parser.add_argument(
        "--input",
        type=str,
        default=str(BASE_INPUT_VIDEO),
        help="Path to the input video",
    )
    args = parser.parse_args()
    asyncio.run(main(Path(args.input)))
