import asyncio
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from qwen_vl_utils import process_vision_info

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT_PATH = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from app.core.config import PROJECT_ROOT, settings
from app.core.utils import format_timestamp_human
from app.services.object_detection.detector import ObjectDetector
from app.services.object_tracker import ObjectTrackerService
from app.services.ocr import OCRService
from app.services.qwen_vlm_hf import NativeQwenTransformersService
from app.services.vlm_utils import clean_json_response
from app.services import vlm_prompt as vlm_prompt_module


VLM_FRAME_METADATA_PROMPT = getattr(vlm_prompt_module, "SHARED_VLM_FRAME_METADATA_PROMPT", None)
if VLM_FRAME_METADATA_PROMPT is None:
    VLM_FRAME_METADATA_PROMPT = getattr(vlm_prompt_module, "VLM_FRAME_METADATA_PROMPT", None)
if VLM_FRAME_METADATA_PROMPT is None:
    raise ImportError(
        "No supported VLM prompt symbol found in app.services.vlm_prompt. "
        "Expected SHARED_VLM_FRAME_METADATA_PROMPT or VLM_FRAME_METADATA_PROMPT."
    )


BENCHMARK_VIDEO_ID = os.getenv("BENCHMARK_VIDEO_ID", "11111111-2222-4333-8444-777777777777")
CASE_ROOT = Path(os.getenv("BENCHMARK_CASE_ROOT", str(PROJECT_ROOT / "tests" / "manual_benchmark_case")))
DATA_ROOT = CASE_ROOT / "data"
INPUT_ROOT = DATA_ROOT / "input"
OUTPUT_ROOT = DATA_ROOT / "output"
FRAME_ROOT = PROJECT_ROOT / "data" / "frames" / BENCHMARK_VIDEO_ID
REASONING_INPUT_ROOT = OUTPUT_ROOT / "reasoning_inputs"


def _resolve_input_video_path(video_path: Path) -> Path:
    if video_path.exists():
        return video_path

    if not video_path.is_absolute():
        input_candidate = INPUT_ROOT / video_path
        if input_candidate.exists():
            return input_candidate

        input_name_candidate = INPUT_ROOT / video_path.name
        if input_name_candidate.exists():
            return input_name_candidate

    return video_path


INPUT_VIDEO_PATH = _resolve_input_video_path(
    Path(os.getenv("BENCHMARK_INPUT_VIDEO", r"C:\Mukul K\test_video\V_ai_test_2min.mp4"))
)

SUMMARY_JSON_PATH = OUTPUT_ROOT / "event_candidate_benchmark_summary.json"
SUMMARY_MD_PATH = OUTPUT_ROOT / "event_candidate_benchmark_summary.md"
TIMELINE_JSON_PATH = OUTPUT_ROOT / "event_candidate_timeline.json"
PROMPT_EXAMPLES_PATH = OUTPUT_ROOT / "reasoning_prompt_examples.json"
INPUT_COPY_PATH = INPUT_ROOT / INPUT_VIDEO_PATH.name

FRAME_GAP_SECONDS = 4.0
MAX_CLUSTER_DURATION_SECONDS = float(os.getenv("MAX_CLUSTER_DURATION_SECONDS", "15.0"))
TARGET_KEYFRAME_BUDGET = int(os.getenv("TARGET_KEYFRAME_BUDGET", "24"))
MAX_KEYFRAMES_PER_EVENT = int(os.getenv("MAX_KEYFRAMES_PER_EVENT", "8"))
PERIODIC_SAFETY_SECONDS = 10.0
BAG_CLASSES = {"backpack", "handbag", "suitcase"}
DYNAMIC_CLASSES = {
    "person",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "backpack",
    "handbag",
    "suitcase",
    "cell phone",
    "book",
    "bottle",
    "cup",
}
PROMPT_MAX_NEW_TOKENS = 64


@dataclass(frozen=True)
class BenchmarkVariant:
    name: str
    image_layout: str
    max_new_tokens: int
    batch_size: int


BENCHMARK_VARIANTS = [
    BenchmarkVariant(name="strip_tokens150_batch1", image_layout="strip", max_new_tokens=350, batch_size=1),
    BenchmarkVariant(name="strip_tokens150_batch4", image_layout="strip", max_new_tokens=350, batch_size=4),
    BenchmarkVariant(name="single_peak_tokens150_batch1", image_layout="single_peak", max_new_tokens=350, batch_size=1),
    BenchmarkVariant(name="single_peak_tokens150_batch4", image_layout="single_peak", max_new_tokens=350, batch_size=4),
]

BENCHMARK_VARIANT_BY_NAME = {variant.name: variant for variant in BENCHMARK_VARIANTS}
DEFAULT_BENCHMARK_MODE_NAMES = ("candidate_only",)
DEFAULT_BENCHMARK_VARIANT_NAMES = ("strip_tokens150_batch4",)


def _parse_csv_env(value: str | None, default_values: Tuple[str, ...]) -> List[str]:
    if not value:
        return list(default_values)
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    return parsed or list(default_values)


def _resolve_selected_modes() -> List[str]:
    selected = _parse_csv_env(os.getenv("BENCHMARK_MODES"), DEFAULT_BENCHMARK_MODE_NAMES)
    valid_modes = {"candidate_only", "candidate_plus_periodic10s"}
    unknown = [mode for mode in selected if mode not in valid_modes]
    if unknown:
        raise ValueError(f"Unknown benchmark mode(s): {unknown}. Valid modes: {sorted(valid_modes)}")
    return selected


def _resolve_selected_variants() -> List[BenchmarkVariant]:
    selected_names = _parse_csv_env(os.getenv("BENCHMARK_VARIANTS"), DEFAULT_BENCHMARK_VARIANT_NAMES)
    unknown = [name for name in selected_names if name not in BENCHMARK_VARIANT_BY_NAME]
    if unknown:
        raise ValueError(
            f"Unknown benchmark variant(s): {unknown}. "
            f"Valid variants: {sorted(BENCHMARK_VARIANT_BY_NAME.keys())}"
        )
    return [BENCHMARK_VARIANT_BY_NAME[name] for name in selected_names]


@dataclass
class FrameSignal:
    frame_id: str
    video_id: str
    timestamp_seconds: float
    frame_path: Path
    track_ids: List[int]
    class_counts: Dict[str, int]
    tracked_entities: List[Dict[str, Any]]
    reason_flags: List[str]
    score: float


@dataclass
class CandidateEvent:
    event_id: str
    event_type: str
    reason: str
    start_seconds: float
    end_seconds: float
    frame_ids: List[str]
    selected_frame_ids: List[str]
    persons_max: int
    bags_max: int
    dwell_seconds: float
    track_ids: List[int]
    ocr_text: List[str]


@dataclass
class ReasoningJob:
    mode: str
    variant_name: str
    event_id: str
    image_path: Path
    image_paths: List[Path]
    selected_frame_ids: List[str]
    image_layout: str
    max_new_tokens: int
    batch_size: int
    structured_context: Dict[str, Any]
    prompt: str
    periodic: bool


def _ensure_dirs() -> None:
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REASONING_INPUT_ROOT.mkdir(parents=True, exist_ok=True)


def _copy_input_video() -> None:
    if not INPUT_VIDEO_PATH.exists():
        raise FileNotFoundError(f"Input video not found: {INPUT_VIDEO_PATH}")
    if INPUT_VIDEO_PATH.resolve() == INPUT_COPY_PATH.resolve():
        return
    shutil.copy2(INPUT_VIDEO_PATH, INPUT_COPY_PATH)


def _safe_remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


def _clean_previous_outputs() -> None:
    _safe_remove(FRAME_ROOT)
    _safe_remove(REASONING_INPUT_ROOT)
    for path in (SUMMARY_JSON_PATH, SUMMARY_MD_PATH, TIMELINE_JSON_PATH, PROMPT_EXAMPLES_PATH):
        _safe_remove(path)


def _get_video_duration_seconds(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    if fps <= 0.0:
        return 0.0
    return float(total_frames / fps)


def _extract_one_fps_frames(video_id: str, video_path: Path) -> List[Tuple[str, str, float, Path]]:
    FRAME_ROOT.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0.0:
        fps = 30.0
    frame_interval = max(1, int(round(fps)))
    current_raw_frame = 0
    frame_idx = 1
    extracted_tuples: List[Tuple[str, str, float, Path]] = []

    try:
        while True:
            if current_raw_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_raw_frame)
            success, frame = cap.read()
            if not success:
                break

            second = current_raw_frame / fps
            frame_id = f"{video_id}_f{frame_idx:04d}"
            frame_path = FRAME_ROOT / f"frame_{frame_idx:04d}.jpg"
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"Failed to save frame {frame_id}")

            extracted_tuples.append((frame_id, video_id, second, frame_path))
            frame_idx += 1
            current_raw_frame += frame_interval
    finally:
        cap.release()

    return extracted_tuples


def _run_detection_and_tracking(
    extracted_tuples: List[Tuple[str, str, float, Path]]
) -> Tuple[List[Any], Dict[str, Dict[str, Any]]]:
    detector = ObjectDetector()
    frame_detections = [
        detector.detect_frame(path, frame_id, video_id, ts)
        for frame_id, video_id, ts, path in extracted_tuples
    ]
    tracking_map = ObjectTrackerService.track_frames(frame_detections)
    return frame_detections, tracking_map


def _build_frame_signals(
    extracted_tuples: List[Tuple[str, str, float, Path]],
    tracking_map: Dict[str, Dict[str, Any]],
) -> List[FrameSignal]:
    signals: List[FrameSignal] = []
    previous_tracks: List[int] = []
    previous_counts: Dict[str, int] = {}

    for frame_id, video_id, ts, frame_path in extracted_tuples:
        tracking = tracking_map.get(frame_id, {})
        tracked_entities = [
            entity for entity in tracking.get("tracked_entities", [])
            if str(entity.get("class_name", "")).lower() in DYNAMIC_CLASSES
        ]
        track_ids = [int(entity.get("track_id")) for entity in tracked_entities]
        class_counts: Dict[str, int] = {}
        for entity in tracked_entities:
            class_name = str(entity.get("class_name", "")).lower()
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        reason_flags: List[str] = []
        score = 0.0
        if tracked_entities:
            reason_flags.append("dynamic_object_detected")
            score += 2.0
        if tracking.get("new_track_count", 0) > 0 and track_ids:
            reason_flags.append("new_track")
            score += 3.0
        if tracking.get("ended_track_count", 0) > 0 and previous_tracks:
            reason_flags.append("track_ended")
            score += 3.0
        if class_counts != previous_counts and class_counts:
            reason_flags.append("dynamic_count_changed")
            score += 2.0
        if track_ids != previous_tracks and track_ids:
            reason_flags.append("dynamic_track_set_changed")
            score += 2.0

        signals.append(
            FrameSignal(
                frame_id=frame_id,
                video_id=video_id,
                timestamp_seconds=ts,
                frame_path=frame_path,
                track_ids=track_ids,
                class_counts=class_counts,
                tracked_entities=tracked_entities,
                reason_flags=reason_flags,
                score=score,
            )
        )
        previous_tracks = track_ids
        previous_counts = class_counts

    return signals


def _cluster_active_frames(signals: List[FrameSignal]) -> List[List[FrameSignal]]:
    active = [signal for signal in signals if signal.reason_flags]
    if not active:
        return []

    clusters: List[List[FrameSignal]] = []
    current_cluster = [active[0]]
    for signal in active[1:]:
        gap_seconds = signal.timestamp_seconds - current_cluster[-1].timestamp_seconds
        span_seconds = signal.timestamp_seconds - current_cluster[0].timestamp_seconds
        if gap_seconds <= FRAME_GAP_SECONDS and span_seconds <= MAX_CLUSTER_DURATION_SECONDS:
            current_cluster.append(signal)
        else:
            clusters.append(current_cluster)
            current_cluster = [signal]
    clusters.append(current_cluster)
    return clusters


def _class_count(cluster: List[FrameSignal], class_name: str) -> int:
    return max((signal.class_counts.get(class_name, 0) for signal in cluster), default=0)


def _allocate_keyframe_budgets(clusters: List[List[FrameSignal]]) -> List[int]:
    durations = [max(1.0, cluster[-1].timestamp_seconds - cluster[0].timestamp_seconds) for cluster in clusters]
    total_duration = sum(durations) or 1.0
    budgets = []
    for cluster, duration in zip(clusters, durations):
        proportional = max(1, int(math.floor(TARGET_KEYFRAME_BUDGET * duration / total_duration)))
        budgets.append(min(len(cluster), MAX_KEYFRAMES_PER_EVENT, proportional))
    return budgets


def _select_keyframes(cluster: List[FrameSignal], target_count: int | None = None) -> List[str]:
    if not cluster:
        return []
    target_count = min(len(cluster), MAX_KEYFRAMES_PER_EVENT, max(1, target_count or MAX_KEYFRAMES_PER_EVENT))

    chosen_indices = {0, len(cluster) - 1}
    best_index = max(range(len(cluster)), key=lambda idx: cluster[idx].score)
    chosen_indices.add(best_index)

    previous_counts: Dict[str, int] = {}
    previous_tracks: List[int] = []
    state_change_indices: List[int] = []
    interaction_indices: List[int] = []

    for idx, signal in enumerate(cluster):
        if signal.class_counts != previous_counts and signal.class_counts:
            state_change_indices.append(idx)
        if signal.track_ids != previous_tracks and signal.track_ids:
            state_change_indices.append(idx)
        if signal.class_counts.get("person", 0) > 0 and any(signal.class_counts.get(cls, 0) > 0 for cls in BAG_CLASSES):
            interaction_indices.append(idx)
        previous_counts = signal.class_counts
        previous_tracks = signal.track_ids

    if state_change_indices:
        chosen_indices.add(state_change_indices[0])
        chosen_indices.add(state_change_indices[-1])
    if interaction_indices:
        chosen_indices.add(interaction_indices[len(interaction_indices) // 2])

    if target_count > len(chosen_indices):
        step = (len(cluster) - 1) / max(1, target_count - 1)
        for slot in range(target_count):
            chosen_indices.add(round(slot * step))
            if len(chosen_indices) >= target_count:
                break

    if len(chosen_indices) > MAX_KEYFRAMES_PER_EVENT:
        required = {0, len(cluster) - 1, best_index}
        optional = sorted(chosen_indices - required, key=lambda idx: cluster[idx].score, reverse=True)
        chosen_indices = set(required)
        for idx in optional:
            if len(chosen_indices) >= MAX_KEYFRAMES_PER_EVENT:
                break
            chosen_indices.add(idx)

    ordered = sorted(chosen_indices)
    return [cluster[idx].frame_id for idx in ordered]


def _guess_event_type(cluster: List[FrameSignal]) -> Tuple[str, str]:
    persons_max = _class_count(cluster, "person")
    bags_max = max((_class_count(cluster, cls) for cls in BAG_CLASSES), default=0)
    dwell_seconds = cluster[-1].timestamp_seconds - cluster[0].timestamp_seconds

    if bags_max > 0 and persons_max > 0:
        return "person_object_interaction", "Non-person object appeared while person activity was present."
    if bags_max > 0 and dwell_seconds >= 8.0:
        return "possible_abandoned_object", "Object persisted across multiple seconds."
    if persons_max >= 2 and dwell_seconds >= 10.0:
        return "group_activity", "Multiple people persisted in the monitored area."
    if persons_max >= 1 and dwell_seconds >= 12.0:
        return "long_dwell", "Person remained active in the area for an extended duration."
    return "movement_activity", "Dynamic track changes indicate a movement event."


def _collect_event_ocr(frame_lookup: Dict[str, FrameSignal], selected_frame_ids: List[str]) -> List[str]:
    texts: List[str] = []
    for frame_id in selected_frame_ids:
        signal = frame_lookup[frame_id]
        ocr_result = OCRService.extract_text(signal.frame_path)
        texts.extend(ocr_result.get("detected_text", []))
    return list(dict.fromkeys(texts))


def _build_candidate_events(signals: List[FrameSignal]) -> List[CandidateEvent]:
    frame_lookup = {signal.frame_id: signal for signal in signals}
    clusters = _cluster_active_frames(signals)
    keyframe_budgets = _allocate_keyframe_budgets(clusters)
    events: List[CandidateEvent] = []

    for idx, (cluster, keyframe_budget) in enumerate(zip(clusters, keyframe_budgets), start=1):
        event_type, reason = _guess_event_type(cluster)
        selected_frame_ids = _select_keyframes(cluster, keyframe_budget)
        ocr_text = _collect_event_ocr(frame_lookup, selected_frame_ids)
        events.append(
            CandidateEvent(
                event_id=f"cand_evt_{idx:03d}",
                event_type=event_type,
                reason=reason,
                start_seconds=cluster[0].timestamp_seconds,
                end_seconds=cluster[-1].timestamp_seconds,
                frame_ids=[signal.frame_id for signal in cluster],
                selected_frame_ids=selected_frame_ids,
                persons_max=_class_count(cluster, "person"),
                bags_max=max((_class_count(cluster, cls) for cls in BAG_CLASSES), default=0),
                dwell_seconds=cluster[-1].timestamp_seconds - cluster[0].timestamp_seconds,
                track_ids=sorted({track_id for signal in cluster for track_id in signal.track_ids}),
                ocr_text=ocr_text,
            )
        )

    return events


def _render_reasoning_strip(
    event_id: str,
    frame_paths: List[Path],
    labels: List[str],
    mode: str,
) -> Path:
    REASONING_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    panels = []
    width = 448
    height = 252

    for label, path in zip(labels, frame_paths):
        image = cv2.imread(str(path))
        if image is None:
            image = np.zeros((height, width, 3), dtype=np.uint8)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        cv2.rectangle(image, (0, 0), (width, 34), (0, 0, 0), thickness=-1)
        cv2.putText(image, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(image)

    strip = cv2.hconcat(panels)
    out_path = REASONING_INPUT_ROOT / f"{mode}_{event_id}.jpg"
    cv2.imwrite(str(out_path), strip)
    return out_path


def _render_single_frame_image(
    event_id: str,
    frame_path: Path,
    label: str,
    mode: str,
) -> Path:
    REASONING_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    width = 448
    height = 252
    image = cv2.imread(str(frame_path))
    if image is None:
        image = np.zeros((height, width, 3), dtype=np.uint8)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(image, (0, 0), (width, 34), (0, 0, 0), thickness=-1)
    cv2.putText(image, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    out_path = REASONING_INPUT_ROOT / f"{mode}_{event_id}.jpg"
    cv2.imwrite(str(out_path), image)
    return out_path


def _build_structured_context(event: CandidateEvent) -> Dict[str, Any]:
    return {
        "event_type": event.event_type,
        "persons": event.persons_max,
        "bags": event.bags_max,
        "ocr_text": event.ocr_text[:5],
        "dwell_seconds": round(event.dwell_seconds, 1),
        "reason": event.reason,
        "track_count": len(event.track_ids),
        "start_time": format_timestamp_human(event.start_seconds),
        "end_time": format_timestamp_human(event.end_seconds),
    }


def _build_reasoning_prompt(structured_context: Dict[str, Any]) -> str:
    facts_json = json.dumps(structured_context, indent=2)

    return (
        f"{VLM_FRAME_METADATA_PROMPT}\n\n"
        "Additional structured context:\n"
        f"{facts_json}\n\n"
        "Use the structured context only as auxiliary metadata. "
        "Visual evidence always takes priority."
    )


def _image_path_for_variant(
    mode: str,
    variant: BenchmarkVariant,
    event: CandidateEvent,
    frame_lookup: Dict[str, FrameSignal],
    frame_ids: List[str],
    periodic_label: str | None = None,
) -> Tuple[Path, str]:
    frame_paths = [frame_lookup[frame_id].frame_path for frame_id in frame_ids]
    mode_key = f"{mode}_{variant.name}"
    if variant.image_layout == "single_peak":
        label = periodic_label or "PEAK"
        return _render_single_frame_image(event.event_id, frame_paths[0], label, mode_key), "single_peak"

    labels = []
    for idx, _frame_id in enumerate(frame_ids):
        if idx == 0:
            labels.append(periodic_label or "START")
        elif idx == len(frame_ids) - 1:
            labels.append("END")
        else:
            labels.append("KEY")
    return _render_reasoning_strip(event.event_id, frame_paths, labels, mode_key), "strip"


def _build_jobs_for_mode(
    mode: str,
    variant: BenchmarkVariant,
    events: List[CandidateEvent],
    frame_lookup: Dict[str, FrameSignal],
    extracted_tuples: List[Tuple[str, str, float, Path]],
) -> List[ReasoningJob]:
    jobs: List[ReasoningJob] = []
    for event in events:
        selected_frame_ids = event.selected_frame_ids
        if variant.image_layout == "single_peak":
            selected_frame_ids = [event.selected_frame_ids[len(event.selected_frame_ids) // 2]]
            image_path, image_layout = _image_path_for_variant(mode, variant, event, frame_lookup, selected_frame_ids)
        else:
            image_path = frame_lookup[selected_frame_ids[0]].frame_path
            image_layout = "multi_image_event"
        context = _build_structured_context(event)
        jobs.append(
            ReasoningJob(
                mode=mode,
                variant_name=variant.name,
                event_id=event.event_id,
                image_path=image_path,
                image_paths=[frame_lookup[frame_id].frame_path for frame_id in selected_frame_ids],
                selected_frame_ids=selected_frame_ids,
                image_layout=image_layout,
                max_new_tokens=variant.max_new_tokens,
                batch_size=variant.batch_size,
                structured_context=context,
                prompt=_build_reasoning_prompt(context),
                periodic=False,
            )
        )

    if mode == "candidate_plus_periodic10s":
        seconds_seen = set()
        for frame_id, video_id, ts, path in extracted_tuples:
            bucket = int(ts // PERIODIC_SAFETY_SECONDS)
            if bucket in seconds_seen:
                continue
            seconds_seen.add(bucket)
            event_id = f"periodic_{bucket:03d}"
            context = {
                "event_type": "periodic_context",
                "persons": 0,
                "bags": 0,
                "ocr_text": OCRService.extract_text(path).get("detected_text", [])[:5],
                "dwell_seconds": 0.0,
                "reason": "Periodic safety sample for recall protection.",
                "track_count": 0,
                "start_time": format_timestamp_human(ts),
                "end_time": format_timestamp_human(ts),
            }
            periodic_event = CandidateEvent(
                event_id=event_id,
                event_type="periodic_context",
                reason="Periodic safety sample for recall protection.",
                start_seconds=ts,
                end_seconds=ts,
                frame_ids=[frame_id],
                selected_frame_ids=[frame_id],
                persons_max=0,
                bags_max=0,
                dwell_seconds=0.0,
                track_ids=[],
                ocr_text=context["ocr_text"],
            )
            image_path, image_layout = _image_path_for_variant(
                mode,
                variant,
                periodic_event,
                frame_lookup,
                [frame_id],
                periodic_label="PERIODIC",
            )
            jobs.append(
                ReasoningJob(
                    mode=mode,
                    variant_name=variant.name,
                    event_id=event_id,
                    image_path=image_path,
                    image_paths=[path],
                    selected_frame_ids=[frame_id],
                    image_layout=image_layout,
                    max_new_tokens=variant.max_new_tokens,
                    batch_size=variant.batch_size,
                    structured_context=context,
                    prompt=_build_reasoning_prompt(context),
                    periodic=True,
                )
            )

    return jobs


def _count_tokens(text: str) -> int:
    tokenizer = NativeQwenTransformersService.get_tokenizer()
    if tokenizer is None:
        raise RuntimeError("Qwen tokenizer is unavailable after runtime initialization.")
    encoded = tokenizer(text or "", add_special_tokens=False, return_attention_mask=False)
    return len(encoded["input_ids"])


async def _async_hf_generate_multi_image(jobs: List[ReasoningJob]) -> List[str]:
    def run_hf() -> List[str]:
        service = NativeQwenTransformersService
        model, processor, device = service.get_runtime()

        messages_batch = []
        for job in jobs:
            content = [
                {"type": "image", "image": f"file://{path.absolute()}"}
                for path in job.image_paths
            ]
            content.append({"type": "text", "text": job.prompt})
            messages_batch.append([{"role": "user", "content": content}])

        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_batch
        ]
        image_inputs, video_inputs = process_vision_info(messages_batch)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=service._effective_max_new_tokens(),
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    return await asyncio.to_thread(run_hf)


async def _run_vlm_jobs(jobs: List[ReasoningJob]) -> Dict[str, Any]:
    if not jobs:
        return {
            "frames_sent_to_qwen": 0,
            "successful_responses": 0,
            "failed_responses": 0,
            "avg_output_tokens": 0.0,
            "runtime_seconds": 0.0,
            "results": [],
        }

    old_batch_size = settings.BATCH_SIZE
    old_max_new_tokens = settings.QWEN_MAX_NEW_TOKENS
    settings.BATCH_SIZE = jobs[0].batch_size
    settings.QWEN_MAX_NEW_TOKENS = jobs[0].max_new_tokens

    try:
        NativeQwenTransformersService.get_runtime()
        start = time.perf_counter()
        results: List[Dict[str, Any]] = []

        for index in range(0, len(jobs), settings.BATCH_SIZE):
            batch = jobs[index:index + settings.BATCH_SIZE]
            if any(len(job.image_paths) > 1 for job in batch):
                raw_outputs = await _async_hf_generate_multi_image(batch)
            else:
                image_paths = [job.image_path for job in batch]
                prompts = [job.prompt for job in batch]
                raw_outputs = await NativeQwenTransformersService._async_hf_generate(image_paths, prompts)

            for job, raw_output in zip(batch, raw_outputs):
                parsed = None
                success = False
                cleaned = ""
                output_tokens = _count_tokens(raw_output)
                try:
                    cleaned = clean_json_response(raw_output)
                    parsed = json.loads(cleaned)
                    success = isinstance(parsed, dict)
                except Exception:
                    success = False

                failure_category = _classify_failure(raw_output, cleaned, parsed, success)

                results.append(
                    {
                        "event_id": job.event_id,
                        "mode": job.mode,
                        "variant_name": job.variant_name,
                        "periodic": job.periodic,
                        "selected_frame_ids": job.selected_frame_ids,
                        "image_count": len(job.image_paths),
                        "image_layout": job.image_layout,
                        "batch_size": job.batch_size,
                        "structured_context": job.structured_context,
                        "raw_output": raw_output,
                        "cleaned_output": cleaned,
                        "parsed_output": parsed,
                        "success": success,
                        "output_tokens": output_tokens,
                        "failure_category": failure_category,
                    }
                )

        runtime_seconds = time.perf_counter() - start
    finally:
        settings.BATCH_SIZE = old_batch_size
        settings.QWEN_MAX_NEW_TOKENS = old_max_new_tokens

    successful = sum(1 for item in results if item["success"])
    failed = len(results) - successful
    avg_tokens = sum(item["output_tokens"] for item in results) / max(1, len(results))
    failure_breakdown: Dict[str, int] = {}
    for item in results:
        key = item["failure_category"]
        failure_breakdown[key] = failure_breakdown.get(key, 0) + 1
    return {
        "frames_sent_to_qwen": len(jobs),
        "successful_responses": successful,
        "failed_responses": failed,
        "avg_output_tokens": avg_tokens,
        "runtime_seconds": runtime_seconds,
        "failure_breakdown": failure_breakdown,
        "results": results,
    }


def _classify_failure(raw_output: str, cleaned_output: str, parsed_output: Any, success: bool) -> str:
    if success and isinstance(parsed_output, dict):
        return "json_success"
    raw = (raw_output or "").strip()
    cleaned = (cleaned_output or "").strip()
    if not raw:
        return "empty_response"
    if "```json" in raw.lower() and not success:
        return "json_fenced_parse_failure"
    if cleaned and not success:
        return "json_parse_failure"
    if "{" in raw and "}" not in raw:
        return "likely_truncated_json"
    if any(ord(ch) > 127 for ch in raw) and not cleaned:
        return "non_english_garbage"
    return "non_json_garbage"


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_comparison(
    candidate_modes: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    baseline = _load_json(OUTPUT_ROOT / "summary.json", {})
    dynamic = _load_json(OUTPUT_ROOT / "event_driven_summary.json", {})

    comparison = [
        {
            "pipeline": "Baseline HF Pipeline",
            "frames_sent_to_qwen": baseline.get("frames_sent_to_qwen"),
            "average_output_tokens": None,
            "latency_seconds": baseline.get("wall_clock_seconds"),
            "successful_responses": baseline.get("successful_frames"),
            "failed_responses": baseline.get("failed_frames"),
        },
        {
            "pipeline": "Current Dynamic Selection Pipeline",
            "frames_sent_to_qwen": dynamic.get("frames_sent_to_qwen"),
            "average_output_tokens": None,
            "latency_seconds": dynamic.get("wall_clock_seconds"),
            "successful_responses": dynamic.get("successful_frames"),
            "failed_responses": dynamic.get("failed_frames"),
        },
    ]

    for mode_name, mode_variants in candidate_modes.items():
        for variant_name, mode_summary in mode_variants.items():
            comparison.append(
                {
                    "pipeline": f"Event-Candidate Reasoning ({mode_name}/{variant_name})",
                    "frames_sent_to_qwen": mode_summary.get("frames_sent_to_qwen"),
                    "average_output_tokens": mode_summary.get("avg_output_tokens"),
                    "latency_seconds": mode_summary.get("wall_clock_runtime_seconds"),
                    "successful_responses": mode_summary.get("successful_responses"),
                    "failed_responses": mode_summary.get("failed_responses"),
                }
            )

    return comparison


def _write_outputs(summary: Dict[str, Any], timeline: Dict[str, Any], prompts: List[Dict[str, Any]]) -> None:
    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    with open(TIMELINE_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(timeline, f, indent=4)
    with open(PROMPT_EXAMPLES_PATH, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=4)

    lines = [
        "# Event Candidate Benchmark Summary",
        "",
        f"- Input video: `{summary['input_video_path']}`",
        f"- Video duration: `{summary['video_duration_seconds']:.2f}s`",
        f"- Total frames extracted: `{summary['total_frames_extracted']}`",
        f"- Candidate events: `{summary['total_candidate_events']}`",
        "",
        "## Modes",
        "",
    ]
    for mode_name, mode_variants in summary["modes"].items():
        lines.append(f"### {mode_name}")
        for variant_name, mode_summary in mode_variants.items():
            lines.extend(
                [
                    f"- Variant `{variant_name}`",
                    f"  - Selected keyframes: `{mode_summary['selected_keyframes']}`",
                    f"  - Frames sent to Qwen: `{mode_summary['frames_sent_to_qwen']}`",
                    f"  - Batch size: `{mode_summary['batch_size']}`",
                    f"  - Average frames per event: `{mode_summary['avg_frames_per_event']:.2f}`",
                    f"  - Average output tokens: `{mode_summary['avg_output_tokens']:.2f}`",
                    f"  - Successful responses: `{mode_summary['successful_responses']}`",
                    f"  - Failed responses: `{mode_summary['failed_responses']}`",
                    f"  - Failure breakdown: `{mode_summary['failure_breakdown']}`",
                    f"  - Wall-clock runtime: `{mode_summary['wall_clock_runtime_seconds']:.2f}s`",
                    f"  - Realtime ratio: `{mode_summary['realtime_ratio']:.3f}x`",
                    "",
                ]
            )

    lines.extend(["## Comparison", ""])
    for row in summary["comparison_table"]:
        lines.append(
            f"- {row['pipeline']}: frames={row['frames_sent_to_qwen']}, "
            f"tokens={row['average_output_tokens']}, latency={row['latency_seconds']}, "
            f"success={row['successful_responses']}, failed={row['failed_responses']}"
        )

    with open(SUMMARY_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


async def main() -> None:
    _ensure_dirs()
    _clean_previous_outputs()
    _copy_input_video()
    NativeQwenTransformersService.get_runtime()

    overall_start = time.perf_counter()
    video_duration_seconds = _get_video_duration_seconds(INPUT_COPY_PATH)
    extracted_tuples = _extract_one_fps_frames(BENCHMARK_VIDEO_ID, INPUT_VIDEO_PATH)
    frame_detections, tracking_map = _run_detection_and_tracking(extracted_tuples)
    signals = _build_frame_signals(extracted_tuples, tracking_map)
    frame_lookup = {signal.frame_id: signal for signal in signals}
    candidate_events = _build_candidate_events(signals)
    selected_modes = _resolve_selected_modes()
    selected_variants = _resolve_selected_variants()

    modes: Dict[str, Dict[str, Any]] = {}
    prompt_examples: List[Dict[str, Any]] = []
    timeline: Dict[str, Any] = {
        "video_id": BENCHMARK_VIDEO_ID,
        "candidate_events": [],
        "modes": {},
    }

    for mode_name in selected_modes:
        modes[mode_name] = {}
        timeline["modes"][mode_name] = {}
        for variant in selected_variants:
            mode_start = time.perf_counter()
            jobs = _build_jobs_for_mode(mode_name, variant, candidate_events, frame_lookup, extracted_tuples)
            vlm_summary = await _run_vlm_jobs(jobs)
            wall_clock_runtime_seconds = time.perf_counter() - mode_start
            avg_frames_per_event = (
                sum(len(job.selected_frame_ids) for job in jobs if not job.periodic) / max(1, len(candidate_events))
            )

            modes[mode_name][variant.name] = {
                "variant_name": variant.name,
                "image_layout": variant.image_layout,
                "max_new_tokens": variant.max_new_tokens,
                "batch_size": variant.batch_size,
                "total_candidate_events": len(candidate_events),
                "selected_keyframes": sum(len(job.selected_frame_ids) for job in jobs if not job.periodic),
                "frames_sent_to_qwen": vlm_summary["frames_sent_to_qwen"],
                "avg_frames_per_event": avg_frames_per_event,
                "avg_output_tokens": vlm_summary["avg_output_tokens"],
                "successful_responses": vlm_summary["successful_responses"],
                "failed_responses": vlm_summary["failed_responses"],
                "failure_breakdown": vlm_summary["failure_breakdown"],
                "wall_clock_runtime_seconds": wall_clock_runtime_seconds,
                "realtime_ratio": (wall_clock_runtime_seconds / video_duration_seconds) if video_duration_seconds > 0 else 0.0,
                "results": vlm_summary["results"],
            }

            timeline["modes"][mode_name][variant.name] = [
                {
                    "event_id": job.event_id,
                    "periodic": job.periodic,
                    "variant_name": job.variant_name,
                    "image_layout": job.image_layout,
                    "batch_size": job.batch_size,
                    "selected_frame_ids": job.selected_frame_ids,
                    "structured_context": job.structured_context,
                }
                for job in jobs
            ]
            prompt_examples.extend(
                {
                    "mode": job.mode,
                    "variant_name": job.variant_name,
                    "event_id": job.event_id,
                    "image_layout": job.image_layout,
                    "batch_size": job.batch_size,
                    "prompt": job.prompt,
                    "structured_context": job.structured_context,
                }
                for job in jobs[:4]
            )

    timeline["candidate_events"] = [
        {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "reason": event.reason,
            "start_seconds": event.start_seconds,
            "end_seconds": event.end_seconds,
            "frame_ids": event.frame_ids,
            "selected_frames": event.selected_frame_ids,
            "dwell_seconds": event.dwell_seconds,
            "persons": event.persons_max,
            "bags": event.bags_max,
            "track_ids": event.track_ids,
            "ocr_text": event.ocr_text,
        }
        for event in candidate_events
    ]

    total_runtime_seconds = time.perf_counter() - overall_start
    tokenizer = NativeQwenTransformersService.get_tokenizer()
    tokenizer_padding_side = getattr(tokenizer, "padding_side", None)
    summary = {
        "input_video_path": str(INPUT_VIDEO_PATH),
        "input_copy_path": str(INPUT_COPY_PATH),
        "video_id": BENCHMARK_VIDEO_ID,
        "video_duration_seconds": video_duration_seconds,
        "total_frames_extracted": len(extracted_tuples),
        "total_candidate_events": len(candidate_events),
        "tokenizer_padding_side": tokenizer_padding_side,
        "selected_modes": selected_modes,
        "selected_variants": [variant.name for variant in selected_variants],
        "modes": modes,
        "comparison_table": _build_comparison(modes),
        "overall_runtime_seconds": total_runtime_seconds,
    }
    _write_outputs(summary, timeline, prompt_examples)

    print("EVENT_CANDIDATE_BENCHMARK_SUMMARY_START")
    print(json.dumps(summary))
    print("EVENT_CANDIDATE_BENCHMARK_SUMMARY_END")


if __name__ == "__main__":
    asyncio.run(main())
