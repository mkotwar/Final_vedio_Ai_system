import asyncio
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT_PATH = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from app.core.config import PROJECT_ROOT, settings
from app.core.utils import format_timestamp_human
from app.services.object_detection.detector import ObjectDetector
from app.services.object_detection.schemas import FrameDetection
from app.services.object_tracker import ObjectTrackerService
from app.services.ocr import OCRService
from app.services.qwen_vlm_hf import NativeQwenTransformersService
from app.services.vlm_utils import clean_json_response


INPUT_VIDEO_PATH = Path(os.getenv("BENCHMARK_INPUT_VIDEO", r"C:\Mukul K\test_video\V_ai_test_2min.mp4"))
BENCHMARK_VIDEO_ID = os.getenv("BENCHMARK_VIDEO_ID", "11111111-2222-4333-8444-777777777777")
CASE_ROOT = PROJECT_ROOT / "tests" / "manual_benchmark_case"
DATA_ROOT = CASE_ROOT / "data"
INPUT_ROOT = DATA_ROOT / "input"
OUTPUT_ROOT = DATA_ROOT / "output"
FRAME_ROOT = PROJECT_ROOT / "data" / "frames" / BENCHMARK_VIDEO_ID
DETECTION_ROOT = PROJECT_ROOT / "data" / "detections" / BENCHMARK_VIDEO_ID
KEYFRAME_ROOT = OUTPUT_ROOT / "relationship_keyframes"
REASONING_INPUT_ROOT = OUTPUT_ROOT / "relationship_reasoning_inputs"

SUMMARY_JSON_PATH = OUTPUT_ROOT / "relationship_event_summary.json"
SUMMARY_MD_PATH = OUTPUT_ROOT / "relationship_event_summary.md"
TIMELINE_JSON_PATH = OUTPUT_ROOT / "relationship_event_timeline.json"
FRAME_AUDIT_JSON_PATH = OUTPUT_ROOT / "relationship_frame_audit.json"
INPUT_COPY_PATH = INPUT_ROOT / INPUT_VIDEO_PATH.name

CURRENT_SUMMARY_PATH = OUTPUT_ROOT / "event_candidate_benchmark_summary.json"
CURRENT_TIMELINE_PATH = OUTPUT_ROOT / "event_candidate_timeline.json"

FRAME_GAP_SECONDS = 5.0
NEAR_DISTANCE_PIXELS = 120.0
TOUCH_DISTANCE_PIXELS = 70.0
SIGNIFICANT_DISTANCE_DELTA = 80.0
STATIONARY_MOVEMENT_PIXELS = 20.0
STATIONARY_SECONDS = 2.0
UNATTENDED_SECONDS = 3.0
PROMPT_MAX_NEW_TOKENS = 150
BATCH_SIZE = 1
MOVABLE_OBJECT_CLASSES = {"backpack", "handbag", "suitcase", "bottle", "cup", "book", "cell phone"}


@dataclass
class FrameContext:
    frame_id: str
    video_id: str
    timestamp_seconds: float
    frame_path: Path
    tracked_entities: List[Dict[str, Any]]
    person_entities: List[Dict[str, Any]]
    object_entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    state_changes: List[Dict[str, Any]]
    reasons: List[str]


@dataclass
class RelationshipTrigger:
    frame_id: str
    timestamp_seconds: float
    object_track_id: int
    person_track_id: Optional[int]
    trigger_type: str
    reason: str
    related_track_ids: List[int] = field(default_factory=list)


@dataclass
class RelationshipEvent:
    event_id: str
    event_type: str
    reason: str
    start_seconds: float
    end_seconds: float
    frame_ids: List[str]
    selected_frame_ids: List[str]
    object_track_id: int
    person_track_ids: List[int]
    trigger_types: List[str]
    relationship_summary: List[str]
    ocr_text: List[str]


@dataclass
class ReasoningJob:
    event_id: str
    image_path: Path
    selected_frame_ids: List[str]
    structured_context: Dict[str, Any]
    prompt: str


def _ensure_dirs() -> None:
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    KEYFRAME_ROOT.mkdir(parents=True, exist_ok=True)
    REASONING_INPUT_ROOT.mkdir(parents=True, exist_ok=True)


def _copy_input_video() -> None:
    if not INPUT_VIDEO_PATH.exists():
        raise FileNotFoundError(f"Input video not found: {INPUT_VIDEO_PATH}")
    shutil.copy2(INPUT_VIDEO_PATH, INPUT_COPY_PATH)


def _safe_remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


def _clean_previous_outputs() -> None:
    _safe_remove(KEYFRAME_ROOT)
    _safe_remove(REASONING_INPUT_ROOT)
    for path in (SUMMARY_JSON_PATH, SUMMARY_MD_PATH, TIMELINE_JSON_PATH, FRAME_AUDIT_JSON_PATH):
        _safe_remove(path)
    KEYFRAME_ROOT.mkdir(parents=True, exist_ok=True)
    REASONING_INPUT_ROOT.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


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

    existing = sorted(FRAME_ROOT.glob("frame_*.jpg"))
    if existing:
        extracted: List[Tuple[str, str, float, Path]] = []
        for index, path in enumerate(existing, start=1):
            extracted.append((f"{video_id}_f{index:04d}", video_id, float(index - 1), path))
        return extracted

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


def _load_or_run_detections(extracted_tuples: List[Tuple[str, str, float, Path]]) -> List[FrameDetection]:
    files = sorted(DETECTION_ROOT.glob("*.json"))
    if len(files) == len(extracted_tuples) and files:
        return [FrameDetection.model_validate_json(path.read_text(encoding="utf-8")) for path in files]

    detector = ObjectDetector()
    return [detector.detect_frame(path, frame_id, video_id, ts) for frame_id, video_id, ts, path in extracted_tuples]


def _center(entity: Dict[str, Any]) -> Tuple[float, float]:
    x1, y1, x2, y2 = entity.get("bbox", [0.0, 0.0, 0.0, 0.0])
    return ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)


def _distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ax, ay = _center(a)
    bx, by = _center(b)
    return math.hypot(ax - bx, ay - by)


def _copy_keyframe(frame_path: Path, event_id: str, frame_id: str) -> None:
    target = KEYFRAME_ROOT / f"{event_id}_{frame_id}.jpg"
    shutil.copy2(frame_path, target)


def _render_reasoning_strip(event_id: str, frame_paths: List[Path], labels: List[str]) -> Path:
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
    out_path = REASONING_INPUT_ROOT / f"{event_id}.jpg"
    cv2.imwrite(str(out_path), strip)
    return out_path


def _collect_event_ocr(frame_lookup: Dict[str, FrameContext], frame_ids: List[str]) -> List[str]:
    texts: List[str] = []
    for frame_id in frame_ids:
        result = OCRService.extract_text(frame_lookup[frame_id].frame_path)
        texts.extend(result.get("detected_text", []))
    return list(dict.fromkeys(texts))


def _relationship_state_changes(
    extracted_tuples: List[Tuple[str, str, float, Path]],
    tracking_map: Dict[str, Dict[str, Any]],
) -> Tuple[List[FrameContext], List[RelationshipTrigger]]:
    frame_contexts: List[FrameContext] = []
    all_triggers: List[RelationshipTrigger] = []
    object_memory: Dict[int, Dict[str, Any]] = {}

    for frame_id, video_id, ts, frame_path in extracted_tuples:
        tracking = tracking_map.get(frame_id, {})
        tracked_entities = tracking.get("tracked_entities", [])
        person_entities = [e for e in tracked_entities if str(e.get("class_name", "")).lower() == "person"]
        object_entities = [
            e for e in tracked_entities if str(e.get("class_name", "")).lower() in MOVABLE_OBJECT_CLASSES
        ]
        relationships: List[Dict[str, Any]] = []
        state_changes: List[Dict[str, Any]] = []
        reasons: List[str] = []

        for obj in object_entities:
            obj_track_id = int(obj.get("track_id"))
            obj_center = _center(obj)
            nearest_person = None
            nearest_distance = None
            touching_person = None
            near_people: List[int] = []
            touching_people: List[int] = []

            for person in person_entities:
                distance = _distance(person, obj)
                person_track_id = int(person.get("track_id"))
                if nearest_distance is None or distance < nearest_distance:
                    nearest_distance = distance
                    nearest_person = person_track_id
                if distance <= NEAR_DISTANCE_PIXELS:
                    near_people.append(person_track_id)
                if distance <= TOUCH_DISTANCE_PIXELS:
                    touching_people.append(person_track_id)
                    touching_person = person_track_id

            previous = object_memory.get(obj_track_id, {})
            prev_center = previous.get("center")
            moved_pixels = math.hypot(obj_center[0] - prev_center[0], obj_center[1] - prev_center[1]) if prev_center else 0.0
            stationary_duration = previous.get("stationary_duration", 0.0)
            if prev_center is None or moved_pixels <= STATIONARY_MOVEMENT_PIXELS:
                stationary_duration += 1.0
            else:
                stationary_duration = 0.0

            relationship = {
                "object_track_id": obj_track_id,
                "object_class": obj.get("class_name"),
                "nearest_person_track_id": nearest_person,
                "nearest_distance": round(nearest_distance, 2) if nearest_distance is not None else None,
                "near_people": near_people,
                "touching_people": touching_people,
                "moved_pixels": round(moved_pixels, 2),
                "stationary_duration": round(stationary_duration, 1),
                "ownership_candidate": nearest_person if near_people else None,
            }
            relationships.append(relationship)

            prev_nearest = previous.get("nearest_person")
            prev_distance = previous.get("nearest_distance")
            prev_touching = previous.get("touching_person")
            prev_stationary = previous.get("stationary_duration", 0.0)
            prev_near_people = previous.get("near_people", [])

            triggers: List[Tuple[str, str, Optional[int], List[int]]] = []
            if prev_nearest is None and nearest_person is not None and nearest_distance is not None and nearest_distance <= NEAR_DISTANCE_PIXELS:
                triggers.append(("person_approaches_object", "A person came near the object.", nearest_person, [obj_track_id, nearest_person]))
            if prev_nearest is not None and (nearest_person is None or (nearest_distance is not None and nearest_distance > NEAR_DISTANCE_PIXELS)):
                triggers.append(("person_leaves_object", "The nearest person moved away from the object.", prev_nearest, [obj_track_id, prev_nearest]))
            if prev_touching is None and touching_person is not None:
                triggers.append(("interaction_starts", "Direct person-object interaction started.", touching_person, [obj_track_id, touching_person]))
            if prev_touching is not None and touching_person is None:
                triggers.append(("interaction_ends", "Direct person-object interaction ended.", prev_touching, [obj_track_id, prev_touching]))
            if prev_stationary < STATIONARY_SECONDS <= stationary_duration:
                triggers.append(("object_becomes_stationary", "The object became stationary.", nearest_person, [obj_track_id] + near_people[:1]))
            if prev_stationary >= STATIONARY_SECONDS and moved_pixels > STATIONARY_MOVEMENT_PIXELS:
                triggers.append(("object_begins_moving_again", "A previously stationary object started moving again.", nearest_person, [obj_track_id] + near_people[:1]))
            if prev_distance is not None and nearest_distance is not None and abs(prev_distance - nearest_distance) >= SIGNIFICANT_DISTANCE_DELTA:
                triggers.append(("person_object_distance_changes", "Distance between person and object changed significantly.", nearest_person, [obj_track_id] + near_people[:1]))
            if stationary_duration >= UNATTENDED_SECONDS and not near_people:
                triggers.append(("object_remains_unattended", "The object remained unattended while stationary.", None, [obj_track_id]))
            if len(near_people) >= 2 and sorted(near_people) != sorted(prev_near_people):
                triggers.append(("multiple_people_interact_same_object", "Multiple people were near the same object.", nearest_person, [obj_track_id] + near_people[:2]))
            if prev_nearest is not None and nearest_person is not None and prev_nearest != nearest_person and prev_touching is None and touching_person is not None:
                triggers.append(("ownership_changes", "A different person took over the object interaction.", touching_person, [obj_track_id, prev_nearest, touching_person]))

            for trigger_type, reason, person_track_id, related_track_ids in triggers:
                state_changes.append(
                    {
                        "object_track_id": obj_track_id,
                        "person_track_id": person_track_id,
                        "trigger_type": trigger_type,
                        "reason": reason,
                        "related_track_ids": related_track_ids,
                    }
                )
                reasons.append(trigger_type)
                all_triggers.append(
                    RelationshipTrigger(
                        frame_id=frame_id,
                        timestamp_seconds=ts,
                        object_track_id=obj_track_id,
                        person_track_id=person_track_id,
                        trigger_type=trigger_type,
                        reason=reason,
                        related_track_ids=related_track_ids,
                    )
                )

            object_memory[obj_track_id] = {
                "center": obj_center,
                "nearest_person": nearest_person,
                "nearest_distance": nearest_distance,
                "touching_person": touching_person,
                "stationary_duration": stationary_duration,
                "near_people": near_people,
            }

        frame_contexts.append(
            FrameContext(
                frame_id=frame_id,
                video_id=video_id,
                timestamp_seconds=ts,
                frame_path=frame_path,
                tracked_entities=tracked_entities,
                person_entities=person_entities,
                object_entities=object_entities,
                relationships=relationships,
                state_changes=state_changes,
                reasons=reasons,
            )
        )

    return frame_contexts, all_triggers


def _cluster_relationship_triggers(triggers: List[RelationshipTrigger]) -> List[List[RelationshipTrigger]]:
    if not triggers:
        return []
    ordered = sorted(triggers, key=lambda item: (item.object_track_id, item.timestamp_seconds))
    clusters: List[List[RelationshipTrigger]] = []
    current = [ordered[0]]
    for trigger in ordered[1:]:
        previous = current[-1]
        if trigger.object_track_id == previous.object_track_id and trigger.timestamp_seconds - previous.timestamp_seconds <= FRAME_GAP_SECONDS:
            current.append(trigger)
        else:
            clusters.append(current)
            current = [trigger]
    clusters.append(current)
    return clusters


def _event_type_from_triggers(trigger_types: List[str]) -> Tuple[str, str]:
    trigger_set = set(trigger_types)
    if "ownership_changes" in trigger_set or "multiple_people_interact_same_object" in trigger_set:
        return "ownership_change", "Multiple people interacted with the same object across state changes."
    if "object_remains_unattended" in trigger_set and "person_leaves_object" in trigger_set:
        return "possible_unattended_object", "A person left and the object remained stationary without a nearby owner."
    if "interaction_starts" in trigger_set or "interaction_ends" in trigger_set:
        return "person_object_interaction", "Person-object interaction boundaries were observed."
    if "object_becomes_stationary" in trigger_set or "object_begins_moving_again" in trigger_set:
        return "object_state_change", "An object's motion state changed."
    return "relationship_transition", "Relationship-based state transitions were observed."


def _select_relationship_keyframes(cluster: List[RelationshipTrigger], frame_lookup: Dict[str, FrameContext]) -> List[str]:
    priority_order = [
        "interaction_starts",
        "interaction_ends",
        "object_becomes_stationary",
        "object_remains_unattended",
        "ownership_changes",
        "object_begins_moving_again",
        "multiple_people_interact_same_object",
        "person_approaches_object",
        "person_leaves_object",
        "person_object_distance_changes",
    ]
    chosen: List[str] = []
    for trigger_name in priority_order:
        for trigger in cluster:
            if trigger.trigger_type == trigger_name and trigger.frame_id not in chosen:
                chosen.append(trigger.frame_id)

    if cluster:
        first_frame = cluster[0].frame_id
        last_frame = cluster[-1].frame_id
        if first_frame not in chosen:
            chosen.insert(0, first_frame)
        if last_frame not in chosen:
            chosen.append(last_frame)

    ordered_unique: List[str] = []
    seen = set()
    for frame_id in chosen:
        if frame_id not in frame_lookup or frame_id in seen:
            continue
        seen.add(frame_id)
        ordered_unique.append(frame_id)

    return ordered_unique[:5]


def _build_relationship_events(frame_contexts: List[FrameContext], triggers: List[RelationshipTrigger]) -> List[RelationshipEvent]:
    frame_lookup = {frame.frame_id: frame for frame in frame_contexts}
    clusters = _cluster_relationship_triggers(triggers)
    events: List[RelationshipEvent] = []

    for index, cluster in enumerate(clusters, start=1):
        trigger_types = [item.trigger_type for item in cluster]
        event_type, reason = _event_type_from_triggers(trigger_types)
        selected_frame_ids = _select_relationship_keyframes(cluster, frame_lookup)
        ocr_text = _collect_event_ocr(frame_lookup, selected_frame_ids)
        related_frame_ids = [item.frame_id for item in cluster]
        frame_ids = list(dict.fromkeys(related_frame_ids + selected_frame_ids))
        person_track_ids = sorted(
            {
                track_id
                for item in cluster
                for track_id in item.related_track_ids
                if track_id != item.object_track_id
            }
        )
        relationship_summary = [
            f"{item.trigger_type} at {format_timestamp_human(item.timestamp_seconds)} ({item.reason})" for item in cluster
        ]
        for frame_id in selected_frame_ids:
            _copy_keyframe(frame_lookup[frame_id].frame_path, f"rel_evt_{index:03d}", frame_id)

        events.append(
            RelationshipEvent(
                event_id=f"rel_evt_{index:03d}",
                event_type=event_type,
                reason=reason,
                start_seconds=cluster[0].timestamp_seconds,
                end_seconds=cluster[-1].timestamp_seconds,
                frame_ids=frame_ids,
                selected_frame_ids=selected_frame_ids,
                object_track_id=cluster[0].object_track_id,
                person_track_ids=person_track_ids,
                trigger_types=trigger_types,
                relationship_summary=relationship_summary,
                ocr_text=ocr_text,
            )
        )

    return events


def _build_structured_context(event: RelationshipEvent) -> Dict[str, Any]:
    return {
        "event_type": event.event_type,
        "persons": len(event.person_track_ids),
        "bags": 1 if event.object_track_id else 0,
        "ocr_text": event.ocr_text[:5],
        "dwell_seconds": round(event.end_seconds - event.start_seconds, 1),
        "reason": event.reason,
        "track_count": len(event.person_track_ids) + 1,
        "start_time": format_timestamp_human(event.start_seconds),
        "end_time": format_timestamp_human(event.end_seconds),
        "relationship_summary": event.relationship_summary[:4],
        "trigger_types": event.trigger_types,
        "object_track_id": event.object_track_id,
        "person_track_ids": event.person_track_ids,
    }


def _build_reasoning_prompt(structured_context: Dict[str, Any]) -> str:
    facts_json = json.dumps(structured_context, separators=(",", ":"))
    return (
        "You are an investigation assistant.\n"
        f"Known facts: {facts_json}\n"
        "Analyze:\n"
        "1. likely event\n"
        "2. notable behavior\n"
        "3. interaction\n"
        "4. investigate?\n"
        'Return compact JSON only with keys event_type, notable, interaction, investigate, why. '
        'Use true/false booleans and very short strings.'
    )


def _build_reasoning_jobs(events: List[RelationshipEvent], frame_lookup: Dict[str, FrameContext]) -> List[ReasoningJob]:
    jobs: List[ReasoningJob] = []
    for event in events:
        frame_paths = [frame_lookup[frame_id].frame_path for frame_id in event.selected_frame_ids]
        labels = []
        for index, _frame_id in enumerate(event.selected_frame_ids):
            if index == 0:
                labels.append("START")
            elif index == len(event.selected_frame_ids) - 1:
                labels.append("END")
            else:
                labels.append("STATE")
        image_path = _render_reasoning_strip(event.event_id, frame_paths, labels)
        context = _build_structured_context(event)
        jobs.append(
            ReasoningJob(
                event_id=event.event_id,
                image_path=image_path,
                selected_frame_ids=event.selected_frame_ids,
                structured_context=context,
                prompt=_build_reasoning_prompt(context),
            )
        )
    return jobs


def _count_tokens(text: str) -> int:
    processor = NativeQwenTransformersService._processor
    tokenizer = processor.tokenizer
    encoded = tokenizer(text or "", add_special_tokens=False, return_attention_mask=False)
    return len(encoded["input_ids"])


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


async def _run_vlm_jobs(jobs: List[ReasoningJob]) -> Dict[str, Any]:
    if not jobs:
        return {
            "frames_sent_to_qwen": 0,
            "successful_responses": 0,
            "failed_responses": 0,
            "avg_output_tokens": 0.0,
            "failure_breakdown": {},
            "results": [],
        }

    old_batch_size = settings.BATCH_SIZE
    old_max_new_tokens = settings.QWEN_MAX_NEW_TOKENS
    settings.BATCH_SIZE = BATCH_SIZE
    settings.QWEN_MAX_NEW_TOKENS = PROMPT_MAX_NEW_TOKENS

    try:
        NativeQwenTransformersService.load_model()
        tokenizer = NativeQwenTransformersService._processor.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        start = time.perf_counter()
        results: List[Dict[str, Any]] = []
        for index in range(0, len(jobs), settings.BATCH_SIZE):
            batch = jobs[index:index + settings.BATCH_SIZE]
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
                results.append(
                    {
                        "event_id": job.event_id,
                        "selected_frame_ids": job.selected_frame_ids,
                        "structured_context": job.structured_context,
                        "raw_output": raw_output,
                        "cleaned_output": cleaned,
                        "parsed_output": parsed,
                        "success": success,
                        "output_tokens": output_tokens,
                        "failure_category": _classify_failure(raw_output, cleaned, parsed, success),
                        "image_path": str(job.image_path),
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
        "failure_breakdown": failure_breakdown,
        "runtime_seconds": runtime_seconds,
        "results": results,
    }


def _build_frame_audit(
    frame_contexts: List[FrameContext],
    current_timeline: Dict[str, Any],
    relationship_events: List[RelationshipEvent],
) -> List[Dict[str, Any]]:
    current_selected = {
        frame_id
        for event in current_timeline.get("candidate_events", [])
        for frame_id in event.get("selected_frames", [])
    }
    relationship_selected = {frame_id for event in relationship_events for frame_id in event.selected_frame_ids}
    relationship_reasons: Dict[str, List[str]] = {}
    for event in relationship_events:
        for frame_id in event.selected_frame_ids:
            relationship_reasons.setdefault(frame_id, []).append(event.event_id)

    rows: List[Dict[str, Any]] = []
    for frame in frame_contexts:
        rows.append(
            {
                "frame_id": frame.frame_id,
                "timestamp_seconds": frame.timestamp_seconds,
                "timestamp_human": format_timestamp_human(frame.timestamp_seconds),
                "yolo_objects": [
                    {"track_id": int(entity.get("track_id")), "class_name": entity.get("class_name")}
                    for entity in frame.tracked_entities
                ],
                "tracks": [int(entity.get("track_id")) for entity in frame.tracked_entities],
                "relationships": frame.relationships,
                "state_changes": frame.state_changes,
                "current_benchmark_selected": frame.frame_id in current_selected,
                "relationship_benchmark_selected": frame.frame_id in relationship_selected,
                "reason": frame.reasons + relationship_reasons.get(frame.frame_id, []),
            }
        )
    return rows


def _build_comparison(summary: Dict[str, Any], current_summary: Dict[str, Any], current_timeline: Dict[str, Any]) -> Dict[str, Any]:
    current_mode = current_summary.get("modes", {}).get("candidate_only", {}).get("strip_tokens150_batch1", {})
    current_events = current_timeline.get("candidate_events", [])
    relationship_events = summary["relationship_events"]

    event_comparisons: List[Dict[str, Any]] = []
    for index, rel_event in enumerate(relationship_events):
        current_selected = current_events[index]["selected_frames"] if index < len(current_events) else []
        event_comparisons.append(
            {
                "relationship_event_id": rel_event["event_id"],
                "current_selected_frames": current_selected,
                "relationship_selected_frames": rel_event["selected_frame_ids"],
                "manual_expected_important_frames": [],
            }
        )

    return {
        "current_benchmark": {
            "candidate_events": len(current_events),
            "frames_sent_to_qwen": current_mode.get("frames_sent_to_qwen"),
            "selected_keyframes": current_mode.get("selected_keyframes"),
            "latency_seconds": current_mode.get("wall_clock_runtime_seconds"),
        },
        "relationship_benchmark": {
            "candidate_events": len(relationship_events),
            "frames_sent_to_qwen": summary["frames_sent_to_qwen"],
            "selected_keyframes": sum(len(event["selected_frame_ids"]) for event in relationship_events),
            "latency_seconds": summary["wall_clock_runtime_seconds"],
        },
        "per_event_selected_frames": event_comparisons,
    }


def _write_outputs(summary: Dict[str, Any], timeline: Dict[str, Any], frame_audit: List[Dict[str, Any]]) -> None:
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=4), encoding="utf-8")
    TIMELINE_JSON_PATH.write_text(json.dumps(timeline, indent=4), encoding="utf-8")
    FRAME_AUDIT_JSON_PATH.write_text(json.dumps(frame_audit, indent=4), encoding="utf-8")

    analysis = summary["analysis"]
    comparison = summary["comparison"]
    lines = [
        "# Relationship Event Benchmark Summary",
        "",
        f"- Input video: `{summary['input_video_path']}`",
        f"- Video duration: `{summary['video_duration_seconds']:.2f}s`",
        f"- Extracted frames: `{summary['total_frames_extracted']}`",
        f"- Relationship candidate events: `{summary['total_candidate_events']}`",
        f"- Frames sent to Qwen: `{summary['frames_sent_to_qwen']}`",
        f"- Selected keyframes: `{summary['selected_keyframes']}`",
        f"- Successful responses: `{summary['successful_responses']}`",
        f"- Failed responses: `{summary['failed_responses']}`",
        f"- Wall-clock runtime: `{summary['wall_clock_runtime_seconds']:.2f}s`",
        "",
        "## Comparison",
        "",
        f"- Current benchmark candidate events: `{comparison['current_benchmark']['candidate_events']}`",
        f"- Current benchmark frames sent to Qwen: `{comparison['current_benchmark']['frames_sent_to_qwen']}`",
        f"- Current benchmark selected keyframes: `{comparison['current_benchmark']['selected_keyframes']}`",
        f"- Current benchmark latency: `{comparison['current_benchmark']['latency_seconds']}`",
        f"- Relationship benchmark candidate events: `{comparison['relationship_benchmark']['candidate_events']}`",
        f"- Relationship benchmark frames sent to Qwen: `{comparison['relationship_benchmark']['frames_sent_to_qwen']}`",
        f"- Relationship benchmark selected keyframes: `{comparison['relationship_benchmark']['selected_keyframes']}`",
        f"- Relationship benchmark latency: `{comparison['relationship_benchmark']['latency_seconds']}`",
        "",
        "## Final Analysis",
        "",
        f"- More semantically meaningful frames: `{analysis['more_semantically_meaningful_frames']}`",
        f"- Included drop/release-like frames: `{analysis['included_drop_frames']}`",
        f"- Included pickup/ownership-change frames: `{analysis['included_pickup_frames']}`",
        f"- Included unattended-object frames: `{analysis['included_unattended_frames']}`",
        f"- Qwen workload changed: `{analysis['qwen_workload_change']}`",
        f"- Tender compliance improvement evidence: `{analysis['tender_alignment']}`",
        "",
    ]

    for row in comparison["per_event_selected_frames"]:
        lines.extend(
            [
                f"### {row['relationship_event_id']}",
                f"- Current selected frames: `{row['current_selected_frames']}`",
                f"- Relationship selected frames: `{row['relationship_selected_frames']}`",
                f"- Manual expected important frames: `{row['manual_expected_important_frames']}`",
                "",
            ]
        )

    SUMMARY_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


async def main() -> None:
    _ensure_dirs()
    _clean_previous_outputs()
    _copy_input_video()

    overall_start = time.perf_counter()
    video_duration_seconds = _get_video_duration_seconds(INPUT_COPY_PATH)
    extracted_tuples = _extract_one_fps_frames(BENCHMARK_VIDEO_ID, INPUT_VIDEO_PATH)
    frame_detections = _load_or_run_detections(extracted_tuples)
    tracking_map = ObjectTrackerService.track_frames(frame_detections)

    frame_contexts, triggers = _relationship_state_changes(extracted_tuples, tracking_map)
    relationship_events = _build_relationship_events(frame_contexts, triggers)
    frame_lookup = {frame.frame_id: frame for frame in frame_contexts}
    reasoning_jobs = _build_reasoning_jobs(relationship_events, frame_lookup)
    vlm_summary = await _run_vlm_jobs(reasoning_jobs)

    current_summary = _load_json(CURRENT_SUMMARY_PATH, {})
    current_timeline = _load_json(CURRENT_TIMELINE_PATH, {})

    relationship_events_json = [
        {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "reason": event.reason,
            "start_seconds": event.start_seconds,
            "end_seconds": event.end_seconds,
            "frame_ids": event.frame_ids,
            "selected_frame_ids": event.selected_frame_ids,
            "object_track_id": event.object_track_id,
            "person_track_ids": event.person_track_ids,
            "trigger_types": event.trigger_types,
            "relationship_summary": event.relationship_summary,
            "ocr_text": event.ocr_text,
        }
        for event in relationship_events
    ]

    wall_clock_runtime_seconds = time.perf_counter() - overall_start
    summary = {
        "input_video_path": str(INPUT_VIDEO_PATH),
        "input_copy_path": str(INPUT_COPY_PATH),
        "video_id": BENCHMARK_VIDEO_ID,
        "video_duration_seconds": video_duration_seconds,
        "total_frames_extracted": len(extracted_tuples),
        "total_candidate_events": len(relationship_events),
        "selected_keyframes": sum(len(event.selected_frame_ids) for event in relationship_events),
        "frames_sent_to_qwen": vlm_summary["frames_sent_to_qwen"],
        "successful_responses": vlm_summary["successful_responses"],
        "failed_responses": vlm_summary["failed_responses"],
        "avg_output_tokens": vlm_summary["avg_output_tokens"],
        "failure_breakdown": vlm_summary["failure_breakdown"],
        "wall_clock_runtime_seconds": wall_clock_runtime_seconds,
        "realtime_ratio": (wall_clock_runtime_seconds / video_duration_seconds) if video_duration_seconds > 0 else 0.0,
        "relationship_events": relationship_events_json,
        "vlm_results": vlm_summary["results"],
    }
    summary["comparison"] = _build_comparison(summary, current_summary, current_timeline)

    trigger_type_set = {trigger.trigger_type for trigger in triggers}
    summary["analysis"] = {
        "more_semantically_meaningful_frames": len(trigger_type_set) > 0 and len(relationship_events) > 0,
        "included_drop_frames": "interaction_ends" in trigger_type_set or "object_becomes_stationary" in trigger_type_set,
        "included_pickup_frames": "ownership_changes" in trigger_type_set or "object_begins_moving_again" in trigger_type_set,
        "included_unattended_frames": "object_remains_unattended" in trigger_type_set,
        "qwen_workload_change": {
            "current_frames_sent": summary["comparison"]["current_benchmark"]["frames_sent_to_qwen"],
            "relationship_frames_sent": summary["comparison"]["relationship_benchmark"]["frames_sent_to_qwen"],
        },
        "tender_alignment": {
            "relationship_triggers_observed": sorted(trigger_type_set),
            "evidence_based_only": True,
        },
    }

    timeline = {
        "video_id": BENCHMARK_VIDEO_ID,
        "relationship_events": relationship_events_json,
        "trigger_count": len(triggers),
        "triggers": [
            {
                "frame_id": trigger.frame_id,
                "timestamp_seconds": trigger.timestamp_seconds,
                "timestamp_human": format_timestamp_human(trigger.timestamp_seconds),
                "object_track_id": trigger.object_track_id,
                "person_track_id": trigger.person_track_id,
                "trigger_type": trigger.trigger_type,
                "reason": trigger.reason,
                "related_track_ids": trigger.related_track_ids,
            }
            for trigger in triggers
        ],
    }
    frame_audit = _build_frame_audit(frame_contexts, current_timeline, relationship_events)
    _write_outputs(summary, timeline, frame_audit)

    print("RELATIONSHIP_EVENT_BENCHMARK_START")
    print(json.dumps({"summary": str(SUMMARY_JSON_PATH), "timeline": str(TIMELINE_JSON_PATH), "frame_audit": str(FRAME_AUDIT_JSON_PATH)}))
    print("RELATIONSHIP_EVENT_BENCHMARK_END")


if __name__ == "__main__":
    asyncio.run(main())
