import asyncio
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT_PATH = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from app.core.config import PROJECT_ROOT, settings
from app.core.utils import format_timestamp_human
from app.services.event_aggregation import EventAggregationService
from app.services.frame import FrameExtractionService
from app.services.object_detection.detector import ObjectDetector
from app.services.object_tracker import ObjectTrackerService
from app.services.pipeline_contract import frame_catalog_path, frame_metadata_dir
from app.services.vlm_factory import get_vlm_service


INPUT_VIDEO_PATH = Path(r"C:\Mukul K\test_video\V_ai_test_2min.mp4")
BENCHMARK_VIDEO_ID = "11111111-2222-4333-8444-666666666666"
CASE_ROOT = PROJECT_ROOT / "tests" / "manual_benchmark_case"
DATA_ROOT = CASE_ROOT / "data"
INPUT_ROOT = DATA_ROOT / "input"
OUTPUT_ROOT = DATA_ROOT / "output"
PROJECT_VIDEO_PATH = settings.VIDEOS_DIR / f"{BENCHMARK_VIDEO_ID}{INPUT_VIDEO_PATH.suffix.lower()}"
PROJECT_METADATA_PATH = settings.METADATA_DIR / f"{BENCHMARK_VIDEO_ID}.json"
SUMMARY_JSON_PATH = OUTPUT_ROOT / "event_driven_summary.json"
SUMMARY_MD_PATH = OUTPUT_ROOT / "event_driven_summary.md"
TIMELINE_JSON_PATH = OUTPUT_ROOT / "event_driven_candidate_timeline.json"
OUTPUT_VIDEO_PATH = OUTPUT_ROOT / "event_driven_candidate_frames.mp4"
INPUT_COPY_PATH = INPUT_ROOT / INPUT_VIDEO_PATH.name
EVENT_CLUSTER_GAP_SECONDS = 4.0
LONG_CLUSTER_SECONDS = 10.0
INTRA_CLUSTER_CONTEXT_SPACING_SECONDS = 4.0


@dataclass
class CandidateEvent:
    event_index: int
    start_seconds: float
    end_seconds: float
    frame_ids: List[str]
    representative_frame_id: str
    score: float
    reasons: List[str]
    track_ids: List[int]
    object_counts: Dict[str, int]
    representative_frame_ids: List[str]


def _ensure_dirs() -> None:
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def _copy_input_video() -> None:
    if not INPUT_VIDEO_PATH.exists():
        raise FileNotFoundError(f"Input video not found: {INPUT_VIDEO_PATH}")
    shutil.copy2(INPUT_VIDEO_PATH, INPUT_COPY_PATH)
    shutil.copy2(INPUT_VIDEO_PATH, PROJECT_VIDEO_PATH)


def _write_project_metadata() -> None:
    metadata = {
        "video_id": BENCHMARK_VIDEO_ID,
        "filename": INPUT_VIDEO_PATH.name,
        "upload_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "file_size": PROJECT_VIDEO_PATH.stat().st_size,
        "upload_duration_ms": 0.0,
    }
    with open(PROJECT_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def _safe_remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


def _clean_previous_artifacts() -> None:
    _safe_remove(settings.FRAMES_DIR / BENCHMARK_VIDEO_ID)
    _safe_remove(frame_metadata_dir(BENCHMARK_VIDEO_ID))
    _safe_remove(settings.EVENTS_DIR / BENCHMARK_VIDEO_ID)
    _safe_remove(frame_catalog_path(BENCHMARK_VIDEO_ID))
    _safe_remove(settings.METADATA_DIR / f"{BENCHMARK_VIDEO_ID}_events.json")
    _safe_remove(PROJECT_METADATA_PATH)
    for output_path in (SUMMARY_JSON_PATH, SUMMARY_MD_PATH, TIMELINE_JSON_PATH, OUTPUT_VIDEO_PATH):
        _safe_remove(output_path)


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
    frame_dir = settings.FRAMES_DIR / video_id
    frame_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
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
            frame_path = frame_dir / f"frame_{frame_idx:04d}.jpg"
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"Failed to save frame {frame_id}")

            extracted_tuples.append((frame_id, video_id, second, frame_path))
            frame_idx += 1
            current_raw_frame += frame_interval
    finally:
        cap.release()

    return extracted_tuples


def _detect_and_track(
    extracted_tuples: List[Tuple[str, str, float, Path]]
) -> Tuple[List[Any], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    detector = ObjectDetector()
    frame_detections = [
        detector.detect_frame(path, frame_id, video_id, ts)
        for frame_id, video_id, ts, path in extracted_tuples
    ]
    tracking_map = ObjectTrackerService.track_frames(frame_detections)

    selection_map: Dict[str, Dict[str, Any]] = {}
    previous_dynamic_counts: Dict[str, int] = {}
    previous_dynamic_track_ids: List[int] = []

    for detection in frame_detections:
        tracking = tracking_map.get(detection.frame_id, {})
        tracked_entities = tracking.get("tracked_entities", [])
        dynamic_entities = [
            entity for entity in tracked_entities
            if str(entity.get("class_name", "")).lower() in {
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
                "sports ball",
            }
        ]
        dynamic_track_ids = [int(entity["track_id"]) for entity in dynamic_entities]
        dynamic_counts: Dict[str, int] = {}
        for entity in dynamic_entities:
            class_name = str(entity.get("class_name", "")).lower()
            dynamic_counts[class_name] = dynamic_counts.get(class_name, 0) + 1

        reasons: List[str] = []
        score = 0.0

        if dynamic_entities:
            reasons.append("dynamic_object_detected")
            score += 2.0
        if tracking.get("new_track_count", 0) > 0 and dynamic_track_ids:
            reasons.append("new_track")
            score += 3.0
        if tracking.get("ended_track_count", 0) > 0 and previous_dynamic_track_ids:
            reasons.append("track_ended")
            score += 3.0
        if dynamic_counts != previous_dynamic_counts and dynamic_counts:
            reasons.append("dynamic_count_changed")
            score += 2.0
        if dynamic_track_ids != previous_dynamic_track_ids and dynamic_track_ids:
            reasons.append("dynamic_track_set_changed")
            score += 2.0

        selection_map[detection.frame_id] = {
            "selected": bool(reasons),
            "candidate_reasons": reasons,
            "tracked_entities": dynamic_entities,
            "track_ids": dynamic_track_ids,
            "object_counts": dynamic_counts,
            "event_signal_score": score,
        }

        previous_dynamic_counts = dynamic_counts
        previous_dynamic_track_ids = dynamic_track_ids

    return frame_detections, tracking_map, selection_map


def _cluster_candidate_events(
    extracted_tuples: List[Tuple[str, str, float, Path]],
    selection_map: Dict[str, Dict[str, Any]],
) -> List[CandidateEvent]:
    frame_lookup = {frame_id: (video_id, ts, path) for frame_id, video_id, ts, path in extracted_tuples}
    active_frames: List[Tuple[str, float, Dict[str, Any]]] = []
    for frame_id, _video_id, ts, _path in extracted_tuples:
        selection = selection_map.get(frame_id, {})
        if selection.get("selected"):
            active_frames.append((frame_id, ts, selection))

    if not active_frames:
        return []

    clusters: List[List[Tuple[str, float, Dict[str, Any]]]] = []
    current_cluster: List[Tuple[str, float, Dict[str, Any]]] = [active_frames[0]]

    for item in active_frames[1:]:
        previous_ts = current_cluster[-1][1]
        if item[1] - previous_ts <= EVENT_CLUSTER_GAP_SECONDS:
            current_cluster.append(item)
        else:
            clusters.append(current_cluster)
            current_cluster = [item]
    clusters.append(current_cluster)

    candidate_events: List[CandidateEvent] = []
    for idx, cluster in enumerate(clusters, start=1):
        best_frame_id = cluster[0][0]
        best_score = -1.0
        all_reasons: List[str] = []
        track_ids: List[int] = []
        object_counts: Dict[str, int] = {}

        for frame_id, _ts, selection in cluster:
            score = float(selection.get("event_signal_score", 0.0))
            if score > best_score:
                best_score = score
                best_frame_id = frame_id
            all_reasons.extend(selection.get("candidate_reasons", []))
            track_ids.extend(selection.get("track_ids", []))
            for class_name, count in selection.get("object_counts", {}).items():
                object_counts[class_name] = max(object_counts.get(class_name, 0), int(count))

        start_seconds = cluster[0][1]
        end_seconds = cluster[-1][1]
        duration_bonus = min(3.0, max(0.0, end_seconds - start_seconds))
        representative_frame_ids = _select_cluster_frame_ids(cluster)

        candidate_events.append(
            CandidateEvent(
                event_index=idx,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                frame_ids=[frame_id for frame_id, _ts, _selection in cluster],
                representative_frame_id=best_frame_id,
                score=best_score + duration_bonus,
                reasons=list(dict.fromkeys(all_reasons)),
                track_ids=sorted(set(track_ids)),
                object_counts=object_counts,
                representative_frame_ids=representative_frame_ids,
            )
        )

    candidate_events.sort(key=lambda item: item.start_seconds)
    for idx, event in enumerate(candidate_events, start=1):
        event.event_index = idx
    return candidate_events


def _select_cluster_frame_ids(
    cluster: List[Tuple[str, float, Dict[str, Any]]]
) -> List[str]:
    if not cluster:
        return []

    chosen_indices = {0, len(cluster) - 1}
    best_index = 0
    best_score = -1.0
    previous_tracks: Tuple[int, ...] = tuple()
    previous_counts: Dict[str, int] = {}
    last_context_ts = cluster[0][1]

    for idx, (_frame_id, ts, selection) in enumerate(cluster):
        score = float(selection.get("event_signal_score", 0.0))
        if score > best_score:
            best_score = score
            best_index = idx

        current_tracks = tuple(selection.get("track_ids", []))
        current_counts = dict(selection.get("object_counts", {}))
        reasons = set(selection.get("candidate_reasons", []))

        if "new_track" in reasons or "track_ended" in reasons:
            chosen_indices.add(idx)
        if current_tracks != previous_tracks and current_tracks:
            chosen_indices.add(idx)
        if current_counts != previous_counts and current_counts:
            chosen_indices.add(idx)
        if (ts - last_context_ts) >= INTRA_CLUSTER_CONTEXT_SPACING_SECONDS:
            chosen_indices.add(idx)
            last_context_ts = ts

        previous_tracks = current_tracks
        previous_counts = current_counts

    chosen_indices.add(best_index)

    duration_seconds = cluster[-1][1] - cluster[0][1]
    if duration_seconds >= LONG_CLUSTER_SECONDS and len(cluster) >= 3:
        mid_index = len(cluster) // 2
        chosen_indices.add(mid_index)

    ordered_indices = sorted(chosen_indices)
    return [cluster[idx][0] for idx in ordered_indices]


def _build_candidate_analysis_tuples(
    extracted_tuples: List[Tuple[str, str, float, Path]],
    candidate_events: List[CandidateEvent],
) -> List[Tuple[str, str, float, Path, Path, Dict[str, Any]]]:
    analysis_pool = FrameExtractionService._build_temporal_context_strips(
        extracted_tuples=extracted_tuples,
        frame_dir=settings.FRAMES_DIR / BENCHMARK_VIDEO_ID,
    )
    analysis_map = {frame_id: (video_id, ts, analysis_path, original_path) for frame_id, video_id, ts, analysis_path, original_path in analysis_pool}

    tuples: List[Tuple[str, str, float, Path, Path, Dict[str, Any]]] = []
    for event in candidate_events:
        for frame_id in event.representative_frame_ids:
            video_id, ts, analysis_path, original_path = analysis_map[frame_id]
            detection_context = {
                "candidate_reasons": [f"event_{event.event_index}"] + event.reasons,
                "track_ids": event.track_ids,
                "detected_objects": [
                    {"class_name": class_name, "confidence": 1.0, "bbox": []}
                    for class_name in sorted(event.object_counts.keys())
                ],
                "benchmark_event_window": {
                    "start_seconds": event.start_seconds,
                    "end_seconds": event.end_seconds,
                    "frame_ids": event.frame_ids,
                    "representative_frame_ids": event.representative_frame_ids,
                },
            }
            tuples.append((frame_id, video_id, ts, analysis_path, original_path, detection_context))
    return tuples


async def _run_qwen(
    analysis_tuples: List[Tuple[str, str, float, Path, Path, Dict[str, Any]]]
) -> Tuple[List[Dict[str, Any]], int, int]:
    if not analysis_tuples:
        return [], 0, 0

    old_batch_size = settings.BATCH_SIZE
    settings.BATCH_SIZE = 4
    rich_frames: List[Dict[str, Any]] = []

    try:
        vlm_service = get_vlm_service()
        batch_size = settings.BATCH_SIZE
        for index in range(0, len(analysis_tuples), batch_size):
            batch = analysis_tuples[index:index + batch_size]
            batch_results = await vlm_service.generate_metadata_batch(batch)
            for rich_meta, _timings in batch_results:
                rich_frames.append(rich_meta.model_dump())
    finally:
        settings.BATCH_SIZE = old_batch_size

    successful = len(rich_frames)
    failed = max(0, len(analysis_tuples) - successful)
    return rich_frames, successful, failed


def _render_candidate_video(candidates: List[Dict[str, Any]]) -> None:
    if not candidates:
        return

    first_image = cv2.imread(str(Path(candidates[0]["frame_path"])))
    if first_image is None:
        return

    height, width = first_image.shape[:2]
    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        1.0,
        (width, height),
    )

    try:
        for candidate in candidates:
            image = cv2.imread(str(Path(candidate["frame_path"])))
            if image is None:
                continue

            cv2.rectangle(image, (0, 0), (width, 116), (0, 0, 0), thickness=-1)
            cv2.putText(image, f"Frame: {candidate['frame_id']}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Time: {candidate['timestamp_human']} ({candidate['timestamp_seconds']:.1f}s)", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Event: {candidate['event_label']} | Score: {candidate['event_score']:.1f}", (12, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
            reason_text = ", ".join(candidate["candidate_reasons"])
            cv2.putText(image, f"Reasons: {reason_text[:110]}", (12, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
            writer.write(image)
    finally:
        writer.release()


def _write_summary(summary: Dict[str, Any]) -> None:
    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    lines = [
        "# Event-Driven Benchmark Summary",
        "",
        f"- Input video: `{summary['input_video_path']}`",
        f"- Output candidate video: `{summary['output_video_path']}`",
        f"- Timeline JSON: `{summary['timeline_json_path']}`",
        f"- VLM engine: `{summary['vlm_engine_type']}`",
        f"- Batch size: `{summary['batch_size']}`",
        f"- Source video duration: `{summary['video_duration_seconds']:.2f}s`",
        f"- Wall-clock latency: `{summary['wall_clock_seconds']:.2f}s`",
        f"- Realtime ratio: `{summary['realtime_ratio']:.3f}x`",
        f"- Faster than video length: `{summary['faster_than_video_length']}`",
        f"- Total extracted at 1 fps: `{summary['total_frames_extracted']}`",
        f"- YOLO analyzed frames: `{summary['yolo_frames_analyzed']}`",
        f"- Candidate event clusters: `{summary['candidate_event_count']}`",
        f"- Frames sent to Qwen: `{summary['frames_sent_to_qwen']}`",
        f"- Successful Qwen frames: `{summary['successful_frames']}`",
        f"- Failed Qwen frames: `{summary['failed_frames']}`",
        f"- Aggregated events: `{summary['event_count']}`",
    ]
    with open(SUMMARY_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


async def main() -> None:
    _ensure_dirs()
    _clean_previous_artifacts()
    _copy_input_video()
    _write_project_metadata()

    start = time.perf_counter()
    extracted_tuples = _extract_one_fps_frames(BENCHMARK_VIDEO_ID, PROJECT_VIDEO_PATH)
    frame_detections, tracking_map, selection_map = _detect_and_track(extracted_tuples)
    candidate_events = _cluster_candidate_events(extracted_tuples, selection_map)
    analysis_tuples = _build_candidate_analysis_tuples(extracted_tuples, candidate_events)
    rich_frames, successful_frames, failed_frames = await _run_qwen(analysis_tuples)
    wall_clock_seconds = time.perf_counter() - start

    catalog_path = frame_catalog_path(BENCHMARK_VIDEO_ID)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(rich_frames, f, indent=4)

    metadata_dir = frame_metadata_dir(BENCHMARK_VIDEO_ID)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for frame in rich_frames:
        frame_path = metadata_dir / f"{frame['frame_id']}.json"
        with open(frame_path, "w", encoding="utf-8") as f:
            json.dump(frame, f, indent=4)

    events = EventAggregationService.process_events(BENCHMARK_VIDEO_ID, rich_frames)

    representative_lookup: Dict[str, CandidateEvent] = {}
    for event in candidate_events:
        for frame_id in event.representative_frame_ids:
            representative_lookup[frame_id] = event
    frame_lookup = {frame_id: path for frame_id, _video_id, _ts, path in extracted_tuples}
    timeline: List[Dict[str, Any]] = []
    render_candidates: List[Dict[str, Any]] = []
    for frame_id, video_id, ts, _analysis_path, original_path, detection_context in analysis_tuples:
        event = representative_lookup.get(frame_id)
        row = {
            "frame_id": frame_id,
            "video_id": video_id,
            "timestamp_seconds": ts,
            "timestamp_human": format_timestamp_human(ts),
            "frame_path": str(original_path),
            "event_label": f"event_{event.event_index}" if event else "event_unknown",
            "event_score": event.score if event else 0.0,
            "event_start_seconds": event.start_seconds if event else ts,
            "event_end_seconds": event.end_seconds if event else ts,
            "candidate_reasons": detection_context.get("candidate_reasons", []),
            "track_ids": detection_context.get("track_ids", []),
            "detected_objects": detection_context.get("detected_objects", []),
            "cluster_frame_ids": event.frame_ids if event else [frame_id],
        }
        timeline.append(row)
        render_candidates.append(row)

    with open(TIMELINE_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(timeline, f, indent=4)

    _render_candidate_video(render_candidates)

    video_duration_seconds = _get_video_duration_seconds(INPUT_COPY_PATH)
    realtime_ratio = (wall_clock_seconds / video_duration_seconds) if video_duration_seconds > 0 else 0.0
    faster_than_video_length = wall_clock_seconds < video_duration_seconds if video_duration_seconds > 0 else None

    summary = {
        "input_video_path": str(INPUT_VIDEO_PATH),
        "input_copy_path": str(INPUT_COPY_PATH),
        "output_video_path": str(OUTPUT_VIDEO_PATH),
        "timeline_json_path": str(TIMELINE_JSON_PATH),
        "video_id": BENCHMARK_VIDEO_ID,
        "vlm_engine_type": settings.VLM_ENGINE_TYPE,
        "batch_size": 4,
        "video_duration_seconds": video_duration_seconds,
        "wall_clock_seconds": wall_clock_seconds,
        "realtime_ratio": realtime_ratio,
        "faster_than_video_length": faster_than_video_length,
        "total_frames_extracted": len(extracted_tuples),
        "yolo_frames_analyzed": len(frame_detections),
        "tracked_frame_count": len(tracking_map),
        "candidate_event_count": len(candidate_events),
        "frames_sent_to_qwen": len(analysis_tuples),
        "successful_frames": successful_frames,
        "failed_frames": failed_frames,
        "event_count": len(events),
        "event_ids": [event.get("event_id") for event in events],
        "candidate_events": [
            {
                "event_index": event.event_index,
                "start_seconds": event.start_seconds,
                "end_seconds": event.end_seconds,
                "frame_ids": event.frame_ids,
                "representative_frame_id": event.representative_frame_id,
                "representative_frame_ids": event.representative_frame_ids,
                "score": event.score,
                "reasons": event.reasons,
                "track_ids": event.track_ids,
                "object_counts": event.object_counts,
            }
            for event in candidate_events
        ],
    }
    _write_summary(summary)

    print("EVENT_DRIVEN_BENCHMARK_SUMMARY_START")
    print(json.dumps(summary))
    print("EVENT_DRIVEN_BENCHMARK_SUMMARY_END")


if __name__ == "__main__":
    asyncio.run(main())
