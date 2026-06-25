import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import cv2

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT_PATH = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from app.core.config import PROJECT_ROOT
from app.core.utils import format_timestamp_human
from app.services.object_detection.detector import ObjectDetector
from app.services.object_detection.schemas import FrameDetection
from app.services.object_tracker import ObjectTrackerService
from app.services.ocr import OCRService


INPUT_VIDEO_PATH = Path(os.getenv("BENCHMARK_INPUT_VIDEO", r"C:\Mukul K\test_video\V_ai_test_2min.mp4"))
BENCHMARK_VIDEO_ID = os.getenv("BENCHMARK_VIDEO_ID", "11111111-2222-4333-8444-777777777777")

CASE_ROOT = PROJECT_ROOT / "tests" / "manual_benchmark_case"
DATA_ROOT = CASE_ROOT / "data"
INPUT_ROOT = DATA_ROOT / "input"
OUTPUT_ROOT = DATA_ROOT / "output"
FRAME_ROOT = PROJECT_ROOT / "data" / "frames" / BENCHMARK_VIDEO_ID
DETECTION_ROOT = PROJECT_ROOT / "data" / "detections" / BENCHMARK_VIDEO_ID

TIMELINES_JSON_PATH = OUTPUT_ROOT / "entity_timelines.json"
SUMMARY_MD_PATH = OUTPUT_ROOT / "entity_timeline_summary.md"
INPUT_COPY_PATH = INPUT_ROOT / INPUT_VIDEO_PATH.name

VEHICLE_CLASSES = {
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
}

STATIONARY_SPEED_PIXELS_PER_SECOND = 8.0
OCR_NEARBY_METHOD = "same_frame_text_only_no_ocr_bounding_boxes_available"


@dataclass
class EntityObservation:
    frame_id: str
    timestamp_seconds: float
    timestamp_human: str
    bbox: List[float]
    centroid: Dict[str, float]
    confidence: float
    zone: str
    ocr_observed_nearby: List[str]
    license_plates_observed: List[str]


@dataclass
class EntityTimeline:
    track_id: int
    entity_type: str
    class_name: str
    observations: List[EntityObservation] = field(default_factory=list)


def _ensure_dirs() -> None:
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def _copy_input_video() -> None:
    if not INPUT_VIDEO_PATH.exists():
        raise FileNotFoundError(f"Input video not found: {INPUT_VIDEO_PATH}")
    shutil.copy2(INPUT_VIDEO_PATH, INPUT_COPY_PATH)


def _safe_remove(path: Path) -> None:
    if path.exists():
        path.unlink()


def _clean_previous_outputs() -> None:
    for path in (TIMELINES_JSON_PATH, SUMMARY_MD_PATH):
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
    existing = sorted(FRAME_ROOT.glob("frame_*.jpg"))
    if existing:
        return [(f"{video_id}_f{index:04d}", video_id, float(index - 1), path) for index, path in enumerate(existing, start=1)]

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


def _entity_type_for_class(class_name: str) -> str:
    normalized = class_name.lower()
    if normalized == "person":
        return "person"
    if normalized in VEHICLE_CLASSES:
        return "vehicle"
    return "object"


def _centroid(bbox: List[float]) -> Dict[str, float]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return {"x": (x1 + x2) / 2.0, "y": (y1 + y2) / 2.0}


def _zone_for_centroid(centroid: Dict[str, float], frame_width: int, frame_height: int) -> str:
    if frame_width <= 0 or frame_height <= 0:
        return "unknown"

    x_ratio = centroid["x"] / float(frame_width)
    y_ratio = centroid["y"] / float(frame_height)

    horizontal = "left" if x_ratio < 1.0 / 3.0 else "right" if x_ratio >= 2.0 / 3.0 else "center"
    vertical = "top" if y_ratio < 1.0 / 3.0 else "bottom" if y_ratio >= 2.0 / 3.0 else "middle"
    return f"{vertical}_{horizontal}"


def _movement_direction(start: Dict[str, float], end: Dict[str, float]) -> str:
    dx = end["x"] - start["x"]
    dy = end["y"] - start["y"]
    if math.hypot(dx, dy) <= STATIONARY_SPEED_PIXELS_PER_SECOND:
        return "stationary"

    horizontal = ""
    vertical = ""
    if abs(dx) >= STATIONARY_SPEED_PIXELS_PER_SECOND:
        horizontal = "right" if dx > 0 else "left"
    if abs(dy) >= STATIONARY_SPEED_PIXELS_PER_SECOND:
        vertical = "down" if dy > 0 else "up"

    if horizontal and vertical:
        return f"{vertical}_{horizontal}"
    return horizontal or vertical or "stationary"


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _build_timelines(
    extracted_tuples: List[Tuple[str, str, float, Path]],
    frame_detections: List[FrameDetection],
    tracking_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    frame_detection_lookup = {frame.frame_id: frame for frame in frame_detections}
    timelines: Dict[int, EntityTimeline] = {}
    ocr_cache: Dict[str, Dict[str, List[str]]] = {}

    for frame_id, _video_id, ts, frame_path in extracted_tuples:
        frame_detection = frame_detection_lookup[frame_id]
        ocr_result = ocr_cache.setdefault(str(frame_path), OCRService.extract_text(frame_path))
        frame_text = _dedupe_preserve_order(ocr_result.get("detected_text", []))
        frame_plates = _dedupe_preserve_order(ocr_result.get("license_plates", []))

        for entity in tracking_map.get(frame_id, {}).get("tracked_entities", []):
            track_id = int(entity["track_id"])
            class_name = str(entity.get("class_name", "unknown"))
            bbox = [float(value) for value in entity.get("bbox", [0.0, 0.0, 0.0, 0.0])]
            center = _centroid(bbox)
            timeline = timelines.setdefault(
                track_id,
                EntityTimeline(
                    track_id=track_id,
                    entity_type=_entity_type_for_class(class_name),
                    class_name=class_name,
                ),
            )
            timeline.observations.append(
                EntityObservation(
                    frame_id=frame_id,
                    timestamp_seconds=float(ts),
                    timestamp_human=format_timestamp_human(ts),
                    bbox=bbox,
                    centroid=center,
                    confidence=float(entity.get("confidence", 0.0)),
                    zone=_zone_for_centroid(center, frame_detection.frame_width, frame_detection.frame_height),
                    ocr_observed_nearby=frame_text,
                    license_plates_observed=frame_plates,
                )
            )

    return [_serialize_timeline(timeline) for timeline in sorted(timelines.values(), key=lambda item: item.track_id)]


def _serialize_timeline(timeline: EntityTimeline) -> Dict[str, Any]:
    observations = sorted(timeline.observations, key=lambda item: item.timestamp_seconds)
    first = observations[0]
    last = observations[-1]
    duration_seconds = max(0.0, last.timestamp_seconds - first.timestamp_seconds)

    distances = []
    segment_speeds = []
    directions = []
    stationary_seconds = 0.0
    zone_history = []
    last_zone: Optional[str] = None
    last_zone_start: Optional[EntityObservation] = None

    for previous, current in zip(observations, observations[1:]):
        dx = current.centroid["x"] - previous.centroid["x"]
        dy = current.centroid["y"] - previous.centroid["y"]
        distance = math.hypot(dx, dy)
        elapsed = max(0.0, current.timestamp_seconds - previous.timestamp_seconds)
        speed = distance / elapsed if elapsed > 0.0 else 0.0
        distances.append(distance)
        segment_speeds.append(speed)
        directions.append(_movement_direction(previous.centroid, current.centroid))
        if speed <= STATIONARY_SPEED_PIXELS_PER_SECOND:
            stationary_seconds += elapsed

    for observation in observations:
        if observation.zone != last_zone:
            if last_zone is not None and last_zone_start is not None:
                zone_history.append(
                    {
                        "zone": last_zone,
                        "entered_frame_id": last_zone_start.frame_id,
                        "entered_at_seconds": last_zone_start.timestamp_seconds,
                        "entered_at": last_zone_start.timestamp_human,
                        "left_frame_id": observation.frame_id,
                        "left_at_seconds": observation.timestamp_seconds,
                        "left_at": observation.timestamp_human,
                        "duration_seconds": max(0.0, observation.timestamp_seconds - last_zone_start.timestamp_seconds),
                    }
                )
            last_zone = observation.zone
            last_zone_start = observation

    if last_zone is not None and last_zone_start is not None:
        zone_history.append(
            {
                "zone": last_zone,
                "entered_frame_id": last_zone_start.frame_id,
                "entered_at_seconds": last_zone_start.timestamp_seconds,
                "entered_at": last_zone_start.timestamp_human,
                "left_frame_id": last.frame_id,
                "left_at_seconds": last.timestamp_seconds,
                "left_at": last.timestamp_human,
                "duration_seconds": max(0.0, last.timestamp_seconds - last_zone_start.timestamp_seconds),
            }
        )

    all_text = _dedupe_preserve_order([text for observation in observations for text in observation.ocr_observed_nearby])
    all_plates = _dedupe_preserve_order([plate for observation in observations for plate in observation.license_plates_observed])
    frame_indices = [_frame_index(observation.frame_id) for observation in observations]
    missing_frame_gaps = _missing_frame_gaps(frame_indices)
    primary_direction = _primary_value(directions) if directions else "single_frame"
    active_segments = sum(1 for speed in segment_speeds if speed > STATIONARY_SPEED_PIXELS_PER_SECOND)

    return {
        "track_id": timeline.track_id,
        "entity_type": timeline.entity_type,
        "class_name": timeline.class_name,
        "first_appearance": {
            "frame_id": first.frame_id,
            "timestamp_seconds": first.timestamp_seconds,
            "timestamp_human": first.timestamp_human,
        },
        "last_appearance": {
            "frame_id": last.frame_id,
            "timestamp_seconds": last.timestamp_seconds,
            "timestamp_human": last.timestamp_human,
        },
        "total_duration_seconds": duration_seconds,
        "frame_ids": [observation.frame_id for observation in observations],
        "frame_count": len(observations),
        "centroid_positions": [
            {
                "frame_id": observation.frame_id,
                "timestamp_seconds": observation.timestamp_seconds,
                "x": observation.centroid["x"],
                "y": observation.centroid["y"],
            }
            for observation in observations
        ],
        "movement_direction": primary_direction,
        "movement_direction_segments": directions,
        "average_speed_pixels_per_second": mean(segment_speeds) if segment_speeds else 0.0,
        "total_distance_pixels": sum(distances),
        "dwell_time_seconds": stationary_seconds,
        "ocr_observed_nearby": all_text,
        "license_plates_observed_nearby": all_plates,
        "ocr_nearby_method": OCR_NEARBY_METHOD,
        "zone_history": zone_history,
        "observations": [
            {
                "frame_id": observation.frame_id,
                "timestamp_seconds": observation.timestamp_seconds,
                "timestamp_human": observation.timestamp_human,
                "bbox": observation.bbox,
                "centroid": observation.centroid,
                "confidence": observation.confidence,
                "zone": observation.zone,
                "ocr_observed_nearby": observation.ocr_observed_nearby,
                "license_plates_observed": observation.license_plates_observed,
            }
            for observation in observations
        ],
        "track_quality": {
            "missing_frame_count": sum(missing_frame_gaps),
            "gap_count": len([gap for gap in missing_frame_gaps if gap > 0]),
            "max_gap_frames": max(missing_frame_gaps) if missing_frame_gaps else 0,
            "is_broken_track": any(gap > 0 for gap in missing_frame_gaps),
            "lost_at_end": False,
            "active_segment_count": active_segments,
        },
    }


def _frame_index(frame_id: str) -> Optional[int]:
    try:
        return int(frame_id.rsplit("_f", 1)[1])
    except (IndexError, ValueError):
        return None


def _missing_frame_gaps(frame_indices: List[Optional[int]]) -> List[int]:
    clean_indices = [index for index in frame_indices if index is not None]
    return [max(0, current - previous - 1) for previous, current in zip(clean_indices, clean_indices[1:])]


def _primary_value(values: List[str]) -> str:
    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _mark_lost_tracks(timelines: List[Dict[str, Any]], extracted_tuples: List[Tuple[str, str, float, Path]]) -> None:
    if not extracted_tuples:
        return
    final_timestamp = extracted_tuples[-1][2]
    frame_step = _estimate_frame_step_seconds(extracted_tuples)
    lost_margin = frame_step * ObjectTrackerService.MAX_MISSED_FRAMES
    for timeline in timelines:
        last_ts = float(timeline["last_appearance"]["timestamp_seconds"])
        timeline["track_quality"]["lost_at_end"] = (final_timestamp - last_ts) >= lost_margin


def _estimate_frame_step_seconds(extracted_tuples: List[Tuple[str, str, float, Path]]) -> float:
    deltas = [
        current[2] - previous[2]
        for previous, current in zip(extracted_tuples, extracted_tuples[1:])
        if current[2] > previous[2]
    ]
    return mean(deltas) if deltas else 1.0


def _build_summary(timelines: List[Dict[str, Any]], video_duration_seconds: float, runtime_seconds: float) -> Dict[str, Any]:
    durations = [float(timeline["total_duration_seconds"]) for timeline in timelines]
    frame_counts = [int(timeline["frame_count"]) for timeline in timelines]
    broken_tracks = [timeline for timeline in timelines if timeline["track_quality"]["is_broken_track"]]
    lost_tracks = [timeline for timeline in timelines if timeline["track_quality"]["lost_at_end"]]

    by_type: Dict[str, int] = {}
    for timeline in timelines:
        by_type[timeline["entity_type"]] = by_type.get(timeline["entity_type"], 0) + 1

    return {
        "input_video_path": str(INPUT_VIDEO_PATH),
        "input_copy_path": str(INPUT_COPY_PATH),
        "video_id": BENCHMARK_VIDEO_ID,
        "video_duration_seconds": video_duration_seconds,
        "wall_clock_runtime_seconds": runtime_seconds,
        "total_entities_tracked": len(timelines),
        "entities_by_type": by_type,
        "average_track_length_seconds": mean(durations) if durations else 0.0,
        "average_track_frame_count": mean(frame_counts) if frame_counts else 0.0,
        "broken_tracks": _track_refs(broken_tracks),
        "lost_tracks": _track_refs(lost_tracks),
        "longest_tracks": _track_refs(sorted(timelines, key=lambda item: item["total_duration_seconds"], reverse=True)[:10]),
        "most_active_tracks": _track_refs(
            sorted(
                timelines,
                key=lambda item: (item["track_quality"]["active_segment_count"], item["total_distance_pixels"]),
                reverse=True,
            )[:10]
        ),
        "output_files": {
            "entity_timelines_json": str(TIMELINES_JSON_PATH),
            "entity_timeline_summary_md": str(SUMMARY_MD_PATH),
        },
        "notes": [
            "This benchmark performs no event reasoning and no VLM reasoning.",
            "Detection, tracking, and OCR are invoked through production services without changing thresholds or inference settings.",
            f"OCR proximity uses `{OCR_NEARBY_METHOD}` because OCRService.extract_text does not expose text bounding boxes.",
            "Zone history uses deterministic 3x3 frame-position zones because no production polygon zone service is present.",
        ],
    }


def _track_refs(timelines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "track_id": timeline["track_id"],
            "entity_type": timeline["entity_type"],
            "class_name": timeline["class_name"],
            "first_appearance": timeline["first_appearance"],
            "last_appearance": timeline["last_appearance"],
            "total_duration_seconds": timeline["total_duration_seconds"],
            "frame_count": timeline["frame_count"],
            "total_distance_pixels": timeline["total_distance_pixels"],
            "active_segment_count": timeline["track_quality"]["active_segment_count"],
            "missing_frame_count": timeline["track_quality"]["missing_frame_count"],
            "gap_count": timeline["track_quality"]["gap_count"],
        }
        for timeline in timelines
    ]


def _write_outputs(summary: Dict[str, Any], timelines: List[Dict[str, Any]]) -> None:
    payload = {
        "benchmark": "entity_timeline_builder_phase_1",
        "summary": summary,
        "entity_timelines": timelines,
    }
    TIMELINES_JSON_PATH.write_text(json.dumps(payload, indent=4), encoding="utf-8")

    lines = [
        "# Entity Timeline Summary",
        "",
        f"- Input video: `{summary['input_video_path']}`",
        f"- Video duration: `{summary['video_duration_seconds']:.2f}s`",
        f"- Entities tracked: `{summary['total_entities_tracked']}`",
        f"- Entities by type: `{summary['entities_by_type']}`",
        f"- Average track length: `{summary['average_track_length_seconds']:.2f}s`",
        f"- Average track frame count: `{summary['average_track_frame_count']:.2f}`",
        f"- Broken tracks: `{len(summary['broken_tracks'])}`",
        f"- Lost tracks: `{len(summary['lost_tracks'])}`",
        f"- Wall-clock runtime: `{summary['wall_clock_runtime_seconds']:.2f}s`",
        "",
        "## Broken Tracks",
        "",
    ]
    lines.extend(_track_lines(summary["broken_tracks"]))
    lines.extend(["", "## Lost Tracks", ""])
    lines.extend(_track_lines(summary["lost_tracks"]))
    lines.extend(["", "## Longest Tracks", ""])
    lines.extend(_track_lines(summary["longest_tracks"]))
    lines.extend(["", "## Most Active Tracks", ""])
    lines.extend(_track_lines(summary["most_active_tracks"]))
    lines.extend(["", "## Notes", ""])
    for note in summary["notes"]:
        lines.append(f"- {note}")

    SUMMARY_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def _track_lines(rows: List[Dict[str, Any]]) -> List[str]:
    if not rows:
        return ["- None"]
    return [
        (
            f"- Track #{row['track_id']} ({row['entity_type']}/{row['class_name']}): "
            f"{row['total_duration_seconds']:.2f}s, frames={row['frame_count']}, "
            f"distance={row['total_distance_pixels']:.2f}px, gaps={row['gap_count']}"
        )
        for row in rows
    ]


def main() -> None:
    _ensure_dirs()
    _clean_previous_outputs()
    _copy_input_video()

    start = time.perf_counter()
    video_duration_seconds = _get_video_duration_seconds(INPUT_COPY_PATH)
    extracted_tuples = _extract_one_fps_frames(BENCHMARK_VIDEO_ID, INPUT_VIDEO_PATH)
    frame_detections = _load_or_run_detections(extracted_tuples)
    tracking_map = ObjectTrackerService.track_frames(frame_detections)
    timelines = _build_timelines(extracted_tuples, frame_detections, tracking_map)
    _mark_lost_tracks(timelines, extracted_tuples)
    summary = _build_summary(timelines, video_duration_seconds, time.perf_counter() - start)
    _write_outputs(summary, timelines)

    print("ENTITY_TIMELINE_BENCHMARK_START")
    print(json.dumps({"timelines": str(TIMELINES_JSON_PATH), "summary": str(SUMMARY_MD_PATH)}))
    print("ENTITY_TIMELINE_BENCHMARK_END")


if __name__ == "__main__":
    main()
