import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict
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
OVERLAY_FRAME_ROOT = OUTPUT_ROOT / "perception_overlay_frames"

PERCEPTION_AUDIT_JSON_PATH = OUTPUT_ROOT / "perception_audit.json"
PERCEPTION_SUMMARY_MD_PATH = OUTPUT_ROOT / "perception_summary.md"
PER_CLASS_STATS_JSON_PATH = OUTPUT_ROOT / "per_class_statistics.json"
TRACK_STATS_JSON_PATH = OUTPUT_ROOT / "track_statistics.json"
TENDER_OBJECT_REPORT_JSON_PATH = OUTPUT_ROOT / "tender_object_report.json"
OVERLAY_VIDEO_PATH = OUTPUT_ROOT / "perception_frame_overlay.mp4"
INPUT_COPY_PATH = INPUT_ROOT / INPUT_VIDEO_PATH.name

LEGACY_SUMMARY_JSON_PATH = OUTPUT_ROOT / "perception_audit_summary.json"
LEGACY_SUMMARY_MD_PATH = OUTPUT_ROOT / "perception_audit_summary.md"
LEGACY_FRAME_AUDIT_JSON_PATH = OUTPUT_ROOT / "perception_frame_audit.json"
LEGACY_CLASS_STATS_JSON_PATH = OUTPUT_ROOT / "perception_class_statistics.json"
LEGACY_TRACKER_STATS_JSON_PATH = OUTPUT_ROOT / "perception_tracker_statistics.json"

VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"}
PERSON_CLASSES = {"person"}
TENDER_OBJECTS = [
    "handbag",
    "backpack",
    "suitcase",
    "laptop bag",
    "parcel",
    "bottle",
    "phone",
    "box",
    "helmet",
    "fire extinguisher",
    "any movable object",
]
TENDER_CLASS_ALIASES = {
    "handbag": ["handbag"],
    "backpack": ["backpack"],
    "suitcase": ["suitcase"],
    "laptop bag": ["laptop bag", "laptop", "backpack", "handbag"],
    "parcel": ["parcel", "package", "box"],
    "bottle": ["bottle"],
    "phone": ["cell phone", "phone"],
    "box": ["box", "package", "parcel"],
    "helmet": ["helmet", "sports ball"],
    "fire extinguisher": ["fire extinguisher"],
    "any movable object": [
        "handbag",
        "backpack",
        "suitcase",
        "bottle",
        "cell phone",
        "phone",
        "cup",
        "book",
        "laptop",
        "box",
        "parcel",
        "package",
    ],
}
MOVABLE_OBJECT_CLASSES = set(TENDER_CLASS_ALIASES["any movable object"])


def _ensure_dirs() -> None:
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OVERLAY_FRAME_ROOT.mkdir(parents=True, exist_ok=True)


def _safe_remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


def _clean_previous_outputs() -> None:
    _safe_remove(OVERLAY_FRAME_ROOT)
    OVERLAY_FRAME_ROOT.mkdir(parents=True, exist_ok=True)
    for path in (
        PERCEPTION_AUDIT_JSON_PATH,
        PERCEPTION_SUMMARY_MD_PATH,
        PER_CLASS_STATS_JSON_PATH,
        TRACK_STATS_JSON_PATH,
        TENDER_OBJECT_REPORT_JSON_PATH,
        OVERLAY_VIDEO_PATH,
        LEGACY_SUMMARY_JSON_PATH,
        LEGACY_SUMMARY_MD_PATH,
        LEGACY_FRAME_AUDIT_JSON_PATH,
        LEGACY_CLASS_STATS_JSON_PATH,
        LEGACY_TRACKER_STATS_JSON_PATH,
    ):
        _safe_remove(path)


def _copy_input_video() -> None:
    if not INPUT_VIDEO_PATH.exists():
        raise FileNotFoundError(f"Input video not found: {INPUT_VIDEO_PATH}")
    shutil.copy2(INPUT_VIDEO_PATH, INPUT_COPY_PATH)


def _get_video_duration_seconds(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    return float(total_frames / fps) if fps > 0.0 else 0.0


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


def _area_metrics(bbox: List[float], frame_width: int, frame_height: int) -> Dict[str, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area = width * height
    image_area = max(1.0, float(frame_width * frame_height))
    return {
        "bbox_width": width,
        "bbox_height": height,
        "bbox_area": area,
        "bbox_size": {"width": width, "height": height, "area": area},
        "image_occupancy_percent": (area / image_area) * 100.0,
    }


def _center(bbox: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _category_for_class(class_name: str) -> str:
    normalized = class_name.lower()
    if normalized in PERSON_CLASSES:
        return "person"
    if normalized in VEHICLE_CLASSES:
        return "vehicle"
    if normalized in MOVABLE_OBJECT_CLASSES:
        return "tender_object"
    return "object"


def _match_track(detection: Any, tracked_entities: List[Dict[str, Any]], used_track_ids: set[int]) -> Optional[Dict[str, Any]]:
    for entity in tracked_entities:
        track_id = int(entity.get("track_id"))
        if track_id in used_track_ids:
            continue
        if entity.get("class_name") == detection.class_name and entity.get("bbox") == detection.bbox:
            used_track_ids.add(track_id)
            return entity
    for entity in tracked_entities:
        track_id = int(entity.get("track_id"))
        if track_id in used_track_ids:
            continue
        if entity.get("class_name") == detection.class_name:
            used_track_ids.add(track_id)
            return entity
    return None


def _build_raw_observations(
    extracted_tuples: List[Tuple[str, str, float, Path]],
    frame_detections: List[FrameDetection],
    tracking_map: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    frame_rows: List[Dict[str, Any]] = []
    tracks: Dict[int, Dict[str, Any]] = {}
    prior_track_ids: set[int] = set()
    frame_detection_lookup = {frame.frame_id: frame for frame in frame_detections}

    for frame_number, (frame_id, _video_id, ts, frame_path) in enumerate(extracted_tuples, start=1):
        frame_detection = frame_detection_lookup[frame_id]
        tracking = tracking_map.get(frame_id, {})
        tracked_entities = tracking.get("tracked_entities", [])
        used_track_ids: set[int] = set()
        current_track_ids = {int(tid) for tid in tracking.get("track_ids", [])}
        ended_track_ids = sorted(prior_track_ids - current_track_ids) if prior_track_ids else []
        ocr_result = OCRService.extract_text(frame_path)
        detections = []

        for detection in frame_detection.detections:
            matched = _match_track(detection, tracked_entities, used_track_ids)
            track_id = int(matched["track_id"]) if matched is not None else None
            bbox = [float(v) for v in detection.bbox]
            center_x, center_y = _center(bbox)
            metrics = _area_metrics(bbox, frame_detection.frame_width, frame_detection.frame_height)
            category = _category_for_class(detection.class_name)

            if track_id is not None:
                track_entry = tracks.setdefault(
                    track_id,
                    {
                        "track_id": track_id,
                        "class_name": detection.class_name,
                        "category": category,
                        "birth_frame": frame_id,
                        "birth_frame_number": frame_number,
                        "birth_timestamp_seconds": float(ts),
                        "death_frame": frame_id,
                        "death_frame_number": frame_number,
                        "death_timestamp_seconds": float(ts),
                        "observations": [],
                        "confidences": [],
                        "frame_numbers": [],
                    },
                )
                track_entry["death_frame"] = frame_id
                track_entry["death_frame_number"] = frame_number
                track_entry["death_timestamp_seconds"] = float(ts)
                track_entry["observations"].append(
                    {
                        "frame_id": frame_id,
                        "frame_number": frame_number,
                        "timestamp_seconds": float(ts),
                        "bbox": bbox,
                        "centroid": {"x": center_x, "y": center_y},
                        "confidence": float(detection.confidence),
                    }
                )
                track_entry["confidences"].append(float(detection.confidence))
                track_entry["frame_numbers"].append(frame_number)

            detections.append(
                {
                    "class_name": detection.class_name,
                    "category": category,
                    "confidence": float(detection.confidence),
                    "bbox": bbox,
                    "center": {"x": center_x, "y": center_y},
                    "track_id": track_id,
                    "track_confidence": float(matched.get("confidence", detection.confidence)) if matched is not None else None,
                    "is_new_track": track_id is not None and track_id not in prior_track_ids,
                    "is_ended_track_in_this_frame": track_id in ended_track_ids if track_id is not None else False,
                    **metrics,
                }
            )

        class_names = [row["class_name"] for row in detections]
        frame_rows.append(
            {
                "frame_number": frame_number,
                "frame_id": frame_id,
                "timestamp_seconds": float(ts),
                "timestamp_human": format_timestamp_human(ts),
                "frame_filename": frame_path.name,
                "frame_path": str(frame_path),
                "frame_resolution": {
                    "width": int(frame_detection.frame_width),
                    "height": int(frame_detection.frame_height),
                },
                "detected_classes": sorted(set(class_names)),
                "number_of_objects": len(detections),
                "number_of_persons": sum(1 for row in detections if row["category"] == "person"),
                "number_of_vehicles": sum(1 for row in detections if row["category"] == "vehicle"),
                "number_of_tender_objects": sum(1 for row in detections if row["category"] == "tender_object"),
                "new_tracks": sorted(current_track_ids - prior_track_ids),
                "new_track_count": tracking.get("new_track_count", 0),
                "ended_tracks": ended_track_ids,
                "ended_track_count": len(ended_track_ids),
                "ocr_text": ocr_result.get("detected_text", []),
                "ocr_license_plates": ocr_result.get("license_plates", []),
                "detections": detections,
            }
        )
        prior_track_ids = current_track_ids

    return frame_rows, tracks


def _finalize_tracks(tracks: Dict[int, Dict[str, Any]], total_frames: int) -> List[Dict[str, Any]]:
    birth_by_class_frame = defaultdict(list)
    death_by_class_frame = defaultdict(list)
    for track in tracks.values():
        birth_by_class_frame[(track["class_name"], track["birth_frame_number"])].append(track["track_id"])
        death_by_class_frame[(track["class_name"], track["death_frame_number"])].append(track["track_id"])

    rows = []
    for track_id, track in sorted(tracks.items()):
        observations = track["observations"]
        frame_numbers = track["frame_numbers"]
        gaps = [max(0, current - previous - 1) for previous, current in zip(frame_numbers, frame_numbers[1:])]
        distances = []
        speeds = []
        for previous, current in zip(observations, observations[1:]):
            dx = current["centroid"]["x"] - previous["centroid"]["x"]
            dy = current["centroid"]["y"] - previous["centroid"]["y"]
            distance = math.hypot(dx, dy)
            elapsed = max(0.0, current["timestamp_seconds"] - previous["timestamp_seconds"])
            distances.append(distance)
            speeds.append(distance / elapsed if elapsed > 0.0 else 0.0)

        birth = track["birth_frame_number"]
        death = track["death_frame_number"]
        same_class_prior_deaths = []
        same_class_next_births = []
        for frame_number in (birth - 1, birth):
            same_class_prior_deaths.extend(death_by_class_frame.get((track["class_name"], frame_number), []))
        for frame_number in (death, death + 1):
            same_class_next_births.extend(birth_by_class_frame.get((track["class_name"], frame_number), []))

        merged_candidates = sorted(tid for tid in same_class_prior_deaths if tid != track_id)
        split_candidates = sorted(tid for tid in same_class_next_births if tid != track_id)
        rows.append(
            {
                "track_id": track_id,
                "class_name": track["class_name"],
                "category": track["category"],
                "birth_frame": track["birth_frame"],
                "birth_frame_number": birth,
                "birth_timestamp_seconds": track["birth_timestamp_seconds"],
                "birth_timestamp_human": format_timestamp_human(track["birth_timestamp_seconds"]),
                "death_frame": track["death_frame"],
                "death_frame_number": death,
                "death_timestamp_seconds": track["death_timestamp_seconds"],
                "death_timestamp_human": format_timestamp_human(track["death_timestamp_seconds"]),
                "lifetime_seconds": float(track["death_timestamp_seconds"] - track["birth_timestamp_seconds"]),
                "frames_present": len(frame_numbers),
                "interrupted": any(gap > 0 for gap in gaps),
                "missing_frame_count": sum(gaps),
                "interruption_count": sum(1 for gap in gaps if gap > 0),
                "merged": bool(merged_candidates),
                "merged_candidate_track_ids": merged_candidates,
                "split": bool(split_candidates),
                "split_candidate_track_ids": split_candidates,
                "confidence": mean(track["confidences"]) if track["confidences"] else 0.0,
                "min_confidence": min(track["confidences"]) if track["confidences"] else 0.0,
                "max_confidence": max(track["confidences"]) if track["confidences"] else 0.0,
                "average_speed_pixels_per_second": mean(speeds) if speeds else 0.0,
                "total_distance_pixels": sum(distances),
                "observations": observations,
                "lost_before_final_frame": death < total_frames,
            }
        )
    return rows


def _attach_track_lifetimes_to_frames(frame_rows: List[Dict[str, Any]], track_rows: List[Dict[str, Any]]) -> None:
    track_lookup = {row["track_id"]: row for row in track_rows}
    for frame in frame_rows:
        lifetimes = []
        for detection in frame["detections"]:
            track_id = detection.get("track_id")
            if track_id is None or track_id not in track_lookup:
                continue
            track = track_lookup[track_id]
            detection["track_lifetime_seconds"] = track["lifetime_seconds"]
            lifetimes.append(
                {
                    "track_id": track_id,
                    "class_name": track["class_name"],
                    "birth_frame": track["birth_frame"],
                    "death_frame": track["death_frame"],
                    "lifetime_seconds": track["lifetime_seconds"],
                }
            )
        frame["track_lifetimes"] = lifetimes


def _build_class_statistics(frame_rows: List[Dict[str, Any]], track_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_class: Dict[str, Dict[str, Any]] = {}
    for frame in frame_rows:
        for detection in frame["detections"]:
            class_name = detection["class_name"]
            stats = by_class.setdefault(
                class_name,
                {
                    "class_name": class_name,
                    "category": detection["category"],
                    "detections": 0,
                    "confidences": [],
                    "frames_seen": set(),
                    "bbox_areas": [],
                    "bbox_widths": [],
                    "bbox_heights": [],
                    "track_ids": set(),
                },
            )
            stats["detections"] += 1
            stats["confidences"].append(float(detection["confidence"]))
            stats["frames_seen"].add(frame["frame_id"])
            stats["bbox_areas"].append(float(detection["bbox_area"]))
            stats["bbox_widths"].append(float(detection["bbox_width"]))
            stats["bbox_heights"].append(float(detection["bbox_height"]))
            if detection.get("track_id") is not None:
                stats["track_ids"].add(int(detection["track_id"]))

    tracks_by_class = defaultdict(list)
    for track in track_rows:
        tracks_by_class[track["class_name"]].append(track)

    result = {}
    for class_name, stats in sorted(by_class.items()):
        confidences = stats["confidences"]
        class_tracks = tracks_by_class.get(class_name, [])
        result[class_name] = {
            "class_name": class_name,
            "category": stats["category"],
            "detections": stats["detections"],
            "average_confidence": mean(confidences) if confidences else 0.0,
            "minimum_confidence": min(confidences) if confidences else 0.0,
            "maximum_confidence": max(confidences) if confidences else 0.0,
            "frames_seen": len(stats["frames_seen"]),
            "frame_ids_seen": sorted(stats["frames_seen"]),
            "track_count": len(stats["track_ids"]),
            "average_box_size": {
                "width": mean(stats["bbox_widths"]) if stats["bbox_widths"] else 0.0,
                "height": mean(stats["bbox_heights"]) if stats["bbox_heights"] else 0.0,
                "area": mean(stats["bbox_areas"]) if stats["bbox_areas"] else 0.0,
            },
            "average_lifetime_seconds": mean([track["lifetime_seconds"] for track in class_tracks]) if class_tracks else 0.0,
            "average_speed_pixels_per_second": mean([track["average_speed_pixels_per_second"] for track in class_tracks]) if class_tracks else 0.0,
            "track_fragmentation": {
                "interrupted_tracks": sum(1 for track in class_tracks if track["interrupted"]),
                "total_interruptions": sum(track["interruption_count"] for track in class_tracks),
                "total_missing_frames": sum(track["missing_frame_count"] for track in class_tracks),
                "merged_tracks": sum(1 for track in class_tracks if track["merged"]),
                "split_tracks": sum(1 for track in class_tracks if track["split"]),
            },
        }
    return result


def _build_tender_report(class_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for tender_class in TENDER_OBJECTS:
        aliases = TENDER_CLASS_ALIASES.get(tender_class, [tender_class])
        matched = [class_stats[alias] for alias in aliases if alias in class_stats]
        detections = sum(row["detections"] for row in matched)
        track_count = sum(row["track_count"] for row in matched)
        frame_ids = sorted({frame_id for row in matched for frame_id in row["frame_ids_seen"]})
        confidences = [row["average_confidence"] for row in matched if row["detections"] > 0]
        rows.append(
            {
                "tender_class": tender_class,
                "aliases_checked": aliases,
                "detected": detections > 0,
                "never_detected": detections == 0,
                "tracked": track_count > 0,
                "detections": detections,
                "track_count": track_count,
                "average_confidence": mean(confidences) if confidences else 0.0,
                "frames_seen": len(frame_ids),
                "frame_ids_seen": frame_ids,
            }
        )
    return rows


def _draw_overlay_frame(frame: Dict[str, Any]) -> Path:
    image = cv2.imread(frame["frame_path"])
    if image is None:
        raise RuntimeError(f"Unable to read frame for overlay: {frame['frame_path']}")
    for detection in frame["detections"]:
        x1, y1, x2, y2 = [int(float(v)) for v in detection["bbox"]]
        color = (0, 255, 0)
        if detection["category"] == "person":
            color = (255, 180, 0)
        elif detection["category"] == "vehicle":
            color = (0, 180, 255)
        elif detection["category"] == "tender_object":
            color = (255, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{detection['class_name']} t{detection.get('track_id')} {detection['confidence']:.2f}"
        cv2.rectangle(image, (x1, max(0, y1 - 24)), (min(image.shape[1], x1 + 270), y1), (0, 0, 0), -1)
        cv2.putText(image, label, (x1 + 4, max(16, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    header = (
        f"Frame {frame['frame_number']} | {frame['timestamp_human']} | "
        f"objects={frame['number_of_objects']} persons={frame['number_of_persons']} "
        f"vehicles={frame['number_of_vehicles']} tender={frame['number_of_tender_objects']}"
    )
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(image, header[:110], (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    out_path = OVERLAY_FRAME_ROOT / f"{frame['frame_number']:04d}_{frame['frame_id']}.jpg"
    cv2.imwrite(str(out_path), image)
    return out_path


def _write_overlay_video(frame_rows: List[Dict[str, Any]]) -> List[str]:
    overlay_paths = [_draw_overlay_frame(frame) for frame in frame_rows]
    if not overlay_paths:
        return []
    first = cv2.imread(str(overlay_paths[0]))
    if first is None:
        return [str(path) for path in overlay_paths]
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(OVERLAY_VIDEO_PATH), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (width, height))
    try:
        for path in overlay_paths:
            image = cv2.imread(str(path))
            if image is None:
                continue
            if image.shape[1] != width or image.shape[0] != height:
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(image)
    finally:
        writer.release()
    return [str(path) for path in overlay_paths]


def _build_conclusions(class_stats: Dict[str, Any], track_rows: List[Dict[str, Any]], tender_report: List[Dict[str, Any]]) -> List[str]:
    conclusions = []
    if class_stats:
        dominant = sorted(class_stats.values(), key=lambda row: row["detections"], reverse=True)[:5]
        conclusions.append("Dominant detected classes: " + ", ".join(f"{row['class_name']} ({row['detections']})" for row in dominant) + ".")
    reliable = [row for row in class_stats.values() if row["frames_seen"] >= 3 and row["average_confidence"] >= 0.5]
    if reliable:
        conclusions.append("Reliably visible classes by frequency/confidence: " + ", ".join(sorted(row["class_name"] for row in reliable)) + ".")
    missing_tender = [row["tender_class"] for row in tender_report if row["never_detected"]]
    if missing_tender:
        conclusions.append("Tender objects completely missing: " + ", ".join(missing_tender) + ".")
    unstable = [track for track in track_rows if track["interrupted"] or track["merged"] or track["split"]]
    conclusions.append(f"Unstable tracks observed: {len(unstable)} of {len(track_rows)} tracks.")
    weak_classes = [
        row["class_name"]
        for row in class_stats.values()
        if row["average_confidence"] < 0.5 or row["track_fragmentation"]["interrupted_tracks"] > 0
    ]
    if weak_classes:
        conclusions.append("Classes needing perception review: " + ", ".join(sorted(weak_classes)) + ".")
    return conclusions


def _write_outputs(
    audit: Dict[str, Any],
    frame_rows: List[Dict[str, Any]],
    class_stats: Dict[str, Any],
    track_rows: List[Dict[str, Any]],
    tender_report: List[Dict[str, Any]],
) -> None:
    PERCEPTION_AUDIT_JSON_PATH.write_text(json.dumps(audit, indent=4), encoding="utf-8")
    PER_CLASS_STATS_JSON_PATH.write_text(json.dumps(class_stats, indent=4), encoding="utf-8")
    TRACK_STATS_JSON_PATH.write_text(json.dumps({"tracks": track_rows}, indent=4), encoding="utf-8")
    TENDER_OBJECT_REPORT_JSON_PATH.write_text(json.dumps(tender_report, indent=4), encoding="utf-8")

    # Legacy aliases retained for older benchmark notebooks.
    LEGACY_SUMMARY_JSON_PATH.write_text(json.dumps(audit, indent=4), encoding="utf-8")
    LEGACY_FRAME_AUDIT_JSON_PATH.write_text(json.dumps(frame_rows, indent=4), encoding="utf-8")
    LEGACY_CLASS_STATS_JSON_PATH.write_text(json.dumps(class_stats, indent=4), encoding="utf-8")
    LEGACY_TRACKER_STATS_JSON_PATH.write_text(json.dumps({"tracks": track_rows}, indent=4), encoding="utf-8")

    summary = audit["summary"]
    lines = [
        "# Perception Audit",
        "",
        f"- Input video: `{summary['input_video_path']}`",
        f"- Video duration: `{summary['video_duration_seconds']:.2f}s`",
        f"- Frames analyzed: `{summary['total_frames_analyzed']}`",
        f"- Total detections: `{summary['total_detections']}`",
        f"- Total tracks: `{summary['total_tracks']}`",
        f"- Overlay video: `{summary['overlay_video_path']}`",
        "",
        "## Detected Classes",
        "",
    ]
    for class_name, row in class_stats.items():
        lines.append(
            f"- {class_name}: detections={row['detections']}, frames={row['frames_seen']}, "
            f"avg_conf={row['average_confidence']:.3f}, tracks={row['track_count']}, "
            f"avg_lifetime={row['average_lifetime_seconds']:.2f}s, avg_speed={row['average_speed_pixels_per_second']:.2f}px/s"
        )
    lines.extend(["", "## Tender Object Audit", ""])
    for row in tender_report:
        lines.append(
            f"- {row['tender_class']}: detected={row['detected']}, never_detected={row['never_detected']}, "
            f"tracked={row['tracked']}, detections={row['detections']}, frames={row['frames_seen']}, "
            f"avg_conf={row['average_confidence']:.3f}"
        )
    lines.extend(["", "## Track Quality", ""])
    lines.append(f"- Interrupted tracks: `{summary['interrupted_track_count']}`")
    lines.append(f"- Merge candidates: `{summary['merged_track_count']}`")
    lines.append(f"- Split candidates: `{summary['split_track_count']}`")
    lines.extend(["", "## Evidence-Based Answers", ""])
    for conclusion in audit["conclusions"]:
        lines.append(f"- {conclusion}")
    PERCEPTION_SUMMARY_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    LEGACY_SUMMARY_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _ensure_dirs()
    _clean_previous_outputs()
    _copy_input_video()

    start = time.perf_counter()
    video_duration_seconds = _get_video_duration_seconds(INPUT_COPY_PATH)
    extracted_tuples = _extract_one_fps_frames(BENCHMARK_VIDEO_ID, INPUT_VIDEO_PATH)
    frame_detections = _load_or_run_detections(extracted_tuples)
    tracking_map = ObjectTrackerService.track_frames(frame_detections)
    frame_rows, raw_tracks = _build_raw_observations(extracted_tuples, frame_detections, tracking_map)
    track_rows = _finalize_tracks(raw_tracks, len(frame_rows))
    _attach_track_lifetimes_to_frames(frame_rows, track_rows)
    class_stats = _build_class_statistics(frame_rows, track_rows)
    tender_report = _build_tender_report(class_stats)
    overlay_paths = _write_overlay_video(frame_rows)
    conclusions = _build_conclusions(class_stats, track_rows, tender_report)

    total_detections = sum(len(frame["detections"]) for frame in frame_rows)
    audit = {
        "benchmark": "perception_audit",
        "rules": {
            "benchmark_only": True,
            "production_code_modified": False,
            "production_pipeline_changed": False,
            "detector_replaced": False,
            "tracker_replaced": False,
        },
        "summary": {
            "input_video_path": str(INPUT_VIDEO_PATH),
            "input_copy_path": str(INPUT_COPY_PATH),
            "video_id": BENCHMARK_VIDEO_ID,
            "video_duration_seconds": video_duration_seconds,
            "total_frames_analyzed": len(frame_rows),
            "total_detections": total_detections,
            "total_tracks": len(track_rows),
            "detected_classes": sorted(class_stats.keys()),
            "dominant_classes": [
                {"class_name": row["class_name"], "detections": row["detections"]}
                for row in sorted(class_stats.values(), key=lambda item: item["detections"], reverse=True)[:10]
            ],
            "missing_tender_objects": [row["tender_class"] for row in tender_report if row["never_detected"]],
            "interrupted_track_count": sum(1 for track in track_rows if track["interrupted"]),
            "merged_track_count": sum(1 for track in track_rows if track["merged"]),
            "split_track_count": sum(1 for track in track_rows if track["split"]),
            "wall_clock_runtime_seconds": time.perf_counter() - start,
            "overlay_video_path": str(OVERLAY_VIDEO_PATH),
            "overlay_frame_folder": str(OVERLAY_FRAME_ROOT),
            "perception_audit_json": str(PERCEPTION_AUDIT_JSON_PATH),
            "perception_summary_md": str(PERCEPTION_SUMMARY_MD_PATH),
            "per_class_statistics_json": str(PER_CLASS_STATS_JSON_PATH),
            "track_statistics_json": str(TRACK_STATS_JSON_PATH),
            "tender_object_report_json": str(TENDER_OBJECT_REPORT_JSON_PATH),
        },
        "frame_audit": frame_rows,
        "class_statistics": class_stats,
        "track_statistics": {"tracks": track_rows},
        "tender_object_report": tender_report,
        "overlay_frames": overlay_paths,
        "conclusions": conclusions,
    }
    _write_outputs(audit, frame_rows, class_stats, track_rows, tender_report)

    print("PERCEPTION_AUDIT_START")
    print(
        json.dumps(
            {
                "perception_audit": str(PERCEPTION_AUDIT_JSON_PATH),
                "perception_summary": str(PERCEPTION_SUMMARY_MD_PATH),
                "per_class_statistics": str(PER_CLASS_STATS_JSON_PATH),
                "track_statistics": str(TRACK_STATS_JSON_PATH),
                "tender_object_report": str(TENDER_OBJECT_REPORT_JSON_PATH),
                "overlay_video": str(OVERLAY_VIDEO_PATH),
            }
        )
    )
    print("PERCEPTION_AUDIT_END")


if __name__ == "__main__":
    main()
