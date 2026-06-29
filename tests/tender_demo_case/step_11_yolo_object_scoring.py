from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2


IMPORTANT_OBJECT_CLASSES = {
    "backpack",
    "handbag",
    "suitcase",
    "cell phone",
    "knife",
    "bottle",
    "box",
    "sports ball",
}


def _safe_round(value: float) -> float:
    return round(float(value), 6)


def _load_yolo_detections(run_dir: Path) -> list[dict[str, Any]]:
    detections_path = run_dir / "10_yolo_detections.json"
    if not detections_path.exists():
        raise FileNotFoundError(f"Missing YOLO detections file: {detections_path}")

    detections = json.loads(detections_path.read_text(encoding="utf-8"))
    if not isinstance(detections, list):
        raise ValueError(f"Expected a list in YOLO detections file: {detections_path}")
    return detections


def _to_abs_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / path


def _largest_bbox_area_ratio(detections: list[dict[str, Any]], image_area: float) -> float:
    if image_area <= 0:
        return 0.0
    largest_area = 0.0
    for detection in detections:
        largest_area = max(largest_area, float(detection.get("bbox_area", 0.0) or 0.0))
    return largest_area / image_area if largest_area > 0 else 0.0


def _build_evidence_labels(
    person_count: int,
    vehicle_count: int,
    important_object_count: int,
    motion_score_norm: float,
    bbox_prominence_score: float,
    detection_count: int,
) -> list[str]:
    labels: list[str] = []
    if person_count > 0:
        labels.append("person_present")
    if person_count >= 2:
        labels.append("multiple_people")
    if vehicle_count > 0:
        labels.append("vehicle_present")
    if important_object_count > 0:
        labels.append("important_object_present")
    if motion_score_norm >= 0.7:
        labels.append("high_motion_context")
    if bbox_prominence_score >= 0.5:
        labels.append("large_visible_object")
    if detection_count == 0:
        labels.append("no_object_detected")
    return labels


def _annotate_frame(
    frame_path: Path,
    detections: list[dict[str, Any]],
    output_dir: Path,
    frame_idx: int,
) -> str | None:
    image = cv2.imread(str(frame_path))
    if image is None:
        print(f"[tender-demo] Warning: unable to read frame for annotation: {frame_path}")
        return None

    for detection in detections:
        bbox = detection.get("bbox_xyxy", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
        class_name = str(detection.get("class_name", "object"))
        confidence = float(detection.get("confidence", 0.0) or 0.0)
        label = f"{class_name} {confidence:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(image, (x1, max(0, y1 - 24)), (x1 + 220, y1), (0, 255, 0), thickness=-1)
        cv2.putText(
            image,
            label,
            (x1 + 4, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    output_path = output_dir / f"yolo_annotated_{frame_idx:06d}.jpg"
    cv2.imwrite(str(output_path), image)
    repo_root = Path(__file__).resolve().parents[2]
    return output_path.resolve().relative_to(repo_root).as_posix()


def _top_detected_classes(scored_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    class_counts: dict[str, int] = {}
    for item in scored_items:
        for class_name, count in item.get("object_counts", {}).items():
            class_counts[class_name] = class_counts.get(class_name, 0) + int(count)
    return [
        {"class_name": class_name, "count": count}
        for class_name, count in sorted(class_counts.items(), key=lambda pair: (-pair[1], pair[0]))
    ]


def _build_usefulness_summary(
    frames_with_detections: int,
    frames_with_person: int,
    frames_with_important_objects: int,
    important_classes_present: list[str],
) -> str:
    if frames_with_detections == 0:
        return "YOLO did not detect useful objects in the selected frames."

    parts: list[str] = []
    if frames_with_person > 0:
        parts.append(
            "YOLO detected people in many selected motion frames, which confirms the motion segments contain human activity."
        )
    if frames_with_important_objects > 0 and important_classes_present:
        parts.append(
            "YOLO also detected important object classes such as "
            + ", ".join(important_classes_present[:5])
            + ", which may be useful for later object-aware event scoring."
        )
    if not parts:
        parts.append(
            "YOLO produced limited detections on the selected frames. Later stages should rely more on motion and VLM context for this video."
        )
    return " ".join(parts)


def _build_recommendation(
    frames_with_detections: int,
    frames_with_person: int,
    frames_with_important_objects: int,
) -> str:
    if frames_with_detections == 0:
        return "Do not use YOLO scoring for this video unless the confidence threshold or input scope is adjusted."
    if frames_with_important_objects > 0:
        return "Use important object detections as supporting evidence in later suspicious-event scoring."
    if frames_with_person > 0:
        return "Use YOLO person detections in a later step to improve event importance scoring."
    return "Use YOLO detections as optional supporting context in later object-aware event scoring."


def run_yolo_object_scoring(run_dir: Path) -> dict[str, Any]:
    print("[tender-demo] Starting Step 11: YOLO object usefulness scoring")
    yolo_items = _load_yolo_detections(run_dir)

    annotated_dir = run_dir / "11_yolo_annotated_frames"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    scored_items: list[dict[str, Any]] = []
    frames_with_detections = 0
    total_detections = 0
    frames_with_person = 0
    frames_with_multiple_people = 0
    frames_with_vehicle = 0
    frames_with_important_objects = 0

    for item in yolo_items:
        detections = item.get("detections", [])
        if not isinstance(detections, list):
            detections = []

        frame_path = _to_abs_path(str(item.get("frame_path", "")))
        image = cv2.imread(str(frame_path))
        image_area = 0.0
        if image is not None:
            image_area = float(image.shape[0] * image.shape[1])

        detection_count = int(item.get("detection_count", len(detections)) or 0)
        person_count = int(item.get("person_count", 0) or 0)
        vehicle_count = int(item.get("vehicle_count", 0) or 0)
        important_object_count = int(item.get("important_object_count", 0) or 0)
        unique_object_class_count = len(item.get("object_classes_present", []))
        motion_score_norm = float(item.get("motion_score_norm", 0.0) or 0.0)

        person_presence_score = 1.0 if person_count > 0 else 0.0
        person_count_score = min(person_count / 3.0, 1.0)
        vehicle_score = min(vehicle_count / 2.0, 1.0)
        important_object_score = min(important_object_count / 2.0, 1.0)
        object_diversity_score = min(unique_object_class_count / 5.0, 1.0)
        largest_bbox_area_ratio = _largest_bbox_area_ratio(detections, image_area)
        bbox_prominence_score = min(largest_bbox_area_ratio * 5.0, 1.0)
        motion_context_score = motion_score_norm

        object_importance_score = _safe_round(
            (0.25 * person_presence_score)
            + (0.20 * person_count_score)
            + (0.10 * vehicle_score)
            + (0.15 * important_object_score)
            + (0.10 * object_diversity_score)
            + (0.10 * bbox_prominence_score)
            + (0.10 * motion_context_score)
        )

        evidence_labels = _build_evidence_labels(
            person_count=person_count,
            vehicle_count=vehicle_count,
            important_object_count=important_object_count,
            motion_score_norm=motion_score_norm,
            bbox_prominence_score=bbox_prominence_score,
            detection_count=detection_count,
        )

        annotated_frame_path = None
        if detections:
            annotated_frame_path = _annotate_frame(
                frame_path=frame_path,
                detections=detections,
                output_dir=annotated_dir,
                frame_idx=int(item.get("frame_idx", 0) or 0),
            )

        if detection_count > 0:
            frames_with_detections += 1
        if person_count > 0:
            frames_with_person += 1
        if person_count >= 2:
            frames_with_multiple_people += 1
        if vehicle_count > 0:
            frames_with_vehicle += 1
        if important_object_count > 0:
            frames_with_important_objects += 1
        total_detections += detection_count

        scored_items.append(
            {
                **item,
                "score_components": {
                    "person_presence_score": _safe_round(person_presence_score),
                    "person_count_score": _safe_round(person_count_score),
                    "vehicle_score": _safe_round(vehicle_score),
                    "important_object_score": _safe_round(important_object_score),
                    "object_diversity_score": _safe_round(object_diversity_score),
                    "bbox_prominence_score": _safe_round(bbox_prominence_score),
                    "motion_context_score": _safe_round(motion_context_score),
                },
                "object_importance_score": object_importance_score,
                "evidence_labels": evidence_labels,
                "annotated_frame_path": annotated_frame_path,
            }
        )

    scored_items.sort(key=lambda item: float(item.get("object_importance_score", 0.0)), reverse=True)

    scores_output_path = run_dir / "11_yolo_object_scores.json"
    scores_output_path.write_text(json.dumps(scored_items, indent=2), encoding="utf-8")

    top_detected_classes = _top_detected_classes(scored_items)
    important_classes_present = [
        item["class_name"] for item in top_detected_classes if item["class_name"] in IMPORTANT_OBJECT_CLASSES
    ]
    top_object_frames = [
        {
            "frame_idx": item.get("frame_idx"),
            "timestamp_seconds": item.get("timestamp_seconds"),
            "object_importance_score": item.get("object_importance_score"),
            "person_count": item.get("person_count"),
            "vehicle_count": item.get("vehicle_count"),
            "important_object_count": item.get("important_object_count"),
            "object_classes_present": item.get("object_classes_present", []),
            "evidence_labels": item.get("evidence_labels", []),
            "frame_path": item.get("frame_path"),
            "annotated_frame_path": item.get("annotated_frame_path"),
        }
        for item in scored_items[:10]
    ]

    report = {
        "total_frames_analyzed": len(scored_items),
        "frames_with_detections": frames_with_detections,
        "frames_without_detections": len(scored_items) - frames_with_detections,
        "total_detections": total_detections,
        "frames_with_person": frames_with_person,
        "frames_with_multiple_people": frames_with_multiple_people,
        "frames_with_vehicle": frames_with_vehicle,
        "frames_with_important_objects": frames_with_important_objects,
        "top_detected_classes": top_detected_classes,
        "top_object_frames": top_object_frames,
        "yolo_usefulness_summary": _build_usefulness_summary(
            frames_with_detections=frames_with_detections,
            frames_with_person=frames_with_person,
            frames_with_important_objects=frames_with_important_objects,
            important_classes_present=important_classes_present,
        ),
        "recommendation": _build_recommendation(
            frames_with_detections=frames_with_detections,
            frames_with_person=frames_with_person,
            frames_with_important_objects=frames_with_important_objects,
        ),
    }

    report_output_path = run_dir / "11_yolo_usefulness_report.json"
    report_output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[tender-demo] Total frames analyzed: {len(scored_items)}")
    print(f"[tender-demo] Frames with detections: {frames_with_detections}")
    print(f"[tender-demo] Frames with person: {frames_with_person}")
    print(f"[tender-demo] Frames with multiple people: {frames_with_multiple_people}")
    print(f"[tender-demo] Frames with vehicles: {frames_with_vehicle}")
    print(f"[tender-demo] Frames with important objects: {frames_with_important_objects}")
    print(f"[tender-demo] Top detected classes: {top_detected_classes[:10]}")
    print(f"[tender-demo] YOLO object scores output path: {scores_output_path}")
    print(f"[tender-demo] YOLO usefulness report output path: {report_output_path}")
    print(f"[tender-demo] Annotated frames folder path: {annotated_dir}")
    return {
        "scored_items": scored_items,
        "report": report,
        "scores_output_path": str(scores_output_path),
        "report_output_path": str(report_output_path),
        "annotated_frames_dir": str(annotated_dir),
    }
