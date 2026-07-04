from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


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

VEHICLE_CLASS_NAMES = {"car", "truck", "bus", "motorcycle", "bicycle"}
BAG_CLASS_NAMES = {"backpack", "handbag", "suitcase"}


def _safe_round(value: float) -> float:
    return round(float(value), 6)


def _crop_image(image, bbox_xyxy: list[float]):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox_xyxy]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return image[y1:y2, x1:x2]


def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
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
    denom = area_a + area_b - inter_area
    return inter_area / denom if denom > 0 else 0.0


def _classify_bgr_color(bgr_pixel) -> str:
    b, g, r = [int(v) for v in bgr_pixel.tolist()]
    sample = np.uint8([[[b, g, r]]])
    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = [int(x) for x in sample.tolist()]
    if v < 40:
        return "dark"
    if s < 30:
        if v > 210:
            return "white"
        if v > 130:
            return "grey"
        return "black"
    if h < 8 or h >= 170:
        return "red"
    if h < 18:
        return "orange"
    if h < 32:
        return "yellow"
    if h < 85:
        return "green"
    if h < 130:
        return "blue"
    if h < 150:
        return "purple"
    if h < 170:
        return "pink"
    return "brown"


def _dominant_color_label(image) -> str:
    if image is None or image.size == 0:
        return "unknown"
    trimmed = image
    height, width = image.shape[:2]
    if height > 12 and width > 12:
        pad_h = max(1, int(height * 0.12))
        pad_w = max(1, int(width * 0.12))
        trimmed = image[pad_h: height - pad_h, pad_w: width - pad_w]
        if trimmed.size == 0:
            trimmed = image
    small = cv2.resize(trimmed, (24, 24), interpolation=cv2.INTER_AREA)
    pixels = small.reshape((-1, 3))
    sample_pixel = pixels.mean(axis=0)
    return _classify_bgr_color(sample_pixel)


def _build_detection_appearance_terms(detection: dict[str, Any], crop_image, all_detections: list[dict[str, Any]]) -> list[str]:
    class_name = str(detection.get("class_name", "")).strip().lower()
    color_label = _dominant_color_label(crop_image)
    terms = [class_name]
    if class_name == "person":
        carrying_bag = any(
            str(other.get("class_name", "")).strip().lower() in BAG_CLASS_NAMES
            and _bbox_iou(detection.get("bbox_xyxy", []), other.get("bbox_xyxy", [])) > 0.01
            for other in all_detections
            if isinstance(other, dict)
        )
        upper_crop = crop_image[: max(1, crop_image.shape[0] // 2), :] if crop_image is not None and crop_image.size else crop_image
        lower_crop = crop_image[max(0, crop_image.shape[0] // 2):, :] if crop_image is not None and crop_image.size else crop_image
        upper_color = _dominant_color_label(upper_crop)
        lower_color = _dominant_color_label(lower_crop)
        terms.extend(
            [
                upper_color,
                f"{upper_color} shirt",
                f"{upper_color} upper clothing",
                lower_color,
                f"{lower_color} lower clothing",
            ]
        )
        if carrying_bag:
            terms.extend(["bag", "person with bag", "carrying bag"])
    elif class_name in VEHICLE_CLASS_NAMES:
        terms.extend(["vehicle", color_label, f"{color_label} {class_name}", f"{color_label} vehicle"])
        if color_label in {"white", "black", "grey", "red", "blue", "yellow", "green"}:
            terms.append(f"{color_label} car" if class_name == "car" else f"{color_label} {class_name}")
    else:
        terms.extend([color_label, f"{color_label} {class_name}"])
    cleaned_terms: list[str] = []
    for term in terms:
        term_str = str(term).strip().lower()
        if term_str and term_str not in cleaned_terms:
            cleaned_terms.append(term_str)
    return cleaned_terms


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
    object_crops_dir = run_dir / "11_yolo_object_crops"
    object_crops_dir.mkdir(parents=True, exist_ok=True)

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

        enriched_detections: list[dict[str, Any]] = []
        for detection_index, detection in enumerate(detections):
            detection_copy = dict(detection)
            bbox_xyxy = detection_copy.get("bbox_xyxy", [])
            crop_path_relative = None
            appearance_terms: list[str] = []
            if image is not None and isinstance(bbox_xyxy, list) and len(bbox_xyxy) == 4:
                crop_image = _crop_image(image, bbox_xyxy)
                if crop_image is not None and crop_image.size > 0:
                    crop_name = (
                        f"frame_{int(item.get('frame_idx', 0) or 0):06d}"
                        f"_{detection_index:02d}_{str(detection_copy.get('class_name', 'object')).strip().lower()}.jpg"
                    )
                    crop_path = object_crops_dir / crop_name
                    cv2.imwrite(str(crop_path), crop_image)
                    repo_root = Path(__file__).resolve().parents[2]
                    crop_path_relative = crop_path.resolve().relative_to(repo_root).as_posix()
                    appearance_terms = _build_detection_appearance_terms(detection_copy, crop_image, detections)
            detection_copy["crop_path"] = crop_path_relative
            detection_copy["appearance_terms"] = appearance_terms
            enriched_detections.append(detection_copy)

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
                "detections": enriched_detections,
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
    print(f"[tender-demo] Object crops folder path: {object_crops_dir}")
    return {
        "scored_items": scored_items,
        "report": report,
        "scores_output_path": str(scores_output_path),
        "report_output_path": str(report_output_path),
        "annotated_frames_dir": str(annotated_dir),
        "object_crops_dir": str(object_crops_dir),
    }
