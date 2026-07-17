from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_object_reviews_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "local_object_id",
        "manual_real_object_id",
        "object_family",
        "automated_class",
        "manual_class",
        "start_timestamp",
        "end_timestamp",
        "object_review_status",
        "crop_review_status",
        "timeline_review_status",
        "class_review_status",
        "downstream_decision",
        "same_real_object_as",
        "reviewer_notes",
        "reviewed_at",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def export_summary_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Manual Review Summary",
        "",
        "## Review Coverage",
        f"- Total local objects: {summary['total_local_objects']}",
        f"- Reviewed local objects: {summary['reviewed_local_objects']}",
        f"- Unreviewed local objects: {summary['unreviewed_local_objects']}",
        "",
        "## Object Decisions",
        f"- Correct single objects: {summary['correct_single_objects']}",
        f"- Fragmented objects: {summary['fragmented_objects']}",
        f"- Duplicate tracks: {summary['duplicate_tracks']}",
        f"- False detections: {summary['false_detections']}",
        f"- Wrong classes: {summary['wrong_classes']}",
        f"- Track switches: {summary['track_switches']}",
        f"- Uncertain objects: {summary['uncertain_objects']}",
        "",
        "## Manual Counts",
        f"- Manual estimated real object count: {summary['manual_real_object_count']}",
        f"- Manual vehicles count: {summary['manual_vehicles_count']}",
        f"- Manual persons count: {summary['manual_persons_count']}",
        "",
        "## Merge Review",
        f"- Accepted merges correct: {summary['accepted_merges_correct']}",
        f"- Accepted merges incorrect: {summary['accepted_merges_incorrect']}",
        f"- Possible merges accepted: {summary['possible_merges_accepted']}",
        f"- Possible merges rejected: {summary['possible_merges_rejected']}",
        "",
        "## Crop Review",
        f"- Good primary crops: {summary['good_primary_crops']}",
        f"- Bad primary crops: {summary['bad_primary_crops']}",
        f"- Alternative crops preferred: {summary['alternative_crops_preferred']}",
        f"- All crops bad: {summary['all_crops_bad']}",
        "",
        "## Downstream",
        f"- Ready after review: {summary['ready_after_review']}",
        f"- Fallback after review: {summary['fallback_after_review']}",
        f"- Manual review remaining: {summary['manual_review_remaining']}",
        f"- Rejected after review: {summary['rejected_after_review']}",
        "",
        "## Comparison Metrics",
        f"- Automated local object count: {summary['automated_local_object_count']}",
        f"- Manual real object count: {summary['manual_real_object_count']}",
        f"- Overcount: {summary['overcount']}",
        f"- Undercount: {summary['undercount']}",
        f"- Fragmented local object count: {summary['fragmented_local_object_count']}",
        f"- False detection count: {summary['false_detection_count']}",
        f"- Incorrect merge count: {summary['incorrect_merge_count']}",
        f"- Missed merge count: {summary['missed_merge_count']}",
        "",
        "## Reviewed-Only Ratios",
        f"- Precision-like object validity ratio: {summary['precision_like_object_validity_ratio']}",
        f"- Fragmentation ratio: {summary['fragmentation_ratio']}",
        f"- False-positive ratio: {summary['false_positive_ratio']}",
        f"- Class accuracy: {summary['class_accuracy']}",
        f"- Primary-crop usability rate: {summary['primary_crop_usability_rate']}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
