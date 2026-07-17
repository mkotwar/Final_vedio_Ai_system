from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def build_manual_ground_truth(
    *,
    camera_id: str,
    video_name: str,
    packages: list[dict[str, Any]],
    object_reviews: list[dict[str, Any]],
) -> dict[str, Any]:
    package_by_id = {int(item["local_object_id"]): item for item in packages}
    grouped: dict[str, dict[str, Any]] = {}
    uncertain_groups: list[dict[str, Any]] = []
    for review in object_reviews:
        local_object_id = int(review["local_object_id"])
        package = package_by_id.get(local_object_id)
        if package is None:
            continue
        manual_real_object_id = str(review.get("manual_real_object_id", "")).strip()
        related_ids = [local_object_id, *list(review.get("same_real_object_as_local_object_ids", []))]
        if not manual_real_object_id:
            manual_real_object_id = f"{package['object_family']}_{local_object_id:04d}"
        target = grouped.setdefault(
            manual_real_object_id,
            {
                "manual_real_object_id": manual_real_object_id,
                "object_family": package["object_family"],
                "manual_class": review.get("manual_class") or package["final_class"],
                "matched_local_object_ids": [],
                "matched_raw_track_ids": [],
                "visible_start_timestamp": package["start_timestamp_seconds"],
                "visible_end_timestamp": package["end_timestamp_seconds"],
                "review_status": "confirmed",
                "notes": "",
            },
        )
        target["matched_local_object_ids"] = sorted(set([*target["matched_local_object_ids"], *related_ids]))
        target["matched_raw_track_ids"] = sorted(
            set([*target["matched_raw_track_ids"], *list(package.get("source_raw_track_ids", []))])
        )
        target["visible_start_timestamp"] = min(
            float(target["visible_start_timestamp"]),
            float(package["start_timestamp_seconds"]),
        )
        target["visible_end_timestamp"] = max(
            float(target["visible_end_timestamp"]),
            float(package["end_timestamp_seconds"]),
        )
        if review.get("object_review_status") == "uncertain":
            target["review_status"] = "uncertain"
        notes = str(review.get("reviewer_notes", "")).strip()
        if notes:
            target["notes"] = notes
        if review.get("object_review_status") in {"fragmented_object", "duplicate_track"} and not related_ids:
            uncertain_groups.append(
                {
                    "manual_real_object_id": manual_real_object_id,
                    "local_object_id": local_object_id,
                    "notes": notes,
                }
            )
    return {
        "camera_id": camera_id,
        "video_name": video_name,
        "objects": sorted(grouped.values(), key=lambda item: str(item["manual_real_object_id"])),
        "uncertain_groups": uncertain_groups,
    }


def build_progress(
    *,
    total_objects: int,
    packages: list[dict[str, Any]],
    object_reviews: list[dict[str, Any]],
    merge_reviews: list[dict[str, Any]],
    possible_merge_reviews: list[dict[str, Any]],
    accepted_merge_count: int,
    possible_merge_count: int,
) -> dict[str, Any]:
    reviewed_ids = {int(item["local_object_id"]) for item in object_reviews}
    vehicle_ids = {int(item["local_object_id"]) for item in packages if item.get("object_family") == "vehicle"}
    person_ids = {int(item["local_object_id"]) for item in packages if item.get("object_family") == "person"}
    last_reviewed = max(reviewed_ids) if reviewed_ids else None
    return {
        "total_objects": total_objects,
        "reviewed_objects": len(reviewed_ids),
        "unreviewed_objects": max(0, total_objects - len(reviewed_ids)),
        "reviewed_vehicle_objects": len(reviewed_ids & vehicle_ids),
        "reviewed_person_objects": len(reviewed_ids & person_ids),
        "accepted_merges_reviewed": len(merge_reviews),
        "accepted_merges_total": accepted_merge_count,
        "possible_merges_reviewed": len(possible_merge_reviews),
        "possible_merges_total": possible_merge_count,
        "last_reviewed_object_id": last_reviewed,
        "last_updated_timestamp": object_reviews[-1]["reviewed_at"] if object_reviews else "",
    }


def build_summary(
    *,
    packages: list[dict[str, Any]],
    object_reviews: list[dict[str, Any]],
    merge_reviews: list[dict[str, Any]],
    possible_merge_reviews: list[dict[str, Any]],
    accepted_merge_count: int,
    possible_merge_count: int,
    manual_ground_truth: dict[str, Any],
) -> dict[str, Any]:
    total_local_objects = len(packages)
    reviewed_local_objects = len(object_reviews)
    package_by_id = {int(item["local_object_id"]): item for item in packages}
    object_status_counts = Counter(str(item.get("object_review_status", "uncertain")) for item in object_reviews)
    crop_status_counts = Counter(str(item.get("crop_review_status", "crop_uncertain")) for item in object_reviews)
    downstream_counts = Counter(str(item.get("downstream_decision", "manual_review")) for item in object_reviews)
    merge_counts = Counter(str(item.get("decision", "uncertain")) for item in merge_reviews)
    possible_merge_counts = Counter(str(item.get("decision", "uncertain")) for item in possible_merge_reviews)

    manual_objects = list(manual_ground_truth.get("objects", []))
    family_counts = Counter(str(item.get("object_family", "")) for item in manual_objects)

    valid_objects = object_status_counts["correct_single_object"] + object_status_counts["fragmented_object"] + object_status_counts["duplicate_track"]
    class_correct = 0
    for review in object_reviews:
        local_object_id = int(review["local_object_id"])
        package = package_by_id.get(local_object_id)
        if package is None:
            continue
        if str(review.get("class_review_status")) == "class_correct":
            class_correct += 1
            continue
        manual_class = str(review.get("manual_class", "")).strip()
        if manual_class and manual_class == str(package.get("final_class")):
            class_correct += 1

    summary = {
        "total_local_objects": total_local_objects,
        "reviewed_local_objects": reviewed_local_objects,
        "unreviewed_local_objects": max(0, total_local_objects - reviewed_local_objects),
        "correct_single_objects": object_status_counts["correct_single_object"],
        "fragmented_objects": object_status_counts["fragmented_object"],
        "duplicate_tracks": object_status_counts["duplicate_track"],
        "false_detections": object_status_counts["false_detection"],
        "wrong_classes": object_status_counts["wrong_class"],
        "track_switches": object_status_counts["track_switch"],
        "uncertain_objects": object_status_counts["uncertain"],
        "manual_real_object_count": len(manual_objects),
        "manual_vehicles_count": family_counts["vehicle"],
        "manual_persons_count": family_counts["person"],
        "accepted_merges_correct": merge_counts["merge_correct"],
        "accepted_merges_incorrect": merge_counts["merge_incorrect"],
        "possible_merges_accepted": possible_merge_counts["accept_merge"],
        "possible_merges_rejected": possible_merge_counts["reject_merge"],
        "good_primary_crops": crop_status_counts["primary_crop_good"],
        "bad_primary_crops": crop_status_counts["crop_contains_wrong_object"]
        + crop_status_counts["crop_too_blurry"]
        + crop_status_counts["crop_too_small"]
        + crop_status_counts["crop_too_clipped"]
        + crop_status_counts["all_crops_bad"],
        "alternative_crops_preferred": crop_status_counts["alternative_crop_better"],
        "all_crops_bad": crop_status_counts["all_crops_bad"],
        "ready_after_review": downstream_counts["ready"],
        "fallback_after_review": downstream_counts["fallback"],
        "manual_review_remaining": max(0, total_local_objects - reviewed_local_objects) + downstream_counts["manual_review"],
        "rejected_after_review": downstream_counts["reject"],
        "estimated_fragmentation_rate": _ratio(object_status_counts["fragmented_object"], reviewed_local_objects),
        "estimated_false_positive_rate": _ratio(object_status_counts["false_detection"], reviewed_local_objects),
        "estimated_class_error_rate": _ratio(object_status_counts["wrong_class"], reviewed_local_objects),
        "estimated_crop_success_rate": _ratio(crop_status_counts["primary_crop_good"], reviewed_local_objects),
        "automated_local_object_count": total_local_objects,
        "overcount": max(0, total_local_objects - len(manual_objects)),
        "undercount": max(0, len(manual_objects) - total_local_objects),
        "fragmented_local_object_count": object_status_counts["fragmented_object"],
        "false_detection_count": object_status_counts["false_detection"],
        "incorrect_merge_count": merge_counts["merge_incorrect"],
        "missed_merge_count": possible_merge_counts["accept_merge"],
        "precision_like_object_validity_ratio": _ratio(valid_objects, reviewed_local_objects),
        "fragmentation_ratio": _ratio(object_status_counts["fragmented_object"], reviewed_local_objects),
        "false_positive_ratio": _ratio(object_status_counts["false_detection"], reviewed_local_objects),
        "class_accuracy": _ratio(class_correct, reviewed_local_objects),
        "primary_crop_usability_rate": _ratio(crop_status_counts["primary_crop_good"], reviewed_local_objects),
        "accepted_merge_count": accepted_merge_count,
        "possible_merge_count": possible_merge_count,
    }
    return summary
