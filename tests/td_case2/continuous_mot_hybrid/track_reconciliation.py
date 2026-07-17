from __future__ import annotations

from collections import Counter
from typing import Any

from tests.td_case2.hybrid_tracking_test.reconciliation_candidate_index import CandidateIndexConfig, generate_reconciliation_candidates
from tests.td_case2.hybrid_tracking_test.track_merge_scoring import MergeScoringConfig, compute_merge_candidate


def reconcile_track_fragments(
    tracks: list[dict[str, Any]],
    *,
    camera_id: str,
    camera_group: str,
    camera_timezone: str,
    maximum_gap_seconds: float,
    duplicate_overlap_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    id_lookup = {str(item["track_id"]): index + 1 for index, item in enumerate(tracks)}
    normalized_tracks = [{**item, "track_id": id_lookup[str(item["track_id"])]} for item in tracks]
    candidate_config = CandidateIndexConfig(
        maximum_merge_gap_seconds=maximum_gap_seconds,
        maximum_overlap_duplicate_seconds=duplicate_overlap_seconds,
        maximum_predicted_center_distance_ratio=2.0,
        minimum_area_ratio=0.33,
        maximum_area_ratio=3.0,
    )
    candidates, candidate_report = generate_reconciliation_candidates(normalized_tracks, config=candidate_config)
    scoring_config = MergeScoringConfig(
        maximum_merge_gap_seconds=maximum_gap_seconds,
        automatic_merge_score=0.80,
        possible_merge_score=0.68,
    )
    reverse_lookup = {value: key for key, value in id_lookup.items()}
    track_by_id = {str(item["track_id"]): item for item in tracks}
    accepted: list[dict[str, Any]] = []
    possible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    assigned_target_ids: set[str] = set()
    for candidate in candidates:
        candidate = {
            **candidate,
            "from_track_id": reverse_lookup[int(candidate["from_track_id"])],
            "to_track_id": reverse_lookup[int(candidate["to_track_id"])],
        }
        left = track_by_id[str(candidate["from_track_id"])]
        right = track_by_id[str(candidate["to_track_id"])]
        scored = {**candidate, **compute_merge_candidate({**left, "_candidate_type": candidate["candidate_type"]}, {**right, "_candidate_type": candidate["candidate_type"]}, config=scoring_config)}
        target_list = rejected
        if scored["decision"] == "auto_merge" and str(candidate["to_track_id"]) not in assigned_target_ids:
            target_list = accepted
            assigned_target_ids.add(str(candidate["to_track_id"]))
        elif scored["decision"] == "possible_merge":
            target_list = possible
        target_list.append(scored)
    reconciled_rows: list[dict[str, Any]] = []
    consumed: set[str] = set()
    next_local_object_id = 1
    accepted_by_source = {(str(item["from_track_id"]), str(item["to_track_id"])): item for item in accepted}
    ordered_tracks = sorted(tracks, key=lambda item: (float(item["sanitized_start_timestamp_seconds"]), str(item["track_id"])))
    for track in ordered_tracks:
        track_id = str(track["track_id"])
        if track_id in consumed:
            continue
        combined = dict(track)
        combined["source_raw_track_ids"] = [track_id]
        for candidate_key, candidate in list(accepted_by_source.items()):
            if candidate_key[0] == track_id:
                right = track_by_id[candidate_key[1]]
                combined["source_raw_track_ids"].append(str(right["track_id"]))
                combined["sanitized_valid_timeline"] = sorted(
                    [*list(combined.get("sanitized_valid_timeline", [])), *list(right.get("sanitized_valid_timeline", []))],
                    key=lambda item: (float(item["timestamp_seconds"]), int(item["source_frame_index"])),
                )
                combined["sanitized_end_timestamp_seconds"] = right["sanitized_end_timestamp_seconds"]
                combined["sanitized_duration_seconds"] = round(
                    float(combined["sanitized_end_timestamp_seconds"]) - float(combined["sanitized_start_timestamp_seconds"]),
                    6,
                )
                consumed.add(str(right["track_id"]))
        combined["camera_id"] = camera_id
        combined["camera_group"] = camera_group
        combined["camera_timezone"] = camera_timezone
        combined["local_object_id"] = next_local_object_id
        combined["local_object_key"] = f"{camera_id}:{next_local_object_id}"
        combined["final_class"] = str(track["class_name"])
        combined["quality_score"] = 1.0 if str(track["track_integrity_status"]) == "usable" else 0.6
        combined["quality_level"] = "high" if combined["quality_score"] >= 0.9 else ("medium" if combined["quality_score"] >= 0.7 else "low")
        combined["confirmed"] = bool(track.get("confirmed"))
        combined["downstream_status"] = "ready" if bool(track.get("confirmed")) and float(track.get("sanitized_duration_seconds", 0.0)) >= 0.5 else "manual_review"
        combined["class_votes"] = {str(track["class_name"]): float(track.get("detector_hits", 0))}
        combined["warnings"] = list(track.get("integrity_flags", []))
        combined["quality_flags"] = list(track.get("integrity_flags", []))
        combined["start_timestamp_seconds"] = combined["sanitized_start_timestamp_seconds"]
        combined["end_timestamp_seconds"] = combined["sanitized_end_timestamp_seconds"]
        combined["duration_seconds"] = combined["sanitized_duration_seconds"]
        combined["first_source_frame_index"] = int(track["first_source_frame_index"])
        combined["last_source_frame_index"] = int(track["last_source_frame_index"])
        reconciled_rows.append(combined)
        consumed.add(track_id)
        next_local_object_id += 1
    report = {
        "status": "success",
        "reconciled_objects": len(reconciled_rows),
        "accepted_merges": len(accepted),
        "possible_merges": len(possible),
        "rejected_merges": len(rejected),
        "candidate_report": candidate_report,
        "quality_distribution": dict(sorted(Counter(str(item["quality_level"]) for item in reconciled_rows).items())),
    }
    return (
        {"status": "success", "candidates": candidates},
        {"status": "success", "merges": accepted},
        {"status": "success", "merges": possible},
        {"status": "success", "merges": rejected},
        {"status": "success", "tracks": reconciled_rows, "report": report},
    )
