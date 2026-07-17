from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .track_merge_scoring import MergeScoringConfig, compute_merge_candidate
from .track_quality import build_track_quality_markdown, build_track_quality_report, object_family_for_class


def load_hybrid_inputs(hybrid_output_dir: Path) -> dict[str, Any]:
    required_files = {
        "tracks": hybrid_output_dir / "04c_hybrid_tracks.json",
        "track_summary": hybrid_output_dir / "04c_hybrid_track_summary.json",
        "frame_metrics": hybrid_output_dir / "04c_hybrid_frame_metrics.json",
        "events": hybrid_output_dir / "04c_hybrid_tracking_events.json",
        "report": hybrid_output_dir / "04c_hybrid_tracking_report.json",
        "config": hybrid_output_dir / "04c_hybrid_config.json",
    }
    payloads: dict[str, Any] = {}
    for key, path in required_files.items():
        payloads[key] = json.loads(path.read_text(encoding="utf-8"))
    payloads["required_files"] = {key: str(path) for key, path in required_files.items()}
    return payloads


def _sorted_raw_tracks(raw_tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        raw_tracks,
        key=lambda item: (
            float(item.get("sanitized_start_timestamp_seconds", item.get("start_timestamp_seconds", 0.0)) or 0.0),
            float(item.get("sanitized_end_timestamp_seconds", item.get("end_timestamp_seconds", 0.0)) or 0.0),
            int(item.get("track_id", 0) or 0),
        ),
    )


def _local_object_stub(track: dict[str, Any], quality_item: dict[str, Any], *, camera_id: str, camera_group: str, camera_timezone: str, local_object_id: int) -> dict[str, Any]:
    active_trajectory = list(track.get("sanitized_valid_timeline", track.get("trajectory", [])))
    return {
        "camera_id": camera_id,
        "camera_group": camera_group,
        "camera_timezone": camera_timezone,
        "local_object_id": int(local_object_id),
        "local_object_key": f"{camera_id}:{int(local_object_id)}",
        "global_object_id": None,
        "object_family": str(quality_item.get("object_family") or object_family_for_class(str(track.get("class_name", "")))),
        "final_class": str(track.get("class_name", "unknown")),
        "source_raw_track_ids": [int(track.get("track_id", 0))],
        "start_timestamp_seconds": round(float(track.get("sanitized_start_timestamp_seconds", track.get("start_timestamp_seconds", 0.0)) or 0.0), 6),
        "end_timestamp_seconds": round(float(track.get("sanitized_end_timestamp_seconds", track.get("end_timestamp_seconds", 0.0)) or 0.0), 6),
        "duration_seconds": round(float(track.get("sanitized_duration_seconds", track.get("duration_seconds", 0.0)) or 0.0), 6),
        "first_source_frame_index": int(track.get("sanitized_first_source_frame_index", track.get("first_source_frame_index", 0)) or 0),
        "last_source_frame_index": int(track.get("sanitized_last_source_frame_index", track.get("last_source_frame_index", 0)) or 0),
        "combined_trajectory": active_trajectory,
        "class_votes": dict(track.get("class_votes", {})),
        "quality_level": str(quality_item.get("quality_level", "low")),
        "quality_score": round(float(quality_item.get("quality_score", 0.0) or 0.0), 6),
        "downstream_status": str(quality_item.get("downstream_status", "manual_review")),
        "merge_confidence": 0.0,
        "merge_evidence": [],
        "entry_boundary": quality_item.get("entry_boundary"),
        "exit_boundary": quality_item.get("exit_boundary"),
        "termination_reason": track.get("termination_reason"),
        "confirmed": bool(track.get("is_confirmed", False)),
        "warnings": list(quality_item.get("quality_flags", [])),
        "quality_flags": list(quality_item.get("quality_flags", [])),
        "reactivation_count": int(track.get("reactivation_count", 0) or 0),
        "track_integrity_status": str(track.get("track_integrity_status", quality_item.get("integrity_status", "usable"))),
        "timeline_correction_applied": bool(track.get("timeline_correction_applied", False)),
        "invalid_observation_count": int(track.get("invalid_observation_count", 0) or 0),
        "trimmed_kcf_duration_seconds": round(float(track.get("trimmed_kcf_duration_seconds", 0.0) or 0.0), 6),
        "frozen_kcf_detected": bool(track.get("frozen_kcf_detected", False)),
        "boundary_stuck_detected": bool(track.get("boundary_stuck_detected", False)),
    }


def _merge_local_object(current: dict[str, Any], track: dict[str, Any], quality_item: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    merged = dict(current)
    merged["source_raw_track_ids"] = sorted(set([*list(current["source_raw_track_ids"]), int(track.get("track_id", 0))]))
    merged["end_timestamp_seconds"] = round(float(track.get("sanitized_end_timestamp_seconds", track.get("end_timestamp_seconds", merged["end_timestamp_seconds"])) or merged["end_timestamp_seconds"]), 6)
    merged["last_source_frame_index"] = int(track.get("sanitized_last_source_frame_index", track.get("last_source_frame_index", merged["last_source_frame_index"])) or merged["last_source_frame_index"])
    merged["duration_seconds"] = round(float(merged["end_timestamp_seconds"]) - float(merged["start_timestamp_seconds"]), 6)
    merged["combined_trajectory"] = sorted(
        [*list(current["combined_trajectory"]), *list(track.get("sanitized_valid_timeline", track.get("trajectory", [])))],
        key=lambda item: (float(item.get("timestamp_seconds", 0.0) or 0.0), int(item.get("source_frame_index", 0) or 0)),
    )
    class_votes = dict(current.get("class_votes", {}))
    for key, value in dict(track.get("class_votes", {})).items():
        class_votes[str(key)] = float(class_votes.get(str(key), 0.0)) + float(value)
    merged["class_votes"] = {str(key): round(float(value), 6) for key, value in class_votes.items()}
    if class_votes:
        merged["final_class"] = max(sorted(class_votes), key=lambda item: (float(class_votes[item]), item))
    merged["quality_score"] = round(max(float(current.get("quality_score", 0.0)), float(quality_item.get("quality_score", 0.0))), 6)
    quality_order = {"high": 3, "medium": 2, "low": 1, "invalid": 0}
    merged["quality_level"] = max(
        [str(current.get("quality_level", "low")), str(quality_item.get("quality_level", "low"))],
        key=lambda item: quality_order.get(item, -1),
    )
    merged["confirmed"] = bool(current.get("confirmed", False) or track.get("is_confirmed", False))
    merged["merge_confidence"] = round(max(float(current.get("merge_confidence", 0.0)), float(candidate.get("merge_score", 0.0))), 6)
    merged["merge_evidence"] = [
        *list(current.get("merge_evidence", [])),
        {
            "from_track_id": int(current["source_raw_track_ids"][-1]),
            "to_track_id": int(track.get("track_id", 0)),
            **{key: value for key, value in candidate.items() if key not in {"compatible", "decision", "reasons"}},
        },
    ]
    merged["warnings"] = sorted(set([*list(current.get("warnings", [])), *list(quality_item.get("quality_flags", []))]))
    merged["quality_flags"] = sorted(set([*list(current.get("quality_flags", [])), *list(quality_item.get("quality_flags", []))]))
    merged["reactivation_count"] = int(current.get("reactivation_count", 0)) + int(track.get("reactivation_count", 0) or 0)
    merged["invalid_observation_count"] = int(current.get("invalid_observation_count", 0)) + int(track.get("invalid_observation_count", 0) or 0)
    merged["trimmed_kcf_duration_seconds"] = round(float(current.get("trimmed_kcf_duration_seconds", 0.0) or 0.0) + float(track.get("trimmed_kcf_duration_seconds", 0.0) or 0.0), 6)
    merged["frozen_kcf_detected"] = bool(current.get("frozen_kcf_detected", False) or track.get("frozen_kcf_detected", False))
    merged["boundary_stuck_detected"] = bool(current.get("boundary_stuck_detected", False) or track.get("boundary_stuck_detected", False))
    if current.get("exit_boundary") is None:
        merged["exit_boundary"] = quality_item.get("exit_boundary")
    return merged


def reconcile_track_fragments(
    raw_tracks: list[dict[str, Any]],
    quality_report: dict[str, Any],
    *,
    camera_id: str,
    camera_group: str,
    camera_timezone: str,
    scoring_config: MergeScoringConfig,
    candidate_records: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    quality_by_track_id = {int(item["track_id"]): item for item in list(quality_report.get("tracks", []))}
    ordered_tracks = _sorted_raw_tracks(raw_tracks)
    accepted_merges: list[dict[str, Any]] = []
    rejected_candidates: list[dict[str, Any]] = []
    possible_candidates: list[dict[str, Any]] = []
    local_objects: list[dict[str, Any]] = []
    used_track_ids: set[int] = set()
    next_local_object_id = 1
    indexed_candidates: dict[tuple[int, int], dict[str, Any]] = {
        (int(item["from_track_id"]), int(item["to_track_id"])): item for item in list(candidate_records or [])
    }

    for raw_track in ordered_tracks:
        raw_track_id = int(raw_track.get("track_id", 0))
        if raw_track_id in used_track_ids:
            continue
        quality_item = quality_by_track_id[raw_track_id]
        local_object = _local_object_stub(
            raw_track,
            quality_item,
            camera_id=camera_id,
            camera_group=camera_group,
            camera_timezone=camera_timezone,
            local_object_id=next_local_object_id,
        )
        next_local_object_id += 1
        used_track_ids.add(raw_track_id)

        while True:
            best_candidate: tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None = None
            tail_track_id = int(local_object["source_raw_track_ids"][-1])
            tail_track = next(track for track in ordered_tracks if int(track.get("track_id", 0)) == tail_track_id)
            for candidate_track in ordered_tracks:
                candidate_track_id = int(candidate_track.get("track_id", 0))
                if candidate_track_id in used_track_ids:
                    continue
                candidate_quality = quality_by_track_id[candidate_track_id]
                if candidate_records is not None:
                    indexed = indexed_candidates.get((tail_track_id, candidate_track_id))
                    if indexed is None:
                        continue
                    left_payload = {**tail_track, "_candidate_type": str(indexed.get("candidate_type", "sequential_fragment"))}
                    right_payload = {**candidate_track, "_candidate_type": str(indexed.get("candidate_type", "sequential_fragment"))}
                    candidate = compute_merge_candidate(left_payload, right_payload, config=scoring_config)
                    candidate = {**indexed, **candidate}
                else:
                    candidate = compute_merge_candidate(tail_track, candidate_track, config=scoring_config)
                candidate_record = {
                    "from_track_id": tail_track_id,
                    "to_track_id": candidate_track_id,
                    "from_local_object_id": int(local_object["local_object_id"]),
                    **candidate,
                }
                if candidate["decision"] == "auto_merge":
                    if best_candidate is None or float(candidate["merge_score"]) > float(best_candidate[2]["merge_score"]):
                        best_candidate = (candidate_track, candidate_quality, candidate)
                elif candidate["decision"] == "possible_merge":
                    possible_candidates.append(candidate_record)
                else:
                    rejected_candidates.append(candidate_record)
            if best_candidate is None:
                break
            merge_track, merge_quality, merge_candidate = best_candidate
            used_track_ids.add(int(merge_track.get("track_id", 0)))
            local_object = _merge_local_object(local_object, merge_track, merge_quality, merge_candidate)
            accepted_merges.append(
                {
                    "local_object_id": int(local_object["local_object_id"]),
                    "source_track_ids": list(local_object["source_raw_track_ids"]),
                    "merge_evidence": list(local_object["merge_evidence"]),
                }
            )
        local_objects.append(local_object)

    reconciled_objects_by_class = Counter(str(item.get("final_class", "unknown")) for item in local_objects)
    reconciled_objects_by_family = Counter(str(item.get("object_family", "other")) for item in local_objects)
    quality_breakdown = Counter(str(item.get("quality_level", "low")) for item in local_objects)
    report = {
        "status": "success",
        "raw_track_id_count": len(raw_tracks),
        "confirmed_raw_track_segment_count": len([item for item in raw_tracks if bool(item.get("is_confirmed"))]),
        "tentative_raw_track_segment_count": len([item for item in raw_tracks if not bool(item.get("is_confirmed"))]),
        "raw_tracks_by_class": dict(sorted(Counter(str(item.get("class_name", "unknown")) for item in raw_tracks).items())),
        "raw_tracks_by_family": dict(sorted(Counter(object_family_for_class(str(item.get("class_name", ""))) for item in raw_tracks).items())),
        "reconciled_local_object_count": len(local_objects),
        "reconciled_confirmed_object_count": len([item for item in local_objects if bool(item.get("confirmed"))]),
        "reconciled_objects_by_class": dict(sorted(reconciled_objects_by_class.items())),
        "reconciled_objects_by_family": dict(sorted(reconciled_objects_by_family.items())),
        "accepted_merge_count": len(accepted_merges),
        "rejected_merge_candidate_count": len(rejected_candidates),
        "possible_merge_count": len(possible_candidates),
        "reconciliation_candidate_count": len(candidate_records) if candidate_records is not None else len(rejected_candidates) + len(possible_candidates) + len(accepted_merges),
        "merged_fragment_group_count": len([item for item in local_objects if len(list(item.get("source_raw_track_ids", []))) > 1]),
        "fragment_reduction_percent": round((1.0 - (len(local_objects) / max(len(raw_tracks), 1))) * 100.0, 6),
        "short_raw_tracks_under_0_5_seconds": len([item for item in raw_tracks if float(item.get("duration_seconds", 0.0) or 0.0) < 0.5]),
        "short_raw_tracks_under_1_second": len([item for item in raw_tracks if float(item.get("duration_seconds", 0.0) or 0.0) < 1.0]),
        "short_reconciled_objects_under_0_5_seconds": len([item for item in local_objects if float(item.get("duration_seconds", 0.0) or 0.0) < 0.5]),
        "short_reconciled_objects_under_1_second": len([item for item in local_objects if float(item.get("duration_seconds", 0.0) or 0.0) < 1.0]),
        "quality_breakdown": dict(sorted(quality_breakdown.items())),
        "merge_score_distribution": {
            "accepted_mean": round(sum(float(item["merge_evidence"][-1]["merge_score"]) for item in accepted_merges if item["merge_evidence"]) / max(len([item for item in accepted_merges if item["merge_evidence"]]), 1), 6),
            "possible_mean": round(sum(float(item.get("merge_score", 0.0)) for item in possible_candidates) / max(len(possible_candidates), 1), 6),
            "rejected_mean": round(sum(float(item.get("merge_score", 0.0)) for item in rejected_candidates if item.get("merge_score") is not None) / max(len([item for item in rejected_candidates if item.get("merge_score") is not None]), 1), 6),
        },
        "warnings": [],
        "manual_review_candidates": possible_candidates,
    }
    return local_objects, accepted_merges, rejected_candidates, report


def write_track_quality_outputs(post_tracking_dir: Path, quality_report: dict[str, Any]) -> None:
    (post_tracking_dir / "04d_track_quality_report.json").write_text(json.dumps(quality_report, indent=2), encoding="utf-8")
    (post_tracking_dir / "04d_track_quality_report.md").write_text(build_track_quality_markdown(quality_report), encoding="utf-8")


def write_reconciliation_outputs(
    post_tracking_dir: Path,
    *,
    reconciled_tracks: list[dict[str, Any]],
    merge_events: list[dict[str, Any]],
    rejected_candidates: list[dict[str, Any]],
    report: dict[str, Any],
) -> None:
    (post_tracking_dir / "04d_reconciled_tracks.json").write_text(json.dumps({"status": "success", "tracks": reconciled_tracks}, indent=2), encoding="utf-8")
    (post_tracking_dir / "04d_track_merge_events.json").write_text(json.dumps({"status": "success", "events": merge_events}, indent=2), encoding="utf-8")
    (post_tracking_dir / "04d_rejected_merge_candidates.json").write_text(json.dumps({"status": "success", "candidates": rejected_candidates}, indent=2), encoding="utf-8")
    (post_tracking_dir / "04d_track_reconciliation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_lines = [
        "# Track Reconciliation Report",
        "",
        f"- Raw track IDs: {report['raw_track_id_count']}",
        f"- Confirmed raw track segments: {report['confirmed_raw_track_segment_count']}",
        f"- Reconciled local objects: {report['reconciled_local_object_count']}",
        f"- Reconciled confirmed objects: {report['reconciled_confirmed_object_count']}",
        f"- Accepted fragment merges: {report['accepted_merge_count']}",
        f"- Rejected merge candidates: {report['rejected_merge_candidate_count']}",
        f"- Fragment reduction percent: {report['fragment_reduction_percent']:.3f}%",
        "",
        "Reconciled local-object counts are estimated single-camera physical-object counts based on track continuity evidence, not ground truth.",
    ]
    (post_tracking_dir / "04d_track_reconciliation_report.md").write_text("\n".join(markdown_lines), encoding="utf-8")


__all__ = [
    "MergeScoringConfig",
    "build_track_quality_report",
    "load_hybrid_inputs",
    "reconcile_tracks",
    "reconcile_track_fragments",
    "write_reconciliation_outputs",
    "write_track_quality_outputs",
]


def reconcile_tracks(raw_tracks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    quality_report = build_track_quality_report(raw_tracks, frame_width=1280, frame_height=720)
    local_objects, merge_events, _rejected, report = reconcile_track_fragments(
        raw_tracks,
        quality_report,
        camera_id="test_cam_01",
        camera_group="single_camera_test",
        camera_timezone="Asia/Kolkata",
        scoring_config=MergeScoringConfig(),
    )
    return local_objects, merge_events, {"reconciled_track_count": report["reconciled_local_object_count"], **report}
