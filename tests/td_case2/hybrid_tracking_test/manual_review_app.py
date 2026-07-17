from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import streamlit as st

try:
    from tests.td_case2.hybrid_tracking_test.manual_review_data import ManualReviewRepository
    from tests.td_case2.hybrid_tracking_test.manual_review_schema import (
        CLASS_REVIEW_STATUSES,
        CROP_REVIEW_STATUSES,
        DOWNSTREAM_DECISIONS,
        FALSE_DETECTION_REASONS,
        MERGE_INCORRECT_REASONS,
        MERGE_REVIEW_DECISIONS,
        OBJECT_REVIEW_STATUSES,
        POSSIBLE_MERGE_REVIEW_DECISIONS,
        TIMELINE_REVIEW_STATUSES,
    )
except ImportError:  # pragma: no cover
    from manual_review_data import ManualReviewRepository
    from manual_review_schema import (
        CLASS_REVIEW_STATUSES,
        CROP_REVIEW_STATUSES,
        DOWNSTREAM_DECISIONS,
        FALSE_DETECTION_REASONS,
        MERGE_INCORRECT_REASONS,
        MERGE_REVIEW_DECISIONS,
        OBJECT_REVIEW_STATUSES,
        POSSIBLE_MERGE_REVIEW_DECISIONS,
        TIMELINE_REVIEW_STATUSES,
    )


DEFAULT_POST_TRACKING_DIR = Path(
    r"C:\Users\PC\mk\Final_vedio_Ai_system\debug_runs\hybrid_test_run_v2\hybrid_tracking_test\post_tracking_v2"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-dir", default=str(DEFAULT_POST_TRACKING_DIR.parent.parent))
    parser.add_argument("--post-tracking-dir", default=str(DEFAULT_POST_TRACKING_DIR))
    parser.add_argument("--video-path", default="")
    parser.add_argument("--camera-id", default="")
    return parser.parse_known_args()[0]


@st.cache_resource(show_spinner=False)
def _load_repository(
    run_dir: str,
    post_tracking_dir: str,
    video_path: str,
    camera_id: str,
) -> ManualReviewRepository:
    return ManualReviewRepository(
        run_dir=run_dir,
        post_tracking_dir=post_tracking_dir,
        video_path=video_path or None,
        camera_id=camera_id or None,
    )


def _repo() -> ManualReviewRepository:
    config = st.session_state["review_config"]
    return _load_repository(
        config["run_dir"],
        config["post_tracking_dir"],
        config["video_path"],
        config["camera_id"],
    )


def _seed_state(key: str, value: Any) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


def _object_field_key(local_object_id: int, field_name: str) -> str:
    return f"object_{int(local_object_id)}_{field_name}"


def _save_object_review(local_object_id: int) -> None:
    repo = _repo()
    package = repo.get_object_record(local_object_id)
    payload = {
        "local_object_id": int(local_object_id),
        "camera_id": str(package["camera_id"]),
        "manual_real_object_id": st.session_state.get(_object_field_key(local_object_id, "manual_real_object_id"), ""),
        "object_review_status": st.session_state.get(_object_field_key(local_object_id, "object_review_status"), "uncertain"),
        "crop_review_status": st.session_state.get(_object_field_key(local_object_id, "crop_review_status"), "crop_uncertain"),
        "timeline_review_status": st.session_state.get(_object_field_key(local_object_id, "timeline_review_status"), "timeline_uncertain"),
        "class_review_status": st.session_state.get(_object_field_key(local_object_id, "class_review_status"), "class_uncertain"),
        "downstream_decision": st.session_state.get(_object_field_key(local_object_id, "downstream_decision"), "manual_review"),
        "manual_class": st.session_state.get(_object_field_key(local_object_id, "manual_class"), ""),
        "same_real_object_as_local_object_ids": st.session_state.get(_object_field_key(local_object_id, "same_real_object_as"), ""),
        "suggested_real_object_group": st.session_state.get(_object_field_key(local_object_id, "suggested_group"), ""),
        "false_detection_reason": st.session_state.get(_object_field_key(local_object_id, "false_detection_reason"), ""),
        "switch_timestamp_seconds": st.session_state.get(_object_field_key(local_object_id, "switch_timestamp_seconds"), None),
        "switch_original_object": st.session_state.get(_object_field_key(local_object_id, "switch_original_object"), ""),
        "switch_new_object": st.session_state.get(_object_field_key(local_object_id, "switch_new_object"), ""),
        "track_should_be_split": st.session_state.get(_object_field_key(local_object_id, "track_should_be_split"), None),
        "reviewer_notes": st.session_state.get(_object_field_key(local_object_id, "reviewer_notes"), ""),
    }
    repo.save_object_review(payload)


def _save_merge_review(local_object_id: int) -> None:
    repo = _repo()
    event = repo.accepted_merges_by_local_object_id[int(local_object_id)]
    payload = {
        "local_object_id": int(local_object_id),
        "source_track_ids": event.get("source_track_ids", []),
        "decision": st.session_state.get(f"merge_{local_object_id}_decision", "uncertain"),
        "incorrect_reason": st.session_state.get(f"merge_{local_object_id}_incorrect_reason", ""),
        "reviewer_notes": st.session_state.get(f"merge_{local_object_id}_notes", ""),
    }
    repo.save_merge_review(payload)


def _save_possible_merge_review(candidate_key: str, from_track_id: int, to_track_id: int) -> None:
    repo = _repo()
    payload = {
        "candidate_key": candidate_key,
        "from_track_id": int(from_track_id),
        "to_track_id": int(to_track_id),
        "decision": st.session_state.get(f"possible_merge_{candidate_key}_decision", "uncertain"),
        "reviewer_notes": st.session_state.get(f"possible_merge_{candidate_key}_notes", ""),
    }
    repo.save_possible_merge_review(payload)


def _render_image_list(label: str, paths: list[str]) -> None:
    st.markdown(f"**{label}**")
    if not paths:
        st.caption("Not available.")
        return
    existing_paths = [path for path in paths if Path(path).exists()]
    if not existing_paths:
        st.caption("Referenced files are missing on disk.")
        return
    st.image(existing_paths, use_container_width=True)


def _render_object_page() -> None:
    repo = _repo()
    families = ["all", *sorted({str(item.get("object_family", "")) for item in repo.packages})]
    classes = ["all", *sorted({str(item.get("final_class", "")) for item in repo.packages})]
    downstream_values = ["all", *sorted({str(item.get("downstream_status", "")) for item in repo.packages})]
    warning_values = ["all", *sorted({warning for item in repo.packages for warning in item.get("warnings", [])})]
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        object_family = st.selectbox("Filter by family", families)
        final_class = st.selectbox("Filter by class", classes)
    with filter_col2:
        downstream_status = st.selectbox("Filter by downstream status", downstream_values)
        warning = st.selectbox("Filter by warning", warning_values)
    with filter_col3:
        only_unreviewed = st.checkbox("Show only unreviewed")
        only_merged = st.checkbox("Show only merged objects")
        only_possible_merges = st.checkbox("Show only possible merges")
    filtered_ids = repo.filter_object_ids(
        object_family=object_family,
        final_class=final_class,
        downstream_status=downstream_status,
        warning=warning,
        only_unreviewed=only_unreviewed,
        only_merged=only_merged,
        only_possible_merges=only_possible_merges,
    )
    if not filtered_ids:
        st.warning("No objects match the current filters.")
        return
    progress = repo.get_progress()
    default_object_id = progress.get("last_reviewed_object_id") or filtered_ids[0]
    _seed_state("current_object_id", default_object_id if default_object_id in filtered_ids else filtered_ids[0])
    current_object_id = st.session_state["current_object_id"]
    if current_object_id not in filtered_ids:
        current_object_id = filtered_ids[0]
        st.session_state["current_object_id"] = current_object_id
    current_index = filtered_ids.index(current_object_id)
    nav1, nav2, nav3 = st.columns([1, 1, 2])
    with nav1:
        if st.button("Previous object", disabled=current_index == 0):
            st.session_state["current_object_id"] = filtered_ids[current_index - 1]
            st.rerun()
    with nav2:
        if st.button("Next object", disabled=current_index >= len(filtered_ids) - 1):
            st.session_state["current_object_id"] = filtered_ids[current_index + 1]
            st.rerun()
    with nav3:
        selected_object_id = st.selectbox(
            "Jump to object ID",
            filtered_ids,
            index=current_index,
            format_func=lambda item: f"Object {item}",
        )
        if selected_object_id != current_object_id:
            st.session_state["current_object_id"] = selected_object_id
            st.rerun()
    record = repo.get_object_record(st.session_state["current_object_id"])
    review_payload = repo.default_object_review_payload(int(record["local_object_id"]))
    st.subheader(f"Local object {record['local_object_id']}")
    st.caption(
        f"Camera {record['camera_id']} | {record['object_family']} | {record['final_class']} | "
        f"{current_index + 1} of {len(filtered_ids)} filtered objects"
    )
    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.write(
            {
                "source_raw_track_ids": record.get("source_raw_track_ids", []),
                "start_timestamp": record.get("start_timestamp_seconds"),
                "end_timestamp": record.get("end_timestamp_seconds"),
                "duration_seconds": record.get("duration_seconds"),
                "source_frame_indexes": [
                    record.get("first_source_frame_index"),
                    record.get("last_source_frame_index"),
                ],
            }
        )
    with info_col2:
        st.write(
            {
                "quality_level": record.get("quality_level"),
                "quality_score": record.get("quality_score"),
                "downstream_status": record.get("downstream_status"),
                "entry_boundary": record.get("entry_boundary"),
                "exit_boundary": record.get("exit_boundary"),
                "motion_direction": record.get("motion_direction"),
            }
        )
    with info_col3:
        accepted_merge_event = record.get("accepted_merge_event")
        st.write(
            {
                "manual_review_reasons": record.get("manual_review_reasons", []),
                "track_integrity_status": record.get("track_integrity_status"),
                "accepted_merge_present": bool(accepted_merge_event),
            }
        )
    grouped_warnings = repo.classify_warnings(list(record.get("warnings", [])))
    warning_cols = st.columns(3)
    for column, level in zip(warning_cols, ["critical", "important", "informational"]):
        with column:
            st.markdown(f"**{level.title()} warnings**")
            warnings = grouped_warnings.get(level, [])
            if warnings:
                for item in warnings:
                    st.write(f"- {item}")
            else:
                st.caption("None")
    other_warnings = grouped_warnings.get("other", [])
    if other_warnings:
        st.markdown("**Other warnings**")
        for item in other_warnings:
            st.write(f"- {item}")
    frame_detail = record.get("representative_frames_detail", {})
    primary = frame_detail.get("primary", {}) if isinstance(frame_detail, dict) else {}
    alternatives = list(frame_detail.get("alternatives", [])) if isinstance(frame_detail, dict) else []
    plate_candidate = frame_detail.get("plate_candidate", {}) if isinstance(frame_detail, dict) else {}
    _render_image_list("Primary crop", [primary.get("crop_path")] if primary.get("crop_path") else [])
    _render_image_list(
        "Alternative crops",
        [item.get("crop_path") for item in alternatives if item.get("crop_path")],
    )
    _render_image_list(
        "Plate candidate crop",
        [plate_candidate.get("crop_path")] if plate_candidate.get("crop_path") else [],
    )
    _render_image_list(
        "Full-scene frames",
        [primary.get("full_frame_path"), *[item.get("full_frame_path") for item in alternatives]],
    )
    if record.get("accepted_merge_event"):
        st.markdown("**Accepted merge evidence**")
        st.json(record["accepted_merge_event"])
    clip = repo.ensure_review_clip(int(record["local_object_id"]))
    st.markdown("**Video review**")
    st.write(
        {
            "clip_start_seconds": clip.get("clip_start"),
            "clip_end_seconds": clip.get("clip_end"),
            "video_path": clip.get("video_path"),
        }
    )
    if clip.get("clip_path") and Path(str(clip["clip_path"])).exists():
        st.video(str(clip["clip_path"]))
    else:
        st.caption(clip.get("message", "Clip could not be generated."))

    for field_name, default_value in review_payload.items():
        if field_name in {"local_object_id", "camera_id"}:
            continue
        _seed_state(_object_field_key(int(record["local_object_id"]), field_name), default_value)

    local_object_id = int(record["local_object_id"])
    st.selectbox(
        "Object review status",
        OBJECT_REVIEW_STATUSES,
        key=_object_field_key(local_object_id, "object_review_status"),
        on_change=_save_object_review,
        args=(local_object_id,),
    )
    st.selectbox(
        "Crop review status",
        CROP_REVIEW_STATUSES,
        key=_object_field_key(local_object_id, "crop_review_status"),
        on_change=_save_object_review,
        args=(local_object_id,),
    )
    st.selectbox(
        "Timeline review status",
        TIMELINE_REVIEW_STATUSES,
        key=_object_field_key(local_object_id, "timeline_review_status"),
        on_change=_save_object_review,
        args=(local_object_id,),
    )
    st.selectbox(
        "Class review status",
        CLASS_REVIEW_STATUSES,
        key=_object_field_key(local_object_id, "class_review_status"),
        on_change=_save_object_review,
        args=(local_object_id,),
    )
    st.selectbox(
        "Downstream decision",
        DOWNSTREAM_DECISIONS,
        key=_object_field_key(local_object_id, "downstream_decision"),
        on_change=_save_object_review,
        args=(local_object_id,),
    )
    st.text_input(
        "Manual real-object ID",
        key=_object_field_key(local_object_id, "manual_real_object_id"),
        on_change=_save_object_review,
        args=(local_object_id,),
    )
    st.text_input(
        "Manual class",
        key=_object_field_key(local_object_id, "manual_class"),
        on_change=_save_object_review,
        args=(local_object_id,),
    )
    object_status = st.session_state.get(_object_field_key(local_object_id, "object_review_status"))
    if object_status in {"fragmented_object", "duplicate_track"}:
        st.text_input(
            "Same real object as local object IDs",
            help="Comma-separated IDs, for example: 12,13,19",
            key=_object_field_key(local_object_id, "same_real_object_as"),
            on_change=_save_object_review,
            args=(local_object_id,),
        )
        st.text_input(
            "Suggested real-object group",
            key=_object_field_key(local_object_id, "suggested_group"),
            on_change=_save_object_review,
            args=(local_object_id,),
        )
    if object_status == "false_detection":
        st.selectbox(
            "False detection reason",
            FALSE_DETECTION_REASONS,
            key=_object_field_key(local_object_id, "false_detection_reason"),
            on_change=_save_object_review,
            args=(local_object_id,),
        )
    if object_status == "track_switch":
        st.number_input(
            "Approximate switch timestamp",
            min_value=0.0,
            value=float(st.session_state.get(_object_field_key(local_object_id, "switch_timestamp_seconds")) or 0.0),
            key=_object_field_key(local_object_id, "switch_timestamp_seconds"),
            on_change=_save_object_review,
            args=(local_object_id,),
        )
        st.text_input(
            "Original object before switch",
            key=_object_field_key(local_object_id, "switch_original_object"),
            on_change=_save_object_review,
            args=(local_object_id,),
        )
        st.text_input(
            "Object after switch",
            key=_object_field_key(local_object_id, "switch_new_object"),
            on_change=_save_object_review,
            args=(local_object_id,),
        )
        st.checkbox(
            "Track should be split",
            key=_object_field_key(local_object_id, "track_should_be_split"),
            on_change=_save_object_review,
            args=(local_object_id,),
        )
    st.text_area(
        "Reviewer notes",
        key=_object_field_key(local_object_id, "reviewer_notes"),
        on_change=_save_object_review,
        args=(local_object_id,),
    )
    if st.button("Save current object review"):
        _save_object_review(local_object_id)
        st.success("Review saved.")


def _render_merge_page() -> None:
    repo = _repo()
    rows = repo.get_accepted_merge_rows()
    if not rows:
        st.info("No accepted merges were found.")
        return
    selected = st.selectbox("Accepted merge", rows, format_func=lambda item: f"Local object {item['local_object_id']}")
    local_object_id = int(selected["local_object_id"])
    review = selected.get("review")
    _seed_state(f"merge_{local_object_id}_decision", review.decision if review else "uncertain")
    _seed_state(f"merge_{local_object_id}_incorrect_reason", review.incorrect_reason if review else "")
    _seed_state(f"merge_{local_object_id}_notes", review.reviewer_notes if review else "")
    st.json(
        {
            "source_track_ids": selected.get("source_track_ids", []),
            "local_object_id": local_object_id,
            "merge_evidence": selected.get("merge_evidence", []),
        }
    )
    package = selected.get("package", {})
    detail = repo.get_object_record(local_object_id).get("representative_frames_detail", {})
    primary = detail.get("primary", {}) if isinstance(detail, dict) else {}
    st.image([path for path in [primary.get("crop_path"), primary.get("full_frame_path")] if path], use_container_width=True)
    clip = repo.ensure_review_clip(local_object_id)
    if clip.get("clip_path") and Path(str(clip["clip_path"])).exists():
        st.video(str(clip["clip_path"]))
    st.selectbox(
        "Merge decision",
        MERGE_REVIEW_DECISIONS,
        key=f"merge_{local_object_id}_decision",
        on_change=_save_merge_review,
        args=(local_object_id,),
    )
    if st.session_state.get(f"merge_{local_object_id}_decision") == "merge_incorrect":
        st.selectbox(
            "Incorrect reason",
            MERGE_INCORRECT_REASONS,
            key=f"merge_{local_object_id}_incorrect_reason",
            on_change=_save_merge_review,
            args=(local_object_id,),
        )
    st.text_area(
        "Reviewer notes",
        key=f"merge_{local_object_id}_notes",
        on_change=_save_merge_review,
        args=(local_object_id,),
    )
    if st.button("Save merge review"):
        _save_merge_review(local_object_id)
        st.success("Merge review saved.")


def _render_possible_merge_page() -> None:
    repo = _repo()
    candidates = repo.get_possible_merge_candidates()
    if not candidates:
        st.info("No possible merges are pending review.")
        return
    selected = st.selectbox(
        "Possible merge candidate",
        candidates,
        format_func=lambda item: f"{item['from_track_id']} -> {item['to_track_id']}",
    )
    candidate_key = str(selected["candidate_key"])
    review = selected.get("review")
    _seed_state(f"possible_merge_{candidate_key}_decision", review.decision if review else "uncertain")
    _seed_state(f"possible_merge_{candidate_key}_notes", review.reviewer_notes if review else "")
    st.json(selected)
    crop_paths = [
        selected.get("from_track", {}).get("crop_path"),
        selected.get("to_track", {}).get("crop_path"),
        selected.get("from_track", {}).get("full_frame_path"),
        selected.get("to_track", {}).get("full_frame_path"),
    ]
    existing = [path for path in crop_paths if path and Path(path).exists()]
    if existing:
        st.image(existing, use_container_width=True)
    local_ids = list(selected.get("related_local_object_ids", []))
    if local_ids:
        local_object_id = int(local_ids[0])
        clip = repo.ensure_review_clip(local_object_id)
        if clip.get("clip_path") and Path(str(clip["clip_path"])).exists():
            st.video(str(clip["clip_path"]))
    st.selectbox(
        "Possible merge decision",
        POSSIBLE_MERGE_REVIEW_DECISIONS,
        key=f"possible_merge_{candidate_key}_decision",
        on_change=_save_possible_merge_review,
        args=(candidate_key, int(selected["from_track_id"]), int(selected["to_track_id"])),
    )
    st.text_area(
        "Reviewer notes",
        key=f"possible_merge_{candidate_key}_notes",
        on_change=_save_possible_merge_review,
        args=(candidate_key, int(selected["from_track_id"]), int(selected["to_track_id"])),
    )
    if st.button("Save possible merge review"):
        _save_possible_merge_review(candidate_key, int(selected["from_track_id"]), int(selected["to_track_id"]))
        st.success("Possible merge review saved.")


def _render_summary_page() -> None:
    repo = _repo()
    st.subheader("Manual review outputs")
    st.json(repo.get_progress())
    st.json(repo.get_summary())
    st.write(f"Review output path: `{repo.paths.root}`")
    st.write(f"Input files found: {len(repo.available_input_files())}")
    st.write(repo.available_input_files())


def main() -> None:
    args = _parse_args()
    st.set_page_config(page_title="td_case2 Manual Review", layout="wide")
    if "review_config" not in st.session_state:
        st.session_state["review_config"] = {
            "run_dir": args.run_dir,
            "post_tracking_dir": args.post_tracking_dir,
            "video_path": args.video_path,
            "camera_id": args.camera_id,
        }
    st.title("td_case2 Hybrid Tracking Manual Review")
    with st.sidebar:
        st.header("Configuration")
        run_dir = st.text_input("Run dir", value=st.session_state["review_config"]["run_dir"])
        post_tracking_dir = st.text_input("Post-tracking dir", value=st.session_state["review_config"]["post_tracking_dir"])
        video_path = st.text_input("Video path", value=st.session_state["review_config"]["video_path"])
        camera_id = st.text_input("Camera ID", value=st.session_state["review_config"]["camera_id"])
        if st.button("Reload review data"):
            st.session_state["review_config"] = {
                "run_dir": run_dir,
                "post_tracking_dir": post_tracking_dir,
                "video_path": video_path,
                "camera_id": camera_id,
            }
            _load_repository.clear()
            st.rerun()
        page = st.radio("Page", ["Object Review", "Accepted Merges", "Possible Merges", "Summary"])
    repo = _repo()
    progress = repo.get_progress()
    reviewed = int(progress.get("reviewed_objects", 0))
    total = max(1, int(progress.get("total_objects", len(repo.packages))))
    st.progress(min(1.0, reviewed / total), text=f"Reviewed {reviewed} of {total} objects")
    if page == "Object Review":
        _render_object_page()
    elif page == "Accepted Merges":
        _render_merge_page()
    elif page == "Possible Merges":
        _render_possible_merge_page()
    else:
        _render_summary_page()


if __name__ == "__main__":
    main()
