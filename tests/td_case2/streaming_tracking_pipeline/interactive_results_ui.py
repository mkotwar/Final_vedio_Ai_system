from __future__ import annotations

import argparse
import subprocess
import sys
import types
from pathlib import Path
from typing import Any


def _install_namespace_package(name: str, package_path: Path) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    module.__path__ = [str(package_path)]
    module.__package__ = name


try:
    from .ui_data_loader import (
        append_manual_plate_review,
        build_object_evidence,
        ensure_ui_state_files,
        get_record_detail,
        load_manual_plate_reviews,
        load_run_artifacts,
        summarize_evidence_availability,
        summarize_records,
    )
    from .ui_filters import UIRecordFilters, available_filter_values, filter_records, paginate_records
    from .ui_media import collect_record_media, image_status, plate_crop_caption, render_evidence_image, render_object_evidence_pair, video_preview_info
except ImportError:
    repo_root = Path(__file__).resolve().parents[3]
    tests_root = repo_root / "tests"
    td_case2_root = tests_root / "td_case2"
    package_root = td_case2_root / "streaming_tracking_pipeline"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    _install_namespace_package("tests", tests_root)
    _install_namespace_package("tests.td_case2", td_case2_root)
    _install_namespace_package("tests.td_case2.streaming_tracking_pipeline", package_root)
    from tests.td_case2.streaming_tracking_pipeline.ui_data_loader import (
        append_manual_plate_review,
        build_object_evidence,
        ensure_ui_state_files,
        get_record_detail,
        load_manual_plate_reviews,
        load_run_artifacts,
        summarize_evidence_availability,
        summarize_records,
    )
    from tests.td_case2.streaming_tracking_pipeline.ui_filters import UIRecordFilters, available_filter_values, filter_records, paginate_records
    from tests.td_case2.streaming_tracking_pipeline.ui_media import collect_record_media, image_status, plate_crop_caption, render_evidence_image, render_object_evidence_pair, video_preview_info


DEFAULT_RUN_DIR = Path("debug_runs") / "streaming_tracking_anpr_10fps_anpr_test_5min_20260718_163012"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect isolated ANPR run artifacts in Streamlit.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit("Streamlit is required for this UI. Install streamlit in the active environment.") from exc

    st.set_page_config(page_title="Object Results Inspector", layout="wide")
    st.title("Object Results Inspector")
    run_dir = Path(args.run_dir)
    repo_root = Path.cwd()
    artifacts = _cached_load(str(run_dir), str(repo_root))
    ensure_ui_state_files(run_dir)

    if artifacts.missing_artifacts:
        st.warning("Missing optional artifacts: " + ", ".join(artifacts.missing_artifacts))

    tabs = st.tabs(
        [
            "Dashboard",
            "Search",
            "All Objects",
            "Object Details",
            "Plate Review",
            "Tracking Review",
            "Object Colour Review",
            "Pipeline Metrics",
            "Run Artifacts",
            "Run New Video",
        ]
    )
    with tabs[0]:
        _dashboard_page(st, artifacts)
    with tabs[1]:
        _search_page(st, artifacts)
    with tabs[2]:
        _all_objects_page(st, artifacts)
    with tabs[3]:
        _vehicle_details_page(st, artifacts)
    with tabs[4]:
        _plate_review_page(st, artifacts)
    with tabs[5]:
        _tracking_review_page(st, artifacts)
    with tabs[6]:
        _colour_review_page(st, artifacts)
    with tabs[7]:
        _pipeline_metrics_page(st, artifacts)
    with tabs[8]:
        _run_artifacts_page(st, artifacts)
    with tabs[9]:
        _run_new_video_page(st, repo_root, run_dir)


def _dashboard_page(st: Any, artifacts: Any) -> None:
    summary = summarize_records(artifacts.searchable_records, artifacts.source_metadata)
    evidence_summary = summarize_evidence_availability(artifacts.searchable_records, artifacts)
    metric_keys = [
        ("Tracked objects", "total_tracked_objects"),
        ("Cars", "cars"),
        ("Motorcycles", "motorcycles_two_wheelers"),
        ("Buses", "buses"),
        ("Trucks", "trucks"),
        ("Bicycles", "bicycles"),
        ("Persons", "persons"),
        ("Verified plates", "verified_plates"),
        ("Weak plates", "weak_plates"),
        ("Invalid OCR", "invalid_ocr"),
        ("No plate", "no_plate_records"),
    ]
    columns = st.columns(4)
    for index, (label, key) in enumerate(metric_keys):
        columns[index % 4].metric(label, summary.get(key, 0))
    st.subheader("Dominant Colour")
    st.bar_chart(summary["records_by_dominant_colour"])
    st.subheader("Run")
    st.json(
        {
            "processing_runtime_sec": artifacts.step11_summary.get("runtime_sec")
            or artifacts.step10_summary.get("total_pipeline_runtime_sec"),
            "gpu_device": artifacts.step10_summary.get("gpu_name") or artifacts.step11_summary.get("gpu_device"),
            "peak_gpu_memory": artifacts.step10_summary.get("peak_gpu_memory") or artifacts.step11_summary.get("peak_gpu_memory"),
            "source_fps": summary.get("source_fps"),
            "processed_fps": summary.get("processed_fps"),
            "duration_sec": summary.get("duration_sec"),
            "records_with_full_frames": evidence_summary["records_with_full_frames"],
            "records_missing_full_frames": evidence_summary["records_missing_full_frames"],
            "records_with_frame_mismatch": evidence_summary["records_with_frame_mismatch"],
        }
    )


def _search_page(st: Any, artifacts: Any) -> None:
    values = available_filter_values(artifacts.searchable_records)
    query = st.text_input("Natural query", value="white car")
    col1, col2, col3, col4 = st.columns(4)
    object_class = col1.selectbox("Class", [""] + values["classes"])
    colour = col2.selectbox("Dominant colour", [""] + values["colours"])
    status = col3.selectbox("Plate status", [""] + values["plate_statuses"])
    sort_by = col4.selectbox("Sort", ["relevance", "time", "confidence", "track_id"])
    col5, col6, col7, col8 = st.columns(4)
    exact_plate = col5.text_input("Exact plate")
    plate_prefix = col6.text_input("Plate prefix")
    track_id = col7.number_input("Track ID", min_value=0, value=0)
    track_generation = col8.number_input("Generation", min_value=0, value=0)
    duration = float(artifacts.source_metadata.get("duration_sec") or 0.0)
    time_range = st.slider("Time range seconds", min_value=0.0, max_value=max(duration, 1.0), value=(0.0, max(duration, 1.0)))
    col9, col10, col11, col12 = st.columns(4)
    min_conf = col9.slider("Minimum confidence", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    verified_only = col10.toggle("Verified only", value=False)
    include_weak = col11.toggle("Include weak OCR", value=True)
    presence = col12.selectbox("Plate presence", ["any", "with_plate", "without_plate"])
    top_k = st.number_input("Top K", min_value=1, max_value=500, value=50)
    filters = UIRecordFilters(
        text_query=query,
        object_class=object_class or None,
        dominant_colour=colour or None,
        plate_status=status or None,
        exact_plate=exact_plate or None,
        plate_prefix=plate_prefix or None,
        start_time_sec=time_range[0],
        end_time_sec=time_range[1],
        track_id=int(track_id) if track_id else None,
        track_generation=int(track_generation) if track_generation else None,
        minimum_confidence=min_conf if min_conf > 0 else None,
        verified_only=verified_only,
        include_weak_plates=include_weak,
        plate_presence=presence,
        sort_by=sort_by,
        top_k=int(top_k),
    )
    results = filter_records(artifacts.searchable_records, filters)
    st.caption(f"{len(results)} records matched")
    _render_result_cards(st, artifacts, results, page_key="search")


def _all_objects_page(st: Any, artifacts: Any) -> None:
    st.caption(f"{len(artifacts.searchable_records)} records loaded")
    _render_result_cards(st, artifacts, artifacts.searchable_records, page_key="all")


def _vehicle_details_page(st: Any, artifacts: Any) -> None:
    record_id = _selected_record_id(st, artifacts)
    if not record_id:
        st.info("Select a result card first, or choose a record below.")
        record_id = st.selectbox("Record", [""] + sorted(artifacts.records_by_id))
    if not record_id:
        return
    detail = get_record_detail(artifacts, record_id)
    record = detail.get("record") or {}
    st.subheader(record_id)
    st.json(
        {
            key: record.get(key)
            for key in [
                "object_class",
                "raw_class_name",
                "dominant_colour",
                "upper_clothing_color",
                "lower_clothing_color",
                "dominant_clothing_color",
                "clothing_color_status",
                "raw_colour",
                "plate_text",
                "plate_status",
                "first_frame_index",
                "last_frame_index",
                "first_seen_sec",
                "last_seen_sec",
                "warnings",
            ]
        }
    )
    st.subheader("Scene and Object Evidence")
    render_object_evidence_pair(st, detail.get("object_evidence") or build_object_evidence(record, artifacts), run_dir=artifacts.run_dir, repo_root=artifacts.repo_root)
    _render_media_sections(st, artifacts, record)
    st.subheader("Selected Crops")
    st.json(detail.get("selected_crops") or {})
    st.subheader("Step 8 Final Result")
    st.json(detail.get("plate_validation") or {})
    st.subheader("Lifecycle")
    st.json(detail.get("completed_track") or {})
    st.subheader("Video Preview")
    preview = video_preview_info(record, artifacts.source_metadata)
    st.json(preview)
    video_status = image_status(preview.get("source_path"), run_dir=artifacts.run_dir, repo_root=artifacts.repo_root)
    if video_status["resolved_path"]:
        st.video(video_status["resolved_path"], start_time=int(float(preview.get("timestamp_sec") or 0)))


def _plate_review_page(st: Any, artifacts: Any) -> None:
    status = st.selectbox("Plate status", ["verified", "weak", "invalid", "no_plate_detected", ""])
    records = [
        record
        for record in artifacts.searchable_records
        if record.get("object_group") != "person" and (not status or record.get("plate_status") == status)
    ]
    page_records, _ = paginate_records(records, page=int(st.number_input("Page", min_value=1, value=1, key="plate_page")), page_size=10)
    reviews = load_manual_plate_reviews(artifacts.run_dir)
    st.caption(f"{len(records)} records, {len(reviews)} manual review decisions")
    for record in page_records:
        with st.expander(f"{record.get('record_id')} | {record.get('plate_text') or 'no plate'} | {record.get('plate_status')}"):
            evidence = build_object_evidence(record, artifacts)
            render_object_evidence_pair(st, evidence, run_dir=artifacts.run_dir, repo_root=artifacts.repo_root)
            cols = st.columns([1, 2])
            render_evidence_image(
                cols[0],
                evidence.get("plate_crop_path"),
                "plate_crop",
                run_dir=artifacts.run_dir,
                repo_root=artifacts.repo_root,
                caption=plate_crop_caption(evidence),
                missing_message="Plate crop unavailable",
            )
            cols[1].json(_plate_review_payload(record, artifacts))
            decision = cols[1].selectbox("Decision", ["looks_correct", "looks_incorrect", "needs_review"], key=f"decision_{record.get('record_id')}")
            notes = cols[1].text_input("Notes", key=f"notes_{record.get('record_id')}")
            if cols[1].button("Save manual review", key=f"save_{record.get('record_id')}"):
                append_manual_plate_review(artifacts.run_dir, record_id=str(record.get("record_id")), decision=decision, notes=notes)
                st.success("Manual review saved separately.")


def _tracking_review_page(st: Any, artifacts: Any) -> None:
    status = st.selectbox("Lifecycle filter", ["", "confirmed", "tentative", "short track", "temporarily lost", "multiple generations"])
    rows = []
    generation_counts: dict[int, int] = {}
    for record in artifacts.searchable_records:
        generation_counts[int(record.get("track_id") or 0)] = generation_counts.get(int(record.get("track_id") or 0), 0) + 1
    for record in artifacts.searchable_records:
        lifecycle = artifacts.completed_tracks_by_identity.get(_identity_key(record)) or {}
        warnings = list(record.get("warnings") or [])
        short_track = float(record.get("duration_sec") or 0.0) < 1.0
        multiple_generations = generation_counts.get(int(record.get("track_id") or 0), 0) > 1
        row = {
            "record_id": record.get("record_id"),
            "track_id": record.get("track_id"),
            "generation": record.get("track_generation"),
            "class": record.get("normalized_class_name") or record.get("object_class"),
            "first_seen_sec": record.get("first_seen_sec"),
            "last_seen_sec": record.get("last_seen_sec"),
            "observation_count": lifecycle.get("observation_count"),
            "lifecycle_status": lifecycle.get("status") or record.get("metadata", {}).get("lifecycle_status"),
            "completion_reason": lifecycle.get("completion_reason") or record.get("metadata", {}).get("lifecycle_completion_reason"),
            "short_track": short_track,
            "multiple_generations": multiple_generations,
            "warnings": warnings,
        }
        if _tracking_row_matches(row, status):
            rows.append(row)
    st.dataframe(rows, use_container_width=True)
    matching_records = [record for record in artifacts.searchable_records if str(record.get("record_id")) in {str(row["record_id"]) for row in rows}]
    _render_result_cards(st, artifacts, matching_records, page_key="tracking")


def _colour_review_page(st: Any, artifacts: Any) -> None:
    vehicle_records = [
        record
        for record in artifacts.searchable_records
        if record.get("object_group") != "person" and (record.get("normalized_colour") or record.get("dominant_colour") or record.get("raw_colour"))
    ]
    person_records = [
        record
        for record in artifacts.searchable_records
        if record.get("object_group") == "person"
        and (record.get("dominant_clothing_color") or record.get("upper_clothing_color") or record.get("lower_clothing_color"))
    ]
    st.subheader("Vehicle Colour")
    _render_result_cards(st, artifacts, vehicle_records, page_key="vehicle_colour", show_colour_debug=True)
    st.subheader("Person Clothing Colour")
    _render_result_cards(st, artifacts, person_records, page_key="person_clothing_colour", show_colour_debug=True)


def _pipeline_metrics_page(st: Any, artifacts: Any) -> None:
    st.subheader("Source Metadata")
    st.json({key: value for key, value in artifacts.source_metadata.items() if key != "selected_frame_indices"})
    st.subheader("Search Metrics")
    st.json(artifacts.step10_summary)
    st.subheader("Result Card Metrics")
    st.json(artifacts.step11_summary)


def _run_artifacts_page(st: Any, artifacts: Any) -> None:
    st.subheader("Loaded")
    st.json(artifacts.loaded_artifacts)
    st.subheader("Missing")
    st.json(artifacts.missing_artifacts)
    st.subheader("UI State")
    st.json({key: str(path) for key, path in ensure_ui_state_files(artifacts.run_dir).items()})
    st.subheader("Evidence Diagnostics")
    st.json(summarize_evidence_availability(artifacts.searchable_records, artifacts))


def _run_new_video_page(st: Any, repo_root: Path, current_run_dir: Path) -> None:
    st.warning("Disabled by default. This starts the existing validation runner only after pressing the button.")
    enabled = st.checkbox("Enable new-video runner")
    video_path = st.text_input("Video path")
    target_fps = st.number_input("Target FPS", min_value=1, max_value=60, value=10)
    full_video = st.checkbox("Full video", value=True)
    device = st.selectbox("Device mode", ["auto", "cpu", "cuda"])
    output_dir = st.text_input("Output directory", value=str(current_run_dir.parent))
    if "ui_subprocess_running" not in st.session_state:
        st.session_state.ui_subprocess_running = False
    if st.button("Start processing", disabled=not enabled or st.session_state.ui_subprocess_running or not video_path):
        st.session_state.ui_subprocess_running = True
        command = [
            sys.executable,
            "-m",
            "tests.td_case2.streaming_tracking_pipeline.run_combined_vehicle_person_pipeline",
            "--video",
            video_path,
            "--target-fps",
            str(target_fps),
            "--device",
            device,
            "--output-root",
            output_dir,
        ]
        if full_video:
            command.append("--full-video")
        with subprocess.Popen(command, cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
            log_box = st.empty()
            lines: list[str] = []
            for line in process.stdout or []:
                lines.append(line.rstrip())
                log_box.code("\n".join(lines[-200:]))
            st.session_state.ui_subprocess_running = False
            st.caption(f"Process exited with {process.returncode}")


def _cached_load(run_dir: str, repo_root: str) -> Any:
    import streamlit as st

    @st.cache_data(show_spinner="Loading ANPR artifacts")
    def _load(run_dir_text: str, repo_root_text: str) -> Any:
        return load_run_artifacts(run_dir_text, repo_root=repo_root_text)

    return _load(run_dir, repo_root)


def _render_result_cards(st: Any, artifacts: Any, records: list[dict[str, Any]], *, page_key: str, show_colour_debug: bool = False) -> None:
    page = st.number_input("Page", min_value=1, value=1, key=f"{page_key}_page")
    page_size = st.selectbox("Cards per page", [10, 20, 50], index=1, key=f"{page_key}_page_size")
    page_records, meta = paginate_records(records, page=int(page), page_size=int(page_size))
    st.caption(f"Page {meta['page']} of {meta['total_pages']} | {meta['total_records']} records")
    for record in page_records:
        title = f"#{record.get('_ui_rank') or '-'} {record.get('record_id')} | {record.get('normalized_class_name') or record.get('object_class')} | {record.get('dominant_colour') or record.get('normalized_colour') or 'unknown'}"
        with st.container(border=True):
            evidence = build_object_evidence(record, artifacts)
            cols = st.columns([3, 1, 2])
            render_object_evidence_pair(cols[0], evidence, run_dir=artifacts.run_dir, repo_root=artifacts.repo_root, compact=True)
            if (record.get("normalized_class_name") or record.get("object_class")) == "person":
                cols[1].caption("Plate fields not meaningful for persons.")
            else:
                render_evidence_image(
                    cols[1],
                    evidence.get("plate_crop_path"),
                    "plate_crop",
                    run_dir=artifacts.run_dir,
                    repo_root=artifacts.repo_root,
                    caption=plate_crop_caption(evidence),
                    missing_message="Plate crop unavailable",
                )
            cols[2].markdown(f"**{title}**")
            cols[2].write(
                {
                    "raw_class": record.get("raw_class_name"),
                    "raw_florence_colour": record.get("raw_colour"),
                    "upper_clothing_color": record.get("upper_clothing_color"),
                    "lower_clothing_color": record.get("lower_clothing_color"),
                    "dominant_clothing_color": record.get("dominant_clothing_color"),
                    "clothing_color_status": record.get("clothing_color_status"),
                    "plate_text": record.get("plate_text"),
                    "plate_status": record.get("plate_status"),
                    "confidence": record.get("plate_confidence"),
                    "first_seen": record.get("first_seen_sec"),
                    "last_seen": record.get("last_seen_sec"),
                    "duration": record.get("duration_sec"),
                    "track": f"{record.get('track_id')} gen {record.get('track_generation')}",
                    "matched_filters": record.get("_ui_matched_filters") or [],
                    "matched_tokens": record.get("_ui_matched_tokens") or [],
                    "warnings": record.get("warnings") or [],
                }
            )
            if cols[2].button("Open detail", key=f"detail_{page_key}_{record.get('record_id')}"):
                st.session_state.selected_record_id = record.get("record_id")
            if show_colour_debug:
                _render_media_sections(st, artifacts, record, colour_only=True)


def _render_media_sections(st: Any, artifacts: Any, record: dict[str, Any], *, colour_only: bool = False) -> None:
    media = collect_record_media(record, artifacts)
    if not colour_only:
        st.subheader("Scene / Object Evidence")
        render_object_evidence_pair(st, media["object_evidence"], run_dir=artifacts.run_dir, repo_root=artifacts.repo_root)
        st.subheader("Additional Vehicle / Person Crops")
        _render_image_list(st, media["vehicle_images"], artifacts, image_type="object_crop")
        st.subheader("Plate Crops")
        _render_image_list(st, media["plate_images"], artifacts, image_type="plate_crop")
        st.subheader("Annotated Plate Diagnostics")
        _render_image_list(st, media["annotated_plate_images"], artifacts, image_type="full_frame")
    st.subheader("Dominant Colour Evidence")
    _render_image_list(st, media["colour_debug_images"], artifacts, image_type="object_crop")


def _render_image_list(st: Any, images: list[dict[str, Any]], artifacts: Any, *, image_type: str) -> None:
    if not images:
        st.caption("No images available.")
        return
    columns = st.columns(3)
    for index, image in enumerate(images[:30]):
        column = columns[index % 3]
        render_evidence_image(
            column,
            image.get("requested_path") or image.get("resolved_path"),
            image_type,
            run_dir=artifacts.run_dir,
            repo_root=artifacts.repo_root,
            caption=Path(image["resolved_path"]).name if image.get("resolved_path") else None,
            missing_message=image.get("placeholder") or "Image missing",
        )


def _render_single_image(st: Any, column: Any, artifacts: Any, path_value: Any, caption: str) -> None:
    status = image_status(path_value, run_dir=artifacts.run_dir, repo_root=artifacts.repo_root)
    if status["resolved_path"]:
        column.image(status["resolved_path"], caption=caption, use_container_width=True)
    else:
        column.caption(f"{caption}: missing")


def _selected_record_id(st: Any, artifacts: Any) -> str | None:
    selected = st.session_state.get("selected_record_id")
    if selected in artifacts.records_by_id:
        return selected
    return None


def _plate_review_payload(record: dict[str, Any], artifacts: Any) -> dict[str, Any]:
    validation = artifacts.plate_validation_by_identity.get(_identity_key(record)) or {}
    return {
        "raw_ocr": validation.get("all_candidates"),
        "corrected_ocr": validation.get("selected_candidate"),
        "final_plate": validation.get("final_plate_text") or record.get("plate_text"),
        "confidence": validation.get("confidence") or record.get("plate_confidence"),
        "corrections_applied": validation.get("metadata", {}).get("corrections_applied"),
    }


def _tracking_row_matches(row: dict[str, Any], status: str) -> bool:
    if not status:
        return True
    if status == "short track":
        return bool(row["short_track"])
    if status == "multiple generations":
        return bool(row["multiple_generations"])
    if status == "temporarily lost":
        return "lost" in " ".join(map(str, row.get("warnings") or [])).lower()
    return str(row.get("lifecycle_status") or "").lower() == status


def _identity_key(record: dict[str, Any]) -> str:
    return f"{record.get('source_id')}:{int(record.get('track_id') or 0)}:{int(record.get('track_generation') or 0)}"


if __name__ == "__main__":
    main()
