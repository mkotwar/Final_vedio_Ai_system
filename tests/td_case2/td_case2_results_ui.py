from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st


DEFAULT_RUN_DIR = Path(
    r"C:\Mukul K\vinfo1\video-search-engine\tests\td_case2\debug_runs\anpr_test_5min_20260707_122758"
)
FILE_NAMES = [
    "00_stage_gate_report.json",
    "01_video_info.json",
    "11_5_vlm_filter_report.json",
    "12_event_candidate_ranking_report.json",
    "13_vlm_event_input_report.json",
    "13_vlm_event_inputs.json",
    "14_vlm_event_review_report.json",
    "14_vlm_event_reviews.json",
    "14_vlm_event_reviews_flat.json",
    "14_final_video_summary.json",
]


def _read_json_if_exists(run_dir: Path, file_name: str) -> tuple[dict[str, Any] | list[Any] | None, str | None]:
    path = run_dir / file_name
    if not path.exists():
        return None, f"Missing: {file_name}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"Could not read {file_name}: {exc}"


def _resolve_run_path(run_dir: Path, path_value: str | None) -> Path | None:
    if not path_value:
        return None
    normalized = str(path_value).strip().replace("\\", "/")
    if not normalized:
        return None
    path = Path(normalized)
    if path.is_absolute():
        return path
    return (run_dir / path).resolve()


def _safe_get(payload: dict[str, Any] | None, key: str, default: Any = "—") -> Any:
    if not isinstance(payload, dict):
        return default
    return payload.get(key, default)


def _status_badge(status_value: str) -> str:
    normalized = str(status_value or "").lower()
    if normalized == "success":
        return "✅ success"
    if normalized in {"failed", "error"}:
        return "❌ failed"
    if normalized in {"missing", "not_found"}:
        return "⚠️ missing"
    return f"ℹ️ {normalized or 'unknown'}"


def _render_missing_messages(messages: list[str]) -> None:
    if not messages:
        return
    st.subheader("Missing Files / Read Warnings")
    for message in messages:
        st.warning(message)


def _build_stage_rows(stage_gate: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(stage_gate, dict):
        return []
    steps = stage_gate.get("steps", {})
    if not isinstance(steps, dict):
        return []
    rows: list[dict[str, Any]] = []
    for step_name, step_payload in steps.items():
        payload = step_payload if isinstance(step_payload, dict) else {}
        count_parts: list[str] = []
        preferred_keys = [
            "input_candidate_count",
            "candidate_events_created",
            "selected_top_k_count",
            "vlm_inputs_created",
            "inputs_reviewed",
            "event_visible_count",
            "normal_context_count",
            "searchable_records",
            "records_with_verified_plate",
        ]
        for key in preferred_keys:
            if key in payload:
                count_parts.append(f"{key}={payload.get(key)}")
        if not count_parts:
            for key, value in payload.items():
                if key == "status":
                    continue
                if isinstance(value, (int, float, bool, str)):
                    count_parts.append(f"{key}={value}")
                if len(count_parts) >= 4:
                    break
        rows.append(
            {
                "step_name": step_name,
                "status": _status_badge(str(payload.get("status", "unknown"))),
                "important_counts": " | ".join(count_parts) if count_parts else "—",
            }
        )
    return rows


def _moment_image_path(run_dir: Path, review: dict[str, Any]) -> tuple[Path | None, str]:
    strip_path = _resolve_run_path(run_dir, review.get("temporal_strip_path"))
    if strip_path is not None and strip_path.exists():
        return strip_path, "Temporal strip"
    contact_sheet_path = _resolve_run_path(run_dir, review.get("contact_sheet_path"))
    if contact_sheet_path is not None and contact_sheet_path.exists():
        return contact_sheet_path, "Contact sheet fallback"
    return None, "Image missing"


def _review_list(reviews_payload: dict[str, Any] | None, flat_payload: list[Any] | None) -> list[dict[str, Any]]:
    if isinstance(reviews_payload, dict):
        reviews = reviews_payload.get("reviews", [])
        if isinstance(reviews, list):
            return [item for item in reviews if isinstance(item, dict)]
    if isinstance(flat_payload, list):
        return [item for item in flat_payload if isinstance(item, dict)]
    return []


def main() -> None:
    st.set_page_config(page_title="TD Case 2 Video Review Dashboard", layout="wide")
    st.title("TD Case 2 Video Review Dashboard")

    st.sidebar.header("Run Directory")
    run_dir_input = st.sidebar.text_input("run_dir", value=str(DEFAULT_RUN_DIR))
    run_dir = Path(run_dir_input).expanduser()
    if not run_dir.is_absolute():
        run_dir = run_dir.resolve()

    st.sidebar.caption("This dashboard only reads existing td_case2 JSON outputs and images.")
    st.caption(f"Current run directory: `{run_dir}`")

    loaded: dict[str, dict[str, Any] | list[Any] | None] = {}
    messages: list[str] = []
    for file_name in FILE_NAMES:
        payload, message = _read_json_if_exists(run_dir, file_name)
        loaded[file_name] = payload
        if message:
            messages.append(message)

    stage_gate = loaded["00_stage_gate_report.json"] if isinstance(loaded["00_stage_gate_report.json"], dict) else None
    video_info = loaded["01_video_info.json"] if isinstance(loaded["01_video_info.json"], dict) else None
    step11_5_report = loaded["11_5_vlm_filter_report.json"] if isinstance(loaded["11_5_vlm_filter_report.json"], dict) else None
    step12_report = loaded["12_event_candidate_ranking_report.json"] if isinstance(loaded["12_event_candidate_ranking_report.json"], dict) else None
    step13_report = loaded["13_vlm_event_input_report.json"] if isinstance(loaded["13_vlm_event_input_report.json"], dict) else None
    reviews_payload = loaded["14_vlm_event_reviews.json"] if isinstance(loaded["14_vlm_event_reviews.json"], dict) else None
    flat_reviews_payload = loaded["14_vlm_event_reviews_flat.json"] if isinstance(loaded["14_vlm_event_reviews_flat.json"], list) else None
    step14_report = loaded["14_vlm_event_review_report.json"] if isinstance(loaded["14_vlm_event_review_report.json"], dict) else None
    final_summary = loaded["14_final_video_summary.json"] if isinstance(loaded["14_final_video_summary.json"], dict) else None

    _render_missing_messages(messages)

    st.subheader("Summary")
    summary_cols = st.columns(4)
    summary_cols[0].metric("Video Name", _safe_get(video_info, "video_name"))
    summary_cols[1].metric("Duration", _safe_get(video_info, "duration_text"))
    summary_cols[2].metric("Overall Status", _safe_get(final_summary, "overall_status"))
    summary_cols[3].metric("Headline", _safe_get(final_summary, "headline"))

    summary_cols_2 = st.columns(4)
    summary_cols_2[0].metric("Event Count", _safe_get(final_summary, "event_count", 0))
    summary_cols_2[1].metric("Normal Context Count", _safe_get(final_summary, "normal_context_count", 0))
    summary_cols_2[2].metric("Uncertain Count", _safe_get(final_summary, "uncertain_count", 0))
    summary_cols_2[3].metric("Recommended Action", _safe_get(final_summary, "recommended_action"))

    st.subheader("Pipeline Stage Status")
    stage_rows = _build_stage_rows(stage_gate)
    if stage_rows:
        st.dataframe(stage_rows, use_container_width=True, hide_index=True)
    else:
        st.info("Missing: 00_stage_gate_report.json")

    st.subheader("Step 11.5 Filter Summary")
    if step11_5_report:
        filter_cols = st.columns(4)
        filter_cols[0].metric("Input Candidates", _safe_get(step11_5_report, "input_candidate_count", 0))
        filter_cols[1].metric("Checked by VLM", _safe_get(step11_5_report, "candidates_checked_by_vlm", 0))
        filter_cols[2].metric("Accepted Yes", _safe_get(step11_5_report, "accepted_yes_count", 0))
        filter_cols[3].metric("Uncertain", _safe_get(step11_5_report, "uncertain_count", 0))

        filter_cols_2 = st.columns(4)
        filter_cols_2[0].metric("Rejected No", _safe_get(step11_5_report, "rejected_no_count", 0))
        filter_cols_2[1].metric(
            "Fallback Normal Context",
            _safe_get(step11_5_report, "fallback_normal_context_count", 0),
        )
        filter_cols_2[2].metric(
            "Final Filtered Candidates",
            _safe_get(step11_5_report, "final_filtered_candidate_count", 0),
        )
        filter_cols_2[3].metric(
            "Avg Inference Time (s)",
            _safe_get(step11_5_report, "average_inference_time_seconds", "—"),
        )
    else:
        st.info("Missing: 11_5_vlm_filter_report.json")

    st.subheader("Step 13 / Step 14 Review Summary")
    review_cols = st.columns(6)
    review_cols[0].metric(
        "Step 13 Inputs Created",
        _safe_get(step13_report, "vlm_inputs_created", "—"),
    )
    review_cols[1].metric(
        "Temporal Strips",
        _safe_get(step13_report, "temporal_strips_created", "—"),
    )
    review_cols[2].metric(
        "Step 14 Inputs Reviewed",
        _safe_get(step14_report, "inputs_reviewed", "—"),
    )
    review_cols[3].metric(
        "Event Visible",
        _safe_get(step14_report, "event_visible_count", "—"),
    )
    review_cols[4].metric(
        "Normal Context",
        _safe_get(step14_report, "normal_context_count", "—"),
    )
    review_cols[5].metric(
        "Uncertain",
        _safe_get(step14_report, "uncertain_count", "—"),
    )

    st.subheader("Reviewed Moments Timeline")
    reviews = _review_list(reviews_payload, flat_reviews_payload)
    if not reviews:
        st.info("Missing: 14_vlm_event_reviews.json / 14_vlm_event_reviews_flat.json")
    else:
        for review in reviews:
            model_review = review.get("model_review", review)
            image_path, image_label = _moment_image_path(run_dir, review)
            with st.container(border=True):
                header_cols = st.columns([1.1, 1.1, 1.0, 1.0, 0.9, 0.9])
                header_cols[0].markdown(f"**Timestamp**  \n{review.get('best_timestamp_text', '—')}")
                header_cols[1].markdown(f"**Review Decision**  \n{model_review.get('review_decision', '—')}")
                header_cols[2].markdown(f"**Event Visible**  \n{model_review.get('event_visible', '—')}")
                header_cols[3].markdown(f"**Event Type**  \n{model_review.get('event_type', '—')}")
                header_cols[4].markdown(f"**Risk Level**  \n{model_review.get('risk_level', '—')}")
                header_cols[5].markdown(f"**Confidence**  \n{model_review.get('confidence', '—')}")

                st.markdown(f"**Summary Caption**: {model_review.get('summary_caption', '—')}")
                meta_cols = st.columns(2)
                meta_cols[0].markdown(f"**Needs Human Review**: {model_review.get('needs_human_review', '—')}")
                meta_cols[1].markdown(
                    f"**Source Candidate IDs**: {', '.join(review.get('source_candidate_ids', [])) or '—'}"
                )

                if image_path is not None:
                    st.image(str(image_path), caption=f"{image_label}: {image_path.name}", use_container_width=True)
                else:
                    st.warning(
                        f"Missing image for {review.get('vlm_input_id', 'unknown review')}. "
                        f"Tried strip/contact sheet relative to run_dir."
                    )

                with st.expander("Optional Details"):
                    st.markdown(f"**What Is Visible**: {model_review.get('what_is_visible', '—')}")
                    st.markdown(f"**Why Decision**: {model_review.get('why_decision', '—')}")
                    objects_seen = model_review.get("objects_seen", [])
                    if isinstance(objects_seen, list):
                        st.markdown(f"**Objects Seen**: {', '.join(str(item) for item in objects_seen) or '—'}")
                    else:
                        st.markdown(f"**Objects Seen**: {objects_seen}")
                    st.markdown("**Step 11.5 Filter Context**")
                    st.json(review.get("step11_5_filter_context", {}))
                    raw_output = review.get("raw_output_text")
                    if raw_output:
                        st.markdown("**Raw Model Output**")
                        st.code(str(raw_output), language="json")

    st.subheader("Search / Object Summary")
    search_payload = {}
    if stage_gate and isinstance(stage_gate.get("steps"), dict):
        search_payload = stage_gate["steps"].get("07_search_index_enrichment", {}) or {}
    search_cols = st.columns(4)
    search_cols[0].metric("Total Vehicle Records", _safe_get(search_payload, "total_vehicle_records", "—"))
    search_cols[1].metric("Verified Plates", _safe_get(search_payload, "records_with_verified_plate", "—"))
    search_cols[2].metric("Records With Colors", _safe_get(search_payload, "records_with_color", "—"))
    search_cols[3].metric("Searchable Records", _safe_get(search_payload, "searchable_records", "—"))

    st.subheader("Final Summary")
    if final_summary:
        st.info(_safe_get(final_summary, "summary"))
        st.json(final_summary)
    else:
        st.info("Missing: 14_final_video_summary.json")

    st.divider()
    st.caption(r'Run with: cd "C:\Mukul K\vinfo1\video-search-engine"')
    st.caption(r".venv\Scripts\python.exe -m streamlit run tests\td_case2\td_case2_results_ui.py")


if __name__ == "__main__":
    main()
