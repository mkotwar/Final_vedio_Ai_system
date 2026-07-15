from __future__ import annotations

import base64
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from event_preview_video import build_event_preview_clip, get_existing_event_preview, resolve_run_path as resolve_preview_path
from traffic_search_common import resolve_run_path, run_traffic_search


SESSION_RUN_DIR_KEY = "td_case2_run_dir"
DEBUG_RUNS_DIR = Path(__file__).resolve().parent / "debug_runs"
IMAGE_SIZE_MAP = {
    "Small": {"full": (320, 180), "crop": (180, 140)},
    "Medium": {"full": (360, 200), "crop": (220, 160)},
    "Large": {"full": (420, 236), "crop": (260, 190)},
}
SEARCH_READY_FILES = [
    "01_video_info.json",
    "07B_traffic_object_search_index.json",
    "09B_universal_search_cards.json",
    "07B_traffic_object_search_index_report.json",
]


def _read_json_if_exists(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_payloads(run_dir: Path) -> dict[str, Any]:
    names = [
        "00_stage_gate_report.json",
        "01_video_info.json",
        "03A_yolo_model_audit.json",
        "04B_tracking_report.json",
        "07B_traffic_object_search_index.json",
        "07B_traffic_object_search_index_report.json",
        "08B_dynamic_search_validation_report.json",
        "09B_universal_search_cards_flat.json",
        "10B_universal_search_demo_report.json",
        "14_vlm_event_review_report.json",
        "pipeline_status.json",
    ]
    return {name: _read_json_if_exists(run_dir / name) for name in names}


def _is_search_ready_run_dir(run_dir: Path) -> bool:
    return run_dir.exists() and run_dir.is_dir() and all((run_dir / name).exists() for name in SEARCH_READY_FILES)


def _find_latest_valid_run_dir() -> Path | None:
    if not DEBUG_RUNS_DIR.exists() or not DEBUG_RUNS_DIR.is_dir():
        return None
    candidates = [path for path in DEBUG_RUNS_DIR.iterdir() if _is_search_ready_run_dir(path)]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _resolve_initial_run_dir() -> tuple[Path | None, str]:
    env_run_dir = os.environ.get("TD_CASE2_RUN_DIR", "").strip()
    if env_run_dir:
        resolved = Path(env_run_dir).expanduser()
        if not resolved.is_absolute():
            resolved = resolved.resolve()
        st.session_state[SESSION_RUN_DIR_KEY] = str(resolved)
        return resolved, "environment variable TD_CASE2_RUN_DIR"

    session_value = str(st.session_state.get(SESSION_RUN_DIR_KEY, "") or "").strip()
    if session_value:
        resolved = Path(session_value).expanduser()
        if not resolved.is_absolute():
            resolved = resolved.resolve()
        return resolved, "session state"

    latest = _find_latest_valid_run_dir()
    if latest is not None:
        st.session_state[SESSION_RUN_DIR_KEY] = str(latest)
        return latest, "latest valid run"
    return None, "manual input"


def _format_seconds(value: float | int | None) -> str:
    total_seconds = int(round(float(value or 0.0)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .traffic-metric-card {
            border: 1px solid #d9dde3;
            border-radius: 14px;
            padding: 14px 16px;
            background: #fbfcfe;
            min-height: 88px;
        }
        .traffic-metric-label {
            font-size: 0.82rem;
            color: #64707d;
            margin-bottom: 6px;
        }
        .traffic-metric-value {
            font-size: 1.2rem;
            font-weight: 600;
            color: #15202b;
        }
        .traffic-card {
            border: 1px solid #d9dde3;
            border-radius: 16px;
            padding: 16px;
            margin-bottom: 16px;
            background: #fafbfd;
        }
        .traffic-preview-box {
            display: flex;
            align-items: center;
            justify-content: center;
            background: #f4f6f8;
            border: 1px solid #d8dde5;
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 8px;
        }
        .traffic-preview-box img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
            margin: 0 auto;
            background: #f4f6f8;
        }
        .traffic-kicker {
            color: #64707d;
            font-size: 0.85rem;
            margin-bottom: 10px;
        }
        .traffic-badges {
            font-size: 0.8rem;
            color: #55606c;
            margin-bottom: 10px;
        }
        .traffic-empty {
            border: 1px dashed #cfd7df;
            border-radius: 14px;
            padding: 18px;
            background: #fafbfd;
            color: #5d6875;
        }
        .traffic-fallback-timeline {
            position: relative;
            height: 82px;
            border: 1px solid #d9dde3;
            border-radius: 12px;
            background: linear-gradient(90deg, #f8fafc 0%, #eef3f8 100%);
            margin: 10px 0 12px 0;
        }
        .traffic-fallback-dot {
            position: absolute;
            top: 24px;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #0b6bcb;
            transform: translateX(-50%);
            box-shadow: 0 0 0 4px rgba(11, 107, 203, 0.12);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_metric_card(label: str, value: Any) -> None:
    st.markdown(
        f"""
        <div class="traffic-metric-card">
            <div class="traffic-metric-label">{label}</div>
            <div class="traffic-metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _image_to_data_uri(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(path.suffix.lower())
    if not mime:
        return None
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _render_image_box(path: Path | None, width: int, height: int, label: str) -> None:
    data_uri = _image_to_data_uri(path)
    if not data_uri:
        st.caption(f"{label}: missing")
        return
    st.markdown(
        f"""
        <div class="traffic-preview-box" style="width:{width}px; height:{height}px;">
            <img src="{data_uri}" alt="{label}">
        </div>
        """,
        unsafe_allow_html=True,
    )


def _match_explanation_text(explanation: dict[str, Any] | None) -> str:
    if not explanation:
        return "-"
    ordered = [
        "matched_class",
        "matched_verified_color",
        "matched_possible_color",
        "matched_plate",
        "matched_timestamp",
        "matched_text_fallback",
    ]
    enabled = [name for name in ordered if explanation.get(name)]
    return ", ".join(enabled) if enabled else "-"


def _record_badges(record: dict[str, Any]) -> list[str]:
    badges: list[str] = []
    if record.get("verified_vehicle_color"):
        badges.append("verified_color")
    elif record.get("possible_vehicle_color"):
        badges.append("possible_color")
    else:
        badges.append("unknown_color")
    if (record.get("_match_explanation") or {}).get("matched_verified_color"):
        badges.append("verified_color_match")
    if (record.get("_match_explanation") or {}).get("matched_possible_color"):
        badges.append("possible_color_match")
    if record.get("color_warning") == "possible_tail_light_color_confusion":
        badges.append("tail_light_warning")
    if record.get("color_status") == "conflict":
        badges.append("color_conflict")
    if record.get("color_warning") == "single_or_fallback_color_evidence":
        badges.append("fallback_color_evidence")
    return badges


def _has_active_filters(
    query: str,
    selected_class: str,
    selected_verified_color: str,
    selected_possible_color: str,
    plate_filter: str,
    selected_object_type: str,
    selected_quality: str,
    time_range: tuple[float, float],
    max_time: float,
    include_possible_colors: bool,
    include_possible_plates: bool,
) -> bool:
    return any(
        [
            bool(query.strip()),
            selected_class != "All",
            selected_verified_color != "All",
            include_possible_colors and selected_possible_color != "All",
            plate_filter != "All",
            selected_object_type != "All",
            selected_quality != "All",
            time_range != (0.0, float(max_time)),
            include_possible_plates,
        ]
    )


def _filter_records(
    records: list[dict[str, Any]],
    run_dir: Path,
    query: str,
    selected_class: str,
    selected_verified_color: str,
    selected_possible_color: str,
    plate_filter: str,
    selected_object_type: str,
    selected_quality: str,
    time_range: tuple[float, float],
    include_possible_colors: bool,
    include_possible_plates: bool,
) -> list[dict[str, Any]]:
    query = query.strip()
    if query:
        search_result = run_traffic_search(
            records,
            query,
            top_k=max(500, len(records)),
            time_tolerance_seconds=5.0,
            require_full_frame=False,
            run_dir=run_dir,
            include_uncertain_colors=include_possible_colors,
            include_possible_plates=include_possible_plates,
        )
        matched_records = []
        for match in list(search_result.get("matches", [])):
            record = dict(match.get("record", {}))
            record["_match_explanation"] = match.get("match_explanation")
            record["_match_score"] = match.get("score")
            matched_records.append(record)
    else:
        matched_records = [dict(record) for record in records]

    filtered_records = []
    for record in matched_records:
        if selected_class != "All" and record.get("class_name") != selected_class:
            continue
        verified_color = record.get("verified_vehicle_color")
        possible_color = record.get("possible_vehicle_color")
        if selected_verified_color != "All" and verified_color != selected_verified_color:
            continue
        if selected_possible_color != "All":
            if not include_possible_colors:
                continue
            if possible_color != selected_possible_color and verified_color != selected_possible_color:
                continue
        if selected_object_type != "All" and record.get("object_type") != selected_object_type:
            continue
        if selected_quality != "All" and record.get("quality") != selected_quality:
            continue
        plate_status = str(record.get("verified_plate_status", "") or "")
        if plate_filter == "Verified Plate Only" and plate_status != "verified":
            continue
        if plate_filter == "Possible Plate Allowed" and plate_status not in {"verified", "possible"}:
            continue
        if plate_filter == "No Verified Plate" and plate_status == "verified":
            continue
        timestamp_seconds = float(record.get("timestamp_seconds", 0.0) or 0.0)
        if not (time_range[0] <= timestamp_seconds <= time_range[1]):
            continue
        filtered_records.append(record)
    return filtered_records


def _build_search_summary(records: list[dict[str, Any]], query: str) -> dict[str, Any]:
    timestamps = [float(item.get("timestamp_seconds", 0.0) or 0.0) for item in records]
    explanation_counts = Counter()
    for item in records:
        for key, value in (item.get("_match_explanation") or {}).items():
            if value:
                explanation_counts[key] += 1
    return {
        "query": query or "Filters only",
        "total_matches": len(records),
        "first_seen": _format_seconds(min(timestamps)) if timestamps else "-",
        "last_seen": _format_seconds(max(timestamps)) if timestamps else "-",
        "unique_tracks": len({str(item.get("track_id")) for item in records if item.get("track_id")}),
        "verified_plates_found": sum(1 for item in records if item.get("verified_license_plate")),
        "verified_colors_found": sum(1 for item in records if item.get("verified_vehicle_color")),
        "verified_color_matches": int(explanation_counts.get("matched_verified_color", 0)),
        "possible_color_matches": int(explanation_counts.get("matched_possible_color", 0)),
        "verified_plate_matches": int(explanation_counts.get("matched_plate", 0)),
        "class_matches": int(explanation_counts.get("matched_class", 0)),
        "class_distribution": Counter(str(item.get("class_name", "") or "unknown") for item in records),
        "verified_color_distribution": Counter(
            str(item.get("verified_vehicle_color", "") or "") for item in records if item.get("verified_vehicle_color")
        ),
        "possible_color_distribution": Counter(
            str(item.get("possible_vehicle_color", "") or "") for item in records if item.get("possible_vehicle_color")
        ),
        "quality_distribution": Counter(str(item.get("quality", "") or "unknown") for item in records),
    }


def _make_distribution_df(counter: Counter[str], key_name: str) -> pd.DataFrame:
    return pd.DataFrame([{key_name: key, "count": value} for key, value in counter.most_common()])


def _make_cluster(existing_clusters: list[dict[str, Any]], cluster_records: list[dict[str, Any]]) -> dict[str, Any]:
    start_seconds = float(cluster_records[0].get("timestamp_seconds", 0.0) or 0.0)
    end_seconds = float(cluster_records[-1].get("timestamp_seconds", 0.0) or 0.0)
    class_counts = Counter(str(item.get("class_name", "") or "unknown") for item in cluster_records)
    color_counts = Counter(str(item.get("verified_vehicle_color") or item.get("possible_vehicle_color") or "-") for item in cluster_records)
    plate_values = sorted({str(item.get("verified_license_plate") or "-") for item in cluster_records if item.get("verified_license_plate")})
    track_ids = sorted({str(item.get("track_id")) for item in cluster_records if item.get("track_id")})
    return {
        "cluster_id": f"cluster_{len(existing_clusters) + 1:03d}",
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "start_time": _format_seconds(start_seconds),
        "end_time": _format_seconds(end_seconds),
        "count": len(cluster_records),
        "classes": ", ".join(name for name, _ in class_counts.most_common(3)),
        "colors": ", ".join(name for name, _ in color_counts.most_common(3) if name and name != "-"),
        "plates": ", ".join(plate_values) if plate_values else "-",
        "track_ids": ", ".join(track_ids) if track_ids else "-",
        "label": f"{_format_seconds(start_seconds)}-{ _format_seconds(end_seconds)} | {len(cluster_records)} matches",
        "records": cluster_records,
    }


def _cluster_records(records: list[dict[str, Any]], threshold_seconds: float) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda item: float(item.get("timestamp_seconds", 0.0) or 0.0))
    clusters: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for record in ordered:
        ts = float(record.get("timestamp_seconds", 0.0) or 0.0)
        if not current:
            current = [record]
            continue
        prev_ts = float(current[-1].get("timestamp_seconds", 0.0) or 0.0)
        if ts - prev_ts <= threshold_seconds:
            current.append(record)
        else:
            clusters.append(_make_cluster(clusters, current))
            current = [record]
    if current:
        clusters.append(_make_cluster(clusters, current))
    return clusters


def _render_timeline(clusters: list[dict[str, Any]], duration_seconds: float) -> None:
    if not clusters:
        st.info("No timeline clusters for the current result set.")
        return
    cluster_centers = [(item["start_seconds"] + item["end_seconds"]) / 2.0 for item in clusters]
    hover_text = [
        "<br>".join(
            [
                f"{item['start_time']} to {item['end_time']}",
                f"Matches: {item['count']}",
                f"Classes: {item['classes'] or '-'}",
                f"Colors: {item['colors'] or '-'}",
                f"Plates: {item['plates'] or '-'}",
            ]
        )
        for item in clusters
    ]
    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=cluster_centers,
                y=[1] * len(cluster_centers),
                mode="markers",
                marker={"size": [max(12, min(26, 10 + item["count"])) for item in clusters], "color": "#0b6bcb"},
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                name="Clusters",
            )
        )
        fig.update_layout(
            height=200,
            margin={"l": 20, "r": 20, "t": 10, "b": 20},
            showlegend=False,
            yaxis={"visible": False},
            xaxis_title="Video time (seconds)",
        )
        fig.update_xaxes(range=[0, max(duration_seconds, max(cluster_centers, default=0.0), 1.0)])
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        safe_duration = max(duration_seconds, max(cluster_centers, default=0.0), 1.0)
        dots = []
        for center in cluster_centers:
            left_percent = max(0.0, min(100.0, (center / safe_duration) * 100.0))
            dots.append(f'<div class="traffic-fallback-dot" style="left:{left_percent:.2f}%"></div>')
        html = '<div class="traffic-fallback-timeline">' + "".join(dots) + "</div>"
        st.markdown(html, unsafe_allow_html=True)


def _journey_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("track_id"):
            grouped[str(record.get("track_id"))].append(record)
    rows = []
    for track_id, items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda item: float(item.get("timestamp_seconds", 0.0) or 0.0))
        first = ordered[0]
        rows.append(
            {
                "track_id": track_id,
                "class": first.get("class_name"),
                "verified_color": first.get("verified_vehicle_color") or "Unknown",
                "possible_color": first.get("possible_vehicle_color") or "-",
                "verified_plate": first.get("verified_license_plate") or "-",
                "first_seen": _format_seconds(float(ordered[0].get("timestamp_seconds", 0.0) or 0.0)),
                "last_seen": _format_seconds(float(ordered[-1].get("timestamp_seconds", 0.0) or 0.0)),
                "number_of_detections": len(ordered),
                "quality": first.get("quality") or "-",
                "records": ordered,
            }
        )
    return rows


def _render_card(record: dict[str, Any], run_dir: Path, image_size: str) -> None:
    dims = IMAGE_SIZE_MAP[image_size]
    full_frame = resolve_run_path(run_dir, record.get("full_frame_path"))
    crop = resolve_run_path(run_dir, record.get("crop_path"))
    st.markdown('<div class="traffic-card">', unsafe_allow_html=True)
    st.markdown(f"### {str(record.get('class_name', 'Object')).title()}")
    st.markdown(f'<div class="traffic-kicker">{record.get("timestamp_text", "-")}</div>', unsafe_allow_html=True)

    image_cols = st.columns([1.35, 1.0])
    with image_cols[0]:
        _render_image_box(full_frame, dims["full"][0], dims["full"][1], "Full frame")
    with image_cols[1]:
        _render_image_box(crop, dims["crop"][0], dims["crop"][1], "Crop")

    info_cols = st.columns(4)
    info_cols[0].markdown(f"**Verified color**  \n{record.get('verified_vehicle_color') or 'Unknown'}")
    info_cols[1].markdown(f"**Possible color**  \n{record.get('possible_vehicle_color') or '-'}")
    info_cols[2].markdown(f"**Verified plate**  \n{record.get('verified_license_plate') or '-'}")
    info_cols[3].markdown(f"**Track ID**  \n{record.get('track_id') or '-'}")

    meta_cols = st.columns(3)
    meta_cols[0].markdown(f"**Confidence / Quality**  \n{record.get('confidence', '-')} / {record.get('quality', '-')}")
    first_seen = record.get("first_seen_seconds")
    last_seen = record.get("last_seen_seconds")
    duration_seconds = float(record.get("duration_seconds", 0.0) or 0.0)
    if isinstance(first_seen, (int, float)) and isinstance(last_seen, (int, float)):
        detection_window = f"{_format_seconds(first_seen)} to {_format_seconds(last_seen)}"
    else:
        detection_window = record.get("timestamp_text", "-")
    meta_cols[1].markdown(f"**Detected window**  \n{detection_window} ({duration_seconds:.2f}s)")
    badges = _record_badges(record)
    meta_cols[2].markdown(f"**Labels**  \n{', '.join(badges) if badges else '-'}")

    existing_clip = get_existing_event_preview(run_dir, record)
    clip_cols = st.columns([1.1, 1.0])
    with clip_cols[0]:
        if existing_clip:
            clip_path = resolve_preview_path(run_dir, existing_clip.get("event_clip_path"))
            if clip_path is not None and clip_path.exists():
                st.video(str(clip_path))
                st.caption(
                    "Event preview clip | "
                    f"{existing_clip.get('start_timestamp_text', '-')} to {existing_clip.get('end_timestamp_text', '-')} | "
                    f"{float(existing_clip.get('detection_duration_seconds', 0.0) or 0.0):.2f}s"
                )
        else:
            if st.button("Generate event clip", key=f"generate_clip_{record.get('object_record_id')}"):
                with st.spinner("Building event preview clip..."):
                    build_event_preview_clip(run_dir, record)
                st.rerun()
    with clip_cols[1]:
        if existing_clip:
            st.markdown(
                f"**Clip frames**  \n{existing_clip.get('clip_frame_count', '-')}"
            )
            st.markdown(
                f"**Source frames**  \n{existing_clip.get('source_frame_count', '-')}"
            )
            st.markdown(
                f"**Best frame**  \n{record.get('timestamp_text', '-')}"
            )
        else:
            st.info("Generate the event clip to preview the processed event frames in the UI.")

    with st.expander("Details"):
        st.write(
            {
                "possible_ocr": record.get("possible_plate_text") or "-",
                "match_explanation": _match_explanation_text(record.get("_match_explanation")),
                "warnings": record.get("warnings", []),
                "search_text": record.get("search_text"),
                "bbox_xyxy": record.get("bbox_xyxy"),
                "source_type": record.get("source_type"),
                "object_record_id": record.get("object_record_id"),
                "full_frame_path": record.get("full_frame_path"),
                "crop_path": record.get("crop_path"),
                "existing_event_clip": existing_clip,
            }
        )
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="TD Case 2 Traffic Search UI", layout="wide")
    _inject_styles()
    st.title("TD Case 2 Traffic Search UI")

    st.sidebar.header("Run Directory")
    resolved_run_dir, run_dir_source = _resolve_initial_run_dir()
    initial_value = str(resolved_run_dir) if resolved_run_dir is not None else ""
    run_dir_input = st.sidebar.text_input("Run Directory", value=initial_value)
    button_cols = st.sidebar.columns(3)
    load_selected_run = button_cols[0].button("Load selected run")
    load_latest_run = button_cols[1].button("Load latest run")
    clear_cached_data = button_cols[2].button("Clear cached data")

    if clear_cached_data:
        st.cache_data.clear()
        st.rerun()

    if load_latest_run:
        latest = _find_latest_valid_run_dir()
        if latest is None:
            st.sidebar.error("No valid search-ready run directory found under tests/td_case2/debug_runs.")
            st.stop()
        st.session_state[SESSION_RUN_DIR_KEY] = str(latest)
        st.rerun()

    if load_selected_run:
        st.session_state[SESSION_RUN_DIR_KEY] = run_dir_input.strip()
        st.rerun()

    active_run_dir_text = str(st.session_state.get(SESSION_RUN_DIR_KEY) or run_dir_input.strip() or initial_value).strip()
    if not active_run_dir_text:
        st.error("Selected run directory is invalid or search outputs are missing.")
        st.stop()

    run_dir = Path(active_run_dir_text).expanduser()
    if not run_dir.is_absolute():
        run_dir = run_dir.resolve()

    if os.environ.get("TD_CASE2_RUN_DIR", "").strip():
        run_dir_source = "environment variable TD_CASE2_RUN_DIR"
    elif str(st.session_state.get(SESSION_RUN_DIR_KEY, "") or "").strip():
        run_dir_source = "session state" if not load_latest_run else "latest valid run"

    if not _is_search_ready_run_dir(run_dir):
        st.error("Selected run directory is invalid or search outputs are missing.")
        st.caption("Selected run directory is not search-ready.")
        with st.expander("Run directory debug"):
            st.write("os.environ TD_CASE2_RUN_DIR:", os.environ.get("TD_CASE2_RUN_DIR"))
            st.write("session_state run_dir:", st.session_state.get(SESSION_RUN_DIR_KEY))
            st.write("resolved_run_dir:", str(run_dir))
            for name in SEARCH_READY_FILES:
                st.write(f"{name} exists:", (run_dir / name).exists())
        st.stop()

    payloads = _load_payloads(run_dir)
    video_info = payloads.get("01_video_info.json") or {}
    audit = payloads.get("03A_yolo_model_audit.json") or {}
    tracking_report = payloads.get("04B_tracking_report.json") or {}
    index_payload = payloads.get("07B_traffic_object_search_index.json") or {}
    index_report = payloads.get("07B_traffic_object_search_index_report.json") or {}
    validation_report = payloads.get("08B_dynamic_search_validation_report.json") or {}
    cards_flat = payloads.get("09B_universal_search_cards_flat.json") or []
    demo_report = payloads.get("10B_universal_search_demo_report.json") or {}
    step14_report = payloads.get("14_vlm_event_review_report.json") or {}
    pipeline_status = payloads.get("pipeline_status.json") or {}

    records = list(index_payload.get("records", [])) if isinstance(index_payload, dict) else []
    cards = [item for item in cards_flat if isinstance(item, dict)]
    duration_seconds = float(video_info.get("duration_seconds", 0.0) or 0.0)
    possible_color_warning_count = sum(
        1 for item in records if str(item.get("color_warning", "") or "") == "single_or_fallback_color_evidence"
    )

    missing_files = [name for name, payload in payloads.items() if payload is None]
    for file_name in missing_files:
        st.warning(f"Missing: {file_name}")

    st.caption("Active run directory:")
    st.code(str(run_dir))
    st.caption(f"Run directory source: {run_dir_source}")

    folder_name = run_dir.name.lower()
    video_name = str(video_info.get("video_name", "") or "")
    if folder_name and video_name:
        normalized_video_name = video_name.lower().replace(".mp4", "")
        if normalized_video_name not in folder_name and folder_name not in normalized_video_name:
            st.warning("WARNING: Loaded video info does not match selected run folder. Clear cache and reload.")

    st.subheader("Prerequisite Health")
    health_cols = st.columns(6)
    with health_cols[0]:
        _render_metric_card("Video", video_info.get("video_name", "-"))
    with health_cols[1]:
        _render_metric_card("Duration", video_info.get("duration_text", "-"))
    with health_cols[2]:
        _render_metric_card("Searchable objects", len(records))
    with health_cols[3]:
        _render_metric_card("Verified plates", index_report.get("records_with_verified_plate", 0))
    with health_cols[4]:
        _render_metric_card("Verified colors", index_report.get("records_with_verified_color", 0))
    with health_cols[5]:
        _render_metric_card("Validation pass rate", validation_report.get("pass_rate", "-"))

    status_cols = st.columns(6)
    object_search_status = "ready" if _is_search_ready_run_dir(run_dir) else "not ready"
    vlm_backend = pipeline_status.get("vlm_backend") or step14_report.get("vlm_backend") or os.environ.get("TD_CASE2_VLM_BACKEND", "local_qwen")
    vlm_status = pipeline_status.get("vlm_status") or step14_report.get("status") or "not run"
    api_model = pipeline_status.get("api_model") or step14_report.get("api_model") or "-"
    step14_status = step14_report.get("status", "not run")
    last_updated_path = run_dir / "pipeline_status.json"
    if not last_updated_path.exists():
        last_updated_path = run_dir / "14_vlm_event_review_report.json"
    last_updated = _format_seconds(0)
    if last_updated_path.exists():
        last_updated = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_updated_path.stat().st_mtime))
    with status_cols[0]:
        _render_metric_card("Object Search status", object_search_status)
    with status_cols[1]:
        _render_metric_card("VLM Summary status", vlm_status)
    with status_cols[2]:
        _render_metric_card("VLM backend", vlm_backend)
    with status_cols[3]:
        _render_metric_card("API model", api_model)
    with status_cols[4]:
        _render_metric_card("Step 14 status", step14_status)
    with status_cols[5]:
        _render_metric_card("Last updated", last_updated)
    if vlm_backend == "api_qwen":
        if vlm_status == "ready":
            st.caption("VLM Summary: Ready from Qwen API")
        elif vlm_status == "running":
            st.caption("VLM Summary: Running with Qwen API")

    with st.expander("Advanced diagnostics"):
        diag_cols = st.columns(5)
        diag_cols[0].metric("YOLO models loaded", sum(1 for item in audit.get("models", []) if item.get("load_status") == "success"))
        diag_cols[1].metric("Traffic classes detected", len(index_report.get("class_counts", {})))
        diag_cols[2].metric("Tracking quality", tracking_report.get("track_quality_counts", {}).get("good", "-"))
        diag_cols[3].metric("Possible colors", index_report.get("records_with_possible_color", 0))
        diag_cols[4].metric("Unknown colors", index_report.get("records_with_unknown_color", 0))
        diag_cols_2 = st.columns(5)
        diag_cols_2[0].metric("Tail-light warnings", index_report.get("tail_light_confusion_warning_count", 0))
        diag_cols_2[1].metric("Conflict warnings", index_report.get("color_conflict_count", 0))
        diag_cols_2[2].metric("Fallback warnings", possible_color_warning_count)
        diag_cols_2[3].metric("Possible plates", index_report.get("records_with_possible_plate", 0))
        diag_cols_2[4].metric(
            "object_vehicle model",
            "warning" if any(item.get("model_role") == "object_vehicle" and item.get("load_status") != "success" for item in audit.get("models", [])) else "ok",
        )
        st.caption(f"Demo queries ready: {demo_report.get('queries_run', '-')}")

    st.subheader("Search")
    class_options = sorted({str(item.get("class_name", "") or "unknown") for item in records})
    verified_color_options = sorted({str(item.get("verified_vehicle_color", "") or "").strip() for item in records if item.get("verified_vehicle_color")})
    possible_color_options = sorted({str(item.get("possible_vehicle_color", "") or "").strip() for item in records if item.get("possible_vehicle_color")})
    object_type_options = sorted({str(item.get("object_type", "") or "unknown") for item in records})
    quality_options = sorted({str(item.get("quality", "") or "unknown") for item in records})
    max_time = max([float(item.get("timestamp_seconds", 0.0) or 0.0) for item in records], default=0.0)

    control_cols = st.columns(4)
    result_view_mode = control_cols[0].selectbox("Result view mode", ["Card Grid", "Timeline", "Object Journey", "Table"])
    image_size = control_cols[1].selectbox("Image size", ["Small", "Medium", "Large"], index=1)
    cluster_window = control_cols[2].slider("Group nearby matches within seconds", 1, 15, 5)
    max_cards_to_show = control_cols[3].selectbox("Max cards to show", [10, 25, 50, 100], index=1)

    query = st.text_input("Free text search", value="")
    filter_cols = st.columns(8)
    selected_class = filter_cols[0].selectbox("Class", ["All"] + class_options)
    selected_verified_color = filter_cols[1].selectbox("Verified Color", ["All"] + verified_color_options)
    include_possible_colors = filter_cols[2].checkbox("Include possible colors in search", value=True)
    selected_possible_color = filter_cols[3].selectbox("Possible Color", ["All"] + possible_color_options, disabled=not include_possible_colors)
    plate_filter = filter_cols[4].selectbox("Plate Trust", ["All", "Verified Plate Only", "Possible Plate Allowed", "No Verified Plate"])
    selected_object_type = filter_cols[5].selectbox("Object Type", ["All"] + object_type_options)
    selected_quality = filter_cols[6].selectbox("Quality", ["All"] + quality_options)
    include_possible_plates = filter_cols[7].checkbox("Include possible plates", value=False)
    time_range = st.slider("Time Range (seconds)", 0.0, float(max_time), (0.0, float(max_time)))
    show_all_objects = st.button("Show all objects")
    if include_possible_colors:
        st.caption("Color search includes possible colors. Possible color results are labelled and ranked below verified colors.")

    active_filters = _has_active_filters(
        query,
        selected_class,
        selected_verified_color,
        selected_possible_color,
        plate_filter,
        selected_object_type,
        selected_quality,
        time_range,
        max_time,
        include_possible_colors,
        include_possible_plates,
    )

    filtered_records = _filter_records(
        records,
        run_dir,
        query,
        selected_class,
        selected_verified_color,
        selected_possible_color,
        plate_filter,
        selected_object_type,
        selected_quality,
        time_range,
        include_possible_colors,
        include_possible_plates,
    ) if (active_filters or show_all_objects) else []

    if not active_filters and not show_all_objects:
        st.markdown('<div class="traffic-empty">Enter a search query or choose filters to view results.</div>', unsafe_allow_html=True)
    else:
        summary = _build_search_summary(filtered_records, query)
        clusters = _cluster_records(filtered_records, float(cluster_window))

        st.subheader("Search Overview")
        overview_cols = st.columns(8)
        with overview_cols[0]:
            _render_metric_card("Query", summary["query"])
        with overview_cols[1]:
            _render_metric_card("Total results", summary["total_matches"])
        with overview_cols[2]:
            _render_metric_card("First seen", summary["first_seen"])
        with overview_cols[3]:
            _render_metric_card("Last seen", summary["last_seen"])
        with overview_cols[4]:
            _render_metric_card("Unique tracks", summary["unique_tracks"])
        with overview_cols[5]:
            _render_metric_card("Verified color matches", summary["verified_color_matches"])
        with overview_cols[6]:
            _render_metric_card("Possible color matches", summary["possible_color_matches"])
        with overview_cols[7]:
            _render_metric_card("Verified plates found", summary["verified_plates_found"])

        with st.expander("Distribution details"):
            dist_cols = st.columns(4)
            with dist_cols[0]:
                st.dataframe(_make_distribution_df(summary["class_distribution"], "class"), use_container_width=True, hide_index=True)
            with dist_cols[1]:
                st.dataframe(_make_distribution_df(summary["verified_color_distribution"], "verified_color"), use_container_width=True, hide_index=True)
            with dist_cols[2]:
                st.dataframe(_make_distribution_df(summary["possible_color_distribution"], "possible_color"), use_container_width=True, hide_index=True)
            with dist_cols[3]:
                st.dataframe(_make_distribution_df(summary["quality_distribution"], "quality"), use_container_width=True, hide_index=True)

        st.subheader("Timeline")
        if filtered_records:
            _render_timeline(clusters, duration_seconds or max_time)
            cluster_df = pd.DataFrame(
                [
                    {
                        "cluster_id": item["cluster_id"],
                        "time_range": f"{item['start_time']}-{item['end_time']}",
                        "count": item["count"],
                        "classes": item["classes"] or "-",
                        "colors": item["colors"] or "-",
                        "plates": item["plates"] or "-",
                    }
                    for item in clusters
                ]
            )
            st.dataframe(cluster_df, use_container_width=True, hide_index=True)
        else:
            st.info("No results for the current search or filters.")

        cluster_options = ["All matches"] + [f"{item['cluster_id']} | {item['label']}" for item in clusters]
        selected_cluster_label = st.selectbox("Selected Cluster Details", cluster_options)
        selected_cluster_records = filtered_records
        if selected_cluster_label != "All matches":
            cluster_id = selected_cluster_label.split("|", 1)[0].strip()
            selected_cluster = next((item for item in clusters if item["cluster_id"] == cluster_id), None)
            selected_cluster_records = list(selected_cluster.get("records", [])) if selected_cluster else []

        if selected_cluster_records:
            ordered = sorted(selected_cluster_records, key=lambda item: float(item.get("timestamp_seconds", 0.0) or 0.0))
            best = max(ordered, key=lambda item: float(item.get("confidence", 0.0) or 0.0))
            cluster_cols = st.columns(3)
            cluster_cols[0].markdown(f"**First frame**  \n{ordered[0].get('timestamp_text', '-')}")
            cluster_cols[1].markdown(f"**Best frame**  \n{best.get('timestamp_text', '-')}")
            cluster_cols[2].markdown(f"**Last frame**  \n{ordered[-1].get('timestamp_text', '-')}")

        journey_rows = _journey_rows(filtered_records)
        selected_track_label = "None"
        if journey_rows:
            with st.expander("Object Journey"):
                journey_df = pd.DataFrame(
                    [
                        {
                            "track_id": item["track_id"],
                            "class": item["class"],
                            "verified_color": item["verified_color"],
                            "possible_color": item["possible_color"],
                            "verified_plate": item["verified_plate"],
                            "first_seen": item["first_seen"],
                            "last_seen": item["last_seen"],
                            "detections": item["number_of_detections"],
                            "quality": item["quality"],
                        }
                        for item in journey_rows
                    ]
                )
                st.dataframe(journey_df, use_container_width=True, hide_index=True)
                selected_track_label = st.selectbox(
                    "Select track",
                    ["None"] + [f"{item['track_id']} | {item['class']} | {item['first_seen']} to {item['last_seen']}" for item in journey_rows],
                )

        records_to_render = filtered_records[:max_cards_to_show]
        if selected_cluster_label != "All matches":
            records_to_render = selected_cluster_records[:max_cards_to_show]
        if selected_track_label != "None":
            selected_track_id = selected_track_label.split("|", 1)[0].strip()
            selected_track = next((item for item in journey_rows if item["track_id"] == selected_track_id), None)
            records_to_render = list(selected_track.get("records", []))[:max_cards_to_show] if selected_track else []

        generate_clip_batch = st.button("Generate event clips for shown results", disabled=not records_to_render)
        if generate_clip_batch and records_to_render:
            with st.spinner("Building event preview clips for the current result set..."):
                for record in records_to_render:
                    build_event_preview_clip(run_dir, record)
            st.rerun()

        st.subheader(f"Results ({len(records_to_render)} shown)")
        if result_view_mode == "Table":
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "class": item.get("class_name"),
                            "timestamp": item.get("timestamp_text"),
                            "verified_color": item.get("verified_vehicle_color") or "Unknown",
                            "possible_color": item.get("possible_vehicle_color") or "-",
                            "verified_plate": item.get("verified_license_plate") or "-",
                            "track_id": item.get("track_id") or "-",
                            "confidence": item.get("confidence"),
                            "quality": item.get("quality"),
                        }
                        for item in records_to_render
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        elif result_view_mode == "Object Journey" and selected_track_label == "None":
            st.info("Open Object Journey above and select a track to view its frames.")
        else:
            for record in records_to_render:
                _render_card(record, run_dir, image_size)

    with st.expander("Advanced diagnostics"):
        st.write({"missing_files": missing_files, "packaged_cards_loaded": len(cards)})

    with st.expander("Noisy / excluded detections"):
        ignored_counts = index_report.get("ignored_class_counts", {})
        if ignored_counts:
            st.dataframe(
                pd.DataFrame([{"class_name": key, "count": value} for key, value in ignored_counts.items()]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No ignored noisy classes were recorded.")
        st.caption(f"Excluded by whitelist: {index_report.get('excluded_by_whitelist_detection_count', 0)}")

    with st.expander("Debug JSON"):
        st.write(
            {
                "video_info": video_info,
                "index_report_summary": {
                    "total_object_records": index_report.get("total_object_records"),
                    "records_with_verified_plate": index_report.get("records_with_verified_plate"),
                    "records_with_verified_color": index_report.get("records_with_verified_color"),
                },
            }
        )
    with st.expander("Run directory debug"):
        cards_path = run_dir / "09B_universal_search_cards.json"
        st.write("os.environ TD_CASE2_RUN_DIR:", os.environ.get("TD_CASE2_RUN_DIR"))
        st.write("session_state run_dir:", st.session_state.get(SESSION_RUN_DIR_KEY))
        st.write("resolved_run_dir:", str(run_dir))
        st.write("video_name:", video_info.get("video_name"))
        st.write("cards path:", str(cards_path))
        st.write("cards path exists:", cards_path.exists())


if __name__ == "__main__":
    main()
