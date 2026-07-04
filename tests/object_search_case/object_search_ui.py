from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import streamlit as st


def _case_root() -> Path:
    return Path(__file__).resolve().parent


def _debug_runs_root() -> Path:
    return _case_root() / "debug_runs"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_index(run_dir: Path) -> list[dict[str, Any]]:
    index_path = run_dir / "06_searchable_object_index.json"
    if not index_path.exists():
        return []
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    items = payload.get("items", []) if isinstance(payload, dict) else []
    return items if isinstance(items, list) else []


def _load_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "07_summary.json"
    if not summary_path.exists():
        return {}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_video_info(run_dir: Path) -> dict[str, Any]:
    video_info_path = run_dir / "01_video_info.json"
    if not video_info_path.exists():
        return {}
    payload = json.loads(video_info_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _resolve_repo_relative(path_value: Any) -> Path | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        return path if path.exists() else None
    candidate = _repo_root() / path
    return candidate if candidate.exists() else None


def _format_seconds(value: Any) -> str:
    try:
        total = max(0.0, float(value))
    except (TypeError, ValueError):
        return "unknown"
    minutes = int(total // 60)
    seconds = total - (minutes * 60)
    if abs(seconds - round(seconds)) < 1e-6:
        return f"{minutes:02d}:{int(round(seconds)):02d}"
    return f"{minutes:02d}:{seconds:04.1f}"


def _score_item(item: dict[str, Any], query_terms: list[str], class_name: str, start_seconds: float | None, end_seconds: float | None) -> int | None:
    if class_name and str(item.get("class_name", "")).strip().lower() != class_name:
        return None
    start_time = float(item.get("start_time", 0.0) or 0.0)
    end_time_value = float(item.get("end_time", 0.0) or 0.0)
    if start_seconds is not None and end_time_value < start_seconds:
        return None
    if end_seconds is not None and start_time > end_seconds:
        return None

    appearance_terms = [str(value).lower() for value in item.get("appearance_terms", [])]
    search_text = str(item.get("search_text", "")).lower()
    score = 0
    for term in query_terms:
        if term in search_text:
            score += 5
        elif any(term in value for value in appearance_terms):
            score += 3
    if query_terms and score == 0:
        return None
    score += int(item.get("frame_hit_count", 0) or 0)
    return score


def _search_items(
    items: list[dict[str, Any]],
    *,
    query: str,
    class_name: str,
    start_seconds: float | None,
    end_seconds: float | None,
) -> list[dict[str, Any]]:
    query_terms = [term for term in re.split(r"\s+", query.strip().lower()) if term]
    ranked: list[tuple[int, dict[str, Any]]] = []
    for item in items:
        score = _score_item(item, query_terms, class_name, start_seconds, end_seconds)
        if score is None:
            continue
        enriched = dict(item)
        enriched["search_score"] = score
        ranked.append((score, enriched))
    ranked.sort(key=lambda pair: (-pair[0], float(pair[1].get("start_time", 0.0) or 0.0), str(pair[1].get("object_id", ""))))
    return [item for _score, item in ranked]


def _available_runs() -> list[Path]:
    root = _debug_runs_root()
    if not root.exists():
        return []
    return sorted([path for path in root.iterdir() if path.is_dir()], key=lambda path: path.stat().st_mtime, reverse=True)


def _class_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        class_name = str(item.get("class_name", "")).strip().lower() or "unknown"
        counts[class_name] = counts.get(class_name, 0) + 1
    return dict(sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])))


def _pick_best_evidence_hits(frame_hits: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    ranked = sorted(
        frame_hits,
        key=lambda hit: (
            float(hit.get("confidence", 0.0) or 0.0),
            len(hit.get("license_plates", []) or []),
            float(hit.get("timestamp_seconds", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return ranked[:limit]


def _render_overview(summary: dict[str, Any], video_info: dict[str, Any], items: list[dict[str, Any]]) -> None:
    tracked_by_class = summary.get("tracked_by_class") if isinstance(summary.get("tracked_by_class"), dict) else _class_counts(items)
    st.subheader("Run Overview")
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Tracked objects", int(summary.get("tracked_object_count", len(items)) or len(items)))
    with metric_cols[1]:
        st.metric("Sampled frames", int(summary.get("sampled_frame_count", video_info.get("sampled_frame_count", 0)) or 0))
    with metric_cols[2]:
        st.metric("Duration", _format_seconds(video_info.get("duration_seconds", summary.get("duration_seconds"))))
    with metric_cols[3]:
        st.metric("Classes found", len(tracked_by_class))
    if summary.get("tracks_with_license_plate") is not None:
        st.caption(f"Tracks with readable plate text: {summary.get('tracks_with_license_plate')}")

    detail_cols = st.columns([1, 1])
    with detail_cols[0]:
        st.write("Tracked classes")
        if tracked_by_class:
            st.write(tracked_by_class)
        else:
            st.info("No tracked classes were found in this run.")
    with detail_cols[1]:
        st.write("Try these searches")
        suggestions = [
            "person",
            "car",
            "motorcycle",
            "truck",
            "pink upper clothing",
            "person with bag",
        ]
        st.write(", ".join(suggestions))


def _render_frame_hit_gallery(frame_hits: list[dict[str, Any]], *, key_prefix: str) -> None:
    preview_hits = _pick_best_evidence_hits(frame_hits, limit=3)
    if not preview_hits:
        return
    st.caption("Best evidence images")
    columns = st.columns(len(preview_hits))
    for index, hit in enumerate(preview_hits):
        frame_path = _resolve_repo_relative(hit.get("frame_path"))
        crop_path = _resolve_repo_relative(hit.get("crop_path"))
        plate_crop_path = _resolve_repo_relative(hit.get("plate_crop_path"))
        with columns[index]:
            st.caption(f"{_format_seconds(hit.get('timestamp_seconds'))} | {hit.get('frame_id', 'unknown')}")
            if frame_path is not None:
                st.image(str(frame_path), use_container_width=True)
            else:
                st.write("Frame unavailable")
            if crop_path is not None:
                st.image(str(crop_path), use_container_width=True)
            else:
                st.write("Crop unavailable")
            if plate_crop_path is not None:
                st.caption("Plate crop")
                st.image(str(plate_crop_path), use_container_width=True)
            license_plates = hit.get("license_plates", []) or []
            if license_plates:
                st.caption(f"Plate: {', '.join(str(value) for value in license_plates)}")
            st.caption(f"Confidence: {hit.get('confidence', 'n/a')}")


def _render_result(item: dict[str, Any]) -> None:
    st.markdown(f"### {item.get('object_id', 'unknown')} | {item.get('class_name', 'unknown')}")
    st.write(
        {
            "time_range": f"{item.get('start_time_text', 'unknown')} - {item.get('end_time_text', 'unknown')}",
            "duration_seconds": item.get("duration_seconds"),
            "best_timestamp": item.get("best_timestamp_text"),
            "frame_hits": item.get("frame_hit_count"),
            "search_score": item.get("search_score", 0),
            "best_license_plate": item.get("best_license_plate") or "n/a",
            "plate_ocr_status": item.get("plate_ocr_status", "n/a"),
        }
    )
    st.write(f"Appearance terms: {', '.join(item.get('appearance_terms', [])) or 'n/a'}")
    if item.get("license_plates"):
        st.write(f"Detected license plates: {', '.join(item.get('license_plates', []))}")
    elif item.get("plate_detected_text"):
        st.write(f"OCR text near plate region: {', '.join(item.get('plate_detected_text', []))}")

    frame_path = _resolve_repo_relative(item.get("best_frame_path"))
    crop_path = _resolve_repo_relative(item.get("best_crop_path"))
    plate_crop_path = _resolve_repo_relative(item.get("best_license_plate_crop_path"))
    cols = st.columns(3 if plate_crop_path is not None else 2)
    with cols[0]:
        st.caption("Best frame")
        if frame_path is not None:
            st.image(str(frame_path), use_container_width=True)
        else:
            st.write("Frame unavailable")
    with cols[1]:
        st.caption("Best crop")
        if crop_path is not None:
            st.image(str(crop_path), use_container_width=True)
        else:
            st.write("Crop unavailable")
    if plate_crop_path is not None:
        with cols[2]:
            st.caption("Best plate crop")
            st.image(str(plate_crop_path), use_container_width=True)
            plate_text = item.get("best_license_plate") or "unreadable"
            st.caption(f"Plate text: {plate_text}")

    _render_frame_hit_gallery(item.get("frame_hits", []), key_prefix=str(item.get("object_id", "result")))
    hit_rows = [
        {
            "timestamp": hit.get("timestamp_seconds"),
            "frame_id": hit.get("frame_id"),
            "confidence": hit.get("confidence"),
            "plate_text": ", ".join(hit.get("license_plates", []) or []),
            "plate_status": hit.get("plate_ocr_status"),
        }
        for hit in item.get("frame_hits", [])[:20]
    ]
    if hit_rows:
        with st.expander("Show frame-hit metadata", expanded=False):
            st.dataframe(hit_rows, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Object Search Case UI", layout="wide")
    st.title("Object Search Case UI")
    st.caption("Isolated YOLO + tracking + searchable object index viewer")

    runs = _available_runs()
    if not runs:
        st.warning("No object-search runs found yet. Run the testcase first.")
        return

    run_labels = [run.name for run in runs]
    default_run = runs[0]

    with st.sidebar:
        st.header("Run Selection")
        selected_label = st.selectbox("Run", run_labels, index=0)
        selected_run = next((run for run in runs if run.name == selected_label), default_run)
        st.caption(str(selected_run))

        summary = _load_summary(selected_run)
        video_info = _load_video_info(selected_run)
        st.header("Search")
        query = st.text_input("Query", placeholder="person bag, pink upper clothing, car, motorcycle, truck")
        items = _load_index(selected_run)
        class_options = ["All"] + sorted({str(item.get("class_name", "")).strip().lower() for item in items if str(item.get("class_name", "")).strip()})
        class_name = st.selectbox("Class name", class_options, index=0)
        time_start = st.number_input("Start seconds", min_value=0.0, value=0.0, step=1.0)
        time_end = st.number_input("End seconds", min_value=0.0, value=0.0, step=1.0)
        limit = st.slider("Max results", min_value=5, max_value=100, value=20, step=5)

        st.header("Summary")
        if summary:
            st.write(summary)

    items = _load_index(selected_run)
    _render_overview(summary, video_info, items)

    video_path = _resolve_repo_relative(video_info.get("video_path"))
    if video_info:
        with st.expander("Original Video", expanded=False):
            overview_cols = st.columns([2, 1])
            with overview_cols[0]:
                if video_path is not None:
                    st.video(str(video_path))
                else:
                    st.warning(f"Original video not found: {video_info.get('video_path', 'unknown')}")
            with overview_cols[1]:
                st.write(
                    {
                        "video_name": video_info.get("video_name"),
                        "duration": _format_seconds(video_info.get("duration_seconds")),
                        "fps": video_info.get("fps"),
                        "resolution": f"{video_info.get('width', 'unknown')}x{video_info.get('height', 'unknown')}",
                        "sample_interval_seconds": video_info.get("sample_every_seconds"),
                        "sampled_frames": video_info.get("sampled_frame_count"),
                    }
                )
                if video_path is not None:
                    st.caption(str(video_path))

    results = _search_items(
        items,
        query=query,
        class_name="" if class_name == "All" else class_name,
        start_seconds=time_start if time_start > 0 else None,
        end_seconds=time_end if time_end > 0 else None,
    )

    st.subheader("Search Results")
    st.write(f"Found {len(results)} result(s)")
    if not results:
        st.info("No results matched the current filters.")
        return

    quick_preview = [
        {
            "object_id": item.get("object_id"),
            "class_name": item.get("class_name"),
            "time_range": f"{item.get('start_time_text', 'unknown')} - {item.get('end_time_text', 'unknown')}",
            "best_timestamp": item.get("best_timestamp_text"),
            "frame_hits": item.get("frame_hit_count"),
            "license_plate": item.get("best_license_plate") or "",
            "appearance_terms": ", ".join(item.get("appearance_terms", [])[:4]),
        }
        for item in results[: min(limit, 10)]
    ]
    st.dataframe(quick_preview, use_container_width=True)

    for item in results[:limit]:
        _render_result(item)
        st.divider()


if __name__ == "__main__":
    main()
