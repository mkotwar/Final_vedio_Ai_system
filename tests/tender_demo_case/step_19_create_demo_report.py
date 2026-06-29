from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _load_required_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Step 19 input file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in JSON file: {path}")
    return payload


def _load_optional_json(path: Path) -> dict[str, Any] | list[dict[str, Any]] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def make_relative_media_path(run_dir: Path, path_value: str | None) -> str | None:
    if not path_value:
        return None

    repo_root = _repo_root()
    raw_value = str(path_value).strip()
    path = Path(raw_value)
    run_marker = f"tests/tender_demo_case/debug_runs/{run_dir.name}/"

    try:
        normalized_raw = raw_value.replace("\\", "/")
        if run_marker in normalized_raw:
            return normalized_raw.split(run_marker, 1)[1]

        if path.is_absolute():
            try:
                return path.resolve().relative_to(run_dir.resolve()).as_posix()
            except ValueError:
                try:
                    return path.resolve().relative_to(repo_root.resolve()).as_posix()
                except ValueError:
                    return str(path)

        candidate_from_run_dir = run_dir / path
        if candidate_from_run_dir.exists():
            return candidate_from_run_dir.resolve().relative_to(run_dir.resolve()).as_posix()

        candidate_from_repo = repo_root / path
        if candidate_from_repo.exists():
            return candidate_from_repo.resolve().relative_to(run_dir.resolve()).as_posix()

        return path.as_posix()
    except Exception:
        return str(path_value)


def _escape(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _json_block(value: Any) -> str:
    return _escape(json.dumps(value, indent=2, ensure_ascii=True))


def _badge_class(category: str) -> str:
    mapping = {
        "priority_suspicious_event": "badge-danger",
        "possible_review_clip": "badge-warning",
        "normal_activity": "badge-success",
        "uncertain_activity": "badge-muted",
    }
    return mapping.get(str(category), "badge-muted")


def _event_export_map(export_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    items = export_manifest.get("exported_clips", []) if isinstance(export_manifest, dict) else []
    if not isinstance(items, list):
        return {}
    return {
        str(item.get("clip_id")): item
        for item in items
        if isinstance(item, dict) and str(item.get("clip_id", "")).strip()
    }


def _vlm_output_map(vlm_outputs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    items = vlm_outputs.get("items", []) if isinstance(vlm_outputs, dict) else []
    if not isinstance(items, list):
        return {}
    return {
        str(item.get("source_clip_id")): item
        for item in items
        if isinstance(item, dict) and str(item.get("source_clip_id", "")).strip()
    }


def _media_image(run_dir: Path, path_value: str | None, alt_text: str, class_name: str = "thumb") -> str:
    relative = make_relative_media_path(run_dir, path_value)
    if relative is None:
        return '<div class="media-missing">Media not found</div>'
    resolved = run_dir / relative
    if not resolved.exists():
        return f'<div class="media-missing">Media not found</div><div class="path-text">{_escape(relative)}</div>'
    return f'<img class="{class_name}" src="{_escape(relative)}" alt="{_escape(alt_text)}">'


def _media_video(run_dir: Path, path_value: str | None) -> str:
    relative = make_relative_media_path(run_dir, path_value)
    if relative is None:
        return '<div class="media-missing">Media not found</div>'
    resolved = run_dir / relative
    if not resolved.exists():
        return f'<div class="media-missing">Media not found</div><div class="path-text">{_escape(relative)}</div>'
    suffix = Path(str(path_value)).suffix.lower()
    mime_type = "video/mp4"
    if suffix == ".avi":
        mime_type = "video/x-msvideo"
    return (
        f'<video controls preload="metadata" class="video-player">'
        f'<source src="{_escape(relative)}" type="{_escape(mime_type)}">'
        "Your browser does not support the video tag."
        "</video>"
        f'<div class="path-text">{_escape(relative)}</div>'
    )


def _warnings(optional_payloads: dict[str, Any | None]) -> list[str]:
    warnings: list[str] = []
    for label, payload in optional_payloads.items():
        if payload is None:
            warnings.append(f"Optional file missing: {label}")
    return warnings


def _summary_cards(processing_summary: dict[str, Any]) -> str:
    cards = [
        ("Top-K Clips", processing_summary.get("topk_inputs", 0)),
        ("Successful Parses", processing_summary.get("successful_parses", 0)),
        ("Failed Parses", processing_summary.get("failed_parses", 0)),
        ("Priority Events", processing_summary.get("priority_suspicious_events", 0)),
        ("Review Clips", processing_summary.get("possible_review_clips", 0)),
        ("Normal Clips", processing_summary.get("normal_activity_clips", 0)),
        ("Uncertain Clips", processing_summary.get("uncertain_clips", 0)),
    ]
    return "".join(
        f'<div class="stat-card"><div class="stat-label">{_escape(label)}</div><div class="stat-value">{_escape(value)}</div></div>'
        for label, value in cards
    )


def _scene_overview_html(scene_overview: dict[str, Any]) -> str:
    if not isinstance(scene_overview, dict) or not scene_overview:
        return ""
    people_counts = scene_overview.get("people_count_observed", {})
    if not isinstance(people_counts, dict):
        people_counts = {}
    activities = scene_overview.get("common_activities", [])
    objects = scene_overview.get("common_objects", [])
    return f"""
    <section class="section">
      <h2>Scene Overview</h2>
      <div class="kv-grid">
        <div><strong>Dominant Scene Type</strong><br>{_escape(scene_overview.get('dominant_scene_type', 'unknown'))}</div>
        <div><strong>Common Activities</strong><br>{_escape(', '.join(activities) if isinstance(activities, list) and activities else 'unavailable')}</div>
        <div><strong>Common Objects</strong><br>{_escape(', '.join(objects) if isinstance(objects, list) and objects else 'unavailable')}</div>
        <div><strong>People Count Observed</strong><br>{_escape(people_counts.get('min', 0))} to {_escape(people_counts.get('max', 0))}</div>
      </div>
    </section>
    """


def _event_card(
    run_dir: Path,
    item: dict[str, Any],
    export_map: dict[str, dict[str, Any]],
    vlm_map: dict[str, dict[str, Any]],
) -> str:
    clip_id = str(item.get("clip_id", "unknown_clip"))
    export_item = export_map.get(clip_id, {})
    vlm_item = vlm_map.get(clip_id, {})
    strip_html = _media_image(run_dir, item.get("strip_path"), f"{clip_id} temporal strip", "strip-image")
    yolo_html = _media_image(run_dir, item.get("top_annotated_frame_path"), f"{clip_id} annotated YOLO frame", "thumb")
    exported_video_html = _media_video(run_dir, export_item.get("output_path")) if export_item else '<div class="media-missing">Media not found</div>'
    qwen_block = ""
    if vlm_item:
        qwen_block = (
            "<details><summary>Qwen Parsed JSON</summary>"
            f"<pre>{_json_block(vlm_item.get('parsed_json'))}</pre>"
            "</details>"
            "<details><summary>Raw Qwen Output</summary>"
            f"<pre>{_escape(vlm_item.get('raw_vlm_output', ''))}</pre>"
            "</details>"
        )

    compiled_reference = ""
    if not export_item:
        compiled_reference = "<div class='path-text'>Using compiled review video reference.</div>"

    return f"""
    <article class="event-card">
      <div class="event-header">
        <h3>{_escape(clip_id)}</h3>
        <span class="badge {_badge_class(str(item.get('final_category', '')))}">{_escape(item.get('final_category'))}</span>
      </div>
      <div class="kv-grid">
        <div><strong>Time</strong><br>{_escape(item.get('time_range'))}</div>
        <div><strong>Risk</strong><br>{_escape(item.get('risk_level'))}</div>
        <div><strong>Confidence</strong><br>{_escape(item.get('confidence'))}</div>
        <div><strong>Event Label</strong><br>{_escape(item.get('event_label'))}</div>
        <div><strong>Ranked Score</strong><br>{_escape(item.get('ranked_clip_score'))}</div>
        <div><strong>Motion Score</strong><br>{_escape(item.get('motion_score'))}</div>
        <div><strong>YOLO Person Max</strong><br>{_escape(item.get('yolo_person_max'))}</div>
        <div><strong>YOLO Top Classes</strong><br>{_escape(", ".join(item.get('yolo_top_classes', [])))}</div>
      </div>
      <p class="description">{_escape(item.get('best_event_description'))}</p>
      <div class="path-text"><strong>Selection Reasons:</strong> {_escape(", ".join(item.get('selection_reasons', [])))}</div>
      <div class="path-text"><strong>Why Selected:</strong> {_escape(item.get('why_selected'))}</div>
      <div class="path-text"><strong>Review Note:</strong> {_escape(item.get('review_note'))}</div>
      <div class="media-grid">
        <div><h4>Temporal Strip</h4>{strip_html}</div>
        <div><h4>Annotated YOLO Frame</h4>{yolo_html}</div>
        <div><h4>Exported Clip Video</h4>{exported_video_html}{compiled_reference}</div>
      </div>
      {qwen_block}
    </article>
    """


def _compact_cards(
    run_dir: Path,
    items: list[dict[str, Any]],
    export_map: dict[str, dict[str, Any]],
    vlm_map: dict[str, dict[str, Any]],
    include_normal: bool = False,
) -> str:
    rendered: list[str] = []
    for item in items:
        clip_id = str(item.get("clip_id", "unknown_clip"))
        vlm_item = vlm_map.get(clip_id, {})
        qwen_block = ""
        if vlm_item:
            qwen_block = (
                "<details><summary>Qwen Output</summary>"
                f"<pre>{_json_block(vlm_item.get('parsed_json'))}</pre>"
                "</details>"
            )
        strip_html = _media_image(run_dir, item.get("strip_path"), f"{clip_id} strip")
        yolo_html = _media_image(run_dir, item.get("top_annotated_frame_path"), f"{clip_id} yolo frame")
        video_html = ""
        export_item = export_map.get(clip_id, {})
        if export_item:
            video_html = f'<div class="path-text">{_escape(make_relative_media_path(run_dir, export_item.get("output_path")) or export_item.get("output_path"))}</div>'

        extra = ""
        if not include_normal:
            extra = (
                f"<div><strong>Motion</strong>: {_escape(item.get('motion_score'))}</div>"
                f"<div><strong>YOLO People</strong>: {_escape(item.get('yolo_person_max'))}</div>"
                f"<div><strong>YOLO Classes</strong>: {_escape(', '.join(item.get('yolo_top_classes', [])))}</div>"
            )

        rendered.append(
            f"""
            <article class="compact-card">
              <div class="event-header">
                <h4>{_escape(clip_id)}</h4>
                <span class="badge {_badge_class(str(item.get('final_category', '')))}">{_escape(item.get('final_category'))}</span>
              </div>
              <div><strong>Time</strong>: {_escape(item.get('time_range'))}</div>
              <div><strong>Event Label</strong>: {_escape(item.get('event_label'))}</div>
              <div><strong>Risk</strong>: {_escape(item.get('risk_level'))}</div>
              <div><strong>Confidence</strong>: {_escape(item.get('confidence'))}</div>
              <div><strong>Description</strong>: {_escape(item.get('best_event_description'))}</div>
              {extra}
              <div class="thumb-row">
                {strip_html}
                {yolo_html if not include_normal else ""}
              </div>
              {video_html}
              {qwen_block}
            </article>
            """
        )
    return "".join(rendered) or '<div class="media-missing">No clips available.</div>'


def _timeline_rows(items: list[dict[str, Any]]) -> str:
    rows = []
    for item in sorted(items, key=lambda entry: float(entry.get("start_time", 0.0) or 0.0)):
        rows.append(
            f"""
            <tr>
              <td>{_escape(item.get('time_range'))}</td>
              <td>{_escape(item.get('clip_id'))}</td>
              <td><span class="badge {_badge_class(str(item.get('final_category', '')))}">{_escape(item.get('final_category'))}</span></td>
              <td>{_escape(item.get('event_label'))}</td>
              <td>{_escape(item.get('risk_level'))}</td>
              <td>{_escape(item.get('confidence'))}</td>
              <td>{_escape(item.get('best_event_description'))}</td>
            </tr>
            """
        )
    return "".join(rows)


def _evidence_paths(run_dir: Path) -> list[tuple[str, str]]:
    candidates = [
        ("13_ranked_clips.json", run_dir / "13_ranked_clips.json"),
        ("14_selected_top_clips.json", run_dir / "14_selected_top_clips.json"),
        ("15_topk_vlm_inputs.json", run_dir / "15_topk_vlm_inputs.json"),
        ("16_topk_vlm_outputs.json", run_dir / "16_topk_vlm_outputs.json"),
        ("17_topk_final_summary.json", run_dir / "17_topk_final_summary.json"),
        ("17_topk_final_summary.md", run_dir / "17_topk_final_summary.md"),
        ("18_exported_clips.json", run_dir / "18_exported_clips.json"),
        ("18_compiled_review_video.json", run_dir / "18_compiled_review_video.json"),
        ("19_demo_report.html", run_dir / "19_demo_report.html"),
    ]
    return [(label, path.resolve().relative_to(run_dir.resolve()).as_posix() if path.exists() else path.name) for label, path in candidates]


def create_demo_report_html(run_dir: Path) -> dict[str, Any]:
    print("[tender-demo] Starting Step 19")

    final_summary = _load_required_json(run_dir / "17_topk_final_summary.json")
    vlm_outputs = _load_optional_json(run_dir / "16_topk_vlm_outputs.json")
    export_manifest = _load_optional_json(run_dir / "18_exported_clips.json")
    compiled_manifest = _load_optional_json(run_dir / "18_compiled_review_video.json")
    if compiled_manifest is None:
        compiled_manifest = _load_optional_json(run_dir / "18_exported_clips" / "18_compiled_review_video.json")
    selected_top_clips = _load_optional_json(run_dir / "14_selected_top_clips.json")
    ranked_report = _load_optional_json(run_dir / "13_ranked_clips_report.json")
    video_info = _load_optional_json(run_dir / "01_video_info.json")

    optional_payloads = {
        "16_topk_vlm_outputs.json": vlm_outputs,
        "18_exported_clips.json": export_manifest,
        "18_compiled_review_video.json": compiled_manifest,
        "14_selected_top_clips.json": selected_top_clips,
        "13_ranked_clips_report.json": ranked_report,
        "01_video_info.json": video_info,
    }
    warnings = _warnings(optional_payloads)

    if not isinstance(video_info, dict):
        video_info = final_summary.get("video_info", {}) if isinstance(final_summary.get("video_info"), dict) else {}

    processing_summary = final_summary.get("processing_summary", {})
    if not isinstance(processing_summary, dict):
        processing_summary = {}

    priority_items = final_summary.get("priority_suspicious_events", [])
    review_items = final_summary.get("possible_review_clips", [])
    normal_items = final_summary.get("normal_activity_clips", [])
    timeline_items = final_summary.get("event_timeline", [])
    scene_overview = final_summary.get("scene_overview", {})

    if not isinstance(priority_items, list):
        priority_items = []
    if not isinstance(review_items, list):
        review_items = []
    if not isinstance(normal_items, list):
        normal_items = []
    if not isinstance(timeline_items, list):
        timeline_items = []

    export_map = _event_export_map(export_manifest if isinstance(export_manifest, dict) else {})
    vlm_map = _vlm_output_map(vlm_outputs if isinstance(vlm_outputs, dict) else {})

    compiled_video_path = None
    compiled_video_warning = ""
    compiled_video_metadata_html = ""
    if isinstance(export_manifest, dict):
        compiled_video_info = export_manifest.get("compiled_review_video", {})
        if isinstance(compiled_video_info, dict):
            compiled_video_path = compiled_video_info.get("playback_recommended_file") or compiled_video_info.get("output_path")
            verification = compiled_video_info.get("playback_recommended_verification") or compiled_video_info.get("video_verification", {})
            compiled_video_metadata_html = (
                f"<div class='kv-grid'>"
                f"<div><strong>Backend</strong><br>{_escape(compiled_video_info.get('backend'))}</div>"
                f"<div><strong>Playable</strong><br>{_escape('yes' if compiled_video_info.get('playable') else 'no')}</div>"
                f"<div><strong>FPS</strong><br>{_escape(verification.get('fps', 'unknown'))}</div>"
                f"<div><strong>Total Frames</strong><br>{_escape(verification.get('frame_count', 'unknown'))}</div>"
                f"<div><strong>Estimated Duration</strong><br>{_escape(compiled_video_info.get('total_frames_written', 0))} frames</div>"
                f"<div><strong>Verification</strong><br>{_escape('readable' if verification.get('readable_by_opencv') else 'not readable')}</div>"
                f"</div>"
                f"<div class='path-text'><strong>File Path:</strong> {_escape(compiled_video_path)}</div>"
            )
            if not compiled_video_info.get("playable"):
                compiled_video_warning = "Compiled review video exists but is marked not playable."
    if compiled_video_path is None and isinstance(compiled_manifest, dict):
        compiled_video_path = compiled_manifest.get("playback_recommended_file") or compiled_manifest.get("compiled_video_path")
        verification = compiled_manifest.get("playback_recommended_verification") or compiled_manifest.get("video_verification", {})
        compiled_video_metadata_html = (
            f"<div class='kv-grid'>"
            f"<div><strong>Backend</strong><br>{_escape(compiled_manifest.get('compiled_video_backend'))}</div>"
            f"<div><strong>FFmpeg Available</strong><br>{_escape('yes' if compiled_manifest.get('ffmpeg_available') else 'no')}</div>"
            f"<div><strong>FPS</strong><br>{_escape(compiled_manifest.get('fps', 'unknown'))}</div>"
            f"<div><strong>Total Frames</strong><br>{_escape(verification.get('frame_count', 'unknown'))}</div>"
            f"<div><strong>Estimated Duration</strong><br>{_escape(compiled_manifest.get('duration_seconds_estimated', 'unknown'))} seconds</div>"
            f"<div><strong>Verification</strong><br>{_escape('readable' if verification.get('readable_by_opencv') else 'not readable')}</div>"
            f"</div>"
            f"<div class='path-text'><strong>File Path:</strong> {_escape(compiled_video_path)}</div>"
        )
        if not verification.get("readable_by_opencv"):
            compiled_video_warning = "Compiled review video is missing or unreadable."

    compiled_video_html = (
        _media_video(run_dir, compiled_video_path)
        if compiled_video_path
        else '<div class="warning">Compiled review video is missing.</div>'
    )
    if compiled_video_warning:
        compiled_video_html = f'<div class="warning">{_escape(compiled_video_warning)}</div>' + compiled_video_html
    if compiled_video_metadata_html:
        compiled_video_html += compiled_video_metadata_html

    warning_html = "".join(f'<div class="warning">{_escape(message)}</div>' for message in warnings)
    evidence_items_html = "".join(
        f"<li><strong>{_escape(label)}</strong>: {_escape(path_value)}</li>"
        for label, path_value in _evidence_paths(run_dir)
    )

    html_output = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tender Demo Video Analysis Report</title>
  <style>
    :root {{
      --bg: #0f1720;
      --surface: #162231;
      --surface-2: #1d2d40;
      --text: #e8eef5;
      --muted: #9bb0c5;
      --accent: #5bc0eb;
      --danger: #ff6b6b;
      --warning: #f7b267;
      --success: #7bd389;
      --gray: #7f8c9a;
      --border: #29415a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Tahoma, sans-serif;
      background: linear-gradient(180deg, #0b1118, #122030 35%, #0f1720 100%);
      color: var(--text);
      line-height: 1.45;
    }}
    .container {{
      width: min(1400px, calc(100% - 32px));
      margin: 0 auto;
      padding: 24px 0 48px;
    }}
    header {{
      background: linear-gradient(135deg, #101926, #1b3147);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 28px;
      margin-bottom: 24px;
      box-shadow: 0 18px 45px rgba(0, 0, 0, 0.25);
    }}
    h1, h2, h3, h4 {{ margin-top: 0; }}
    .subtitle, .path-text {{ color: var(--muted); word-break: break-all; }}
    .section {{
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 20px;
      margin-bottom: 20px;
      color: #102030;
    }}
    .stats-grid, .kv-grid, .media-grid, .card-grid {{
      display: grid;
      gap: 14px;
    }}
    .stats-grid {{
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    }}
    .stat-card, .compact-card, .event-card {{
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 16px;
      color: #102030;
    }}
    .stat-label {{
      color: #516679;
      font-size: 0.9rem;
    }}
    .stat-value {{
      font-size: 1.8rem;
      font-weight: 700;
      margin-top: 8px;
    }}
    .event-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
    }}
    .badge {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 0.78rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .badge-danger {{ background: rgba(255, 107, 107, 0.18); color: #ffd2d2; }}
    .badge-warning {{ background: rgba(247, 178, 103, 0.18); color: #ffe1bf; }}
    .badge-success {{ background: rgba(123, 211, 137, 0.18); color: #d9ffe0; }}
    .badge-muted {{ background: rgba(127, 140, 154, 0.2); color: #dde5ec; }}
    .kv-grid {{
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      margin: 12px 0;
    }}
    .media-grid {{
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      margin-top: 12px;
    }}
    .thumb-row {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin-top: 10px;
    }}
    .thumb, .strip-image, .video-player {{
      width: 100%;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: #091119;
    }}
    .strip-image {{ max-height: 220px; object-fit: cover; }}
    .video-player {{ max-height: 520px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      border-radius: 12px;
    }}
    th, td {{
      border-bottom: 1px solid var(--border);
      padding: 12px 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      background: rgba(255, 255, 255, 0.02);
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #f2f6fa;
      border: 1px solid #d6e1ea;
      border-radius: 12px;
      padding: 12px;
      overflow-x: auto;
      color: #102030;
    }}
    details {{
      margin-top: 12px;
      background: #f8fbfd;
      border: 1px solid #d6e1ea;
      border-radius: 12px;
      padding: 10px 12px;
    }}
    .warning, .media-missing {{
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(247, 178, 103, 0.18);
      color: #ffe1bf;
      border: 1px solid rgba(247, 178, 103, 0.35);
      margin-bottom: 10px;
    }}
    .description {{
      font-size: 1rem;
      color: #f1f6fb;
    }}
    @media (max-width: 860px) {{
      .container {{ width: min(100% - 20px, 1400px); }}
      header, .section {{ padding: 16px; }}
      .event-header {{ flex-direction: column; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Tender Demo Video Analysis Report</h1>
      <p class="subtitle">Optimized Top-K + Safety Guardrails</p>
      <div class="kv-grid">
        <div><strong>Video</strong><br>{_escape(video_info.get('video_name', 'unknown'))}</div>
        <div><strong>Duration</strong><br>{_escape(video_info.get('duration_seconds', 'unknown'))} seconds</div>
        <div><strong>Resolution</strong><br>{_escape(video_info.get('resolution', 'unknown'))}</div>
        <div><strong>Run Folder</strong><br>{_escape(run_dir.name)}</div>
        <div><strong>Generated Files Location</strong><br>{_escape(run_dir.as_posix())}</div>
      </div>
    </header>

    <section class="section">
      <h2>Executive Summary</h2>
      <p>{_escape(final_summary.get('descriptive_summary') or final_summary.get('final_summary_text') or 'Summary not available.')}</p>
      {warning_html}
    </section>

    {_scene_overview_html(scene_overview)}

    <section class="section">
      <h2>Processing Summary</h2>
      <div class="stats-grid">
        {_summary_cards(processing_summary)}
      </div>
    </section>

    <section class="section">
      <h2>Compiled Review Video</h2>
      {compiled_video_html}
    </section>

    <section class="section">
      <h2>Priority Suspicious Events</h2>
      {''.join(_event_card(run_dir, item, export_map, vlm_map) for item in priority_items) or '<div class="media-missing">No priority suspicious events found.</div>'}
    </section>

    <section class="section">
      <h2>Possible Review Clips</h2>
      <div class="card-grid">
        {_compact_cards(run_dir, review_items, export_map, vlm_map)}
      </div>
    </section>

    <section class="section">
      <h2>Normal Activity Clips</h2>
      <div class="card-grid">
        {_compact_cards(run_dir, normal_items, export_map, vlm_map, include_normal=True)}
      </div>
    </section>

    <section class="section">
      <h2>Event Timeline</h2>
      <table>
        <thead>
          <tr>
            <th>Time</th>
            <th>Clip</th>
            <th>Category</th>
            <th>Event Label</th>
            <th>Risk</th>
            <th>Confidence</th>
            <th>Description</th>
          </tr>
        </thead>
        <tbody>
          {_timeline_rows(timeline_items)}
        </tbody>
      </table>
    </section>

    <section class="section">
      <h2>Evidence Files</h2>
      <ul>
        {evidence_items_html}
      </ul>
    </section>
  </div>
</body>
</html>
"""

    output_path = run_dir / "19_demo_report.html"
    output_path.write_text(html_output, encoding="utf-8")

    result = {
        "html_report_path": str(output_path),
        "priority_events": len(priority_items),
        "possible_review_clips": len(review_items),
        "normal_clips": len(normal_items),
        "compiled_video_available": compiled_video_path is not None,
    }

    print(f"[tender-demo] HTML report path: {output_path}")
    print(f"[tender-demo] Priority events count: {len(priority_items)}")
    print(f"[tender-demo] Possible review clips count: {len(review_items)}")
    print(f"[tender-demo] Normal clips count: {len(normal_items)}")
    print(f"[tender-demo] Compiled video available: {'yes' if compiled_video_path else 'no'}")
    return result
