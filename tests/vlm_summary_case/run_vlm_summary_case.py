from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


SCHEMA_NOISE_TERMS = (
    "scene_type",
    "caption",
    "json schema",
    "current panel",
    "short factual sentence",
    "return only",
    "short sentence",
    "people|",
    "shop|office|road",
    "street|parking|warehouse",
    "risk_level",
    "event_label",
)

GENERIC_NOISE_TERMS = (
    "there is no indication of suspicious activity",
    "no indication of suspicious activity",
    "no suspicious activity",
    "no usable vlm text was returned",
    "selected clip contains visually important activity",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _debug_root() -> Path:
    return Path(__file__).resolve().parent / "debug_runs"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_seconds(seconds: float | int | None) -> str:
    if seconds is None:
        return "unknown"
    total_seconds = float(seconds)
    if total_seconds < 0:
        return "unknown"
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    remaining = total_seconds - (hours * 3600) - (minutes * 60)
    if float(remaining).is_integer():
        return f"{hours:02d}:{minutes:02d}:{int(remaining):02d}"
    return f"{hours:02d}:{minutes:02d}:{remaining:04.1f}"


def _clean_text(text: Any) -> str:
    cleaned = str(text or "")
    cleaned = re.sub(r"```(?:json)?", " ", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([,.:;!?])", r"\1", cleaned)
    return cleaned


def _clean_phrase(text: Any) -> str:
    cleaned = _clean_text(text).strip().rstrip(".")
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    if any(term in lowered for term in SCHEMA_NOISE_TERMS):
        return ""
    if any(term in lowered for term in GENERIC_NOISE_TERMS):
        return ""
    if "{" in cleaned or "}" in cleaned or '": "' in cleaned:
        return ""
    if cleaned.count("|") >= 2:
        return ""
    if cleaned.startswith("[clip_") or ", rank_" in cleaned.lower() or "strategy:" in cleaned.lower():
        return ""
    if len(cleaned) > 180:
        return ""
    return cleaned


def _dedupe_preserve_order(values: list[str], limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = _clean_phrase(value)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
        if limit is not None and len(result) >= limit:
            break
    return result


def _join_terms(values: list[str], limit: int = 3) -> str:
    items = _dedupe_preserve_order(values, limit=limit)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _resolve_run_inputs(args: argparse.Namespace) -> tuple[Path | None, Path, Path | None, Path | None]:
    run_dir: Path | None = None
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser()
        if not run_dir.is_absolute():
            run_dir = (_repo_root() / run_dir).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Run dir not found: {run_dir}")
        vlm_outputs_path = run_dir / "16_topk_vlm_outputs.json"
        step15_path = run_dir / "15_topk_vlm_inputs.json"
        video_info_path = run_dir / "01_video_info.json"
    else:
        vlm_outputs_path = Path(args.vlm_outputs).expanduser()
        if not vlm_outputs_path.is_absolute():
            vlm_outputs_path = (_repo_root() / vlm_outputs_path).resolve()
        if not vlm_outputs_path.exists():
            raise FileNotFoundError(f"VLM outputs file not found: {vlm_outputs_path}")
        run_dir = vlm_outputs_path.parent if vlm_outputs_path.parent.exists() else None
        step15_path = run_dir / "15_topk_vlm_inputs.json" if run_dir else None
        video_info_path = run_dir / "01_video_info.json" if run_dir else None
    if not vlm_outputs_path.exists():
        raise FileNotFoundError(f"Missing Step 16 outputs: {vlm_outputs_path}")
    return run_dir, vlm_outputs_path, step15_path if step15_path and step15_path.exists() else None, video_info_path if video_info_path and video_info_path.exists() else None


def _extract_summary_sentence(item: dict[str, Any]) -> str:
    parsed_json = item.get("parsed_json", {}) if isinstance(item.get("parsed_json"), dict) else {}
    candidates = [
        parsed_json.get("description"),
        parsed_json.get("caption"),
        item.get("raw_vlm_output"),
    ]
    for value in candidates:
        cleaned = _clean_phrase(value)
        if cleaned:
            return cleaned
    return "No clear VLM description available."


def _extract_activity_terms(item: dict[str, Any]) -> list[str]:
    parsed_json = item.get("parsed_json", {}) if isinstance(item.get("parsed_json"), dict) else {}
    values: list[str] = []
    for key in ("main_activities", "main_interactions"):
        value = parsed_json.get(key, [])
        if isinstance(value, list):
            values.extend(str(entry).replace("_", " ") for entry in value if _clean_phrase(str(entry).replace("_", " ")))
    sentence = _extract_summary_sentence(item)
    if sentence:
        values.append(sentence)
    return _dedupe_preserve_order(values, limit=6)


def _extract_object_terms(item: dict[str, Any]) -> list[str]:
    parsed_json = item.get("parsed_json", {}) if isinstance(item.get("parsed_json"), dict) else {}
    values: list[str] = []
    if isinstance(parsed_json.get("main_objects"), list):
        values.extend(str(entry).replace("_", " ") for entry in parsed_json.get("main_objects", []) if _clean_phrase(str(entry).replace("_", " ")))
    yolo = item.get("yolo", {})
    if isinstance(yolo, dict) and isinstance(yolo.get("top_classes"), list):
        values.extend(
            str(entry.get("class_name", "")).replace("_", " ")
            for entry in yolo.get("top_classes", [])
            if isinstance(entry, dict) and _clean_phrase(str(entry.get("class_name", "")).replace("_", " "))
        )
    return _dedupe_preserve_order(values, limit=8)


def _build_scene_overview(items: list[dict[str, Any]]) -> dict[str, Any]:
    scene_counter: Counter[str] = Counter()
    activities: list[str] = []
    objects: list[str] = []
    people_counts: list[int] = []
    for item in items:
        parsed_json = item.get("parsed_json", {}) if isinstance(item.get("parsed_json"), dict) else {}
        scene_type = _clean_phrase(parsed_json.get("scene_type")).lower()
        if scene_type and scene_type != "unknown":
            scene_counter[scene_type] += 1
        activities.extend(_extract_activity_terms(item))
        objects.extend(_extract_object_terms(item))
        people_counts.append(
            max(
                _safe_int(parsed_json.get("people_count"), 0),
                _safe_int(item.get("yolo", {}).get("person_max") if isinstance(item.get("yolo"), dict) else 0, 0),
            )
        )
    dominant_scene_type = scene_counter.most_common(1)[0][0] if scene_counter else "unknown"
    return {
        "dominant_scene_type": dominant_scene_type,
        "common_activities": _dedupe_preserve_order(activities, limit=6),
        "common_objects": _dedupe_preserve_order(objects, limit=6),
        "people_count_observed": {
            "min": min(people_counts) if people_counts else 0,
            "max": max(people_counts) if people_counts else 0,
        },
    }


def _build_timeline(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for item in items:
        parsed_json = item.get("parsed_json", {}) if isinstance(item.get("parsed_json"), dict) else {}
        start_time = _safe_float(item.get("start_time"), 0.0)
        end_time = _safe_float(item.get("end_time"), start_time)
        timeline.append(
            {
                "clip_id": item.get("source_clip_id") or item.get("clip_id") or item.get("topk_vlm_input_id"),
                "time_range": f"{_format_seconds(start_time)} - {_format_seconds(end_time)}",
                "scene_type": _clean_phrase(parsed_json.get("scene_type")) or "unknown",
                "event_label": _clean_phrase(parsed_json.get("event_label") or parsed_json.get("primary_event_label")) or "unknown",
                "summary_sentence": _extract_summary_sentence(item),
                "main_activities": _extract_activity_terms(item),
                "main_objects": _extract_object_terms(item),
            }
        )
    timeline.sort(key=lambda entry: entry["time_range"])
    return timeline


def _build_descriptive_summary(scene_overview: dict[str, Any], timeline: list[dict[str, Any]], video_info: dict[str, Any]) -> str:
    video_name = str(video_info.get("video_name", "")).strip() or "the video"
    scene_type = str(scene_overview.get("dominant_scene_type", "unknown")).strip() or "unknown"
    activities = scene_overview.get("common_activities", []) if isinstance(scene_overview.get("common_activities"), list) else []
    objects = scene_overview.get("common_objects", []) if isinstance(scene_overview.get("common_objects"), list) else []
    people_counts = scene_overview.get("people_count_observed", {}) if isinstance(scene_overview.get("people_count_observed"), dict) else {}
    activity_text = _join_terms(activities, limit=3)
    object_text = _join_terms(objects, limit=4)
    max_people = _safe_int(people_counts.get("max"), 0)

    parts = [f"VLM-first summary for {video_name}."]
    if scene_type != "unknown":
        parts.append(f"The scene mainly looks like a {scene_type.replace('_', ' ')} setting.")
    if max_people > 0:
        parts.append(f"Up to {max_people} people are visible across the selected clips.")
    if activity_text:
        parts.append(f"The main visible activity is: {activity_text}.")
    if object_text:
        parts.append(f"Common visible objects include {object_text}.")
    if timeline:
        first_note = timeline[0].get("summary_sentence", "")
        if first_note:
            parts.append(f"One representative VLM description is: {first_note}.")
    return " ".join(part.strip() for part in parts if part.strip())


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    scene_overview = payload.get("scene_overview", {})
    timeline = payload.get("timeline", [])
    people_counts = scene_overview.get("people_count_observed", {}) if isinstance(scene_overview, dict) else {}
    lines = [
        "# VLM-First Summary",
        "",
        "## Summary",
        "",
        payload.get("descriptive_summary", ""),
        "",
        "## Scene Overview",
        "",
        f"- Dominant scene type: `{scene_overview.get('dominant_scene_type', 'unknown')}`",
        f"- Common activities: `{', '.join(scene_overview.get('common_activities', [])) if isinstance(scene_overview.get('common_activities'), list) else ''}`",
        f"- Common objects: `{', '.join(scene_overview.get('common_objects', [])) if isinstance(scene_overview.get('common_objects'), list) else ''}`",
        f"- People count observed: `{people_counts.get('min', 0)} to {people_counts.get('max', 0)}`",
        "",
        "## Timeline",
        "",
    ]
    for item in timeline:
        lines.append(f"- {item.get('time_range', 'unknown')}: {item.get('summary_sentence', '')}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a plain VLM-first summary from tender-demo Step 16 outputs.")
    parser.add_argument("--run-dir", default="", help="Tender demo run directory.")
    parser.add_argument("--vlm-outputs", default="", help="Direct path to 16_topk_vlm_outputs.json.")
    parser.add_argument("--output-dir", default="", help="Optional output directory.")
    args = parser.parse_args()

    if not args.run_dir and not args.vlm_outputs:
        raise ValueError("Provide --run-dir or --vlm-outputs.")

    run_dir, vlm_outputs_path, step15_path, video_info_path = _resolve_run_inputs(args)
    payload = _load_json(vlm_outputs_path)
    items = payload.get("items", []) if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError("Expected a list or object with 'items' in Step 16 outputs.")

    step15_payload = _load_json(step15_path) if step15_path else {}
    video_info = _load_json(video_info_path) if video_info_path else {}
    if not isinstance(step15_payload, dict):
        step15_payload = {}
    if not isinstance(video_info, dict):
        video_info = {}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = (_repo_root() / output_dir).resolve()
    else:
        run_name = run_dir.name if run_dir else vlm_outputs_path.parent.name
        output_dir = _debug_root() / f"{run_name}_vlm_summary_{timestamp}"
    _ensure_dir(output_dir)

    scene_overview = _build_scene_overview([item for item in items if isinstance(item, dict)])
    timeline = _build_timeline([item for item in items if isinstance(item, dict)])
    summary_payload = {
        "source_run_dir": str(run_dir) if run_dir else None,
        "source_vlm_outputs_path": str(vlm_outputs_path),
        "source_step15_path": str(step15_path) if step15_path else None,
        "source_video_info_path": str(video_info_path) if video_info_path else None,
        "input_snapshot": {
            "total_vlm_items": len(items),
            "vlm_backend": payload.get("vlm_backend") if isinstance(payload, dict) else None,
            "successful_outputs": payload.get("successful_outputs") if isinstance(payload, dict) else None,
            "failed_outputs": payload.get("failed_outputs") if isinstance(payload, dict) else None,
            "vlm_input_strategy": step15_payload.get("vlm_input_strategy"),
            "video_name": video_info.get("video_name"),
        },
        "scene_overview": scene_overview,
        "descriptive_summary": _build_descriptive_summary(scene_overview, timeline, video_info),
        "timeline": timeline,
    }

    (output_dir / "01_input_snapshot.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir) if run_dir else None,
                "vlm_outputs_path": str(vlm_outputs_path),
                "step15_path": str(step15_path) if step15_path else None,
                "video_info_path": str(video_info_path) if video_info_path else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "02_vlm_first_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    _write_markdown(output_dir / "03_vlm_first_summary.md", summary_payload)

    print(f"[vlm-summary-case] Output dir: {output_dir}")
    print(f"[vlm-summary-case] Source VLM outputs: {vlm_outputs_path}")
    print(f"[vlm-summary-case] Timeline items: {len(timeline)}")


if __name__ == "__main__":
    main()
