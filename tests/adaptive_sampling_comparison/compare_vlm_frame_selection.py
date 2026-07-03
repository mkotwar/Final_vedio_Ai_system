from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from tests.adaptive_sampling_comparison.adaptive_sampler_prototype import format_seconds_label
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    from adaptive_sampler_prototype import format_seconds_label


def _load_json(path: Path) -> dict[str, Any] | list[dict[str, Any]] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path_value: Any) -> Path | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        return path if path.exists() else None
    repo_root = _repo_root()
    candidate = repo_root / path
    return candidate if candidate.exists() else None


def _safe_float_env(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _nearest_item(target_seconds: float, items: list[dict[str, Any]], time_field: str) -> tuple[dict[str, Any] | None, float | None]:
    nearest_item = None
    nearest_distance = None
    for item in items:
        candidate_seconds = float(item.get(time_field, 0.0) or 0.0)
        distance = abs(candidate_seconds - target_seconds)
        if nearest_distance is None or distance < nearest_distance:
            nearest_item = item
            nearest_distance = distance
    return nearest_item, nearest_distance


def _coerce_item_list(payload: dict[str, Any] | list[dict[str, Any]] | None, candidate_keys: list[str]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in candidate_keys:
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_tender_vlm_candidates(tender_run_dir: Path) -> dict[str, Any]:
    vlm_inputs_payload = _load_json(tender_run_dir / "15_topk_vlm_inputs.json")
    selected_topk_payload = _load_json(tender_run_dir / "14_selected_top_clips.json")
    ranked_payload = _load_json(tender_run_dir / "13_ranked_clips.json")
    step16_payload = _load_json(tender_run_dir / "16_topk_vlm_outputs.json")
    step16b_payload = _load_json(tender_run_dir / "16b_incident_recheck_outputs.json")

    vlm_input_items = _coerce_item_list(vlm_inputs_payload, ["items", "vlm_inputs", "inputs"])
    selected_topk_items = _coerce_item_list(selected_topk_payload, ["selected_clips", "items"])
    ranked_items = _coerce_item_list(ranked_payload, ["ranked_clips", "items"])
    step16_items = _coerce_item_list(step16_payload, ["items"])
    step16b_items = _coerce_item_list(step16b_payload, ["items"])

    selected_by_clip_id = {
        str(item.get("clip_id", "")).strip(): item
        for item in selected_topk_items
        if isinstance(item, dict) and str(item.get("clip_id", "")).strip()
    }
    ranked_by_clip_id = {
        str(item.get("clip_id", "")).strip(): item
        for item in ranked_items
        if isinstance(item, dict) and str(item.get("clip_id", "")).strip()
    }
    step16_by_clip_id = {
        str(item.get("source_clip_id", "")).strip(): item
        for item in step16_items
        if isinstance(item, dict) and str(item.get("source_clip_id", "")).strip()
    }
    step16b_by_clip_id = {
        str(item.get("source_clip_id", "")).strip(): item
        for item in step16b_items
        if isinstance(item, dict) and str(item.get("source_clip_id", "")).strip()
    }

    tender_frames: list[dict[str, Any]] = []
    for item in vlm_input_items:
        if not isinstance(item, dict):
            continue
        clip_id = str(item.get("source_clip_id", "")).strip()
        current_time = None
        previous_time = _safe_float(item.get("previous_time"))
        next_time = _safe_float(item.get("next_time"))
        source_frame_times = item.get("source_frame_times", {}) if isinstance(item.get("source_frame_times"), dict) else {}

        if previous_time is None and "previous" in source_frame_times:
            previous_time = _safe_float(source_frame_times.get("previous"))
        if next_time is None and "next" in source_frame_times:
            next_time = _safe_float(source_frame_times.get("next"))

        if "current" in source_frame_times:
            current_time = _safe_float(source_frame_times.get("current"))
            panel_time_confidence = "high"
        elif item.get("current_time") is not None:
            current_time = _safe_float(item.get("current_time"))
            panel_time_confidence = "high"
        elif item.get("center_time") is not None:
            current_time = _safe_float(item.get("center_time"))
            panel_time_confidence = "high"
        elif item.get("timestamp_seconds") is not None:
            current_time = _safe_float(item.get("timestamp_seconds"))
            panel_time_confidence = "medium"
        else:
            current_time = _safe_float(item.get("start_time"))
            panel_time_confidence = "low"

        if current_time is None:
            current_time = 0.0
            panel_time_confidence = "low"

        if previous_time is None:
            previous_time = _safe_float(item.get("expanded_start_time"))
        if previous_time is None:
            previous_time = _safe_float(item.get("start_time"))
        if previous_time is None:
            previous_time = current_time
            panel_time_confidence = "low"

        if next_time is None:
            next_time = _safe_float(item.get("expanded_end_time"))
        if next_time is None:
            next_time = _safe_float(item.get("end_time"))
        if next_time is None:
            next_time = current_time
            panel_time_confidence = "low"

        clip_meta = selected_by_clip_id.get(clip_id, {})
        ranked_meta = ranked_by_clip_id.get(clip_id, {})
        step16_meta = step16_by_clip_id.get(clip_id, {})
        step16b_meta = step16b_by_clip_id.get(clip_id, {})
        tender_frames.append(
            {
                "clip_id": clip_id,
                "timestamp_seconds": current_time,
                "previous_panel_time": previous_time,
                "current_panel_time": current_time,
                "next_panel_time": next_time,
                "panel_time_confidence": panel_time_confidence,
                "time_candidates": [
                    previous_time,
                    current_time,
                    next_time,
                ],
                "strip_path": str(item.get("strip_path", "")),
                "start_time": float(item.get("start_time", 0.0) or 0.0),
                "end_time": float(item.get("end_time", 0.0) or 0.0),
                "selection_reasons": list(item.get("selection_reasons", []) or []),
                "ranking_reasons": list(item.get("ranking_reasons", []) or []),
                "ranked_clip_score": float(item.get("ranked_clip_score", ranked_meta.get("ranked_clip_score", 0.0)) or 0.0),
                "selected_clip": clip_meta,
                "step16_output": step16_meta,
                "step16b_output": step16b_meta,
            }
        )

    return {
        "vlm_input_items": vlm_input_items,
        "tender_frames": tender_frames,
        "selected_topk_items": selected_topk_items,
        "ranked_items": ranked_items,
        "step16_items": step16_items,
        "step16b_items": step16b_items,
    }


def _panel_candidate(item: dict[str, Any], panel_name: str, panel_time: float, target_seconds: float) -> dict[str, Any]:
    distance = round(abs(panel_time - target_seconds), 3)
    return {
        "item": item,
        "panel_name": panel_name,
        "panel_time": round(panel_time, 3),
        "distance_seconds": distance,
        "image_path": item.get("strip_path"),
        "clip_id": item.get("clip_id"),
        "panel_time_confidence": item.get("panel_time_confidence", "unknown"),
    }


def _find_tender_panel_coverage(target_seconds: float, tender_frames: list[dict[str, Any]]) -> dict[str, dict[str, Any] | None]:
    nearest_previous = None
    nearest_current = None
    nearest_next = None
    nearest_any = None

    for item in tender_frames:
        candidates = {
            "previous": _panel_candidate(
                item,
                "previous",
                float(item.get("previous_panel_time", item.get("timestamp_seconds", 0.0)) or 0.0),
                target_seconds,
            ),
            "current": _panel_candidate(
                item,
                "current",
                float(item.get("current_panel_time", item.get("timestamp_seconds", 0.0)) or 0.0),
                target_seconds,
            ),
            "next": _panel_candidate(
                item,
                "next",
                float(item.get("next_panel_time", item.get("timestamp_seconds", 0.0)) or 0.0),
                target_seconds,
            ),
        }

        if nearest_previous is None or candidates["previous"]["distance_seconds"] < nearest_previous["distance_seconds"]:
            nearest_previous = candidates["previous"]
        if nearest_current is None or candidates["current"]["distance_seconds"] < nearest_current["distance_seconds"]:
            nearest_current = candidates["current"]
        if nearest_next is None or candidates["next"]["distance_seconds"] < nearest_next["distance_seconds"]:
            nearest_next = candidates["next"]

        best_for_item = min(
            candidates.values(),
            key=lambda candidate: (candidate["distance_seconds"], candidate["panel_name"]),
        )
        if nearest_any is None or best_for_item["distance_seconds"] < nearest_any["distance_seconds"]:
            nearest_any = best_for_item

    return {
        "nearest_previous": nearest_previous,
        "nearest_current": nearest_current,
        "nearest_next": nearest_next,
        "nearest_any": nearest_any,
    }


def _load_adaptive_candidates(adaptive_output_dir: Path) -> list[dict[str, Any]]:
    payload = _load_json(adaptive_output_dir / "adaptive_retained_frames.json")
    items = payload.get("items", []) if isinstance(payload, dict) else []
    return [item for item in items if isinstance(item, dict)]


def _format_optional_seconds_label(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return format_seconds_label(float(value))
    except (TypeError, ValueError):
        return str(value)


def create_contact_sheet(
    comparison_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    cell_width = 380
    cell_height = 260
    header_height = 92
    row_height = header_height + cell_height + 20
    sheet = np.full((max(1, len(comparison_rows)) * row_height, cell_width * 2, 3), 245, dtype=np.uint8)

    for row_index, row in enumerate(comparison_rows):
        top_y = row_index * row_height
        cv2.putText(
            sheet,
            f"{row['timestamp']} | {row.get('meaning', '')}",
            (10, top_y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            sheet,
            f"Tender: {row.get('tender_coverage_type', 'n/a')}",
            (10, top_y + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            sheet,
            f"current distance: {row.get('nearest_tender_current_distance', 'n/a')}s",
            (10, top_y + 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            sheet,
            f"nearest any: {str(row.get('nearest_tender_any_panel_name', 'n/a')).upper()} {_format_optional_seconds_label(row.get('nearest_tender_any_panel_time'))} | {row.get('nearest_tender_any_panel_distance', 'n/a')}s",
            (10, top_y + 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            sheet,
            f"Adaptive: current/focal frame {_format_optional_seconds_label(row.get('nearest_adaptive_time'))}",
            (cell_width + 10, top_y + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            sheet,
            f"distance: {row.get('adaptive_distance_seconds', 'n/a')}s",
            (cell_width + 10, top_y + 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )

        tender_path = _resolve_path(row.get("chosen_tender_display_image_path"))
        adaptive_path = _resolve_path(row.get("nearest_adaptive_image_path"))
        for column_index, image_path in enumerate([tender_path, adaptive_path]):
            x0 = column_index * cell_width
            image = None
            if image_path and image_path.exists():
                image = cv2.imread(str(image_path))
            if image is None:
                image = np.full((cell_height, cell_width, 3), 220, dtype=np.uint8)
                cv2.putText(
                    image,
                    "Image not found",
                    (40, cell_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (60, 60, 60),
                    2,
                    cv2.LINE_AA,
                )
            else:
                image = cv2.resize(image, (cell_width, cell_height), interpolation=cv2.INTER_AREA)
            sheet[top_y + header_height : top_y + header_height + cell_height, x0 : x0 + cell_width] = image
            label = "Tender selected display" if column_index == 0 else "Adaptive retained frame"
            cv2.putText(
                sheet,
                label,
                (x0 + 10, top_y + header_height - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (10, 10, 10),
                2,
                cv2.LINE_AA,
            )

    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"Failed to write comparison contact sheet: {output_path}")
    return output_path


def compare_vlm_frame_selection(
    tender_run_dir: Path,
    adaptive_output_dir: Path,
    output_dir: Path,
    target_timestamps: list[float],
    target_labels: list[str],
    coverage_window_seconds: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage_threshold_seconds = _safe_float_env(
        "ADAPTIVE_COMPARE_COVERAGE_THRESHOLD_SECONDS",
        coverage_window_seconds,
    )

    tender_data = _extract_tender_vlm_candidates(tender_run_dir)
    tender_frames = tender_data["tender_frames"]
    adaptive_frames = _load_adaptive_candidates(adaptive_output_dir)

    comparison_rows: list[dict[str, Any]] = []
    summary_counts = {
        "tender_current_panel_covered_count": 0,
        "tender_context_only_covered_count": 0,
        "tender_missing_count": 0,
        "adaptive_covered_count": 0,
        "adaptive_better_current_focus_count": 0,
        "tender_better_count": 0,
        "both_missed_count": 0,
    }

    for index, target_seconds in enumerate(target_timestamps):
        tender_coverage = _find_tender_panel_coverage(target_seconds, tender_frames)
        adaptive_nearest, adaptive_distance = _nearest_item(target_seconds, adaptive_frames, "timestamp_seconds")
        nearest_previous = tender_coverage.get("nearest_previous")
        nearest_current = tender_coverage.get("nearest_current")
        nearest_next = tender_coverage.get("nearest_next")
        nearest_any = tender_coverage.get("nearest_any")
        distance_to_previous = nearest_previous.get("distance_seconds") if nearest_previous else None
        distance_to_current = nearest_current.get("distance_seconds") if nearest_current else None
        distance_to_next = nearest_next.get("distance_seconds") if nearest_next else None
        distance_to_any = nearest_any.get("distance_seconds") if nearest_any else None

        tender_current_panel_covered = bool(distance_to_current is not None and float(distance_to_current) <= coverage_threshold_seconds)
        tender_context_panel_covered = bool(
            not tender_current_panel_covered
            and (
                (distance_to_previous is not None and float(distance_to_previous) <= coverage_threshold_seconds)
                or (distance_to_next is not None and float(distance_to_next) <= coverage_threshold_seconds)
            )
        )
        tender_missing = not tender_current_panel_covered and not tender_context_panel_covered
        if tender_current_panel_covered:
            tender_coverage_type = "current_panel_covered"
            summary_counts["tender_current_panel_covered_count"] += 1
        elif tender_context_panel_covered:
            tender_coverage_type = "context_panel_covered"
            summary_counts["tender_context_only_covered_count"] += 1
        else:
            tender_coverage_type = "missing"
            summary_counts["tender_missing_count"] += 1

        adaptive_covered = bool(adaptive_distance is not None and adaptive_distance <= coverage_window_seconds)

        if adaptive_covered:
            summary_counts["adaptive_covered_count"] += 1

        if adaptive_covered and not tender_current_panel_covered:
            verdict = "adaptive_better_current_focus"
            summary_counts["adaptive_better_current_focus_count"] += 1
        elif tender_current_panel_covered and not adaptive_covered:
            verdict = "tender_better_current_focus"
            summary_counts["tender_better_count"] += 1
        elif tender_context_panel_covered and not adaptive_covered:
            verdict = "tender_context_only_better_than_missing"
            summary_counts["tender_better_count"] += 1
        elif tender_current_panel_covered and adaptive_covered:
            verdict = "both_current_capable"
        elif tender_context_panel_covered and adaptive_covered:
            verdict = "adaptive_better_current_focus"
            summary_counts["adaptive_better_current_focus_count"] += 1
        elif tender_missing and not adaptive_covered:
            verdict = "both_missed"
            summary_counts["both_missed_count"] += 1
        else:
            verdict = "uncertain"

        chosen_tender_display = nearest_current if tender_current_panel_covered and nearest_current else nearest_any

        comparison_rows.append(
            {
                "timestamp": format_seconds_label(target_seconds),
                "meaning": target_labels[index] if index < len(target_labels) else "",
                "tender_coverage_type": tender_coverage_type,
                "tender_current_panel_covered": tender_current_panel_covered,
                "tender_context_panel_covered": tender_context_panel_covered,
                "tender_missing": tender_missing,
                "panel_time_confidence": nearest_current.get("panel_time_confidence") if nearest_current else "unknown",
                "nearest_tender_previous_time": nearest_previous.get("panel_time") if nearest_previous else None,
                "nearest_tender_previous_distance": distance_to_previous,
                "nearest_tender_previous_clip_id": nearest_previous.get("clip_id") if nearest_previous else None,
                "nearest_tender_previous_image_path": nearest_previous.get("image_path") if nearest_previous else None,
                "nearest_tender_current_time": nearest_current.get("panel_time") if nearest_current else None,
                "nearest_tender_current_distance": distance_to_current,
                "nearest_tender_current_clip_id": nearest_current.get("clip_id") if nearest_current else None,
                "nearest_tender_current_image_path": nearest_current.get("image_path") if nearest_current else None,
                "nearest_tender_next_time": nearest_next.get("panel_time") if nearest_next else None,
                "nearest_tender_next_distance": distance_to_next,
                "nearest_tender_next_clip_id": nearest_next.get("clip_id") if nearest_next else None,
                "nearest_tender_next_image_path": nearest_next.get("image_path") if nearest_next else None,
                "nearest_tender_any_panel_name": nearest_any.get("panel_name") if nearest_any else None,
                "nearest_tender_any_panel_time": nearest_any.get("panel_time") if nearest_any else None,
                "nearest_tender_any_panel_distance": distance_to_any,
                "nearest_tender_any_clip_id": nearest_any.get("clip_id") if nearest_any else None,
                "nearest_tender_any_image_path": nearest_any.get("image_path") if nearest_any else None,
                "chosen_tender_display_image_path": chosen_tender_display.get("image_path") if chosen_tender_display else None,
                "chosen_tender_display_clip_id": chosen_tender_display.get("clip_id") if chosen_tender_display else None,
                "distance_to_previous_panel_seconds": distance_to_previous,
                "distance_to_current_panel_seconds": distance_to_current,
                "distance_to_next_panel_seconds": distance_to_next,
                "distance_to_nearest_any_panel_seconds": distance_to_any,
                "adaptive_covered": adaptive_covered,
                "nearest_adaptive_time": float(adaptive_nearest.get("timestamp_seconds", 0.0) or 0.0) if adaptive_nearest else None,
                "adaptive_distance_seconds": round(float(adaptive_distance), 3) if adaptive_distance is not None else None,
                "nearest_adaptive_image_path": adaptive_nearest.get("frame_path") if adaptive_nearest else None,
                "verdict": verdict,
            }
        )

    comparison_json = {
        "target_timestamps": [format_seconds_label(value) for value in target_timestamps],
        "coverage_threshold_seconds": coverage_threshold_seconds,
        "comparison": comparison_rows,
        "summary": summary_counts,
    }

    json_path = output_dir / "vlm_selection_comparison.json"
    json_path.write_text(json.dumps(comparison_json, indent=2), encoding="utf-8")

    md_lines = [
        "# VLM Frame Selection Comparison",
        "",
        "Tender coverage distinguishes CURRENT panel coverage from weaker PREVIOUS/NEXT context coverage.",
        "Qwen focuses on the CURRENT panel and uses PREVIOUS/NEXT only as context, so context-only coverage is weaker than true focal coverage.",
        "",
        "| Timestamp | Meaning | Tender coverage type | Nearest current | Nearest context | Nearest any | Adaptive focal | Verdict |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in comparison_rows:
        context_label = (
            f"PREV {_format_optional_seconds_label(row.get('nearest_tender_previous_time'))} ({row.get('nearest_tender_previous_distance')}s) / "
            f"NEXT {_format_optional_seconds_label(row.get('nearest_tender_next_time'))} ({row.get('nearest_tender_next_distance')}s)"
        )
        md_lines.append(
            f"| {row['timestamp']} | {row.get('meaning', '')} | {row['tender_coverage_type']} | "
            f"{_format_optional_seconds_label(row.get('nearest_tender_current_time'))} ({row.get('nearest_tender_current_distance')}s) | "
            f"{context_label} | "
            f"{str(row.get('nearest_tender_any_panel_name', 'n/a')).upper()} {_format_optional_seconds_label(row.get('nearest_tender_any_panel_time'))} ({row.get('nearest_tender_any_panel_distance')}s) | "
            f"{_format_optional_seconds_label(row.get('nearest_adaptive_time'))} ({row.get('adaptive_distance_seconds')}s) | {row['verdict']} |"
        )
    md_lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Tender current-panel covered count: {summary_counts['tender_current_panel_covered_count']}",
            f"- Tender context-only covered count: {summary_counts['tender_context_only_covered_count']}",
            f"- Tender missing count: {summary_counts['tender_missing_count']}",
            f"- Adaptive covered count: {summary_counts['adaptive_covered_count']}",
            f"- Adaptive better current-focus count: {summary_counts['adaptive_better_current_focus_count']}",
            f"- Tender better count: {summary_counts['tender_better_count']}",
            f"- Both missed count: {summary_counts['both_missed_count']}",
        ]
    )
    md_path = output_dir / "vlm_selection_comparison.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    csv_path = output_dir / "vlm_selection_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "timestamp",
                "meaning",
                "tender_coverage_type",
                "tender_current_panel_covered",
                "tender_context_panel_covered",
                "tender_missing",
                "nearest_tender_previous_time",
                "nearest_tender_previous_distance",
                "nearest_tender_previous_clip_id",
                "nearest_tender_previous_image_path",
                "nearest_tender_current_time",
                "nearest_tender_current_distance",
                "nearest_tender_current_clip_id",
                "nearest_tender_current_image_path",
                "nearest_tender_next_time",
                "nearest_tender_next_distance",
                "nearest_tender_next_clip_id",
                "nearest_tender_next_image_path",
                "nearest_tender_any_panel_name",
                "nearest_tender_any_panel_time",
                "nearest_tender_any_panel_distance",
                "nearest_tender_any_clip_id",
                "nearest_tender_any_image_path",
                "chosen_tender_display_image_path",
                "chosen_tender_display_clip_id",
                "adaptive_covered",
                "nearest_adaptive_time",
                "adaptive_distance_seconds",
                "verdict",
            ],
        )
        writer.writeheader()
        for row in comparison_rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})

    contact_sheet_path = create_contact_sheet(comparison_rows, output_dir / "comparison_contact_sheet.jpg")

    return {
        "comparison_json_path": json_path,
        "comparison_md_path": md_path,
        "comparison_csv_path": csv_path,
        "contact_sheet_path": contact_sheet_path,
        "comparison": comparison_json,
    }
