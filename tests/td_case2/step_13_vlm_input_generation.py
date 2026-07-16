from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stage_checks import format_seconds_text, read_json, write_json
from step_09_search_result_packaging import write_json_any


ALLOWED_STRIP_MODES = {"three_panel", "five_panel"}
PRIORITY_ORDER = {"high": 3, "medium": 2, "low": 1, None: 0}
FILTERED_SOURCE_FILE = "11_5_vlm_filtered_event_candidates.json"
STEP11_SOURCE_FILE = "11_full_scene_event_candidates.json"


def _normalize_rel_path(path_value: str | None) -> str | None:
    """Normalize a run-relative path into forward-slash form."""

    if not path_value:
        return None
    normalized = str(path_value).strip().replace("\\", "/")
    return normalized or None


def _resolve_run_path(run_dir: Path, path_value: str | None) -> Path | None:
    """Resolve an image path relative to the run directory."""

    normalized = _normalize_rel_path(path_value)
    if not normalized:
        return None
    path = Path(normalized)
    if path.is_absolute():
        return path
    return (run_dir / path).resolve()


def _max_priority(values: list[str | None]) -> str | None:
    """Return the strongest VLM priority label."""

    if not values:
        return None
    return max(values, key=lambda item: PRIORITY_ORDER.get(item, 0))


def _candidate_summary_text(event_types: list[str], best_timestamp_text: str) -> str:
    """Build a compact candidate-only summary string."""

    label = " / ".join(dict.fromkeys(event_types))
    label = label.replace("_", " ")
    return f"{label.capitalize()} candidate around {best_timestamp_text}."


def _load_sampled_frames(run_dir: Path) -> list[dict[str, Any]]:
    """Load sampled full-scene frame manifest if available."""

    manifest_path = run_dir / "02_sampled_frames.json"
    if not manifest_path.exists():
        return []
    payload = read_json(manifest_path)
    return list(payload.get("frames", payload.get("sampled_frames", [])))


def _frame_item_from_path(sampled_frames: list[dict[str, Any]], image_path: str | None) -> dict[str, Any] | None:
    """Find a sampled frame by image path."""

    normalized = _normalize_rel_path(image_path)
    if not normalized:
        return None
    for frame in sampled_frames:
        if _normalize_rel_path(frame.get("image_path")) == normalized:
            return frame
    return None


def _nearest_frame_by_timestamp(sampled_frames: list[dict[str, Any]], target_seconds: float) -> dict[str, Any] | None:
    """Find the nearest sampled frame to a target timestamp."""

    if not sampled_frames:
        return None
    return min(
        sampled_frames,
        key=lambda item: abs(float(item.get("timestamp_seconds", 0.0) or 0.0) - target_seconds),
    )


def _candidate_with_enrichment(
    selected_candidate: dict[str, Any],
    ranked_map: dict[str, dict[str, Any]],
    step11_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Merge the selected Step 12 shape with richer ranked/Step 11 fields."""

    candidate_id = str(selected_candidate.get("candidate_event_id", "") or "")
    enriched = dict(step11_map.get(candidate_id, {}))
    enriched.update(ranked_map.get(candidate_id, {}))
    for key, value in selected_candidate.items():
        if value is not None:
            enriched[key] = value
    return enriched


def _selected_candidate_valid(candidate: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate minimum Step 13 candidate requirements."""

    warnings: list[str] = []
    if not bool(candidate.get("selected_for_vlm")):
        warnings.append("selected_for_vlm_false")
    if not bool(candidate.get("needs_vlm_review")):
        warnings.append("needs_vlm_review_false")
    if candidate.get("context_start_seconds") is None or candidate.get("context_end_seconds") is None:
        warnings.append("missing_context_window")
    if not candidate.get("representative_frame_path"):
        warnings.append("missing_representative_frame_path")
    if not list(candidate.get("full_frame_paths", [])):
        warnings.append("missing_full_frame_paths")
    return len(warnings) == 0, warnings


def _should_merge(left: dict[str, Any], right: dict[str, Any], merge_gap_seconds: float) -> bool:
    """Return whether two selected candidates should be merged into one VLM group."""

    if bool(dict(left.get("ranking", {})).get("critical_event")) or bool(dict(right.get("ranking", {})).get("critical_event")):
        return False
    left_start = float(left.get("context_start_seconds", 0.0) or 0.0)
    left_end = float(left.get("context_end_seconds", 0.0) or 0.0)
    right_start = float(right.get("context_start_seconds", 0.0) or 0.0)
    right_end = float(right.get("context_end_seconds", 0.0) or 0.0)
    left_best = float(left.get("best_timestamp_seconds", 0.0) or 0.0)
    right_best = float(right.get("best_timestamp_seconds", 0.0) or 0.0)
    left_cluster = str(left.get("selection", {}).get("temporal_cluster_id", "") or "")
    right_cluster = str(right.get("selection", {}).get("temporal_cluster_id", "") or "")
    windows_overlap = not (left_end < right_start or right_end < left_start)
    nearby_best = abs(left_best - right_best) <= merge_gap_seconds
    same_cluster = bool(left_cluster) and left_cluster == right_cluster
    return windows_overlap or nearby_best or same_cluster


def _merge_selected_candidates(
    selected_candidates: list[dict[str, Any]],
    *,
    merge_enabled: bool,
    merge_gap_seconds: float,
    max_group_duration_seconds: float,
    context_before_seconds: float,
    context_after_seconds: float,
) -> list[dict[str, Any]]:
    """Merge nearby selected candidates into VLM input groups."""

    ordered = sorted(selected_candidates, key=lambda item: float(item.get("best_timestamp_seconds", 0.0) or 0.0))
    if not merge_enabled:
        return [
            {
                "group_index": index,
                "candidates": [candidate],
                "highest_ranked_candidate": candidate,
                "source_candidate_ids": [str(candidate.get("candidate_event_id", "") or "")],
                "source_event_types": [str(candidate.get("event_type", "") or "")],
                "source_best_timestamps": [str(candidate.get("best_timestamp_text", "") or "")],
                "selection_ranks": [
                    int(candidate.get("selection_rank", candidate.get("selection", {}).get("selection_rank", 0)) or 0)
                ],
                "merged_context_start_seconds": round(float(candidate.get("context_start_seconds", 0.0) or 0.0), 6),
                "merged_context_end_seconds": round(float(candidate.get("context_end_seconds", 0.0) or 0.0), 6),
                "group_best_timestamp_seconds": round(float(candidate.get("best_timestamp_seconds", 0.0) or 0.0), 6),
                "max_ranking_score": round(
                    float(candidate.get("ranking", {}).get("ranking_score", candidate.get("ranking_score", 0.0)) or 0.0),
                    6,
                ),
                "max_vlm_priority": candidate.get("ranking", {}).get("vlm_priority", candidate.get("vlm_priority")),
                "combined_trigger_reasons": sorted(list(candidate.get("trigger_reasons", []))),
                "combined_involved_classes": sorted(list(candidate.get("involved_classes", []))),
                "temporal_cluster_ids": [
                    str(candidate.get("selection", {}).get("temporal_cluster_id", "") or "")
                ]
                if str(candidate.get("selection", {}).get("temporal_cluster_id", "") or "")
                else [],
                "merged": False,
            }
            for index, candidate in enumerate(ordered, start=1)
        ]

    grouped: list[list[dict[str, Any]]] = []
    for candidate in ordered:
        if not grouped or not _should_merge(grouped[-1][-1], candidate, merge_gap_seconds):
            grouped.append([candidate])
        else:
            grouped[-1].append(candidate)

    merged_groups: list[dict[str, Any]] = []
    for index, group_candidates in enumerate(grouped, start=1):
        highest_ranked = max(
            group_candidates,
            key=lambda item: float(item.get("ranking", {}).get("ranking_score", item.get("ranking_score", 0.0)) or 0.0),
        )
        group_best_timestamp_seconds = float(highest_ranked.get("best_timestamp_seconds", 0.0) or 0.0)
        context_start_seconds = min(float(item.get("context_start_seconds", 0.0) or 0.0) for item in group_candidates)
        context_end_seconds = max(float(item.get("context_end_seconds", 0.0) or 0.0) for item in group_candidates)
        if context_end_seconds - context_start_seconds > max_group_duration_seconds:
            context_start_seconds = max(0.0, group_best_timestamp_seconds - context_before_seconds)
            context_end_seconds = group_best_timestamp_seconds + context_after_seconds
            if context_end_seconds - context_start_seconds > max_group_duration_seconds:
                context_end_seconds = context_start_seconds + max_group_duration_seconds
        merged_groups.append(
            {
                "group_index": index,
                "candidates": group_candidates,
                "highest_ranked_candidate": highest_ranked,
                "source_candidate_ids": [str(item.get("candidate_event_id", "") or "") for item in group_candidates],
                "source_event_types": [str(item.get("event_type", "") or "") for item in group_candidates],
                "source_best_timestamps": [str(item.get("best_timestamp_text", "") or "") for item in group_candidates],
                "selection_ranks": [int(item.get("selection_rank", item.get("selection", {}).get("selection_rank", 0)) or 0) for item in group_candidates],
                "merged_context_start_seconds": round(context_start_seconds, 6),
                "merged_context_end_seconds": round(context_end_seconds, 6),
                "group_best_timestamp_seconds": round(group_best_timestamp_seconds, 6),
                "max_ranking_score": round(
                    max(float(item.get("ranking", {}).get("ranking_score", item.get("ranking_score", 0.0)) or 0.0) for item in group_candidates),
                    6,
                ),
                "max_vlm_priority": _max_priority(
                    [item.get("ranking", {}).get("vlm_priority", item.get("vlm_priority")) for item in group_candidates]
                ),
                "combined_trigger_reasons": sorted(
                    {reason for item in group_candidates for reason in list(item.get("trigger_reasons", []))}
                ),
                "combined_involved_classes": sorted(
                    {class_name for item in group_candidates for class_name in list(item.get("involved_classes", []))}
                ),
                "temporal_cluster_ids": sorted(
                    {
                        str(item.get("selection", {}).get("temporal_cluster_id", "") or "")
                        for item in group_candidates
                        if str(item.get("selection", {}).get("temporal_cluster_id", "") or "")
                    }
                ),
                "merged": len(group_candidates) > 1,
            }
        )
    return merged_groups


def _load_image_or_placeholder(image_path: Path | None, panel_height: int, label: str) -> np.ndarray:
    """Load an image for strip assembly or create a placeholder panel."""

    if image_path is not None and image_path.exists():
        image = cv2.imread(str(image_path))
        if image is not None:
            return image
    placeholder = np.full((panel_height, max(320, panel_height // 2), 3), 28, dtype=np.uint8)
    cv2.putText(
        placeholder,
        "MISSING FRAME",
        (20, panel_height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        placeholder,
        label,
        (20, min(panel_height - 24, panel_height // 2 + 50)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (180, 180, 180),
        2,
        cv2.LINE_AA,
    )
    return placeholder


def _fit_panel(image: np.ndarray, panel_width: int, panel_height: int) -> np.ndarray:
    """Resize an image into a fixed panel while keeping aspect ratio."""

    image_height, image_width = image.shape[:2]
    scale = min(panel_width / max(1, image_width), panel_height / max(1, image_height))
    resized_width = max(1, int(round(image_width * scale)))
    resized_height = max(1, int(round(image_height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    panel = np.full((panel_height, panel_width, 3), 22, dtype=np.uint8)
    offset_x = (panel_width - resized_width) // 2
    offset_y = (panel_height - resized_height) // 2
    panel[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
    return panel


def _annotate_panel(panel: np.ndarray, top_label: str, timestamp_text: str, bottom_label: str) -> np.ndarray:
    """Draw lightweight panel annotations."""

    annotated = panel.copy()
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 50), (12, 12, 12), thickness=-1)
    cv2.rectangle(
        annotated,
        (0, annotated.shape[0] - 46),
        (annotated.shape[1], annotated.shape[0]),
        (12, 12, 12),
        thickness=-1,
    )
    cv2.putText(annotated, top_label, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(annotated, timestamp_text, (18, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(
        annotated,
        bottom_label,
        (18, annotated.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (210, 210, 210),
        2,
        cv2.LINE_AA,
    )
    return annotated


def _build_strip_image(
    *,
    run_dir: Path,
    output_path: Path,
    frames: list[dict[str, Any]],
    strip_mode: str,
    strip_width: int,
    panel_height: int,
    add_labels: bool,
    group_id: str,
) -> bool:
    """Create one temporal strip image."""

    panel_count = 3 if strip_mode == "three_panel" else 5
    panel_width = max(1, strip_width // panel_count)
    panels: list[np.ndarray] = []
    for frame in frames:
        image = _load_image_or_placeholder(_resolve_run_path(run_dir, frame.get("image_path")), panel_height, str(frame.get("panel_label", "")))
        panel = _fit_panel(image, panel_width, panel_height)
        if add_labels:
            panel = _annotate_panel(
                panel,
                str(frame.get("panel_label", "")),
                str(frame.get("timestamp_text", "")),
                group_id,
            )
        panels.append(panel)
    if not panels:
        return False
    strip = np.concatenate(panels, axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), strip))


def _build_contact_sheet(
    *,
    run_dir: Path,
    output_path: Path,
    frame_items: list[dict[str, Any]],
    panel_height: int,
) -> bool:
    """Create a simple multi-panel contact sheet from available group frames."""

    unique_frames: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for frame in frame_items:
        normalized = _normalize_rel_path(frame.get("image_path"))
        if normalized and normalized not in seen_paths:
            seen_paths.add(normalized)
            unique_frames.append(frame)
    if not unique_frames:
        return False
    rows: list[np.ndarray] = []
    columns = min(3, len(unique_frames))
    panel_width = 420
    panels: list[np.ndarray] = []
    for frame in unique_frames:
        image = _load_image_or_placeholder(_resolve_run_path(run_dir, frame.get("image_path")), panel_height, "CONTACT")
        panel = _fit_panel(image, panel_width, panel_height)
        panel = _annotate_panel(panel, "CONTEXT", str(frame.get("timestamp_text", "")), str(frame.get("frame_id", "")))
        panels.append(panel)
    for row_start in range(0, len(panels), columns):
        row_panels = panels[row_start : row_start + columns]
        while len(row_panels) < columns:
            row_panels.append(np.full((panel_height, panel_width, 3), 20, dtype=np.uint8))
        rows.append(np.concatenate(row_panels, axis=1))
    sheet = np.concatenate(rows, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), sheet))


def _panel_targets(
    *,
    strip_mode: str,
    best_timestamp_seconds: float,
    context_start_seconds: float,
    context_end_seconds: float,
) -> list[tuple[str, float]]:
    """Return target timestamps and labels for strip assembly."""

    if strip_mode == "five_panel":
        return [
            ("CONTEXT START", context_start_seconds),
            ("PREVIOUS", max(context_start_seconds, best_timestamp_seconds - 1.0)),
            ("CURRENT / EVENT CENTER", best_timestamp_seconds),
            ("NEXT", min(context_end_seconds, best_timestamp_seconds + 1.0)),
            ("CONTEXT END", context_end_seconds),
        ]
    return [
        ("PREVIOUS", max(context_start_seconds, best_timestamp_seconds - 2.0)),
        ("CURRENT / EVENT CENTER", best_timestamp_seconds),
        ("NEXT", min(context_end_seconds, best_timestamp_seconds + 2.0)),
    ]


def _frame_package_for_group(
    *,
    sampled_frames: list[dict[str, Any]],
    group: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Choose frame packages for one VLM group."""

    warnings: list[str] = []
    targets = _panel_targets(
        strip_mode=str(group["strip_mode"]),
        best_timestamp_seconds=float(group["group_best_timestamp_seconds"]),
        context_start_seconds=float(group["merged_context_start_seconds"]),
        context_end_seconds=float(group["merged_context_end_seconds"]),
    )
    candidate_path_frames = []
    for candidate in list(group.get("candidates", [])):
        if candidate.get("representative_frame_path"):
            found = _frame_item_from_path(sampled_frames, candidate.get("representative_frame_path"))
            if found:
                candidate_path_frames.append(found)
        for image_path in list(candidate.get("full_frame_paths", [])):
            found = _frame_item_from_path(sampled_frames, image_path)
            if found:
                candidate_path_frames.append(found)

    selected_frames: list[dict[str, Any]] = []
    for label, target_seconds in targets:
        frame_item = _nearest_frame_by_timestamp(sampled_frames, target_seconds)
        if frame_item is None and candidate_path_frames:
            frame_item = min(
                candidate_path_frames,
                key=lambda item: abs(float(item.get("timestamp_seconds", 0.0) or 0.0) - target_seconds),
            )
        if frame_item is None:
            warnings.append(f"missing_frame_for_{label.lower().replace(' ', '_')}")
            selected_frames.append(
                {
                    "panel_label": label,
                    "timestamp_seconds": target_seconds,
                    "timestamp_text": format_seconds_text(target_seconds),
                    "image_path": None,
                    "frame_id": None,
                }
            )
            continue
        selected_frames.append(
            {
                "panel_label": label,
                "timestamp_seconds": float(frame_item.get("timestamp_seconds", 0.0) or 0.0),
                "timestamp_text": str(frame_item.get("timestamp_text", "") or format_seconds_text(target_seconds)),
                "image_path": _normalize_rel_path(frame_item.get("image_path")),
                "frame_id": frame_item.get("frame_id"),
            }
        )
    return selected_frames, warnings


def run_vlm_input_generation(
    *,
    run_dir: Path,
    vlm_config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Create Step 13 full-scene VLM input packages from selected Step 12 event candidates."""

    selected_payload = read_json(run_dir / "12_selected_top_event_candidates.json")
    ranking_report_payload = read_json(run_dir / "12_event_candidate_ranking_report.json")
    selected_source_file = str(selected_payload.get("source_file", "") or "")
    preferred_source_path = run_dir / FILTERED_SOURCE_FILE
    if selected_source_file == FILTERED_SOURCE_FILE and preferred_source_path.exists():
        step11_payload = read_json(preferred_source_path)
        candidate_source_file = FILTERED_SOURCE_FILE
    elif preferred_source_path.exists():
        filtered_payload = read_json(preferred_source_path)
        if filtered_payload.get("status") == "success" and list(filtered_payload.get("candidate_events", [])):
            step11_payload = filtered_payload
            candidate_source_file = FILTERED_SOURCE_FILE
        else:
            step11_payload = read_json(run_dir / STEP11_SOURCE_FILE)
            candidate_source_file = STEP11_SOURCE_FILE
    else:
        step11_payload = read_json(run_dir / STEP11_SOURCE_FILE)
        candidate_source_file = STEP11_SOURCE_FILE
    video_info_payload = read_json(run_dir / "01_video_info.json")
    ranked_payload = read_json(run_dir / "12_ranked_event_candidates.json") if (run_dir / "12_ranked_event_candidates.json").exists() else {}

    selected_candidates = list(selected_payload.get("selected_candidates", []))
    ranked_candidates = list(ranked_payload.get("ranked_candidates", []))
    ranked_map = {str(item.get("candidate_event_id", "") or ""): item for item in ranked_candidates}
    step11_map = {str(item.get("candidate_event_id", "") or ""): item for item in list(step11_payload.get("candidate_events", []))}
    sampled_frames = _load_sampled_frames(run_dir)
    output_dir = run_dir / "13_vlm_event_inputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not selected_candidates:
        output_payload = {
            "status": "success",
            "source_file": "12_selected_top_event_candidates.json",
            "config": vlm_config,
            "summary": {
                "selected_candidates_loaded": 0,
                "merged_vlm_input_groups": 0,
                "vlm_inputs_created": 0,
                "temporal_strips_created": 0,
                "contact_sheets_created": 0,
                "inputs_ready_for_vlm": 0,
                "inputs_skipped": 0,
                "ready_for_step14_vlm_event_review": False,
            },
            "vlm_inputs": [],
        }
        report_payload = {
            "status": "success",
            "candidate_source_file": candidate_source_file,
            "selected_candidates_loaded": 0,
            "merged_groups_created": 0,
            "vlm_inputs_created": 0,
            "temporal_strips_created": 0,
            "contact_sheets_created": 0,
            "inputs_ready_for_vlm": 0,
            "inputs_skipped": 0,
            "merge_summary": {"merged_candidate_count": 0, "unmerged_candidate_count": 0, "largest_group_size": 0, "groups": []},
            "priority_counts": {},
            "warnings": [],
            "recommendation": "No selected candidates available for VLM input generation.",
        }
        write_json(run_dir / "13_vlm_event_inputs.json", output_payload)
        write_json_any(run_dir / "13_vlm_event_inputs_flat.json", [])
        write_json(run_dir / "13_vlm_event_input_report.json", report_payload)
        return output_payload, [], report_payload

    enriched_selected: list[dict[str, Any]] = []
    global_warnings: list[str] = []
    for item in selected_candidates[: int(vlm_config["max_inputs"])]:
        enriched = _candidate_with_enrichment(item, ranked_map, step11_map)
        enriched.setdefault("selection", {})
        if item.get("selection_rank") is not None:
            enriched["selection"]["selection_rank"] = int(item.get("selection_rank", 0) or 0)
        enriched["selection_rank"] = int(item.get("selection_rank", enriched.get("selection", {}).get("selection_rank", 0)) or 0)
        valid, warnings = _selected_candidate_valid(enriched)
        enriched["step13_validation"] = {
            "selected_candidate_valid": valid,
            "warnings": warnings,
        }
        enriched_selected.append(enriched)
        global_warnings.extend(f"{enriched.get('candidate_event_id')}: {warning}" for warning in warnings)

    merged_groups = _merge_selected_candidates(
        enriched_selected,
        merge_enabled=bool(vlm_config["merge_nearby_selected"]),
        merge_gap_seconds=float(vlm_config["merge_gap_seconds"]),
        max_group_duration_seconds=float(vlm_config["max_group_duration_seconds"]),
        context_before_seconds=float(vlm_config["context_before_seconds"]),
        context_after_seconds=float(vlm_config["context_after_seconds"]),
    )

    vlm_inputs: list[dict[str, Any]] = []
    flat_inputs: list[dict[str, Any]] = []
    strips_created = 0
    contact_sheets_created = 0
    inputs_skipped = 0
    priority_counts: Counter[str] = Counter()

    for index, group in enumerate(merged_groups, start=1):
        group["strip_mode"] = str(vlm_config["strip_mode"])
        group_id = f"vlm_event_input_{index:06d}"
        frame_packages, frame_warnings = _frame_package_for_group(sampled_frames=sampled_frames, group=group)
        global_warnings.extend(f"{group_id}: {warning}" for warning in frame_warnings)
        existing_frame_paths = [frame for frame in frame_packages if frame.get("image_path")]
        if not existing_frame_paths and bool(vlm_config["require_full_frame_exists"]):
            inputs_skipped += 1
            global_warnings.append(f"{group_id}: skipped because no full-scene frame could be resolved.")
            continue

        strip_path = output_dir / f"{group_id}_strip.jpg"
        strip_ok = _build_strip_image(
            run_dir=run_dir,
            output_path=strip_path,
            frames=frame_packages,
            strip_mode=str(vlm_config["strip_mode"]),
            strip_width=int(vlm_config["strip_width"]),
            panel_height=int(vlm_config["strip_panel_height"]),
            add_labels=bool(vlm_config["add_labels"]),
            group_id=group_id,
        )
        if strip_ok:
            strips_created += 1

        contact_sheet_path = output_dir / f"{group_id}_contact_sheet.jpg"
        contact_sheet_rel: str | None = None
        if bool(vlm_config["save_contact_sheet"]):
            if _build_contact_sheet(
                run_dir=run_dir,
                output_path=contact_sheet_path,
                frame_items=frame_packages,
                panel_height=max(280, int(vlm_config["strip_panel_height"]) // 2),
            ):
                contact_sheets_created += 1
                contact_sheet_rel = contact_sheet_path.relative_to(run_dir).as_posix()

        highest_ranked = dict(group["highest_ranked_candidate"])
        best_timestamp_seconds = float(group["group_best_timestamp_seconds"])
        best_timestamp_text = format_seconds_text(best_timestamp_seconds)
        max_vlm_priority = _max_priority(
            [candidate.get("ranking", {}).get("vlm_priority", candidate.get("vlm_priority")) for candidate in group["candidates"]]
        )
        priority_counts[str(max_vlm_priority or "low")] += 1
        source_candidate_ids = list(group["source_candidate_ids"])
        source_event_types = list(dict.fromkeys(group["source_event_types"]))
        combined_trigger_reasons = list(group["combined_trigger_reasons"])
        combined_involved_classes = list(group["combined_involved_classes"])
        representative_frame_path = highest_ranked.get("representative_frame", {}).get("image_path") or highest_ranked.get("representative_frame_path")
        metadata_payload = {
            "vlm_input_id": group_id,
            "input_type": "full_scene_event_temporal_strip",
            "schema_version": "v1",
            "source_candidate_ids": source_candidate_ids,
            "source_event_types": source_event_types,
            "merged": bool(group["merged"]),
            "selection_ranks": [int(rank) for rank in group["selection_ranks"] if int(rank) > 0],
            "best_timestamp_seconds": round(best_timestamp_seconds, 6),
            "best_timestamp_text": best_timestamp_text,
            "context_start_seconds": float(group["merged_context_start_seconds"]),
            "context_end_seconds": float(group["merged_context_end_seconds"]),
            "ranking": {
                "max_ranking_score": float(group["max_ranking_score"]),
                "max_vlm_priority": max_vlm_priority,
            },
            "scene_context": {
                "candidate_event_summary": _candidate_summary_text(source_event_types, best_timestamp_text),
                "trigger_reasons": combined_trigger_reasons,
                "involved_classes": combined_involved_classes,
                "final_event_truth": "unknown_candidate_only",
                "instruction": "Review full scene and decide what is visibly happening. Do not assume accident unless visible.",
            },
            "frames": {
                "previous": frame_packages[0] if frame_packages else None,
                "current": frame_packages[len(frame_packages) // 2] if frame_packages else None,
                "next": frame_packages[-1] if frame_packages else None,
                "all_panels": frame_packages,
            },
            "media": {
                "temporal_strip_path": strip_path.relative_to(run_dir).as_posix() if strip_ok else None,
                "contact_sheet_path": contact_sheet_rel,
                "primary_frame_path": _normalize_rel_path(representative_frame_path),
            },
            "prompt_context": {
                "vlm_task": "analyze_event_candidate",
                "candidate_only": True,
                "important_instruction": "This is only a candidate event. Report visible facts only. Do not confirm collision, accident, or violation unless visible.",
                "question": "What is happening in the CURRENT panel? Use previous and next panels only as temporal context.",
                "expected_output": "strict_json",
            },
            "ready_for_vlm": bool(strip_ok and existing_frame_paths),
        }
        metadata_path = output_dir / f"{group_id}_metadata.json"
        write_json(metadata_path, metadata_payload)
        vlm_inputs.append(metadata_payload)
        flat_inputs.append(
            {
                "vlm_input_id": group_id,
                "source_candidate_ids": ", ".join(source_candidate_ids),
                "source_event_types": ", ".join(source_event_types),
                "best_timestamp_text": best_timestamp_text,
                "context_start_seconds": float(group["merged_context_start_seconds"]),
                "context_end_seconds": float(group["merged_context_end_seconds"]),
                "max_ranking_score": float(group["max_ranking_score"]),
                "max_vlm_priority": max_vlm_priority,
                "temporal_strip_path": strip_path.relative_to(run_dir).as_posix() if strip_ok else None,
                "contact_sheet_path": contact_sheet_rel,
                "ready_for_vlm": metadata_payload["ready_for_vlm"],
            }
        )

    merge_groups_summary = [
        {
            "vlm_input_id": item["vlm_input_id"],
            "source_candidate_ids": item["source_candidate_ids"],
            "best_timestamp_text": item["best_timestamp_text"],
        }
        for item in vlm_inputs
    ]
    report_payload = {
        "status": "success",
        "candidate_source_file": candidate_source_file,
        "selected_candidates_loaded": len(selected_candidates[: int(vlm_config["max_inputs"])]),
        "merged_groups_created": len(merged_groups),
        "vlm_inputs_created": len(vlm_inputs),
        "temporal_strips_created": strips_created,
        "contact_sheets_created": contact_sheets_created,
        "inputs_ready_for_vlm": sum(1 for item in vlm_inputs if bool(item.get("ready_for_vlm"))),
        "inputs_skipped": inputs_skipped,
        "merge_summary": {
            "merged_candidate_count": sum(max(0, len(group["candidates"]) - 1) for group in merged_groups),
            "unmerged_candidate_count": sum(1 for group in merged_groups if len(group["candidates"]) == 1),
            "largest_group_size": max((len(group["candidates"]) for group in merged_groups), default=0),
            "groups": merge_groups_summary,
        },
        "priority_counts": dict(priority_counts),
        "warnings": global_warnings,
        "recommendation": "Proceed to Step 14 VLM Event Review / Qwen inference." if vlm_inputs else "No VLM-ready input was created; inspect selected candidates and frame availability.",
    }
    output_payload = {
        "status": "success",
        "source_file": "12_selected_top_event_candidates.json",
        "config": vlm_config,
        "summary": {
            "selected_candidates_loaded": len(selected_candidates[: int(vlm_config["max_inputs"])]),
            "merged_vlm_input_groups": len(merged_groups),
            "vlm_inputs_created": len(vlm_inputs),
            "temporal_strips_created": strips_created,
            "contact_sheets_created": contact_sheets_created,
            "inputs_ready_for_vlm": sum(1 for item in vlm_inputs if bool(item.get("ready_for_vlm"))),
            "inputs_skipped": inputs_skipped,
            "ready_for_step14_vlm_event_review": len(vlm_inputs) > 0,
        },
        "vlm_inputs": vlm_inputs,
    }

    write_json(run_dir / "13_vlm_event_inputs.json", output_payload)
    write_json_any(run_dir / "13_vlm_event_inputs_flat.json", flat_inputs)
    write_json(run_dir / "13_vlm_event_input_report.json", report_payload)
    return output_payload, flat_inputs, report_payload
