from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from run_td_case2_step01_02 import log
from stage_checks import build_failure_payload, read_json, update_stage_gate_report, write_json
from step_09_search_result_packaging import write_json_any


ENV_RUN_DIR = "TD_CASE2_RUN_DIR"


@dataclass(frozen=True)
class Step09BConfig:
    run_dir: Path


def read_config() -> Step09BConfig:
    raw_run_dir = os.environ.get(ENV_RUN_DIR, "").strip()
    if not raw_run_dir:
        raise ValueError(f"Environment variable {ENV_RUN_DIR} is required for Step 09B.")
    run_dir = Path(raw_run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"TD_CASE2_RUN_DIR does not point to an existing directory: {run_dir}")
    for required_path in [
        run_dir / "07B_traffic_object_search_index.json",
        run_dir / "07B_traffic_object_search_index_report.json",
        run_dir / "08B_dynamic_search_validation_results.json",
    ]:
        if not required_path.exists():
            raise FileNotFoundError(f"Required Step 09B input is missing: {required_path}")
    return Step09BConfig(run_dir=run_dir.resolve())


def _write_failed_reports(run_dir: Path, error_message: str) -> None:
    write_json(run_dir / "09B_universal_search_cards.json", {"status": "failed", "cards": [], "error_message": error_message})
    write_json_any(run_dir / "09B_universal_search_cards_flat.json", [])
    write_json(run_dir / "09B_universal_search_card_schema.json", {"status": "failed", "error_message": error_message})
    write_json(run_dir / "09B_universal_search_packaging_report.json", {"status": "failed", "error_message": error_message})


def _build_badges(record: dict) -> list[str]:
    badges: list[str] = []
    if record.get("verified_license_plate"):
        badges.append("verified_plate")
    elif record.get("possible_plate_text"):
        badges.append("possible_plate")
    else:
        badges.append("no_verified_plate")
    if record.get("verified_vehicle_color"):
        badges.append("verified_color")
    elif record.get("possible_vehicle_color"):
        badges.append("possible_color")
    if record.get("quality") == "fallback":
        badges.append("fallback_track")
    if record.get("source_type") == "detection":
        badges.append("single_detection")
    if record.get("color_warning") == "possible_tail_light_color_confusion":
        badges.append("tail_light_warning")
    if record.get("color_status") == "conflict":
        badges.append("color_conflict")
    return badges


def main() -> None:
    config = read_config()
    log(f"Run directory: {config.run_dir}")
    try:
        index_payload = read_json(config.run_dir / "07B_traffic_object_search_index.json")
        index_report = read_json(config.run_dir / "07B_traffic_object_search_index_report.json")
        records = list(index_payload.get("records", []))
        cards = []
        for index, record in enumerate(records, start=1):
            cards.append(
                {
                    "card_id": f"traffic_card_{index:06d}",
                    "title": f"{str(record.get('class_name', 'object')).title()} @ {record.get('timestamp_text', '-')}",
                    "subtitle": f"{record.get('object_type', 'object')} | {record.get('track_id') or record.get('detection_id') or 'no-id'}",
                    "timestamp_text": record.get("timestamp_text"),
                    "object_type": record.get("object_type"),
                    "class_name": record.get("class_name"),
                    "color": record.get("verified_vehicle_color"),
                    "possible_color": record.get("possible_vehicle_color"),
                    "color_status": record.get("color_status"),
                    "color_warning": record.get("color_warning"),
                    "plate": record.get("verified_license_plate"),
                    "possible_plate": record.get("possible_plate_text"),
                    "verified_plate_status": record.get("verified_plate_status"),
                    "plate_warning": record.get("plate_warning"),
                    "confidence": record.get("confidence"),
                    "quality": record.get("quality"),
                    "full_frame_path": record.get("full_frame_path"),
                    "crop_path": record.get("crop_path"),
                    "bbox_xyxy": record.get("bbox_xyxy"),
                    "track_id": record.get("track_id"),
                    "search_text": record.get("search_text"),
                    "searchable_tokens": record.get("searchable_tokens"),
                    "match_explanation": {},
                    "badges": _build_badges(record),
                }
            )
        payload = {
            "status": "success",
            "source_index_file": "07B_traffic_object_search_index.json",
            "summary": {"total_cards_created": len(cards)},
            "cards": cards,
        }
        schema = {
            "schema_name": "td_case2_universal_traffic_search_card",
            "schema_version": "v2",
            "field_definitions": {
                "card_id": "stable card identifier",
                "title": "main result title",
                "subtitle": "secondary object summary",
                "timestamp_text": "video timestamp label",
                "object_type": "vehicle/person/traffic_signal/other_road_object",
                "class_name": "raw detected class",
                "color": "trusted verified vehicle color only",
                "possible_color": "possible untrusted vehicle color for warning display only",
                "color_status": "verified/possible/unknown/conflict",
                "color_warning": "why color was downgraded or flagged",
                "plate": "trusted verified plate only",
                "possible_plate": "possible OCR candidate not trusted as plate",
                "verified_plate_status": "verified/possible/none/rejected",
                "plate_warning": "plate trust warning if any",
                "confidence": "best confidence for the object record",
                "quality": "primary/fallback/single_detection/low_quality",
                "full_frame_path": "full-scene image path",
                "crop_path": "crop image path if available",
                "bbox_xyxy": "best box coordinates",
                "track_id": "linked track id when available",
                "search_text": "combined search text",
                "searchable_tokens": "normalized searchable token list",
                "match_explanation": "query-time explanation placeholder for UI reuse",
                "badges": "UI badges",
            },
            "example_card": cards[0] if cards else {},
        }
        report = {
            "status": "success",
            "total_cards_created": len(cards),
            "cards_with_full_frame": sum(1 for item in cards if item.get("full_frame_path")),
            "cards_with_crop": sum(1 for item in cards if item.get("crop_path")),
            "badge_counts": dict(Counter(badge for item in cards for badge in list(item.get("badges", [])))),
            "records_with_verified_color": index_report.get("records_with_verified_color", 0),
            "records_with_possible_color": index_report.get("records_with_possible_color", 0),
            "records_with_unknown_color": index_report.get("records_with_unknown_color", 0),
            "records_with_verified_plate": index_report.get("records_with_verified_plate", 0),
            "records_with_possible_plate": index_report.get("records_with_possible_plate", 0),
            "rejected_plate_count": index_report.get("rejected_plate_count", 0),
            "color_conflict_count": index_report.get("color_conflict_count", 0),
            "tail_light_confusion_warning_count": index_report.get("tail_light_confusion_warning_count", 0),
            "image_path_readiness": index_report.get("image_path_ready", False),
            "recommendation": "Universal traffic search cards are ready for UI use.",
        }
        write_json(config.run_dir / "09B_universal_search_cards.json", payload)
        write_json_any(config.run_dir / "09B_universal_search_cards_flat.json", cards)
        write_json(config.run_dir / "09B_universal_search_card_schema.json", schema)
        write_json(config.run_dir / "09B_universal_search_packaging_report.json", report)
        update_stage_gate_report(
            config.run_dir,
            "09B_universal_search_cards",
            {
                "status": "success",
                "total_cards_created": len(cards),
                "cards_with_full_frame": report["cards_with_full_frame"],
                "cards_with_crop": report["cards_with_crop"],
                "records_with_verified_color": report["records_with_verified_color"],
                "records_with_verified_plate": report["records_with_verified_plate"],
                "ready_for_step10b_universal_search_demo": len(cards) > 0,
            },
        )
        log(f"Cards created: {len(cards)}")
    except Exception as exc:
        _write_failed_reports(config.run_dir, str(exc))
        update_stage_gate_report(config.run_dir, "09B_universal_search_cards", build_failure_payload(exc))
        log(f"Step 09B failed: {exc}")
        log(f"Run directory: {config.run_dir}")
        raise


if __name__ == "__main__":
    main()
