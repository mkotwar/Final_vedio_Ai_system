from __future__ import annotations

import logging
from typing import Any


NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "postgrest",
    "supabase",
    "supabase_py",
)


def configure_cli_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, str(level_name).upper(), logging.INFO),
        format="%(levelname)s | %(message)s",
    )
    for logger_name in NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def summarize_worker_report(report: dict[str, Any]) -> list[str]:
    run_id = str(report.get("run_id") or report.get("run_code") or "UNKNOWN")
    camera_codes = ", ".join(str(code) for code in report.get("camera_codes", []))
    persistence = report.get("persistence", {}) if isinstance(report.get("persistence"), dict) else {}
    errors = report.get("errors", [])
    return [
        f"Run complete: {run_id}",
        f"Cameras processed: {camera_codes or 'none'}",
        (
            "Frames="
            f"{int(report.get('total_frames_processed', 0) or 0)} "
            f"Detections={int(report.get('total_detections', 0) or 0)} "
            f"Observations={int(report.get('total_track_observations', 0) or 0)}"
        ),
        (
            "Tracks="
            f"{int(report.get('total_completed_tracks', 0) or 0)} completed, "
            f"{int(report.get('total_discarded_tracks', 0) or 0)} discarded"
        ),
        (
            "Persistence="
            f"{str(persistence.get('backend', 'disabled'))} "
            f"(enabled={bool(persistence.get('enabled', False))}, "
            f"dry_run={bool(persistence.get('dry_run', False))})"
        ),
        f"Pipeline errors: {len(errors) if isinstance(errors, list) else 0}",
    ]
