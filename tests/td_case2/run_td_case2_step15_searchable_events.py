from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from run_td_case2_step01_02 import log
from stage_checks import build_failure_payload, update_stage_gate_report, write_json
from step_15_searchable_event_generation import run_searchable_event_generation


@dataclass(frozen=True)
class Step15Config:
    run_dir: Path


def read_config() -> Step15Config:
    raw_run_dir = os.environ.get("TD_CASE2_RUN_DIR", "").strip()
    if not raw_run_dir:
        raise ValueError("Environment variable TD_CASE2_RUN_DIR is required for Step 15.")
    run_dir = Path(raw_run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"TD_CASE2_RUN_DIR does not point to an existing directory: {run_dir}")
    for required_path in (
        run_dir / "11_full_scene_event_candidates.json",
        run_dir / "12_selected_top_event_candidates.json",
        run_dir / "14_vlm_event_reviews.json",
        run_dir / "14_final_video_summary.json",
    ):
        if not required_path.exists():
            raise FileNotFoundError(f"Required Step 15 input is missing: {required_path}")
    return Step15Config(run_dir=run_dir.resolve())


def _write_failed_reports(run_dir: Path, error_message: str) -> None:
    write_json(run_dir / "15_searchable_events.json", {"status": "failed", "records": [], "error_message": error_message})
    write_json(
        run_dir / "15_searchable_event_report.json",
        {"status": "failed", "event_visible_reviews": 0, "critical_event_count": 0, "error_message": error_message},
    )


def main() -> None:
    config = read_config()
    log(f"Run directory: {config.run_dir}")
    try:
        output_payload, report_payload, _flat_payload = run_searchable_event_generation(config.run_dir)
        update_stage_gate_report(
            config.run_dir,
            "15_searchable_event_generation",
            {
                "status": "success",
                "event_visible_reviews": output_payload["summary"]["event_visible_reviews"],
                "critical_event_count": output_payload["summary"]["critical_event_count"],
                "ready_for_step16_evidence_video": output_payload["summary"]["ready_for_step16_evidence_video"],
            },
        )
        log(f"Searchable reviewed events: {report_payload['event_visible_reviews']}")
        log(f"Critical events: {report_payload['critical_event_count']}")
        log(
            "Output paths: "
            f"{config.run_dir / '15_searchable_events.json'} | "
            f"{config.run_dir / '15_searchable_events_flat.json'} | "
            f"{config.run_dir / '15_searchable_event_report.json'}"
        )
    except Exception as exc:
        _write_failed_reports(config.run_dir, str(exc))
        update_stage_gate_report(config.run_dir, "15_searchable_event_generation", build_failure_payload(exc))
        log(f"Step 15 failed: {exc}")
        log(f"Run directory: {config.run_dir}")
        raise


if __name__ == "__main__":
    main()
