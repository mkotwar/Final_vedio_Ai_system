from __future__ import annotations

import os
from pathlib import Path

from config import ENV_RUN_DIR
from run_td_case2_step01_02 import log
from run_td_case2_step01_02_02a import main as step01_main
from run_td_case2_step03_yolo import main as step03_main
from run_td_case2_step04a_florence_audit import main as step04a_main
from run_td_case2_step04b_tracking import main as step04b_main
from run_td_case2_step05_best_frames import main as step05_main
from run_td_case2_step06_ocr_color import main as step06_main
from run_td_case2_step07b_traffic_object_search_index import main as step07b_main
from run_td_case2_step08b_dynamic_search_validation import main as step08b_main
from run_td_case2_step09b_universal_search_cards import main as step09b_main
from run_td_case2_step10b_universal_search_demo import main as step10b_main
from stage_checks import write_json


def _write_pipeline_status(run_dir: Path, latest_completed_step: str, object_search_status: str, error_message: str | None = None) -> None:
    existing = {}
    status_path = run_dir / "pipeline_status.json"
    if status_path.exists():
        try:
            import json

            existing = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    existing.update(
        {
            "object_search_status": object_search_status,
            "latest_completed_step": latest_completed_step,
            "error_message": error_message,
        }
    )
    write_json(status_path, existing)


def main() -> None:
    step01_main()
    raw_run_dir = os.environ.get(ENV_RUN_DIR, "").strip()
    if not raw_run_dir:
        raise ValueError(f"Environment variable {ENV_RUN_DIR} is required after Step 01.")
    run_dir = Path(raw_run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = run_dir.resolve()
    log(f"Search-ready run directory: {run_dir}")
    _write_pipeline_status(run_dir, "step01_02_02a", "running")
    try:
        step03_main()
        _write_pipeline_status(run_dir, "step03", "running")
        step04a_main()
        _write_pipeline_status(run_dir, "step04a", "running")
        step04b_main()
        _write_pipeline_status(run_dir, "step04b", "running")
        step05_main()
        _write_pipeline_status(run_dir, "step05", "running")
        step06_main()
        _write_pipeline_status(run_dir, "step06", "running")
        step07b_main()
        _write_pipeline_status(run_dir, "step07b", "running")
        step08b_main()
        _write_pipeline_status(run_dir, "step08b", "running")
        step09b_main()
        _write_pipeline_status(run_dir, "step09b", "running")
        step10b_main()
        _write_pipeline_status(run_dir, "step10b", "ready")
        log(f"Search-ready pipeline complete. TD_CASE2_RUN_DIR={run_dir}")
    except Exception as exc:
        _write_pipeline_status(run_dir, "failed", "failed", error_message=str(exc))
        raise


if __name__ == "__main__":
    main()
