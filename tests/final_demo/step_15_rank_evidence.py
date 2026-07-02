from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


def _add_project_root_to_sys_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root


_add_project_root_to_sys_path()

from tests.final_demo.services.evidence_ranker import (  # noqa: E402
    build_evidence_ranking_outputs,
    update_run_manifest_for_evidence_ranking,
)
from tests.final_demo.services.video_io import write_json  # noqa: E402


def run_step_15_rank_evidence(
    run_dir: Path,
    *,
    debug_full: bool = False,
) -> dict[str, Any]:
    print("[final-demo] Starting Step 15: evidence ranking")

    step_result = build_evidence_ranking_outputs(run_dir, debug_full=debug_full)
    ranked_path = run_dir / "15_ranked_evidence.json"
    report_path = run_dir / "15_evidence_ranking_report.json"
    debug_path = run_dir / "15_ranked_evidence_debug.json"

    write_json(ranked_path, step_result["results_payload"])
    write_json(report_path, step_result["report_payload"])
    write_json(debug_path, step_result["debug_payload"])
    update_run_manifest_for_evidence_ranking(run_dir / "run_manifest.json")

    report_payload = step_result["report_payload"]
    print(f"[final-demo] Enriched records loaded: {report_payload['enriched_records_loaded']}")
    print(f"[final-demo] Ranked evidence count: {report_payload['global_ranked_evidence_count']}")
    print(f"[final-demo] Timeline groups created: {report_payload['timeline_groups_created']}")
    print(f"[final-demo] Top VLM candidates created: {report_payload['top_vlm_candidates_created']}")
    print(f"[final-demo] Records by ranking bucket: {report_payload['records_by_ranking_bucket']}")
    print(f"[final-demo] Ranked evidence path: {ranked_path}")
    print(f"[final-demo] Report path: {report_path}")
    print(f"[final-demo] Debug path: {debug_path}")

    return {
        "run_dir": run_dir,
        "ranked_path": ranked_path,
        "report_path": report_path,
        "debug_path": debug_path,
        "step_result": step_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final_demo Step 15 evidence ranking.")
    parser.add_argument("--run-dir", type=str, default=os.environ.get("FINAL_DEMO_RUN_DIR", "").strip())
    parser.add_argument("--debug-full", action="store_true")
    args = parser.parse_args()
    run_dir_value = str(args.run_dir or "").strip()
    if not run_dir_value:
        raise ValueError(
            "Provide --run-dir or set FINAL_DEMO_RUN_DIR before running Step 15 directly."
        )
    run_step_15_rank_evidence(
        Path(run_dir_value),
        debug_full=bool(args.debug_full),
    )


if __name__ == "__main__":
    main()
