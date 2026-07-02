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

from tests.final_demo.services.video_io import write_json  # noqa: E402
from tests.final_demo.services.vlm_evidence_verifier import (  # noqa: E402
    build_vlm_verification_outputs,
    update_run_manifest_for_vlm_verification,
)


def run_step_16_vlm_evidence_verification(
    run_dir: Path,
    *,
    debug_full: bool = False,
) -> dict[str, Any]:
    print("[final-demo] Starting Step 16: VLM evidence verification")

    step_result = build_vlm_verification_outputs(run_dir, debug_full=debug_full)
    inputs_dir = run_dir / "16_vlm_inputs"
    verification_path = run_dir / "16_vlm_evidence_verification.json"
    report_path = run_dir / "16_vlm_verification_report.json"
    debug_path = run_dir / "16_vlm_verification_debug.json"

    write_json(verification_path, step_result["results_payload"])
    write_json(report_path, step_result["report_payload"])
    write_json(debug_path, step_result["debug_payload"])
    update_run_manifest_for_vlm_verification(run_dir / "run_manifest.json")

    report_payload = step_result["report_payload"]
    print(f"[final-demo] VLM enabled: {report_payload['vlm_enabled']}")
    print(f"[final-demo] Selected candidates: {report_payload['candidates_selected']}")
    print(f"[final-demo] VLM inputs created: {report_payload['vlm_inputs_created']}")
    print(f"[final-demo] Calls attempted: {report_payload['vlm_calls_attempted']}")
    print(f"[final-demo] Verification status counts: {report_payload['verifications_by_status']}")
    print(f"[final-demo] Inputs dir: {inputs_dir}")
    print(f"[final-demo] Verification path: {verification_path}")
    print(f"[final-demo] Report path: {report_path}")
    print(f"[final-demo] Debug path: {debug_path}")

    return {
        "run_dir": run_dir,
        "inputs_dir": inputs_dir,
        "verification_path": verification_path,
        "report_path": report_path,
        "debug_path": debug_path,
        "step_result": step_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final_demo Step 16 VLM evidence verification.")
    parser.add_argument("--run-dir", type=str, default=os.environ.get("FINAL_DEMO_RUN_DIR", "").strip())
    parser.add_argument("--debug-full", action="store_true")
    args = parser.parse_args()
    run_dir_value = str(args.run_dir or "").strip()
    if not run_dir_value:
        raise ValueError(
            "Provide --run-dir or set FINAL_DEMO_RUN_DIR before running Step 16 directly."
        )
    run_step_16_vlm_evidence_verification(
        Path(run_dir_value),
        debug_full=bool(args.debug_full),
    )


if __name__ == "__main__":
    main()
