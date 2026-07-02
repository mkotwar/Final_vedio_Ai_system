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

from tests.final_demo.services.entity_association import (  # noqa: E402
    build_entity_association_outputs,
    update_run_manifest_for_entity_association,
)
from tests.final_demo.services.video_io import write_json  # noqa: E402


def run_step_12_associate_entities(run_dir: Path) -> dict[str, Any]:
    print("[final-demo] Starting Step 12: entity association")

    step_result = build_entity_association_outputs(run_dir)
    associations_path = run_dir / "12_entity_associations.json"
    report_path = run_dir / "12_entity_association_report.json"
    debug_path = run_dir / "12_entity_association_debug.json"

    write_json(associations_path, step_result["associations_payload"])
    write_json(report_path, step_result["report_payload"])
    write_json(debug_path, step_result["debug_payload"])
    update_run_manifest_for_entity_association(run_dir / "run_manifest.json")

    report_payload = step_result["report_payload"]
    print(f"[final-demo] Persons loaded: {report_payload['person_records_loaded']}")
    print(f"[final-demo] Objects loaded: {report_payload['object_records_loaded']}")
    print(f"[final-demo] Vehicles loaded: {report_payload['vehicle_records_loaded']}")
    print(f"[final-demo] Associations created: {report_payload['associations_created']}")
    print(f"[final-demo] Associations by relationship: {report_payload['associations_by_relationship']}")
    print(f"[final-demo] Associations needing review: {report_payload['associations_needing_review']}")
    print(f"[final-demo] Associations path: {associations_path}")
    print(f"[final-demo] Report path: {report_path}")
    print(f"[final-demo] Debug path: {debug_path}")

    return {
        "run_dir": run_dir,
        "associations_path": associations_path,
        "report_path": report_path,
        "debug_path": debug_path,
        "step_result": step_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final_demo Step 12 entity association.")
    parser.add_argument("--run-dir", type=str, default=os.environ.get("FINAL_DEMO_RUN_DIR", "").strip())
    args = parser.parse_args()
    run_dir_value = str(args.run_dir or "").strip()
    if not run_dir_value:
        raise ValueError(
            "Provide --run-dir or set FINAL_DEMO_RUN_DIR before running Step 12 directly."
        )
    run_step_12_associate_entities(Path(run_dir_value))


if __name__ == "__main__":
    main()
