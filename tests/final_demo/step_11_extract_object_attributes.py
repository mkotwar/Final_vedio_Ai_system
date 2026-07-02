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

from tests.final_demo.services.object_attribute_extractor import (  # noqa: E402
    build_object_attribute_outputs,
    update_run_manifest_for_object_attributes,
)
from tests.final_demo.services.video_io import write_json  # noqa: E402


def run_step_11_extract_object_attributes(run_dir: Path) -> dict[str, Any]:
    print("[final-demo] Starting Step 11: generic object attribute extraction")

    step_result = build_object_attribute_outputs(run_dir)
    attributes_path = run_dir / "11_object_attributes.json"
    report_path = run_dir / "11_object_attribute_report.json"

    write_json(attributes_path, step_result["attributes_payload"])
    write_json(report_path, step_result["report_payload"])
    update_run_manifest_for_object_attributes(run_dir / "run_manifest.json")

    report_payload = step_result["report_payload"]
    print(f"[final-demo] Object records created: {report_payload['object_attribute_records']}")
    print(f"[final-demo] Records by object type: {report_payload['records_by_object_type']}")
    print(f"[final-demo] Records by color: {report_payload['records_by_color']}")
    print(f"[final-demo] Records needing review: {report_payload['records_needing_review']}")
    print(f"[final-demo] Attributes path: {attributes_path}")
    print(f"[final-demo] Report path: {report_path}")

    return {
        "run_dir": run_dir,
        "attributes_path": attributes_path,
        "report_path": report_path,
        "step_result": step_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final_demo Step 11 object attribute extraction.")
    parser.add_argument("--run-dir", type=str, default=os.environ.get("FINAL_DEMO_RUN_DIR", "").strip())
    args = parser.parse_args()
    run_dir_value = str(args.run_dir or "").strip()
    if not run_dir_value:
        raise ValueError(
            "Provide --run-dir or set FINAL_DEMO_RUN_DIR before running Step 11 directly."
        )
    run_step_11_extract_object_attributes(Path(run_dir_value))


if __name__ == "__main__":
    main()
