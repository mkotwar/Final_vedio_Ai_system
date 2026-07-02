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

from tests.final_demo.services.enriched_search_indexer import (  # noqa: E402
    build_enriched_search_index_outputs,
    update_run_manifest_for_enriched_search_index,
)
from tests.final_demo.services.video_io import write_json  # noqa: E402


def run_step_13_build_enriched_search_index(run_dir: Path) -> dict[str, Any]:
    print("[final-demo] Starting Step 13: enriched search index")

    step_result = build_enriched_search_index_outputs(run_dir)
    index_path = run_dir / "13_enriched_search_index.json"
    report_path = run_dir / "13_enriched_search_index_report.json"
    smoke_path = run_dir / "13_enriched_search_smoke_test.json"

    write_json(index_path, step_result["index_payload"])
    write_json(report_path, step_result["report_payload"])
    write_json(smoke_path, step_result["smoke_payload"])
    update_run_manifest_for_enriched_search_index(run_dir / "run_manifest.json")

    report_payload = step_result["report_payload"]
    print(f"[final-demo] Base records loaded: {report_payload['base_records_loaded']}")
    print(f"[final-demo] Person attributes loaded: {report_payload['person_attribute_records_loaded']}")
    print(f"[final-demo] Object attributes loaded: {report_payload['object_attribute_records_loaded']}")
    print(f"[final-demo] Associations loaded: {report_payload['association_records_loaded']}")
    print(f"[final-demo] Enriched records created: {report_payload['enriched_records_created']}")
    print(f"[final-demo] Records by family: {report_payload['records_by_entity_family']}")
    print(f"[final-demo] Records needing review: {report_payload['records_needing_review']}")
    print(f"[final-demo] Index path: {index_path}")
    print(f"[final-demo] Report path: {report_path}")
    print(f"[final-demo] Smoke path: {smoke_path}")

    return {
        "run_dir": run_dir,
        "index_path": index_path,
        "report_path": report_path,
        "smoke_path": smoke_path,
        "step_result": step_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final_demo Step 13 enriched search index.")
    parser.add_argument("--run-dir", type=str, default=os.environ.get("FINAL_DEMO_RUN_DIR", "").strip())
    args = parser.parse_args()
    run_dir_value = str(args.run_dir or "").strip()
    if not run_dir_value:
        raise ValueError(
            "Provide --run-dir or set FINAL_DEMO_RUN_DIR before running Step 13 directly."
        )
    run_step_13_build_enriched_search_index(Path(run_dir_value))


if __name__ == "__main__":
    main()
