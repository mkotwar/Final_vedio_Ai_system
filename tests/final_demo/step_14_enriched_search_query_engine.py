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

from tests.final_demo.services.enriched_search_query_engine import (  # noqa: E402
    DEFAULT_ENRICHED_DEMO_QUERIES,
    build_enriched_search_query_outputs,
    update_run_manifest_for_enriched_search_query_engine,
)
from tests.final_demo.services.video_io import write_json  # noqa: E402


def run_step_14_enriched_search_query_engine(
    run_dir: Path,
    queries: list[str] | None = None,
    *,
    debug_full: bool = False,
) -> dict[str, Any]:
    print("[final-demo] Starting Step 14: enriched search query engine")

    selected_queries = list(queries or DEFAULT_ENRICHED_DEMO_QUERIES)
    step_result = build_enriched_search_query_outputs(run_dir, selected_queries, debug_full=debug_full)
    results_path = run_dir / "14_enriched_search_results.json"
    report_path = run_dir / "14_enriched_search_query_report.json"
    debug_path = run_dir / "14_enriched_search_debug_matches.json"

    write_json(results_path, step_result["results_payload"])
    write_json(report_path, step_result["report_payload"])
    write_json(debug_path, step_result["debug_payload"])
    update_run_manifest_for_enriched_search_query_engine(run_dir / "run_manifest.json")

    report_payload = step_result["report_payload"]
    print(f"[final-demo] Loaded enriched records: {report_payload['index_records_loaded']}")
    print(f"[final-demo] Queries run: {report_payload['queries_run']}")
    print(f"[final-demo] Grouped result count: {report_payload['total_grouped_results']}")
    print(f"[final-demo] Results by strength: {report_payload['results_by_strength']}")
    print(f"[final-demo] Results path: {results_path}")
    print(f"[final-demo] Report path: {report_path}")
    print(f"[final-demo] Debug path: {debug_path}")

    return {
        "run_dir": run_dir,
        "results_path": results_path,
        "report_path": report_path,
        "debug_path": debug_path,
        "step_result": step_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final_demo Step 14 enriched search query engine.")
    parser.add_argument("--run-dir", type=str, default=os.environ.get("FINAL_DEMO_RUN_DIR", "").strip())
    parser.add_argument("--query", action="append", default=None)
    parser.add_argument("--debug-full", action="store_true")
    args = parser.parse_args()
    run_dir_value = str(args.run_dir or "").strip()
    if not run_dir_value:
        raise ValueError(
            "Provide --run-dir or set FINAL_DEMO_RUN_DIR before running Step 14 directly."
        )
    run_step_14_enriched_search_query_engine(
        Path(run_dir_value),
        queries=list(args.query or []) or None,
        debug_full=bool(args.debug_full),
    )


if __name__ == "__main__":
    main()
