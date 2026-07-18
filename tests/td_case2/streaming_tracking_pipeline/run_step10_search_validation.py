from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .search_metrics import build_search_metrics
from .search_result_artifacts import SearchResultArtifactSink
from .serialization import read_json, write_json
from .structured_search_index import STEP10_INPUT_RELATIVE_PATH, StructuredVehicleSearchIndex


DEFAULT_RUN_DIR = Path("debug_runs") / "streaming_tracking_anpr_10fps_anpr_test_5min_20260718_163012"
DEFAULT_VALIDATION_QUERIES = [
    "white car",
    "verified plates",
    "UP81CH4158",
    "UP81",
    "red vehicle",
    "vehicles between 60 and 120 seconds",
    "weak OCR",
    "no plate",
    "motorcycle without plate",
    "white car between 2 and 3 minutes",
    "truck with verified plate",
]
EXPECTED_COUNTS = {
    "white car": 27,
    "verified plates": 58,
    "UP81CH4158": 1,
    "UP81": 2,
    "red vehicle": 17,
    "vehicles between 60 and 120 seconds": 16,
    "weak OCR": 28,
    "no plate": 67,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step 10 structured search validation over Step 9 vehicle records.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR), help="Run directory containing 09_searchable_objects artifacts.")
    parser.add_argument("--query", help="Single query to execute and print as a compact table.")
    parser.add_argument("--queries-file", help="Optional JSON or text file containing validation queries.")
    parser.add_argument("--top-k", type=int, default=25, help="Maximum results retained per query in artifacts/table.")
    parser.add_argument("--include-weak-plates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", help="Optional output directory. Defaults to <run-dir>/10_structured_search.")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    input_path = run_dir / STEP10_INPUT_RELATIVE_PATH
    if not input_path.exists():
        raise FileNotFoundError(f"Missing Step 10 input artifact: {input_path}")

    index = StructuredVehicleSearchIndex.from_run_dir(run_dir, include_weak_plates=args.include_weak_plates)
    queries = [args.query] if args.query else _load_queries(args.queries_file)
    responses = [index.search(query, top_k=args.top_k) for query in queries]
    metrics = build_search_metrics(responses, records_indexed=len(index.records))
    validation_counts = {response.query.raw_query: response.total_matches for response in responses}
    mismatches = _expected_count_mismatches(validation_counts)
    index_summary = index.summary()
    summary: dict[str, Any] = {
        **metrics,
        "run_dir": str(run_dir),
        "input_artifact": str(input_path),
        "output_dir": str(Path(args.output_dir) if args.output_dir else run_dir / "10_structured_search"),
        "validation_counts": validation_counts,
        "expected_count_mismatches": mismatches,
        "include_weak_plates": args.include_weak_plates,
        "top_k": args.top_k,
    }
    report = {
        "summary": summary,
        "index_summary": index_summary,
        "responses": [response.to_dict() for response in responses],
        "query_parser_rules": [
            "class and colour aliases are parsed from whole words",
            "verified/weak/no-plate/invalid phrases become plate status filters",
            "between X and Y seconds/minutes becomes an overlap time filter",
            "alphanumeric plate-like token length >= 6 becomes exact plate search",
            "short alphanumeric plate-like token becomes plate prefix search",
            "unknown words remain free-text tokens",
        ],
    }
    paths = SearchResultArtifactSink(run_dir, args.output_dir).write(
        index_summary=index_summary,
        responses=responses,
        summary=summary,
        report=report,
    )
    write_json(Path(paths["summary"]).with_name("step10_search_paths.json"), paths)

    if args.query:
        _print_compact_table(responses[0])
    else:
        print("Step 10 structured search validation complete")
        print(f"run_dir={run_dir}")
        print(f"records_indexed={len(index.records)}")
        print(f"queries_executed={len(responses)}")
        print(f"summary={paths['summary']}")
    return 0


def _load_queries(queries_file: str | None) -> list[str]:
    if not queries_file:
        return list(DEFAULT_VALIDATION_QUERIES)
    path = Path(queries_file)
    if path.suffix.lower() == ".json":
        payload = read_json(path)
        if isinstance(payload, list):
            return [str(item) for item in payload]
        if isinstance(payload, dict) and isinstance(payload.get("queries"), list):
            return [str(item) for item in payload["queries"]]
        raise ValueError("JSON queries file must be a list or contain a 'queries' list.")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _expected_count_mismatches(validation_counts: dict[str, int]) -> dict[str, dict[str, int | str]]:
    mismatches: dict[str, dict[str, int | str]] = {}
    for query, expected in EXPECTED_COUNTS.items():
        if query not in validation_counts:
            continue
        actual = validation_counts[query]
        if actual != expected:
            mismatches[query] = {
                "expected": expected,
                "actual": actual,
                "reason": "Step 10 uses structured filters plus overlap-aware time matching.",
            }
    return mismatches


def _print_compact_table(response: Any) -> None:
    print(f"query={response.query.raw_query}")
    print(f"matches={response.total_matches} searched={response.total_records_searched} runtime_sec={response.runtime_sec}")
    print("rank score record_id class colour plate_status plate_text time vehicle_crop plate_crop")
    for result in response.results:
        time_range = f"{result.first_seen_sec}-{result.last_seen_sec}"
        print(
            f"{result.rank} {result.score:.1f} {result.record_id} {result.object_class} {result.colour} "
            f"{result.plate_status} {result.plate_text} {time_range} "
            f"{result.representative_vehicle_crop_path} {result.representative_plate_crop_path}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
