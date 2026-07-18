from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from .run_step10_search_validation import DEFAULT_VALIDATION_QUERIES
from .search_result_card_artifacts import SearchResultCardArtifactSink
from .search_result_card_builder import build_vehicle_result_card_package
from .search_result_card_metrics import build_result_card_metrics
from .serialization import read_json, read_jsonl, write_json


DEFAULT_RUN_DIR = Path("debug_runs") / "streaming_tracking_anpr_10fps_anpr_test_5min_20260718_163012"
STEP9_RECORDS_RELATIVE_PATH = Path("09_searchable_objects") / "searchable_vehicle_records.jsonl"
STEP10_RESULTS_RELATIVE_PATH = Path("10_structured_search") / "validation_search_results.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step 11 UI-ready result-card packaging over Step 10 search results.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR), help="Run directory containing Step 9 and Step 10 artifacts.")
    parser.add_argument("--query", help="Package one query already present in Step 10 validation results.")
    parser.add_argument("--top-k", type=int, default=20, help="Cards per query.")
    parser.add_argument("--queries-file", help="Optional JSON or text file containing query names to package from Step 10 artifacts.")
    parser.add_argument("--write-html-preview", action="store_true", help="Write a static HTML preview.")
    parser.add_argument("--output-dir", help="Optional output directory. Defaults to <run-dir>/11_result_cards.")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    step9_path = run_dir / STEP9_RECORDS_RELATIVE_PATH
    step10_path = run_dir / STEP10_RESULTS_RELATIVE_PATH
    if not step9_path.exists():
        raise FileNotFoundError(f"Missing Step 9 records artifact: {step9_path}")
    if not step10_path.exists():
        raise FileNotFoundError(f"Missing Step 10 results artifact: {step10_path}")

    started = time.perf_counter()
    records_by_id = {str(record.get("record_id") or ""): record for record in read_jsonl(step9_path)}
    responses = read_jsonl(step10_path)
    selected_queries = [args.query] if args.query else _load_queries(args.queries_file)
    selected_responses = _select_responses(responses, selected_queries)
    packages = [
        build_vehicle_result_card_package(response, records_by_id, top_k=args.top_k)
        for response in selected_responses
    ]
    packaging_runtime = time.perf_counter() - started
    summary = {
        **build_result_card_metrics(packages, packaging_runtime=packaging_runtime),
        "run_dir": str(run_dir),
        "step9_records_artifact": str(step9_path),
        "step10_results_artifact": str(step10_path),
        "output_dir": str(Path(args.output_dir) if args.output_dir else run_dir / "11_result_cards"),
        "top_k": args.top_k,
        "html_preview_requested": bool(args.write_html_preview),
    }
    report: dict[str, Any] = {
        "summary": summary,
        "packages": [package.to_dict() for package in packages],
        "card_building_rules": [
            "preserve Step 10 rank and score",
            "join Step 9 records by record_id for confidence and duration",
            "use representative_vehicle_crop_path as thumbnail_path",
            "use representative_plate_crop_path as secondary_image_path",
            "keep cards even when image evidence is missing",
            "do not display invalid OCR as a valid number plate",
        ],
    }
    paths = SearchResultCardArtifactSink(run_dir, args.output_dir).write(
        packages=packages,
        summary=summary,
        report=report,
        write_html_preview=args.write_html_preview,
    )
    write_json(Path(paths["summary"]).with_name("step11_result_card_paths.json"), paths)

    if args.query:
        _print_compact_cards(packages[0])
    else:
        print("Step 11 result-card packaging complete")
        print(f"run_dir={run_dir}")
        print(f"queries_packaged={summary['queries_packaged']}")
        print(f"cards_created={summary['cards_created']}")
        print(f"summary={paths['summary']}")
        if "html_preview" in paths:
            print(f"html_preview={paths['html_preview']}")
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


def _select_responses(responses: list[dict[str, Any]], query_names: list[str]) -> list[dict[str, Any]]:
    by_query = {str((response.get("query") or {}).get("raw_query") or ""): response for response in responses}
    missing = [query for query in query_names if query not in by_query]
    if missing:
        available = ", ".join(sorted(by_query))
        raise ValueError(
            "Requested query was not found in saved Step 10 results. "
            f"Missing: {missing}. Available queries: {available}"
        )
    return [by_query[query] for query in query_names]


def _print_compact_cards(package: Any) -> None:
    print(f"query={package.raw_query}")
    print(f"total_matches={package.total_matches} returned_cards={package.returned_cards} runtime_sec={package.runtime_sec}")
    print("rank score record_id title status time thumbnail plate_image")
    for card in package.cards:
        print(
            f"{card.rank} {card.search_score:.1f} {card.record_id} {card.title} {card.status_badge} "
            f"{card.time_label} {card.thumbnail_path} {card.secondary_image_path}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
