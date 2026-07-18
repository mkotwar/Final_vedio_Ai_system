from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .searchable_object_artifacts import SearchableObjectArtifactSink, load_step9_inputs
from .searchable_object_metrics import build_searchable_object_metrics
from .class_normalization import normalize_class_name
from .searchable_object_records import build_searchable_object_record, identity, run_validation_queries
from .serialization import read_jsonl, to_json_safe


DEFAULT_RUN_DIR = "debug_runs/streaming_tracking_anpr_10fps_anpr_test_5min_20260718_163012"


def run(
    *,
    run_dir: str | Path = DEFAULT_RUN_DIR,
    output_dir: str | Path | None = None,
    include_weak_plate_text: bool = True,
    write_flat_json: bool = True,
) -> dict[str, Any]:
    inputs = load_step9_inputs(run_dir)
    source_metadata = inputs["source_metadata"]
    video_path = source_metadata.get("source_path")
    completed_tracks = inputs["completed_tracks"]
    selected_by_id = {identity(row): row for row in inputs["selected_track_crop_sets"]}
    final_by_id = {identity(row): row for row in inputs["final_track_anpr_results"]}
    clothing_by_id = _load_optional_person_clothing(run_dir)
    join_failures: list[str] = []
    records = []
    for lifecycle in sorted(completed_tracks, key=lambda row: identity(row)):
        key = identity(lifecycle)
        selected = selected_by_id.get(key)
        final = final_by_id.get(key)
        raw_class_name = lifecycle.get("last_class_name") or next(iter((lifecycle.get("class_votes") or {}) or []), None)
        object_group = normalize_class_name(raw_class_name).object_group
        if selected is None:
            join_failures.append(f"{key}:missing_selected_track_crop_set")
        if final is None and object_group != "person":
            join_failures.append(f"{key}:missing_step8_final_anpr")
        records.append(
            build_searchable_object_record(
                lifecycle,
                video_path=video_path,
                selected_crop_set=selected,
                final_anpr=final,
                person_clothing=clothing_by_id.get(key),
                include_weak_plate_text=include_weak_plate_text,
            )
        )
    queries = run_validation_queries(records)
    metrics = build_searchable_object_metrics(records, join_failures=join_failures, missing_input_artifacts=[])
    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir or Path(run_dir) / "09_searchable_objects"),
        "include_weak_plate_text": include_weak_plate_text,
        "write_flat_json": write_flat_json,
        **metrics,
        "validation_queries": queries,
        "input_artifact_counts": {
            "completed_tracks": len(completed_tracks),
            "completed_track_crop_bundles": len(inputs["completed_track_crop_bundles"]),
            "selected_track_crop_sets": len(inputs["selected_track_crop_sets"]),
            "track_anpr_colour_results": len(inputs["track_anpr_colour_results"]),
            "final_track_anpr_results": len(inputs["final_track_anpr_results"]),
        },
    }
    report = {
        "summary": summary,
        "validation_queries": queries,
        "example_records": [record.to_dict() for record in records[:10]],
        "records": [record.to_dict() for record in records],
    }
    paths = SearchableObjectArtifactSink(run_dir, output_dir).write(records, summary, report, write_flat_json=write_flat_json)
    return {"summary": summary, "report": report, "artifact_paths": paths, "records": records}


def _load_optional_person_clothing(run_dir: str | Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    path = Path(run_dir) / "07_person_clothing_colour" / "person_clothing_colour_results.jsonl"
    if not path.exists():
        return {}
    return {identity(row): row for row in read_jsonl(path)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 9: build searchable vehicle records from saved ANPR artifacts.")
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--include-weak-plate-text", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-flat-json", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        include_weak_plate_text=args.include_weak_plate_text,
        write_flat_json=args.write_flat_json,
    )
    summary = result["summary"]
    print("Step 9 searchable object records complete")
    print(f"run_dir={summary['run_dir']}")
    print(f"vehicle_records_created={summary['vehicle_records_created']}")
    print(f"verified_plate_records={summary['verified_plate_records']}")
    print(f"weak_plate_records={summary['weak_plate_records']}")
    print(f"no_plate_records={summary['no_plate_records']}")
    print(f"report={result['artifact_paths']['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
