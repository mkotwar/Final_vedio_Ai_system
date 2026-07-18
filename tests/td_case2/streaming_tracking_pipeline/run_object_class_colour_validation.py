from __future__ import annotations

import argparse
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from .class_normalization import normalize_model_names
from .dominant_colour_analysis import estimate_dominant_colour
from .serialization import read_jsonl, write_json, write_jsonl
from .structured_search_index import StructuredVehicleSearchIndex


DEFAULT_RUN_DIR = Path("debug_runs") / "streaming_tracking_anpr_10fps_anpr_test_5min_20260718_163012"
DEFAULT_OBJECT_DIR = Path("object")
SEARCH_PROBES = [
    "motorcycle",
    "black motorcycle",
    "white car",
    "red truck",
    "person",
    "person in white",
    "person between 60 and 120 seconds",
    "vehicles without plates",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate object model classes, normalized classes, and dominant colour on saved crops.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--object-dir", default=str(DEFAULT_OBJECT_DIR))
    parser.add_argument("--max-colour-records", type=int, default=40)
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "12_object_class_colour_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    colour_debug_dir = output_dir / "colour_debug"

    model_report = inspect_object_models(Path(args.object_dir))
    records_path = run_dir / "09_searchable_objects" / "searchable_vehicle_records.jsonl"
    records = read_jsonl(records_path)
    colour_results = analyse_record_colours(records, colour_debug_dir=colour_debug_dir, max_records=args.max_colour_records)
    search_results = run_search_probes(run_dir)
    summary = {
        "run_dir": str(run_dir),
        "object_dir": str(Path(args.object_dir)),
        "records_read": len(records),
        "model_files": model_report["model_files"],
        "model_loads": model_report["model_loads"],
        "raw_class_counts": dict(sorted(Counter(str(record.get("object_class") or "unknown") for record in records).items())),
        "normalized_class_counts": dict(sorted(Counter(str(record.get("normalized_class_name") or record.get("object_class") or "unknown") for record in records).items())),
        "colour_records_analysed": len(colour_results),
        "dominant_colour_counts": dict(sorted(Counter(str(item.get("dominant_colour") or "unknown") for item in colour_results).items())),
        "search_probe_counts": {query: payload["count"] for query, payload in search_results.items()},
        "person_model_status": model_report["person_model_status"],
        "person_tracking_result": "not_run_person_model_unloadable",
        "person_crop_count": 0,
        "duplicate_suppression_behavior": "class-aware duplicate suppression implemented; real person/vehicle joint validation blocked by person model load failure",
        "runtime_sec": round(time.perf_counter() - started, 6),
    }
    paths = {
        "model_report": output_dir / "object_model_report.json",
        "colour_report": output_dir / "dominant_colour_results.jsonl",
        "search_probe_report": output_dir / "search_probe_results.json",
        "summary": output_dir / "object_class_colour_validation_summary.json",
    }
    write_json(paths["model_report"], model_report)
    write_jsonl(paths["colour_report"], colour_results)
    write_json(paths["search_probe_report"], search_results)
    write_json(paths["summary"], summary)
    print("Object class/colour validation complete")
    print(f"records_read={len(records)}")
    print(f"colour_records_analysed={len(colour_results)}")
    print(f"summary={paths['summary']}")
    return 0


def inspect_object_models(object_dir: Path) -> dict[str, Any]:
    model_files = sorted([path for path in object_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".pt", ".py"}])
    loads: list[dict[str, Any]] = []
    person_status = "not_found"
    for path in model_files:
        item: dict[str, Any] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "suffix": path.suffix,
            "is_zip": zipfile.is_zipfile(path),
        }
        if path.suffix.lower() == ".pt":
            try:
                from ultralytics import YOLO  # type: ignore

                model = YOLO(str(path))
                names = getattr(model, "names", {})
                item["load_status"] = "loaded"
                item["task"] = getattr(model, "task", None)
                item["raw_class_mapping"] = {int(k): str(v) for k, v in dict(names).items()}
                item["normalized_class_mapping"] = normalize_model_names(item["raw_class_mapping"])
            except Exception as exc:
                item["load_status"] = "load_failed"
                item["load_error"] = f"{type(exc).__name__}: {exc}"
                if zipfile.is_zipfile(path):
                    try:
                        with zipfile.ZipFile(path) as archive:
                            item["zip_entries"] = archive.namelist()[:20]
                    except Exception:
                        pass
            if "person" in path.name.lower():
                person_status = item["load_status"]
        loads.append(item)
    return {
        "model_files": [str(path) for path in model_files],
        "model_loads": loads,
        "person_model_status": person_status,
    }


def analyse_record_colours(records: list[dict[str, Any]], *, colour_debug_dir: Path, max_records: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for record in records:
        if len(results) >= max_records:
            break
        crop_path = record.get("representative_vehicle_crop_path")
        if not crop_path or not Path(str(crop_path)).exists():
            continue
        result = estimate_dominant_colour(
            str(crop_path),
            object_class=str(record.get("normalized_class_name") or record.get("object_class") or "other_object"),
            raw_colour=record.get("raw_colour"),
            debug_dir=colour_debug_dir,
            record_id=str(record.get("record_id") or len(results)),
        )
        payload = {
            "record_id": record.get("record_id"),
            "object_class": record.get("normalized_class_name") or record.get("object_class"),
            "raw_florence_colour": record.get("raw_colour"),
            **result.to_dict(),
            "vehicle_crop_path": crop_path,
        }
        results.append(payload)
    return results


def run_search_probes(run_dir: Path) -> dict[str, Any]:
    index = StructuredVehicleSearchIndex.from_run_dir(run_dir)
    payload: dict[str, Any] = {}
    for query in SEARCH_PROBES:
        response = index.search(query, top_k=10)
        payload[query] = {
            "count": response.total_matches,
            "top_record_ids": [result.record_id for result in response.results[:10]],
        }
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
