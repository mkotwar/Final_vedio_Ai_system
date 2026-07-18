from __future__ import annotations

from pathlib import Path
from typing import Any

from .searchable_object_schemas import SearchableVehicleRecord
from .serialization import read_json, read_jsonl, write_json, write_jsonl


REQUIRED_STEP9_INPUTS = (
    "01_source/source_metadata.json",
    "04_lifecycle/completed_tracks.jsonl",
    "05_crops/completed_track_crop_bundles.jsonl",
    "06_selected_crops/selected_track_crop_sets.jsonl",
    "07_anpr/track_anpr_colour_results.jsonl",
    "08_plate_validation/final_track_anpr_results.jsonl",
)


def validate_required_step9_inputs(run_dir: str | Path) -> list[Path]:
    root = Path(run_dir)
    missing = [root / relative for relative in REQUIRED_STEP9_INPUTS if not (root / relative).exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required Step 9 artifact(s):\n{missing_text}")
    return [root / relative for relative in REQUIRED_STEP9_INPUTS]


def load_step9_inputs(run_dir: str | Path) -> dict[str, Any]:
    validate_required_step9_inputs(run_dir)
    root = Path(run_dir)
    return {
        "source_metadata": read_json(root / "01_source" / "source_metadata.json"),
        "completed_tracks": read_jsonl(root / "04_lifecycle" / "completed_tracks.jsonl"),
        "completed_track_crop_bundles": read_jsonl(root / "05_crops" / "completed_track_crop_bundles.jsonl"),
        "selected_track_crop_sets": read_jsonl(root / "06_selected_crops" / "selected_track_crop_sets.jsonl"),
        "track_anpr_colour_results": read_jsonl(root / "07_anpr" / "track_anpr_colour_results.jsonl"),
        "final_track_anpr_results": read_jsonl(root / "08_plate_validation" / "final_track_anpr_results.jsonl"),
    }


class SearchableObjectArtifactSink:
    def __init__(self, run_dir: str | Path, output_dir: str | Path | None = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else Path(run_dir) / "09_searchable_objects"
        self.report_dir = self.output_dir / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        records: list[SearchableVehicleRecord],
        summary: dict[str, Any],
        report: dict[str, Any],
        *,
        write_flat_json: bool = True,
    ) -> dict[str, str]:
        paths = {
            "searchable_vehicle_records": self.output_dir / "searchable_vehicle_records.jsonl",
            "searchable_vehicle_records_flat": self.output_dir / "searchable_vehicle_records_flat.json",
            "verified_plate_vehicle_records": self.output_dir / "verified_plate_vehicle_records.jsonl",
            "weak_plate_vehicle_records": self.output_dir / "weak_plate_vehicle_records.jsonl",
            "no_plate_vehicle_records": self.output_dir / "no_plate_vehicle_records.jsonl",
            "summary": self.report_dir / "step9_searchable_objects_summary.json",
            "report": self.report_dir / "step9_searchable_objects_report.json",
        }
        write_jsonl(paths["searchable_vehicle_records"], records)
        if write_flat_json:
            write_json(paths["searchable_vehicle_records_flat"], [record.to_dict() for record in records])
        write_jsonl(paths["verified_plate_vehicle_records"], [record for record in records if record.plate_status == "verified"])
        write_jsonl(paths["weak_plate_vehicle_records"], [record for record in records if record.plate_status == "weak"])
        write_jsonl(paths["no_plate_vehicle_records"], [record for record in records if record.plate_status == "no_plate_detected"])
        write_json(paths["summary"], summary)
        write_json(paths["report"], report)
        return {key: str(value) for key, value in paths.items()}
