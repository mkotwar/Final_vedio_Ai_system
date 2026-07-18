from __future__ import annotations

from pathlib import Path
from typing import Any

from .plate_validation_schemas import FinalTrackAnprResult, PlateAgreementResult, PlateTextCandidate
from .serialization import read_json, read_jsonl, write_json, write_jsonl


REQUIRED_STEP8_INPUTS = (
    "07_anpr/florence_ocr_results.jsonl",
    "07_anpr/track_anpr_colour_results.jsonl",
    "07_5_plate_diagnostics/plate_diagnostic_attempts.jsonl",
    "07_5_plate_diagnostics/raw_plate_box_diagnostics.jsonl",
    "07_5_plate_diagnostics/track_plate_diagnostic_results.jsonl",
    "06_selected_crops/selected_track_crop_sets.jsonl",
    "reports/full_video_anpr_summary.json",
)


def validate_required_step8_inputs(run_dir: str | Path) -> list[Path]:
    root = Path(run_dir)
    missing = [root / relative for relative in REQUIRED_STEP8_INPUTS if not (root / relative).exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required Step 8 artifact(s):\n{missing_text}")
    return [root / relative for relative in REQUIRED_STEP8_INPUTS]


def load_step8_inputs(run_dir: str | Path) -> dict[str, Any]:
    validate_required_step8_inputs(run_dir)
    root = Path(run_dir)
    return {
        "ocr_results": read_jsonl(root / "07_anpr" / "florence_ocr_results.jsonl"),
        "florence_colour_results": read_jsonl(root / "07_anpr" / "florence_colour_results.jsonl")
        if (root / "07_anpr" / "florence_colour_results.jsonl").exists()
        else [],
        "track_anpr_colour_results": read_jsonl(root / "07_anpr" / "track_anpr_colour_results.jsonl"),
        "plate_diagnostic_attempts": read_jsonl(root / "07_5_plate_diagnostics" / "plate_diagnostic_attempts.jsonl"),
        "raw_plate_box_diagnostics": read_jsonl(root / "07_5_plate_diagnostics" / "raw_plate_box_diagnostics.jsonl"),
        "track_plate_diagnostic_results": read_jsonl(root / "07_5_plate_diagnostics" / "track_plate_diagnostic_results.jsonl"),
        "selected_track_crop_sets": read_jsonl(root / "06_selected_crops" / "selected_track_crop_sets.jsonl"),
        "full_video_anpr_summary": read_json(root / "reports" / "full_video_anpr_summary.json"),
    }


class PlateValidationArtifactSink:
    def __init__(self, run_dir: str | Path, output_dir: str | Path | None = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else Path(run_dir) / "08_plate_validation"
        self.report_dir = self.output_dir / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        candidates: list[PlateTextCandidate],
        agreements: list[PlateAgreementResult],
        finals: list[FinalTrackAnprResult],
        summary: dict[str, Any],
        report: dict[str, Any],
    ) -> dict[str, str]:
        paths = {
            "plate_text_candidates": self.output_dir / "plate_text_candidates.jsonl",
            "plate_agreement_results": self.output_dir / "plate_agreement_results.jsonl",
            "final_track_anpr_results": self.output_dir / "final_track_anpr_results.jsonl",
            "verified_plate_results": self.output_dir / "verified_plate_results.jsonl",
            "weak_plate_results": self.output_dir / "weak_plate_results.jsonl",
            "invalid_plate_results": self.output_dir / "invalid_plate_results.jsonl",
            "summary": self.report_dir / "step8_plate_validation_summary.json",
            "report": self.report_dir / "step8_plate_validation_report.json",
        }
        write_jsonl(paths["plate_text_candidates"], candidates)
        write_jsonl(paths["plate_agreement_results"], agreements)
        write_jsonl(paths["final_track_anpr_results"], finals)
        write_jsonl(paths["verified_plate_results"], [item for item in finals if item.plate_status == "verified"])
        write_jsonl(paths["weak_plate_results"], [item for item in finals if item.plate_status == "weak"])
        write_jsonl(paths["invalid_plate_results"], [item for item in finals if item.plate_status == "invalid"])
        write_json(paths["summary"], summary)
        write_json(paths["report"], report)
        return {key: str(value) for key, value in paths.items()}
