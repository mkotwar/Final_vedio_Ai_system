from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.final_demo.services.attribute_extractor import (
    build_attribute_outputs,
    update_run_manifest_for_attributes,
)
from tests.final_demo.services.video_io import write_json


def run_step_06_extract_attributes(run_dir: Path) -> dict[str, Any]:
    print("[final-demo] Starting Step 6: track attribute extraction")

    attribute_result = build_attribute_outputs(run_dir)
    attributes_path = run_dir / "06_track_attributes.json"
    report_path = run_dir / "06_attribute_report.json"

    write_json(attributes_path, attribute_result["attributes_payload"])
    write_json(report_path, attribute_result["report_payload"])
    update_run_manifest_for_attributes(
        run_dir / "run_manifest.json",
        list(attribute_result["attributes_payload"].get("attributes") or []),
    )

    report_payload = attribute_result["report_payload"]
    print(f"[final-demo] Selected track source: {report_payload['selected_track_source']}")
    print(f"[final-demo] Total tracks input: {report_payload['total_tracks_input']}")
    print(f"[final-demo] Attributes created: {report_payload['total_attributes_created']}")
    print(f"[final-demo] Attributes by class: {report_payload['attributes_by_class']}")
    print(
        f"[final-demo] Summary-ready track count: "
        f"{report_payload['summary_ready_track_count']}"
    )
    print(f"[final-demo] Review track count: {report_payload['review_track_count']}")
    print(f"[final-demo] Noise track count: {report_payload['noise_track_count']}")
    print(f"[final-demo] Crop output paths: {report_payload['crop_output_dirs']}")
    print(f"[final-demo] Attributes path: {attributes_path}")
    print(f"[final-demo] Report path: {report_path}")

    return {
        "run_dir": run_dir,
        "attributes_path": attributes_path,
        "report_path": report_path,
        "attribute_result": attribute_result,
    }


def main() -> None:
    raise RuntimeError(
        "Run Step 6 from the final demo pipeline or call run_step_06_extract_attributes(run_dir)."
    )


if __name__ == "__main__":
    main()
