from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.final_demo.services.track_cleanup import (
    build_track_cleanup_outputs,
    update_run_manifest_for_track_cleanup,
)
from tests.final_demo.services.video_io import write_json


def run_step_05B_track_cleanup(run_dir: Path) -> dict[str, Any]:
    print("[final-demo] Starting Step 5B: tracklet cleanup / strong fragment merge")

    cleanup_result = build_track_cleanup_outputs(run_dir)
    clean_tracks_path = run_dir / "05B_clean_tracks.json"
    cleanup_report_path = run_dir / "05B_track_cleanup_report.json"
    mapping_path = run_dir / "05B_track_id_mapping.json"

    write_json(clean_tracks_path, cleanup_result["clean_tracks_payload"])
    write_json(cleanup_report_path, cleanup_result["report_payload"])
    write_json(mapping_path, cleanup_result["mapping_payload"])
    update_run_manifest_for_track_cleanup(run_dir / "run_manifest.json")

    report_payload = cleanup_result["report_payload"]
    print(
        f"[final-demo] Selected input track source: "
        f"{report_payload['selected_input_track_source']}"
    )
    print(f"[final-demo] Input tracks: {report_payload['input_track_count']}")
    print(f"[final-demo] Clean tracks: {report_payload['clean_track_count']}")
    print(
        f"[final-demo] Strong clean tracks: {report_payload['strong_clean_track_count']}"
    )
    print(f"[final-demo] Review tracks: {report_payload['review_track_count']}")
    print(f"[final-demo] Tracks merged: {report_payload['tracks_merged_count']}")
    print(f"[final-demo] Tracks removed as noise: {report_payload['tracks_removed_as_noise']}")
    print(
        f"[final-demo] Cleanup parameter profile: "
        f"{report_payload['cleanup_parameter_profile']}"
    )
    if report_payload["raw_person_tracks_input"] > 0:
        print(f"[final-demo] Person clean tracks: {report_payload['clean_person_tracks']}")
        print(
            "[final-demo] Max people visible at once: "
            f"{report_payload['max_people_visible_at_once']}"
        )
        print(f"[final-demo] Cleanup status: {report_payload['cleanup_status']}")
    print(f"[final-demo] Clean tracks path: {clean_tracks_path}")
    print(f"[final-demo] Cleanup report path: {cleanup_report_path}")
    print(f"[final-demo] Track mapping path: {mapping_path}")

    return {
        "run_dir": run_dir,
        "clean_tracks_path": clean_tracks_path,
        "cleanup_report_path": cleanup_report_path,
        "mapping_path": mapping_path,
        "cleanup_result": cleanup_result,
    }


def main() -> None:
    raise RuntimeError(
        "Run Step 5B from the final demo pipeline or call run_step_05B_track_cleanup(run_dir)."
    )


if __name__ == "__main__":
    main()
