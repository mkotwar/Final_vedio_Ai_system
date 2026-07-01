from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.final_demo.services.tracking_quality import (
    build_tracking_quality_report,
    update_run_manifest_for_tracking_quality,
)
from tests.final_demo.services.video_io import write_json


def run_step_05A_tracking_quality(run_dir: Path) -> dict[str, Any]:
    print("[final-demo] Starting Step 5A: tracking quality QA")

    qa_report = build_tracking_quality_report(run_dir)
    report_path = run_dir / "05A_tracking_quality_report.json"
    write_json(report_path, qa_report)
    update_run_manifest_for_tracking_quality(run_dir / "run_manifest.json")

    person_quality = next(
        (item for item in qa_report["quality_by_class"] if str(item["class_name"]) == "person"),
        None,
    )

    print(f"[final-demo] Overall status: {qa_report['overall_status']}")
    print(f"[final-demo] Main problem: {', '.join(qa_report['main_problem']) or 'none'}")
    if person_quality is not None:
        print(f"[final-demo] Person raw tracks: {person_quality['raw_person_tracks']}")
        print(f"[final-demo] Person clean tracks: {person_quality['clean_person_tracks']}")
        print(
            "[final-demo] Max people visible at once: "
            f"{person_quality['max_people_visible_at_once']}"
        )
    print(f"[final-demo] Output path: {report_path}")

    return {
        "run_dir": run_dir,
        "report_path": report_path,
        "qa_report": qa_report,
    }


def main() -> None:
    raise RuntimeError(
        "Run Step 5A from the final demo pipeline or call run_step_05A_tracking_quality(run_dir)."
    )


if __name__ == "__main__":
    main()
