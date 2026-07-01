from __future__ import annotations

import sys
from pathlib import Path


def _add_project_root_to_sys_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root


def main() -> None:
    _add_project_root_to_sys_path()
    from tests.final_demo.services.video_io import ensure_final_demo_directories
    from tests.final_demo.step_01_video_ingest import run_step_01_video_ingest
    from tests.final_demo.step_02_chunk_planner import run_step_02_chunk_planner
    from tests.final_demo.step_03_sample_frames import run_step_03_sample_frames
    from tests.final_demo.step_04_yolo_detect import run_step_04_yolo_detect
    from tests.final_demo.step_05_track_objects import run_step_05_track_objects
    from tests.final_demo.step_05A_tracking_quality import (
        run_step_05A_tracking_quality,
    )
    from tests.final_demo.step_05B_track_cleanup import run_step_05B_track_cleanup
    from tests.final_demo.step_05C_class_propagation_audit import (
        run_step_05C_class_propagation_audit,
    )
    from tests.final_demo.step_06_extract_attributes import (
        run_step_06_extract_attributes,
    )
    from tests.final_demo.step_06A_detect_plate_candidates import (
        run_step_06A_detect_plate_candidates,
    )
    from tests.final_demo.step_07A_plate_ocr import run_step_07A_plate_ocr
    from tests.final_demo.step_07B_event_candidates import run_step_07B_event_candidates
    from tests.final_demo.step_07C_class_propagation_audit import (
        run_step_07C_class_propagation_audit,
    )

    ensure_final_demo_directories()
    step_01_result = run_step_01_video_ingest()
    step_02_result = run_step_02_chunk_planner(step_01_result["run_dir"])
    step_03_result = run_step_03_sample_frames(step_02_result["run_dir"])
    step_04_result = run_step_04_yolo_detect(step_03_result["run_dir"])
    step_05_result = run_step_05_track_objects(step_04_result["run_dir"])
    step_05a_result = run_step_05A_tracking_quality(step_05_result["run_dir"])
    step_05b_result = run_step_05B_track_cleanup(step_05a_result["run_dir"])
    step_05c_result = run_step_05C_class_propagation_audit(step_05b_result["run_dir"])
    step_06_result = run_step_06_extract_attributes(step_05c_result["run_dir"])
    step_06a_result = run_step_06A_detect_plate_candidates(step_06_result["run_dir"])
    step_07a_result = run_step_07A_plate_ocr(step_06a_result["run_dir"])
    step_07b_result = run_step_07B_event_candidates(step_07a_result["run_dir"])
    run_step_07C_class_propagation_audit(step_07b_result["run_dir"])


if __name__ == "__main__":
    main()
