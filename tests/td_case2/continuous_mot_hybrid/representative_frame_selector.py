from __future__ import annotations

from typing import Any

from tests.td_case2.hybrid_tracking_test.representative_frame_selector import build_representative_frames_v2


def build_representative_frames(
    *,
    video_path,
    local_objects: list[dict[str, Any]],
    frame_width: int,
    frame_height: int,
    run_dir,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    return build_representative_frames_v2(
        video_path=video_path,
        local_objects=local_objects,
        frame_width=frame_width,
        frame_height=frame_height,
        post_tracking_dir=run_dir / "07_representative_frames",
        maximum_ready_crop_clipping_ratio=0.25,
        maximum_fallback_crop_clipping_ratio=0.45,
        minimum_plate_candidate_score=0.45,
    )

