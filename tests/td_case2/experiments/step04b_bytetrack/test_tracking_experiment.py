from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.step04b_bytetrack.bytetrack_adapter import run_bytetrack_tracking
from experiments.step04b_bytetrack import fragment_merger
from experiments.step04b_bytetrack.fragment_merger import merge_track_fragments
from experiments.step04b_bytetrack.tracking_metrics import build_step05_compatible_tracks, validate_step05_compatibility


def _frame(frame_idx: int, timestamp_seconds: float) -> dict[str, object]:
    return {
        "frame_id": f"frame_{frame_idx:06d}",
        "frame_idx": frame_idx,
        "timestamp_seconds": timestamp_seconds,
        "timestamp_text": f"{timestamp_seconds:.3f}s",
        "image_path": f"frames/frame_{frame_idx:06d}.jpg",
    }


def _det(frame_idx: int, timestamp_seconds: float, x1: float, y1: float, x2: float, y2: float, *, detection_id: str, class_name: str = "car", confidence: float = 0.8) -> dict[str, object]:
    return {
        "frame_id": f"frame_{frame_idx:06d}",
        "frame_idx": frame_idx,
        "timestamp_seconds": timestamp_seconds,
        "timestamp_text": f"{timestamp_seconds:.3f}s",
        "image_path": f"frames/frame_{frame_idx:06d}.jpg",
        "detection_id": detection_id,
        "class_id": 2 if class_name == "car" else 0,
        "class_name": class_name,
        "confidence": confidence,
        "bbox_xyxy": [x1, y1, x2, y2],
        "bbox_area_ratio": 0.02,
        "crop_path": "",
        "crop_exists": False,
    }


def _track_synthetic(frames: list[dict[str, object]], detections_by_frame: dict[str, list[dict[str, object]]]):
    return run_bytetrack_tracking(
        frame_items=frames,
        detections_by_frame_id=detections_by_frame,
        image_width=1280,
        image_height=720,
        tracking_fps=5.0,
        track_buffer_seconds=2.0,
        high_confidence=0.25,
        low_confidence=0.10,
        match_threshold=0.80,
    )


@contextmanager
def _patched_appearance(value: float | None):
    original = fragment_merger._appearance_similarity
    fragment_merger._appearance_similarity = lambda run_dir, source, target: value
    try:
        yield
    finally:
        fragment_merger._appearance_similarity = original


def _raw_track(track_id: str, detections: list[dict[str, object]]) -> dict[str, object]:
    return {
        "track_id": track_id,
        "track_type": "vehicle",
        "source": "bytetrack_raw",
        "detections": detections,
    }


def test_one_vehicle_with_missed_detections():
    frames = [_frame(index, index * 0.2) for index in range(8)]
    detections = {
        frames[0]["frame_id"]: [_det(0, 0.0, 100, 100, 180, 160, detection_id="d0")],
        frames[1]["frame_id"]: [_det(1, 0.2, 118, 102, 198, 162, detection_id="d1")],
        frames[3]["frame_id"]: [_det(3, 0.6, 154, 106, 236, 168, detection_id="d3")],
        frames[4]["frame_id"]: [_det(4, 0.8, 172, 108, 255, 171, detection_id="d4")],
        frames[6]["frame_id"]: [_det(6, 1.2, 210, 112, 294, 176, detection_id="d6")],
    }
    tracks, _meta = _track_synthetic(frames, detections)
    vehicle_tracks = [track for track in tracks if track["track_type"] == "vehicle"]
    assert len(vehicle_tracks) == 1
    assert len(vehicle_tracks[0]["detections"]) == 5


def test_temporary_occlusion_can_merge_or_refind(tmp_path: Path):
    raw_tracks = [
        _raw_track(
            "vehicle_track_0001",
            [
                _det(0, 0.0, 100, 100, 180, 160, detection_id="a0"),
                _det(1, 0.2, 118, 102, 198, 162, detection_id="a1"),
            ],
        ),
        _raw_track(
            "vehicle_track_0002",
            [
                _det(3, 0.6, 156, 105, 238, 168, detection_id="b0"),
                _det(4, 0.8, 173, 108, 255, 171, detection_id="b1"),
            ],
        ),
    ]
    merged, audit, meta = merge_track_fragments(run_dir=tmp_path, raw_tracks=raw_tracks, image_diagonal=1468.6)
    assert meta["post_merge_track_count"] <= 2
    assert audit


def test_two_nearby_vehicles_stay_separate():
    frames = [_frame(index, index * 0.2) for index in range(5)]
    detections = {}
    for index, frame in enumerate(frames):
        detections[frame["frame_id"]] = [
            _det(index, index * 0.2, 100 + (index * 20), 100, 180 + (index * 20), 160, detection_id=f"a{index}"),
            _det(index, index * 0.2, 240 + (index * 20), 110, 320 + (index * 20), 170, detection_id=f"b{index}"),
        ]
    tracks, _meta = _track_synthetic(frames, detections)
    vehicle_tracks = [track for track in tracks if track["track_type"] == "vehicle"]
    assert len(vehicle_tracks) == 2


def test_crossing_vehicles_do_not_unsafe_merge(tmp_path: Path):
    raw_tracks = [
        _raw_track(
            "vehicle_track_0001",
            [
                _det(0, 0.0, 100, 100, 180, 160, detection_id="a0"),
                _det(1, 0.2, 160, 110, 240, 170, detection_id="a1"),
            ],
        ),
        _raw_track(
            "vehicle_track_0002",
            [
                _det(2, 0.4, 240, 110, 320, 170, detection_id="b0"),
                _det(3, 0.6, 180, 100, 260, 160, detection_id="b1"),
            ],
        ),
    ]
    merged, _audit, meta = merge_track_fragments(run_dir=tmp_path, raw_tracks=raw_tracks, image_diagonal=1468.6, max_prediction_distance=0.05)
    assert meta["post_merge_track_count"] == 2
    assert len(merged) == 2


def test_overlapping_track_times_rejected(tmp_path: Path):
    raw_tracks = [
        _raw_track(
            "vehicle_track_0001",
            [_det(0, 0.0, 100, 100, 180, 160, detection_id="a0"), _det(1, 0.2, 120, 100, 200, 160, detection_id="a1")],
        ),
        _raw_track(
            "vehicle_track_0002",
            [_det(1, 0.1, 125, 100, 205, 160, detection_id="b0"), _det(2, 0.3, 145, 100, 225, 160, detection_id="b1")],
        ),
    ]
    merged, audit, meta = merge_track_fragments(run_dir=tmp_path, raw_tracks=raw_tracks, image_diagonal=1468.6)
    assert meta["post_merge_track_count"] == 2
    assert any(item["source_track_id"] == "vehicle_track_0001" for item in audit)
    assert len(merged) == 2


def test_similar_appearance_but_impossible_motion_rejected(tmp_path: Path):
    raw_tracks = [
        _raw_track(
            "vehicle_track_0001",
            [_det(0, 0.0, 100, 100, 180, 160, detection_id="a0"), _det(1, 0.2, 118, 102, 198, 162, detection_id="a1")],
        ),
        _raw_track(
            "vehicle_track_0002",
            [_det(3, 0.6, 900, 500, 980, 560, detection_id="b0"), _det(4, 0.8, 918, 502, 998, 562, detection_id="b1")],
        ),
    ]
    merged, _audit, meta = merge_track_fragments(run_dir=tmp_path, raw_tracks=raw_tracks, image_diagonal=1468.6, max_prediction_distance=0.08)
    assert meta["post_merge_track_count"] == 2
    assert len(merged) == 2


def test_curved_path_exception_merges_one_vehicle(tmp_path: Path):
    raw_tracks = [
        _raw_track(
            "vehicle_track_0001",
            [
                _det(0, 0.0, 555, 70, 645, 130, detection_id="a0"),
                _det(1, 0.2, 515, 85, 605, 145, detection_id="a1"),
                _det(2, 0.4, 495, 100, 585, 160, detection_id="a2"),
            ],
        ),
        _raw_track(
            "vehicle_track_0002",
            [
                _det(5, 1.4, 445, 145, 535, 205, detection_id="b0"),
                _det(6, 1.6, 505, 205, 605, 275, detection_id="b1"),
                _det(7, 1.8, 565, 265, 675, 345, detection_id="b2"),
            ],
        ),
    ]
    with _patched_appearance(0.86):
        merged, audit, meta = merge_track_fragments(run_dir=tmp_path, raw_tracks=raw_tracks, image_diagonal=1468.6)
    assert meta["post_merge_track_count"] == 1
    assert len(merged) == 1
    accepted = [item for item in audit if item["accepted"]]
    assert accepted
    assert accepted[0]["final_merge_reason"] == "accepted_curved_path_or_abrupt_turn_exception"
    assert accepted[0]["curved_path_or_abrupt_turn_exception_checks"]["mutual_best_match"] is True


def test_sharp_turn_exception_merges_one_vehicle(tmp_path: Path):
    raw_tracks = [
        _raw_track(
            "vehicle_track_0001",
            [
                _det(0, 0.0, 980, 220, 1080, 290, detection_id="a0"),
                _det(1, 0.2, 955, 225, 1055, 295, detection_id="a1"),
            ],
        ),
        _raw_track(
            "vehicle_track_0002",
            [
                _det(4, 0.8, 885, 245, 985, 315, detection_id="b0"),
                _det(5, 1.0, 940, 325, 1050, 405, detection_id="b1"),
            ],
        ),
    ]
    with _patched_appearance(0.83):
        merged, audit, meta = merge_track_fragments(run_dir=tmp_path, raw_tracks=raw_tracks, image_diagonal=1468.6)
    assert meta["post_merge_track_count"] == 1
    assert len(merged) == 1
    assert any(item["accepted"] and item["direction_mismatch"] for item in audit)


def test_two_opposite_direction_vehicles_remain_separate(tmp_path: Path):
    raw_tracks = [
        _raw_track(
            "vehicle_track_0001",
            [
                _det(0, 0.0, 100, 100, 180, 160, detection_id="a0"),
                _det(1, 0.2, 140, 104, 220, 164, detection_id="a1"),
            ],
        ),
        _raw_track(
            "vehicle_track_0002",
            [
                _det(4, 0.8, 142, 108, 222, 168, detection_id="b0"),
                _det(5, 1.0, 100, 112, 180, 172, detection_id="b1"),
            ],
        ),
    ]
    with _patched_appearance(0.91):
        merged, audit, meta = merge_track_fragments(run_dir=tmp_path, raw_tracks=raw_tracks, image_diagonal=1468.6)
    assert meta["post_merge_track_count"] == 2
    rejected = [item for item in audit if item["source_track_id"] == "vehicle_track_0001" and item["candidate_track_id"] == "vehicle_track_0002"]
    assert rejected
    assert rejected[0]["accepted"] is False


def test_similar_looking_nearby_vehicles_not_merged(tmp_path: Path):
    raw_tracks = [
        _raw_track(
            "vehicle_track_0001",
            [
                _det(0, 0.0, 600, 300, 690, 365, detection_id="a0"),
                _det(1, 0.2, 620, 306, 710, 371, detection_id="a1"),
            ],
        ),
        _raw_track(
            "vehicle_track_0002",
            [
                _det(5, 1.0, 705, 190, 795, 255, detection_id="b0"),
                _det(6, 1.2, 685, 150, 775, 215, detection_id="b1"),
            ],
        ),
    ]
    with _patched_appearance(0.89):
        merged, audit, meta = merge_track_fragments(
            run_dir=tmp_path,
            raw_tracks=raw_tracks,
            image_diagonal=1468.6,
            max_prediction_distance=0.06,
        )
    assert meta["post_merge_track_count"] == 2
    assert len(merged) == 2
    rejected = [item for item in audit if item["source_track_id"] == "vehicle_track_0001" and item["candidate_track_id"] == "vehicle_track_0002"]
    assert rejected
    assert rejected[0]["accepted"] is False


def test_competing_successor_candidates_block_exception_merge(tmp_path: Path):
    raw_tracks = [
        _raw_track(
            "vehicle_track_0001",
            [
                _det(0, 0.0, 600, 80, 690, 140, detection_id="a0"),
                _det(1, 0.2, 560, 88, 650, 150, detection_id="a1"),
                _det(2, 0.4, 520, 102, 615, 168, detection_id="a2"),
            ],
        ),
        _raw_track(
            "vehicle_track_0002",
            [
                _det(5, 1.4, 500, 190, 600, 270, detection_id="b0"),
                _det(6, 1.6, 520, 235, 630, 320, detection_id="b1"),
            ],
        ),
        _raw_track(
            "vehicle_track_0003",
            [
                _det(5, 1.4, 506, 194, 606, 274, detection_id="c0"),
                _det(6, 1.6, 526, 239, 636, 324, detection_id="c1"),
            ],
        ),
    ]
    with _patched_appearance(0.87):
        merged, audit, meta = merge_track_fragments(run_dir=tmp_path, raw_tracks=raw_tracks, image_diagonal=1468.6)
    assert meta["post_merge_track_count"] == 3
    assert len(merged) == 3
    rejected = [item for item in audit if item["source_track_id"] == "vehicle_track_0001" and item["candidate_track_id"] == "vehicle_track_0002"]
    assert rejected
    assert rejected[0]["curved_path_or_abrupt_turn_exception_checks"]["no_competing_successor"] is False
    assert rejected[0]["accepted"] is False


def test_output_compatibility():
    track_payload, _report = build_step05_compatible_tracks(
        run_dir=Path("."),
        tracks=[
            {
                "track_id": "vehicle_track_0001",
                "track_type": "vehicle",
                "source": "post_merge",
                "detections": [
                    _det(0, 0.0, 100, 100, 180, 160, detection_id="d0"),
                    _det(1, 0.2, 120, 100, 200, 160, detection_id="d1"),
                    _det(2, 0.4, 140, 100, 220, 160, detection_id="d2"),
                    _det(3, 0.6, 160, 100, 240, 160, detection_id="d3"),
                ],
            }
        ],
        image_width=1280,
        image_height=720,
        min_track_length=2,
    )
    compatibility = validate_step05_compatibility(track_payload)
    assert compatibility["compatible"] is True
