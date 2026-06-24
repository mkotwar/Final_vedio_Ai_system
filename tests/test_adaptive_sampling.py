"""Unit and integration tests for Adaptive Frame Sampling.
"""

import json
import pytest
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings, PROJECT_ROOT
from app.services.frame import FrameExtractionService


@pytest.fixture(name="client")
def client_fixture():
    """Yields a test client with lifespan context active, forcing adaptive sampling on."""
    old_sampling = settings.ENABLE_ADAPTIVE_SAMPLING
    old_motion_windowing = settings.ENABLE_MOTION_WINDOWING
    settings.ENABLE_ADAPTIVE_SAMPLING = True
    settings.ENABLE_MOTION_WINDOWING = False
    try:
        with TestClient(app) as client:
            yield client
    finally:
        settings.ENABLE_ADAPTIVE_SAMPLING = old_sampling
        settings.ENABLE_MOTION_WINDOWING = old_motion_windowing


def create_dynamic_video(file_path: Path, duration_sec: int = 5, fps: int = 10) -> None:
    """Helper to generate a video file where some frames are identical and others change."""
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(file_path), fourcc, float(fps), (width, height))

    if not out.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for path: {file_path}")

    total_frames = duration_sec * fps
    for i in range(total_frames):
        # Create a frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Second 0 to 2 (first 20 frames): identical black frames
        # Second 2 to 5 (next 30 frames): white frames with changing text (significant scene change)
        if i >= 20:
            frame.fill(255)
            cv2.putText(
                frame,
                f"Change: {i}",
                (40, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2,
            )
        out.write(frame)

    out.release()


def test_compute_similarity_metrics():
    """Test individual metrics on identical and different frames."""
    # Create identical frames
    frame1 = np.zeros((240, 320, 3), dtype=np.uint8)
    frame2 = np.zeros((240, 320, 3), dtype=np.uint8)
    
    hist_diff, ssim_diff, motion_score = FrameExtractionService.compute_similarity_metrics(frame1, frame2)
    assert hist_diff == 0.0
    assert ssim_diff == 0.0
    assert motion_score == 0.0

    # Create completely different frames (black and white)
    frame3 = np.ones((240, 320, 3), dtype=np.uint8) * 255
    hist_diff2, ssim_diff2, motion_score2 = FrameExtractionService.compute_similarity_metrics(frame1, frame3)
    assert hist_diff2 > 0.5
    assert ssim_diff2 > 0.5
    assert motion_score2 > 0.5


def test_adaptive_sampling_integration(client: TestClient) -> None:
    """Verify that adaptive sampling reduces the frames sent to Qwen VLM and populates metrics."""
    video_id = "00000000-0000-0000-0000-000000000005"
    video_filename = f"{video_id}.mp4"
    video_path = settings.VIDEOS_DIR / video_filename

    create_dynamic_video(video_path, duration_sec=5, fps=10)

    # Write custom video metadata
    video_metadata = {
        "video_id": video_id,
        "filename": "adaptive_test.mp4",
        "upload_time": "2026-06-03T18:00:00Z",
        "file_size": video_path.stat().st_size,
    }
    metadata_path = settings.METADATA_DIR / f"{video_id}.json"
    with open(metadata_path, "w", encoding="utf-8") as meta_file:
        json.dump(video_metadata, meta_file)

    # Call endpoint with adaptive sampling enabled (via client_fixture)
    payload = {"video_id": video_id}
    response = client.post("/frames/extract", json=payload)

    assert response.status_code == 200
    data = response.json()
    
    assert data["video_id"] == video_id
    assert data["total_frames_extracted"] == 5
    
    # Frames 0, 1, 2 should be similar (first frame processed, next two skipped)
    # Frames 3, 4 should be different (scene changes, so sent)
    # Total sent should be less than 5
    assert data["frames_sent_to_qwen"] < 5
    assert data["frames_skipped"] > 0
    assert data["reduction_percent"] == round((data["frames_skipped"] / 5) * 100.0, 2)

    # Check report path
    report_path = PROJECT_ROOT / "ADAPTIVE_SAMPLING_REPORT.md"
    assert report_path.exists()
    
    report_content = report_path.read_text(encoding="utf-8")
    assert "ADAPTIVE SAMPLING REPORT" in report_content
    assert str(video_id) in report_content
    assert "Estimated Runtime Savings" in report_content


@patch("app.services.motion_window_service.MotionWindowService.detect_motion_windows")
def test_motion_windowing_keeps_sparse_baseline_outside_windows(mock_windows, client: TestClient) -> None:
    video_id = "00000000-0000-0000-0000-000000000006"
    video_filename = f"{video_id}.mp4"
    video_path = settings.VIDEOS_DIR / video_filename

    create_dynamic_video(video_path, duration_sec=12, fps=10)

    video_metadata = {
        "video_id": video_id,
        "filename": "baseline_test.mp4",
        "upload_time": "2026-06-24T12:00:00Z",
        "file_size": video_path.stat().st_size,
    }
    metadata_path = settings.METADATA_DIR / f"{video_id}.json"
    with open(metadata_path, "w", encoding="utf-8") as meta_file:
        json.dump(video_metadata, meta_file)

    old_motion_windowing = settings.ENABLE_MOTION_WINDOWING
    old_baseline = settings.OUT_OF_WINDOW_BASELINE_SECONDS
    old_max_gap = settings.MAX_FRAME_GAP_SECONDS
    settings.ENABLE_MOTION_WINDOWING = True
    settings.OUT_OF_WINDOW_BASELINE_SECONDS = 5.0
    settings.MAX_FRAME_GAP_SECONDS = 10.0
    mock_windows.return_value = [(2.0, 3.0)]
    try:
        response = client.post("/frames/extract", json={"video_id": video_id})
        assert response.status_code == 200
        data = response.json()

        timestamps = [frame["timestamp_seconds"] for frame in data["frames"]]
        assert 0.0 in timestamps
        assert 5.0 in timestamps
        assert 10.0 in timestamps
        assert any(ts in timestamps for ts in (2.0, 3.0))
    finally:
        settings.ENABLE_MOTION_WINDOWING = old_motion_windowing
        settings.OUT_OF_WINDOW_BASELINE_SECONDS = old_baseline
        settings.MAX_FRAME_GAP_SECONDS = old_max_gap


@patch("app.services.motion_window_service.MotionWindowService.detect_motion_windows")
def test_motion_windowing_applies_hard_gap_safeguard(mock_windows, client: TestClient) -> None:
    video_id = "00000000-0000-0000-0000-000000000007"
    video_filename = f"{video_id}.mp4"
    video_path = settings.VIDEOS_DIR / video_filename

    create_dynamic_video(video_path, duration_sec=12, fps=10)

    video_metadata = {
        "video_id": video_id,
        "filename": "gap_safeguard_test.mp4",
        "upload_time": "2026-06-24T12:30:00Z",
        "file_size": video_path.stat().st_size,
    }
    metadata_path = settings.METADATA_DIR / f"{video_id}.json"
    with open(metadata_path, "w", encoding="utf-8") as meta_file:
        json.dump(video_metadata, meta_file)

    old_motion_windowing = settings.ENABLE_MOTION_WINDOWING
    old_baseline = settings.OUT_OF_WINDOW_BASELINE_SECONDS
    old_max_gap = settings.MAX_FRAME_GAP_SECONDS
    settings.ENABLE_MOTION_WINDOWING = True
    settings.OUT_OF_WINDOW_BASELINE_SECONDS = 30.0
    settings.MAX_FRAME_GAP_SECONDS = 3.0
    mock_windows.return_value = [(2.0, 3.0)]
    try:
        response = client.post("/frames/extract", json={"video_id": video_id})
        assert response.status_code == 200
        data = response.json()

        timestamps = [frame["timestamp_seconds"] for frame in data["frames"]]
        assert 0.0 in timestamps
        assert 2.0 in timestamps
        assert 3.0 in timestamps
        assert 6.0 in timestamps
        assert 9.0 in timestamps

        debug_path = PROJECT_ROOT / "data" / "reports" / "sampling_debug" / f"{video_id}.json"
        assert debug_path.exists()
        debug_data = json.loads(debug_path.read_text(encoding="utf-8"))
        assert debug_data["metrics"]["frames_retained_by_gap_safeguard"] >= 2
    finally:
        settings.ENABLE_MOTION_WINDOWING = old_motion_windowing
        settings.OUT_OF_WINDOW_BASELINE_SECONDS = old_baseline
        settings.MAX_FRAME_GAP_SECONDS = old_max_gap
