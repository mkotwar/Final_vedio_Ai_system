import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app.services.motion_window_service import MotionWindowService
from app.core.config import settings

@pytest.fixture
def mock_settings():
    original_threshold = settings.MOTION_THRESHOLD_PERCENT
    original_consecutive = settings.MOTION_CONSECUTIVE_FRAMES
    original_pre = settings.PRE_EVENT_BUFFER_SECONDS
    original_post = settings.POST_EVENT_BUFFER_SECONDS
    original_min = settings.MIN_MOTION_WINDOW_SECONDS
    
    settings.MOTION_THRESHOLD_PERCENT = 0.05
    settings.MOTION_CONSECUTIVE_FRAMES = 2
    settings.PRE_EVENT_BUFFER_SECONDS = 1
    settings.POST_EVENT_BUFFER_SECONDS = 1
    settings.MIN_MOTION_WINDOW_SECONDS = 6
    
    yield
    
    settings.MOTION_THRESHOLD_PERCENT = original_threshold
    settings.MOTION_CONSECUTIVE_FRAMES = original_consecutive
    settings.PRE_EVENT_BUFFER_SECONDS = original_pre
    settings.POST_EVENT_BUFFER_SECONDS = original_post
    settings.MIN_MOTION_WINDOW_SECONDS = original_min

def test_merge_windows():
    windows = [(0.0, 2.0), (1.5, 3.0), (4.0, 5.0)]
    merged = MotionWindowService._merge_windows(windows)
    assert merged == [(0.0, 3.0), (4.0, 5.0)]

    windows_2 = [(0.0, 1.0), (2.0, 3.0), (2.5, 4.0), (3.5, 5.0)]
    merged_2 = MotionWindowService._merge_windows(windows_2)
    assert merged_2 == [(0.0, 1.0), (2.0, 5.0)]

def test_expand_short_windows():
    windows = [(10.0, 11.0)]

    expanded = MotionWindowService._expand_short_windows(windows, 6.0, 120.0)

    assert expanded == [(7.5, 13.5)]

@patch("cv2.VideoCapture")
def test_detect_motion_windows_static(mock_cap, mock_settings, tmp_path):
    # Setup mock video
    mock_video = MagicMock()
    mock_cap.return_value = mock_video
    mock_video.isOpened.return_value = True
    mock_video.get.side_effect = lambda prop: 30.0 if prop == 5 else 90 # FPS=30, Total=90
    
    # Create identical frames (static video)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    
    returns = [(True, frame)] * 90 + [(False, None)]
    mock_video.read.side_effect = returns
    
    dummy_path = tmp_path / "dummy.mp4"
    windows = MotionWindowService.detect_motion_windows(dummy_path)
    
    # Should be no motion detected
    assert len(windows) == 0

@patch("cv2.VideoCapture")
def test_detect_motion_windows_with_motion(mock_cap, mock_settings, tmp_path):
    # Setup mock video
    mock_video = MagicMock()
    mock_cap.return_value = mock_video
    mock_video.isOpened.return_value = True
    mock_video.get.side_effect = lambda prop: 30.0 if prop == 5 else 90 # FPS=30, Total=90
    
    frame_static = np.zeros((240, 320, 3), dtype=np.uint8)
    frame_motion = np.ones((240, 320, 3), dtype=np.uint8) * 255
    
    # Frames 0-29: Static
    # Frames 30-33: Motion (Exceeds consecutive 2 frames needed)
    # Frames 34-89: Static
    frames = [(True, frame_static)] * 30 + \
             [(True, frame_motion), (True, frame_static), (True, frame_motion), (True, frame_static)] * 2 + \
             [(True, frame_static)] * 52 + \
             [(False, None)]
             
    mock_video.read.side_effect = frames
    
    dummy_path = tmp_path / "dummy.mp4"
    windows = MotionWindowService.detect_motion_windows(dummy_path)
    
    # Motion detected around sec 1.0 to 1.2
    # Pre-buffer is 1s, Post-buffer is 1s
    # So window should be around 0.0 to 2.2
    assert len(windows) == 1
    assert windows[0][0] == 0.0  # bounded by 0
    assert windows[0][1] > 1.5
