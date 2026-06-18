import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from app.services.object_detection.detector import ObjectDetector
from app.services.object_detection.schemas import FrameDetection, Detection

# Pytest fixture to mock YOLO
@pytest.fixture
def mock_yolo():
    with patch("app.services.object_detection.detector.YOLO") as mock:
        # Create a mock result object that YOLO returns
        mock_result = MagicMock()
        mock_box = MagicMock()
        
        # Configure the box properties
        # Format: [x1, y1, x2, y2]
        mock_box.xyxy = [[100.0, 50.0, 200.0, 150.0]]
        mock_box.conf = [0.95]
        mock_box.cls = [0]
        
        mock_result.boxes = [mock_box]
        mock_result.names = {0: "person", 1: "car"}
        mock_result.orig_shape = (1080, 1920) # (height, width)
        
        # Configure the YOLO instance to return a list of mock results
        mock_instance = MagicMock()
        mock_instance.return_value = [mock_result]
        mock.return_value = mock_instance
        
        yield mock

@pytest.fixture
def temp_detections_dir(tmp_path):
    # Mock the save directory to use tmp_path
    with patch("app.services.object_detection.detector.Path") as mock_path:
        # We need to only mock Path("data/detections") and let other Paths work
        # A simpler way is to patch the save_dir directly or use monkeypatch
        pass

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the ObjectDetector singleton before each test."""
    ObjectDetector._instance = None
    ObjectDetector._model = None
    yield
    ObjectDetector._instance = None
    ObjectDetector._model = None


def test_singleton_initialization(mock_yolo):
    """Test that the model is initialized correctly and lazily."""
    detector1 = ObjectDetector()
    detector2 = ObjectDetector()
    
    assert detector1 is detector2
    
    # YOLO should NOT have been called yet (lazy loading)
    mock_yolo.assert_not_called()
    
    # Initialize model explicitly for test or by calling detect_frame
    detector1._initialize_model()
    mock_yolo.assert_called_once_with("yolo11m.pt")
    
    # Second call should not re-initialize
    detector1._initialize_model()
    assert mock_yolo.call_count == 1

def test_detect_frame(mock_yolo, tmp_path, monkeypatch):
    """Test the detect_frame logic and file saving."""
    
    # Monkeypatch the data directory to use a temporary directory
    mock_data_dir = tmp_path / "data"
    
    # Create a custom Path class to intercept Path("data/detections")
    original_path = Path
    
    class MockPath(original_path):
        def __new__(cls, *args, **kwargs):
            if args and str(args[0]) == "data/detections":
                return original_path(mock_data_dir / "detections")
            return original_path(*args, **kwargs)

    monkeypatch.setattr("app.services.object_detection.detector.Path", MockPath)
    
    detector = ObjectDetector()
    
    # Create a dummy image path
    dummy_image = tmp_path / "dummy.jpg"
    dummy_image.touch()
    
    # Run detection
    result = detector.detect_frame(
        frame_path=dummy_image,
        frame_id="frame_001",
        video_id="vid_123",
        timestamp_seconds=1.5
    )
    
    # Assert result structure
    assert isinstance(result, FrameDetection)
    assert result.frame_id == "frame_001"
    assert result.video_id == "vid_123"
    assert result.timestamp_seconds == 1.5
    assert result.frame_width == 1920
    assert result.frame_height == 1080
    assert len(result.detections) == 1
    
    # Assert detection values based on our mock
    # bbox: [100.0, 50.0, 200.0, 150.0]
    det = result.detections[0]
    assert det.class_id == 0
    assert det.class_name == "person"
    assert det.confidence == 0.95
    assert det.bbox == [100.0, 50.0, 200.0, 150.0]
    assert det.width == 100.0  # 200 - 100
    assert det.height == 100.0 # 150 - 50
    assert det.center_x == 150.0 # 100 + 100/2
    assert det.center_y == 100.0 # 50 + 100/2
    
    # Assert file was saved
    expected_save_path = mock_data_dir / "detections" / "vid_123" / "frame_001.json"
    assert expected_save_path.exists()
    
    # Assert JSON contents
    with open(expected_save_path, "r") as f:
        saved_data = json.load(f)
        assert saved_data["frame_id"] == "frame_001"
        assert saved_data["video_id"] == "vid_123"
        assert saved_data["frame_width"] == 1920
        assert saved_data["frame_height"] == 1080
        assert len(saved_data["detections"]) == 1
        assert saved_data["detections"][0]["class_id"] == 0
        assert saved_data["detections"][0]["class_name"] == "person"
