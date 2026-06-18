from typing import List
from pydantic import BaseModel, Field

class Detection(BaseModel):
    class_id: int = Field(..., description="ID of the detected class")
    class_name: str = Field(..., description="Name of the detected class")
    confidence: float = Field(..., description="Confidence score between 0 and 1")
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    center_x: float = Field(..., description="X coordinate of the bounding box center")
    center_y: float = Field(..., description="Y coordinate of the bounding box center")
    width: float = Field(..., description="Width of the bounding box")
    height: float = Field(..., description="Height of the bounding box")

class FrameDetection(BaseModel):
    frame_id: str = Field(..., description="Unique ID for the frame")
    video_id: str = Field(..., description="Unique ID for the video")
    timestamp_seconds: float = Field(..., description="Timestamp of the frame in the video")
    frame_width: int = Field(..., description="Original width of the frame")
    frame_height: int = Field(..., description="Original height of the frame")
    detections: List[Detection] = Field(default_factory=list, description="List of detections in the frame")
