import os
import cv2
import json
import numpy as np
from pathlib import Path
from app.services.poster_service import PosterService
from app.schemas.summary import AggregatedEvent
from app.schemas.incident import IncidentChain

def test_poster_service():
    print("Testing Poster Service...")
    video_id = "test_vid_123"
    
    # Create mock frames dir
    os.makedirs(f"data/frames/{video_id}", exist_ok=True)
    os.makedirs(f"data/thumbnails/{video_id}", exist_ok=True)
    
    # Create mock image (frame_1) - Clear
    img_clear = np.ones((720, 1280, 3), dtype=np.uint8) * 150
    cv2.putText(img_clear, "Clear Frame", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.imwrite(f"data/frames/{video_id}/frame_1.jpg", img_clear)
    
    # Create mock image (frame_2) - Blurry
    img_blur = cv2.GaussianBlur(img_clear, (55, 55), 0)
    cv2.imwrite(f"data/frames/{video_id}/frame_2.jpg", img_blur)
    
    # Create mock image (frame_3) - Dark
    img_dark = np.ones((720, 1280, 3), dtype=np.uint8) * 10
    cv2.imwrite(f"data/frames/{video_id}/frame_3.jpg", img_dark)

    frames = [
        {"frame_id": f"{video_id}_f1", "timestamp_seconds": 1.0, "activities": ["walking"], "objects": ["person"], "confidence": 0.8},
        {"frame_id": f"{video_id}_f2", "timestamp_seconds": 2.0, "activities": ["running", "accident"], "objects": ["person", "car"], "confidence": 0.9},
        {"frame_id": f"{video_id}_f3", "timestamp_seconds": 3.0, "activities": ["standing"], "objects": ["person"], "confidence": 0.7}
    ]
    
    event = {
        "event_id": "evt_001",
        "event_severity": 10
    }
    
    event_res = PosterService.select_event_poster(video_id, event, frames)
    
    print("Event Poster Selection Result:")
    print(json.dumps(event_res, indent=2))
    
    # Test Incident Poster
    incident = IncidentChain(
        incident_id="inc_001",
        primary_incident_type="accident",
        severity="high",
        summary="Test incident",
        chain_events=[
            AggregatedEvent(
                video_id=video_id,
                event_id="evt_001",
                start_time="00:00:01",
                end_time="00:00:05",
                event_type="accident",
                description="test",
                event_severity=8,
                poster_frame="/api/static/thumbnails/vid_1/f1.jpg"
            ),
            AggregatedEvent(
                video_id=video_id,
                event_id="evt_002",
                start_time="00:00:05",
                end_time="00:00:10",
                event_type="accident",
                description="test",
                event_severity=15,
                poster_frame="/api/static/thumbnails/vid_1/f2.jpg"
            )
        ]
    )
    
    incident_res = PosterService.select_incident_poster(incident)
    print("\nIncident Poster Selection Result:")
    print(incident_res.poster_frame)
    assert incident_res.poster_frame == "/api/static/thumbnails/vid_1/f2.jpg"

if __name__ == "__main__":
    test_poster_service()
