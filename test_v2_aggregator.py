import json
import os
from pathlib import Path
import sys

# Setup python path
project_root = Path(r"c:\Mukul K\vinfo1\video-search-engine")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.services.event_aggregation_v2 import EventAggregationServiceV2
from app.services.event_aggregation import EventAggregationService

def test_side_by_side():
    metadata_dir = project_root / "data" / "metadata"
    frames_files = list(metadata_dir.glob("*_frames.json"))
    
    if not frames_files:
        print("No frames.json found to test.")
        return
        
    test_file = frames_files[0]
    video_id = test_file.name.replace("_frames.json", "")
    print(f"Testing with video_id: {video_id} using file: {test_file}")
    
    with open(test_file, "r", encoding="utf-8") as f:
        frames_metadata = json.load(f)
        
    print(f"Loaded {len(frames_metadata)} frames.")
    
    print("\n--- Running V1 ---")
    try:
        events_v1 = EventAggregationService.process_events(video_id, frames_metadata)
        print(f"V1 created {len(events_v1)} events.")
        if events_v1:
            print("V1 Sample Event:")
            print(json.dumps(events_v1[0], indent=2))
    except Exception as e:
        print(f"V1 failed: {e}")
        
    print("\n--- Running V2 ---")
    try:
        events_v2 = EventAggregationServiceV2.process_events(video_id, frames_metadata)
        print(f"V2 created {len(events_v2)} events.")
        if events_v2:
            print("V2 Sample Event:")
            print(json.dumps(events_v2[0], indent=2))
    except Exception as e:
        print(f"V2 failed: {e}")

if __name__ == "__main__":
    test_side_by_side()
