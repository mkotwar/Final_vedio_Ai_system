import os
import cv2
import json
import shutil
from pathlib import Path

VIDEO_PATH = r"C:\Mukul K\test_video\person_walking_30sec.mp4"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEST_VIDEO_DIR = PROJECT_ROOT / "validation" / "videos"
DEST_GT_DIR = PROJECT_ROOT / "validation" / "ground_truth"

def main():
    print(f"Checking if video exists at: {VIDEO_PATH}")
    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: Video file not found at {VIDEO_PATH}")
        return

    # Probe video specs
    print("Probing video details...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("ERROR: Could not open video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0
    cap.release()

    print(f"Video Probed - Duration: {duration:.2f}s, FPS: {fps:.2f}, Total Frames: {total_frames}")

    # Ensure destination directories exist
    DEST_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    DEST_GT_DIR.mkdir(parents=True, exist_ok=True)

    # Copy video file
    video_filename = "person_walking_30sec.mp4"
    dest_video_path = DEST_VIDEO_DIR / video_filename
    print(f"Copying video to: {dest_video_path}")
    shutil.copy(VIDEO_PATH, dest_video_path)

    # Generate a unique video ID
    # Since it's a test video, we can hash the filename or use a clean string
    video_id = "person_walking_30sec"

    # Define ground truth JSON schema
    # Since the video is named person_walking_30sec, we can assume a general event 'person_walking' for the duration
    expected_base_frames = int(round(duration))
    
    gt_data = {
        "video_id": video_id,
        "filename": video_filename,
        "original_filename": video_filename,
        "duration_seconds": round(duration, 2),
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "expected_base_frames": expected_base_frames,
        "expected_events": [
            {
                "event_id": "evt_001",
                "event_type": "person_walking",
                "start_seconds": 0.0,
                "end_seconds": round(duration, 1),
                "description": "A person is walking in the video."
            }
        ]
    }

    gt_file_path = DEST_GT_DIR / f"{video_id}.json"
    print(f"Generating ground truth JSON at: {gt_file_path}")
    with open(gt_file_path, "w", encoding="utf-8") as f:
        json.dump(gt_data, f, indent=2)

    print("Setup completed successfully!")

if __name__ == "__main__":
    main()
