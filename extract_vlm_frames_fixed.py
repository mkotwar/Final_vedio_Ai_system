import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
import cv2
from pathlib import Path

def extract_frames(video_path: str, timestamps: list[float], out_dir: str, prefix: str):
    os.makedirs(out_dir, exist_ok=True)
    # Force FFMPEG backend to avoid MSMF hanging in headless environment
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0
    
    for ts in timestamps:
        frame_idx = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            out_name = f"{prefix}_ts{int(ts)}.jpg"
            out_path = os.path.join(out_dir, out_name)
            cv2.imwrite(out_path, frame)
            print(f"Extracted {out_path}")
        else:
            print(f"Failed to extract ts {ts} from {video_path}")
            
    cap.release()

vinfo_dir = r"c:\Mukul K\vinfo1\video-search-engine"
vid_dir = os.path.join(vinfo_dir, "validation", "videos")
out_dir = os.path.join(vinfo_dir, "validation", "vlm", "frames")

print("Starting extraction...")
extract_frames(os.path.join(vid_dir, "empy_room_15sec.mp4"), [1.0, 7.0, 14.0], out_dir, "empy_room")
extract_frames(os.path.join(vid_dir, "person_walking_30sec.mp4"), [5.0, 18.0, 28.0], out_dir, "person_walk")
extract_frames(os.path.join(vid_dir, "customer_interaticio_60sec.mp4"), [5.0, 15.0, 35.0], out_dir, "customer_int")
print("Done!")
