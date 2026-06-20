import os
import imageio.v3 as iio
from PIL import Image

def extract_frames_iio(video_path, timestamps, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    try:
        props = iio.improps(video_path, plugin="pyav")
        fps = props.shape[0] / props.shape[1] if hasattr(props, 'shape') else 30.0 
        # Fallback to estimating fps, imageio pyav might not return easy fps. Let's just read metadata:
        meta = iio.immeta(video_path, plugin="pyav")
        fps = meta.get("fps", 30.0)
    except:
        fps = 30.0

    print(f"Reading {video_path} at FPS ~{fps}")
    for ts in timestamps:
        frame_idx = int(ts * fps)
        try:
            frame = iio.imread(video_path, index=frame_idx, plugin="pyav")
            out_name = f"{prefix}_ts{int(ts)}.jpg"
            out_path = os.path.join(out_dir, out_name)
            Image.fromarray(frame).save(out_path)
            print(f"Extracted {out_path}")
        except Exception as e:
            print(f"Failed to extract frame {frame_idx} from {video_path}: {e}")

vinfo_dir = r"c:\Mukul K\vinfo1\video-search-engine"
vid_dir = os.path.join(vinfo_dir, "validation", "videos")
out_dir = os.path.join(vinfo_dir, "validation", "vlm", "frames")

extract_frames_iio(os.path.join(vid_dir, "empy_room_15sec.mp4"), [1.0, 7.0, 14.0], out_dir, "empy_room")
extract_frames_iio(os.path.join(vid_dir, "person_walking_30sec.mp4"), [5.0, 18.0, 28.0], out_dir, "person_walk")
extract_frames_iio(os.path.join(vid_dir, "customer_interaticio_60sec.mp4"), [5.0, 15.0, 35.0], out_dir, "customer_int")
