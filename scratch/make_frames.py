import cv2
import numpy as np
from pathlib import Path

frames_dir = Path("data/frames/sample_video")
frames_dir.mkdir(parents=True, exist_ok=True)

# Create a black image
img = np.zeros((1080, 1920, 3), dtype=np.uint8)

# Add some fake rectangles so YOLO might detect something or at least process it
cv2.rectangle(img, (500, 500), (800, 800), (255, 255, 255), -1)

for i in range(10):
    path = frames_dir / f"frame_{i:03d}.jpg"
    cv2.imwrite(str(path), img)
    print(f"Created {path}")
