import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Create dummy frames
frames_dir = Path("data/frames/sample_video")
frames_dir.mkdir(parents=True, exist_ok=True)
img = np.zeros((1080, 1920, 3), dtype=np.uint8)
cv2.rectangle(img, (500, 500), (800, 800), (255, 255, 255), -1)

for i in range(10):
    cv2.imwrite(str(frames_dir / f"frame_{i:03d}.jpg"), img)

# Import and run test_detector
import test_detector
test_detector.main()
