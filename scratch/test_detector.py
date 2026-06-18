import sys
import os
import cv2
import json
from pathlib import Path
from collections import defaultdict

# Add the project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.object_detection.detector import ObjectDetector

def main():
    detector = ObjectDetector()
    
    frames_dir = Path("data/frames")
    output_dir = Path("scratch/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect up to 10 frame paths
    all_frames = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        all_frames.extend(list(frames_dir.rglob(ext)))
        if len(all_frames) >= 10:
            break
            
    sample_frames = all_frames[:10]
    
    if not sample_frames:
        print("No frames found in data/frames")
        return
        
    print(f"Found {len(sample_frames)} frames to process.")
    
    total_detections = 0
    detections_per_class = defaultdict(int)
    total_confidence = 0.0
    
    for idx, frame_path in enumerate(sample_frames):
        video_id = frame_path.parent.name
        frame_id = frame_path.stem
        
        # Run detection
        result = detector.detect_frame(
            frame_path=frame_path,
            frame_id=frame_id,
            video_id=video_id,
            timestamp_seconds=idx * 1.0 # arbitrary
        )
        
        # Load image for drawing
        img = cv2.imread(str(frame_path))
        if img is None:
            print(f"Warning: Could not read {frame_path}")
            continue
            
        # Draw detections
        for det in result.detections:
            total_detections += 1
            detections_per_class[det.class_name] += 1
            total_confidence += det.confidence
            
            x1, y1, x2, y2 = map(int, det.bbox)
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{det.class_name}: {det.confidence:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - h - 5), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
        # Save output image
        out_path = output_dir / f"{video_id}_{frame_id}.jpg"
        cv2.imwrite(str(out_path), img)
        print(f"Processed and saved: {out_path}")
        
        
    # Summary Report
    summary = []
    summary.append("="*40)
    summary.append("DETECTION SUMMARY REPORT")
    summary.append("="*40)
    summary.append(f"Total Frames Processed: {len(sample_frames)}")
    summary.append(f"Total Detections: {total_detections}")
    
    if total_detections > 0:
        avg_conf = total_confidence / total_detections
        summary.append(f"Average Confidence: {avg_conf:.4f}")
    else:
        summary.append("Average Confidence: 0.0000")
        
    summary.append("\nDetections per class:")
    for cls_name, count in sorted(detections_per_class.items(), key=lambda x: x[1], reverse=True):
        summary.append(f" - {cls_name}: {count}")
    summary.append("="*40)

    report_text = "\n".join(summary)
    print(report_text)
    
    with open("scratch/summary.txt", "w") as f:
        f.write(report_text)

if __name__ == "__main__":
    main()
