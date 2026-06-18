import json
import logging
import time
from pathlib import Path
from typing import Optional, Any

from loguru import logger
from ultralytics import YOLO

from app.services.object_detection.schemas import Detection, FrameDetection

# Setup standard logging to match the requirement if loguru isn't enough, 
# but loguru is in requirements.txt. We'll use loguru for consistency.
# If they specifically requested "logging", we can use python's built-in, 
# but we'll stick to loguru which is standard here.

class ObjectDetector:
    _instance: Optional['ObjectDetector'] = None
    _model: Optional[Any] = None
    
    def __new__(cls) -> 'ObjectDetector':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _initialize_model(self) -> None:
        """Initialize the YOLO model exactly once."""
        if self._model is not None:
            return
            
        logger.info("Initializing ObjectDetector model YOLO('yolo11m.pt')...")
        start_time = time.time()
        self._model = YOLO("yolo11m.pt")
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully. Model load time: {load_time:.4f} seconds.")

    def detect_frame(self, frame_path: Path, frame_id: str, video_id: str, timestamp_seconds: float) -> FrameDetection:
        """
        Detect objects in a single frame.
        
        Args:
            frame_path: Path to the image file.
            frame_id: Unique ID for the frame.
            video_id: Unique ID for the video.
            timestamp_seconds: Timestamp of the frame.
            
        Returns:
            FrameDetection object containing the detection results.
        """
        if not self._model:
            self._initialize_model()

        logger.info(f"Running detection on frame {frame_id} from video {video_id}.")
        
        start_inference = time.time()
        # Perform inference
        # model() returns a list of Results objects
        results = self._model(str(frame_path))
        inference_time = time.time() - start_inference
        
        detections_list = []
        total_confidence = 0.0
        detection_count = 0
        frame_height = 0
        frame_width = 0
        
        if results and len(results) > 0:
            result = results[0]  # We expect 1 result because we passed 1 image
            
            # Extract boxes and original image shape
            boxes = result.boxes
            names = result.names
            
            if hasattr(result, 'orig_shape') and result.orig_shape:
                frame_height, frame_width = result.orig_shape
            
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # box.xyxy is [x1, y1, x2, y2]
                    # box.conf is confidence
                    # box.cls is class id
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    class_name = names.get(cls_id, str(cls_id))
                    
                    x1, y1, x2, y2 = xyxy
                    width = x2 - x1
                    height = y2 - y1
                    center_x = x1 + width / 2.0
                    center_y = y1 + height / 2.0
                    
                    detection = Detection(
                        class_id=cls_id,
                        class_name=class_name,
                        confidence=conf,
                        bbox=xyxy,
                        center_x=center_x,
                        center_y=center_y,
                        width=width,
                        height=height
                    )
                    detections_list.append(detection)
                    total_confidence += conf
                    detection_count += 1

        avg_confidence = total_confidence / detection_count if detection_count > 0 else 0.0
        
        # Structured logging requirements
        logger.info(
            f"Object detection completed | "
            f"Video: {video_id} | Frame: {frame_id} | "
            f"Inference time: {inference_time:.4f}s | "
            f"Detection count: {detection_count} | "
            f"Average confidence: {avg_confidence:.4f}"
        )
        
        frame_detection = FrameDetection(
            frame_id=frame_id,
            video_id=video_id,
            timestamp_seconds=timestamp_seconds,
            frame_width=frame_width,
            frame_height=frame_height,
            detections=detections_list
        )
        
        # Save to data/detections/{video_id}/{frame_id}.json
        save_dir = Path("data/detections") / video_id
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = save_dir / f"{frame_id}.json"
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(frame_detection.model_dump_json(indent=2))
            
        logger.info(f"Saved frame detections to {save_path}")
        
        return frame_detection
