import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
from loguru import logger
from app.core.config import settings

class MotionWindowService:
    """Service to detect temporal windows of motion activity in a video."""

    @classmethod
    def detect_motion_windows(cls, video_path: Path) -> List[Tuple[float, float]]:
        """
        Performs a fast pass over the video to detect windows of motion.
        
        Args:
            video_path: Absolute path to the video file.
            
        Returns:
            List of (start_seconds, end_seconds) tuples denoting active windows.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Failed to open video for motion windowing: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Configuration
        threshold_percent = settings.MOTION_THRESHOLD_PERCENT
        consecutive_frames_needed = settings.MOTION_CONSECUTIVE_FRAMES
        pre_buffer = settings.PRE_EVENT_BUFFER_SECONDS
        post_buffer = settings.POST_EVENT_BUFFER_SECONDS
        max_duration = settings.MAX_FRAMES_PER_WINDOW
        
        # State
        active_windows: List[Tuple[float, float]] = []
        is_active = False
        consecutive_motion = 0
        window_start_sec = 0.0
        
        frame_idx = 0
        
        # Use MOG2 Background Subtractor to handle smooth/slow motion correctly
        # instead of simplistic frame-to-frame absdiff.
        back_sub = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=25, detectShadows=False)

        while True:
            success, frame = cap.read()
            if not success:
                break
                
            current_sec = frame_idx / fps
            
            # Downscale for performance
            small_frame = cv2.resize(frame, (320, 240))
            
            # Apply MOG2 mask
            fg_mask = back_sub.apply(small_frame)
            
            # Clean up noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            
            non_zero = np.count_nonzero(fg_mask)
            total_pixels = fg_mask.size
            motion_percent = non_zero / total_pixels
            
            if motion_percent >= threshold_percent:
                consecutive_motion += 1
            else:
                consecutive_motion = 0
                
            if not is_active and consecutive_motion >= consecutive_frames_needed:
                is_active = True
                # Start window slightly before the first detected motion frame
                window_start_sec = max(0.0, current_sec - (consecutive_frames_needed / fps))
                
            elif is_active and consecutive_motion == 0:
                # End of motion window
                is_active = False
                active_windows.append((window_start_sec, current_sec))
                
            elif is_active:
                # Check for max duration cap
                if (current_sec - window_start_sec) >= max_duration:
                    is_active = False
                    active_windows.append((window_start_sec, current_sec))
                    consecutive_motion = 0 # reset to require new trigger
            
            frame_idx += 1
            
        if is_active:
            active_windows.append((window_start_sec, frame_idx / fps))
            
        cap.release()
        
        # Apply Buffers and Merge Overlapping Windows
        buffered_windows = []
        for start_sec, end_sec in active_windows:
            padded_start = max(0.0, start_sec - pre_buffer)
            padded_end = end_sec + post_buffer
            buffered_windows.append((padded_start, padded_end))
            
        merged_windows = cls._merge_windows(buffered_windows)
        
        return merged_windows
        
    @classmethod
    def _merge_windows(cls, windows: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Merges overlapping time windows."""
        if not windows:
            return []
            
        # Sort by start time
        sorted_windows = sorted(windows, key=lambda w: w[0])
        merged = [sorted_windows[0]]
        
        for current in sorted_windows[1:]:
            last = merged[-1]
            if current[0] <= last[1]:
                # Overlap, merge them
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)
                
        return merged
