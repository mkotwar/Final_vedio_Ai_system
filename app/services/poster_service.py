"""Service for generating and selecting representative 'hero' frames (posters) for events and incidents."""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger

from app.core.config import settings
from app.core.utils import format_timestamp_human

class PosterService:
    """Handles logic for selecting the best thumbnail poster for an event or incident."""
    
    @classmethod
    def check_image_quality(cls, frame_path: str) -> int:
        """Analyzes a frame using OpenCV to detect blurriness, over/under exposure, etc.
        Returns a penalty score (negative int) if the frame is of poor quality.
        """
        penalty = 0
        if not Path(frame_path).exists():
            return -1000
            
        try:
            # Read image in grayscale for pixel analysis
            img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return -1000
                
            # 1. Blur Detection (Laplacian Variance)
            variance = cv2.Laplacian(img, cv2.CV_64F).var()
            if variance < 50.0:  # blurry threshold
                penalty -= 100
                
            # 2. Exposure Check (Mean Brightness)
            mean_val = np.mean(img)
            if mean_val < 20:  # nearly black
                penalty -= 100
            elif mean_val > 240: # nearly white
                penalty -= 100
                
            return penalty
        except Exception as e:
            logger.warning(f"Failed to analyze image quality for {frame_path}: {e}")
            return -50
            
    @classmethod
    def score_frame(cls, frame: Dict[str, Any], event_severity: int = 15) -> int:
        """Scores a single frame based on semantic richness and activity.
        Formula: (Severity * 1.5) + (Activities * 10) + (Objects * 5) + (Flags * 20) + (Confidence * 10)
        """
        score = 0
        
        # A. Severity contribution
        score += int(event_severity * 1.5)
        
        # B. Activity richness
        activities = frame.get("activities", [])
        score += len(activities) * 10
        
        # C. Object count
        objects = frame.get("objects", [])
        if not objects:
            score -= 50  # Empty scene penalty
        score += len(objects) * 5
        
        # D. Behavioral Flags (if any extracted at frame level, else infer from activities)
        flags = 0
        acts_str = " ".join(activities).lower()
        if any(kw in acts_str for kw in ["accident", "collision", "crash", "fall", "fire", "intrusion"]):
            flags += 1
        if any(kw in acts_str for kw in ["running", "carrying", "weapon"]):
            flags += 1
        score += flags * 20
        
        # E. Semantic importance / Confidence
        confidence = frame.get("confidence", 0.5)
        score += int(confidence * 10)
        
        # F. OCR Only check
        ocr = frame.get("ocr", {})
        if ocr and "detected_text" in ocr and len(ocr["detected_text"]) > 5 and not objects and not activities:
            score -= 50  # OCR heavy but no semantic visual content
            
        return score

    @classmethod
    def generate_thumbnail(cls, video_id: str, frame_id: str, original_path: Path) -> Optional[str]:
        """Resizes the original frame to 320x180 and saves it to data/thumbnails/video_id/."""
        if not original_path.exists():
            return None
            
        thumb_dir = settings.DATA_DIR / "thumbnails" / video_id
        thumb_dir.mkdir(parents=True, exist_ok=True)
        
        thumb_path = thumb_dir / f"{frame_id}.jpg"
        if thumb_path.exists():
            return f"/api/static/thumbnails/{video_id}/{frame_id}.jpg"
            
        try:
            img = cv2.imread(str(original_path))
            if img is None:
                return None
            
            # Resize for UI performance
            resized = cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(thumb_path), resized)
            return f"/api/static/thumbnails/{video_id}/{frame_id}.jpg"
        except Exception as e:
            logger.error(f"Thumbnail generation failed for {frame_id}: {e}")
            return None

    @classmethod
    def select_event_poster(cls, video_id: str, event_data: Dict[str, Any], group_frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Selects the best frame for an AggregatedEvent, generates a thumbnail, and updates the event."""
        if not group_frames:
            return event_data
            
        best_frame = None
        highest_score = -9999
        
        event_severity = event_data.get("event_severity", 15)
        frames_dir = settings.FRAMES_DIR / video_id
        
        for frame in group_frames:
            frame_id = frame.get("frame_id")
            if not frame_id:
                continue
                
            frame_idx_str = frame_id.split("_f")[-1]
            original_path = frames_dir / f"frame_{frame_idx_str}.jpg"
            
            # 1. Score semantics
            base_score = cls.score_frame(frame, event_severity)
            
            # 2. Quality penalty
            quality_penalty = cls.check_image_quality(str(original_path))
            
            total_score = base_score + quality_penalty
            
            if total_score > highest_score:
                highest_score = total_score
                best_frame = (frame, original_path)
                
        if best_frame:
            frame_meta, original_path = best_frame
            frame_id = frame_meta.get("frame_id")
            timestamp_sec = frame_meta.get("timestamp_seconds", 0.0)
            
            thumb_url = cls.generate_thumbnail(video_id, frame_id, original_path)
            
            event_data["poster_frame"] = thumb_url
            event_data["poster_timestamp"] = format_timestamp_human(timestamp_sec)
            event_data["poster_frame_id"] = frame_id
            event_data["thumbnail_path"] = thumb_url
            
        return event_data

    @classmethod
    def select_incident_poster(cls, incident: Any) -> Any:
        """Selects the best poster from an IncidentChain's constituent events."""
        is_pydantic = hasattr(incident, "chain_events")
        events = incident.chain_events if is_pydantic else incident.get("chain_events", [])
        
        if not events:
            return incident
            
        # The best poster is the one from the event with the highest severity.
        # If tie, pick the one with the most objects/activities (can use duration as proxy).
        best_event = None
        highest_sev = -1
        
        for ev in events:
            if hasattr(ev, "model_dump"):
                ev = ev.model_dump()
            elif hasattr(ev, "dict"):
                ev = ev.dict()
                
            sev = ev.get("event_severity", 0)
            if sev > highest_sev:
                highest_sev = sev
                best_event = ev
            elif sev == highest_sev and best_event is not None:
                if len(ev.get("activities", [])) > len(best_event.get("activities", [])):
                    best_event = ev
                    
        if best_event:
            pf = best_event.get("poster_frame") or best_event.get("thumbnail_path")
            if pf:
                if is_pydantic:
                    incident.poster_frame = pf
                    incident.poster_timestamp = best_event.get("poster_timestamp")
                    incident.poster_event_id = best_event.get("event_id")
                else:
                    incident["poster_frame"] = pf
                    incident["poster_timestamp"] = best_event.get("poster_timestamp")
                    incident["poster_event_id"] = best_event.get("event_id")
            
        return incident
