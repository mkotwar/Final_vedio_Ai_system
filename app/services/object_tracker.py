from typing import Any, Dict, List, Optional


class ObjectTrackerService:
    """Lightweight IoU-based tracker for stable per-video entity ids."""

    IOU_THRESHOLD = 0.30
    MAX_MISSED_FRAMES = 2

    @staticmethod
    def _iou(box_a: List[float], box_b: List[float]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0.0:
            return 0.0

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter_area
        if denom <= 0.0:
            return 0.0
        return inter_area / denom

    @classmethod
    def track_frames(cls, frame_detections: List[Any]) -> Dict[str, Dict[str, Any]]:
        tracks: List[Dict[str, Any]] = []
        next_track_id = 1
        per_frame: Dict[str, Dict[str, Any]] = {}

        for frame_detection in frame_detections:
            matched_track_ids = set()
            tracked_entities: List[Dict[str, Any]] = []
            new_track_count = 0
            ended_track_count = 0
            frame_counts: Dict[str, int] = {}

            for detection in frame_detection.detections:
                best_track: Optional[Dict[str, Any]] = None
                best_iou = 0.0

                for track in tracks:
                    if track["class_name"] != detection.class_name:
                        continue
                    iou_score = cls._iou(track["bbox"], detection.bbox)
                    if iou_score >= cls.IOU_THRESHOLD and iou_score > best_iou:
                        best_iou = iou_score
                        best_track = track

                if best_track is None:
                    best_track = {
                        "track_id": next_track_id,
                        "class_name": detection.class_name,
                        "first_seen": frame_detection.timestamp_seconds,
                        "last_seen": frame_detection.timestamp_seconds,
                        "bbox": detection.bbox,
                        "missed_frames": 0,
                    }
                    tracks.append(best_track)
                    next_track_id += 1
                    new_track_count += 1
                else:
                    best_track["last_seen"] = frame_detection.timestamp_seconds
                    best_track["bbox"] = detection.bbox
                    best_track["missed_frames"] = 0

                matched_track_ids.add(best_track["track_id"])
                frame_counts[detection.class_name] = frame_counts.get(detection.class_name, 0) + 1
                tracked_entities.append(
                    {
                        "track_id": best_track["track_id"],
                        "class_name": detection.class_name,
                        "confidence": detection.confidence,
                        "bbox": detection.bbox,
                    }
                )

            for track in tracks:
                if track["track_id"] not in matched_track_ids:
                    track["missed_frames"] += 1
                    if track["missed_frames"] == cls.MAX_MISSED_FRAMES:
                        ended_track_count += 1

            per_frame[frame_detection.frame_id] = {
                "tracked_entities": tracked_entities,
                "track_ids": [entity["track_id"] for entity in tracked_entities],
                "class_counts": frame_counts,
                "new_track_count": new_track_count,
                "ended_track_count": ended_track_count,
            }

        return per_frame
