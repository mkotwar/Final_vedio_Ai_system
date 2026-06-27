from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics.trackers.bot_sort import BOTSORT


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_DEBUG_ROOT = SCRIPT_PATH.parent / "benchmark_debug_runs" / "tracking"


@dataclass
class _DetectionBatch:
    xywh: np.ndarray
    conf: np.ndarray
    cls: np.ndarray

    @property
    def xyxy(self) -> np.ndarray:
        if self.xywh.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
        xyxy = np.asarray(self.xywh, dtype=np.float32).copy()
        xyxy[:, 0] = self.xywh[:, 0] - (self.xywh[:, 2] / 2.0)
        xyxy[:, 1] = self.xywh[:, 1] - (self.xywh[:, 3] / 2.0)
        xyxy[:, 2] = self.xywh[:, 0] + (self.xywh[:, 2] / 2.0)
        xyxy[:, 3] = self.xywh[:, 1] + (self.xywh[:, 3] / 2.0)
        return xyxy

    def __len__(self) -> int:
        return int(self.conf.shape[0])

    def __getitem__(self, item: Any) -> "_DetectionBatch":
        return _DetectionBatch(
            xywh=np.asarray(self.xywh[item], dtype=np.float32),
            conf=np.asarray(self.conf[item], dtype=np.float32),
            cls=np.asarray(self.cls[item], dtype=np.float32),
        )


@dataclass
class _RegistryEntry:
    global_actor_id: str
    track_history: List[int]
    first_seen: float
    last_seen: float
    appearance_embedding: np.ndarray
    class_name: str
    last_bbox: List[float]
    last_track_id: int

    def to_json(self) -> Dict[str, Any]:
        return {
            "global_actor_id": self.global_actor_id,
            "track_history": self.track_history,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "appearance_embedding": self.appearance_embedding.tolist(),
            "class_name": self.class_name,
            "last_bbox": self.last_bbox,
            "last_track_id": self.last_track_id,
        }


class BenchmarkBoTSORTTracker:
    """Benchmark-only wrapper around Ultralytics BoT-SORT with extra debug artifacts."""

    TRACKER_ARGS = SimpleNamespace(
        tracker_type="botsort",
        track_high_thresh=0.25,
        track_low_thresh=0.1,
        new_track_thresh=0.25,
        track_buffer=30,
        match_thresh=0.8,
        fuse_score=True,
        gmc_method="sparseOptFlow",
        proximity_thresh=0.5,
        appearance_thresh=0.8,
        with_reid=bool(int(os.getenv("BENCHMARK_BOTSORT_WITH_REID", "0"))),
        model=os.getenv("BENCHMARK_BOTSORT_REID_MODEL", "auto"),
    )

    GLOBAL_MATCH_THRESHOLD = 0.72
    GLOBAL_IOU_THRESHOLD = 0.30
    GLOBAL_EMBEDDING_ALPHA = 0.85

    @classmethod
    def track_frames(
        cls,
        frame_detections: Sequence[Any],
        *,
        extracted_tuples: Optional[Sequence[Tuple[str, str, float, Path]]] = None,
        debug_output_dir: Optional[Path | str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        tracker = BOTSORT(cls.TRACKER_ARGS)
        debug_root = Path(
            debug_output_dir
            or os.getenv("TRACKING_DEBUG_DIR")
            or DEFAULT_DEBUG_ROOT
        )
        tracked_frames_dir = debug_root / "tracked_frames"
        tracked_frames_dir.mkdir(parents=True, exist_ok=True)

        frame_path_map = cls._build_frame_path_map(extracted_tuples or ())
        results_rows: List[Dict[str, Any]] = []
        id_switch_events: List[Dict[str, Any]] = []
        registry: List[_RegistryEntry] = []
        track_to_actor: Dict[int, str] = {}
        previous_active_track_ids: set[int] = set()
        previously_removed_ids: set[int] = set()
        next_actor_index = 1

        for frame_index, frame_detection in enumerate(frame_detections, start=1):
            frame_id = frame_detection.frame_id
            frame_path = frame_path_map.get(frame_id)
            image = cls._load_frame_image(frame_path, frame_detection.frame_width, frame_detection.frame_height)
            detection_batch = cls._build_detection_batch(frame_detection)

            tracker.update(detection_batch, img=image)
            active_tracks = [track for track in tracker.tracked_stracks if track.is_activated]
            current_track_ids = {int(track.track_id) for track in active_tracks}
            removed_ids = {int(track.track_id) for track in getattr(tracker, "removed_stracks", [])}
            ended_track_count = len(removed_ids - previously_removed_ids)
            previously_removed_ids = removed_ids

            per_frame_entities: List[Dict[str, Any]] = []
            frame_counts: Dict[str, int] = {}
            new_track_count = 0

            for track in active_tracks:
                track_id = int(track.track_id)
                detection_index = int(getattr(track, "idx", -1))
                detection = frame_detection.detections[detection_index] if 0 <= detection_index < len(frame_detection.detections) else None
                class_name = str(getattr(detection, "class_name", getattr(track, "cls", "unknown")))
                confidence = float(getattr(detection, "confidence", getattr(track, "score", 0.0)))
                bbox = [float(value) for value in track.xyxy.tolist()]

                if track.start_frame == frame_index:
                    new_track_count += 1

                embedding = cls._appearance_embedding(image, bbox)
                prior_actor = track_to_actor.get(track_id)
                actor_id, actor_was_new, reidentified = cls._assign_global_actor(
                    registry=registry,
                    track_to_actor=track_to_actor,
                    track_id=track_id,
                    class_name=class_name,
                    timestamp_seconds=float(frame_detection.timestamp_seconds),
                    bbox=bbox,
                    embedding=embedding,
                    next_actor_index=next_actor_index,
                )
                if actor_was_new:
                    next_actor_index += 1

                if prior_actor is not None and prior_actor != actor_id:
                    id_switch_events.append(
                        {
                            "frame_id": frame_id,
                            "track_id": track_id,
                            "previous_global_actor_id": prior_actor,
                            "current_global_actor_id": actor_id,
                            "reason": "track_id_reassociated_to_different_actor",
                        }
                    )
                elif reidentified:
                    id_switch_events.append(
                        {
                            "frame_id": frame_id,
                            "track_id": track_id,
                            "previous_global_actor_id": actor_id,
                            "current_global_actor_id": actor_id,
                            "reason": "reidentified_after_gap",
                        }
                    )
                track_to_actor[track_id] = actor_id

                per_frame_entities.append(
                    {
                        "track_id": track_id,
                        "global_actor_id": actor_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": bbox,
                        "first_seen": cls._registry_entry_first_seen(registry, actor_id),
                        "last_seen": cls._registry_entry_last_seen(registry, actor_id),
                    }
                )
                frame_counts[class_name] = frame_counts.get(class_name, 0) + 1

            if frame_path is not None and per_frame_entities:
                cls._render_tracked_frame(
                    image=image,
                    detections=per_frame_entities,
                    output_path=tracked_frames_dir / f"{frame_id}.jpg",
                )

            frame_result = {
                "frame_id": frame_id,
                "video_id": frame_detection.video_id,
                "timestamp_seconds": float(frame_detection.timestamp_seconds),
                "tracked_entities": per_frame_entities,
                "track_ids": [entity["track_id"] for entity in per_frame_entities],
                "class_counts": frame_counts,
                "new_track_count": new_track_count,
                "ended_track_count": ended_track_count,
                "active_track_count": len(per_frame_entities),
                "removed_track_ids": sorted(removed_ids),
                "previous_active_track_ids": sorted(previous_active_track_ids),
            }
            results_rows.append(frame_result)
            previous_active_track_ids = current_track_ids

        tracking_results = {
            "frames": results_rows,
            "summary": {
                "total_frames": len(results_rows),
                "total_tracks": len(registry),
                "total_id_switches": len(id_switch_events),
                "reidentifications": sum(1 for event in id_switch_events if event["reason"] in {"track_id_reassociated_to_different_actor", "reidentified_after_gap"}),
                "actor_count": len(registry),
            },
        }
        actor_registry = [entry.to_json() for entry in registry]
        id_switch_report = {
            "total_tracks": len(registry),
            "total_id_switches": len(id_switch_events),
            "reidentifications": sum(1 for event in id_switch_events if event["reason"] in {"track_id_reassociated_to_different_actor", "reidentified_after_gap"}),
            "events": id_switch_events,
        }

        debug_root.mkdir(parents=True, exist_ok=True)
        (debug_root / "tracking_results.json").write_text(json.dumps(tracking_results, indent=4), encoding="utf-8")
        (debug_root / "actor_registry.json").write_text(json.dumps(actor_registry, indent=4), encoding="utf-8")
        (debug_root / "id_switch_report.json").write_text(json.dumps(id_switch_report, indent=4), encoding="utf-8")

        return {
            row["frame_id"]: {
                "tracked_entities": row["tracked_entities"],
                "track_ids": row["track_ids"],
                "class_counts": row["class_counts"],
                "new_track_count": row["new_track_count"],
                "ended_track_count": row["ended_track_count"],
            }
            for row in results_rows
        }

    @staticmethod
    def _build_frame_path_map(
        extracted_tuples: Sequence[Tuple[str, str, float, Path]]
    ) -> Dict[str, Path]:
        return {frame_id: Path(frame_path) for frame_id, _video_id, _ts, frame_path in extracted_tuples}

    @staticmethod
    def _build_detection_batch(frame_detection: Any) -> _DetectionBatch:
        boxes_xywh: List[List[float]] = []
        confidences: List[float] = []
        class_ids: List[float] = []
        for detection in frame_detection.detections:
            x1, y1, x2, y2 = detection.bbox
            boxes_xywh.append([
                float((x1 + x2) / 2.0),
                float((y1 + y2) / 2.0),
                float(x2 - x1),
                float(y2 - y1),
            ])
            confidences.append(float(detection.confidence))
            class_ids.append(float(detection.class_id))
        return _DetectionBatch(
            xywh=np.asarray(boxes_xywh, dtype=np.float32) if boxes_xywh else np.zeros((0, 4), dtype=np.float32),
            conf=np.asarray(confidences, dtype=np.float32) if confidences else np.zeros((0,), dtype=np.float32),
            cls=np.asarray(class_ids, dtype=np.float32) if class_ids else np.zeros((0,), dtype=np.float32),
        )

    @staticmethod
    def _load_frame_image(frame_path: Optional[Path], width: int, height: int) -> np.ndarray:
        if frame_path is not None and frame_path.exists():
            image = cv2.imread(str(frame_path))
            if image is not None:
                return image
        return np.zeros((max(1, height), max(1, width), 3), dtype=np.uint8)

    @staticmethod
    def _render_tracked_frame(image: np.ndarray, detections: List[Dict[str, Any]], output_path: Path) -> None:
        canvas = image.copy()
        for detection in detections:
            bbox = detection["bbox"]
            label = f'{detection["global_actor_id"]} | {detection["class_name"]} | t{detection["track_id"]}'
            x1, y1, x2, y2 = [int(round(value)) for value in bbox]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
            text_width = min(canvas.shape[1] - 1, x1 + 340)
            cv2.rectangle(canvas, (x1, max(0, y1 - 28)), (text_width, y1), (0, 0, 0), thickness=-1)
            cv2.putText(
                canvas,
                label,
                (x1 + 4, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(output_path), canvas)

    @staticmethod
    def _appearance_embedding(image: np.ndarray, bbox: Sequence[float]) -> np.ndarray:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((48,), dtype=np.float32)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        channels = [0, 1, 2]
        bins = [16, 16, 16]
        hist_parts = []
        for channel, bin_count in zip(channels, bins):
            hist = cv2.calcHist([hsv], [channel], None, [bin_count], [0, 256]).flatten()
            hist_parts.append(hist)
        embedding = np.concatenate(hist_parts).astype(np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm > 0.0:
            embedding /= norm
        return embedding

    @classmethod
    def _assign_global_actor(
        cls,
        *,
        registry: List[_RegistryEntry],
        track_to_actor: Dict[int, str],
        track_id: int,
        class_name: str,
        timestamp_seconds: float,
        bbox: Sequence[float],
        embedding: np.ndarray,
        next_actor_index: int,
    ) -> Tuple[str, bool, bool]:
        if track_id in track_to_actor:
            actor_id = track_to_actor[track_id]
            entry = next((item for item in registry if item.global_actor_id == actor_id), None)
            if entry is not None:
                entry.last_seen = timestamp_seconds
                entry.last_bbox = [float(value) for value in bbox]
                entry.last_track_id = track_id
                entry.track_history = cls._append_unique(entry.track_history, track_id)
                entry.appearance_embedding = cls._blend_embedding(entry.appearance_embedding, embedding)
            return actor_id, False, False

        best_entry: Optional[_RegistryEntry] = None
        best_score = -1.0
        for entry in registry:
            if entry.class_name != class_name:
                continue
            iou_score = cls._bbox_iou(entry.last_bbox, bbox)
            embedding_score = cls._cosine_similarity(entry.appearance_embedding, embedding)
            score = 0.55 * embedding_score + 0.45 * iou_score
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is not None and best_score >= cls.GLOBAL_MATCH_THRESHOLD:
            best_entry.last_seen = timestamp_seconds
            best_entry.last_bbox = [float(value) for value in bbox]
            best_entry.last_track_id = track_id
            best_entry.track_history = cls._append_unique(best_entry.track_history, track_id)
            best_entry.appearance_embedding = cls._blend_embedding(best_entry.appearance_embedding, embedding)
            track_to_actor[track_id] = best_entry.global_actor_id
            return best_entry.global_actor_id, False, True

        actor_id = f"actor_{next_actor_index:03d}"
        registry.append(
            _RegistryEntry(
                global_actor_id=actor_id,
                track_history=[track_id],
                first_seen=timestamp_seconds,
                last_seen=timestamp_seconds,
                appearance_embedding=embedding.copy(),
                class_name=class_name,
                last_bbox=[float(value) for value in bbox],
                last_track_id=track_id,
            )
        )
        track_to_actor[track_id] = actor_id
        return actor_id, True, False

    @staticmethod
    def _append_unique(values: List[int], track_id: int) -> List[int]:
        if track_id not in values:
            values = [*values, track_id]
        return values

    @staticmethod
    def _blend_embedding(existing: np.ndarray, incoming: np.ndarray) -> np.ndarray:
        if existing.size == 0:
            return incoming.copy()
        blended = (existing * BenchmarkBoTSORTTracker.GLOBAL_EMBEDDING_ALPHA) + (
            incoming * (1.0 - BenchmarkBoTSORTTracker.GLOBAL_EMBEDDING_ALPHA)
        )
        norm = float(np.linalg.norm(blended))
        if norm > 0.0:
            blended /= norm
        return blended.astype(np.float32)

    @staticmethod
    def _bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
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

    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        if vec_a.size == 0 or vec_b.size == 0:
            return 0.0
        denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        if denom <= 0.0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / denom)

    @staticmethod
    def _registry_entry_first_seen(registry: List[_RegistryEntry], actor_id: str) -> float:
        entry = next((item for item in registry if item.global_actor_id == actor_id), None)
        return float(entry.first_seen if entry is not None else 0.0)

    @staticmethod
    def _registry_entry_last_seen(registry: List[_RegistryEntry], actor_id: str) -> float:
        entry = next((item for item in registry if item.global_actor_id == actor_id), None)
        return float(entry.last_seen if entry is not None else 0.0)
