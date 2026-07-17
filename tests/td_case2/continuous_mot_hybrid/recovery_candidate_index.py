from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .recoverable_track_store import RecoverableTrackSnapshot


def _grid_cell(*, center_x: float, center_y: float, frame_width: int, frame_height: int, grid_size: int = 4) -> tuple[int, int]:
    cell_x = min(grid_size - 1, max(0, int((center_x / max(frame_width, 1)) * grid_size)))
    cell_y = min(grid_size - 1, max(0, int((center_y / max(frame_height, 1)) * grid_size)))
    return cell_x, cell_y


@dataclass
class RecoveryCandidateIndex:
    frame_width: int
    frame_height: int
    candidate_queries: int = 0
    total_candidates_before_scoring: int = 0
    total_rejected_before_scoring: int = 0
    maximum_candidates_per_unmatched_detection: int = 0

    def query(
        self,
        *,
        unmatched_detection: dict[str, Any],
        recoverable_entries: list[RecoverableTrackSnapshot],
        timestamp_seconds: float,
    ) -> list[RecoverableTrackSnapshot]:
        self.candidate_queries += 1
        family = str(unmatched_detection["family"])
        bbox_xyxy = list(unmatched_detection["bbox_xyxy"])
        center_x = (float(bbox_xyxy[0]) + float(bbox_xyxy[2])) / 2.0
        center_y = (float(bbox_xyxy[1]) + float(bbox_xyxy[3])) / 2.0
        detection_cell = _grid_cell(center_x=center_x, center_y=center_y, frame_width=self.frame_width, frame_height=self.frame_height)
        candidates: list[RecoverableTrackSnapshot] = []
        rejected = 0
        for entry in recoverable_entries:
            if entry.object_family != family:
                rejected += 1
                continue
            if float(entry.recovery_expiry_timestamp) < float(timestamp_seconds):
                rejected += 1
                continue
            entry_cell = _grid_cell(center_x=entry.last_center[0], center_y=entry.last_center[1], frame_width=self.frame_width, frame_height=self.frame_height)
            if max(abs(entry_cell[0] - detection_cell[0]), abs(entry_cell[1] - detection_cell[1])) > 1:
                rejected += 1
                continue
            if entry.movement_direction not in {"unknown", unmatched_detection.get("direction_group", "unknown")}:
                rejected += 1
                continue
            candidates.append(entry)
        self.total_candidates_before_scoring += len(candidates)
        self.total_rejected_before_scoring += rejected
        self.maximum_candidates_per_unmatched_detection = max(self.maximum_candidates_per_unmatched_detection, len(candidates))
        return candidates

    def build_report(self) -> dict[str, Any]:
        return {
            "status": "success",
            "candidate_queries": self.candidate_queries,
            "average_candidates_per_unmatched_detection": round(self.total_candidates_before_scoring / max(self.candidate_queries, 1), 6),
            "maximum_candidates_per_unmatched_detection": self.maximum_candidates_per_unmatched_detection,
            "candidates_rejected_before_scoring": self.total_rejected_before_scoring,
        }

