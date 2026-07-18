from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .anpr_schemas import TrackAnprColourResult
from .crop_selection import SelectedCropJob
from .serialization import to_json_safe, write_json


def read_selected_crop_jobs(path: str | Path) -> list[SelectedCropJob]:
    job_path = Path(path)
    if job_path.is_dir():
        job_path = job_path / "06_selected_crops" / "selected_crop_jobs.jsonl"
    jobs: list[SelectedCropJob] = []
    for line in job_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        jobs.append(
            SelectedCropJob(
                source_id=str(value["source_id"]),
                track_id=int(value["track_id"]),
                track_generation=int(value.get("track_generation", 0) or 0),
                source_track_id=value.get("source_track_id"),
                object_class=value.get("object_class"),
                lifecycle_completion_reason=value.get("lifecycle_completion_reason"),
                crop_role=str(value["crop_role"]),
                crop_rank=int(value["crop_rank"]),
                frame_index=int(value["frame_index"]),
                timestamp_sec=float(value["timestamp_sec"]),
                vehicle_crop_path=str(value["vehicle_crop_path"]),
                full_frame_path=value.get("full_frame_path"),
                selection_score=float(value.get("selection_score", 0.0) or 0.0),
                quality_warnings=list(value.get("quality_warnings", [])),
                metadata=dict(value.get("metadata", {})),
            )
        )
    return jobs


class AnprArtifactSink:
    """Write Step 7 raw ANPR and colour artifacts."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.anpr_dir = self.run_dir / "07_anpr"
        self.report_dir = self.run_dir / "reports"
        self.anpr_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self._handles = {
            "plate_candidates": (self.anpr_dir / "plate_detection_candidates.jsonl").open("w", encoding="utf-8", newline="\n"),
            "ocr": (self.anpr_dir / "florence_ocr_results.jsonl").open("w", encoding="utf-8", newline="\n"),
            "colour": (self.anpr_dir / "florence_colour_results.jsonl").open("w", encoding="utf-8", newline="\n"),
            "tracks": (self.anpr_dir / "track_anpr_colour_results.jsonl").open("w", encoding="utf-8", newline="\n"),
            "jobs": (self.anpr_dir / "step7_selected_crop_jobs.jsonl").open("w", encoding="utf-8", newline="\n"),
        }
        self.counts = defaultdict(int)
        self.closed = False

    def write_result(self, result: TrackAnprColourResult) -> None:
        for job in result.selected_crop_jobs:
            self._write("jobs", job)
        for candidate in result.plate_candidates:
            self._write("plate_candidates", candidate)
        for ocr in result.ocr_results:
            self._write("ocr", ocr)
        if result.colour_result is not None:
            self._write("colour", result.colour_result)
        self._write("tracks", result)

    def write_summary(self, summary: dict[str, Any]) -> None:
        enriched = dict(summary)
        enriched["artifact_counts"] = dict(self.counts)
        write_json(self.report_dir / "anpr_colour_summary.json", enriched)
        write_json(self.report_dir / "step7_anpr_colour_report.json", enriched)

    def close(self) -> None:
        if self.closed:
            return
        for handle in self._handles.values():
            handle.flush()
            handle.close()
        self.closed = True

    def _write(self, name: str, value: Any) -> None:
        if self.closed:
            raise RuntimeError("Cannot write to closed AnprArtifactSink.")
        self._handles[name].write(json.dumps(to_json_safe(value), ensure_ascii=False, sort_keys=True))
        self._handles[name].write("\n")
        self._handles[name].flush()
        self.counts[name] += 1
