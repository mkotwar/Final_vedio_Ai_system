from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .plate_diagnostics import TrackPlateDiagnosticResult
from .serialization import to_json_safe, write_json


class PlateDiagnosticArtifactSink:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.diagnostic_dir = self.run_dir / "07_5_plate_diagnostics"
        self.report_dir = self.run_dir / "reports"
        self.diagnostic_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self._handles = {
            "attempts": (self.diagnostic_dir / "plate_diagnostic_attempts.jsonl").open("w", encoding="utf-8", newline="\n"),
            "raw_boxes": (self.diagnostic_dir / "raw_plate_box_diagnostics.jsonl").open("w", encoding="utf-8", newline="\n"),
            "tracks": (self.diagnostic_dir / "track_plate_diagnostic_results.jsonl").open("w", encoding="utf-8", newline="\n"),
        }
        self.counts = defaultdict(int)
        self.closed = False

    def write_result(self, result: TrackPlateDiagnosticResult) -> None:
        for attempt in result.attempts:
            self._write("attempts", attempt)
            for raw_box in attempt.raw_boxes:
                self._write("raw_boxes", raw_box)
        self._write("tracks", result)

    def write_summary(self, summary: dict[str, Any]) -> None:
        enriched = dict(summary)
        enriched["artifact_counts"] = dict(self.counts)
        write_json(self.report_dir / "plate_diagnostic_summary.json", enriched)
        write_json(self.report_dir / "step75_plate_diagnostic_report.json", enriched)

    def close(self) -> None:
        if self.closed:
            return
        for handle in self._handles.values():
            handle.flush()
            handle.close()
        self.closed = True

    def _write(self, name: str, value: Any) -> None:
        if self.closed:
            raise RuntimeError("Cannot write to closed PlateDiagnosticArtifactSink.")
        self._handles[name].write(json.dumps(to_json_safe(value), ensure_ascii=False, sort_keys=True))
        self._handles[name].write("\n")
        self._handles[name].flush()
        self.counts[name] += 1
