from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .crop_artifacts import CompletedTrackCropBundle
from .crop_pipeline import CropPipelineReport, SequentialCropCollectionPipeline
from .crop_selection import FinalBestCropSelector, SelectedTrackCropSet
from .crop_selection_artifacts import CropSelectionArtifactSink
from .crop_selection_metrics import build_selection_summary
from .serialization import dataclass_to_dict, read_jsonl, write_json


@dataclass(frozen=True)
class BestCropSelectionPipelineReport:
    run_id: str
    mode: str
    source_path: str
    source_id: str
    tracking_backend: str
    completed_crop_bundles: int
    selection_summary: dict[str, Any]
    crop_collection_report: dict[str, Any] | None = None
    selected_track_crop_sets: list[SelectedTrackCropSet] = field(default_factory=list)
    runtime_sec: float = 0.0
    sink_closed: bool = False
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


class SequentialBestCropSelectionPipeline:
    """Run crop collection and final best-crop selection in strict sequential order."""

    def __init__(
        self,
        *,
        crop_collection_pipeline: SequentialCropCollectionPipeline,
        selector: FinalBestCropSelector,
        sink: CropSelectionArtifactSink | None = None,
        mode: str = "real_best_crop_selection",
    ) -> None:
        self.crop_collection_pipeline = crop_collection_pipeline
        self.selector = selector
        self.sink = sink
        self.mode = mode
        self.last_report: BestCropSelectionPipelineReport | None = None

    def run(self) -> BestCropSelectionPipelineReport:
        started_at = time.perf_counter()
        sink_closed = False
        errors: list[str] = []
        crop_report: CropPipelineReport | None = None
        results: list[SelectedTrackCropSet] = []
        try:
            crop_report = self.crop_collection_pipeline.run()
            bundles = list(self.crop_collection_pipeline.completed_bundles)
            results = select_completed_crop_bundles(
                bundles,
                selector=self.selector,
                sink=self.sink,
            )
            summary = build_selection_summary(results, primary_target=self.selector.config.primary_crop_count)
            if self.sink is not None:
                self.sink.write_summary(summary)
        except Exception as exc:
            errors.append(str(exc))
            raise
        finally:
            if self.sink is not None:
                self.sink.close()
                sink_closed = bool(getattr(self.sink, "closed", False))
        self.last_report = BestCropSelectionPipelineReport(
            run_id=crop_report.run_id if crop_report is not None else "unknown",
            mode=self.mode,
            source_path=crop_report.source_path if crop_report is not None else "",
            source_id=crop_report.source_id if crop_report is not None else "",
            tracking_backend=crop_report.tracking_backend if crop_report is not None else "",
            completed_crop_bundles=len(self.crop_collection_pipeline.completed_bundles),
            selection_summary=build_selection_summary(results, primary_target=self.selector.config.primary_crop_count),
            crop_collection_report=crop_report.to_dict() if crop_report is not None else None,
            selected_track_crop_sets=results,
            runtime_sec=round(time.perf_counter() - started_at, 6),
            sink_closed=sink_closed,
            errors=errors,
        )
        return self.last_report


def select_completed_crop_bundles(
    bundles: Sequence[CompletedTrackCropBundle],
    *,
    selector: FinalBestCropSelector,
    sink: CropSelectionArtifactSink | None = None,
) -> list[SelectedTrackCropSet]:
    results: list[SelectedTrackCropSet] = []
    for bundle in sorted(bundles, key=lambda item: (item.source_id, item.track_id, item.track_generation)):
        result = selector.select(bundle)
        results.append(result)
        if sink is not None:
            sink.write_result(result)
    return results


def run_selection_for_existing_bundles(
    *,
    run_id: str,
    mode: str,
    bundles: Sequence[CompletedTrackCropBundle],
    selector: FinalBestCropSelector,
    sink: CropSelectionArtifactSink | None,
    source_path: str = "",
    source_id: str = "",
    tracking_backend: str = "",
) -> BestCropSelectionPipelineReport:
    started_at = time.perf_counter()
    sink_closed = False
    try:
        results = select_completed_crop_bundles(bundles, selector=selector, sink=sink)
        summary = build_selection_summary(results, primary_target=selector.config.primary_crop_count)
        if sink is not None:
            sink.write_summary(summary)
    finally:
        if sink is not None:
            sink.close()
            sink_closed = bool(getattr(sink, "closed", False))
    return BestCropSelectionPipelineReport(
        run_id=run_id,
        mode=mode,
        source_path=source_path,
        source_id=source_id,
        tracking_backend=tracking_backend,
        completed_crop_bundles=len(bundles),
        selection_summary=summary,
        selected_track_crop_sets=results,
        runtime_sec=round(time.perf_counter() - started_at, 6),
        sink_closed=sink_closed,
    )


def finalize_step6_artifacts(run_dir: str | Path, report: BestCropSelectionPipelineReport) -> None:
    base = Path(run_dir)
    write_json(base / "reports" / "step6_best_crop_pipeline_report.json", report)
    read_jsonl(base / "06_selected_crops" / "selected_track_crop_sets.jsonl")
    read_jsonl(base / "06_selected_crops" / "selected_primary_crops.jsonl")
    read_jsonl(base / "06_selected_crops" / "selected_fallback_crops.jsonl")
    read_jsonl(base / "06_selected_crops" / "crop_selection_rejections.jsonl")
    read_jsonl(base / "06_selected_crops" / "selected_crop_jobs.jsonl")
