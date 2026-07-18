from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .crop_artifacts import CompletedTrackCropBundle
from .crop_selection import SelectedCrop, SelectedTrackCropSet
from .schemas import BoundingBox, CropCandidate, CropQualityMetrics
from .serialization import to_json_safe, write_json


def _cv2() -> Any:
    try:
        import cv2  # type: ignore

        return cv2
    except Exception:
        return None


def _bbox_from_value(value: Any) -> BoundingBox:
    if isinstance(value, BoundingBox):
        return value
    if isinstance(value, dict):
        return BoundingBox(float(value["x1"]), float(value["y1"]), float(value["x2"]), float(value["y2"]))
    items = list(value)
    return BoundingBox(float(items[0]), float(items[1]), float(items[2]), float(items[3]))


def _quality_from_dict(value: dict[str, Any]) -> CropQualityMetrics:
    return CropQualityMetrics(
        detection_confidence=float(value.get("detection_confidence", 0.0) or 0.0),
        bbox_area_ratio=float(value.get("bbox_area_ratio", 0.0) or 0.0),
        sharpness=value.get("sharpness"),
        brightness=value.get("brightness"),
        edge_touching=bool(value.get("edge_touching", False)),
        occlusion_score=value.get("occlusion_score"),
        plate_visibility_score=value.get("plate_visibility_score"),
        combined_score=value.get("combined_score"),
        crop_width=value.get("crop_width"),
        crop_height=value.get("crop_height"),
        contrast=value.get("contrast"),
        crop_completeness=value.get("crop_completeness"),
        padding_clipped=bool(value.get("padding_clipped", False)),
        preliminary_score=value.get("preliminary_score"),
    )


def _candidate_from_dict(value: dict[str, Any]) -> CropCandidate:
    return CropCandidate(
        track_id=int(value["track_id"]),
        frame_index=int(value["frame_index"]),
        timestamp_sec=float(value["timestamp_sec"]),
        bbox=_bbox_from_value(value["bbox"]),
        full_frame_path=value.get("full_frame_path"),
        vehicle_crop_path=value.get("vehicle_crop_path"),
        quality=_quality_from_dict(value["quality"]),
        is_primary=bool(value.get("is_primary", False)),
        is_fallback=bool(value.get("is_fallback", False)),
        source_id=value.get("source_id"),
        track_generation=int(value.get("track_generation", 0) or 0),
        source_track_id=value.get("source_track_id"),
        crop_bbox=_bbox_from_value(value.get("crop_bbox") or value["bbox"]),
        class_name=value.get("class_name"),
        detection_confidence=value.get("detection_confidence"),
        preliminary_rank_score=value.get("preliminary_rank_score"),
        retention_reason=value.get("retention_reason"),
        metadata=dict(value.get("metadata", {})),
    )


def bundle_from_dict(value: dict[str, Any]) -> CompletedTrackCropBundle:
    candidates = [_candidate_from_dict(item) for item in list(value.get("candidates", []))]
    return CompletedTrackCropBundle(
        source_id=str(value["source_id"]),
        track_id=int(value["track_id"]),
        track_generation=int(value.get("track_generation", 0) or 0),
        source_track_id=value.get("source_track_id"),
        completion_reason=value.get("completion_reason"),
        lifecycle_status=str(value.get("lifecycle_status", "completed")),
        observation_count=int(value.get("observation_count", len(candidates)) or 0),
        retained_candidate_count=int(value.get("retained_candidate_count", len(candidates)) or 0),
        candidates=candidates,
        metadata=dict(value.get("metadata", {})),
    )


def read_completed_crop_bundles(path: str | Path) -> list[CompletedTrackCropBundle]:
    bundle_path = Path(path)
    if bundle_path.is_dir():
        bundle_path = bundle_path / "05_crops" / "completed_track_crop_bundles.jsonl"
    bundles: list[CompletedTrackCropBundle] = []
    for line in bundle_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        bundles.append(bundle_from_dict(json.loads(line)))
    return bundles


class CropSelectionArtifactSink:
    """Write Step 6 selected crop artifacts."""

    def __init__(self, run_dir: str | Path, *, create_previews: bool = True) -> None:
        self.run_dir = Path(run_dir)
        self.selection_dir = self.run_dir / "06_selected_crops"
        self.preview_dir = self.selection_dir / "previews"
        self.report_dir = self.run_dir / "reports"
        self.create_previews = create_previews
        self.selection_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        if create_previews:
            self.preview_dir.mkdir(parents=True, exist_ok=True)
        self._handles = {
            "sets": (self.selection_dir / "selected_track_crop_sets.jsonl").open("w", encoding="utf-8", newline="\n"),
            "primary": (self.selection_dir / "selected_primary_crops.jsonl").open("w", encoding="utf-8", newline="\n"),
            "fallback": (self.selection_dir / "selected_fallback_crops.jsonl").open("w", encoding="utf-8", newline="\n"),
            "rejections": (self.selection_dir / "crop_selection_rejections.jsonl").open("w", encoding="utf-8", newline="\n"),
            "jobs": (self.selection_dir / "selected_crop_jobs.jsonl").open("w", encoding="utf-8", newline="\n"),
        }
        self.counts = {"sets": 0, "primary": 0, "fallback": 0, "rejections": 0, "jobs": 0, "previews": 0, "preview_failures": 0}
        self.preview_paths: list[str] = []
        self.preview_failures: list[dict[str, Any]] = []
        self.closed = False

    def write_result(self, result: SelectedTrackCropSet) -> None:
        self._write("sets", result)
        for crop in result.primary_crops:
            self._write("primary", crop)
        if result.fallback_crop is not None:
            self._write("fallback", result.fallback_crop)
        for job in result.to_crop_jobs():
            self._write("jobs", job)
        for reason, count in result.rejection_reason_counts.items():
            self._write(
                "rejections",
                {
                    "source_id": result.source_id,
                    "track_id": result.track_id,
                    "track_generation": result.track_generation,
                    "reason": reason,
                    "count": count,
                },
            )
        if self.create_previews:
            self._write_preview(result)

    def write_summary(self, payload: dict[str, Any]) -> None:
        enriched = dict(payload)
        enriched["artifact_counts"] = dict(self.counts)
        enriched["preview_paths"] = list(self.preview_paths)
        enriched["preview_failures"] = list(self.preview_failures)
        write_json(self.report_dir / "crop_selection_summary.json", enriched)
        write_json(self.report_dir / "step6_best_crop_report.json", enriched)

    def close(self) -> None:
        if self.closed:
            return
        for handle in self._handles.values():
            handle.flush()
            handle.close()
        self.closed = True

    def _write(self, name: str, value: Any) -> None:
        if self.closed:
            raise RuntimeError("Cannot write to closed CropSelectionArtifactSink.")
        self._handles[name].write(json.dumps(to_json_safe(value), ensure_ascii=False, sort_keys=True))
        self._handles[name].write("\n")
        self._handles[name].flush()
        self.counts[name] += 1

    def _write_preview(self, result: SelectedTrackCropSet) -> None:
        crops = list(result.primary_crops)
        if result.fallback_crop is not None:
            crops.append(result.fallback_crop)
        if not crops:
            return
        cv2 = _cv2()
        if cv2 is None:
            self._preview_failed(result, "cv2_unavailable")
            return
        tiles = []
        for crop in crops:
            if not crop.vehicle_crop_path:
                continue
            path = Path(crop.vehicle_crop_path)
            if not path.exists():
                resolved = self.run_dir / crop.vehicle_crop_path
                path = resolved if resolved.exists() else path
            image = cv2.imread(str(path))
            if image is None:
                continue
            resized = cv2.resize(image, (180, 140), interpolation=cv2.INTER_AREA)
            cv2.putText(resized, f"{crop.role} r{crop.rank} f{crop.frame_index}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(resized, f"{crop.final_score:.2f}", (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(resized)
        if not tiles:
            self._preview_failed(result, "no_readable_selected_crop_images")
            return
        sheet = tiles[0] if len(tiles) == 1 else cv2.hconcat(tiles)
        path = self.preview_dir / f"track_{result.track_id:06d}_gen_{result.track_generation:03d}_selection_preview.jpg"
        if not bool(cv2.imwrite(str(path), sheet)):
            self._preview_failed(result, "cv2_imwrite_failed")
            return
        self.preview_paths.append(str(path))
        self.counts["previews"] += 1

    def _preview_failed(self, result: SelectedTrackCropSet, reason: str) -> None:
        self.counts["preview_failures"] += 1
        self.preview_failures.append(
            {
                "source_id": result.source_id,
                "track_id": result.track_id,
                "track_generation": result.track_generation,
                "reason": reason,
            }
        )


def copy_step5_artifact_context(step5_run_dir: str | Path, step6_run_dir: str | Path) -> None:
    """Copy small JSON context folders for artifact-mode reports; crop images remain referenced in place."""

    source = Path(step5_run_dir)
    target = Path(step6_run_dir)
    for folder_name in ("01_source", "02_detections", "03_tracks", "04_lifecycle", "05_crops"):
        src = source / folder_name
        dst = target / folder_name
        if not src.exists():
            continue
        if folder_name == "05_crops":
            dst.mkdir(parents=True, exist_ok=True)
            for item in src.iterdir():
                if item.is_file():
                    shutil.copy2(item, dst / item.name)
            continue
        if dst.exists():
            continue
        shutil.copytree(src, dst)
