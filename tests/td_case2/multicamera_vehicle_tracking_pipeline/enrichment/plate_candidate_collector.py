from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..models.plate_detector_runtime import PlateDetectorRuntime
from .anpr_config import AnprConfig, FallbackRegionConfig
from .plate_models import PlateCandidate, VehicleEvidenceInput


@dataclass(slots=True)
class PlateCandidateCollectionMetrics:
    detector_calls: int = 0
    detector_candidates: int = 0
    heuristic_candidates: int = 0
    padded_crops_examined: int = 0
    original_frame_regions_examined: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)

    def record_rejection(self, reason: str) -> None:
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1


class PlateCandidateCollector:
    def __init__(self, *, detector_runtime: PlateDetectorRuntime, config: AnprConfig, artifact_root: Path) -> None:
        self.detector_runtime = detector_runtime
        self.config = config
        self.artifact_root = artifact_root.resolve()
        self.metrics = PlateCandidateCollectionMetrics()

    def collect(self, vehicle_evidence: Sequence[VehicleEvidenceInput]) -> list[PlateCandidate]:
        try:
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("OpenCV is required for plate candidate collection.") from exc
        candidates: list[PlateCandidate] = []
        candidate_index = 0
        for evidence in vehicle_evidence:
            image_sources = self._build_image_sources(evidence)
            evidence_candidates_before = len(candidates)
            for source in image_sources:
                image = cv2.imread(str(source.local_file_path))
                if image is None:
                    self.metrics.record_rejection("image_load_failed")
                    continue
                self.metrics.detector_calls += 1
                detections = self.detector_runtime.detect(source.local_file_path)
                self.metrics.detector_candidates += len(detections)
                for detection in detections:
                    candidate = self._build_detection_candidate(
                        image=image,
                        source=source,
                        detection_bbox=detection.bbox_xyxy,
                        detector_confidence=detection.confidence,
                        class_id=detection.class_id,
                        class_name=detection.class_name,
                        candidate_index=candidate_index + 1,
                    )
                    if candidate is None:
                        continue
                    candidate_index += 1
                    candidates.append(candidate)
            if len(candidates) == evidence_candidates_before:
                for heuristic_candidate in self._build_heuristic_candidates(evidence=evidence, candidate_index_start=candidate_index + 1):
                    candidate_index += 1
                    self.metrics.heuristic_candidates += 1
                    candidates.append(heuristic_candidate)
        return candidates

    def _build_image_sources(self, evidence: VehicleEvidenceInput) -> list[VehicleEvidenceInput]:
        sources = [evidence]
        if self.config.fallback.enable_padded_vehicle_crop:
            padded = self._build_padded_vehicle_crop(evidence)
            if padded is not None:
                self.metrics.padded_crops_examined += 1
                sources.append(padded)
        if self.config.fallback.enable_original_frame_region:
            original = self._build_original_frame_region(evidence)
            if original is not None:
                self.metrics.original_frame_regions_examined += 1
                sources.append(original)
        return sources

    def _build_padded_vehicle_crop(self, evidence: VehicleEvidenceInput) -> VehicleEvidenceInput | None:
        try:
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("OpenCV is required for padded plate candidate collection.") from exc
        image = cv2.imread(str(evidence.local_file_path))
        if image is None:
            self.metrics.record_rejection("padded_source_missing")
            return None
        height, width = image.shape[:2]
        pad_x = max(1, int(round(width * float(self.config.fallback.vehicle_crop_padding_ratio))))
        pad_y = max(1, int(round(height * float(self.config.fallback.vehicle_crop_padding_ratio))))
        padded = cv2.copyMakeBorder(image, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)
        output_path = _artifact_output_directory(evidence.local_file_path, self.config.media.artifact_subdirectory) / f"{evidence.local_file_path.stem}_padded.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), padded):
            self.metrics.record_rejection("padded_write_failed")
            return None
        return VehicleEvidenceInput(
            track_uuid=evidence.track_uuid,
            camera_code=evidence.camera_code,
            source_vehicle_role=evidence.source_vehicle_role,
            source_vehicle_storage_uri=evidence.source_vehicle_storage_uri,
            local_file_path=output_path.resolve(),
            frame_number=evidence.frame_number,
            video_time_seconds=evidence.video_time_seconds,
            confidence=evidence.confidence,
            bbox_xyxy=(0.0, 0.0, float(padded.shape[1]), float(padded.shape[0])),
            crop_width=int(padded.shape[1]),
            crop_height=int(padded.shape[0]),
            sharpness_score=evidence.sharpness_score,
            edge_penalty=evidence.edge_penalty,
            overall_score=evidence.overall_score,
            source_image_kind="PADDED_VEHICLE_CROP",
            metadata={**evidence.metadata, "padding_ratio": self.config.fallback.vehicle_crop_padding_ratio},
        )

    def _build_original_frame_region(self, evidence: VehicleEvidenceInput) -> VehicleEvidenceInput | None:
        try:
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("OpenCV is required for original-frame ANPR fallback.") from exc
        source_path = Path(str(evidence.metadata.get("source_path", "") or "")).expanduser()
        if not source_path.exists():
            self.metrics.record_rejection("original_source_missing")
            return None
        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            self.metrics.record_rejection("original_source_unreadable")
            return None
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(evidence.frame_number))
            success, frame = capture.read()
        finally:
            capture.release()
        if not success or frame is None:
            self.metrics.record_rejection("original_frame_missing")
            return None
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = _expand_bbox(
            evidence.bbox_xyxy,
            width=frame_width,
            height=frame_height,
            padding_ratio=float(self.config.fallback.vehicle_crop_padding_ratio),
        )
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            self.metrics.record_rejection("original_region_empty")
            return None
        output_path = _artifact_output_directory(evidence.local_file_path, self.config.media.artifact_subdirectory) / f"{evidence.local_file_path.stem}_frame_region.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), region):
            self.metrics.record_rejection("original_region_write_failed")
            return None
        return VehicleEvidenceInput(
            track_uuid=evidence.track_uuid,
            camera_code=evidence.camera_code,
            source_vehicle_role=evidence.source_vehicle_role,
            source_vehicle_storage_uri=evidence.source_vehicle_storage_uri,
            local_file_path=output_path.resolve(),
            frame_number=evidence.frame_number,
            video_time_seconds=evidence.video_time_seconds,
            confidence=evidence.confidence,
            bbox_xyxy=(0.0, 0.0, float(region.shape[1]), float(region.shape[0])),
            crop_width=int(region.shape[1]),
            crop_height=int(region.shape[0]),
            sharpness_score=evidence.sharpness_score,
            edge_penalty=evidence.edge_penalty,
            overall_score=evidence.overall_score,
            source_image_kind="ORIGINAL_FRAME_REGION",
            source_frame_path=source_path.resolve(),
            source_frame_region_bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
            metadata={**evidence.metadata, "source_frame_region_bbox_xyxy": [x1, y1, x2, y2]},
        )

    def _build_detection_candidate(
        self,
        *,
        image,
        source: VehicleEvidenceInput,
        detection_bbox: tuple[float, float, float, float],
        detector_confidence: float,
        class_id: int | None,
        class_name: str | None,
        candidate_index: int,
    ) -> PlateCandidate | None:
        try:
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("OpenCV is required for plate candidate collection.") from exc
        x1, y1, x2, y2 = (int(round(value)) for value in detection_bbox)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            self.metrics.record_rejection("empty_after_clipping")
            return None
        height, width = crop.shape[:2]
        area = int(width * height)
        aspect_ratio = float(width) / float(height)
        if width < self.config.plate_selection.minimum_width:
            self.metrics.record_rejection("width_below_minimum")
            return None
        if height < self.config.plate_selection.minimum_height:
            self.metrics.record_rejection("height_below_minimum")
            return None
        if area < self.config.plate_selection.minimum_area:
            self.metrics.record_rejection("area_below_minimum")
            return None
        if aspect_ratio < self.config.plate_selection.minimum_aspect_ratio:
            self.metrics.record_rejection("aspect_ratio_below_minimum")
            return None
        if aspect_ratio > self.config.plate_selection.maximum_aspect_ratio:
            self.metrics.record_rejection("aspect_ratio_above_maximum")
            return None
        sharpness_score = _sharpness_score(crop)
        edge_penalty = _edge_penalty(detection_bbox, image_width=image.shape[1], image_height=image.shape[0])
        output_path = _artifact_output_directory(source.local_file_path, self.config.media.artifact_subdirectory) / f"candidate_{candidate_index:03d}.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), crop)
        relative_storage_uri = output_path.resolve().relative_to(self.artifact_root).as_posix()
        return PlateCandidate(
            track_uuid=source.track_uuid,
            camera_code=source.camera_code,
            source_vehicle_role=source.source_vehicle_role,
            source_vehicle_storage_uri=source.source_vehicle_storage_uri,
            plate_bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
            detector_confidence=detector_confidence,
            crop_width=width,
            crop_height=height,
            area=area,
            aspect_ratio=aspect_ratio,
            sharpness_score=sharpness_score,
            edge_penalty=edge_penalty,
            overall_score=source.overall_score,
            local_file_path=output_path.resolve(),
            relative_storage_uri=relative_storage_uri,
            frame_number=source.frame_number,
            video_time_seconds=source.video_time_seconds,
            candidate_source="DETECTOR",
            source_image_kind=source.source_image_kind,
            metadata={
                "source_role": source.source_vehicle_role,
                "source_image_kind": source.source_image_kind,
                "class_id": class_id,
                "class_name": class_name,
            },
        )

    def _build_heuristic_candidates(self, *, evidence: VehicleEvidenceInput, candidate_index_start: int) -> list[PlateCandidate]:
        try:
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("OpenCV is required for heuristic plate regions.") from exc
        if not self.config.fallback.enabled:
            return []
        class_name = str(evidence.metadata.get("track_class_name", "") or "").strip().upper()
        regions = self.config.fallback.heuristic_regions_by_class.get(class_name, ())
        if not regions:
            return []
        image = cv2.imread(str(evidence.local_file_path))
        if image is None:
            self.metrics.record_rejection("heuristic_source_missing")
            return []
        candidates: list[PlateCandidate] = []
        for offset, region in enumerate(regions):
            candidate = self._build_heuristic_candidate(
                image=image,
                evidence=evidence,
                region=region,
                candidate_index=candidate_index_start + offset,
            )
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def _build_heuristic_candidate(
        self,
        *,
        image,
        evidence: VehicleEvidenceInput,
        region: FallbackRegionConfig,
        candidate_index: int,
    ) -> PlateCandidate | None:
        try:
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("OpenCV is required for heuristic plate regions.") from exc
        image_height, image_width = image.shape[:2]
        x1 = int(round(image_width * float(region.x_min_ratio)))
        y1 = int(round(image_height * float(region.y_min_ratio)))
        x2 = int(round(image_width * float(region.x_max_ratio)))
        y2 = int(round(image_height * float(region.y_max_ratio)))
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            self.metrics.record_rejection("heuristic_region_empty")
            return None
        height, width = crop.shape[:2]
        area = int(width * height)
        if width < self.config.plate_selection.minimum_width or height < self.config.plate_selection.minimum_height:
            self.metrics.record_rejection("heuristic_region_too_small")
            return None
        output_path = _artifact_output_directory(evidence.local_file_path, self.config.media.artifact_subdirectory) / f"heuristic_{candidate_index:03d}.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), crop)
        relative_storage_uri = output_path.resolve().relative_to(self.artifact_root).as_posix()
        sharpness_score = _sharpness_score(crop)
        edge_penalty = _edge_penalty((x1, y1, x2, y2), image_width=image_width, image_height=image_height)
        return PlateCandidate(
            track_uuid=evidence.track_uuid,
            camera_code=evidence.camera_code,
            source_vehicle_role=evidence.source_vehicle_role,
            source_vehicle_storage_uri=evidence.source_vehicle_storage_uri,
            plate_bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
            detector_confidence=0.0,
            crop_width=width,
            crop_height=height,
            area=area,
            aspect_ratio=float(width) / float(height),
            sharpness_score=sharpness_score,
            edge_penalty=edge_penalty,
            overall_score=max(0.0, evidence.overall_score - self.config.plate_selection.heuristic_region_penalty),
            local_file_path=output_path.resolve(),
            relative_storage_uri=relative_storage_uri,
            frame_number=evidence.frame_number,
            video_time_seconds=evidence.video_time_seconds,
            candidate_source="HEURISTIC_REGION",
            source_image_kind=evidence.source_image_kind,
            heuristic_region_name=region.name,
            metadata={
                "source_role": evidence.source_vehicle_role,
                "source_image_kind": evidence.source_image_kind,
                "heuristic_region_name": region.name,
            },
        )


def _sharpness_score(image) -> float:
    import cv2  # type: ignore

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return max(0.0, min(1.0, variance / 1000.0))


def _edge_penalty(bbox_xyxy: tuple[float, float, float, float], *, image_width: int, image_height: int) -> float:
    x1, y1, x2, y2 = bbox_xyxy
    penalty = 0.0
    if x1 <= 0 or y1 <= 0:
        penalty += 0.5
    if x2 >= image_width or y2 >= image_height:
        penalty += 0.5
    return min(1.0, penalty)


def _expand_bbox(
    bbox_xyxy: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    pad_x = box_width * padding_ratio
    pad_y = box_height * padding_ratio
    return (
        max(0, int(round(x1 - pad_x))),
        max(0, int(round(y1 - pad_y))),
        min(width, int(round(x2 + pad_x))),
        min(height, int(round(y2 + pad_y))),
    )


def _artifact_output_directory(local_file_path: Path, artifact_subdirectory: str) -> Path:
    parent = local_file_path.parent
    return parent if parent.name == artifact_subdirectory else parent / artifact_subdirectory
