from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..models.plate_detector_runtime import PlateDetectorRuntime
from .anpr_config import AnprConfig
from .plate_models import PlateCandidate, VehicleEvidenceInput


class PlateCandidateCollector:
    def __init__(self, *, detector_runtime: PlateDetectorRuntime, config: AnprConfig, artifact_root: Path) -> None:
        self.detector_runtime = detector_runtime
        self.config = config
        self.artifact_root = artifact_root.resolve()

    def collect(self, vehicle_evidence: Sequence[VehicleEvidenceInput]) -> list[PlateCandidate]:
        try:
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("OpenCV is required for plate candidate collection.") from exc
        candidates: list[PlateCandidate] = []
        candidate_index = 0
        for evidence in vehicle_evidence:
            image = cv2.imread(str(evidence.local_file_path))
            if image is None:
                continue
            detections = self.detector_runtime.detect(evidence.local_file_path)
            for detection in detections:
                x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox_xyxy)
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                height, width = crop.shape[:2]
                area = int(width * height)
                if width < self.config.plate_selection.minimum_width or height < self.config.plate_selection.minimum_height:
                    continue
                if area < self.config.plate_selection.minimum_area:
                    continue
                aspect_ratio = float(width) / float(height)
                sharpness_score = _sharpness_score(crop)
                edge_penalty = _edge_penalty(detection.bbox_xyxy, image_width=image.shape[1], image_height=image.shape[0])
                candidate_index += 1
                output_path = evidence.local_file_path.parent / self.config.media.artifact_subdirectory / f"candidate_{candidate_index:03d}.jpg"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), crop)
                relative_storage_uri = output_path.resolve().relative_to(self.artifact_root).as_posix()
                candidates.append(
                    PlateCandidate(
                        track_uuid=evidence.track_uuid,
                        camera_code=evidence.camera_code,
                        source_vehicle_role=evidence.source_vehicle_role,
                        source_vehicle_storage_uri=evidence.source_vehicle_storage_uri,
                        plate_bbox_xyxy=detection.bbox_xyxy,
                        detector_confidence=detection.confidence,
                        crop_width=width,
                        crop_height=height,
                        area=area,
                        aspect_ratio=aspect_ratio,
                        sharpness_score=sharpness_score,
                        edge_penalty=edge_penalty,
                        overall_score=evidence.overall_score,
                        local_file_path=output_path.resolve(),
                        relative_storage_uri=relative_storage_uri,
                        frame_number=evidence.frame_number,
                        video_time_seconds=evidence.video_time_seconds,
                        metadata={"source_role": evidence.source_vehicle_role, "class_id": detection.class_id, "class_name": detection.class_name},
                    )
                )
        return candidates


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
