from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..detection.detection_models import DetectionPacket
from ..tracking.tracking_models import LocalVehicleTrack, TrackObservation
from .evidence_config import EvidenceConfig
from .evidence_models import EvidenceCandidate, TrackEvidencePackage


@dataclass(slots=True)
class _TrackEvidenceState:
    run_id: str
    camera_code: str
    local_track_id: int
    track_uuid: str
    class_name: str
    first_frame_number: int | None = None
    last_frame_number: int | None = None
    accepted_frame_numbers: list[int] = field(default_factory=list)
    candidates: dict[str, EvidenceCandidate] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _CropExtraction:
    crop: np.ndarray
    original_bbox_xyxy: tuple[float, float, float, float]
    expanded_bbox_xyxy: tuple[float, float, float, float]
    clipped_bbox_xyxy: tuple[float, float, float, float]
    crop_width: int
    crop_height: int
    visible_bbox_ratio: float
    crop_clipped: bool
    touches_left_edge: bool
    touches_right_edge: bool
    touches_top_edge: bool
    touches_bottom_edge: bool


class TrackEvidenceCollector:
    def __init__(self, config: EvidenceConfig, *, run_id: str) -> None:
        self.config = config
        self.run_id = run_id
        self._states_by_track_uuid: dict[str, _TrackEvidenceState] = {}

    def update(self, packet: DetectionPacket, observations: list[TrackObservation]) -> None:
        if not self.config.enabled or packet.frame is None or not observations:
            return
        frame = packet.frame
        frame_height, frame_width = frame.shape[:2]
        frame_observation_records = self._frame_observation_records(observations)
        for observation in observations:
            if observation.track_uuid == "":
                continue
            if float(observation.confidence) < self.config.minimum_detection_confidence:
                continue
            extracted = self._extract_crop(frame, frame_width, frame_height, observation.bbox_xyxy)
            if extracted is None:
                continue
            if extracted.crop_width < self.config.minimum_crop_width or extracted.crop_height < self.config.minimum_crop_height:
                continue
            encoded = self._encode_image(extracted.crop)
            state = self._states_by_track_uuid.setdefault(
                observation.track_uuid,
                _TrackEvidenceState(
                    run_id=self.run_id,
                    camera_code=observation.camera_code,
                    local_track_id=observation.local_track_id,
                    track_uuid=observation.track_uuid,
                    class_name=observation.class_name,
                ),
            )
            state.class_name = observation.class_name
            state.first_frame_number = observation.frame_number if state.first_frame_number is None else state.first_frame_number
            state.last_frame_number = observation.frame_number
            state.accepted_frame_numbers.append(observation.frame_number)
            candidate = self._build_candidate(
                observation=observation,
                extracted=extracted,
                encoded=encoded,
                crop=extracted.crop,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            self._consider_candidate(
                state,
                "first",
                candidate,
                frame=frame,
                frame_observations=frame_observation_records,
                condition=self.config.collect_first and "first" not in state.candidates,
            )
            self._consider_candidate(
                state,
                "last",
                candidate,
                frame=frame,
                frame_observations=frame_observation_records,
                condition=self.config.collect_last,
            )
            self._consider_candidate(
                state,
                "highest_confidence",
                candidate,
                frame=frame,
                frame_observations=frame_observation_records,
                condition=self.config.collect_highest_confidence
                and ("highest_confidence" not in state.candidates or candidate.confidence > state.candidates["highest_confidence"].confidence),
            )
            self._consider_candidate(
                state,
                "largest",
                candidate,
                frame=frame,
                frame_observations=frame_observation_records,
                condition=self.config.collect_largest and ("largest" not in state.candidates or candidate.area > state.candidates["largest"].area),
            )
            self._consider_candidate(
                state,
                "sharpest",
                candidate,
                frame=frame,
                frame_observations=frame_observation_records,
                condition=self.config.collect_sharpest
                and ("sharpest" not in state.candidates or candidate.sharpness_score > state.candidates["sharpest"].sharpness_score),
            )
            self._consider_candidate(
                state,
                "best_overall",
                candidate,
                frame=frame,
                frame_observations=frame_observation_records,
                condition=self.config.collect_best_overall
                and ("best_overall" not in state.candidates or candidate.overall_score > state.candidates["best_overall"].overall_score),
            )
            if self.config.collect_middle:
                self._consider_middle_candidate(state, candidate, frame=frame, frame_observations=frame_observation_records)

    def finalize_track(self, track: LocalVehicleTrack) -> TrackEvidencePackage | None:
        if not self.config.enabled:
            return None
        state = self._states_by_track_uuid.pop(track.track_uuid, None)
        if state is None or not state.candidates:
            return None
        output_directory = None
        manifest_path = None
        full_frame_metadata: dict[str, object] = {}
        candidates = dict(state.candidates)
        if self.config.save_final_selected_crops:
            output_directory, manifest_path, full_frame_metadata = self._persist_candidates(track, candidates)
        return TrackEvidencePackage(
            run_id=self.run_id,
            camera_code=track.camera_code,
            local_track_id=track.local_track_id,
            track_uuid=track.track_uuid,
            class_name=track.class_name,
            candidates=candidates,
            output_directory=output_directory,
            manifest_path=manifest_path,
            full_frame_path=full_frame_metadata.get("full_frame_path"),
            annotated_full_frame_path=full_frame_metadata.get("annotated_full_frame_path"),
            full_frame_frame_number=full_frame_metadata.get("full_frame_frame_number"),
            full_frame_video_time_seconds=full_frame_metadata.get("full_frame_video_time_seconds"),
            full_frame_bbox_xyxy=full_frame_metadata.get("full_frame_bbox_xyxy"),
            full_frame_width=full_frame_metadata.get("full_frame_width"),
            full_frame_height=full_frame_metadata.get("full_frame_height"),
        )

    def drop_camera(self, camera_code: str) -> None:
        doomed = [track_uuid for track_uuid, state in self._states_by_track_uuid.items() if state.camera_code == camera_code]
        for track_uuid in doomed:
            del self._states_by_track_uuid[track_uuid]

    def _extract_crop(
        self,
        frame: Any,
        frame_width: int,
        frame_height: int,
        bbox_xyxy: tuple[float, float, float, float],
    ) -> _CropExtraction | None:
        x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
        if x2 <= x1 or y2 <= y1:
            return None
        width = x2 - x1
        height = y2 - y1
        pad_x = max(width * self.config.padding_ratio_x, width * self.config.bbox_padding_ratio, float(self.config.minimum_padding_pixels))
        pad_y = max(height * self.config.padding_ratio_y, height * self.config.bbox_padding_ratio, float(self.config.minimum_padding_pixels))
        expanded = (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y)
        if self.config.reject_out_of_frame_bbox and not self.config.clamp_bbox_to_frame:
            if expanded[0] < 0.0 or expanded[1] < 0.0 or expanded[2] > float(frame_width) or expanded[3] > float(frame_height):
                return None
        clipped = (
            max(0.0, min(float(frame_width), expanded[0])),
            max(0.0, min(float(frame_height), expanded[1])),
            max(0.0, min(float(frame_width), expanded[2])),
            max(0.0, min(float(frame_height), expanded[3])),
        )
        if not self.config.clip_to_frame and clipped != expanded:
            return None
        ix1, iy1, ix2, iy2 = (
            int(math.floor(clipped[0])),
            int(math.floor(clipped[1])),
            int(math.ceil(clipped[2])),
            int(math.ceil(clipped[3])),
        )
        ix1 = max(0, min(frame_width, ix1))
        iy1 = max(0, min(frame_height, iy1))
        ix2 = max(0, min(frame_width, ix2))
        iy2 = max(0, min(frame_height, iy2))
        if ix2 <= ix1 or iy2 <= iy1:
            return None
        crop = frame[iy1:iy2, ix1:ix2]
        if crop is None or crop.size == 0:
            return None
        requested_area = max((expanded[2] - expanded[0]) * (expanded[3] - expanded[1]), 1.0)
        visible_bbox_ratio = max(0.0, min(1.0, ((clipped[2] - clipped[0]) * (clipped[3] - clipped[1])) / requested_area))
        crop_clipped = any(abs(clipped[index] - expanded[index]) > 1e-6 for index in range(4))
        touches_left_edge = x1 <= 0.5
        touches_right_edge = x2 >= float(frame_width) - 0.5
        touches_top_edge = y1 <= 0.5
        touches_bottom_edge = y2 >= float(frame_height) - 0.5
        return _CropExtraction(
            crop=crop.copy(),
            original_bbox_xyxy=(x1, y1, x2, y2),
            expanded_bbox_xyxy=expanded,
            clipped_bbox_xyxy=clipped,
            crop_width=ix2 - ix1,
            crop_height=iy2 - iy1,
            visible_bbox_ratio=visible_bbox_ratio,
            crop_clipped=crop_clipped,
            touches_left_edge=touches_left_edge,
            touches_right_edge=touches_right_edge,
            touches_top_edge=touches_top_edge,
            touches_bottom_edge=touches_bottom_edge,
        )

    def _encode_image(self, image: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.config.jpeg_quality)])
        if not ok:
            raise RuntimeError("Failed to JPEG-encode evidence image.")
        return bytes(encoded.tobytes())

    def _build_candidate(
        self,
        *,
        observation: TrackObservation,
        extracted: _CropExtraction,
        encoded: bytes,
        crop: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> EvidenceCandidate:
        area = int(extracted.crop_width * extracted.crop_height)
        sharpness = self._sharpness_score(crop) if self.config.sharpness_enabled else 0.0
        normalized_sharpness = self._normalized_sharpness_score(sharpness) if self.config.sharpness_enabled else 0.0
        edge_penalty = self._edge_penalty(extracted, frame_width, frame_height) if self.config.edge_penalty_enabled else 0.0
        normalized_area = math.sqrt(area / max(1.0, float(frame_width * frame_height)))
        centeredness = self._centeredness_score(extracted.original_bbox_xyxy, frame_width, frame_height)
        visibility_score = self._visibility_score(extracted, frame_width, frame_height)
        positive_score = (
            (visibility_score * self.config.visibility_weight)
            + (float(observation.confidence) * self.config.detection_confidence_weight)
            + (normalized_sharpness * self.config.sharpness_weight)
            + (centeredness * self.config.centeredness_weight)
            + (normalized_area * self.config.bbox_area_weight)
        )
        positive_weights = (
            self.config.visibility_weight
            + self.config.detection_confidence_weight
            + self.config.sharpness_weight
            + self.config.centeredness_weight
            + self.config.bbox_area_weight
        )
        overall_score = max(
            0.0,
            min(
                1.0,
                (positive_score / max(positive_weights, 1e-9)) - (edge_penalty * self.config.edge_penalty_weight),
            ),
        )
        return EvidenceCandidate(
            candidate_type="generic",
            frame_number=observation.frame_number,
            video_time_seconds=observation.video_time_seconds,
            confidence=float(observation.confidence),
            original_bbox_xyxy=extracted.original_bbox_xyxy,
            expanded_bbox_xyxy=extracted.expanded_bbox_xyxy,
            bbox_xyxy=extracted.clipped_bbox_xyxy,
            crop_width=extracted.crop_width,
            crop_height=extracted.crop_height,
            area=area,
            sharpness_score=sharpness,
            visibility_score=visibility_score,
            centeredness_score=centeredness,
            visible_bbox_ratio=extracted.visible_bbox_ratio,
            edge_penalty=edge_penalty,
            overall_score=overall_score,
            crop_clipped=extracted.crop_clipped,
            touches_left_edge=extracted.touches_left_edge,
            touches_right_edge=extracted.touches_right_edge,
            touches_top_edge=extracted.touches_top_edge,
            touches_bottom_edge=extracted.touches_bottom_edge,
            encoded_jpeg=encoded,
        )

    def _consider_candidate(
        self,
        state: _TrackEvidenceState,
        candidate_name: str,
        candidate: EvidenceCandidate,
        *,
        frame: np.ndarray,
        frame_observations: tuple[dict[str, object], ...],
        condition: bool,
    ) -> None:
        if not condition:
            return
        frame_height, frame_width = frame.shape[:2]
        state.candidates[candidate_name] = EvidenceCandidate(
            candidate_type=candidate_name,
            frame_number=candidate.frame_number,
            video_time_seconds=candidate.video_time_seconds,
            confidence=candidate.confidence,
            original_bbox_xyxy=candidate.original_bbox_xyxy,
            expanded_bbox_xyxy=candidate.expanded_bbox_xyxy,
            bbox_xyxy=candidate.bbox_xyxy,
            crop_width=candidate.crop_width,
            crop_height=candidate.crop_height,
            area=candidate.area,
            sharpness_score=candidate.sharpness_score,
            visibility_score=candidate.visibility_score,
            centeredness_score=candidate.centeredness_score,
            visible_bbox_ratio=candidate.visible_bbox_ratio,
            edge_penalty=candidate.edge_penalty,
            overall_score=candidate.overall_score,
            crop_clipped=candidate.crop_clipped,
            touches_left_edge=candidate.touches_left_edge,
            touches_right_edge=candidate.touches_right_edge,
            touches_top_edge=candidate.touches_top_edge,
            touches_bottom_edge=candidate.touches_bottom_edge,
            encoded_jpeg=candidate.encoded_jpeg,
            file_path=candidate.file_path,
            source_frame_path=candidate.source_frame_path,
            annotated_frame_path=candidate.annotated_frame_path,
            source_frame_jpeg=self._encode_image(frame),
            source_frame_width=frame_width,
            source_frame_height=frame_height,
            frame_observations=frame_observations,
        )

    def _consider_middle_candidate(
        self,
        state: _TrackEvidenceState,
        candidate: EvidenceCandidate,
        *,
        frame: np.ndarray,
        frame_observations: tuple[dict[str, object], ...],
    ) -> None:
        if state.first_frame_number is None or state.last_frame_number is None:
            return
        desired_mid = (state.first_frame_number + state.last_frame_number) / 2.0
        existing = state.candidates.get("middle")
        if existing is None or abs(candidate.frame_number - desired_mid) < abs(existing.frame_number - desired_mid):
            self._consider_candidate(state, "middle", candidate, frame=frame, frame_observations=frame_observations, condition=True)

    @staticmethod
    def _sharpness_score(crop: np.ndarray) -> float:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _normalized_sharpness_score(raw_sharpness: float) -> float:
        if raw_sharpness <= 0.0:
            return 0.0
        return raw_sharpness / (raw_sharpness + 300.0)

    @staticmethod
    def _edge_penalty(extracted: _CropExtraction, frame_width: int, frame_height: int) -> float:
        x1, y1, x2, y2 = extracted.original_bbox_xyxy
        margin_x = min(max(x1, 0.0), max(float(frame_width) - x2, 0.0))
        margin_y = min(max(y1, 0.0), max(float(frame_height) - y2, 0.0))
        normalized = min(margin_x / max(1.0, frame_width), margin_y / max(1.0, frame_height))
        clipped_penalty = 1.0 - extracted.visible_bbox_ratio
        edge_touch_penalty = 0.15 if (
            extracted.touches_left_edge or extracted.touches_right_edge or extracted.touches_top_edge or extracted.touches_bottom_edge
        ) else 0.0
        return max(0.0, min(1.0, max(0.0, 0.1 - normalized) + clipped_penalty + edge_touch_penalty))

    @staticmethod
    def _centeredness_score(bbox_xyxy: tuple[float, float, float, float], frame_width: int, frame_height: int) -> float:
        x1, y1, x2, y2 = bbox_xyxy
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        normalized_dx = abs(center_x - (float(frame_width) / 2.0)) / max(float(frame_width) / 2.0, 1.0)
        normalized_dy = abs(center_y - (float(frame_height) / 2.0)) / max(float(frame_height) / 2.0, 1.0)
        return max(0.0, 1.0 - ((normalized_dx * 0.6) + (normalized_dy * 0.4)))

    @staticmethod
    def _visibility_score(extracted: _CropExtraction, frame_width: int, frame_height: int) -> float:
        x1, y1, x2, y2 = extracted.original_bbox_xyxy
        left_margin = max(0.0, x1) / max(float(frame_width), 1.0)
        right_margin = max(0.0, float(frame_width) - x2) / max(float(frame_width), 1.0)
        top_margin = max(0.0, y1) / max(float(frame_height), 1.0)
        bottom_margin = max(0.0, float(frame_height) - y2) / max(float(frame_height), 1.0)
        tightest_margin = min(left_margin, right_margin, top_margin, bottom_margin)
        margin_score = max(0.0, min(1.0, tightest_margin / 0.08))
        return max(0.0, min(1.0, (margin_score * 0.6) + (extracted.visible_bbox_ratio * 0.4)))

    @staticmethod
    def _frame_observation_records(observations: list[TrackObservation]) -> tuple[dict[str, object], ...]:
        records = []
        for item in observations:
            records.append(
                {
                    "track_uuid": item.track_uuid,
                    "local_track_id": item.local_track_id,
                    "class_name": item.class_name,
                    "confidence": float(item.confidence),
                    "bbox_xyxy": [float(value) for value in item.bbox_xyxy],
                }
            )
        return tuple(records)

    def _persist_candidates(self, track: LocalVehicleTrack, candidates: dict[str, EvidenceCandidate]) -> tuple[str, str, dict[str, object]]:
        base_dir = Path(self.config.output_root) / self.run_id / track.camera_code / f"track_{track.local_track_id:06d}" / track.track_uuid.replace(":", "_")
        vehicle_dir = base_dir / "vehicle"
        full_frames_dir = base_dir / "full_frames"
        annotated_frames_dir = base_dir / "annotated_frames"
        for directory in (base_dir, vehicle_dir, full_frames_dir, annotated_frames_dir):
            directory.mkdir(parents=True, exist_ok=True)

        persisted_by_signature: dict[tuple[object, ...], tuple[str, str, str]] = {}
        manifest_records: dict[str, dict[str, object]] = {}
        ordered_roles = ("best_overall", "highest_confidence", "largest", "sharpest", "first", "middle", "last")

        for candidate_name in ordered_roles:
            candidate = candidates.get(candidate_name)
            if candidate is None:
                continue
            signature = (
                candidate.frame_number,
                round(candidate.confidence, 6),
                tuple(round(value, 2) for value in candidate.original_bbox_xyxy),
                tuple(round(value, 2) for value in candidate.bbox_xyxy),
            )
            if signature in persisted_by_signature:
                vehicle_path, full_frame_path, annotated_frame_path = persisted_by_signature[signature]
            else:
                vehicle_target = vehicle_dir / f"{candidate_name}.jpg"
                source_target = full_frames_dir / f"{candidate_name}.jpg"
                annotated_target = annotated_frames_dir / f"{candidate_name}.jpg"
                vehicle_target.write_bytes(candidate.encoded_jpeg)
                if candidate.source_frame_jpeg is None:
                    raise RuntimeError(f"Evidence candidate '{candidate_name}' has no source frame bytes.")
                source_target.write_bytes(candidate.source_frame_jpeg)
                annotated_bytes = self._encode_image(self._annotate_source_frame(track, candidate, candidate_name))
                annotated_target.write_bytes(annotated_bytes)
                vehicle_path = str(vehicle_target)
                full_frame_path = str(source_target)
                annotated_frame_path = str(annotated_target)
                persisted_by_signature[signature] = (vehicle_path, full_frame_path, annotated_frame_path)

            candidates[candidate_name] = EvidenceCandidate(
                candidate_type=candidate.candidate_type,
                frame_number=candidate.frame_number,
                video_time_seconds=candidate.video_time_seconds,
                confidence=candidate.confidence,
                original_bbox_xyxy=candidate.original_bbox_xyxy,
                expanded_bbox_xyxy=candidate.expanded_bbox_xyxy,
                bbox_xyxy=candidate.bbox_xyxy,
                crop_width=candidate.crop_width,
                crop_height=candidate.crop_height,
                area=candidate.area,
                sharpness_score=candidate.sharpness_score,
                visibility_score=candidate.visibility_score,
                centeredness_score=candidate.centeredness_score,
                visible_bbox_ratio=candidate.visible_bbox_ratio,
                edge_penalty=candidate.edge_penalty,
                overall_score=candidate.overall_score,
                crop_clipped=candidate.crop_clipped,
                touches_left_edge=candidate.touches_left_edge,
                touches_right_edge=candidate.touches_right_edge,
                touches_top_edge=candidate.touches_top_edge,
                touches_bottom_edge=candidate.touches_bottom_edge,
                encoded_jpeg=candidate.encoded_jpeg,
                file_path=vehicle_path,
                source_frame_path=full_frame_path,
                annotated_frame_path=annotated_frame_path,
                source_frame_jpeg=candidate.source_frame_jpeg,
                source_frame_width=candidate.source_frame_width,
                source_frame_height=candidate.source_frame_height,
                frame_observations=candidate.frame_observations,
            )
            manifest_records[candidate_name] = {
                "role": candidate_name,
                "frame_number": candidate.frame_number,
                "timestamp_seconds": candidate.video_time_seconds,
                "original_bbox": [round(value, 3) for value in candidate.original_bbox_xyxy],
                "expanded_bbox": [round(value, 3) for value in candidate.bbox_xyxy],
                "detection_confidence": round(candidate.confidence, 6),
                "class_name": track.class_name,
                "crop_clipped": candidate.crop_clipped,
                "visible_bbox_ratio": round(candidate.visible_bbox_ratio, 6),
                "touches_left_edge": candidate.touches_left_edge,
                "touches_right_edge": candidate.touches_right_edge,
                "touches_top_edge": candidate.touches_top_edge,
                "touches_bottom_edge": candidate.touches_bottom_edge,
                "source_frame": Path(full_frame_path).relative_to(base_dir).as_posix(),
                "annotated_frame": Path(annotated_frame_path).relative_to(base_dir).as_posix(),
                "vehicle_crop": Path(vehicle_path).relative_to(base_dir).as_posix(),
                "plate_crop": None,
                "sharpness_score": round(candidate.sharpness_score, 6),
                "visibility_score": round(candidate.visibility_score, 6),
                "centeredness_score": round(candidate.centeredness_score, 6),
                "overall_score": round(candidate.overall_score, 6),
            }

        manifest_path = base_dir / "evidence_manifest.json"
        manifest_payload = {
            "run_id": self.run_id,
            "camera_code": track.camera_code,
            "track_uuid": track.track_uuid,
            "local_track_id": track.local_track_id,
            "class_name": track.class_name,
            "roles": manifest_records,
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

        full_frame_metadata: dict[str, object] = {}
        representative = self._select_representative_full_frame_candidate(candidates)
        if representative is not None and representative.source_frame_path is not None:
            full_frame_metadata = {
                "full_frame_path": representative.source_frame_path,
                "annotated_full_frame_path": representative.annotated_frame_path,
                "full_frame_frame_number": representative.frame_number,
                "full_frame_video_time_seconds": representative.video_time_seconds,
                "full_frame_bbox_xyxy": representative.original_bbox_xyxy,
                "full_frame_width": representative.source_frame_width,
                "full_frame_height": representative.source_frame_height,
            }
            legacy_full_frame = base_dir / "full_frame.jpg"
            if not legacy_full_frame.exists():
                legacy_full_frame.write_bytes(Path(representative.source_frame_path).read_bytes())
            legacy_annotated = base_dir / "annotated_full_frame.jpg"
            if representative.annotated_frame_path is not None and not legacy_annotated.exists():
                legacy_annotated.write_bytes(Path(representative.annotated_frame_path).read_bytes())
        return str(base_dir), str(manifest_path), full_frame_metadata

    def _annotate_source_frame(self, track: LocalVehicleTrack, candidate: EvidenceCandidate, role_name: str) -> np.ndarray:
        if candidate.source_frame_jpeg is None:
            raise RuntimeError(f"Evidence candidate '{role_name}' is missing source-frame bytes.")
        frame_buffer = np.frombuffer(candidate.source_frame_jpeg, dtype=np.uint8)
        frame = cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Failed to decode source-frame bytes for evidence candidate '{role_name}'.")
        observations = candidate.frame_observations if self.config.annotate_all_observations else tuple(
            item for item in candidate.frame_observations if str(item.get("track_uuid", "")) == track.track_uuid
        )
        if not observations:
            observations = (
                {
                    "track_uuid": track.track_uuid,
                    "local_track_id": track.local_track_id,
                    "class_name": track.class_name,
                    "confidence": candidate.confidence,
                    "bbox_xyxy": [float(value) for value in candidate.original_bbox_xyxy],
                },
            )
        for record in observations:
            bbox = record.get("bbox_xyxy") or []
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
            is_selected = str(record.get("track_uuid", "")) == track.track_uuid
            color = (0, 215, 255) if is_selected else (120, 120, 120)
            thickness = 3 if is_selected else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            if not is_selected:
                continue
            label = self._build_selected_label(track, candidate, record)
            self._draw_label(frame, label, x1, y1, color)
        return frame

    @staticmethod
    def _build_selected_label(track: LocalVehicleTrack, candidate: EvidenceCandidate, record: dict[str, object]) -> str:
        raw_class_name = str(record.get("raw_class_name") or record.get("class_name") or track.class_name).strip().upper()
        if str(track.class_status or "").strip().upper() == "MIXED_IDENTITY":
            final_class_name = "MIXED_IDENTITY"
        else:
            final_class_name = str(track.stable_class_name or "UNKNOWN").strip().upper()
        confidence = float(record.get("confidence") or candidate.confidence)
        frame_number = int(record.get("frame_number") or candidate.frame_number)
        video_time_seconds = record.get("video_time_seconds")
        if video_time_seconds is None:
            video_time_seconds = candidate.video_time_seconds
        label = (
            f"{track.camera_code} | TRACK_{track.local_track_id} | "
            f"RAW {raw_class_name} | FINAL {final_class_name} | "
            f"{confidence:.2f} | frame {frame_number}"
        )
        if video_time_seconds is not None:
            label = f"{label} | {float(video_time_seconds):.2f}s"
        return label

    @staticmethod
    def _draw_label(frame: np.ndarray, text: str, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        label_x = max(0, min(frame.shape[1] - text_width - 6, x1))
        above_y = y1 - 8
        label_y = above_y if above_y - text_height - baseline >= 0 else min(frame.shape[0] - baseline - 2, y1 + text_height + 8)
        top_left = (label_x, max(0, label_y - text_height - baseline - 2))
        bottom_right = (min(frame.shape[1] - 1, label_x + text_width + 6), min(frame.shape[0] - 1, label_y + baseline + 2))
        cv2.rectangle(frame, top_left, bottom_right, color, -1)
        cv2.putText(frame, text, (label_x + 3, label_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    @staticmethod
    def _select_representative_full_frame_candidate(candidates: dict[str, EvidenceCandidate]) -> EvidenceCandidate | None:
        for key in ("best_overall", "highest_confidence", "largest", "sharpest", "middle", "first", "last"):
            candidate = candidates.get(key)
            if candidate is not None and candidate.source_frame_path is not None:
                return candidate
        return next((candidate for candidate in candidates.values() if candidate.source_frame_path is not None), None)
