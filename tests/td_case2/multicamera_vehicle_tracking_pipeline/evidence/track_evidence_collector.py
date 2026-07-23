from __future__ import annotations

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
        for observation in observations:
            if observation.track_uuid == "":
                continue
            if float(observation.confidence) < self.config.minimum_detection_confidence:
                continue
            crop_result = self._extract_crop(frame, frame_width, frame_height, observation.bbox_xyxy)
            if crop_result is None:
                continue
            crop, adjusted_bbox = crop_result
            crop_height, crop_width = crop.shape[:2]
            if crop_width < self.config.minimum_crop_width or crop_height < self.config.minimum_crop_height:
                continue
            encoded = self._encode_crop(crop)
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
            candidate = self._build_candidate(observation, adjusted_bbox, crop_width, crop_height, encoded, crop, frame_width, frame_height)
            self._consider_candidate(state, "first", candidate, condition=self.config.collect_first and "first" not in state.candidates)
            self._consider_candidate(state, "last", candidate, condition=self.config.collect_last)
            self._consider_candidate(
                state,
                "highest_confidence",
                candidate,
                condition=self.config.collect_highest_confidence and (
                    "highest_confidence" not in state.candidates or candidate.confidence > state.candidates["highest_confidence"].confidence
                ),
            )
            self._consider_candidate(
                state,
                "largest",
                candidate,
                condition=self.config.collect_largest and (
                    "largest" not in state.candidates or candidate.area > state.candidates["largest"].area
                ),
            )
            self._consider_candidate(
                state,
                "sharpest",
                candidate,
                condition=self.config.collect_sharpest and (
                    "sharpest" not in state.candidates or candidate.sharpness_score > state.candidates["sharpest"].sharpness_score
                ),
            )
            self._consider_candidate(
                state,
                "best_overall",
                candidate,
                condition=self.config.collect_best_overall and (
                    "best_overall" not in state.candidates or candidate.overall_score > state.candidates["best_overall"].overall_score
                ),
            )
            if self.config.collect_middle:
                self._consider_middle_candidate(state, candidate)

    def finalize_track(self, track: LocalVehicleTrack) -> TrackEvidencePackage | None:
        if not self.config.enabled:
            return None
        state = self._states_by_track_uuid.pop(track.track_uuid, None)
        if state is None or not state.candidates:
            return None
        output_directory = None
        candidates = dict(state.candidates)
        if self.config.save_final_selected_crops:
            output_directory = self._persist_candidates(track, candidates)
        return TrackEvidencePackage(
            run_id=self.run_id,
            camera_code=track.camera_code,
            local_track_id=track.local_track_id,
            track_uuid=track.track_uuid,
            class_name=track.class_name,
            candidates=candidates,
            output_directory=output_directory,
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
    ) -> tuple[np.ndarray, tuple[float, float, float, float]] | None:
        x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
        width = x2 - x1
        height = y2 - y1
        pad_x = width * self.config.bbox_padding_ratio
        pad_y = height * self.config.bbox_padding_ratio
        padded = (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y)
        if self.config.reject_out_of_frame_bbox and (
            padded[0] < 0.0 or padded[1] < 0.0 or padded[2] > float(frame_width) or padded[3] > float(frame_height)
        ):
            if not self.config.clamp_bbox_to_frame:
                return None
        clamped = (
            max(0.0, min(float(frame_width), padded[0])),
            max(0.0, min(float(frame_height), padded[1])),
            max(0.0, min(float(frame_width), padded[2])),
            max(0.0, min(float(frame_height), padded[3])),
        )
        ix1, iy1, ix2, iy2 = (int(round(clamped[0])), int(round(clamped[1])), int(round(clamped[2])), int(round(clamped[3])))
        if ix2 <= ix1 or iy2 <= iy1:
            return None
        crop = frame[iy1:iy2, ix1:ix2]
        if crop is None or crop.size == 0:
            return None
        return crop.copy(), clamped

    def _encode_crop(self, crop: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.config.jpeg_quality)])
        if not ok:
            raise RuntimeError("Failed to JPEG-encode evidence crop.")
        return bytes(encoded.tobytes())

    def _build_candidate(
        self,
        observation: TrackObservation,
        bbox_xyxy: tuple[float, float, float, float],
        crop_width: int,
        crop_height: int,
        encoded: bytes,
        crop: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> EvidenceCandidate:
        area = int(crop_width * crop_height)
        sharpness = self._sharpness_score(crop) if self.config.sharpness_enabled else 0.0
        edge_penalty = self._edge_penalty(bbox_xyxy, frame_width, frame_height) if self.config.edge_penalty_enabled else 0.0
        normalized_area = area / max(1.0, float(frame_width * frame_height))
        overall_score = (float(observation.confidence) * 0.5) + (normalized_area * 0.25) + (sharpness * 0.25) - edge_penalty
        return EvidenceCandidate(
            candidate_type="generic",
            frame_number=observation.frame_number,
            video_time_seconds=observation.video_time_seconds,
            confidence=float(observation.confidence),
            bbox_xyxy=bbox_xyxy,
            crop_width=crop_width,
            crop_height=crop_height,
            area=area,
            sharpness_score=sharpness,
            edge_penalty=edge_penalty,
            overall_score=overall_score,
            encoded_jpeg=encoded,
        )

    def _consider_candidate(self, state: _TrackEvidenceState, candidate_name: str, candidate: EvidenceCandidate, *, condition: bool) -> None:
        if not condition:
            return
        state.candidates[candidate_name] = EvidenceCandidate(
            candidate_type=candidate_name,
            frame_number=candidate.frame_number,
            video_time_seconds=candidate.video_time_seconds,
            confidence=candidate.confidence,
            bbox_xyxy=candidate.bbox_xyxy,
            crop_width=candidate.crop_width,
            crop_height=candidate.crop_height,
            area=candidate.area,
            sharpness_score=candidate.sharpness_score,
            edge_penalty=candidate.edge_penalty,
            overall_score=candidate.overall_score,
            encoded_jpeg=candidate.encoded_jpeg,
            file_path=candidate.file_path,
        )

    def _consider_middle_candidate(self, state: _TrackEvidenceState, candidate: EvidenceCandidate) -> None:
        if state.first_frame_number is None or state.last_frame_number is None:
            return
        desired_mid = (state.first_frame_number + state.last_frame_number) / 2.0
        existing = state.candidates.get("middle")
        if existing is None or abs(candidate.frame_number - desired_mid) < abs(existing.frame_number - desired_mid):
            self._consider_candidate(state, "middle", candidate, condition=True)

    @staticmethod
    def _sharpness_score(crop: np.ndarray) -> float:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return variance

    @staticmethod
    def _edge_penalty(bbox_xyxy: tuple[float, float, float, float], frame_width: int, frame_height: int) -> float:
        x1, y1, x2, y2 = bbox_xyxy
        margin_x = min(x1, float(frame_width) - x2)
        margin_y = min(y1, float(frame_height) - y2)
        normalized = min(margin_x / max(1.0, frame_width), margin_y / max(1.0, frame_height))
        return max(0.0, 0.1 - normalized)

    def _persist_candidates(self, track: LocalVehicleTrack, candidates: dict[str, EvidenceCandidate]) -> str:
        base_dir = Path(self.config.output_root) / self.run_id / track.camera_code / f"track_{track.local_track_id:06d}" / track.track_uuid.replace(":", "_")
        base_dir.mkdir(parents=True, exist_ok=True)
        for candidate_name, candidate in list(candidates.items()):
            target_path = base_dir / f"{candidate_name}.jpg"
            target_path.write_bytes(candidate.encoded_jpeg)
            candidates[candidate_name] = EvidenceCandidate(
                candidate_type=candidate.candidate_type,
                frame_number=candidate.frame_number,
                video_time_seconds=candidate.video_time_seconds,
                confidence=candidate.confidence,
                bbox_xyxy=candidate.bbox_xyxy,
                crop_width=candidate.crop_width,
                crop_height=candidate.crop_height,
                area=candidate.area,
                sharpness_score=candidate.sharpness_score,
                edge_penalty=candidate.edge_penalty,
                overall_score=candidate.overall_score,
                encoded_jpeg=candidate.encoded_jpeg,
                file_path=str(target_path),
            )
        return str(base_dir)
