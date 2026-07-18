from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import CropCollectionConfig
from .crop_artifacts import CompletedTrackCropBundle, CropImageWriter
from .crop_quality import compute_crop_quality, extract_crop
from .lifecycle import LifecycleUpdateResult
from .observations import ObservationCollectionResult, TrackIdentity, TrackObservationCollector
from .schemas import CropCandidate, TrackRecord, TrackedFramePacket
from .serialization import dataclass_to_dict


@dataclass(frozen=True)
class CropCollectionUpdateResult:
    observation_result: ObservationCollectionResult
    candidates_created: list[CropCandidate]
    completed_bundles: list[CompletedTrackCropBundle]
    rejected_count: int
    rejection_reasons: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_result": self.observation_result.to_dict(),
            "candidates_created": [dataclass_to_dict(item) for item in self.candidates_created],
            "completed_bundles": [item.to_dict() for item in self.completed_bundles],
            "rejected_count": self.rejected_count,
            "rejection_reasons": dict(sorted(self.rejection_reasons.items())),
        }


class CropCandidateCollector:
    """Collect bounded crop candidates for each lifecycle track identity."""

    def __init__(self, config: CropCollectionConfig, *, image_writer: CropImageWriter | None = None) -> None:
        self.config = config
        self.observation_collector = TrackObservationCollector(config)
        self.image_writer = image_writer
        self._candidates_by_identity: dict[TrackIdentity, list[CropCandidate]] = {}
        self._emitted_bundles: set[TrackIdentity] = set()
        self.rejection_reasons: dict[str, int] = {}
        self.total_candidates_created = 0

    def update(self, packet: TrackedFramePacket, lifecycle_result: LifecycleUpdateResult) -> CropCollectionUpdateResult:
        observation_result = self.observation_collector.collect(packet, lifecycle_result)
        created: list[CropCandidate] = []
        local_rejections: dict[str, int] = {}
        for runtime_observation in observation_result.observations:
            observation = runtime_observation.observation
            area_ratio = observation.bbox.area / max(float(packet.frame_width * packet.frame_height), 1.0)
            if area_ratio < self.config.minimum_bbox_area_ratio:
                self._count(local_rejections, "bbox_area_too_small")
                continue
            if self.config.maximum_bbox_area_ratio is not None and area_ratio > self.config.maximum_bbox_area_ratio:
                self._count(local_rejections, "bbox_area_too_large")
                continue
            extracted = extract_crop(
                runtime_observation.frame,
                observation.bbox,
                frame_width=packet.frame_width,
                frame_height=packet.frame_height,
                config=self.config,
            )
            if extracted is None:
                self._count(local_rejections, "invalid_crop")
                continue
            quality = compute_crop_quality(
                crop=extracted.crop,
                source_bbox=observation.bbox,
                extracted=extracted,
                frame_width=packet.frame_width,
                frame_height=packet.frame_height,
                detection_confidence=observation.confidence,
                config=self.config,
            )
            crop_path = self.image_writer.write_crop(observation=observation, crop=extracted.crop) if self.image_writer is not None else None
            full_frame_path = observation.full_frame_path
            if full_frame_path is None and self.image_writer is not None:
                full_frame_path = self.image_writer.write_full_frame(observation=observation, frame=runtime_observation.frame)
            candidate = CropCandidate(
                source_id=observation.source_id,
                track_id=observation.track_id,
                track_generation=observation.track_generation,
                source_track_id=observation.source_track_id,
                frame_index=observation.frame_index,
                timestamp_sec=observation.timestamp_sec,
                bbox=observation.bbox,
                crop_bbox=extracted.crop_bbox,
                full_frame_path=full_frame_path,
                vehicle_crop_path=crop_path,
                quality=quality,
                class_name=observation.class_name,
                detection_confidence=observation.confidence,
                preliminary_rank_score=quality.preliminary_score,
                retention_reason=self.config.retention_policy,
                metadata={
                    "lifecycle_status": observation.lifecycle_status.value,
                    "requested_crop_bbox": extracted.requested_bbox.to_xyxy(),
                    "padding_clipped": extracted.padding_clipped,
                },
            )
            self.total_candidates_created += 1
            created.append(candidate)
            self._retain_candidate(observation.identity, candidate)

        bundles = self.complete_tracks(lifecycle_result.newly_completed_tracks)
        self._merge_counts(local_rejections)
        return CropCollectionUpdateResult(
            observation_result=observation_result,
            candidates_created=created,
            completed_bundles=bundles,
            rejected_count=sum(local_rejections.values()),
            rejection_reasons=local_rejections,
        )

    def complete_tracks(self, tracks: list[TrackRecord] | tuple[TrackRecord, ...]) -> list[CompletedTrackCropBundle]:
        bundles: list[CompletedTrackCropBundle] = []
        for track in sorted(tracks, key=lambda item: (item.source_id, item.track_id, item.track_generation)):
            identity = TrackIdentity(track.source_id, track.track_id, track.track_generation)
            if identity in self._emitted_bundles:
                continue
            candidates = list(self._candidates_by_identity.get(identity, ()))
            bundle = CompletedTrackCropBundle.from_track(track, candidates)
            bundles.append(bundle)
            self._emitted_bundles.add(identity)
        return bundles

    def flush(self, active_tracks: list[TrackRecord] | tuple[TrackRecord, ...]) -> list[CompletedTrackCropBundle]:
        return self.complete_tracks(active_tracks)

    def candidates_for(self, identity: TrackIdentity) -> tuple[CropCandidate, ...]:
        return tuple(self._candidates_by_identity.get(identity, ()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_count_with_candidates": len(self._candidates_by_identity),
            "retained_candidate_count": sum(len(items) for items in self._candidates_by_identity.values()),
            "total_candidates_created": self.total_candidates_created,
            "rejection_reasons": dict(sorted(self.rejection_reasons.items())),
            "observations": self.observation_collector.to_dict(),
            "retention_policy": self.config.retention_policy,
        }

    def _retain_candidate(self, identity: TrackIdentity, candidate: CropCandidate) -> None:
        candidates = list(self._candidates_by_identity.get(identity, ()))
        candidates.append(candidate)
        self._candidates_by_identity[identity] = self._trim_candidates(candidates)

    def _trim_candidates(self, candidates: list[CropCandidate]) -> list[CropCandidate]:
        limit = self.config.max_candidates_per_track
        if len(candidates) <= limit:
            return sorted(candidates, key=self._stable_order)
        if self.config.retention_policy == "highest_preliminary_score":
            return sorted(sorted(candidates, key=self._stable_order), key=self._quality_order)[:limit]
        if self.config.retention_policy == "uniform_temporal":
            ordered = sorted(candidates, key=self._stable_order)
            if limit == 1:
                return [ordered[len(ordered) // 2]]
            selected_indices = {round(index * (len(ordered) - 1) / (limit - 1)) for index in range(limit)}
            return [ordered[index] for index in sorted(selected_indices)]
        return self._hybrid_quality_temporal(candidates, limit)

    def _hybrid_quality_temporal(self, candidates: list[CropCandidate], limit: int) -> list[CropCandidate]:
        ordered = sorted(candidates, key=self._stable_order)
        buckets: list[list[CropCandidate]] = [[] for _ in range(limit)]
        first = ordered[0].frame_index
        last = ordered[-1].frame_index
        span = max(last - first, 1)
        for candidate in ordered:
            bucket_index = min(limit - 1, int(((candidate.frame_index - first) / span) * limit))
            buckets[bucket_index].append(candidate)
        selected: list[CropCandidate] = []
        for bucket in buckets:
            if bucket:
                selected.append(sorted(bucket, key=self._quality_order)[0])
        remaining = [item for item in ordered if item not in selected]
        selected.extend(sorted(remaining, key=self._quality_order)[: max(0, limit - len(selected))])
        return sorted(selected[:limit], key=self._stable_order)

    def _stable_order(self, candidate: CropCandidate) -> tuple[int, float, int]:
        return (candidate.frame_index, candidate.timestamp_sec, candidate.track_id)

    def _quality_order(self, candidate: CropCandidate) -> tuple[float, float, float, int]:
        score = candidate.preliminary_rank_score if candidate.preliminary_rank_score is not None else 0.0
        confidence = candidate.detection_confidence if candidate.detection_confidence is not None else 0.0
        return (-score, -confidence, -candidate.bbox.area, candidate.frame_index)

    def _count(self, counts: dict[str, int], reason: str) -> None:
        counts[reason] = counts.get(reason, 0) + 1

    def _merge_counts(self, counts: dict[str, int]) -> None:
        for reason, count in counts.items():
            self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + count
