from __future__ import annotations

from typing import Sequence

from .anpr_config import PlateSelectionConfig
from .plate_models import PlateCandidate


def select_best_plate_candidates(
    candidates: Sequence[PlateCandidate],
    *,
    maximum_for_ocr: int,
    config: PlateSelectionConfig,
) -> list[PlateCandidate]:
    scored = sorted(
        candidates,
        key=lambda item: (
            _score_candidate(item, config=config),
            item.detector_confidence,
            item.sharpness_score,
            item.area,
            -item.frame_number,
            item.relative_storage_uri,
        ),
        reverse=True,
    )
    return scored[: max(0, int(maximum_for_ocr))]


def _score_candidate(candidate: PlateCandidate, *, config: PlateSelectionConfig) -> float:
    area_score = min(1.0, float(candidate.area) / max(float(config.minimum_area), 1.0) / 10.0)
    aspect_mid = (config.minimum_aspect_ratio + config.maximum_aspect_ratio) / 2.0
    aspect_delta = abs(candidate.aspect_ratio - aspect_mid)
    aspect_window = max(config.maximum_aspect_ratio - config.minimum_aspect_ratio, 1e-6)
    aspect_score = max(0.0, 1.0 - (aspect_delta / aspect_window))
    edge_component = max(0.0, 1.0 - candidate.edge_penalty)
    return (
        candidate.detector_confidence * config.confidence_weight
        + candidate.sharpness_score * config.sharpness_weight
        + area_score * config.size_weight
        + aspect_score * config.aspect_ratio_weight
        + edge_component * config.edge_penalty_weight
    )
