from __future__ import annotations

from dataclasses import dataclass

from .anpr_config import AnprConfig
from .plate_models import PlateCandidate, PlateOcrAttempt


@dataclass(frozen=True, slots=True)
class AggregatedPlateResult:
    status: str
    verification_status: str
    normalized_text: str | None
    raw_text: str | None
    confidence: float
    support_frame_count: int
    support_candidate_count: int
    selected_candidate: PlateCandidate | None
    selected_attempt: PlateOcrAttempt | None
    attempts: tuple[PlateOcrAttempt, ...]


def aggregate_ocr_attempts(
    *,
    candidates: list[PlateCandidate],
    attempts_by_candidate_uri: dict[str, list[PlateOcrAttempt]],
    config: AnprConfig,
) -> AggregatedPlateResult:
    attempts = tuple(attempt for candidate in candidates for attempt in attempts_by_candidate_uri.get(candidate.relative_storage_uri, []))
    if not attempts:
        return AggregatedPlateResult(
            status="NO_PLATE_DETECTED",
            verification_status="UNKNOWN",
            normalized_text=None,
            raw_text=None,
            confidence=0.0,
            support_frame_count=0,
            support_candidate_count=0,
            selected_candidate=None,
            selected_attempt=None,
            attempts=(),
        )
    groups: dict[str, list[PlateOcrAttempt]] = {}
    for attempt in attempts:
        if attempt.normalized_text:
            groups.setdefault(attempt.normalized_text, []).append(attempt)
    if groups:
        ranked = sorted(
            groups.items(),
            key=lambda item: (
                len({attempt.frame_number for attempt in item[1]}),
                len(item[1]),
                max(attempt.confidence for attempt in item[1]),
            ),
            reverse=True,
        )
        best_text, best_attempts = ranked[0]
        competing = ranked[1] if len(ranked) > 1 else None
        best_support_frames = len({attempt.frame_number for attempt in best_attempts})
        best_attempt = max(best_attempts, key=lambda attempt: attempt.confidence)
        selected_candidate = _candidate_for_attempt(candidates, best_attempt.candidate_storage_uri)
        if competing is not None and best_attempt.verification_status == "VERIFIED":
            competing_best = max(competing[1], key=lambda attempt: attempt.confidence)
            if competing_best.verification_status == "VERIFIED" and competing[0] != best_text:
                return AggregatedPlateResult(
                    status="CONFLICTING_CANDIDATES",
                    verification_status="UNVERIFIED",
                    normalized_text=None,
                    raw_text=best_attempt.raw_text,
                    confidence=best_attempt.confidence,
                    support_frame_count=best_support_frames,
                    support_candidate_count=len(best_attempts),
                    selected_candidate=selected_candidate,
                    selected_attempt=best_attempt,
                    attempts=attempts,
                )
        if (
            best_attempt.verification_status == "VERIFIED"
            and (
                best_support_frames >= config.ocr.minimum_support_frames_for_verified
                or (competing is None and best_attempt.confidence >= 0.9 and best_attempt.candidate_source == "DETECTOR")
            )
        ):
            return AggregatedPlateResult(
                status="VERIFIED",
                verification_status="VERIFIED",
                normalized_text=best_text,
                raw_text=best_attempt.raw_text,
                confidence=best_attempt.confidence,
                support_frame_count=best_support_frames,
                support_candidate_count=len(best_attempts),
                selected_candidate=selected_candidate,
                selected_attempt=best_attempt,
                attempts=attempts,
            )
        if competing is not None and len(competing[1]) >= len(best_attempts) and competing[0] != best_text:
            return AggregatedPlateResult(
                status="CONFLICTING_CANDIDATES",
                verification_status="UNVERIFIED",
                normalized_text=None,
                raw_text=best_attempt.raw_text,
                confidence=best_attempt.confidence,
                support_frame_count=best_support_frames,
                support_candidate_count=len(best_attempts),
                selected_candidate=selected_candidate,
                selected_attempt=best_attempt,
                attempts=attempts,
            )
        if best_text:
            status = "PARTIAL" if len(best_text) < config.validation.minimum_normalized_length else best_attempt.status
            return AggregatedPlateResult(
                status=status,
                verification_status="UNVERIFIED",
                normalized_text=best_text,
                raw_text=best_attempt.raw_text,
                confidence=best_attempt.confidence,
                support_frame_count=best_support_frames,
                support_candidate_count=len(best_attempts),
                selected_candidate=selected_candidate,
                selected_attempt=best_attempt,
                attempts=attempts,
            )
    unreadable = max(attempts, key=lambda attempt: attempt.confidence)
    return AggregatedPlateResult(
        status="UNREADABLE",
        verification_status="UNKNOWN",
        normalized_text=None,
        raw_text=unreadable.raw_text,
        confidence=unreadable.confidence,
        support_frame_count=len({attempt.frame_number for attempt in attempts}),
        support_candidate_count=len(attempts),
        selected_candidate=_candidate_for_attempt(candidates, unreadable.candidate_storage_uri),
        selected_attempt=unreadable,
        attempts=attempts,
    )


def _candidate_for_attempt(candidates: list[PlateCandidate], candidate_storage_uri: str) -> PlateCandidate | None:
    for candidate in candidates:
        if candidate.relative_storage_uri == candidate_storage_uri:
            return candidate
    return None
