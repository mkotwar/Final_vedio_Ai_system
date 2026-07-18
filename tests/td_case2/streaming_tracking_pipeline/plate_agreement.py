from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher

from .plate_validation_schemas import PlateAgreementResult, PlateTextCandidate


def string_similarity(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def one_character_disagreement(left: str, right: str) -> bool:
    if len(left) != len(right) or left == right:
        return False
    return sum(1 for a, b in zip(left, right) if a != b) == 1


def build_plate_agreement(
    source_id: str,
    track_id: int,
    track_generation: int,
    candidates: list[PlateTextCandidate],
) -> PlateAgreementResult:
    texts = [candidate.text_for_selection() for candidate in candidates if candidate.text_for_selection()]
    groups = Counter(texts)
    best_text = None
    support = 0
    if groups:
        best_text, support = sorted(groups.items(), key=lambda item: (-item[1], item[0]))[0]
    best_similarity = 0.0
    reasons: list[str] = []
    if len(texts) <= 1:
        reasons.append("single_candidate")
        best_similarity = 1.0 if texts else 0.0
    else:
        pair_scores = [string_similarity(a, b) for index, a in enumerate(texts) for b in texts[index + 1 :]]
        best_similarity = max(pair_scores) if pair_scores else 0.0
        if support >= 2:
            reasons.append("exact_agreement")
        elif any(one_character_disagreement(a, b) for index, a in enumerate(texts) for b in texts[index + 1 :]):
            reasons.append("one_character_disagreement")
        elif best_similarity >= 0.80:
            reasons.append("similarity_agreement")
        else:
            reasons.append("candidate_disagreement")
    return PlateAgreementResult(
        source_id=source_id,
        track_id=track_id,
        track_generation=track_generation,
        candidate_count=len(candidates),
        unique_candidate_count=len(groups),
        exact_groups=dict(sorted(groups.items())),
        best_candidate=best_text,
        best_support_count=support,
        best_similarity_score=round(best_similarity, 6),
        disagreement_reasons=reasons,
    )


def agreement_score(candidate: PlateTextCandidate, agreement: PlateAgreementResult | None) -> float:
    if agreement is None or not candidate.text_for_selection():
        return 0.0
    text = candidate.text_for_selection()
    exact_support = agreement.exact_groups.get(text, 0)
    if exact_support >= 2:
        return 1.0
    if agreement.best_candidate and one_character_disagreement(text, agreement.best_candidate):
        return 0.75
    if agreement.best_candidate:
        return min(0.70, string_similarity(text, agreement.best_candidate))
    return 0.0
