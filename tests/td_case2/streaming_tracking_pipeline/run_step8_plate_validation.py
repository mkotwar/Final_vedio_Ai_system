from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from .plate_agreement import build_plate_agreement
from .plate_validation import candidates_from_ocr_record
from .plate_validation_artifacts import PlateValidationArtifactSink, load_step8_inputs
from .plate_validation_metrics import build_plate_validation_metrics, score_candidate
from .plate_validation_schemas import FinalTrackAnprResult, PlateTextCandidate, PlateValidationConfig
from .serialization import to_json_safe


DEFAULT_RUN_DIR = "debug_runs/streaming_tracking_anpr_10fps_anpr_test_5min_20260718_160910"


def identity(row: dict[str, Any]) -> tuple[str, int, int]:
    return (str(row.get("source_id", "")), int(row.get("track_id", 0) or 0), int(row.get("track_generation", 0) or 0))


def run(
    *,
    run_dir: str | Path = DEFAULT_RUN_DIR,
    minimum_verified_score: float = 0.72,
    minimum_weak_score: float = 0.35,
    maximum_substitutions: int = 2,
    maximum_variants: int = 20,
    minimum_agreement_support: int = 2,
    allow_single_strong_candidate: bool = True,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    config = PlateValidationConfig(
        minimum_verified_score=minimum_verified_score,
        minimum_weak_score=minimum_weak_score,
        maximum_substitutions_per_candidate=maximum_substitutions,
        maximum_generated_variants=maximum_variants,
        minimum_agreement_support=minimum_agreement_support,
        allow_single_strong_candidate=allow_single_strong_candidate,
    )
    inputs = load_step8_inputs(run_dir)
    crop_sets = inputs["selected_track_crop_sets"]
    tracks = inputs["track_plate_diagnostic_results"]
    ocr_rows = inputs["ocr_results"]
    raw_boxes = inputs["raw_plate_box_diagnostics"]
    track_rows = inputs["track_anpr_colour_results"]
    colour_rows = inputs.get("florence_colour_results", [])

    box_by_crop = {str(row.get("plate_crop_path")): row for row in raw_boxes if row.get("plate_crop_path")}
    track_diag_by_id = {identity(row): row for row in tracks}
    crop_set_by_id = {identity(row): row for row in crop_sets}
    colour_by_id = {identity(row): row for row in track_rows}
    colour_by_id.update({identity(row): row for row in colour_rows})

    candidates_by_id: dict[tuple[str, int, int], list[PlateTextCandidate]] = defaultdict(list)
    for row in ocr_rows:
        box = box_by_crop.get(str(row.get("plate_crop_path")))
        vehicle_crop_path = box.get("vehicle_crop_path") if box else None
        if vehicle_crop_path is None:
            vehicle_crop_path = _vehicle_crop_for_ocr(row, track_diag_by_id)
        evidence = {
            "plate_detection_confidence": box.get("raw_confidence") if box else None,
            "timestamp_sec": box.get("timestamp_sec") if box else None,
            "vehicle_crop_path": vehicle_crop_path,
            "raw_plate_box": box,
        }
        for candidate in candidates_from_ocr_record(row, config, evidence):
            candidates_by_id[identity(row)].append(candidate)

    all_candidates = [candidate for values in candidates_by_id.values() for candidate in values]
    agreements = []
    finals = []
    for key in sorted(crop_set_by_id):
        source_id, track_id, track_generation = key
        track_candidates = candidates_by_id.get(key, [])
        agreement = build_plate_agreement(source_id, track_id, track_generation, track_candidates) if track_candidates else None
        if agreement is not None:
            agreements.append(agreement)
        colour_row = colour_by_id.get(key, {})
        diag = track_diag_by_id.get(key, {})
        crop_set = crop_set_by_id.get(key, {})
        final = _finalize_track(
            key,
            track_candidates,
            agreement,
            colour_row,
            diag,
            crop_set,
            config,
        )
        finals.append(final)

    metrics = build_plate_validation_metrics(
        all_candidates,
        agreements,
        finals,
        track_generations_processed=len(crop_set_by_id),
        tracks_with_plate_detection=sum(1 for row in tracks if row.get("selected_plate_candidate")),
        tracks_without_plate_detection=len(crop_set_by_id) - sum(1 for row in tracks if row.get("selected_plate_candidate")),
        raw_ocr_candidate_count=len(ocr_rows),
    )
    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir or Path(run_dir) / "08_plate_validation"),
        "config": to_json_safe(config),
        **metrics,
    }
    report = {
        "summary": summary,
        "final_results": [item.to_dict() for item in finals],
        "input_artifact_counts": {
            "ocr_results": len(ocr_rows),
            "track_anpr_colour_results": len(track_rows),
            "florence_colour_results": len(colour_rows),
            "plate_diagnostic_attempts": len(inputs["plate_diagnostic_attempts"]),
            "raw_plate_box_diagnostics": len(raw_boxes),
            "track_plate_diagnostic_results": len(tracks),
            "selected_track_crop_sets": len(crop_sets),
        },
    }
    paths = PlateValidationArtifactSink(run_dir, output_dir).write(all_candidates, agreements, finals, summary, report)
    return {"summary": summary, "report": report, "artifact_paths": paths, "final_results": finals}


def _finalize_track(
    key: tuple[str, int, int],
    candidates: list[PlateTextCandidate],
    agreement: Any,
    colour_row: dict[str, Any],
    diag: dict[str, Any],
    crop_set: dict[str, Any],
    config: PlateValidationConfig,
) -> FinalTrackAnprResult:
    source_id, track_id, track_generation = key
    warnings: list[str] = []
    scored = [(candidate, score_candidate(candidate, agreement, config)) for candidate in candidates]
    scored = sorted(
        scored,
        key=lambda item: (
            -item[1]["final_candidate_score"],
            -item[0].format_score,
            item[0].crop_rank if item[0].crop_rank is not None else 99,
            item[0].text_for_selection(),
        ),
    )
    selected = scored[0][0] if scored else None
    score = scored[0][1]["final_candidate_score"] if scored else 0.0
    support = agreement.exact_groups.get(selected.text_for_selection(), 0) if selected and agreement else 0
    status = _status_for_track(selected, score, support, agreement, diag, config)
    if selected and selected.corrected_text:
        warnings.append("selected_text_contains_controlled_ocr_correction")
    if agreement and "single_candidate" in agreement.disagreement_reasons:
        warnings.append("single_candidate_evidence")
    return FinalTrackAnprResult(
        source_id=source_id,
        track_id=track_id,
        track_generation=track_generation,
        object_class=diag.get("object_class") or colour_row.get("object_class") or _object_class_from_crop_set(crop_set),
        final_plate_text=selected.text_for_selection() if selected and status in {"verified", "weak"} else None,
        plate_status=status,
        confidence=round(score, 6),
        support_count=support,
        selected_candidate=selected,
        all_candidates=candidates,
        agreement=agreement,
        normalized_colour=colour_row.get("normalized_colour") or _colour_from_track_result(colour_row, "normalized_colour"),
        raw_colour=_colour_from_track_result(colour_row, "raw_text"),
        representative_frame_index=selected.frame_index if selected else _frame_from_diag(diag),
        representative_timestamp_sec=selected.timestamp_sec if selected else _timestamp_from_diag(diag),
        representative_vehicle_crop_path=selected.source_vehicle_crop_path if selected else _vehicle_crop_from_diag(diag),
        representative_plate_crop_path=selected.plate_crop_path if selected else None,
        warnings=warnings,
        metadata={
            "candidate_scores": [
                {"text": candidate.text_for_selection(), "components": components}
                for candidate, components in scored
            ],
            "colour": {
                "raw_colour": colour_row.get("raw_text") or _colour_from_track_result(colour_row, "raw_text"),
                "normalized_colour": colour_row.get("normalized_colour") or _colour_from_track_result(colour_row, "normalized_colour"),
                "colour_confidence": colour_row.get("confidence") or _colour_from_track_result(colour_row, "confidence"),
                "colour_crop_path": colour_row.get("vehicle_crop_path") or _colour_from_track_result(colour_row, "vehicle_crop_path"),
                "colour_status": colour_row.get("status") or _colour_from_track_result(colour_row, "status"),
            },
            "diagnostic_status": diag.get("final_status"),
            "diagnostic_failure_reasons": diag.get("final_failure_reasons", []),
        },
    )


def _status_for_track(
    selected: PlateTextCandidate | None,
    score: float,
    support: int,
    agreement: Any,
    diag: dict[str, Any],
    config: PlateValidationConfig,
) -> str:
    if selected is None:
        if diag.get("selected_plate_candidate") is None:
            return "no_plate_detected"
        if diag.get("selected_ocr_result", {}).get("status") == "empty_output":
            return "ocr_empty"
        return "insufficient_evidence"
    if selected.format_status == "not_plate_like":
        return "invalid"
    has_agreement = support >= config.minimum_agreement_support
    strong_single = (
        config.allow_single_strong_candidate
        and (selected.plate_detection_confidence or 0.0) >= config.strong_detector_confidence
        and selected.format_score >= config.strong_format_score
    )
    if selected.format_status in {"strict_format_match", "relaxed_format_match"} and score >= config.minimum_verified_score and (has_agreement or strong_single):
        return "verified"
    if score >= config.minimum_weak_score or selected.format_status in {"strict_format_match", "relaxed_format_match", "partial_plate"}:
        return "weak"
    if agreement and agreement.candidate_count == 0:
        return "insufficient_evidence"
    return "invalid"


def _vehicle_crop_for_ocr(row: dict[str, Any], diagnostics: dict[tuple[str, int, int], dict[str, Any]]) -> str | None:
    diag = diagnostics.get(identity(row), {})
    selected = diag.get("selected_plate_candidate") or {}
    return selected.get("vehicle_crop_path") or _vehicle_crop_from_diag(diag)


def _vehicle_crop_from_diag(diag: dict[str, Any]) -> str | None:
    attempts = diag.get("attempts") or []
    return attempts[0].get("vehicle_crop_path") if attempts else None


def _frame_from_diag(diag: dict[str, Any]) -> int | None:
    attempts = diag.get("attempts") or []
    return attempts[0].get("source_frame_index") if attempts else None


def _timestamp_from_diag(diag: dict[str, Any]) -> float | None:
    attempts = diag.get("attempts") or []
    return attempts[0].get("timestamp_sec") if attempts else None


def _colour_from_track_result(row: dict[str, Any], field_name: str) -> str | None:
    if row.get(field_name) is not None:
        return row.get(field_name)
    colour = row.get("colour_result") or {}
    return colour.get(field_name)


def _object_class_from_crop_set(row: dict[str, Any]) -> str | None:
    lifecycle = row.get("lifecycle_record") or {}
    return lifecycle.get("last_class_name") or (row.get("metadata") or {}).get("dominant_class")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 8: validate saved raw ANPR OCR artifacts into final plate results.")
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--minimum-verified-score", type=float, default=0.72)
    parser.add_argument("--minimum-weak-score", type=float, default=0.35)
    parser.add_argument("--maximum-substitutions", type=int, default=2)
    parser.add_argument("--maximum-variants", type=int, default=20)
    parser.add_argument("--minimum-agreement-support", type=int, default=2)
    parser.add_argument("--allow-single-strong-candidate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run(
        run_dir=args.run_dir,
        minimum_verified_score=args.minimum_verified_score,
        minimum_weak_score=args.minimum_weak_score,
        maximum_substitutions=args.maximum_substitutions,
        maximum_variants=args.maximum_variants,
        minimum_agreement_support=args.minimum_agreement_support,
        allow_single_strong_candidate=args.allow_single_strong_candidate,
        output_dir=args.output_dir,
    )
    summary = result["summary"]
    print("Step 8 plate validation complete")
    print(f"run_dir={summary['run_dir']}")
    print(f"raw_ocr_candidates={summary['raw_ocr_candidates']}")
    print(f"verified_results={summary['verified_results']}")
    print(f"weak_results={summary['weak_results']}")
    print(f"invalid_results={summary['invalid_results']}")
    print(f"no_plate_results={summary['no_plate_results']}")
    print(f"report={result['artifact_paths']['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
