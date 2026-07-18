from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .serialization import read_json, read_jsonl


ARTIFACT_PATHS: dict[str, Path] = {
    "source_metadata": Path("01_source") / "source_metadata.json",
    "completed_tracks": Path("04_lifecycle") / "completed_tracks.jsonl",
    "track_observations": Path("05_crops") / "track_observations.jsonl",
    "crop_bundles": Path("05_crops") / "completed_track_crop_bundles.jsonl",
    "selected_crop_sets": Path("06_selected_crops") / "selected_track_crop_sets.jsonl",
    "anpr_colour_results": Path("07_anpr") / "track_anpr_colour_results.jsonl",
    "plate_validation_results": Path("08_plate_validation") / "final_track_anpr_results.jsonl",
    "searchable_records": Path("09_searchable_objects") / "searchable_vehicle_records.jsonl",
    "search_index_summary": Path("10_structured_search") / "search_index_summary.json",
    "step10_summary": Path("10_structured_search") / "reports" / "step10_search_summary.json",
    "step11_summary": Path("11_result_cards") / "reports" / "step11_result_cards_summary.json",
}

UI_STATE_DIR = "ui_state"
MANUAL_PLATE_REVIEWS = "manual_plate_reviews.jsonl"
SAVED_SEARCHES = "saved_searches.json"
UI_PREFERENCES = "ui_preferences.json"


@dataclass(frozen=True)
class LoadedRunArtifacts:
    run_dir: Path
    repo_root: Path
    source_metadata: dict[str, Any] = field(default_factory=dict)
    completed_tracks: list[dict[str, Any]] = field(default_factory=list)
    track_observations: list[dict[str, Any]] = field(default_factory=list)
    crop_bundles: list[dict[str, Any]] = field(default_factory=list)
    selected_crop_sets: list[dict[str, Any]] = field(default_factory=list)
    anpr_colour_results: list[dict[str, Any]] = field(default_factory=list)
    plate_validation_results: list[dict[str, Any]] = field(default_factory=list)
    searchable_records: list[dict[str, Any]] = field(default_factory=list)
    search_index_summary: dict[str, Any] = field(default_factory=dict)
    step10_summary: dict[str, Any] = field(default_factory=dict)
    step11_summary: dict[str, Any] = field(default_factory=dict)
    missing_artifacts: list[str] = field(default_factory=list)
    loaded_artifacts: dict[str, str] = field(default_factory=dict)

    @property
    def records_by_id(self) -> dict[str, dict[str, Any]]:
        return {str(record.get("record_id")): record for record in self.searchable_records}

    @property
    def plate_validation_by_identity(self) -> dict[str, dict[str, Any]]:
        return {_identity_key(record): record for record in self.plate_validation_results}

    @property
    def selected_crops_by_identity(self) -> dict[str, dict[str, Any]]:
        return {_identity_key(record): record for record in self.selected_crop_sets}

    @property
    def completed_tracks_by_identity(self) -> dict[str, dict[str, Any]]:
        return {_identity_key(record): record for record in self.completed_tracks}

    @property
    def crop_bundles_by_identity(self) -> dict[str, dict[str, Any]]:
        return {_identity_key(record): record for record in self.crop_bundles}

    @property
    def track_observations_by_identity(self) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in self.track_observations:
            grouped.setdefault(_identity_key(record), []).append(record)
        return grouped


def load_run_artifacts(run_dir: str | Path, *, repo_root: str | Path | None = None) -> LoadedRunArtifacts:
    run_path = Path(run_dir)
    root_path = Path(repo_root) if repo_root is not None else Path.cwd()
    missing: list[str] = []
    loaded: dict[str, str] = {}

    def load_json_artifact(name: str) -> dict[str, Any]:
        path = run_path / ARTIFACT_PATHS[name]
        if not path.exists():
            missing.append(str(ARTIFACT_PATHS[name]))
            return {}
        loaded[name] = str(path)
        value = read_json(path)
        return value if isinstance(value, dict) else {"value": value}

    def load_jsonl_artifact(name: str) -> list[dict[str, Any]]:
        path = run_path / ARTIFACT_PATHS[name]
        if not path.exists():
            missing.append(str(ARTIFACT_PATHS[name]))
            return []
        loaded[name] = str(path)
        return [dict(item) for item in read_jsonl(path)]

    return LoadedRunArtifacts(
        run_dir=run_path,
        repo_root=root_path,
        source_metadata=load_json_artifact("source_metadata"),
        completed_tracks=load_jsonl_artifact("completed_tracks"),
        track_observations=load_jsonl_artifact("track_observations"),
        crop_bundles=load_jsonl_artifact("crop_bundles"),
        selected_crop_sets=load_jsonl_artifact("selected_crop_sets"),
        anpr_colour_results=load_jsonl_artifact("anpr_colour_results"),
        plate_validation_results=load_jsonl_artifact("plate_validation_results"),
        searchable_records=load_jsonl_artifact("searchable_records"),
        search_index_summary=load_json_artifact("search_index_summary"),
        step10_summary=load_json_artifact("step10_summary"),
        step11_summary=load_json_artifact("step11_summary"),
        missing_artifacts=missing,
        loaded_artifacts=loaded,
    )


def summarize_records(records: list[dict[str, Any]], source_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    source_metadata = source_metadata or {}
    return {
        "total_tracked_objects": len(records),
        "cars": _count_class(records, "car"),
        "motorcycles_two_wheelers": _count_class(records, "motorcycle"),
        "buses": _count_class(records, "bus"),
        "trucks": _count_class(records, "truck"),
        "bicycles": _count_class(records, "bicycle"),
        "persons": _count_class(records, "person"),
        "verified_plates": _count_status(records, "verified"),
        "weak_plates": _count_status(records, "weak"),
        "invalid_ocr": _count_status(records, "invalid"),
        "no_plate_records": _count_status(records, "no_plate_detected"),
        "records_by_dominant_colour": count_by(records, "dominant_colour", fallback_key="normalized_colour", missing="unknown"),
        "source_fps": source_metadata.get("source_fps"),
        "processed_fps": source_metadata.get("target_processing_fps"),
        "duration_sec": source_metadata.get("duration_sec"),
        "source_path": source_metadata.get("source_path"),
    }


def build_object_evidence(record: dict[str, Any], artifacts: LoadedRunArtifacts) -> dict[str, Any]:
    identity = _identity_key(record)
    selected = artifacts.selected_crops_by_identity.get(identity) or {}
    plate = artifacts.plate_validation_by_identity.get(identity) or {}
    crop_bundle = artifacts.crop_bundles_by_identity.get(identity) or {}
    completed_track = artifacts.completed_tracks_by_identity.get(identity) or {}
    observations = artifacts.track_observations_by_identity.get(identity) or []

    object_crop = _first_existing_record(
        [
            _crop_record_from_path(record.get("representative_vehicle_crop_path"), record, "search_record_representative_crop"),
            _crop_record_from_path(plate.get("representative_vehicle_crop_path"), plate, "step8_representative_crop"),
            *_selected_crop_records(selected),
            *_candidate_records(crop_bundle.get("candidates"), "crop_bundle_candidate"),
            *_candidate_records(completed_track.get("crop_candidates"), "lifecycle_crop_candidate"),
            *_crop_records_from_paths(record.get("primary_crop_paths"), record, "search_record_primary_crop"),
            *_crop_records_from_paths(record.get("fallback_crop_paths"), record, "search_record_fallback_crop"),
        ],
        artifacts=artifacts,
        path_keys=("object_crop_path",),
    )
    if object_crop is None:
        object_crop = _crop_record_from_path(None, record, "no_object_crop")

    full_frame = _select_full_frame_record(record, artifacts, object_crop, selected, crop_bundle, completed_track, observations)
    plate_crop = _first_existing_record(
        [
            _plate_record_from_path(record.get("representative_plate_crop_path"), record, "search_record_plate_crop"),
            _plate_record_from_path(plate.get("representative_plate_crop_path"), plate, "step8_plate_crop"),
            _plate_record_from_path(_nested_get(plate, ["selected_candidate", "plate_crop_path"]), plate, "step8_selected_candidate"),
        ],
        artifacts=artifacts,
        path_keys=("plate_crop_path",),
    ) or _plate_record_from_path(None, record, "no_plate_crop")

    warnings = list(record.get("warnings") or [])
    if not full_frame.get("full_frame_path"):
        warnings.append("Full frame unavailable")
    if full_frame.get("frame_index") is not None and object_crop.get("frame_index") is not None:
        if int(full_frame["frame_index"]) != int(object_crop["frame_index"]):
            warnings.append("Full frame/object crop frame mismatch")

    return {
        "record_id": record.get("record_id"),
        "source_id": record.get("source_id"),
        "track_id": record.get("track_id"),
        "track_generation": record.get("track_generation"),
        "object_class": record.get("normalized_class_name") or record.get("object_class"),
        "raw_class": record.get("raw_class_name") or record.get("object_class"),
        "timestamp_sec": object_crop.get("timestamp_sec") if object_crop.get("timestamp_sec") is not None else record.get("representative_timestamp_sec"),
        "frame_index": object_crop.get("frame_index") if object_crop.get("frame_index") is not None else record.get("representative_frame_index"),
        "full_frame_path": full_frame.get("full_frame_path"),
        "object_crop_path": object_crop.get("object_crop_path"),
        "plate_crop_path": plate_crop.get("plate_crop_path"),
        "full_frame_source": full_frame.get("source"),
        "object_crop_source": object_crop.get("source"),
        "plate_crop_source": plate_crop.get("source"),
        "full_frame_frame_index": full_frame.get("frame_index"),
        "object_crop_frame_index": object_crop.get("frame_index"),
        "full_frame_timestamp_sec": full_frame.get("timestamp_sec"),
        "object_crop_timestamp_sec": object_crop.get("timestamp_sec"),
        "plate_text": record.get("plate_text"),
        "plate_status": record.get("plate_status"),
        "warnings": _dedupe_text(warnings),
    }


def summarize_evidence_availability(records: list[dict[str, Any]], artifacts: LoadedRunArtifacts) -> dict[str, Any]:
    evidence_records = [build_object_evidence(record, artifacts) for record in records]
    missing = [item for item in evidence_records if not item.get("full_frame_path")]
    mismatched = [item for item in evidence_records if "Full frame/object crop frame mismatch" in item.get("warnings", [])]
    return {
        "records_checked": len(evidence_records),
        "records_with_full_frames": len(evidence_records) - len(missing),
        "records_missing_full_frames": len(missing),
        "records_with_frame_mismatch": len(mismatched),
        "missing_full_frame_record_ids": [str(item.get("record_id")) for item in missing],
        "path_priority": [
            "selected_crop_matching_object_crop_or_frame",
            "search_record_full_frame_fields",
            "crop_bundle_matching_object_crop_or_frame",
            "lifecycle_crop_candidate_matching_object_crop_or_frame",
            "track_observation_matching_frame",
            "no_full_frame_available",
        ],
    }


def count_by(
    records: list[dict[str, Any]],
    key: str,
    *,
    fallback_key: str | None = None,
    missing: str | None = None,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = record.get(key)
        if value is None and fallback_key:
            value = record.get(fallback_key)
        if value is None:
            value = missing
        if value is None:
            continue
        normalized = str(value).lower()
        counts[normalized] = counts.get(normalized, 0) + 1
    return dict(sorted(counts.items()))


def resolve_artifact_path(
    artifact_path: str | Path | None,
    *,
    run_dir: str | Path,
    repo_root: str | Path | None = None,
) -> Path | None:
    if artifact_path is None or str(artifact_path).strip() == "":
        return None
    candidate = Path(artifact_path)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None

    run_path = Path(run_dir)
    root_path = Path(repo_root) if repo_root is not None else Path.cwd()
    for base in (run_path, root_path):
        resolved = base / candidate
        if resolved.exists():
            return resolved
    return None


def get_record_detail(artifacts: LoadedRunArtifacts, record_id: str) -> dict[str, Any]:
    record = artifacts.records_by_id.get(record_id)
    if not record:
        return {}
    identity = _identity_key(record)
    return {
        "record": record,
        "selected_crops": artifacts.selected_crops_by_identity.get(identity),
        "plate_validation": artifacts.plate_validation_by_identity.get(identity),
        "completed_track": artifacts.completed_tracks_by_identity.get(identity),
        "object_evidence": build_object_evidence(record, artifacts),
        "manual_reviews": [review for review in load_manual_plate_reviews(artifacts.run_dir) if review.get("record_id") == record_id],
    }


def ensure_ui_state_files(run_dir: str | Path) -> dict[str, Path]:
    state_dir = Path(run_dir) / UI_STATE_DIR
    state_dir.mkdir(parents=True, exist_ok=True)
    saved_searches = state_dir / SAVED_SEARCHES
    preferences = state_dir / UI_PREFERENCES
    manual_reviews = state_dir / MANUAL_PLATE_REVIEWS
    if not saved_searches.exists():
        saved_searches.write_text("[]\n", encoding="utf-8")
    if not preferences.exists():
        preferences.write_text("{}\n", encoding="utf-8")
    if not manual_reviews.exists():
        manual_reviews.touch()
    return {
        "state_dir": state_dir,
        "manual_plate_reviews": manual_reviews,
        "saved_searches": saved_searches,
        "ui_preferences": preferences,
    }


def append_manual_plate_review(
    run_dir: str | Path,
    *,
    record_id: str,
    decision: str,
    reviewer: str = "local_ui",
    notes: str = "",
) -> Path:
    if decision not in {"looks_correct", "looks_incorrect", "needs_review"}:
        raise ValueError(f"Unsupported manual plate decision: {decision}")
    paths = ensure_ui_state_files(run_dir)
    payload = {
        "record_id": record_id,
        "decision": decision,
        "reviewer": reviewer,
        "notes": notes,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with paths["manual_plate_reviews"].open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
    return paths["manual_plate_reviews"]


def load_manual_plate_reviews(run_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / UI_STATE_DIR / MANUAL_PLATE_REVIEWS
    if not path.exists():
        return []
    reviews: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                reviews.append(json.loads(line))
    return reviews


def _identity_key(record: dict[str, Any]) -> str:
    return f"{record.get('source_id')}:{int(record.get('track_id') or 0)}:{int(record.get('track_generation') or 0)}"


def _select_full_frame_record(
    record: dict[str, Any],
    artifacts: LoadedRunArtifacts,
    object_crop: dict[str, Any],
    selected: dict[str, Any],
    crop_bundle: dict[str, Any],
    completed_track: dict[str, Any],
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    object_path = object_crop.get("object_crop_path")
    frame_index = object_crop.get("frame_index") if object_crop.get("frame_index") is not None else record.get("representative_frame_index")
    candidates = [
        *_full_frame_records_from_crop_records(_selected_crop_records(selected), "selected_crop_evidence"),
        _full_frame_from_record_fields(record, "search_record_full_frame"),
        *_full_frame_records_from_crop_records(_candidate_records(crop_bundle.get("candidates"), "crop_bundle_candidate"), "crop_bundle_source_frame"),
        *_full_frame_records_from_crop_records(_candidate_records(completed_track.get("crop_candidates"), "lifecycle_crop_candidate"), "lifecycle_source_frame"),
        *_full_frame_records_from_observations(observations),
    ]
    matched = _first_matching_full_frame(candidates, artifacts, object_path=object_path, frame_index=frame_index)
    if matched is not None:
        return matched
    existing = _first_existing_record(candidates, artifacts=artifacts, path_keys=("full_frame_path",))
    if existing is not None:
        return existing
    return {"full_frame_path": None, "source": "no_full_frame_available", "frame_index": frame_index, "timestamp_sec": object_crop.get("timestamp_sec")}


def _first_matching_full_frame(
    candidates: list[dict[str, Any]],
    artifacts: LoadedRunArtifacts,
    *,
    object_path: Any,
    frame_index: Any,
) -> dict[str, Any] | None:
    for candidate in candidates:
        if not _record_path_exists(candidate, artifacts, ("full_frame_path",)):
            continue
        if object_path and candidate.get("object_crop_path") and str(candidate.get("object_crop_path")) == str(object_path):
            return candidate
        if frame_index is not None and candidate.get("frame_index") is not None and int(candidate["frame_index"]) == int(frame_index):
            return candidate
    return None


def _first_existing_record(
    records: list[dict[str, Any]],
    *,
    artifacts: LoadedRunArtifacts,
    path_keys: tuple[str, ...],
) -> dict[str, Any] | None:
    for record in records:
        if _record_path_exists(record, artifacts, path_keys):
            return record
    return None


def _record_path_exists(record: dict[str, Any], artifacts: LoadedRunArtifacts, path_keys: tuple[str, ...]) -> bool:
    for key in path_keys:
        if resolve_artifact_path(record.get(key), run_dir=artifacts.run_dir, repo_root=artifacts.repo_root):
            return True
    return False


def _selected_crop_records(selected: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        *_candidate_records(selected.get("primary_crops"), "selected_primary_crop"),
        *_candidate_records([selected.get("fallback_crop")], "selected_fallback_crop"),
    ]


def _candidate_records(candidates: Any, source: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not candidates:
        return records
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        records.append(
            {
                "source": source,
                "object_crop_path": candidate.get("vehicle_crop_path") or candidate.get("object_crop_path"),
                "full_frame_path": _first_value(candidate, ("full_frame_path", "source_frame_path", "evidence_frame_path")),
                "frame_index": candidate.get("frame_index"),
                "timestamp_sec": candidate.get("timestamp_sec"),
            }
        )
    return records


def _crop_record_from_path(path: Any, record: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "source": source,
        "object_crop_path": path,
        "frame_index": record.get("representative_frame_index") or record.get("frame_index"),
        "timestamp_sec": record.get("representative_timestamp_sec") or record.get("timestamp_sec"),
    }


def _crop_records_from_paths(paths: Any, record: dict[str, Any], source: str) -> list[dict[str, Any]]:
    return [_crop_record_from_path(path, record, source) for path in paths or []]


def _plate_record_from_path(path: Any, record: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "source": source,
        "plate_crop_path": path,
        "frame_index": record.get("representative_frame_index") or record.get("frame_index"),
        "timestamp_sec": record.get("representative_timestamp_sec") or record.get("timestamp_sec"),
    }


def _full_frame_records_from_crop_records(records: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    return [
        {
            "source": source,
            "full_frame_path": record.get("full_frame_path"),
            "object_crop_path": record.get("object_crop_path"),
            "frame_index": record.get("frame_index"),
            "timestamp_sec": record.get("timestamp_sec"),
        }
        for record in records
    ]


def _full_frame_records_from_observations(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source": "track_observation_full_frame",
            "full_frame_path": _first_value(observation, ("full_frame_path", "source_frame_path", "evidence_frame_path")),
            "frame_index": observation.get("frame_index"),
            "timestamp_sec": observation.get("timestamp_sec"),
        }
        for observation in observations
    ]


def _full_frame_from_record_fields(record: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "source": source,
        "full_frame_path": _first_value(record, ("full_frame_path", "source_frame_path", "evidence_frame_path")),
        "frame_index": record.get("representative_frame_index") or record.get("frame_index"),
        "timestamp_sec": record.get("representative_timestamp_sec") or record.get("timestamp_sec"),
    }


def _first_value(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return None


def _nested_get(value: dict[str, Any], keys: list[str]) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _dedupe_text(values: list[str]) -> list[str]:
    seen: set[str] = set()
    retained: list[str] = []
    for value in values:
        text = str(value)
        if text not in seen:
            seen.add(text)
            retained.append(text)
    return retained


def _record_class(record: dict[str, Any]) -> str:
    return str(record.get("normalized_class_name") or record.get("object_class") or "").lower()


def _count_class(records: list[dict[str, Any]], class_name: str) -> int:
    return sum(1 for record in records if _record_class(record) == class_name)


def _count_status(records: list[dict[str, Any]], status: str) -> int:
    return sum(1 for record in records if str(record.get("plate_status") or "") == status)
