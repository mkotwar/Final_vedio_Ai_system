from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..persistence.analytics_database_client import AnalyticsDatabaseClient, AnalyticsDatabaseClientError
from ..tracking.class_recalculation import FragmentLinkEvaluation, evaluate_fragment_link, recalculate_track_class
from ..tracking.class_stabilization import class_diagnostics_to_metadata
from ..tracking.tracking_config import TrackingConfig, load_tracking_config
from ..tracking.tracking_models import LocalVehicleTrack, TrackObservation


LOGGER = logging.getLogger("recalculate_track_classes")
EXIT_SUCCESS = 0
EXIT_CONFIGURATION_MISSING = 2
EXIT_QUERY_FAILED = 3


@dataclass(frozen=True, slots=True)
class RecalculatedTrack:
    track_uuid: str
    camera_code: str
    local_track_id: int
    observation_count: int
    old_final_class: str | None
    new_final_class: str | None
    class_confidence: float | None
    class_is_stable: bool
    class_counts: dict[str, int]
    class_confidence_sums: dict[str, float]
    class_max_confidences: dict[str, float]
    class_winner_margin: float | None
    class_status: str | None
    winning_class_ratio: float | None
    runner_up_class: str | None
    runner_up_ratio: float | None
    maximum_consecutive_winner_count: int
    recent_winning_class: str | None
    recent_winning_ratio: float | None
    class_transition_count: int
    incompatible_transition_count: int
    count_winner: str | None
    score_winner: str | None
    winner_agreement: bool
    strong_conflict_detected: bool
    split_recommended: bool
    latest_observation_class: str | None
    persisted: bool
    insufficient_history: bool
    note: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recalculate stabilized track classes from persisted observation history when available.",
    )
    parser.add_argument("--run-code", required=True, help="Run code to inspect, for example RUN_20260727_131724.")
    parser.add_argument("--camera-code", help="Optional single camera filter.")
    parser.add_argument("--track-uuid", help="Optional single track UUID filter.")
    parser.add_argument(
        "--tracking-config",
        default=str(Path("tests/td_case2/multicamera_vehicle_tracking_pipeline/config/tracking.yaml")),
        help="Tracking config path. Defaults to the multicamera tracking config.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Explicitly request dry-run mode. Default behavior is dry-run.")
    parser.add_argument("--persist", action="store_true", help="Persist recalculated final class and diagnostics back to analytics.vehicle_track.")
    parser.add_argument("--output-report", help="Optional JSON report output path.")
    return parser


def _response_rows(response: Any) -> list[dict[str, Any]]:
    rows = getattr(response, "data", None)
    if rows is None:
        return []
    if not isinstance(rows, list):
        raise RuntimeError("Expected list response rows.")
    return [dict(item) for item in rows]


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _load_db_run_rows(
    client: AnalyticsDatabaseClient,
    *,
    run_code: str,
    camera_code: str | None,
    track_uuid: str | None,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    run_rows = _response_rows(client.table("processing_run").select("id,run_code").eq("run_code", run_code).limit(1).execute())
    if not run_rows:
        raise RuntimeError(f"Run {run_code} was not found in analytics.processing_run.")
    run_id = str(run_rows[0]["id"])
    query = client.table("vehicle_track").select("*").eq("processing_run_id", run_id)
    if track_uuid:
        query = query.eq("track_uuid", track_uuid)
    track_rows = _response_rows(query.execute())
    if camera_code:
        camera_rows = _response_rows(client.table("camera").select("id,camera_code").eq("camera_code", camera_code).limit(1).execute())
        if not camera_rows:
            raise RuntimeError(f"Camera {camera_code} was not found.")
        camera_id = str(camera_rows[0]["id"])
        track_rows = [row for row in track_rows if str(row.get("camera_id")) == camera_id]
    if not track_rows:
        raise RuntimeError("No matching tracks were found for the provided filters.")
    observations_by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for track_row in track_rows:
        track_id = str(track_row["id"])
        observation_rows = _response_rows(
            client.table("track_observation")
            .select("frame_number,observed_at,video_time_seconds,detection_confidence,is_key_observation,metadata,bbox_x1,bbox_y1,bbox_x2,bbox_y2")
            .eq("vehicle_track_id", track_id)
            .order("frame_number")
            .execute()
        )
        observations_by_track[track_id] = observation_rows
    return track_rows, observations_by_track


def _find_report_candidates(run_code: str) -> list[Path]:
    root = Path("debug_runs")
    if not root.exists():
        return []
    matches: list[Path] = []
    for report_path in root.rglob("report.json"):
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(payload.get("run_id")) == run_code:
            matches.append(report_path)
    return matches


def _rows_to_observations(track_row: dict[str, Any], observation_rows: list[dict[str, Any]]) -> list[TrackObservation]:
    observations: list[TrackObservation] = []
    for row in observation_rows:
        metadata = _metadata(row)
        class_name = metadata.get("class_name") or metadata.get("raw_class_name")
        if class_name in (None, ""):
            continue
        observations.append(
            TrackObservation(
                camera_code=str(track_row.get("camera_code") or metadata.get("camera_code") or ""),
                local_track_id=int(track_row.get("local_track_id") or 0),
                frame_number=int(row.get("frame_number") or 0),
                video_time_seconds=float(row.get("video_time_seconds") or 0.0),
                camera_timestamp=None,
                class_name=str(class_name),
                confidence=float(row.get("detection_confidence") or 0.0),
                bbox_xyxy=(
                    float(row.get("bbox_x1") or 0.0),
                    float(row.get("bbox_y1") or 0.0),
                    float(row.get("bbox_x2") or 0.0),
                    float(row.get("bbox_y2") or 0.0),
                ),
                track_uuid=str(track_row.get("track_uuid") or ""),
                state="completed",
                raw_class_name=str(metadata.get("raw_class_name")) if metadata.get("raw_class_name") not in (None, "") else None,
            )
        )
    return observations


def _to_local_track(track_row: dict[str, Any], observations: list[TrackObservation], final_class_name: str | None) -> LocalVehicleTrack:
    first = observations[0]
    last = observations[-1]
    return LocalVehicleTrack(
        track_uuid=str(track_row.get("track_uuid") or ""),
        camera_code=str(track_row.get("camera_code") or ""),
        local_track_id=int(track_row.get("local_track_id") or 0),
        class_name=str(final_class_name or track_row.get("vehicle_class") or "unknown"),
        first_frame_number=first.frame_number,
        last_frame_number=last.frame_number,
        first_seen_at=first.camera_timestamp,
        last_seen_at=last.camera_timestamp,
        first_video_time_seconds=first.video_time_seconds,
        last_video_time_seconds=last.video_time_seconds,
        observation_count=len(observations),
        best_confidence=max(float(item.confidence) for item in observations),
        state="completed",
        observations=observations,
    )


def _recalculate_track_rows(
    track_rows: list[dict[str, Any]],
    observations_by_track: dict[str, list[dict[str, Any]]],
    config: TrackingConfig,
) -> tuple[list[RecalculatedTrack], list[FragmentLinkEvaluation]]:
    recalculated: list[RecalculatedTrack] = []
    by_camera: dict[str, list[LocalVehicleTrack]] = defaultdict(list)
    for track_row in sorted(track_rows, key=lambda row: (str(row.get("track_uuid")), int(row.get("first_frame_number") or 0))):
        track_id = str(track_row.get("id"))
        observation_rows = observations_by_track.get(track_id, [])
        observations = _rows_to_observations(track_row, observation_rows)
        metadata = _metadata(track_row)
        if not observations:
            recalculated.append(
                RecalculatedTrack(
                    track_uuid=str(track_row.get("track_uuid") or ""),
                    camera_code=str(track_row.get("camera_code") or ""),
                    local_track_id=int(track_row.get("local_track_id") or 0),
                    observation_count=int(track_row.get("observation_count") or 0),
                    old_final_class=str(track_row.get("vehicle_class")) if track_row.get("vehicle_class") is not None else None,
                    new_final_class=None,
                    class_confidence=None,
                    class_is_stable=False,
                    class_counts={},
                    class_confidence_sums={},
                    class_max_confidences={},
                    class_winner_margin=None,
                    class_status="INSUFFICIENT_OBSERVATIONS",
                    winning_class_ratio=None,
                    runner_up_class=None,
                    runner_up_ratio=None,
                    maximum_consecutive_winner_count=0,
                    recent_winning_class=None,
                    recent_winning_ratio=None,
                    class_transition_count=0,
                    incompatible_transition_count=0,
                    count_winner=None,
                    score_winner=None,
                    winner_agreement=True,
                    strong_conflict_detected=False,
                    split_recommended=False,
                    latest_observation_class=None,
                    persisted=False,
                    insufficient_history=True,
                    note="No persisted observation class metadata was available for recalculation.",
                )
            )
            continue
        diagnostics = recalculate_track_class(observations, config)
        new_final_class = diagnostics.stable_class_name or diagnostics.provisional_class_name
        local_track = _to_local_track(track_row, observations, new_final_class)
        by_camera[local_track.camera_code].append(local_track)
        recalculated.append(
            RecalculatedTrack(
                track_uuid=local_track.track_uuid,
                camera_code=local_track.camera_code,
                local_track_id=local_track.local_track_id,
                observation_count=len(observations),
                old_final_class=str(track_row.get("vehicle_class")) if track_row.get("vehicle_class") is not None else None,
                new_final_class=new_final_class,
                class_confidence=diagnostics.class_confidence,
                class_is_stable=bool(diagnostics.class_is_locked),
                class_counts=dict(diagnostics.class_observation_counts),
                class_confidence_sums=dict(diagnostics.class_scores),
                class_max_confidences=dict(diagnostics.class_max_confidences),
                class_winner_margin=diagnostics.class_winner_margin,
                class_status=diagnostics.class_status,
                winning_class_ratio=diagnostics.winning_class_ratio,
                runner_up_class=diagnostics.runner_up_class_name,
                runner_up_ratio=diagnostics.runner_up_ratio,
                maximum_consecutive_winner_count=diagnostics.maximum_consecutive_winner_count,
                recent_winning_class=diagnostics.recent_winning_class_name,
                recent_winning_ratio=diagnostics.recent_winning_ratio,
                class_transition_count=diagnostics.class_transition_count,
                incompatible_transition_count=diagnostics.incompatible_class_transition_count,
                count_winner=diagnostics.count_winner_class_name,
                score_winner=diagnostics.score_winner_class_name,
                winner_agreement=diagnostics.winners_agree,
                strong_conflict_detected=diagnostics.strong_conflict_detected,
                split_recommended=diagnostics.split_recommended,
                latest_observation_class=diagnostics.latest_observation_class_name,
                persisted=False,
                insufficient_history=False,
            )
        )
    fragment_candidates: list[FragmentLinkEvaluation] = []
    if config.fragment_linking.enabled:
        for camera_code, tracks in by_camera.items():
            ordered = sorted(tracks, key=lambda item: item.first_video_time_seconds)
            for index in range(len(ordered) - 1):
                fragment_candidates.append(evaluate_fragment_link(ordered[index], ordered[index + 1], config))
    return recalculated, fragment_candidates


def _persist_track_updates(
    client: AnalyticsDatabaseClient,
    *,
    track_rows: list[dict[str, Any]],
    observations_by_track: dict[str, list[dict[str, Any]]],
    recalculated_tracks: list[RecalculatedTrack],
    config: TrackingConfig,
) -> list[RecalculatedTrack]:
    by_uuid = {item.track_uuid: item for item in recalculated_tracks}
    updated: list[RecalculatedTrack] = []
    for track_row in track_rows:
        track_uuid = str(track_row.get("track_uuid") or "")
        item = by_uuid.get(track_uuid)
        if item is None or item.insufficient_history or item.new_final_class in (None, ""):
            updated.append(
                item
                or RecalculatedTrack(
                    track_uuid="",
                    camera_code="",
                    local_track_id=0,
                    observation_count=0,
                    old_final_class=None,
                    new_final_class=None,
                    class_confidence=None,
                    class_is_stable=False,
                    class_counts={},
                    class_confidence_sums={},
                    class_max_confidences={},
                    class_winner_margin=None,
                    class_status="INSUFFICIENT_OBSERVATIONS",
                    winning_class_ratio=None,
                    runner_up_class=None,
                    runner_up_ratio=None,
                    maximum_consecutive_winner_count=0,
                    recent_winning_class=None,
                    recent_winning_ratio=None,
                    class_transition_count=0,
                    incompatible_transition_count=0,
                    count_winner=None,
                    score_winner=None,
                    winner_agreement=True,
                    strong_conflict_detected=False,
                    split_recommended=False,
                    latest_observation_class=None,
                    persisted=False,
                    insufficient_history=True,
                )
            )
            continue
        observation_rows = observations_by_track.get(str(track_row.get("id")), [])
        observations = _rows_to_observations(track_row, observation_rows)
        diagnostics = recalculate_track_class(observations, config)
        metadata = _metadata(track_row)
        metadata["class_diagnostics"] = class_diagnostics_to_metadata(diagnostics)
        client.table("vehicle_track").update(
            {
                "vehicle_class": str(item.new_final_class).upper(),
                "metadata": metadata,
            }
        ).eq("id", str(track_row["id"])).execute()
        updated.append(
            RecalculatedTrack(
                **(asdict(item) | {"persisted": True})
            )
        )
    return updated


def generate_report(
    *,
    run_code: str,
    camera_code: str | None,
    track_uuid: str | None,
    tracking_config_path: str,
    persist: bool,
) -> dict[str, Any]:
    config = load_tracking_config(tracking_config_path)
    source = "analytics"
    notes: list[str] = []
    track_rows: list[dict[str, Any]] = []
    observations_by_track: dict[str, list[dict[str, Any]]] = {}
    client: AnalyticsDatabaseClient | None = None
    try:
        client = AnalyticsDatabaseClient(schema_name="analytics")
        track_rows, observations_by_track = _load_db_run_rows(
            client,
            run_code=run_code,
            camera_code=camera_code,
            track_uuid=track_uuid,
        )
    except (AnalyticsDatabaseClientError, RuntimeError) as exc:
        source = "artifact_report_only"
        notes.append(str(exc))
        report_candidates = _find_report_candidates(run_code)
        if not report_candidates:
            raise RuntimeError(f"No analytics data or debug report could be found for {run_code}.") from exc
        notes.append(
            "Debug report was found, but it does not contain per-observation class history for reliable recalculation."
        )
        payload = json.loads(report_candidates[0].read_text(encoding="utf-8"))
        for item in payload.get("completed_tracks", []):
            if camera_code and str(item.get("camera_code")) != camera_code:
                continue
            if track_uuid and str(item.get("track_uuid")) != track_uuid:
                continue
            track_rows.append(
                {
                    "id": str(item.get("track_uuid")),
                    "track_uuid": item.get("track_uuid"),
                    "camera_code": item.get("camera_code"),
                    "local_track_id": item.get("local_track_id"),
                    "vehicle_class": str(item.get("canonical_class_name") or item.get("class_name") or "").upper() or None,
                    "observation_count": item.get("observation_count"),
                    "metadata": {},
                }
            )
            observations_by_track[str(item.get("track_uuid"))] = []
    recalculated_tracks, fragment_candidates = _recalculate_track_rows(track_rows, observations_by_track, config)
    if persist and client is not None and source == "analytics":
        recalculated_tracks = _persist_track_updates(
            client,
            track_rows=track_rows,
            observations_by_track=observations_by_track,
            recalculated_tracks=recalculated_tracks,
            config=config,
        )
    return {
        "run_code": run_code,
        "camera_code": camera_code,
        "track_uuid": track_uuid,
        "source": source,
        "persist_requested": persist,
        "tracking_config_path": str(Path(tracking_config_path)),
        "tracks": [asdict(item) for item in recalculated_tracks],
        "fragment_candidates": [asdict(item) for item in fragment_candidates],
        "notes": notes,
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"Run: {report['run_code']}")
    if report.get("camera_code"):
        print(f"Camera: {report['camera_code']}")
    if report.get("track_uuid"):
        print(f"Track: {report['track_uuid']}")
    print(f"Source: {report['source']}")
    print()
    for track in report["tracks"]:
        print(track["track_uuid"])
        print(f"  old_final_class: {track['old_final_class']}")
        print(f"  new_final_class: {track['new_final_class']}")
        print(f"  class_status: {track.get('class_status')}")
        print(f"  class_counts: {track['class_counts']}")
        print(f"  class_confidence_sums: {track['class_confidence_sums']}")
        print(f"  winning_class_ratio: {track.get('winning_class_ratio')}")
        print(f"  count_winner/score_winner: {track.get('count_winner')} / {track.get('score_winner')}")
        print(f"  latest_observation_class: {track['latest_observation_class']}")
        print(f"  persisted: {track['persisted']}")
        if track.get("note"):
            print(f"  note: {track['note']}")
    if report["fragment_candidates"]:
        print()
        print("Fragment candidates")
        for candidate in report["fragment_candidates"]:
            print(
                f"  {candidate['previous_track_uuid']} -> {candidate['next_track_uuid']}: "
                f"eligible={candidate['eligible']} reasons={candidate['reasons']}"
            )
    if report["notes"]:
        print()
        print("Notes")
        for note in report["notes"]:
            print(f"  - {note}")


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    persist = bool(args.persist)
    try:
        report = generate_report(
            run_code=args.run_code,
            camera_code=args.camera_code,
            track_uuid=args.track_uuid,
            tracking_config_path=args.tracking_config,
            persist=persist,
        )
    except AnalyticsDatabaseClientError as exc:
        print(f"Configuration error: {exc}")
        return EXIT_CONFIGURATION_MISSING
    except RuntimeError as exc:
        print(f"Recalculation failed: {exc}")
        return EXIT_QUERY_FAILED
    print_report(report)
    if args.output_report:
        output_path = Path(args.output_report)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
        print(f"Report written: {output_path}")
    return EXIT_SUCCESS


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
