from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from ..persistence.analytics_database_client import AnalyticsDatabaseClient, AnalyticsDatabaseClientConfig, _normalize_supabase_project_url
from ..persistence.api_read_repository import AnalyticsReadRepository
from ..persistence.persistence_models import TrackMediaRecord
from ..persistence.track_media_repository import TrackMediaRepository
from ..persistence.track_media_types import TRACK_MEDIA_TYPE_ANNOTATED_FULL_FRAME, TRACK_MEDIA_TYPE_FULL_FRAME


REPO_ROOT = Path(__file__).resolve().parents[4]
PIPELINE_ENV = REPO_ROOT / "tests" / "td_case2" / "multicamera_vehicle_tracking_pipeline" / ".env.example"
ARTIFACT_ROOT = REPO_ROOT / "artifacts"


def _load_env(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ[name.strip()] = value.strip().strip('"').strip("'")
    if not os.environ.get("SUPABASE_URL") and os.environ.get("supabase_database_url"):
        os.environ["SUPABASE_URL"] = _normalize_supabase_project_url(os.environ["supabase_database_url"])


def _build_client() -> AnalyticsDatabaseClient:
    _load_env(PIPELINE_ENV)
    config = AnalyticsDatabaseClientConfig(
        supabase_url=_normalize_supabase_project_url(os.environ["SUPABASE_URL"]),
        supabase_service_role_key=os.environ["SUPABASE_SERVICE_ROLE_KEY"],
        schema_name="analytics",
    )
    return AnalyticsDatabaseClient(config=config, schema_name="analytics")


def _artifact_dir(run_code: str, camera_code: str, local_track_id: int) -> Path:
    return ARTIFACT_ROOT / run_code / camera_code / f"track_{local_track_id:06d}" / f"{run_code}_{camera_code}_TRACK_{local_track_id}"


def _relative_storage_uri(path: Path) -> str:
    return path.resolve().relative_to(ARTIFACT_ROOT.resolve()).as_posix()


def _record_for_file(
    *,
    vehicle_track_id: str,
    track_uuid: str,
    local_track_id: int,
    camera_id: str,
    media_type: str,
    file_path: Path,
    frame_number: int | None,
    video_time_seconds: float | None,
    bbox_xyxy: list[float] | None,
    width: int | None,
    height: int | None,
    selection_rank: int,
) -> TrackMediaRecord:
    metadata: dict[str, Any] = {
        "track_uuid": track_uuid,
        "local_track_id": local_track_id,
        "camera_id": camera_id,
        "source_role": "BEST_OVERALL",
    }
    if bbox_xyxy is not None:
        metadata["bbox_xyxy"] = bbox_xyxy
    return TrackMediaRecord(
        vehicle_track_id=vehicle_track_id,
        media_type=media_type,
        storage_uri=_relative_storage_uri(file_path),
        storage_provider="LOCAL",
        mime_type="image/jpeg",
        file_size_bytes=file_path.stat().st_size,
        frame_number=frame_number,
        captured_at=None,
        video_time_seconds=video_time_seconds,
        bbox={"bbox_xyxy": bbox_xyxy} if bbox_xyxy is not None else None,
        width=width,
        height=height,
        quality_score=None,
        sharpness_score=None,
        visibility_score=None,
        occlusion_score=None,
        selection_rank=selection_rank,
        is_primary=False,
        metadata=metadata,
    )


def backfill_run(*, run_code: str, persist: bool) -> dict[str, Any]:
    client = _build_client()
    repository = AnalyticsReadRepository(client)
    media_repository = TrackMediaRepository(client)

    run = repository.find_run_by_code(run_code)
    if run is None:
        raise RuntimeError(f"Run not found: {run_code}")
    run_id = str(run["id"])

    tracks = client.table("vehicle_track").select("id,track_uuid,local_track_id,camera_id").eq("processing_run_id", run_id).execute().data or []
    camera_rows = client.table("camera").select("id,camera_code").execute().data or []
    camera_by_id = {str(row["id"]): str(row["camera_code"]) for row in camera_rows if row.get("id") and row.get("camera_code")}

    report: dict[str, Any] = {
        "run_code": run_code,
        "persist": persist,
        "inserted": 0,
        "skipped_existing": 0,
        "missing_artifacts": 0,
        "failed": 0,
        "tracks": [],
    }

    for track in tracks:
        vehicle_track_id = str(track["id"])
        track_uuid = str(track["track_uuid"])
        local_track_id = int(track["local_track_id"])
        camera_id = str(track["camera_id"])
        camera_code = camera_by_id.get(camera_id)
        if not camera_code:
            report["failed"] += 1
            report["tracks"].append({"track_uuid": track_uuid, "status": "failed", "reason": "camera_code_missing"})
            continue

        base_dir = _artifact_dir(run_code, camera_code, local_track_id)
        manifest_path = base_dir / "evidence_manifest.json"
        if not manifest_path.exists():
            report["missing_artifacts"] += 1
            report["tracks"].append({"track_uuid": track_uuid, "status": "missing_manifest"})
            continue

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        best_overall = manifest.get("roles", {}).get("best_overall")
        if not isinstance(best_overall, dict):
            report["missing_artifacts"] += 1
            report["tracks"].append({"track_uuid": track_uuid, "status": "missing_best_overall"})
            continue

        bbox_xyxy = best_overall.get("original_bbox")
        frame_number = best_overall.get("frame_number")
        video_time_seconds = best_overall.get("timestamp_seconds")

        candidate_files = [
            (TRACK_MEDIA_TYPE_FULL_FRAME, base_dir / str(best_overall.get("source_frame", "")), 5),
            (TRACK_MEDIA_TYPE_ANNOTATED_FULL_FRAME, base_dir / str(best_overall.get("annotated_frame", "")), 5),
        ]

        track_result = {"track_uuid": track_uuid, "status": "skipped", "rows": []}
        for media_type, file_path, selection_rank in candidate_files:
            if not file_path.exists():
                report["missing_artifacts"] += 1
                track_result["rows"].append({"media_type": media_type, "status": "missing_file"})
                continue
            existing = media_repository.get_existing(
                vehicle_track_id=vehicle_track_id,
                media_type=media_type,
                storage_uri=_relative_storage_uri(file_path),
            )
            if existing is not None:
                report["skipped_existing"] += 1
                track_result["rows"].append({"media_type": media_type, "status": "already_exists"})
                continue
            record = _record_for_file(
                vehicle_track_id=vehicle_track_id,
                track_uuid=track_uuid,
                local_track_id=local_track_id,
                camera_id=camera_id,
                media_type=media_type,
                file_path=file_path,
                frame_number=int(frame_number) if frame_number is not None else None,
                video_time_seconds=float(video_time_seconds) if video_time_seconds is not None else None,
                bbox_xyxy=[float(value) for value in bbox_xyxy] if isinstance(bbox_xyxy, list) else None,
                width=None,
                height=None,
                selection_rank=selection_rank,
            )
            if persist:
                media_repository.upsert(record)
                report["inserted"] += 1
                track_result["rows"].append({"media_type": media_type, "status": "inserted"})
            else:
                record.to_payload()
                track_result["rows"].append({"media_type": media_type, "status": "dry_run"})
        report["tracks"].append(track_result)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill missing FULL_FRAME and ANNOTATED_FULL_FRAME media rows for an existing run.")
    parser.add_argument("--run-code", required=True)
    parser.add_argument("--persist", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-report", default="")
    args = parser.parse_args()

    persist = bool(args.persist and not args.dry_run)
    report = backfill_run(run_code=args.run_code, persist=persist)
    if args.output_report:
        output_path = Path(args.output_report)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
