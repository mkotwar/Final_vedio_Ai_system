from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import (
    DEFAULT_STEP16_CLIP_FPS,
    DEFAULT_STEP16_HEADER_SECONDS,
    DEFAULT_STEP16_INCLUDE_NORMAL_CONTEXT_SCENE_EVENTS,
    DEFAULT_STEP16_INCLUDE_PERSON_EVENTS,
    DEFAULT_STEP16_MAX_OBJECT_EVENTS,
    DEFAULT_STEP16_OBJECT_CONTEXT_AFTER_SECONDS,
    DEFAULT_STEP16_OBJECT_CONTEXT_BEFORE_SECONDS,
    DEFAULT_STEP16_SUMMARY_SECONDS,
    ENV_RUN_DIR,
    ENV_STEP16_CLIP_FPS,
    ENV_STEP16_HEADER_SECONDS,
    ENV_STEP16_INCLUDE_NORMAL_CONTEXT_SCENE_EVENTS,
    ENV_STEP16_INCLUDE_PERSON_EVENTS,
    ENV_STEP16_MAX_OBJECT_EVENTS,
    ENV_STEP16_OBJECT_CONTEXT_AFTER_SECONDS,
    ENV_STEP16_OBJECT_CONTEXT_BEFORE_SECONDS,
    ENV_STEP16_SUMMARY_SECONDS,
)
from run_td_case2_step01_02 import log
from stage_checks import build_failure_payload, update_stage_gate_report, write_json
from step_16_evidence_video_generation import EVIDENCE_INDEX_NAME, EVIDENCE_VIDEO_NAME, EvidenceVideoConfig, build_evidence_video


@dataclass(frozen=True)
class Step16Config:
    run_dir: Path
    evidence_config: EvidenceVideoConfig


def _read_bool(env_name: str, default_value: bool) -> bool:
    raw_value = os.environ.get(env_name)
    if raw_value is None or raw_value.strip() == "":
        return default_value
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Environment variable {env_name} must be boolean-like. Received: {raw_value!r}")


def _read_positive_int(env_name: str, default_value: int) -> int:
    raw_value = os.environ.get(env_name, str(default_value)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {env_name} must be a valid integer. Received: {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"Environment variable {env_name} must be greater than 0. Received: {value}")
    return value


def _read_positive_float(env_name: str, default_value: float) -> float:
    raw_value = os.environ.get(env_name, str(default_value)).strip()
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {env_name} must be a valid number. Received: {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"Environment variable {env_name} must be greater than 0. Received: {value}")
    return value


def read_config() -> Step16Config:
    raw_run_dir = os.environ.get(ENV_RUN_DIR, "").strip()
    if not raw_run_dir:
        raise ValueError(f"Environment variable {ENV_RUN_DIR} is required for Step 16.")
    run_dir = Path(raw_run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"TD_CASE2_RUN_DIR does not point to an existing directory: {run_dir}")

    required_inputs = [
        run_dir / "01_video_info.json",
        run_dir / "04B_tracks.json",
        run_dir / "07B_traffic_object_search_index.json",
        run_dir / "11_full_scene_event_candidates.json",
        run_dir / "12_selected_top_event_candidates.json",
    ]
    for required_path in required_inputs:
        if not required_path.exists():
            raise FileNotFoundError(f"Required Step 16 input is missing: {required_path}")

    return Step16Config(
        run_dir=run_dir.resolve(),
        evidence_config=EvidenceVideoConfig(
            clip_fps=_read_positive_int(ENV_STEP16_CLIP_FPS, DEFAULT_STEP16_CLIP_FPS),
            header_seconds=_read_positive_float(ENV_STEP16_HEADER_SECONDS, DEFAULT_STEP16_HEADER_SECONDS),
            summary_seconds=_read_positive_float(ENV_STEP16_SUMMARY_SECONDS, DEFAULT_STEP16_SUMMARY_SECONDS),
            object_context_before_seconds=_read_positive_float(
                ENV_STEP16_OBJECT_CONTEXT_BEFORE_SECONDS,
                DEFAULT_STEP16_OBJECT_CONTEXT_BEFORE_SECONDS,
            ),
            object_context_after_seconds=_read_positive_float(
                ENV_STEP16_OBJECT_CONTEXT_AFTER_SECONDS,
                DEFAULT_STEP16_OBJECT_CONTEXT_AFTER_SECONDS,
            ),
            max_object_events=_read_positive_int(ENV_STEP16_MAX_OBJECT_EVENTS, DEFAULT_STEP16_MAX_OBJECT_EVENTS),
            include_person_events=_read_bool(ENV_STEP16_INCLUDE_PERSON_EVENTS, DEFAULT_STEP16_INCLUDE_PERSON_EVENTS),
            include_normal_context_scene_events=_read_bool(
                ENV_STEP16_INCLUDE_NORMAL_CONTEXT_SCENE_EVENTS,
                DEFAULT_STEP16_INCLUDE_NORMAL_CONTEXT_SCENE_EVENTS,
            ),
        ),
    )


def _write_failed_reports(run_dir: Path, error_message: str) -> None:
    write_json(
        run_dir / EVIDENCE_INDEX_NAME,
        {"status": "failed", "video_file": EVIDENCE_VIDEO_NAME, "clips": [], "error_message": error_message},
    )
    write_json(
        run_dir / "16_evidence_video_report.json",
        {"status": "failed", "video_file": EVIDENCE_VIDEO_NAME, "index_file": EVIDENCE_INDEX_NAME, "error_message": error_message},
    )


def main() -> None:
    config = read_config()
    log(f"Run directory: {config.run_dir}")
    try:
        index_payload, report_payload = build_evidence_video(config.run_dir, config.evidence_config)
        update_stage_gate_report(
            config.run_dir,
            "16_evidence_video_generation",
            {
                "status": "success",
                "event_count": report_payload["event_count"],
                "evidence_duration_seconds": report_payload["evidence_duration_seconds"],
                "vehicles_detected": report_payload["vehicles_detected"],
                "persons_detected": report_payload["persons_detected"],
                "license_plates_detected": report_payload["license_plates_detected"],
                "video_file": EVIDENCE_VIDEO_NAME,
                "index_file": EVIDENCE_INDEX_NAME,
            },
        )
        log(f"Evidence events exported: {report_payload['event_count']}")
        log(f"Evidence duration seconds: {report_payload['evidence_duration_seconds']}")
        log(f"Output paths: {config.run_dir / EVIDENCE_VIDEO_NAME} | {config.run_dir / EVIDENCE_INDEX_NAME}")
    except Exception as exc:
        _write_failed_reports(config.run_dir, str(exc))
        update_stage_gate_report(config.run_dir, "16_evidence_video_generation", build_failure_payload(exc))
        log(f"Step 16 failed: {exc}")
        log(f"Run directory: {config.run_dir}")
        raise


if __name__ == "__main__":
    main()
