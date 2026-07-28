from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..detection.detection_config import detection_overrides_from_env, load_detection_config
from ..evidence.evidence_config import EvidenceConfig, load_evidence_config
from ..ingestion.camera_config import load_camera_configs
from ..persistence.persistence_config import PersistenceConfig, load_persistence_config, persistence_overrides_from_env
from ..tracking.tracking_config import load_tracking_config, tracking_overrides_from_env
from ..workers.worker_config import load_worker_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print the effective multicamera tracking runtime configuration.")
    parser.add_argument("--camera-config", default=None)
    parser.add_argument("--detection-config", required=True)
    parser.add_argument("--tracking-config", required=True)
    parser.add_argument("--worker-config", required=True)
    parser.add_argument("--persistence-config", default=None)
    parser.add_argument("--evidence-config", default=None)
    parser.add_argument("--anpr-config", default=None)
    parser.add_argument("--camera-code", default=None)
    parser.add_argument("--camera-codes", nargs="+", default=None)
    parser.add_argument("--camera-limit", type=int, default=None)
    parser.add_argument("--persist-to-supabase", action="store_true")
    parser.add_argument("--dry-run-persistence", action="store_true")
    parser.add_argument("--json-output", default=None)
    return parser.parse_args()


def _load_yaml_mapping(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def _selected_camera_codes(args: argparse.Namespace) -> list[str]:
    requested: list[str] = []
    if args.camera_code:
        requested.append(str(args.camera_code))
    if args.camera_codes:
        requested.extend(str(value) for value in args.camera_codes)
    seen: set[str] = set()
    ordered: list[str] = []
    for code in requested:
        if code in seen:
            continue
        seen.add(code)
        ordered.append(code)
    return ordered


def _build_worker_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "enabled": True,
        "enable_persistence_worker": args.persist_to_supabase or args.dry_run_persistence,
        "enable_anpr_worker": bool(args.anpr_config),
    }


def _build_persistence_overrides(args: argparse.Namespace) -> dict[str, Any]:
    backend = "disabled"
    if args.dry_run_persistence:
        backend = "dry_run"
    elif args.persist_to_supabase:
        backend = "analytics_supabase"
    return {
        "backend": backend,
        "enabled": args.persist_to_supabase or args.dry_run_persistence,
        "dry_run": args.dry_run_persistence,
    }


def _tracking_parameter_sources(
    yaml_tracking: dict[str, Any],
    env_overrides: dict[str, Any],
    effective: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for key, final_value in effective.items():
        if key in {"class_stabilization", "fragment_linking", "identity_continuity"}:
            continue
        rows[key] = {
            "yaml": yaml_tracking.get(key, "<code default>"),
            "env_override": env_overrides.get(key),
            "cli_override": None,
            "effective": final_value,
        }
    return rows


def _detection_parameter_sources(
    yaml_detection: dict[str, Any],
    env_overrides: dict[str, Any],
    effective: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for key, final_value in effective.items():
        rows[key] = {
            "yaml": yaml_detection.get(key, "<code default>"),
            "env_override": env_overrides.get(key),
            "cli_override": None,
            "effective": final_value,
        }
    return rows


def _worker_parameter_sources(
    yaml_workers: dict[str, Any],
    cli_overrides: dict[str, Any],
    effective: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for key, final_value in effective.items():
        rows[key] = {
            "yaml": yaml_workers.get(key, "<code default>"),
            "env_override": None,
            "cli_override": cli_overrides.get(key),
            "effective": final_value,
        }
    return rows


def _persistence_parameter_sources(
    yaml_persistence: dict[str, Any],
    env_overrides: dict[str, Any],
    cli_overrides: dict[str, Any],
    effective: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for key, final_value in effective.items():
        rows[key] = {
            "yaml": yaml_persistence.get(key, "<code default>"),
            "env_override": env_overrides.get(key),
            "cli_override": cli_overrides.get(key),
            "effective": final_value,
        }
    return rows


def _probe_camera_runtime(camera_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path_exists": camera_path.exists(),
        "source_fps": None,
        "frame_count": None,
        "frame_width": None,
        "frame_height": None,
    }
    if not camera_path.exists():
        return result
    try:
        import cv2
    except Exception:
        return result
    capture = cv2.VideoCapture(str(camera_path))
    if not capture.isOpened():
        capture.release()
        return result
    result["source_fps"] = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or None
    result["frame_count"] = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
    result["frame_width"] = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
    result["frame_height"] = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
    capture.release()
    return result


def generate_report(args: argparse.Namespace) -> dict[str, Any]:
    detection_config_path = Path(args.detection_config).expanduser().resolve()
    tracking_config_path = Path(args.tracking_config).expanduser().resolve()
    worker_config_path = Path(args.worker_config).expanduser().resolve()
    persistence_config_path = Path(args.persistence_config).expanduser().resolve() if args.persistence_config else None
    evidence_config_path = Path(args.evidence_config).expanduser().resolve() if args.evidence_config else worker_config_path.parent / "evidence.yaml"
    camera_config_path = Path(args.camera_config).expanduser().resolve() if args.camera_config else None

    detection_env = detection_overrides_from_env()
    tracking_env = tracking_overrides_from_env()
    persistence_env = persistence_overrides_from_env()
    worker_cli = _build_worker_overrides(args)
    persistence_cli = _build_persistence_overrides(args)

    detection_config = load_detection_config(detection_config_path, overrides=detection_env)
    tracking_config = load_tracking_config(tracking_config_path, overrides=tracking_env)
    worker_config = load_worker_config(worker_config_path, overrides=worker_cli)
    persistence_config = (
        load_persistence_config(persistence_config_path, overrides={**persistence_env, **persistence_cli})
        if persistence_config_path is not None
        else PersistenceConfig(**{**persistence_env, **persistence_cli}) if persistence_env or persistence_cli else PersistenceConfig()
    )
    evidence_config = load_evidence_config(evidence_config_path) if evidence_config_path.exists() else EvidenceConfig()

    yaml_detection = _load_yaml_mapping(detection_config_path).get("vehicle_detector", {})
    yaml_tracking = _load_yaml_mapping(tracking_config_path).get("tracking", {})
    yaml_workers = _load_yaml_mapping(worker_config_path).get("workers", {})
    yaml_persistence = _load_yaml_mapping(persistence_config_path).get("persistence", {}) if persistence_config_path else {}
    yaml_evidence = _load_yaml_mapping(evidence_config_path).get("evidence", {}) if evidence_config_path.exists() else {}

    selected_codes = _selected_camera_codes(args)
    camera_rows: list[dict[str, Any]] = []
    if camera_config_path is not None:
        camera_configs = load_camera_configs(camera_config_path, include_disabled=True, validate_paths=False)
        if selected_codes:
            camera_configs = [config for config in camera_configs if config.camera_code in selected_codes]
        if args.camera_limit is not None:
            camera_configs = camera_configs[: max(0, int(args.camera_limit))]
        for camera_config in camera_configs:
            probed = _probe_camera_runtime(camera_config.source_path)
            source_fps = probed["source_fps"]
            camera_rows.append(
                {
                    "camera_code": camera_config.camera_code,
                    "camera_name": camera_config.camera_name,
                    "enabled": camera_config.enabled,
                    "source_path": str(camera_config.source_path),
                    "configured_start_time": camera_config.start_time.isoformat() if camera_config.start_time is not None else None,
                    "path_exists": probed["path_exists"],
                    "probed_source_fps": source_fps,
                    "probed_frame_count": probed["frame_count"],
                    "probed_frame_width": probed["frame_width"],
                    "probed_frame_height": probed["frame_height"],
                    "effective_tracker_frame_rate": float(source_fps or tracking_config.frame_rate or 30.0),
                    "one_tracker_instance_for_camera": bool(camera_config.enabled),
                }
            )

    tracking_effective = asdict(tracking_config)
    detection_effective = {
        "model_path": detection_config.model_path,
        "fallback_model_path": detection_config.fallback_model_path,
        "allow_fallback": detection_config.allow_fallback,
        "device": detection_config.device,
        "confidence_threshold": detection_config.confidence_threshold,
        "iou_threshold": detection_config.iou_threshold,
        "image_size": detection_config.image_size,
        "allowed_classes": list(detection_config.allowed_classes),
    }
    worker_effective = asdict(worker_config)
    persistence_effective = asdict(persistence_config)
    evidence_effective = asdict(evidence_config)

    report = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "config_files_loaded": {
            "camera_config": str(camera_config_path) if camera_config_path else None,
            "detection_config": str(detection_config_path),
            "tracking_config": str(tracking_config_path),
            "worker_config": str(worker_config_path),
            "persistence_config": str(persistence_config_path) if persistence_config_path else None,
            "evidence_config": str(evidence_config_path) if evidence_config_path.exists() else None,
        },
        "precedence": {
            "detection": "environment override -> YAML -> code default",
            "tracking": "environment override -> YAML -> code default",
            "worker": "CLI/orchestrator override -> YAML -> code default",
            "persistence": "CLI/orchestrator override -> environment override -> YAML -> code default",
            "evidence": "YAML -> code default",
        },
        "environment_overrides": {
            "detection": detection_env,
            "tracking": tracking_env,
            "persistence": persistence_env,
        },
        "cli_overrides": {
            "worker": worker_cli,
            "persistence": persistence_cli,
            "camera_selection": {
                "camera_code": args.camera_code,
                "camera_codes": args.camera_codes,
                "camera_limit": args.camera_limit,
            },
        },
        "effective_runtime": {
            "detection": detection_effective,
            "tracking": tracking_effective,
            "worker": worker_effective,
            "persistence": persistence_effective,
            "evidence": evidence_effective,
        },
        "parameter_sources": {
            "detection": _detection_parameter_sources(yaml_detection, detection_env, detection_effective),
            "tracking": _tracking_parameter_sources(yaml_tracking, tracking_env, tracking_effective),
            "worker": _worker_parameter_sources(yaml_workers, worker_cli, worker_effective),
            "persistence": _persistence_parameter_sources(yaml_persistence, persistence_env, persistence_cli, persistence_effective),
            "evidence": {key: {"yaml": yaml_evidence.get(key, "<code default>"), "effective": value} for key, value in evidence_effective.items()},
        },
        "tracker_isolation": {
            "tracker_backend": tracking_config.backend,
            "preserve_state_per_camera_flag": tracking_config.preserve_state_per_camera,
            "tracker_factory_cache_key": "camera_code",
            "one_tracker_per_selected_camera": True,
            "shared_tracker_state_across_cameras": False,
            "tracker_factory_reset_method": "TrackerFactory.reset()",
            "camera_router_flush_methods": ["flush_camera", "flush_all"],
        },
        "cameras": camera_rows,
    }
    return report


def main() -> None:
    args = parse_args()
    report = generate_report(args)
    payload = json.dumps(report, indent=2)
    if args.json_output:
        output_path = Path(args.json_output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
