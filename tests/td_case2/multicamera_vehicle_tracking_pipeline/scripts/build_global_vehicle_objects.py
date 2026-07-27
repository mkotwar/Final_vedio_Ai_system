from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from ..cross_camera.global_match_config import load_global_match_config
from ..cross_camera.global_match_service import GlobalMatchService
from ..persistence.analytics_database_client import AnalyticsDatabaseClient, AnalyticsDatabaseClientError
from ..persistence.cross_camera_match_repository import CrossCameraMatchRepository
from ..persistence.global_vehicle_object_repository import GlobalVehicleObjectRepository
from .logging_utils import configure_cli_logging


EXIT_SUCCESS = 0
EXIT_QUERY_FAILED = 1
EXIT_CONFIGURATION_MISSING = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build deterministic cross-camera global vehicle objects for one completed analytics run.")
    parser.add_argument("--run-code", required=True)
    parser.add_argument("--global-match-config", default="tests\\td_case2\\multicamera_vehicle_tracking_pipeline\\config\\global_matching.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Preview candidates and object proposals without writes. Default mode.")
    parser.add_argument("--persist", action="store_true", help="Write match/object/member rows to analytics.")
    parser.add_argument("--rebuild", action="store_true", help="Reserved safe reevaluation flag; existing rows are never deleted automatically.")
    parser.add_argument("--json-output")
    parser.add_argument("--limit-samples", type=int, default=5)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"))
    return parser


def configure_logging(level_name: str) -> None:
    configure_cli_logging(level_name)


def write_json_report(path_value: str, report: dict[str, object]) -> None:
    output_path = Path(path_value)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    if args.persist and args.dry_run:
        parser.error("--dry-run and --persist cannot be used together.")
    persist = bool(args.persist)
    try:
        config = load_global_match_config(args.global_match_config)
        client = AnalyticsDatabaseClient(schema_name="analytics")
        service = GlobalMatchService(config, CrossCameraMatchRepository(client), GlobalVehicleObjectRepository(client))
        report = service.build_for_run(args.run_code, persist=persist).to_dict()
    except AnalyticsDatabaseClientError as exc:
        print(f"Configuration error: {exc}")
        return EXIT_CONFIGURATION_MISSING
    except Exception as exc:
        print(f"Build failed: {exc}")
        return EXIT_QUERY_FAILED
    print(json.dumps(report, indent=2, ensure_ascii=True))
    if args.json_output:
        write_json_report(args.json_output, report)
        print(f"JSON report written: {args.json_output}")
    return EXIT_SUCCESS


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
