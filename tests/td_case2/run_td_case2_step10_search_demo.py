from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import (
    DEFAULT_SEARCH_ALLOW_WEAK_OCR,
    DEFAULT_SEARCH_INCLUDE_FALLBACK,
    DEFAULT_SEARCH_MODE,
    DEFAULT_SEARCH_REQUIRE_IMAGE_PATHS,
    DEFAULT_SEARCH_SAVE_DEBUG,
    DEFAULT_SEARCH_TIME_TOLERANCE_SECONDS,
    DEFAULT_SEARCH_TOP_K,
    ENV_RUN_DIR,
    ENV_SEARCH_ALLOW_WEAK_OCR,
    ENV_SEARCH_INCLUDE_FALLBACK,
    ENV_SEARCH_MODE,
    ENV_SEARCH_QUERIES,
    ENV_SEARCH_QUERY,
    ENV_SEARCH_REQUIRE_IMAGE_PATHS,
    ENV_SEARCH_SAVE_DEBUG,
    ENV_SEARCH_TIME_TOLERANCE_SECONDS,
    ENV_SEARCH_TOP_K,
)
from run_td_case2_step01_02 import log
from stage_checks import build_failure_payload, update_stage_gate_report, write_json
from step_10_search_demo_runner import ALLOWED_SEARCH_MODES, DEFAULT_DEMO_QUERIES, run_search_demo, write_json_any


@dataclass(frozen=True)
class Step10Config:
    """Runtime settings for isolated Step 10 search demo runner."""

    run_dir: Path
    query_inputs: list[str]
    search_config: dict[str, Any]


def _read_bool(raw_value: str | None, default_value: bool, env_name: str) -> bool:
    """Read a permissive boolean-like flag."""

    if raw_value is None or raw_value.strip() == "":
        return default_value
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Environment variable {env_name} must be boolean-like. Received: {raw_value!r}")


def _read_positive_int(raw_value: str | None, default_value: int, env_name: str) -> int:
    """Read a positive integer value."""

    candidate = str(default_value) if raw_value is None or raw_value.strip() == "" else raw_value.strip()
    try:
        value = int(candidate)
    except ValueError as exc:
        raise ValueError(f"Environment variable {env_name} must be a valid integer. Received: {candidate!r}") from exc
    if value <= 0:
        raise ValueError(f"Environment variable {env_name} must be greater than 0. Received: {value}")
    return value


def _read_positive_float(raw_value: str | None, default_value: float, env_name: str) -> float:
    """Read a positive float value."""

    candidate = str(default_value) if raw_value is None or raw_value.strip() == "" else raw_value.strip()
    try:
        value = float(candidate)
    except ValueError as exc:
        raise ValueError(f"Environment variable {env_name} must be a valid number. Received: {candidate!r}") from exc
    if value <= 0:
        raise ValueError(f"Environment variable {env_name} must be greater than 0. Received: {value}")
    return value


def _parse_args() -> argparse.Namespace:
    """Parse optional CLI arguments."""

    parser = argparse.ArgumentParser(description="Run isolated td_case2 Step 10 live search demo.")
    parser.add_argument("--query", dest="query", help="Single search query string.", default=None)
    parser.add_argument("--mode", dest="mode", help="Search mode.", default=None)
    parser.add_argument("--top-k", dest="top_k", type=int, help="Maximum matches to return.", default=None)
    parser.add_argument("--run-dir", dest="run_dir", help="Existing td_case2 run directory.", default=None)
    return parser.parse_args()


def _read_query_inputs(cli_args: argparse.Namespace) -> list[str]:
    """Resolve query inputs from CLI or environment."""

    if cli_args.query and cli_args.query.strip():
        return [cli_args.query.strip()]

    env_single_query = os.environ.get(ENV_SEARCH_QUERY, "").strip()
    if env_single_query:
        return [env_single_query]

    env_batch_query = os.environ.get(ENV_SEARCH_QUERIES, "").strip()
    if env_batch_query:
        parsed = json.loads(env_batch_query)
        if not isinstance(parsed, list):
            raise ValueError(f"{ENV_SEARCH_QUERIES} must be a JSON list of query strings.")
        return [str(item).strip() for item in parsed if str(item).strip()]

    return list(DEFAULT_DEMO_QUERIES)


def read_config() -> Step10Config:
    """Read configuration for isolated Step 10 search demo runner."""

    cli_args = _parse_args()
    raw_run_dir = cli_args.run_dir or os.environ.get(ENV_RUN_DIR, "").strip()
    if not raw_run_dir:
        raise ValueError(f"Environment variable {ENV_RUN_DIR} is required for Step 10.")
    run_dir = Path(raw_run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"TD_CASE2_RUN_DIR does not point to an existing directory: {run_dir}")

    required_inputs = [
        run_dir / "07_vehicle_search_index.json",
        run_dir / "07_vehicle_search_index_flat.json",
    ]
    for required_path in required_inputs:
        if not required_path.exists():
            raise FileNotFoundError(f"Required Step 10 input is missing: {required_path}")

    raw_mode = cli_args.mode or os.environ.get(ENV_SEARCH_MODE, DEFAULT_SEARCH_MODE)
    mode = str(raw_mode or DEFAULT_SEARCH_MODE).strip().lower()
    if mode not in ALLOWED_SEARCH_MODES:
        raise ValueError(f"Search mode must be one of {sorted(ALLOWED_SEARCH_MODES)}. Received: {mode!r}")

    raw_top_k = str(cli_args.top_k) if cli_args.top_k is not None else os.environ.get(ENV_SEARCH_TOP_K)
    query_inputs = _read_query_inputs(cli_args)
    return Step10Config(
        run_dir=run_dir.resolve(),
        query_inputs=query_inputs,
        search_config={
            "mode": mode,
            "top_k": _read_positive_int(raw_top_k, DEFAULT_SEARCH_TOP_K, ENV_SEARCH_TOP_K),
            "allow_weak_ocr": _read_bool(
                os.environ.get(ENV_SEARCH_ALLOW_WEAK_OCR),
                DEFAULT_SEARCH_ALLOW_WEAK_OCR,
                ENV_SEARCH_ALLOW_WEAK_OCR,
            ),
            "time_tolerance_seconds": _read_positive_float(
                os.environ.get(ENV_SEARCH_TIME_TOLERANCE_SECONDS),
                DEFAULT_SEARCH_TIME_TOLERANCE_SECONDS,
                ENV_SEARCH_TIME_TOLERANCE_SECONDS,
            ),
            "include_fallback": _read_bool(
                os.environ.get(ENV_SEARCH_INCLUDE_FALLBACK),
                DEFAULT_SEARCH_INCLUDE_FALLBACK,
                ENV_SEARCH_INCLUDE_FALLBACK,
            ),
            "require_image_paths": _read_bool(
                os.environ.get(ENV_SEARCH_REQUIRE_IMAGE_PATHS),
                DEFAULT_SEARCH_REQUIRE_IMAGE_PATHS,
                ENV_SEARCH_REQUIRE_IMAGE_PATHS,
            ),
            "save_debug": _read_bool(
                os.environ.get(ENV_SEARCH_SAVE_DEBUG),
                DEFAULT_SEARCH_SAVE_DEBUG,
                ENV_SEARCH_SAVE_DEBUG,
            ),
        },
    )


def _write_failed_reports(run_dir: Path, error_message: str) -> None:
    """Write Step 10 failure JSON artifacts."""

    response_payload = {
        "status": "failed",
        "schema_version": "v1",
        "cards": [],
        "message": error_message,
        "error_message": error_message,
    }
    report_payload = {
        "status": "failed",
        "source_index_file": "07_vehicle_search_index.json",
        "source_schema_file": "09_search_result_card_schema.json",
        "records_loaded": 0,
        "queries_run": 0,
        "total_cards_returned": 0,
        "queries_with_results": 0,
        "queries_without_results": 0,
        "queries_blocked_invalid_ocr": 0,
        "path_availability": {
            "cards_with_crop": 0,
            "cards_missing_crop": 0,
            "cards_with_full_frame": 0,
            "cards_missing_full_frame": 0,
        },
        "example_queries": [],
        "recommendation": error_message,
        "error_message": error_message,
    }
    write_json(run_dir / "10_search_demo_response.json", response_payload)
    write_json_any(run_dir / "10_search_demo_results_flat.json", [])
    write_json(run_dir / "10_search_demo_report.json", report_payload)
    write_json_any(run_dir / "10_search_demo_query_log.json", [])


def main() -> None:
    """Run isolated Step 10 live search demo."""

    config = read_config()
    log(f"Run directory: {config.run_dir}")
    log(f"Records source: {config.run_dir / '07_vehicle_search_index.json'}")

    try:
        response_payload, flat_results, report_payload, _query_log = run_search_demo(
            run_dir=config.run_dir,
            query_inputs=config.query_inputs,
            search_config=config.search_config,
        )
        queries_run = len(config.query_inputs)
        if queries_run == 1:
            log(f"Query: {config.query_inputs[0]}")
            log(f"Resolved mode: {response_payload['resolved_mode']}")
            log(f"Total matches: {response_payload['total_matches']}")
            log(f"Cards returned: {response_payload['cards_returned']}")
        else:
            log(f"Query count: {queries_run}")
            log(f"Cards returned: {len(flat_results)}")
        update_stage_gate_report(
            config.run_dir,
            "10_search_demo_runner",
            {
                "status": response_payload["status"],
                "records_loaded": report_payload["records_loaded"],
                "queries_run": report_payload["queries_run"],
                "total_cards_returned": report_payload["total_cards_returned"],
                "queries_with_results": report_payload["queries_with_results"],
                "queries_blocked_invalid_ocr": report_payload["queries_blocked_invalid_ocr"],
                "ready_for_step11_event_candidate_generation": response_payload["status"] == "success",
            },
        )
        log(f"Output response path: {config.run_dir / '10_search_demo_response.json'}")
        log(f"Output report path: {config.run_dir / '10_search_demo_report.json'}")
    except Exception as exc:
        _write_failed_reports(config.run_dir, str(exc))
        update_stage_gate_report(config.run_dir, "10_search_demo_runner", build_failure_payload(exc))
        log(f"Step 10 failed: {exc}")
        log(f"Run directory: {config.run_dir}")
        raise


if __name__ == "__main__":
    main()
