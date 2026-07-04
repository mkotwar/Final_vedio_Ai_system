from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _load_index(run_dir: Path) -> list[dict[str, Any]]:
    index_path = run_dir / "06_searchable_object_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing search index: {index_path}")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    items = payload.get("items", []) if isinstance(payload, dict) else []
    if not isinstance(items, list):
        raise ValueError("Expected an 'items' list in 06_searchable_object_index.json")
    return items


def _score_item(item: dict[str, Any], query_terms: list[str], class_name: str, start_seconds: float | None, end_seconds: float | None) -> int | None:
    if class_name and str(item.get("class_name", "")).strip().lower() != class_name:
        return None
    start_time = float(item.get("start_time", 0.0) or 0.0)
    end_time_value = float(item.get("end_time", 0.0) or 0.0)
    if start_seconds is not None and end_time_value < start_seconds:
        return None
    if end_seconds is not None and start_time > end_seconds:
        return None

    search_text = str(item.get("search_text", "")).lower()
    score = 0
    for term in query_terms:
        if term in search_text:
            score += 5
        elif any(term in str(value).lower() for value in item.get("appearance_terms", [])):
            score += 3
    if query_terms and score == 0:
        return None
    score += int(item.get("frame_hit_count", 0) or 0)
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the isolated object-search testcase index")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--query", default="")
    parser.add_argument("--class-name", default="")
    parser.add_argument("--start-seconds", type=float, default=-1.0)
    parser.add_argument("--end-seconds", type=float, default=-1.0)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser()
    items = _load_index(run_dir)
    query_terms = [term for term in re.split(r"\s+", str(args.query).strip().lower()) if term]
    class_name = str(args.class_name).strip().lower()
    start_seconds = args.start_seconds if args.start_seconds >= 0 else None
    end_seconds = args.end_seconds if args.end_seconds >= 0 else None

    ranked: list[tuple[int, dict[str, Any]]] = []
    for item in items:
        score = _score_item(item, query_terms, class_name, start_seconds, end_seconds)
        if score is None:
            continue
        ranked.append((score, item))

    ranked.sort(key=lambda pair: (-pair[0], float(pair[1].get("start_time", 0.0) or 0.0), str(pair[1].get("object_id", ""))))
    limit = max(1, int(args.limit))
    if not ranked:
        print("No matches found.")
        return

    for score, item in ranked[:limit]:
        print(f"object_id: {item.get('object_id')}")
        print(f"class_name: {item.get('class_name')}")
        print(f"time: {item.get('start_time_text')} - {item.get('end_time_text')} ({item.get('duration_seconds')}s)")
        print(f"best_timestamp: {item.get('best_timestamp_text')}")
        print(f"appearance_terms: {', '.join(item.get('appearance_terms', []))}")
        print(f"best_frame_path: {item.get('best_frame_path')}")
        print(f"best_crop_path: {item.get('best_crop_path')}")
        print(f"frame_hits: {item.get('frame_hit_count')}")
        print(f"score: {score}")
        print("-" * 60)


if __name__ == "__main__":
    main()
