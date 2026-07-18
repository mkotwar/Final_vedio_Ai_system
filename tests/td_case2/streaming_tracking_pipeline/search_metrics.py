from __future__ import annotations

from collections import Counter
from typing import Any

from .search_query_schemas import VehicleSearchResponse


def build_search_metrics(responses: list[VehicleSearchResponse], *, records_indexed: int) -> dict[str, Any]:
    runtimes = [response.runtime_sec for response in responses]
    component_counts = Counter()
    status_counts = Counter()
    parser_warnings: list[str] = []
    duplicate_ids: list[str] = []
    for response in responses:
        parser_warnings.extend(response.query.warnings)
        response_ids = [result.record_id for result in response.results]
        duplicate_ids.extend(
            f"{response.query.raw_query}:{record_id}"
            for record_id, count in Counter(response_ids).items()
            if count > 1
        )
        for result in response.results:
            component_counts.update(result.score_components.keys())
            status_counts[result.plate_status] += 1
    return {
        "records_indexed": records_indexed,
        "queries_executed": len(responses),
        "queries_with_matches": sum(1 for response in responses if response.total_matches > 0),
        "queries_without_matches": sum(1 for response in responses if response.total_matches == 0),
        "average_query_runtime": round(sum(runtimes) / len(runtimes), 6) if runtimes else 0.0,
        "exact_plate_matches": component_counts["exact_verified_plate"] + component_counts["exact_weak_plate"],
        "plate_prefix_matches": component_counts["plate_prefix"],
        "class_filter_matches": component_counts["class"],
        "colour_filter_matches": component_counts["colour"],
        "time_filter_matches": component_counts["time_overlap"],
        "verified_results_returned": status_counts["verified"],
        "weak_results_returned": status_counts["weak"],
        "no_plate_results_returned": status_counts["no_plate_detected"],
        "duplicate_result_ids": sorted(duplicate_ids),
        "parser_warnings": sorted(set(parser_warnings)),
    }
