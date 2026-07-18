from __future__ import annotations

from pathlib import Path
from typing import Any

from .search_query_schemas import VehicleSearchResponse
from .serialization import write_json, write_jsonl


class SearchResultArtifactSink:
    def __init__(self, run_dir: str | Path, output_dir: str | Path | None = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else Path(run_dir) / "10_structured_search"
        self.report_dir = self.output_dir / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        *,
        index_summary: dict[str, Any],
        responses: list[VehicleSearchResponse],
        summary: dict[str, Any],
        report: dict[str, Any],
    ) -> dict[str, str]:
        paths = {
            "search_index_summary": self.output_dir / "search_index_summary.json",
            "validation_queries": self.output_dir / "validation_queries.json",
            "validation_search_results": self.output_dir / "validation_search_results.jsonl",
            "validation_search_results_flat": self.output_dir / "validation_search_results_flat.json",
            "summary": self.report_dir / "step10_search_summary.json",
            "report": self.report_dir / "step10_search_report.json",
        }
        write_json(paths["search_index_summary"], index_summary)
        write_json(paths["validation_queries"], [response.query.to_dict() for response in responses])
        write_jsonl(paths["validation_search_results"], responses)
        write_json(paths["validation_search_results_flat"], [response.to_dict() for response in responses])
        write_json(paths["summary"], summary)
        write_json(paths["report"], report)
        return {key: str(value) for key, value in paths.items()}
