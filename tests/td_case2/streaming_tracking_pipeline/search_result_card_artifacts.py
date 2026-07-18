from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

from .search_result_card_schemas import VehicleResultCardPackage
from .serialization import write_json, write_jsonl


RESULT_CARD_SCHEMA = {
    "type": "object",
    "required": [
        "rank",
        "record_id",
        "track_id",
        "track_generation",
        "title",
        "subtitle",
        "time_label",
        "plate_label",
        "status_badge",
        "search_score",
    ],
    "properties": {
        "rank": {"type": "integer"},
        "record_id": {"type": "string"},
        "track_id": {"type": "integer"},
        "track_generation": {"type": "integer"},
        "title": {"type": "string"},
        "subtitle": {"type": "string"},
        "thumbnail_path": {"type": ["string", "null"]},
        "secondary_image_path": {"type": ["string", "null"]},
    },
}


class SearchResultCardArtifactSink:
    def __init__(self, run_dir: str | Path, output_dir: str | Path | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.output_dir = Path(output_dir) if output_dir else self.run_dir / "11_result_cards"
        self.report_dir = self.output_dir / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        *,
        packages: list[VehicleResultCardPackage],
        summary: dict[str, Any],
        report: dict[str, Any],
        write_html_preview: bool = False,
    ) -> dict[str, str]:
        cards = [card for package in packages for card in package.cards]
        paths = {
            "result_card_packages": self.output_dir / "result_card_packages.jsonl",
            "result_cards_flat": self.output_dir / "result_cards_flat.json",
            "demo_query_cards": self.output_dir / "demo_query_cards.json",
            "result_card_schema": self.output_dir / "result_card_schema.json",
            "summary": self.report_dir / "step11_result_cards_summary.json",
            "report": self.report_dir / "step11_result_cards_report.json",
        }
        write_jsonl(paths["result_card_packages"], packages)
        write_json(paths["result_cards_flat"], [card.to_dict() for card in cards])
        write_json(paths["demo_query_cards"], [package.to_dict() for package in packages])
        write_json(paths["result_card_schema"], RESULT_CARD_SCHEMA)
        write_json(paths["summary"], summary)
        write_json(paths["report"], report)
        if write_html_preview:
            paths["html_preview"] = self.output_dir / "demo_result_cards.html"
            paths["html_preview"].write_text(self._render_html(packages), encoding="utf-8")
        return {key: str(path) for key, path in paths.items()}

    def _render_html(self, packages: list[VehicleResultCardPackage]) -> str:
        sections: list[str] = []
        for package in packages:
            cards = "\n".join(self._render_card(card) for card in package.cards)
            sections.append(f"<section><h2>{html.escape(package.raw_query)}</h2><div class=\"grid\">{cards}</div></section>")
        return f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<title>Step 11 Result Cards</title>
<style>
body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; background: #f6f7f9; color: #20242a; }}
h1 {{ margin: 0 0 16px; }}
h2 {{ margin: 28px 0 12px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 14px; }}
.card {{ background: white; border: 1px solid #d8dde5; border-radius: 8px; overflow: hidden; }}
.thumb {{ width: 100%; aspect-ratio: 16 / 10; object-fit: cover; background: #e9edf2; display: block; }}
.body {{ padding: 12px; }}
.title {{ font-weight: 700; margin-bottom: 4px; }}
.sub {{ color: #5b6470; font-size: 13px; margin-bottom: 8px; }}
.badge {{ display: inline-block; padding: 3px 7px; border-radius: 999px; background: #edf2ff; color: #274690; font-size: 12px; margin-bottom: 8px; }}
.meta {{ font-size: 13px; line-height: 1.45; }}
.plate {{ max-width: 130px; max-height: 58px; object-fit: contain; border: 1px solid #d8dde5; background: #fafafa; margin-top: 8px; }}
</style>
</head>
<body>
<h1>Step 11 Result Cards</h1>
{''.join(sections)}
</body>
</html>
"""

    def _render_card(self, card: Any) -> str:
        thumb = self._html_path(card.thumbnail_path)
        plate = self._html_path(card.secondary_image_path)
        image_html = f"<img class=\"thumb\" src=\"{html.escape(thumb)}\" alt=\"vehicle crop\">" if thumb else "<div class=\"thumb\"></div>"
        plate_html = f"<img class=\"plate\" src=\"{html.escape(plate)}\" alt=\"plate crop\">" if plate else ""
        return f"""
<article class=\"card\">
{image_html}
<div class=\"body\">
<div class=\"title\">#{card.rank} {html.escape(card.title)}</div>
<div class=\"sub\">{html.escape(card.subtitle)}</div>
<div class=\"badge\">{html.escape(card.status_badge)}</div>
<div class=\"meta\">
Class: {html.escape(str(card.object_class or 'Unknown'))}<br>
Colour: {html.escape(card.colour_label)}<br>
Plate: {html.escape(card.plate_label)}<br>
Score: {card.search_score:.1f}<br>
Record: {html.escape(card.record_id)}
</div>
{plate_html}
</div>
</article>
"""

    def _html_path(self, value: str | None) -> str | None:
        if not value:
            return None
        path = Path(value)
        absolute = path if path.is_absolute() else Path.cwd() / path
        try:
            return os.path.relpath(absolute, self.output_dir).replace("\\", "/")
        except ValueError:
            return str(value).replace("\\", "/")
