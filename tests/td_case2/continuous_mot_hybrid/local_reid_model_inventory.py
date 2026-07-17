from __future__ import annotations

from pathlib import Path
from typing import Any

from .report_writer import write_json


def discover_local_reid_models() -> list[dict[str, Any]]:
    search_roots = [
        Path.cwd(),
        Path.home() / ".cache",
        Path.home() / ".ultralytics",
        Path.home() / "AppData" / "Roaming" / "Ultralytics",
        Path.home() / "AppData" / "Local" / "Ultralytics",
    ]
    candidate_names = (
        "yolo11n-cls.pt",
        "yolo11s-cls.pt",
        "yolo26n-cls.pt",
        "yolo26s-cls.pt",
        "yolo26n-reid.onnx",
        "yolo26s-reid.onnx",
    )
    rows: list[dict[str, Any]] = []
    for root in search_roots:
        if not root.exists():
            continue
        for name in candidate_names:
            matches = list(root.rglob(name))
            if not matches:
                rows.append(
                    {
                        "path": str((root / name).resolve()),
                        "exists": False,
                        "size": 0,
                        "model_type": "classification" if "-cls." in name else "reid",
                        "ultralytics_loadable": False,
                        "device_compatibility": "unknown",
                        "selected_or_rejected_reason": "not_found",
                    }
                )
                continue
            for match in matches:
                rows.append(
                    {
                        "path": str(match.resolve()),
                        "exists": True,
                        "size": int(match.stat().st_size),
                        "model_type": "classification" if "-cls." in match.name else "reid",
                        "ultralytics_loadable": True,
                        "device_compatibility": "likely_cuda_and_cpu",
                        "selected_or_rejected_reason": "candidate_found",
                    }
                )
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        deduped[row["path"]] = row
    return sorted(deduped.values(), key=lambda item: (not bool(item["exists"]), item["path"]))


def write_local_reid_inventory(output_path: Path) -> dict[str, Any]:
    payload = {"status": "success", "models": discover_local_reid_models()}
    write_json(output_path, payload)
    return payload
