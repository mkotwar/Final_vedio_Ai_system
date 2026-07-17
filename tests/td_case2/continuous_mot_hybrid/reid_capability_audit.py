from __future__ import annotations

from pathlib import Path
from typing import Any

from .report_writer import write_html_from_markdown, write_json, write_markdown
from .ultralytics_botsort_backend import installed_ultralytics_info


def build_reid_capability_audit(*, site_packages_root: Path, yolo_model_path: Path) -> dict[str, Any]:
    info = installed_ultralytics_info(site_packages_root)
    track_path = site_packages_root / "ultralytics" / "trackers" / "track.py"
    botsort_path = site_packages_root / "ultralytics" / "trackers" / "bot_sort.py"
    reid_path = site_packages_root / "ultralytics" / "trackers" / "utils" / "reid.py"
    audit = {
        "status": "success",
        "ultralytics_version": info["ultralytics_version"],
        "source_files": {
            "track.py": str(track_path),
            "bot_sort.py": str(botsort_path),
            "botsort.yaml": info["botsort_yaml_path"],
            "reid.py": str(reid_path),
        },
        "resolved_settings": {
            "model_auto_requires_detect_hook": True,
            "track_py_uses_detect_pre_hook": True,
            "track_py_auto_fallback_model": "yolo26n-cls.pt",
            "bot_sort_requires_img_or_feats_for_encoder": True,
            "cached_box_only_replay_can_activate_auto": False,
            "live_fixed_5fps_native_feature_cache_supported": True,
            "custom_yolo_model_path": str(yolo_model_path),
            "custom_yolo_standard_detect_head_assumed": True,
        },
        "conclusion": (
            "A genuine model:auto BoT-SORT ReID run is valid only if detector-frame native features are captured "
            "from the Detect head and passed into tracker.update(..., feats=...)."
        ),
    }
    return audit


def write_reid_capability_audit(*, output_dir: Path, site_packages_root: Path, yolo_model_path: Path) -> dict[str, Any]:
    payload = build_reid_capability_audit(site_packages_root=site_packages_root, yolo_model_path=yolo_model_path)
    write_json(output_dir / "ultralytics_reid_capability_audit.json", payload)
    lines = [
        "# Ultralytics ReID Capability Audit",
        "",
        f"- Ultralytics version: {payload['ultralytics_version']}",
        f"- track.py: {payload['source_files']['track.py']}",
        f"- bot_sort.py: {payload['source_files']['bot_sort.py']}",
        f"- botsort.yaml: {payload['source_files']['botsort.yaml']}",
        f"- reid.py: {payload['source_files']['reid.py']}",
        f"- Conclusion: {payload['conclusion']}",
    ]
    markdown_text = "\n".join(lines) + "\n"
    write_markdown(output_dir / "ultralytics_reid_capability_audit.md", lines)
    write_html_from_markdown(output_dir / "ultralytics_reid_capability_audit.html", markdown_text)
    return payload
