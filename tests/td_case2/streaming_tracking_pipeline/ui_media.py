from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any

from .ui_data_loader import LoadedRunArtifacts, build_object_evidence, resolve_artifact_path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_DISPLAY_SPECS: dict[str, dict[str, Any]] = {
    "full_frame": {"width": 640, "height": 360, "aspect_ratio": "16 / 9", "fit": "contain"},
    "object_crop": {"width": 320, "height": 240, "aspect_ratio": "4 / 3", "fit": "contain"},
    "plate_crop": {"width": 300, "height": 100, "aspect_ratio": "3 / 1", "fit": "contain"},
    "thumbnail": {"width": 240, "height": 135, "aspect_ratio": "16 / 9", "fit": "contain"},
}


def image_status(path_value: str | Path | None, *, run_dir: str | Path, repo_root: str | Path | None = None) -> dict[str, Any]:
    resolved = resolve_artifact_path(path_value, run_dir=run_dir, repo_root=repo_root)
    return {
        "requested_path": str(path_value) if path_value else None,
        "resolved_path": str(resolved) if resolved else None,
        "exists": bool(resolved and resolved.exists()),
        "is_image": bool(resolved and resolved.suffix.lower() in IMAGE_EXTENSIONS),
        "placeholder": "Image missing" if not resolved else None,
    }


def image_display_spec(image_type: str) -> dict[str, Any]:
    return dict(IMAGE_DISPLAY_SPECS.get(image_type, IMAGE_DISPLAY_SPECS["thumbnail"]))


def render_evidence_image(
    st_target: Any,
    image_path: str | Path | None,
    image_type: str,
    *,
    run_dir: str | Path,
    repo_root: str | Path | None = None,
    caption: str | None = None,
    missing_message: str | None = None,
    fit: str | None = None,
) -> dict[str, Any]:
    status = image_status(image_path, run_dir=run_dir, repo_root=repo_root)
    spec = image_display_spec(image_type)
    if fit:
        spec["fit"] = fit
    status["display_spec"] = spec
    html_payload = _image_html(status, spec, caption=caption, missing_message=missing_message)
    st_target.markdown(html_payload, unsafe_allow_html=True)
    return status


def render_object_evidence_pair(
    st_target: Any,
    evidence: dict[str, Any],
    *,
    run_dir: str | Path,
    repo_root: str | Path | None = None,
    compact: bool = False,
) -> dict[str, Any]:
    columns = st_target.columns([2, 1] if not compact else [1, 1])
    full_frame_type = "thumbnail" if compact else "full_frame"
    crop_type = "object_crop"
    full_status = render_evidence_image(
        columns[0],
        evidence.get("full_frame_path"),
        full_frame_type,
        run_dir=run_dir,
        repo_root=repo_root,
        caption=full_frame_caption(evidence),
        missing_message="Full frame unavailable",
    )
    crop_status = render_evidence_image(
        columns[1],
        evidence.get("object_crop_path"),
        crop_type,
        run_dir=run_dir,
        repo_root=repo_root,
        caption=object_crop_caption(evidence),
        missing_message="Object crop unavailable",
    )
    return {"full_frame": full_status, "object_crop": crop_status}


def collect_record_media(record: dict[str, Any], artifacts: LoadedRunArtifacts) -> dict[str, Any]:
    selected = artifacts.selected_crops_by_identity.get(_identity_key(record)) or {}
    plate = artifacts.plate_validation_by_identity.get(_identity_key(record)) or {}
    vehicle_paths = _dedupe(
        [
            record.get("representative_vehicle_crop_path"),
            plate.get("representative_vehicle_crop_path"),
            *_extract_crop_paths(selected.get("primary_crops")),
            *_extract_crop_paths([selected.get("fallback_crop")]),
            *(record.get("primary_crop_paths") or []),
            *(record.get("fallback_crop_paths") or []),
        ]
    )
    plate_paths = _dedupe(
        [
            record.get("representative_plate_crop_path"),
            plate.get("representative_plate_crop_path"),
            _nested_get(plate, ["selected_candidate", "plate_crop_path"]),
        ]
    )
    annotated_paths = find_plate_diagnostic_images(record, artifacts)
    colour_debug = find_colour_debug_images(record, artifacts)
    evidence = build_object_evidence(record, artifacts)
    return {
        "object_evidence": evidence,
        "full_frame_image": image_status(evidence.get("full_frame_path"), run_dir=artifacts.run_dir, repo_root=artifacts.repo_root),
        "object_crop_image": image_status(evidence.get("object_crop_path"), run_dir=artifacts.run_dir, repo_root=artifacts.repo_root),
        "vehicle_images": [image_status(path, run_dir=artifacts.run_dir, repo_root=artifacts.repo_root) for path in vehicle_paths],
        "plate_images": [image_status(path, run_dir=artifacts.run_dir, repo_root=artifacts.repo_root) for path in plate_paths],
        "annotated_plate_images": annotated_paths,
        "colour_debug_images": colour_debug,
    }


def full_frame_caption(evidence: dict[str, Any]) -> str:
    parts = ["Full frame"]
    frame = evidence.get("full_frame_frame_index")
    if frame is None:
        frame = evidence.get("frame_index")
    timestamp = evidence.get("full_frame_timestamp_sec")
    if timestamp is None:
        timestamp = evidence.get("timestamp_sec")
    if frame is not None:
        parts.append(f"Frame {int(frame)}")
    if timestamp is not None:
        parts.append(_format_timestamp(float(timestamp)))
    return " · ".join(parts)


def object_crop_caption(evidence: dict[str, Any]) -> str:
    label = "Person crop" if str(evidence.get("object_class") or "").lower() == "person" else "Vehicle crop"
    parts = [label]
    if evidence.get("track_id") is not None:
        parts.append(f"Track {int(evidence['track_id'])}")
    if evidence.get("object_class"):
        parts.append(str(evidence["object_class"]))
    return " · ".join(parts)


def plate_crop_caption(evidence: dict[str, Any]) -> str:
    parts = ["Plate crop"]
    if evidence.get("plate_text"):
        parts.append(str(evidence["plate_text"]))
    if evidence.get("plate_status"):
        parts.append(str(evidence["plate_status"]))
    return " · ".join(parts)


def find_plate_diagnostic_images(record: dict[str, Any], artifacts: LoadedRunArtifacts) -> list[dict[str, Any]]:
    diagnostics_dir = artifacts.run_dir / "07_5_plate_diagnostics"
    if not diagnostics_dir.exists():
        return []
    terms = _identity_terms(record)
    matches = [
        path
        for path in diagnostics_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and all(term in path.as_posix() for term in terms)
    ]
    return [image_status(path, run_dir=artifacts.run_dir, repo_root=artifacts.repo_root) for path in sorted(matches)[:50]]


def find_colour_debug_images(record: dict[str, Any], artifacts: LoadedRunArtifacts) -> list[dict[str, Any]]:
    colour_dir = artifacts.run_dir / "12_object_class_colour_validation" / "colour_debug" / _safe_record_id(record)
    if not colour_dir.exists():
        return []
    matches = [path for path in colour_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return [image_status(path, run_dir=artifacts.run_dir, repo_root=artifacts.repo_root) for path in sorted(matches)]


def video_preview_info(record: dict[str, Any], source_metadata: dict[str, Any]) -> dict[str, Any]:
    source_path = source_metadata.get("source_path") or record.get("video_path")
    timestamp = record.get("representative_timestamp_sec")
    if timestamp is None:
        timestamp = record.get("first_seen_sec")
    frame_index = record.get("representative_frame_index")
    if frame_index is None:
        frame_index = record.get("first_frame_index")
    start = max(0.0, float(timestamp or 0.0) - 5.0)
    end = float(timestamp or 0.0) + 5.0
    return {
        "source_path": source_path,
        "timestamp_sec": timestamp,
        "frame_index": frame_index,
        "nearby_window_sec": [round(start, 3), round(end, 3)],
        "is_approximate": True,
        "note": "Streamlit video seeking may be approximate; evidence images remain primary.",
    }


def _image_html(
    status: dict[str, Any],
    spec: dict[str, Any],
    *,
    caption: str | None,
    missing_message: str | None,
) -> str:
    width = int(spec["width"])
    height = int(spec["height"])
    fit = html.escape(str(spec["fit"]))
    caption_html = f"<div class='evidence-caption'>{html.escape(caption)}</div>" if caption else ""
    resolved_path = status.get("resolved_path")
    if resolved_path:
        source = _data_uri(Path(resolved_path))
        if source:
            body = (
                f"<img src='{source}' alt='{html.escape(caption or 'evidence image')}' "
                f"style='width:100%;height:100%;object-fit:{fit};object-position:center center;display:block;'/>"
            )
        else:
            body = _placeholder_html(missing_message or "Image unavailable")
    else:
        body = _placeholder_html(missing_message or status.get("placeholder") or "Image missing")
    return (
        "<div class='evidence-shell' "
        f"style='width:100%;max-width:{width}px;aspect-ratio:{html.escape(str(spec['aspect_ratio']))};"
        f"height:auto;min-height:min({height}px, calc((100vw - 48px) * {height / width:.6f}));"
        "background:#f3f4f6;border:1px solid #d1d5db;border-radius:6px;"
        "overflow:hidden;display:flex;align-items:center;justify-content:center;margin-bottom:0.25rem;'>"
        f"{body}</div>{caption_html}"
    )


def _placeholder_html(message: str) -> str:
    return (
        "<div style='width:100%;height:100%;display:flex;align-items:center;justify-content:center;"
        "color:#6b7280;font-size:0.875rem;text-align:center;padding:0.5rem;'>"
        f"{html.escape(message)}</div>"
    )


def _data_uri(path: Path) -> str | None:
    try:
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None
    mime = _mime_type(path)
    return f"data:{mime};base64,{payload}"


def _mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".bmp":
        return "image/bmp"
    return "application/octet-stream"


def _format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:05.2f}"


def _extract_crop_paths(crops: Any) -> list[str]:
    if not crops:
        return []
    paths: list[str] = []
    for crop in crops:
        if isinstance(crop, dict) and crop.get("vehicle_crop_path"):
            paths.append(str(crop["vehicle_crop_path"]))
    return paths


def _nested_get(value: dict[str, Any], keys: list[str]) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _identity_key(record: dict[str, Any]) -> str:
    return f"{record.get('source_id')}:{int(record.get('track_id') or 0)}:{int(record.get('track_generation') or 0)}"


def _identity_terms(record: dict[str, Any]) -> list[str]:
    return [str(record.get("source_id") or ""), f"track_{int(record.get('track_id') or 0):06d}"]


def _safe_record_id(record: dict[str, Any]) -> str:
    return str(record.get("record_id") or "").replace(":", "_").replace("/", "_").replace("\\", "_")


def _dedupe(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    retained: list[str] = []
    for value in values:
        if value is None or str(value).strip() == "":
            continue
        text = str(value)
        if text not in seen:
            seen.add(text)
            retained.append(text)
    return retained
