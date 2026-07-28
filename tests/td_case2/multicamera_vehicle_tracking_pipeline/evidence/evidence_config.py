from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


class EvidenceConfigError(ValueError):
    """Raised when evidence configuration is invalid."""


@dataclass(frozen=True, slots=True)
class EvidenceConfig:
    enabled: bool = False
    output_root: str = "artifacts"
    collect_first: bool = True
    collect_middle: bool = True
    collect_last: bool = True
    collect_highest_confidence: bool = True
    collect_largest: bool = True
    collect_sharpest: bool = True
    collect_best_overall: bool = True
    max_candidates_per_track: int = 7
    minimum_crop_width: int = 40
    minimum_crop_height: int = 40
    minimum_detection_confidence: float = 0.20
    bbox_padding_ratio: float = 0.05
    padding_ratio_x: float = 0.08
    padding_ratio_y: float = 0.08
    minimum_padding_pixels: int = 8
    clip_to_frame: bool = True
    reject_out_of_frame_bbox: bool = True
    clamp_bbox_to_frame: bool = True
    sharpness_enabled: bool = True
    edge_penalty_enabled: bool = True
    annotate_all_observations: bool = False
    visibility_weight: float = 0.30
    detection_confidence_weight: float = 0.20
    sharpness_weight: float = 0.15
    centeredness_weight: float = 0.15
    bbox_area_weight: float = 0.10
    edge_penalty_weight: float = 0.10
    jpeg_quality: int = 90
    save_candidate_crops: bool = False
    save_final_selected_crops: bool = True

    def __post_init__(self) -> None:
        if not str(self.output_root).strip():
            raise EvidenceConfigError("output_root must not be empty.")
        if int(self.max_candidates_per_track) <= 0:
            raise EvidenceConfigError("max_candidates_per_track must be positive.")
        if int(self.minimum_crop_width) <= 0 or int(self.minimum_crop_height) <= 0:
            raise EvidenceConfigError("minimum crop dimensions must be positive.")
        if int(self.minimum_padding_pixels) < 0:
            raise EvidenceConfigError("minimum_padding_pixels must be non-negative.")
        if not 0.0 <= float(self.minimum_detection_confidence) <= 1.0:
            raise EvidenceConfigError("minimum_detection_confidence must be between 0 and 1.")
        if float(self.bbox_padding_ratio) < 0.0:
            raise EvidenceConfigError("bbox_padding_ratio must be non-negative.")
        if float(self.padding_ratio_x) < 0.0 or float(self.padding_ratio_y) < 0.0:
            raise EvidenceConfigError("padding ratios must be non-negative.")
        for field_name in (
            "visibility_weight",
            "detection_confidence_weight",
            "sharpness_weight",
            "centeredness_weight",
            "bbox_area_weight",
            "edge_penalty_weight",
        ):
            value = float(getattr(self, field_name))
            if value < 0.0:
                raise EvidenceConfigError(f"{field_name} must be non-negative.")
        if not 1 <= int(self.jpeg_quality) <= 100:
            raise EvidenceConfigError("jpeg_quality must be between 1 and 100.")


def _parse_scalar(value: str) -> Any:
    stripped = value.strip()
    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if stripped.startswith('"') and stripped.endswith('"'):
        return stripped[1:-1]
    try:
        if "." in stripped:
            return float(stripped)
        return int(stripped)
    except ValueError:
        return stripped


def _load_yaml_text(text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None
    if yaml is not None:
        payload = yaml.safe_load(text) or {}
        if not isinstance(payload, dict):
            raise EvidenceConfigError("Evidence config root must be a mapping.")
        return payload
    payload: dict[str, Any] = {"evidence": {}}
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or stripped == "evidence:":
            continue
        key, _, value = stripped.partition(":")
        if not _:
            raise EvidenceConfigError("Invalid evidence.yaml structure.")
        payload["evidence"][key.strip()] = _parse_scalar(value)
    return payload


def load_evidence_config(config_path: str | Path, *, overrides: dict[str, Any] | None = None) -> EvidenceConfig:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise EvidenceConfigError(f"Evidence config file does not exist: {path}")
    payload = _load_yaml_text(path.read_text(encoding="utf-8"))
    raw = payload.get("evidence")
    if not isinstance(raw, dict):
        raise EvidenceConfigError("Evidence config must contain an 'evidence' mapping.")
    config = EvidenceConfig(
        enabled=bool(raw.get("enabled", False)),
        output_root=str(raw.get("output_root", "artifacts")),
        collect_first=bool(raw.get("collect_first", True)),
        collect_middle=bool(raw.get("collect_middle", True)),
        collect_last=bool(raw.get("collect_last", True)),
        collect_highest_confidence=bool(raw.get("collect_highest_confidence", True)),
        collect_largest=bool(raw.get("collect_largest", True)),
        collect_sharpest=bool(raw.get("collect_sharpest", True)),
        collect_best_overall=bool(raw.get("collect_best_overall", True)),
        max_candidates_per_track=int(raw.get("max_candidates_per_track", 7)),
        minimum_crop_width=int(raw.get("minimum_crop_width", 40)),
        minimum_crop_height=int(raw.get("minimum_crop_height", 40)),
        minimum_detection_confidence=float(raw.get("minimum_detection_confidence", 0.20)),
        bbox_padding_ratio=float(raw.get("bbox_padding_ratio", 0.05)),
        padding_ratio_x=float(raw.get("padding_ratio_x", raw.get("bbox_padding_ratio", 0.08))),
        padding_ratio_y=float(raw.get("padding_ratio_y", raw.get("bbox_padding_ratio", 0.08))),
        minimum_padding_pixels=int(raw.get("minimum_padding_pixels", 8)),
        clip_to_frame=bool(raw.get("clip_to_frame", True)),
        reject_out_of_frame_bbox=bool(raw.get("reject_out_of_frame_bbox", True)),
        clamp_bbox_to_frame=bool(raw.get("clamp_bbox_to_frame", True)),
        sharpness_enabled=bool(raw.get("sharpness_enabled", True)),
        edge_penalty_enabled=bool(raw.get("edge_penalty_enabled", True)),
        annotate_all_observations=bool(raw.get("annotate_all_observations", False)),
        visibility_weight=float(raw.get("visibility_weight", 0.30)),
        detection_confidence_weight=float(raw.get("detection_confidence_weight", 0.20)),
        sharpness_weight=float(raw.get("sharpness_weight", 0.15)),
        centeredness_weight=float(raw.get("centeredness_weight", 0.15)),
        bbox_area_weight=float(raw.get("bbox_area_weight", 0.10)),
        edge_penalty_weight=float(raw.get("edge_penalty_weight", 0.10)),
        jpeg_quality=int(raw.get("jpeg_quality", 90)),
        save_candidate_crops=bool(raw.get("save_candidate_crops", False)),
        save_final_selected_crops=bool(raw.get("save_final_selected_crops", True)),
    )
    if overrides:
        config = replace(
            config,
            enabled=bool(overrides.get("enabled", config.enabled)),
            output_root=str(overrides.get("output_root", config.output_root)),
            collect_first=bool(overrides.get("collect_first", config.collect_first)),
            collect_middle=bool(overrides.get("collect_middle", config.collect_middle)),
            collect_last=bool(overrides.get("collect_last", config.collect_last)),
            collect_highest_confidence=bool(overrides.get("collect_highest_confidence", config.collect_highest_confidence)),
            collect_largest=bool(overrides.get("collect_largest", config.collect_largest)),
            collect_sharpest=bool(overrides.get("collect_sharpest", config.collect_sharpest)),
            collect_best_overall=bool(overrides.get("collect_best_overall", config.collect_best_overall)),
            max_candidates_per_track=int(overrides.get("max_candidates_per_track", config.max_candidates_per_track)),
            minimum_crop_width=int(overrides.get("minimum_crop_width", config.minimum_crop_width)),
            minimum_crop_height=int(overrides.get("minimum_crop_height", config.minimum_crop_height)),
            minimum_detection_confidence=float(overrides.get("minimum_detection_confidence", config.minimum_detection_confidence)),
            bbox_padding_ratio=float(overrides.get("bbox_padding_ratio", config.bbox_padding_ratio)),
            padding_ratio_x=float(overrides.get("padding_ratio_x", config.padding_ratio_x)),
            padding_ratio_y=float(overrides.get("padding_ratio_y", config.padding_ratio_y)),
            minimum_padding_pixels=int(overrides.get("minimum_padding_pixels", config.minimum_padding_pixels)),
            clip_to_frame=bool(overrides.get("clip_to_frame", config.clip_to_frame)),
            reject_out_of_frame_bbox=bool(overrides.get("reject_out_of_frame_bbox", config.reject_out_of_frame_bbox)),
            clamp_bbox_to_frame=bool(overrides.get("clamp_bbox_to_frame", config.clamp_bbox_to_frame)),
            sharpness_enabled=bool(overrides.get("sharpness_enabled", config.sharpness_enabled)),
            edge_penalty_enabled=bool(overrides.get("edge_penalty_enabled", config.edge_penalty_enabled)),
            annotate_all_observations=bool(overrides.get("annotate_all_observations", config.annotate_all_observations)),
            visibility_weight=float(overrides.get("visibility_weight", config.visibility_weight)),
            detection_confidence_weight=float(overrides.get("detection_confidence_weight", config.detection_confidence_weight)),
            sharpness_weight=float(overrides.get("sharpness_weight", config.sharpness_weight)),
            centeredness_weight=float(overrides.get("centeredness_weight", config.centeredness_weight)),
            bbox_area_weight=float(overrides.get("bbox_area_weight", config.bbox_area_weight)),
            edge_penalty_weight=float(overrides.get("edge_penalty_weight", config.edge_penalty_weight)),
            jpeg_quality=int(overrides.get("jpeg_quality", config.jpeg_quality)),
            save_candidate_crops=bool(overrides.get("save_candidate_crops", config.save_candidate_crops)),
            save_final_selected_crops=bool(overrides.get("save_final_selected_crops", config.save_final_selected_crops)),
        )
    return config
