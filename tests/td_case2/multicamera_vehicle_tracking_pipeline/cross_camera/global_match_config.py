from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


SUPPORTED_TIME_MODES = ("recording_timestamp", "relative_video_time", "disabled")


class GlobalMatchConfigError(ValueError):
    """Raised when global matching configuration is invalid."""


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
            raise GlobalMatchConfigError("Global match config root must be a mapping.")
        return payload
    payload: dict[str, Any] = {}
    current_root: str | None = None
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.endswith(":") and not stripped.startswith("- "):
            current_root = stripped[:-1]
            payload.setdefault(current_root, {})
            continue
        key, _, value = stripped.partition(":")
        if not _ or current_root is None:
            raise GlobalMatchConfigError("Invalid global_matching.yaml structure.")
        payload[current_root][key.strip()] = _parse_scalar(value)
    return payload


@dataclass(frozen=True, slots=True)
class MatchingWeights:
    verified_plate: float = 0.70
    time: float = 0.10
    camera_route: float = 0.10
    vehicle_class: float = 0.05
    vehicle_colour: float = 0.05
    visual_similarity: float = 0.00

    def __post_init__(self) -> None:
        total = 0.0
        for field_name in ("verified_plate", "time", "camera_route", "vehicle_class", "vehicle_colour", "visual_similarity"):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise GlobalMatchConfigError(f"{field_name} weight must be between 0 and 1.")
            total += value
        if round(total, 6) > 1.000001:
            raise GlobalMatchConfigError("Matching weights must sum to at most 1.0.")


@dataclass(frozen=True, slots=True)
class MatchingThresholds:
    confirmed: float = 0.85
    possible: float = 0.55

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.possible) <= 1.0:
            raise GlobalMatchConfigError("possible threshold must be between 0 and 1.")
        if not 0.0 <= float(self.confirmed) <= 1.0:
            raise GlobalMatchConfigError("confirmed threshold must be between 0 and 1.")
        if float(self.confirmed) < float(self.possible):
            raise GlobalMatchConfigError("confirmed threshold must be greater than or equal to possible threshold.")


@dataclass(frozen=True, slots=True)
class TimeMatchingConfig:
    mode: str = "disabled"

    def __post_init__(self) -> None:
        normalized = str(self.mode).strip().lower()
        if normalized not in SUPPORTED_TIME_MODES:
            raise GlobalMatchConfigError(f"Unsupported time matching mode: {self.mode}")
        object.__setattr__(self, "mode", normalized)


@dataclass(frozen=True, slots=True)
class CameraRouteRule:
    allowed: bool = True
    minimum_travel_seconds: float = 0.0
    maximum_travel_seconds: float | None = 300.0

    def __post_init__(self) -> None:
        if float(self.minimum_travel_seconds) < 0:
            raise GlobalMatchConfigError("minimum_travel_seconds must be non-negative.")
        if self.maximum_travel_seconds is not None and float(self.maximum_travel_seconds) < float(self.minimum_travel_seconds):
            raise GlobalMatchConfigError("maximum_travel_seconds must be greater than or equal to minimum_travel_seconds.")


@dataclass(frozen=True, slots=True)
class GlobalMatchConfig:
    rule_version: str = "global_match_v1"
    weights: MatchingWeights = field(default_factory=MatchingWeights)
    thresholds: MatchingThresholds = field(default_factory=MatchingThresholds)
    time_matching: TimeMatchingConfig = field(default_factory=TimeMatchingConfig)
    same_camera_matching: bool = False
    create_single_track_global_objects: bool = True
    camera_routes: dict[str, dict[str, CameraRouteRule]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.rule_version).strip():
            raise GlobalMatchConfigError("rule_version must not be empty.")

    def route_for(self, source_camera_code: str, destination_camera_code: str) -> CameraRouteRule | None:
        forward = self.camera_routes.get(source_camera_code, {})
        if destination_camera_code in forward:
            return forward[destination_camera_code]
        reverse = self.camera_routes.get(destination_camera_code, {})
        if source_camera_code in reverse:
            return reverse[source_camera_code]
        return None


def load_global_match_config(config_path: str | Path, *, overrides: dict[str, Any] | None = None) -> GlobalMatchConfig:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise GlobalMatchConfigError(f"Global matching config file does not exist: {path}")
    payload = _load_yaml_text(path.read_text(encoding="utf-8"))
    matching = payload.get("matching") if isinstance(payload.get("matching"), dict) else {}
    thresholds = matching.get("thresholds") if isinstance(matching.get("thresholds"), dict) else {}
    weights = matching.get("weights") if isinstance(matching.get("weights"), dict) else {}
    time_matching = payload.get("time_matching") if isinstance(payload.get("time_matching"), dict) else {}
    routes_payload = payload.get("camera_routes") if isinstance(payload.get("camera_routes"), dict) else {}
    routes: dict[str, dict[str, CameraRouteRule]] = {}
    for source_camera_code, destination_map in routes_payload.items():
        if not isinstance(destination_map, dict):
            raise GlobalMatchConfigError("camera_routes entries must be mappings.")
        routes[str(source_camera_code)] = {}
        for destination_camera_code, rule_payload in destination_map.items():
            if not isinstance(rule_payload, dict):
                raise GlobalMatchConfigError("camera route rule entries must be mappings.")
            routes[str(source_camera_code)][str(destination_camera_code)] = CameraRouteRule(
                allowed=bool(rule_payload.get("allowed", True)),
                minimum_travel_seconds=float(rule_payload.get("minimum_travel_seconds", 0.0)),
                maximum_travel_seconds=None if rule_payload.get("maximum_travel_seconds") is None else float(rule_payload.get("maximum_travel_seconds")),
            )
    config = GlobalMatchConfig(
        rule_version=str(matching.get("rule_version", "global_match_v1")),
        weights=MatchingWeights(
            verified_plate=float(weights.get("verified_plate", 0.70)),
            time=float(weights.get("time", 0.10)),
            camera_route=float(weights.get("camera_route", 0.10)),
            vehicle_class=float(weights.get("vehicle_class", 0.05)),
            vehicle_colour=float(weights.get("vehicle_colour", 0.05)),
            visual_similarity=float(weights.get("visual_similarity", 0.00)),
        ),
        thresholds=MatchingThresholds(
            confirmed=float(thresholds.get("confirmed", 0.85)),
            possible=float(thresholds.get("possible", 0.55)),
        ),
        time_matching=TimeMatchingConfig(mode=str(time_matching.get("mode", "disabled"))),
        same_camera_matching=bool(payload.get("same_camera_matching", False)),
        create_single_track_global_objects=bool(payload.get("create_single_track_global_objects", True)),
        camera_routes=routes,
    )
    if overrides:
        config = replace(
            config,
            rule_version=str(overrides.get("rule_version", config.rule_version)),
            same_camera_matching=bool(overrides.get("same_camera_matching", config.same_camera_matching)),
            create_single_track_global_objects=bool(overrides.get("create_single_track_global_objects", config.create_single_track_global_objects)),
        )
    return config
