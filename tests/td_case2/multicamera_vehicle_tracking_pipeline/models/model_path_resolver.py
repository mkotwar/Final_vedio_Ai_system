from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ModelPathResolutionError(ValueError):
    """Raised when a model path cannot be resolved safely."""


@dataclass(frozen=True, slots=True)
class ModelPathResolutionAttempt:
    source: str
    raw_value: str | None


@dataclass(frozen=True, slots=True)
class ModelPathResolution:
    path: Path | None
    source: str | None
    attempts: tuple[ModelPathResolutionAttempt, ...]


def resolve_model_path(
    *,
    cli_value: str | Path | None,
    environment_variable: str | None,
    config_value: str | Path | None,
    default_value: str | Path | None = None,
    project_root: Path,
    required: bool = True,
    expect_directory: bool | None = None,
) -> Path | None:
    resolution = resolve_model_path_with_source(
        cli_value=cli_value,
        environment_variable=environment_variable,
        config_value=config_value,
        default_value=default_value,
        project_root=project_root,
        required=required,
        expect_directory=expect_directory,
    )
    return resolution.path


def resolve_model_path_with_source(
    *,
    cli_value: str | Path | None,
    environment_variable: str | None,
    config_value: str | Path | None,
    default_value: str | Path | None = None,
    project_root: Path,
    required: bool = True,
    expect_directory: bool | None = None,
) -> ModelPathResolution:
    attempts: list[ModelPathResolutionAttempt] = []
    candidates: list[tuple[str, str | Path]] = []
    if cli_value not in (None, ""):
        candidates.append(("cli", cli_value))
    env_value = None
    if environment_variable:
        env_value = os.getenv(environment_variable)
        if env_value not in (None, ""):
            candidates.append((f"env:{environment_variable}", env_value))
    if config_value not in (None, ""):
        candidates.append(("config", config_value))
    if default_value not in (None, ""):
        candidates.append(("default", default_value))
    for source, raw_candidate in candidates:
        attempts.append(ModelPathResolutionAttempt(source=source, raw_value=str(raw_candidate)))
        resolved = _normalize_candidate_path(raw_candidate, project_root=project_root)
        if not resolved.exists():
            continue
        if expect_directory is True and not resolved.is_dir():
            continue
        if expect_directory is False and not resolved.is_file():
            continue
        return ModelPathResolution(path=resolved, source=source, attempts=tuple(attempts))
    if not required:
        return ModelPathResolution(path=None, source=None, attempts=tuple(attempts))
    checked = ", ".join(f"{attempt.source}={attempt.raw_value!r}" for attempt in attempts) or "no sources configured"
    raise ModelPathResolutionError(f"Failed to resolve required model path. Checked: {checked}")


def _normalize_candidate_path(value: str | Path, *, project_root: Path) -> Path:
    expanded = Path(os.path.expandvars(str(value))).expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (project_root / expanded).resolve()
