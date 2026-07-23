from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


class FlorenceConfigError(ValueError):
    """Raised when Florence configuration is invalid."""


@dataclass(frozen=True, slots=True)
class FlorenceConfig:
    enabled: bool = False
    model_path: str | None = None
    adapter_path: str | None = None
    processor_path: str | None = None
    model_path_env: str = "FLORENCE_MODEL_PATH"
    adapter_path_env: str = "FLORENCE_ADAPTER_PATH"
    processor_path_env: str = "FLORENCE_PROCESSOR_PATH"
    device: str = "auto"
    dtype: str = "auto"
    trust_remote_code: bool = True
    local_files_only: bool = True
    load_once: bool = True
    allow_cpu_fallback: bool = True
    task_timeout_seconds: int = 60
    max_retries: int = 1
    colour_prompt: str = "Identify the primary exterior colour of the vehicle. Return only one canonical colour label."

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise FlorenceConfigError(f"Unsupported Florence device: {self.device}")
        if self.dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise FlorenceConfigError(f"Unsupported Florence dtype: {self.dtype}")
        if int(self.task_timeout_seconds) <= 0:
            raise FlorenceConfigError("task_timeout_seconds must be positive.")
        if int(self.max_retries) < 0:
            raise FlorenceConfigError("max_retries must be non-negative.")


def load_florence_config(config_path: str | Path, *, overrides: dict[str, Any] | None = None) -> FlorenceConfig:
    payload = _load_yaml_mapping(config_path, root_key="florence")
    config = FlorenceConfig(
        enabled=bool(payload.get("enabled", False)),
        model_path=_optional_str(payload.get("model_path")),
        adapter_path=_optional_str(payload.get("adapter_path")),
        processor_path=_optional_str(payload.get("processor_path")),
        model_path_env=str(payload.get("model_path_env", "FLORENCE_MODEL_PATH")),
        adapter_path_env=str(payload.get("adapter_path_env", "FLORENCE_ADAPTER_PATH")),
        processor_path_env=str(payload.get("processor_path_env", "FLORENCE_PROCESSOR_PATH")),
        device=str(payload.get("device", "auto")),
        dtype=str(payload.get("dtype", "auto")),
        trust_remote_code=bool(payload.get("trust_remote_code", True)),
        local_files_only=bool(payload.get("local_files_only", True)),
        load_once=bool(payload.get("load_once", True)),
        allow_cpu_fallback=bool(payload.get("allow_cpu_fallback", True)),
        task_timeout_seconds=int(payload.get("task_timeout_seconds", 60)),
        max_retries=int(payload.get("max_retries", 1)),
        colour_prompt=str(payload.get("colour_prompt", "Identify the primary exterior colour of the vehicle. Return only one canonical colour label.")),
    )
    if overrides:
        config = replace(
            config,
            enabled=bool(overrides.get("enabled", config.enabled)),
            model_path=_optional_str(overrides.get("model_path", config.model_path)),
            adapter_path=_optional_str(overrides.get("adapter_path", config.adapter_path)),
            processor_path=_optional_str(overrides.get("processor_path", config.processor_path)),
            device=str(overrides.get("device", config.device)),
            dtype=str(overrides.get("dtype", config.dtype)),
            trust_remote_code=bool(overrides.get("trust_remote_code", config.trust_remote_code)),
            local_files_only=bool(overrides.get("local_files_only", config.local_files_only)),
            allow_cpu_fallback=bool(overrides.get("allow_cpu_fallback", config.allow_cpu_fallback)),
            task_timeout_seconds=int(overrides.get("task_timeout_seconds", config.task_timeout_seconds)),
            max_retries=int(overrides.get("max_retries", config.max_retries)),
            colour_prompt=str(overrides.get("colour_prompt", config.colour_prompt)),
        )
    return config


def _load_yaml_mapping(config_path: str | Path, *, root_key: str) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FlorenceConfigError(f"Florence config file does not exist: {path}")
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        payload = yaml.safe_load(text) or {}
        if not isinstance(payload, dict):
            raise FlorenceConfigError("Florence config root must be a mapping.")
        section = payload.get(root_key, {})
        if not isinstance(section, dict):
            raise FlorenceConfigError(f"Florence config must contain a '{root_key}' mapping.")
        return section
    raise FlorenceConfigError("PyYAML is required to load Florence config.")


def _optional_str(value: object) -> str | None:
    if value in (None, "", "null"):
        return None
    return str(value)
