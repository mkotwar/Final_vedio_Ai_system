from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..enrichment.florence_config import FlorenceConfig
from .florence_runtime import FlorenceRuntime
from .model_path_resolver import resolve_model_path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_FLORENCE_MODEL_PATH = PROJECT_ROOT / "models" / "florence" / "Florence-2-base-ft"
DEFAULT_FLORENCE_PROCESSOR_PATH = DEFAULT_FLORENCE_MODEL_PATH
DEFAULT_FLORENCE_ADAPTER_PATH = PROJECT_ROOT / "models" / "florence_adapters" / "adaptor_florance_baseFT"


@dataclass(slots=True)
class FlorenceRuntimeFactory:
    project_root: Path
    runtime: FlorenceRuntime | None = None

    def get_runtime(
        self,
        *,
        config: FlorenceConfig,
        model_path_cli: str | Path | None = None,
        adapter_path_cli: str | Path | None = None,
        processor_path_cli: str | Path | None = None,
        device_override: str | None = None,
    ) -> FlorenceRuntime | None:
        if not config.enabled:
            return None
        if self.runtime is not None and config.load_once:
            return self.runtime
        model_path = resolve_model_path(
            cli_value=model_path_cli,
            environment_variable=config.model_path_env,
            config_value=config.model_path,
            default_value=DEFAULT_FLORENCE_MODEL_PATH,
            project_root=self.project_root,
            required=True,
            expect_directory=True,
        )
        adapter_path = resolve_model_path(
            cli_value=adapter_path_cli,
            environment_variable=config.adapter_path_env,
            config_value=config.adapter_path,
            default_value=DEFAULT_FLORENCE_ADAPTER_PATH,
            project_root=self.project_root,
            required=False,
            expect_directory=True,
        )
        processor_path = resolve_model_path(
            cli_value=processor_path_cli,
            environment_variable=config.processor_path_env,
            config_value=config.processor_path,
            default_value=DEFAULT_FLORENCE_PROCESSOR_PATH,
            project_root=self.project_root,
            required=False,
            expect_directory=True,
        )
        resolved_device = self._resolve_device(device_override or config.device)
        runtime = FlorenceRuntime(
            model_path=model_path,
            adapter_path=adapter_path,
            processor_path=processor_path,
            device=resolved_device,
            dtype=config.dtype,
            local_files_only=config.local_files_only,
            trust_remote_code=config.trust_remote_code,
        )
        runtime.load()
        if config.load_once:
            self.runtime = runtime
        return runtime

    @staticmethod
    def _resolve_device(device: str) -> str:
        normalized = str(device).strip().lower()
        if normalized != "auto":
            return normalized
        try:
            import torch  # type: ignore
        except Exception:
            return "cpu"
        return "cuda" if bool(torch.cuda.is_available()) else "cpu"
