from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..enrichment.anpr_config import AnprConfig
from .model_path_resolver import resolve_model_path
from .plate_detector_runtime import PlateDetectorRuntime


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PLATE_DETECTOR_MODEL_PATH = PROJECT_ROOT / "models" / "plate_detection" / "license_plate_weights.pt"


@dataclass(slots=True)
class PlateDetectorRuntimeFactory:
    project_root: Path
    runtime: PlateDetectorRuntime | None = None

    def get_runtime(
        self,
        *,
        config: AnprConfig,
        model_path_cli: str | Path | None = None,
        device_override: str | None = None,
    ) -> PlateDetectorRuntime | None:
        if not config.enabled:
            return None
        if self.runtime is not None:
            return self.runtime
        model_path = resolve_model_path(
            cli_value=model_path_cli,
            environment_variable=config.plate_detector.model_path_env,
            config_value=config.plate_detector.model_path,
            default_value=DEFAULT_PLATE_DETECTOR_MODEL_PATH,
            project_root=self.project_root,
            required=True,
            expect_directory=False,
        )
        runtime = PlateDetectorRuntime(
            model_path=model_path,
            device=device_override or config.plate_detector.device,
            confidence_threshold=config.plate_detector.confidence_threshold,
            iou_threshold=config.plate_detector.iou_threshold,
            inference_image_size=config.plate_detector.inference_image_size,
            maximum_detections_per_vehicle_crop=config.plate_detector.maximum_detections_per_vehicle_crop,
        )
        runtime.load()
        self.runtime = runtime
        return runtime
