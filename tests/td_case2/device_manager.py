from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from stage_checks import read_json, write_json


ENV_DEVICE = "TD_CASE2_DEVICE"
ENV_DEVICE_INDEX = "TD_CASE2_DEVICE_INDEX"
SUPPORTED_OVERRIDE_VALUES = {"auto", "cpu", "cuda"}


@dataclass(frozen=True)
class RuntimeInfo:
    torch_available: bool
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str | None
    cuda_total_vram_mb: float | None
    cuda_bf16_supported: bool
    opencv_cuda_available: bool
    opencv_cuda_device_count: int
    onnxruntime_cuda_available: bool
    tensorrt_available: bool


@dataclass(frozen=True)
class DeviceDecision:
    component_name: str
    requested: str
    selected: str
    selected_index: int | None
    reason: str
    override_source: str | None
    cuda_available: bool

    @property
    def torch_device(self) -> str:
        if self.selected == "cuda":
            return f"cuda:{self.selected_index or 0}"
        return "cpu"

    @property
    def ultralytics_device(self) -> str:
        if self.selected == "cuda":
            return str(self.selected_index or 0)
        return "cpu"


def _optional_import(module_name: str) -> Any | None:
    try:
        return __import__(module_name)
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_runtime_info() -> RuntimeInfo:
    torch = _optional_import("torch")
    cv2 = _optional_import("cv2")
    onnxruntime = _optional_import("onnxruntime")
    tensorrt = _optional_import("tensorrt")

    torch_available = torch is not None
    cuda_available = bool(torch_available and torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
    cuda_device_name = str(torch.cuda.get_device_name(0)) if cuda_available else None
    cuda_total_vram_mb = None
    if cuda_available:
        try:
            cuda_total_vram_mb = round(float(torch.cuda.get_device_properties(0).total_memory) / (1024**2), 2)
        except Exception:
            cuda_total_vram_mb = None

    opencv_cuda_device_count = 0
    if cv2 is not None and hasattr(cv2, "cuda"):
        try:
            opencv_cuda_device_count = int(cv2.cuda.getCudaEnabledDeviceCount())
        except Exception:
            opencv_cuda_device_count = 0

    onnxruntime_cuda_available = False
    if onnxruntime is not None:
        try:
            providers = set(str(item) for item in onnxruntime.get_available_providers())
            onnxruntime_cuda_available = "CUDAExecutionProvider" in providers
        except Exception:
            onnxruntime_cuda_available = False

    return RuntimeInfo(
        torch_available=torch_available,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_device_name=cuda_device_name,
        cuda_total_vram_mb=cuda_total_vram_mb,
        cuda_bf16_supported=bool(cuda_available and getattr(torch.cuda, "is_bf16_supported", lambda: False)()),
        opencv_cuda_available=opencv_cuda_device_count > 0,
        opencv_cuda_device_count=opencv_cuda_device_count,
        onnxruntime_cuda_available=onnxruntime_cuda_available,
        tensorrt_available=tensorrt is not None,
    )


def _normalize_override(raw_value: str | None) -> str | None:
    if raw_value is None:
        return None
    normalized = raw_value.strip().lower()
    if not normalized:
        return None
    if normalized.startswith("cuda"):
        return "cuda"
    if normalized not in SUPPORTED_OVERRIDE_VALUES:
        raise ValueError(
            f"Unsupported device override value: {raw_value!r}. Supported values: {sorted(SUPPORTED_OVERRIDE_VALUES)}"
        )
    return normalized


def _read_override(*env_names: str) -> tuple[str | None, str | None]:
    import os

    for env_name in env_names:
        raw_value = os.environ.get(env_name)
        normalized = _normalize_override(raw_value)
        if normalized is not None:
            return normalized, env_name
    return None, None


def _read_device_index() -> int:
    import os

    raw_value = os.environ.get(ENV_DEVICE_INDEX, "").strip()
    if not raw_value:
        return 0
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {ENV_DEVICE_INDEX} must be an integer. Received: {raw_value!r}") from exc
    if value < 0:
        raise ValueError(f"Environment variable {ENV_DEVICE_INDEX} must be >= 0. Received: {value}")
    return value


def resolve_device(*, component_name: str, override_env_names: tuple[str, ...] = (), require_cuda: bool = False) -> DeviceDecision:
    runtime = get_runtime_info()
    requested, override_source = _read_override(*override_env_names, ENV_DEVICE)
    requested = requested or "auto"
    selected_index = _read_device_index()

    if requested == "cpu":
        if require_cuda:
            raise RuntimeError(f"{component_name} cannot honor a CPU override because this runtime path requires CUDA.")
        return DeviceDecision(
            component_name=component_name,
            requested=requested,
            selected="cpu",
            selected_index=None,
            reason=f"Using CPU because {override_source} explicitly requested it.",
            override_source=override_source,
            cuda_available=runtime.cuda_available,
        )

    if requested == "cuda":
        if not runtime.cuda_available:
            if require_cuda:
                raise RuntimeError(f"{component_name} requires CUDA, but torch.cuda.is_available() is false on this machine.")
            return DeviceDecision(
                component_name=component_name,
                requested=requested,
                selected="cpu",
                selected_index=None,
                reason=f"Falling back to CPU because {override_source} requested CUDA but CUDA is unavailable.",
                override_source=override_source,
                cuda_available=False,
            )
        return DeviceDecision(
            component_name=component_name,
            requested=requested,
            selected="cuda",
            selected_index=selected_index,
            reason=f"Using CUDA because {override_source} requested it and CUDA is available.",
            override_source=override_source,
            cuda_available=True,
        )

    if runtime.cuda_available:
        return DeviceDecision(
            component_name=component_name,
            requested="auto",
            selected="cuda",
            selected_index=selected_index,
            reason="Using CUDA automatically because torch.cuda.is_available() returned true.",
            override_source=override_source,
            cuda_available=True,
        )

    if require_cuda:
        raise RuntimeError(f"{component_name} requires CUDA, but torch.cuda.is_available() is false on this machine.")

    return DeviceDecision(
        component_name=component_name,
        requested="auto",
        selected="cpu",
        selected_index=None,
        reason="Using CPU because CUDA is unavailable.",
        override_source=override_source,
        cuda_available=False,
    )


def cuda_memory_allocated_mb() -> float | None:
    torch = _optional_import("torch")
    if torch is None or not torch.cuda.is_available():
        return None
    return round(float(torch.cuda.memory_allocated()) / (1024**2), 2)


def cuda_memory_reserved_mb() -> float | None:
    torch = _optional_import("torch")
    if torch is None or not torch.cuda.is_available():
        return None
    return round(float(torch.cuda.memory_reserved()) / (1024**2), 2)


def record_stage_device(
    *,
    run_dir: Path,
    stage_name: str,
    component_name: str,
    supports_gpu: bool,
    selected_device: str,
    reason: str,
    actual_device: str | None = None,
    model_device: str | None = None,
    input_device: str | None = None,
    output_device: str | None = None,
    preprocessing_device: str | None = None,
    inference_device: str | None = None,
    postprocess_device: str | None = None,
    vram_allocated_mb: float | None = None,
    vram_reserved_mb: float | None = None,
    notes: list[str] | None = None,
) -> Path:
    report_path = run_dir / "gpu_utilization_report.json"
    if report_path.exists():
        payload = read_json(report_path)
    else:
        payload = {
            "runtime": asdict(get_runtime_info()),
            "stages": {},
        }

    stages = payload.setdefault("stages", {})
    if not isinstance(stages, dict):
        stages = {}
        payload["stages"] = stages

    stage_payload = stages.setdefault(stage_name, {"components": []})
    components = stage_payload.setdefault("components", [])
    if not isinstance(components, list):
        components = []
        stage_payload["components"] = components

    entry = {
        "component_name": component_name,
        "supports_gpu": supports_gpu,
        "selected_device": selected_device,
        "actual_device": actual_device or selected_device,
        "model_device": model_device,
        "input_device": input_device,
        "output_device": output_device,
        "preprocessing_device": preprocessing_device,
        "inference_device": inference_device,
        "postprocess_device": postprocess_device,
        "vram_allocated_mb": vram_allocated_mb,
        "vram_reserved_mb": vram_reserved_mb,
        "reason": reason,
        "notes": notes or [],
    }

    existing_index = next(
        (index for index, item in enumerate(components) if isinstance(item, dict) and item.get("component_name") == component_name),
        None,
    )
    if existing_index is None:
        components.append(entry)
    else:
        components[existing_index] = entry

    return write_json(report_path, payload)
