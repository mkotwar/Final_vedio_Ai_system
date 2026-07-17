from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass
class PipelineCommand:
    label: str
    command: list[str]
    cwd: str
    env_overrides: dict[str, str]
    started_at: str = field(default_factory=now_iso)
    finished_at: str = ""
    exit_code: int | None = None
    stdout_path: str = ""
    stderr_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineArtifacts:
    run_dir: str
    tracking_runtime_seconds: float | None
    post_processing_runtime_seconds: float | None
    total_runtime_seconds: float | None
    tracker_name: str
    output_files: dict[str, str]
    metrics: dict[str, Any]
    command: dict[str, Any]
    config_snapshot: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
