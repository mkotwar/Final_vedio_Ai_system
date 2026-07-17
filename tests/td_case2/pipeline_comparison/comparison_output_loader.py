from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from .hybrid_runner_adapter import load_hybrid_outputs
    from .td_case2_runner_adapter import load_td_case2_outputs
except ImportError:  # pragma: no cover
    from hybrid_runner_adapter import load_hybrid_outputs
    from td_case2_runner_adapter import load_td_case2_outputs


def load_pipeline_outputs(td_case2_run_dir: Path, hybrid_run_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        "td_case2": load_td_case2_outputs(td_case2_run_dir),
        "hybrid": load_hybrid_outputs(hybrid_run_dir),
    }
