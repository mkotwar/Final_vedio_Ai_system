from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .report_writer import write_json


def save_feature_cache(*, output_path: Path, features_by_frame: dict[int, np.ndarray]) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"frame_{frame_index:06d}": array.astype(np.float32) for frame_index, array in sorted(features_by_frame.items())}
    np.savez_compressed(output_path, **payload)
    metadata = {
        "status": "success",
        "feature_file": str(output_path),
        "frame_count": len(features_by_frame),
        "frames": [
            {
                "processed_frame_index": frame_index,
                "vector_count": int(array.shape[0]) if array.ndim == 2 else 0,
                "feature_dimension": int(array.shape[1]) if array.ndim == 2 and array.shape[0] > 0 else 0,
            }
            for frame_index, array in sorted(features_by_frame.items())
        ],
    }
    return metadata


def write_feature_cache_metadata(*, metadata_path: Path, payload: dict[str, Any]) -> None:
    write_json(metadata_path, payload)


def load_feature_cache(cache_path: Path) -> dict[int, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as cache:
        output: dict[int, np.ndarray] = {}
        for key in cache.files:
            if not key.startswith("frame_"):
                continue
            output[int(key.split("_", 1)[1])] = np.asarray(cache[key], dtype=np.float32)
        return output
