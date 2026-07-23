from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath


class MediaResolutionError(ValueError):
    """Raised when a stored media URI cannot be resolved safely."""


def resolve_local_media_path(
    *,
    storage_uri: str,
    artifact_root: Path,
) -> Path:
    candidate = str(storage_uri).strip().replace("\\", "/")
    if not candidate:
        raise MediaResolutionError("storage_uri must not be empty.")
    if PureWindowsPath(candidate).drive or candidate.startswith("/") or candidate.startswith("//"):
        raise MediaResolutionError("storage_uri must be relative.")
    parts = PurePosixPath(candidate).parts
    if any(part == ".." for part in parts):
        raise MediaResolutionError("storage_uri must not contain path traversal.")
    resolved_root = artifact_root.resolve()
    resolved_path = (resolved_root / Path(*parts)).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise MediaResolutionError("Resolved media path escaped artifact root.") from exc
    if not resolved_path.exists():
        raise FileNotFoundError(f"Resolved media file does not exist: {resolved_path}")
    return resolved_path
