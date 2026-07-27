from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import urlparse

from fastapi.responses import FileResponse

from ..errors import NotFoundError
from ..settings import ApiSettings
from ...persistence.api_read_repository import AnalyticsReadRepository


ALLOWED_IMAGE_EXTENSIONS = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
LOCAL_STORAGE_PROVIDERS = {"LOCAL"}
SUPPORTED_STORAGE_PROVIDERS = {"LOCAL", "SUPABASE_STORAGE"}
MEDIA_AVAILABILITY = {
    "LOCAL_FILE",
    "SIGNED_URL",
    "REFERENCE_ONLY",
    "MISSING",
    "UNSAFE_REFERENCE",
    "UNSUPPORTED_PROVIDER",
}


@dataclass(frozen=True, slots=True)
class MediaStatus:
    availability: str
    path: Path | None = None
    mime_type: str | None = None


class MediaService:
    def __init__(self, repository: AnalyticsReadRepository, *, settings: ApiSettings) -> None:
        self.repository = repository
        self.settings = settings

    def list_media(self, track_uuid: str) -> list[dict]:
        track = self.repository.get_track_row(track_uuid)
        if track is None:
            raise NotFoundError("TRACK_NOT_FOUND", "Track was not found.")
        return [self.decorate_media_reference(row) for row in self.repository.list_track_media(track_uuid)]

    def get_media_reference(self, media_id: str) -> dict:
        row = self._get_media_row(media_id)
        return self.decorate_media_reference(row)

    def get_media_content_response(self, media_id: str) -> FileResponse:
        row = self._get_media_row(media_id)
        status = self._classify_media(row)
        if status.availability in {"MISSING", "REFERENCE_ONLY", "UNSUPPORTED_PROVIDER"}:
            raise NotFoundError("MEDIA_CONTENT_NOT_FOUND", "Media content is not available.")
        if status.availability == "UNSAFE_REFERENCE" or status.path is None or status.mime_type is None:
            raise NotFoundError("MEDIA_CONTENT_NOT_FOUND", "Media content is not available.")
        return FileResponse(
            path=status.path,
            media_type=status.mime_type,
            filename=f"{row.get('id')}{status.path.suffix.lower()}",
            headers={
                "Cache-Control": "private, max-age=60",
                "X-Content-Type-Options": "nosniff",
            },
        )

    def get_signed_url(self, media_id: str) -> dict:
        row = self._get_media_row(media_id)
        status = self._classify_media(row)
        if status.availability != "SIGNED_URL":
            return {
                "media_id": str(row.get("id")),
                "availability": status.availability,
                "url": None,
                "expires_in": None,
            }
        object_key = self._normalize_storage_uri(row.get("storage_uri"))
        bucket_name = self.settings.media_bucket
        if not object_key or not bucket_name:
            return {
                "media_id": str(row.get("id")),
                "availability": "REFERENCE_ONLY",
                "url": None,
                "expires_in": None,
            }
        try:
            signed = self.repository.client.client.storage.from_(bucket_name).create_signed_url(  # type: ignore[attr-defined]
                object_key,
                self.settings.api_media_url_ttl_seconds,
            )
        except Exception:
            return {
                "media_id": str(row.get("id")),
                "availability": "REFERENCE_ONLY",
                "url": None,
                "expires_in": None,
            }
        signed_url = None
        if isinstance(signed, dict):
            signed_url = signed.get("signedURL") or signed.get("signed_url") or signed.get("url")
        return {
            "media_id": str(row.get("id")),
            "availability": "SIGNED_URL" if signed_url else "REFERENCE_ONLY",
            "url": signed_url,
            "expires_in": self.settings.api_media_url_ttl_seconds if signed_url else None,
        }

    def decorate_media_reference(self, row: dict | None) -> dict | None:
        if not row:
            return None
        status = self._classify_media(row)
        media_id = str(row.get("media_id") or row.get("id") or "")
        item = {
            "media_id": media_id or None,
            "media_type": row.get("media_type"),
            "availability": status.availability,
            "content_url": None,
            "thumbnail_url": None,
            "frame_number": row.get("frame_number"),
            "captured_at": row.get("captured_at"),
            "video_time_seconds": row.get("video_time_seconds"),
            "width": row.get("width"),
            "height": row.get("height"),
            "quality_score": row.get("quality_score"),
            "sharpness_score": row.get("sharpness_score"),
            "visibility_score": row.get("visibility_score"),
            "selection_rank": row.get("selection_rank"),
            "is_primary": row.get("is_primary"),
            "error_detail": None,
        }
        if status.availability == "LOCAL_FILE" and media_id:
            item["content_url"] = f"/api/v1/media/{media_id}/content"
        elif status.availability == "SIGNED_URL":
            item["content_url"] = None
        elif status.availability == "MISSING":
            item["error_detail"] = "Local evidence file not found."
        elif status.availability == "REFERENCE_ONLY":
            item["error_detail"] = "Reference only."
        elif status.availability == "UNSAFE_REFERENCE":
            item["error_detail"] = "Unsafe media reference."
        elif status.availability == "UNSUPPORTED_PROVIDER":
            item["error_detail"] = "Unsupported media provider."
        return item

    def resolve_local_media_path(self, storage_uri: str) -> Path | None:
        status = self._classify_local_reference(storage_uri)
        return status.path if status.availability == "LOCAL_FILE" else None

    def _get_media_row(self, media_id: str) -> dict:
        row = self.repository.get_media_by_id(media_id)
        if row is None:
            raise NotFoundError("MEDIA_NOT_FOUND", "Media was not found.")
        return row

    def _classify_media(self, row: dict) -> MediaStatus:
        provider = str(row.get("storage_provider") or "").strip().upper()
        if provider not in SUPPORTED_STORAGE_PROVIDERS:
            return MediaStatus(availability="UNSUPPORTED_PROVIDER")
        if self.settings.media_mode == "REFERENCE_ONLY":
            return MediaStatus(availability="REFERENCE_ONLY")
        if provider == "SUPABASE_STORAGE":
            if self.settings.media_mode == "LOCAL_FILE":
                return MediaStatus(availability="REFERENCE_ONLY")
            return MediaStatus(availability="SIGNED_URL" if self.settings.media_bucket else "REFERENCE_ONLY")
        if self.settings.media_mode == "SUPABASE_STORAGE":
            return MediaStatus(availability="REFERENCE_ONLY")
        return self._classify_local_reference(row.get("storage_uri"))

    def _classify_local_reference(self, storage_uri: object) -> MediaStatus:
        original = str(storage_uri or "").strip()
        if not original:
            return MediaStatus(availability="REFERENCE_ONLY")
        if "\x00" in original:
            return MediaStatus(availability="UNSAFE_REFERENCE")
        parsed = urlparse(original)
        if parsed.scheme and parsed.scheme.lower() not in {"", "file"}:
            return MediaStatus(availability="UNSAFE_REFERENCE")

        normalized = self._normalize_storage_uri(original)
        if not normalized:
            return MediaStatus(availability="UNSAFE_REFERENCE")

        candidate_strings: list[str] = []
        windows_path = PureWindowsPath(original)
        if windows_path.drive or original.startswith("\\\\") or original.startswith("/"):
            candidate_strings.append(original)
        else:
            candidate_strings.append(normalized)

        saw_safe_candidate = False
        for candidate_value in candidate_strings:
            for resolved in self._iter_candidate_paths(candidate_value):
                saw_safe_candidate = True
                extension = resolved.suffix.lower()
                mime_type = ALLOWED_IMAGE_EXTENSIONS.get(extension) or mimetypes.guess_type(resolved.name)[0]
                if extension not in ALLOWED_IMAGE_EXTENSIONS or mime_type is None:
                    return MediaStatus(availability="UNSAFE_REFERENCE")
                if not resolved.exists():
                    continue
                if not resolved.is_file():
                    return MediaStatus(availability="UNSAFE_REFERENCE")
                return MediaStatus(availability="LOCAL_FILE", path=resolved, mime_type=mime_type)
        if saw_safe_candidate:
            return MediaStatus(availability="MISSING")
        return MediaStatus(availability="UNSAFE_REFERENCE")

    def _normalize_storage_uri(self, storage_uri: str) -> str | None:
        candidate = str(storage_uri or "").strip()
        if not candidate:
            return None
        normalized = candidate.replace("\\", "/")
        if any(part == ".." for part in PurePosixPath(normalized).parts):
            return None
        if normalized.startswith("//"):
            return None
        return normalized

    def _iter_candidate_paths(self, candidate_value: str) -> list[Path]:
        candidate_path = Path(candidate_value)
        if candidate_path.is_absolute():
            resolved_candidate = candidate_path.resolve()
            matches: list[Path] = []
            for root in self.settings.media_allowed_roots:
                try:
                    resolved_candidate.relative_to(root)
                    matches.append(resolved_candidate)
                except ValueError:
                    continue
            return matches

        relative_parts = PurePosixPath(candidate_value).parts
        if not relative_parts:
            return []
        matches: list[Path] = []
        for root in self.settings.media_allowed_roots:
            resolved_root = root.resolve()
            try:
                resolved_candidate = (resolved_root / Path(*relative_parts)).resolve()
                resolved_candidate.relative_to(resolved_root)
            except Exception:
                continue
            matches.append(resolved_candidate)
        return matches
