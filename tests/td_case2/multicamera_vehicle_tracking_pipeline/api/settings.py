from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PIPELINE_ENV_FILE = Path(__file__).resolve().parents[1] / ".env.example"
REPO_ROOT = Path(__file__).resolve().parents[4]
CORS_DEFAULT_ORIGINS = "http://127.0.0.1:5173,http://localhost:5173"
MEDIA_DEFAULT_ALLOWED_ROOTS = "artifacts,debug_runs/multicamera_vehicle_tracking_pipeline,debug_runs"
MEDIA_MODES = {"AUTO", "LOCAL_FILE", "SUPABASE_STORAGE", "REFERENCE_ONLY"}


def _parse_list_setting(raw_value: str) -> list[str]:
    candidate = str(raw_value or "").strip()
    if not candidate:
        return []
    if candidate.startswith("["):
        parsed = json.loads(candidate)
        if not isinstance(parsed, list):
            raise ValueError("Expected a JSON array.")
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in candidate.split(",") if item.strip()]


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(PIPELINE_ENV_FILE), env_file_encoding="utf-8", extra="ignore")

    supabase_url: str = Field(
        default="",
        alias="SUPABASE_URL",
        validation_alias=AliasChoices("SUPABASE_URL", "supabase_database_url", "SUPABASE_DATABASE_URL"),
    )
    supabase_service_role_key: str = Field(default="", alias="SUPABASE_SERVICE_ROLE_KEY")
    api_host: str = Field(default="127.0.0.1", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_cors_origins: str = Field(default=CORS_DEFAULT_ORIGINS, alias="API_CORS_ORIGINS")
    api_page_size_default: int = Field(default=25, alias="API_PAGE_SIZE_DEFAULT")
    api_page_size_max: int = Field(default=100, alias="API_PAGE_SIZE_MAX")
    api_log_level: str = Field(default="INFO", alias="API_LOG_LEVEL")
    api_media_mode: str = Field(default="auto", alias="API_MEDIA_MODE")
    api_media_allowed_roots: str = Field(default=MEDIA_DEFAULT_ALLOWED_ROOTS, alias="API_MEDIA_ALLOWED_ROOTS")
    api_media_url_ttl_seconds: int = Field(default=300, alias="API_MEDIA_URL_TTL_SECONDS", ge=60, le=900)
    supabase_media_bucket: str = Field(default="", alias="SUPABASE_MEDIA_BUCKET")
    natural_language_search_enabled: bool = Field(default=True, alias="NATURAL_LANGUAGE_SEARCH_ENABLED")
    natural_language_search_provider: str = Field(default="gemini", alias="NATURAL_LANGUAGE_SEARCH_PROVIDER")
    natural_language_search_model: str = Field(default="gemini-2.5-flash", alias="NATURAL_LANGUAGE_SEARCH_MODEL")
    natural_language_search_timeout_seconds: int = Field(default=20, alias="NATURAL_LANGUAGE_SEARCH_TIMEOUT_SECONDS", ge=1, le=120)
    natural_language_search_max_retries: int = Field(default=2, alias="NATURAL_LANGUAGE_SEARCH_MAX_RETRIES", ge=0, le=5)
    gemini_api_key: str = Field(
        default="",
        alias="GEMINI_API_KEY",
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    )

    @property
    def cors_origins(self) -> list[str]:
        origins = _parse_list_setting(self.api_cors_origins)
        if any(origin == "*" for origin in origins):
            raise ValueError("Wildcard CORS origins are not allowed.")
        return origins

    @property
    def media_mode(self) -> str:
        normalized = str(self.api_media_mode or "").strip().upper() or "AUTO"
        if normalized not in MEDIA_MODES:
            raise ValueError(f"API_MEDIA_MODE must be one of: {', '.join(sorted(MEDIA_MODES))}")
        return normalized

    @property
    def media_allowed_roots(self) -> list[Path]:
        roots = _parse_list_setting(self.api_media_allowed_roots)
        resolved: list[Path] = []
        seen: set[str] = set()
        for item in roots:
            candidate = Path(item)
            absolute = candidate if candidate.is_absolute() else (REPO_ROOT / candidate)
            normalized = absolute.resolve()
            normalized_key = str(normalized).lower()
            if normalized_key in seen:
                continue
            seen.add(normalized_key)
            resolved.append(normalized)
        return resolved

    @property
    def media_bucket(self) -> str | None:
        bucket = str(self.supabase_media_bucket or "").strip()
        return bucket or None

    @property
    def natural_language_provider(self) -> str:
        return str(self.natural_language_search_provider or "").strip().lower() or "gemini"

    def credentials_summary(self) -> dict[str, str]:
        return {
            "SUPABASE_URL": "SET" if self.supabase_url.strip() else "MISSING",
            "SUPABASE_SERVICE_ROLE_KEY": "SET" if self.supabase_service_role_key.strip() else "MISSING",
        }

    def validate_runtime(self) -> None:
        missing: list[str] = []
        if not self.supabase_url.strip():
            missing.append("SUPABASE_URL")
        if not self.supabase_service_role_key.strip():
            missing.append("SUPABASE_SERVICE_ROLE_KEY")
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


@lru_cache(maxsize=1)
def get_settings() -> ApiSettings:
    return ApiSettings()
