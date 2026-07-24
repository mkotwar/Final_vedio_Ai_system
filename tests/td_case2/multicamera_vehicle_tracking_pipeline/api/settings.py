from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .errors import BadRequestError


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    supabase_url: str = Field(default="", alias="SUPABASE_URL")
    supabase_service_role_key: str = Field(default="", alias="SUPABASE_SERVICE_ROLE_KEY")
    api_host: str = Field(default="127.0.0.1", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_cors_origins: str = Field(default="http://localhost:5173", alias="API_CORS_ORIGINS")
    api_page_size_default: int = Field(default=25, alias="API_PAGE_SIZE_DEFAULT")
    api_page_size_max: int = Field(default=100, alias="API_PAGE_SIZE_MAX")
    api_log_level: str = Field(default="INFO", alias="API_LOG_LEVEL")

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.api_cors_origins.split(",") if item.strip()]

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
