from __future__ import annotations

import pytest
from pydantic import ValidationError

from ..api.settings import ApiSettings


def test_api_settings_credentials_summary_masks_values() -> None:
    settings = ApiSettings(SUPABASE_URL="https://example.supabase.co", SUPABASE_SERVICE_ROLE_KEY="secret")
    assert settings.credentials_summary() == {
        "SUPABASE_URL": "SET",
        "SUPABASE_SERVICE_ROLE_KEY": "SET",
    }


def test_api_settings_validate_runtime_requires_credentials() -> None:
    settings = ApiSettings(SUPABASE_URL="", SUPABASE_SERVICE_ROLE_KEY="")
    with pytest.raises(RuntimeError):
        settings.validate_runtime()


def test_api_settings_parses_comma_separated_cors_origins() -> None:
    settings = ApiSettings(
        SUPABASE_URL="https://example.supabase.co",
        SUPABASE_SERVICE_ROLE_KEY="secret",
        API_CORS_ORIGINS="http://127.0.0.1:5173,http://localhost:5173",
    )
    assert settings.cors_origins == [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ]


def test_api_settings_parses_json_array_cors_origins() -> None:
    settings = ApiSettings(
        SUPABASE_URL="https://example.supabase.co",
        SUPABASE_SERVICE_ROLE_KEY="secret",
        API_CORS_ORIGINS='["http://127.0.0.1:5173", "http://localhost:5173"]',
    )
    assert settings.cors_origins == [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ]


def test_api_settings_rejects_wildcard_cors_origins() -> None:
    settings = ApiSettings(
        SUPABASE_URL="https://example.supabase.co",
        SUPABASE_SERVICE_ROLE_KEY="secret",
        API_CORS_ORIGINS="*",
    )
    with pytest.raises(ValueError, match="Wildcard CORS origins are not allowed."):
        _ = settings.cors_origins


def test_api_settings_parses_media_allowed_roots() -> None:
    settings = ApiSettings(
        SUPABASE_URL="https://example.supabase.co",
        SUPABASE_SERVICE_ROLE_KEY="secret",
        API_MEDIA_ALLOWED_ROOTS='["artifacts", "debug_runs/multicamera_vehicle_tracking_pipeline"]',
    )
    roots = settings.media_allowed_roots
    assert any(str(root).endswith("artifacts") for root in roots)
    assert any("debug_runs" in str(root) for root in roots)


def test_api_settings_validates_media_url_ttl_seconds() -> None:
    with pytest.raises(ValidationError):
        ApiSettings(
            SUPABASE_URL="https://example.supabase.co",
            SUPABASE_SERVICE_ROLE_KEY="secret",
            API_MEDIA_URL_TTL_SECONDS=30,
        )
