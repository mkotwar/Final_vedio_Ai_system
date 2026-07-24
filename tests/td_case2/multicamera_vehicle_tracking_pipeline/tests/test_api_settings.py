from __future__ import annotations

import pytest

from ..api.settings import ApiSettings


def test_api_settings_credentials_summary_masks_values() -> None:
    settings = ApiSettings(SUPABASE_URL="https://example.supabase.co", SUPABASE_SERVICE_ROLE_KEY="secret")
    assert settings.credentials_summary() == {
        "SUPABASE_URL": "SET",
        "SUPABASE_SERVICE_ROLE_KEY": "SET",
    }


def test_api_settings_validate_runtime_requires_credentials() -> None:
    settings = ApiSettings()
    with pytest.raises(RuntimeError):
        settings.validate_runtime()
