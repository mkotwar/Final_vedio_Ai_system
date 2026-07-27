from __future__ import annotations

from .test_api_helpers import FakeApiRepository, build_test_client


def test_service_role_and_model_paths_are_not_exposed() -> None:
    client = build_test_client(FakeApiRepository())
    response = client.get("/api/v1/runs/RUN_20260724_151402")
    body = response.text
    assert "super-secret" not in body
    assert "service_role_key" not in body
    assert "model_path" not in body


def test_database_errors_are_sanitized() -> None:
    repository = FakeApiRepository()
    repository.raise_run_list_error = True
    client = build_test_client(repository)
    response = client.get("/api/v1/runs")
    assert response.status_code == 502
    assert "token=abc" not in response.text


def test_openapi_does_not_expose_credentials() -> None:
    client = build_test_client(FakeApiRepository())
    response = client.get("/openapi.json")
    assert response.status_code == 200
    body = response.text
    assert "SUPABASE_SERVICE_ROLE_KEY" not in body
    assert "super-secret" not in body


def test_media_metadata_does_not_expose_absolute_local_paths() -> None:
    client = build_test_client(FakeApiRepository())
    response = client.get("/api/v1/media/media-unsafe-absolute")
    assert response.status_code == 200
    body = response.text
    assert "C:/" not in body
    assert "C:\\" not in body
