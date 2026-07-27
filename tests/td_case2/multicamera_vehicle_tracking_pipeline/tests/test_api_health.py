from __future__ import annotations

from .test_api_helpers import FakeApiRepository, build_test_client


def test_health_endpoint_returns_ok() -> None:
    client = build_test_client(FakeApiRepository())
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "multicamera-vehicle-api",
        "database": "reachable",
        "schema": "analytics",
    }


def test_health_endpoint_handles_database_failure() -> None:
    repository = FakeApiRepository()
    repository.raise_health_error = True
    client = build_test_client(repository)
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded",
        "service": "multicamera-vehicle-api",
        "database": "unreachable",
        "schema": "analytics",
    }


def test_health_response_does_not_expose_credentials() -> None:
    client = build_test_client(FakeApiRepository())
    response = client.get("/api/v1/health")
    body = response.text
    assert "super-secret" not in body
    assert "SUPABASE_SERVICE_ROLE_KEY" not in body


def test_health_endpoint_includes_cors_header_for_127001_origin() -> None:
    client = build_test_client(FakeApiRepository())
    response = client.get("/api/v1/health", headers={"Origin": "http://127.0.0.1:5173"})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"


def test_health_endpoint_includes_cors_header_for_localhost_origin() -> None:
    client = build_test_client(FakeApiRepository())
    response = client.get("/api/v1/health", headers={"Origin": "http://localhost:5173"})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
