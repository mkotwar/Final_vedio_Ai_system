from __future__ import annotations

from .test_api_helpers import FakeApiRepository, build_test_client


def test_track_media_returns_safe_metadata() -> None:
    client = build_test_client(FakeApiRepository())
    response = client.get("/api/v1/tracks/RUN_20260724_151402:CAM_001:TRACK_4/media")
    assert response.status_code == 200
    assert response.json()[0]["storage_uri"] == "safe/ref.jpg"


def test_media_reference_only_response_is_supported() -> None:
    client = build_test_client(FakeApiRepository())
    response = client.get("/api/v1/media/media-1")
    assert response.status_code == 200
    assert response.json()["availability"] == "REFERENCE_ONLY"


def test_media_traversal_and_absolute_paths_are_rejected() -> None:
    client = build_test_client(FakeApiRepository())
    assert client.get("/api/v1/media/media-unsafe-traversal").status_code == 400
    assert client.get("/api/v1/media/media-unsafe-absolute").status_code == 400


def test_missing_media_returns_404() -> None:
    client = build_test_client(FakeApiRepository())
    assert client.get("/api/v1/media/missing-media").status_code == 404
