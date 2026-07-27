from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from fastapi.responses import FileResponse

from ..api.services.media_service import MediaService
from ..api.settings import ApiSettings
from .test_api_helpers import FakeApiRepository, build_test_client


def _write_file(path: Path, content: bytes = b"fake-image") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _build_media_repository(storage_uri: str, *, provider: str = "LOCAL") -> FakeApiRepository:
    repository = FakeApiRepository()
    row = {
        "id": "media-test",
        "vehicle_track_id": "track-1",
        "media_type": "BEST_VEHICLE_CROP",
        "storage_provider": provider,
        "storage_uri": storage_uri,
        "thumbnail_uri": None,
        "frame_number": 42,
        "width": 120,
        "height": 80,
        "quality_score": 0.91,
        "sharpness_score": 0.82,
        "visibility_score": 0.88,
        "selection_rank": 1,
        "is_primary": True,
    }
    repository.get_media_by_id = lambda media_id: row if media_id == "media-test" else None  # type: ignore[method-assign]
    return repository


def test_valid_local_image_is_returned_with_200_and_content_type() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_file(root / "RUN_1" / "CAM_001" / "best_overall.jpg")
        client = build_test_client(
            _build_media_repository("RUN_1/CAM_001/best_overall.jpg"),
            API_MEDIA_ALLOWED_ROOTS=str(root),
        )

        metadata = client.get("/api/v1/media/media-test")
        assert metadata.status_code == 200
        assert metadata.json()["availability"] == "LOCAL_FILE"
        assert metadata.json()["content_url"] == "/api/v1/media/media-test/content"

        response = client.get("/api/v1/media/media-test/content")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("image/jpeg")


def test_missing_file_returns_safe_404() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        client = build_test_client(
            _build_media_repository("RUN_1/CAM_001/missing.jpg"),
            API_MEDIA_ALLOWED_ROOTS=tmpdir,
        )
        response = client.get("/api/v1/media/media-test/content")
        assert response.status_code == 404
        assert "missing.jpg" not in response.text


def test_path_traversal_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        client = build_test_client(
            _build_media_repository("../secret.jpg"),
            API_MEDIA_ALLOWED_ROOTS=tmpdir,
        )
        metadata = client.get("/api/v1/media/media-test")
        assert metadata.status_code == 200
        assert metadata.json()["availability"] == "UNSAFE_REFERENCE"
        assert metadata.json()["content_url"] is None
        response = client.get("/api/v1/media/media-test/content")
        assert response.status_code == 404


def test_absolute_path_outside_allowed_roots_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        outside = Path(tmpdir).parent / "outside.jpg"
        outside.write_bytes(b"outside")
        client = build_test_client(
            _build_media_repository(str(outside)),
            API_MEDIA_ALLOWED_ROOTS=tmpdir,
        )
        metadata = client.get("/api/v1/media/media-test")
        assert metadata.status_code == 200
        body = metadata.json()
        assert body["availability"] == "UNSAFE_REFERENCE"
        assert str(outside) not in metadata.text


def test_symlink_escape_is_rejected() -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("Symlinks are not supported on this platform.")

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        outside_dir = root.parent / "outside-media"
        outside_dir.mkdir(parents=True, exist_ok=True)
        outside_file = outside_dir / "escape.jpg"
        outside_file.write_bytes(b"escape")
        link_path = root / "RUN_1" / "CAM_001" / "escaped.jpg"
        link_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(outside_file, link_path)
        except OSError:
            pytest.skip("Symlink creation is not available in this environment.")

        client = build_test_client(
            _build_media_repository("RUN_1/CAM_001/escaped.jpg"),
            API_MEDIA_ALLOWED_ROOTS=str(root),
        )
        metadata = client.get("/api/v1/media/media-test")
        assert metadata.status_code == 200
        assert metadata.json()["availability"] == "UNSAFE_REFERENCE"


def test_unsupported_extension_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_file(root / "RUN_1" / "CAM_001" / "best_overall.bmp")
        client = build_test_client(
            _build_media_repository("RUN_1/CAM_001/best_overall.bmp"),
            API_MEDIA_ALLOWED_ROOTS=str(root),
        )
        metadata = client.get("/api/v1/media/media-test")
        assert metadata.status_code == 200
        assert metadata.json()["availability"] == "UNSAFE_REFERENCE"


def test_directory_path_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "RUN_1" / "CAM_001" / "folder.jpg").mkdir(parents=True)
        client = build_test_client(
            _build_media_repository("RUN_1/CAM_001/folder.jpg"),
            API_MEDIA_ALLOWED_ROOTS=str(root),
        )
        metadata = client.get("/api/v1/media/media-test")
        assert metadata.status_code == 200
        assert metadata.json()["availability"] == "UNSAFE_REFERENCE"


def test_reference_only_media_has_no_content_url() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        client = build_test_client(
            _build_media_repository("bucket/path/to/object.jpg", provider="SUPABASE_STORAGE"),
            API_MEDIA_ALLOWED_ROOTS=tmpdir,
        )
        metadata = client.get("/api/v1/media/media-test")
        assert metadata.status_code == 200
        assert metadata.json()["availability"] == "REFERENCE_ONLY"
        assert metadata.json()["content_url"] is None


def test_supabase_storage_signed_url_returns_only_when_bucket_configured() -> None:
    repository = _build_media_repository("bucket/path/to/object.jpg", provider="SUPABASE_STORAGE")

    class _FakeBucket:
        def create_signed_url(self, path: str, ttl: int):
            return {"signedURL": f"https://storage.example/{path}?ttl={ttl}"}

    class _FakeStorage:
        def from_(self, bucket_name: str):
            assert bucket_name == "vehicle-evidence"
            return _FakeBucket()

    class _FakeSupabaseClient:
        storage = _FakeStorage()

    class _FakeClientWrapper:
        client = _FakeSupabaseClient()

    repository.client = _FakeClientWrapper()  # type: ignore[attr-defined]

    reference_only_client = build_test_client(repository)
    reference_only_response = reference_only_client.get("/api/v1/media/media-test/url")
    assert reference_only_response.status_code == 200
    assert reference_only_response.json()["availability"] == "REFERENCE_ONLY"
    assert reference_only_response.json()["url"] is None

    signed_client = build_test_client(repository, SUPABASE_MEDIA_BUCKET="vehicle-evidence")
    signed_response = signed_client.get("/api/v1/media/media-test/url")
    assert signed_response.status_code == 200
    assert signed_response.json()["availability"] == "SIGNED_URL"
    assert signed_response.json()["url"].startswith("https://storage.example/")


def test_media_content_response_includes_cors_header() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_file(root / "RUN_1" / "CAM_001" / "best_overall.jpg")
        client = build_test_client(
            _build_media_repository("RUN_1/CAM_001/best_overall.jpg"),
            API_MEDIA_ALLOWED_ROOTS=str(root),
        )
        response = client.get("/api/v1/media/media-test/content", headers={"Origin": "http://127.0.0.1:5173"})
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"


def test_large_image_uses_fileresponse() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_file(root / "RUN_1" / "CAM_001" / "best_overall.jpg", b"x" * 1024 * 1024)
        repository = _build_media_repository("RUN_1/CAM_001/best_overall.jpg")
        settings = ApiSettings(
            SUPABASE_URL="https://example.supabase.co",
            SUPABASE_SERVICE_ROLE_KEY="secret",
            API_MEDIA_ALLOWED_ROOTS=str(root),
        )
        service = MediaService(repository, settings=settings)
        response = service.get_media_content_response("media-test")
        assert isinstance(response, FileResponse)
