from __future__ import annotations

from .test_api_helpers import FakeApiRepository, build_test_client


def test_run_cameras_list_returns_camera_rows() -> None:
    client = build_test_client(FakeApiRepository())
    response = client.get("/api/v1/runs/RUN_20260724_151402/cameras")
    assert response.status_code == 200
    assert response.json()["items"][0]["camera_code"] == "CAM_001"


def test_camera_detail_returns_metrics() -> None:
    client = build_test_client(FakeApiRepository())
    response = client.get("/api/v1/runs/RUN_20260724_151402/cameras/CAM_001")
    assert response.status_code == 200
    assert response.json()["track_count"] == 4


def test_wrong_camera_or_run_returns_404() -> None:
    client = build_test_client(FakeApiRepository())
    assert client.get("/api/v1/runs/RUN_MISSING/cameras/CAM_001").status_code == 404
    assert client.get("/api/v1/runs/RUN_20260724_151402/cameras/CAM_999").status_code == 404
