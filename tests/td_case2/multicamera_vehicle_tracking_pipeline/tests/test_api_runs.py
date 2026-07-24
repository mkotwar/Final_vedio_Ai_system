from __future__ import annotations

from .test_api_helpers import FakeApiRepository, build_test_client


def test_runs_list_supports_pagination_and_status_filter() -> None:
    repository = FakeApiRepository()
    client = build_test_client(repository)
    response = client.get("/api/v1/runs?page=1&page_size=25&status=COMPLETED")
    assert response.status_code == 200
    body = response.json()
    assert body["page"] == 1
    assert body["page_size"] == 25
    assert body["total"] == 1
    assert body["items"][0]["status"] == "COMPLETED"
    assert repository.calls[-1][1]["status"] == "COMPLETED"


def test_runs_list_rejects_invalid_sort_field() -> None:
    client = build_test_client(FakeApiRepository())
    response = client.get("/api/v1/runs?sort_by=invalid")
    assert response.status_code == 422


def test_run_detail_returns_expected_payload() -> None:
    client = build_test_client(FakeApiRepository())
    response = client.get("/api/v1/runs/RUN_20260724_151402")
    assert response.status_code == 200
    assert response.json()["global_object_summary"]["global_vehicle_count"] == 7


def test_missing_run_returns_404() -> None:
    client = build_test_client(FakeApiRepository())
    response = client.get("/api/v1/runs/RUN_MISSING")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "RUN_NOT_FOUND"
