from __future__ import annotations

from .test_api_helpers import FakeApiRepository, build_test_client


def test_global_vehicle_list_supports_filters() -> None:
    repository = FakeApiRepository()
    client = build_test_client(repository)
    response = client.get(
        "/api/v1/global-vehicles"
        "?run_code=RUN_20260724_151402&plate=DL8CBF6268&colour=GREY&vehicle_class=CAR"
        "&minimum_confidence=0.95&minimum_camera_count=2"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["items"][0]["global_vehicle_code"] == "GVO:RUN_20260724_151402:943BD1FE7C62"
    assert body["items"][0]["plate_result"]["display_text"] == "DL8CBF6268"


def test_global_vehicle_detail_returns_multi_camera_object() -> None:
    client = build_test_client(FakeApiRepository())
    response = client.get("/api/v1/global-vehicles/GVO:RUN_20260724_151402:943BD1FE7C62")
    assert response.status_code == 200
    body = response.json()
    assert body["global_vehicle"]["canonical_plate"] == "DL8CBF6268"
    assert body["global_vehicle"]["plate_result"]["status"] == "VERIFIED"
    assert {member["camera_code"] for member in body["members"]} == {"CAM_001", "CAM_002"}


def test_global_vehicle_member_tracks_endpoint() -> None:
    client = build_test_client(FakeApiRepository())
    response = client.get("/api/v1/global-vehicles/GVO:RUN_20260724_151402:943BD1FE7C62/tracks")
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_missing_global_vehicle_returns_404() -> None:
    client = build_test_client(FakeApiRepository())
    assert client.get("/api/v1/global-vehicles/GVO:MISSING").status_code == 404
