from __future__ import annotations

from .test_api_helpers import FakeApiRepository, build_test_client


def test_tracks_list_forwards_filters_and_pagination() -> None:
    repository = FakeApiRepository()
    client = build_test_client(repository)
    response = client.get(
        "/api/v1/runs/RUN_20260724_151402/tracks"
        "?camera_code=CAM_001&vehicle_class=CAR&colour=GREY&plate=DL8CBF6268"
        "&plate_status=VERIFIED&lifecycle_state=COMPLETED&minimum_confidence=0.8"
        "&has_media=true&page=1&page_size=25"
    )
    assert response.status_code == 200
    call = repository.calls[-1]
    assert call[0] == "list_tracks"
    assert call[1]["camera_code"] == "CAM_001"
    assert call[1]["vehicle_class"] == "CAR"
    assert call[1]["colour"] == "GREY"
    assert call[1]["plate"] == "DL8CBF6268"
    assert call[1]["plate_status"] == "VERIFIED"
    assert call[1]["lifecycle_state"] == "COMPLETED"
    assert call[1]["minimum_confidence"] == 0.8
    assert call[1]["has_media"] is True


def test_track_detail_returns_safe_view() -> None:
    client = build_test_client(FakeApiRepository())
    response = client.get("/api/v1/tracks/RUN_20260724_151402:CAM_001:TRACK_4")
    assert response.status_code == 200
    body = response.json()
    assert body["track"]["canonical_plate"] == "DL8CBF6268"
    assert body["track"]["plate_result"]["display_text"] == "DL8CBF6268"
    assert body["plate"]["plate_result"]["status"] == "VERIFIED"
    assert body["global_membership"] == {
        "linked": True,
        "global_vehicle_id": "global-1",
        "global_vehicle_code": "GVO:RUN_20260724_151402:943BD1FE7C62",
        "membership_confidence": 0.95,
        "membership_status": "CONFIRMED",
        "member_track_count": 2,
    }
    assert "association_method" not in response.text
    assert "model_path" not in response.text


def test_track_detail_can_return_unlinked_membership() -> None:
    repository = FakeApiRepository()
    repository.track_detail["global_membership"] = {"linked": False}
    client = build_test_client(repository)
    response = client.get("/api/v1/tracks/RUN_20260724_151402:CAM_001:TRACK_4")
    assert response.status_code == 200
    assert response.json()["global_membership"] == {"linked": False}


def test_track_observations_support_pagination() -> None:
    client = build_test_client(FakeApiRepository())
    response = client.get("/api/v1/tracks/RUN_20260724_151402:CAM_001:TRACK_4/observations?page=1&page_size=10&key_only=true")
    assert response.status_code == 200
    assert response.json()["items"][0]["is_key_observation"] is True


def test_missing_track_returns_404() -> None:
    client = build_test_client(FakeApiRepository())
    assert client.get("/api/v1/tracks/MISSING").status_code == 404
