from __future__ import annotations

from .test_api_helpers import FakeApiRepository, build_test_client


def test_matches_list_supports_decision_and_score_filters() -> None:
    repository = FakeApiRepository()
    client = build_test_client(repository)
    response = client.get("/api/v1/cross-camera-matches?run_code=RUN_20260724_151402&decision=CONFIRMED&minimum_score=0.9")
    assert response.status_code == 200
    body = response.json()
    assert body["items"][0]["decision"] == "CONFIRMED"
    assert body["items"][0]["linked_global_vehicle_code"] == "GVO:RUN_20260724_151402:943BD1FE7C62"


def test_match_detail_returns_complete_safe_payload() -> None:
    client = build_test_client(FakeApiRepository())
    response = client.get("/api/v1/cross-camera-matches/match-1")
    assert response.status_code == 200
    assert response.json()["source_track_uuid"] == "RUN_20260724_151402:CAM_001:TRACK_4"
