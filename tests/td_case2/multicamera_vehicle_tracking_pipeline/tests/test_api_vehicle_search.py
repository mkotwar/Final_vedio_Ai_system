from __future__ import annotations

import unittest

from .test_api_helpers import FakeApiRepository, build_test_client


class VehicleSearchApiTests(unittest.TestCase):
    def test_vehicle_search_forwards_structured_filters(self) -> None:
        repository = FakeApiRepository()
        client = build_test_client(repository)

        response = client.get(
            "/api/v1/search/vehicles"
            "?run_code=RUN_20260725_131944"
            "&result_scope=GLOBAL_VEHICLES"
            "&vehicle_class=CAR"
            "&colour=GREY"
            "&plate=6268"
            "&plate_match_type=ENDS_WITH"
            "&camera_codes=CAM_001,CAM_002"
            "&multi_camera_only=true"
            "&verified_plate_only=true"
            "&limit=25"
            "&offset=0"
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["pagination"]["total"], 1)
        self.assertEqual(body["results"][0]["global_vehicle_code"], "GVO:RUN_20260725_131944:FA3FCF9E3ABC")
        call = repository.calls[-1]
        self.assertEqual(call[0], "search_global_vehicles")
        self.assertEqual(call[1]["camera_codes"], ["CAM_001", "CAM_002"])
        self.assertTrue(call[1]["verified_plate_only"])

    def test_vehicle_search_rejects_invalid_confidence(self) -> None:
        client = build_test_client(FakeApiRepository())
        response = client.get("/api/v1/search/vehicles?run_code=RUN_20260725_131944&minimum_confidence=2")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "VALIDATION_ERROR")

    def test_vehicle_search_rejects_invalid_sort_field(self) -> None:
        client = build_test_client(FakeApiRepository())
        response = client.get("/api/v1/search/vehicles?run_code=RUN_20260725_131944&sort_by=DROP_TABLE")
        self.assertEqual(response.status_code, 422)

    def test_vehicle_search_response_does_not_expose_credentials_or_paths(self) -> None:
        client = build_test_client(FakeApiRepository())
        response = client.get("/api/v1/search/vehicles?run_code=RUN_20260725_131944")
        self.assertEqual(response.status_code, 200)
        text = response.text
        self.assertNotIn("SUPABASE_SERVICE_ROLE_KEY", text)
        self.assertNotIn("C:/", text)


if __name__ == "__main__":
    unittest.main()
