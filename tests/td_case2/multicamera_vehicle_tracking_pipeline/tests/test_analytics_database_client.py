from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from tests.td_case2.multicamera_vehicle_tracking_pipeline.persistence.analytics_database_client import (
    AnalyticsDatabaseClient,
    AnalyticsDatabaseClientConfig,
    AnalyticsDatabaseClientError,
)


class _FakeQuery:
    def __init__(self) -> None:
        self.operations: list[tuple[str, object]] = []

    def select(self, *args, **kwargs):
        self.operations.append(("select", (args, kwargs)))
        return self

    def limit(self, value):
        self.operations.append(("limit", value))
        return self

    def execute(self):
        self.operations.append(("execute", None))
        return {"data": []}


class _FakeSchemaClient:
    def __init__(self) -> None:
        self.tables: list[str] = []
        self.queries: dict[str, _FakeQuery] = {}

    def table(self, name: str):
        self.tables.append(name)
        query = _FakeQuery()
        self.queries[name] = query
        return query


class _FakeSupabaseClient:
    def __init__(self) -> None:
        self.schema_calls: list[str] = []
        self.schema_client = _FakeSchemaClient()

    def schema(self, name: str):
        self.schema_calls.append(name)
        return self.schema_client


class AnalyticsDatabaseClientTests(unittest.TestCase):
    def test_missing_environment_variables_raise_clear_error(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(AnalyticsDatabaseClientError) as ctx:
                AnalyticsDatabaseClientConfig.from_env()
        self.assertEqual(ctx.exception.code, "missing_environment_variables")
        self.assertIn("SUPABASE_URL", str(ctx.exception))

    def test_postgres_url_is_normalized_to_project_url(self) -> None:
        with patch.dict(
            os.environ,
            {
                "supabase_database_url": "postgresql://postgres:secret@db.exampleproj.supabase.co:5432/postgres",
                "SUPABASE_SERVICE_ROLE_KEY": "service-role",
            },
            clear=True,
        ):
            config = AnalyticsDatabaseClientConfig.from_env()
        self.assertEqual(config.supabase_url, "https://exampleproj.supabase.co")

    def test_client_uses_schema_scoped_client(self) -> None:
        fake = _FakeSupabaseClient()
        client = AnalyticsDatabaseClient(supabase_client=fake, schema_name="analytics")
        self.assertEqual(fake.schema_calls, ["analytics"])
        client.table("camera")
        self.assertEqual(fake.schema_client.tables, ["camera"])

    def test_health_check_queries_camera_table(self) -> None:
        fake = _FakeSupabaseClient()
        client = AnalyticsDatabaseClient(supabase_client=fake, schema_name="analytics")
        self.assertTrue(client.health_check())
        query = fake.schema_client.queries["camera"]
        self.assertEqual(query.operations[0][0], "select")
        self.assertEqual(query.operations[1], ("limit", 1))
        self.assertEqual(query.operations[2][0], "execute")

    def test_empty_table_name_is_rejected(self) -> None:
        fake = _FakeSupabaseClient()
        client = AnalyticsDatabaseClient(supabase_client=fake, schema_name="analytics")
        with self.assertRaises(AnalyticsDatabaseClientError) as ctx:
            client.table(" ")
        self.assertEqual(ctx.exception.code, "invalid_table_name")


if __name__ == "__main__":
    unittest.main()
