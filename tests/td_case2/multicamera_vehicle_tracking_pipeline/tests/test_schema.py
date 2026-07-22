from __future__ import annotations

import unittest
from pathlib import Path

from tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts.validate_database_schema import validate_schema_text


class SchemaTests(unittest.TestCase):
    def test_simplified_migration_contains_required_objects(self) -> None:
        sql = (
            Path(__file__).resolve().parents[1]
            / "database"
            / "migrations"
            / "simplified_schema.sql"
        ).read_text(encoding="utf-8")
        validate_schema_text(sql)
        lower_sql = sql.lower()
        self.assertIn("drop table if exists public.stream_sessions cascade;", lower_sql)
        self.assertIn("create or replace view public.searchable_vehicles", lower_sql)


if __name__ == "__main__":
    unittest.main()

