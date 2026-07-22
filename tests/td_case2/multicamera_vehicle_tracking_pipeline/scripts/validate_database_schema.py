from __future__ import annotations

from pathlib import Path


REQUIRED_TABLES = [
    "cameras",
    "vehicle_tracks",
    "vehicle_attributes",
    "vehicle_observations",
    "vehicle_matches",
]
REQUIRED_VIEWS = ["searchable_vehicles"]


def _migration_text() -> str:
    root = Path(__file__).resolve().parents[1]
    migration = root / "database" / "migrations" / "simplified_schema.sql"
    return migration.read_text(encoding="utf-8")


def validate_schema_text(sql: str) -> None:
    lower_sql = sql.lower()
    for table_name in REQUIRED_TABLES:
        marker = f"create table public.{table_name}"
        if marker not in lower_sql:
            raise AssertionError(f"Missing table definition: {table_name}")
    for view_name in REQUIRED_VIEWS:
        marker = f"create or replace view public.{view_name}"
        if marker not in lower_sql:
            raise AssertionError(f"Missing view definition: {view_name}")
    for marker in [
        "create extension if not exists pg_trgm",
        "create or replace function public.set_updated_at()",
        "create or replace function public.validate_vehicle_match_cameras()",
        "enable row level security",
        "gin_trgm_ops",
        "create policy authenticated_read_vehicle_tracks",
        "create policy service_role_all_vehicle_tracks",
    ]:
        if marker not in lower_sql:
            raise AssertionError(f"Missing schema marker: {marker}")


def main() -> None:
    validate_schema_text(_migration_text())
    print({"status": "ok", "validated_tables": len(REQUIRED_TABLES), "validated_views": len(REQUIRED_VIEWS)})


if __name__ == "__main__":
    main()
