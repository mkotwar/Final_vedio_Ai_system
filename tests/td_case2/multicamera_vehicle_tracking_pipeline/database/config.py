from __future__ import annotations

import os
from dataclasses import dataclass


class DatabaseConfigError(RuntimeError):
    """Raised when required database configuration is missing or invalid."""


@dataclass(frozen=True)
class DatabaseConfig:
    supabase_database_url: str
    supabase_service_role_key: str
    supabase_anon_key: str | None
    evidence_bucket: str
    schema_version: str

    @classmethod
    def from_env(cls, *, require_backend_credentials: bool = True) -> "DatabaseConfig":
        supabase_database_url = (
            os.getenv("supabase_database_url", "").strip()
            or os.getenv("SUPABASE_DATABASE_URL", "").strip()
            or os.getenv("SUPABASE_URL", "").strip()
        )
        service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        anon_key = os.getenv("SUPABASE_ANON_KEY", "").strip() or None
        evidence_bucket = os.getenv("SUPABASE_EVIDENCE_BUCKET", "vehicle-evidence").strip() or "vehicle-evidence"
        schema_version = os.getenv("DATABASE_SCHEMA_VERSION", "simplified_schema").strip()

        missing: list[str] = []
        if not supabase_database_url:
            missing.append("supabase_database_url")
        if require_backend_credentials and not service_key:
            missing.append("SUPABASE_SERVICE_ROLE_KEY")
        if missing:
            raise DatabaseConfigError(f"Missing required environment variables: {', '.join(missing)}")
        return cls(
            supabase_database_url=supabase_database_url,
            supabase_service_role_key=service_key,
            supabase_anon_key=anon_key,
            evidence_bucket=evidence_bucket,
            schema_version=schema_version,
        )

    def masked_summary(self) -> dict[str, str]:
        return {
            "supabase_database_url": "<set>" if self.supabase_database_url else "<missing>",
            "supabase_service_role_key": "<set>" if self.supabase_service_role_key else "<missing>",
            "supabase_anon_key": "<set>" if self.supabase_anon_key else "<missing>",
            "evidence_bucket": self.evidence_bucket,
            "schema_version": self.schema_version,
        }
