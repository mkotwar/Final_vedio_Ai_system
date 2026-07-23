from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


try:
    from supabase import Client, create_client
except Exception:  # pragma: no cover - optional dependency
    Client = Any  # type: ignore[assignment]
    create_client = None


class AnalyticsDatabaseClientError(RuntimeError):
    """Raised when the analytics database client cannot be initialized or queried."""

    def __init__(self, message: str, *, code: str, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


@dataclass(frozen=True, slots=True)
class AnalyticsDatabaseClientConfig:
    supabase_url: str
    supabase_service_role_key: str
    schema_name: str = "analytics"

    @classmethod
    def from_env(cls) -> "AnalyticsDatabaseClientConfig":
        raw_url = (
            os.getenv("SUPABASE_URL", "").strip()
            or os.getenv("supabase_database_url", "").strip()
            or os.getenv("SUPABASE_DATABASE_URL", "").strip()
        )
        service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        schema_name = os.getenv("ANALYTICS_DATABASE_SCHEMA", "analytics").strip() or "analytics"

        missing: list[str] = []
        if not raw_url:
            missing.append("SUPABASE_URL")
        if not service_role_key:
            missing.append("SUPABASE_SERVICE_ROLE_KEY")
        if missing:
            raise AnalyticsDatabaseClientError(
                f"Missing required environment variables: {', '.join(missing)}",
                code="missing_environment_variables",
                details={"missing": missing, "schema_name": schema_name},
            )

        return cls(
            supabase_url=_normalize_supabase_project_url(raw_url),
            supabase_service_role_key=service_role_key,
            schema_name=schema_name,
        )

    def masked_summary(self) -> dict[str, str]:
        return {
            "supabase_url": "<set>" if self.supabase_url else "<missing>",
            "supabase_service_role_key": "<set>" if self.supabase_service_role_key else "<missing>",
            "schema_name": self.schema_name,
        }


def _normalize_supabase_project_url(raw_url: str) -> str:
    value = str(raw_url).strip()
    if not value:
        raise AnalyticsDatabaseClientError(
            "Supabase URL must not be empty.",
            code="invalid_supabase_url",
        )
    if value.startswith("http://") or value.startswith("https://"):
        return value
    parsed = urlparse(value)
    hostname = (parsed.hostname or "").strip().lower()
    if hostname.startswith("db.") and hostname.endswith(".supabase.co"):
        project_ref = hostname[len("db.") : -len(".supabase.co")]
        if project_ref:
            return f"https://{project_ref}.supabase.co"
    raise AnalyticsDatabaseClientError(
        "Invalid Supabase URL. Expected SUPABASE_URL or a Supabase-hosted Postgres URL.",
        code="invalid_supabase_url",
        details={"supplied_format": "unrecognized"},
    )


class AnalyticsDatabaseClient:
    def __init__(
        self,
        supabase_client: Client | None = None,
        schema_name: str = "analytics",
        *,
        config: AnalyticsDatabaseClientConfig | None = None,
    ) -> None:
        if config is None and supabase_client is not None:
            config = AnalyticsDatabaseClientConfig(
                supabase_url="https://injected-client.local",
                supabase_service_role_key="<injected>",
                schema_name=schema_name or "analytics",
            )
        self._config = config or AnalyticsDatabaseClientConfig.from_env()
        self.schema_name = str(schema_name or self._config.schema_name).strip() or "analytics"

        if supabase_client is None:
            if create_client is None:
                raise AnalyticsDatabaseClientError(
                    "The 'supabase' package is not installed.",
                    code="missing_supabase_dependency",
                )
            try:
                supabase_client = create_client(
                    self._config.supabase_url,
                    self._config.supabase_service_role_key,
                )
            except Exception as exc:  # pragma: no cover - dependency/network bound
                raise AnalyticsDatabaseClientError(
                    f"Failed to initialize Supabase client: {exc}",
                    code="client_initialization_failed",
                    details={"schema_name": self.schema_name},
                ) from exc

        self._client = supabase_client
        try:
            self._schema_client = self._client.schema(self.schema_name)
        except Exception as exc:
            raise AnalyticsDatabaseClientError(
                f"Failed to scope Supabase client to schema '{self.schema_name}': {exc}",
                code="schema_scoping_failed",
                details={"schema_name": self.schema_name},
            ) from exc

    @property
    def client(self) -> Client:
        return self._client

    @property
    def schema(self) -> Client:
        return self._schema_client

    @property
    def masked_summary(self) -> dict[str, str]:
        return self._config.masked_summary() | {"schema_name": self.schema_name}

    def table(self, table_name: str) -> Any:
        name = str(table_name).strip()
        if not name:
            raise AnalyticsDatabaseClientError(
                "Table name must not be empty.",
                code="invalid_table_name",
                details={"schema_name": self.schema_name},
            )
        try:
            return self.schema.table(name)
        except Exception as exc:
            raise AnalyticsDatabaseClientError(
                f"Failed to open table '{name}' in schema '{self.schema_name}': {exc}",
                code="table_binding_failed",
                details={"schema_name": self.schema_name, "table_name": name},
            ) from exc

    def health_check(self) -> bool:
        try:
            self.table("camera").select("id", count="exact").limit(1).execute()
        except Exception as exc:
            if isinstance(exc, AnalyticsDatabaseClientError):
                raise
            raise AnalyticsDatabaseClientError(
                f"Analytics database health check failed: {exc}",
                code="health_check_failed",
                details={"schema_name": self.schema_name, "table_name": "camera"},
            ) from exc
        return True
