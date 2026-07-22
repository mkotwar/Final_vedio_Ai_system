from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse
from typing import Any

from .config import DatabaseConfig


class DatabaseClientError(RuntimeError):
    """Raised for Supabase client initialization or connectivity failures."""


try:
    from supabase import Client, create_client
except Exception:  # pragma: no cover - optional dependency
    Client = Any  # type: ignore[assignment]
    create_client = None


@dataclass
class SupabaseClients:
    backend: Client
    anon: Client | None = None


def _normalize_supabase_client_url(raw_url: str) -> str:
    value = str(raw_url).strip()
    if value.startswith("http://") or value.startswith("https://"):
        return value
    parsed = urlparse(value)
    hostname = (parsed.hostname or "").strip().lower()
    if hostname.startswith("db.") and hostname.endswith(".supabase.co"):
        project_ref = hostname[len("db.") : -len(".supabase.co")]
        if project_ref:
            return f"https://{project_ref}.supabase.co"
    raise DatabaseClientError("Invalid Supabase URL. Expected an https project URL or a Postgres URL hosted on db.<project-ref>.supabase.co.")


def create_backend_client(config: DatabaseConfig) -> Client:
    if create_client is None:
        raise DatabaseClientError("The 'supabase' package is not installed. Install the optional 'supabase' dependency to connect.")
    return create_client(_normalize_supabase_client_url(config.supabase_database_url), config.supabase_service_role_key)


def create_anon_client(config: DatabaseConfig) -> Client | None:
    if not config.supabase_anon_key:
        return None
    if create_client is None:
        raise DatabaseClientError("The 'supabase' package is not installed. Install the optional 'supabase' dependency to connect.")
    return create_client(_normalize_supabase_client_url(config.supabase_database_url), config.supabase_anon_key)


def create_clients(config: DatabaseConfig) -> SupabaseClients:
    return SupabaseClients(backend=create_backend_client(config), anon=create_anon_client(config))


def health_check(client: Client) -> dict[str, Any]:
    try:
        response = client.table("cameras").select("id", count="exact").limit(1).execute()
    except Exception as exc:  # pragma: no cover - network/dependency bound
        raise DatabaseClientError(f"Supabase health check failed: {exc}") from exc
    return {
        "status": "ok",
        "table": "cameras",
        "row_count_hint": getattr(response, "count", None),
    }
