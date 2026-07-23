from __future__ import annotations

from typing import Any

from .analytics_database_client import AnalyticsDatabaseClient


class AnalyticsRepositoryError(RuntimeError):
    def __init__(
        self,
        *,
        operation: str,
        table_name: str,
        message: str,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.operation = operation
        self.table_name = table_name
        self.cause = cause


class AnalyticsRepositoryBase:
    def __init__(self, client: AnalyticsDatabaseClient, *, table_name: str) -> None:
        self.client = client
        self.table_name = table_name

    def _table(self):
        try:
            return self.client.table(self.table_name)
        except Exception as exc:
            raise self._wrap_error(
                operation="bind_table",
                message=f"Failed to bind analytics table '{self.table_name}'.",
                cause=exc,
            ) from exc

    def _extract_rows(self, response: object) -> list[dict[str, Any]]:
        data = getattr(response, "data", None)
        if data is None:
            return []
        if not isinstance(data, list):
            raise self._wrap_error(
                operation="extract_rows",
                message=f"Analytics response for table '{self.table_name}' did not contain a list payload.",
            )
        return [dict(item) for item in data]

    def _expect_one(self, response: object, *, operation: str) -> dict[str, Any]:
        rows = self._extract_rows(response)
        if len(rows) != 1:
            raise self._wrap_error(
                operation=operation,
                message=f"Expected exactly one row from analytics table '{self.table_name}', got {len(rows)}.",
            )
        return rows[0]

    def _wrap_error(
        self,
        *,
        operation: str,
        message: str,
        cause: Exception | None = None,
    ) -> AnalyticsRepositoryError:
        return AnalyticsRepositoryError(
            operation=operation,
            table_name=self.table_name,
            message=message,
            cause=cause,
        )
