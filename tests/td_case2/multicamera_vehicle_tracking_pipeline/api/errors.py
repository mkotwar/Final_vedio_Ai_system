from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ApiError(Exception):
    code: str
    message: str
    status_code: int
    details: Any = None


class NotFoundError(ApiError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(code=code, message=message, status_code=404, details=None)


class BadRequestError(ApiError):
    def __init__(self, code: str, message: str, *, details: Any = None) -> None:
        super().__init__(code=code, message=message, status_code=400, details=details)
