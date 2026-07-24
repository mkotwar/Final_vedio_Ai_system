from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Callable

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..persistence.analytics_repository_base import AnalyticsRepositoryError
from .dependencies import build_repository_from_settings, sanitize_payload
from .errors import ApiError
from .response_models import ErrorResponse
from .routers import cameras, global_vehicles, health, matches, media, runs, tracks
from .settings import ApiSettings, get_settings


LOGGER = logging.getLogger("multicamera_vehicle_api")


def _error_response(code: str, message: str, status_code: int, details: Any = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(error={"code": code, "message": message, "details": sanitize_payload(details)}).model_dump(),
    )


def create_app(
    *,
    settings: ApiSettings | None = None,
    repository_factory: Callable[[ApiSettings], Any] | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    resolved_factory = repository_factory or build_repository_from_settings

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(level=getattr(logging, resolved_settings.api_log_level.upper(), logging.INFO))
        LOGGER.info("Credentials: SUPABASE_URL=%s SUPABASE_SERVICE_ROLE_KEY=%s", resolved_settings.credentials_summary()["SUPABASE_URL"], resolved_settings.credentials_summary()["SUPABASE_SERVICE_ROLE_KEY"])
        app.state.settings = resolved_settings
        app.state.repository = resolved_factory(resolved_settings)
        yield

    app = FastAPI(
        title="Multicamera Vehicle Tracking API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "OPTIONS"],
        allow_headers=["Accept", "Authorization", "Content-Type"],
    )

    @app.exception_handler(ApiError)
    async def api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
        return _error_response(exc.code, exc.message, exc.status_code, exc.details)

    @app.exception_handler(AnalyticsRepositoryError)
    async def repository_error_handler(_: Request, exc: AnalyticsRepositoryError) -> JSONResponse:
        return _error_response("DATABASE_QUERY_FAILED", "A database query failed.", 502, None)

    @app.exception_handler(httpx.HTTPError)
    async def httpx_error_handler(_: Request, exc: httpx.HTTPError) -> JSONResponse:
        return _error_response("DATABASE_UNREACHABLE", "The analytics database is unreachable.", 502, None)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _error_response("VALIDATION_ERROR", "Request validation failed.", 422, exc.errors())

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
        LOGGER.exception("Unhandled API error", exc_info=exc)
        return _error_response("INTERNAL_SERVER_ERROR", "An internal server error occurred.", 500, None)

    app.include_router(health.router, prefix="/api/v1")
    app.include_router(runs.router, prefix="/api/v1")
    app.include_router(cameras.router, prefix="/api/v1")
    app.include_router(tracks.router, prefix="/api/v1")
    app.include_router(global_vehicles.router, prefix="/api/v1")
    app.include_router(matches.router, prefix="/api/v1")
    app.include_router(media.router, prefix="/api/v1")
    return app


app = create_app()
