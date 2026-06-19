"""Main entry point for the AI Video Search Engine API.

This module initializes the FastAPI application, registers middleware, configures
lifespan event handlers for directory and database setup, and establishes core
base endpoints like the system healthcheck.
"""

import os
import time
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.core.config import settings
from app.core.logging import setup_logging
from app.core.exceptions import VideoValidationError, VideoNotFoundError, FrameExtractionError
from app.api.videos import router as videos_router
from app.api.frames import router as frames_router
from app.api.summary import router as summary_router
from app.api.search import router as search_router
from app.api.events import router as events_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to handle startup and shutdown tasks."""
    # --- Startup Actions ---
    # 1. Setup production-grade logger
    setup_logging()
    logger.info(f"Starting {settings.APP_NAME} in environment: {settings.ENV}")

    logger.info(
        "\nQwen Generation Config\n"
        "----------------------\n"
        f"max_new_tokens: {settings.QWEN_MAX_NEW_TOKENS}"
    )

    # 2. Automatically verify and create necessary application directories
    directories = [
        settings.DATA_DIR,
        settings.VIDEOS_DIR,
        settings.FRAMES_DIR,
        settings.METADATA_DIR,
        settings.LOGS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Verified directory existence: {directory}")

    # 3. Log a separate test entry directly into metadata.log to verify routing
    meta_logger = logger.bind(context="metadata")
    meta_logger.info("Metadata subsystem successfully initialized during startup.")

    # 4. Pre-initialize search and embedding resources
    try:
        from app.services.embedding_service import EmbeddingService
        from app.services.search_service import SearchService
        EmbeddingService.initialize()
        SearchService.get_client()
        SearchService.auto_index_existing_events()
        logger.info("Search and Embedding backend resources pre-initialized and warmed successfully.")
    except Exception as e:
        logger.error(f"Failed to pre-initialize search/embedding services: {e}")

    # 5. Suppress noisy WinError 10054 from asyncio ProactorEventLoop on Windows
    loop = asyncio.get_running_loop()
    default_handler = loop.get_exception_handler()
    def custom_exception_handler(loop, context):
        exc = context.get('exception')
        if isinstance(exc, ConnectionResetError) and getattr(exc, 'winerror', None) == 10054:
            # Benign client disconnect during video streaming, ignore silently
            return
        if default_handler:
            default_handler(loop, context)
        else:
            loop.default_exception_handler(context)
            
    loop.set_exception_handler(custom_exception_handler)
    logger.info("Attached custom asyncio exception handler to suppress WinError 10054.")

    logger.info("Application initialization and storage setup complete.")

    yield

    # --- Shutdown Actions ---
    logger.info("Shutting down the application... Cleaning resources.")


# Create the production-grade FastAPI instance
app = FastAPI(
    title=settings.APP_NAME,
    description="A high-performance AI Video Search Engine API.",
    version="1.0.0",
    debug=settings.DEBUG,
    lifespan=lifespan,
)


# --- Custom Exception Handlers ---
@app.exception_handler(VideoValidationError)
async def video_validation_exception_handler(request: Request, exc: VideoValidationError):
    """Intercept validation errors and return standard HTTP 400 Bad Request."""
    logger.warning(f"Video validation exception triggered: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )


@app.exception_handler(VideoNotFoundError)
async def video_not_found_exception_handler(request: Request, exc: VideoNotFoundError):
    """Intercept lookup errors and return standard HTTP 404 Not Found."""
    logger.warning(f"Video search lookup exception triggered: {str(exc)}")
    return JSONResponse(
        status_code=404,
        content={"detail": str(exc)},
    )


@app.exception_handler(FrameExtractionError)
async def frame_extraction_exception_handler(request: Request, exc: FrameExtractionError):
    """Intercept frame extraction failures and return standard HTTP 500 Internal Error."""
    logger.error(f"Frame extraction failure triggered: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


# --- API Routers Wiring ---
app.include_router(videos_router)
app.include_router(frames_router)
app.include_router(summary_router)
app.include_router(search_router)
app.include_router(events_router)


# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production restrictions
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """HTTP middleware to log details of incoming requests and performance metrics."""
    start_time = time.perf_counter()
    method = request.method
    path = request.url.path

    is_noisy = (
        path == "/" or 
        (path.startswith("/videos/") and path.endswith("/status")) or 
        path == "/videos/"
    )

    if not is_noisy:
        logger.debug(f"Received request: {method} {path}")

    try:
        response = await call_next(request)
        process_time = time.perf_counter() - start_time
        status_code = response.status_code

        # Log differently based on response status
        if status_code >= 400:
            logger.warning(
                f"HTTP {method} {path} - Completed with status {status_code} in {process_time:.4f}s"
            )
        elif not is_noisy:
            logger.info(
                f"HTTP {method} {path} - Completed with status {status_code} in {process_time:.4f}s"
            )

        return response
    except Exception as exc:
        process_time = time.perf_counter() - start_time
        # Log error in global application context
        logger.exception(
            f"HTTP {method} {path} - Failed with exception: {str(exc)} in {process_time:.4f}s"
        )
        raise exc


@app.get("/health", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """Verify application health, configuration loaded, and file-system status.

    Returns:
        dict: High-level health and configuration metadata.
    """
    # Simple write check for paths
    storage_healthy = True
    checks = {}

    for name, path in [
        ("data", settings.DATA_DIR),
        ("videos", settings.VIDEOS_DIR),
        ("frames", settings.FRAMES_DIR),
        ("metadata", settings.METADATA_DIR),
        ("logs", settings.LOGS_DIR),
    ]:
        is_accessible = path.exists() and os.access(path, os.W_OK)
        checks[f"{name}_directory"] = "accessible" if is_accessible else "inaccessible"
        if not is_accessible:
            storage_healthy = False

    # Log access verification into metadata.log
    meta_logger = logger.bind(context="metadata")
    meta_logger.debug(f"Healthcheck triggered. Storage checks: {checks}")

    return {
        "status": "healthy" if storage_healthy else "degraded",
        "app_name": settings.APP_NAME,
        "environment": settings.ENV,
        "debug_mode": settings.DEBUG,
        "storage_checks": checks,
        "timestamp": time.time(),
    }

# --- Mount Web Dashboard (Frontend) ---
frontend_path = settings.DATA_DIR.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
else:
    logger.warning("Frontend directory not found! Web Dashboard will be disabled.")
