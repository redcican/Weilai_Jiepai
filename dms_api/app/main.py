"""
DMS API Gateway - FastAPI Application

Production-grade API gateway for the DMS (Data Management System) backend.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from .config import get_settings
from .core.logging import setup_logging, get_logger
from .core.middleware import (
    RequestContextMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
)
from .core.exceptions import DMSException
from .api.v1 import router as v1_router
from .api.health import router as health_router
from .repositories.dms import close_dms_repository

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    settings = get_settings()
    setup_logging(
        level=settings.log_level,
        format_type=settings.log_format,
        log_file=settings.log_file,
    )
    logger.info(
        f"Starting {settings.app_name} v{settings.app_version} "
        f"in {settings.environment} mode"
    )

    yield

    # Shutdown
    logger.info("Shutting down application...")
    await close_dms_repository()
    logger.info("Application shutdown complete")


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="""
# DMS API Gateway

Production-grade API gateway for the railway DMS (Data Management System).

## Features

- **Abnormal Alerts**: Report operational abnormalities with photo evidence
- **Container Identification**: Submit OCR-identified container information
- **Signal Light Tracking**: Report signal light state changes
- **Ticket OCR**: Parse grouping orders using optical character recognition

## Authentication

API key authentication can be enabled via configuration. When enabled,
provide your API key via:
- Header: `X-API-Key`
- Query parameter: `api_key`

## Rate Limiting

When enabled, requests are limited per client IP address.
Check `Retry-After` header when receiving 429 responses.
        """,
        version=settings.app_version,
        docs_url="/docs" if settings.debug or settings.is_development else None,
        redoc_url="/redoc" if settings.debug or settings.is_development else None,
        openapi_url="/openapi.json" if settings.debug or settings.is_development else None,
        lifespan=lifespan,
    )

    # ===== Middleware (order matters - last added is executed first) =====

    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)

    # Rate limiting
    if settings.rate_limit_enabled:
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_window=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window,
        )

    # Request context (logging, request ID)
    app.add_middleware(RequestContextMiddleware)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time"],
    )

    # ===== Exception Handlers =====

    @app.exception_handler(DMSException)
    async def dms_exception_handler(request: Request, exc: DMSException):
        """Handle DMS-specific exceptions."""
        logger.error(f"DMS exception: {exc}", extra={"extra_fields": exc.to_dict()})
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "error_code": exc.error_code,
                    "message": exc.message,
                    "details": exc.details,
                },
                "request_id": getattr(request.state, "request_id", None),
            },
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors."""
        errors = []
        for error in exc.errors():
            errors.append({
                "field": ".".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
            })

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "success": False,
                "error": {
                    "error_code": "VALIDATION_ERROR",
                    "message": "Request validation failed",
                    "details": {"errors": errors},
                },
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "error_code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred",
                    "details": {} if settings.is_production else {"error": str(exc)},
                },
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    # ===== Routes =====

    # Health check routes (no prefix)
    app.include_router(health_router)

    # API v1 routes
    app.include_router(v1_router)

    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs" if app.docs_url else None,
        }

    return app


# Create application instance
app = create_application()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=settings.workers if not settings.debug else 1,
        log_level=settings.log_level.lower(),
    )
