"""
DMS API Gateway - FastAPI Application

Production-grade API gateway for the DMS (Data Management System) backend.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
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
- **Train ID Recognition**: Identify vehicle type and number from station-entry camera images

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
        docs_url=None,  # custom /docs below with multi-file upload support
        redoc_url="/redoc" if settings.debug or settings.is_development else None,
        openapi_url="/openapi.json" if settings.debug or settings.is_development else None,
        lifespan=lifespan,
    )

    # ===== Middleware (order matters - last added is executed first) =====

    # UTF-8 charset for JSON responses (fixes CJK rendering in browsers)
    @app.middleware("http")
    async def add_charset_to_json(request: Request, call_next):
        response = await call_next(request)
        ct = response.headers.get("content-type", "")
        if ct == "application/json":
            response.headers["content-type"] = "application/json; charset=utf-8"
        return response

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

    # Custom Swagger UI with multi-file upload support
    if settings.debug or settings.is_development:
        from fastapi.responses import HTMLResponse

        @app.get("/docs", include_in_schema=False)
        async def custom_swagger_ui():
            return HTMLResponse("""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>""" + settings.app_name + """ - Swagger UI</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
</head><body>
<div id="swagger-ui"></div>
<script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>
// --- Multi-file upload patch for Swagger UI ---
// Swagger UI only reads the first file from <input multiple>.
// This script stores all selected files, then intercepts fetch()
// to rebuild FormData with every file before the request is sent.
var _pendingMultiFiles = null;

var _origFetch = window.fetch;
window.fetch = function(url, options) {
    if (_pendingMultiFiles && options && options.body instanceof FormData) {
        var newFD = new FormData();
        for (var pair of options.body.entries()) {
            if (pair[1] instanceof File) {
                // Replace the single file with all selected files
                _pendingMultiFiles.forEach(function(f) {
                    newFD.append(pair[0], f, f.name);
                });
            } else {
                newFD.append(pair[0], pair[1]);
            }
        }
        options.body = newFD;
        _pendingMultiFiles = null;
    }
    return _origFetch.call(this, url, options);
};

const ui = SwaggerUIBundle({
    url: '/openapi.json',
    dom_id: '#swagger-ui',
    layout: 'BaseLayout',
    deepLinking: true,
    showExtensions: true,
    showCommonExtensions: true,
    presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset],
    onComplete: function() {
        var root = document.getElementById('swagger-ui');
        new MutationObserver(function() {
            root.querySelectorAll('input[type="file"]').forEach(function(input) {
                if (!input.hasAttribute('multiple')) {
                    input.setAttribute('multiple', 'true');
                }
                if (!input.dataset.mfPatched) {
                    input.dataset.mfPatched = '1';
                    input.addEventListener('change', function() {
                        if (this.files.length > 1) {
                            _pendingMultiFiles = Array.from(this.files);
                        } else {
                            _pendingMultiFiles = null;
                        }
                    });
                }
            });
        }).observe(root, {childList: true, subtree: true});
    }
});
</script>
</body></html>""")

    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
        }

    # ===== OpenAPI schema patch =====
    # FastAPI 0.100+ generates OpenAPI 3.1.0 with contentMediaType for file
    # uploads, but Swagger UI only understands format=binary (OpenAPI 3.0).
    # Patch the schema so upload buttons render correctly.

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        for path_item in schema.get("paths", {}).values():
            for operation in path_item.values():
                body = operation.get("requestBody", {})
                form = body.get("content", {}).get("multipart/form-data", {})
                props = form.get("schema", {}).get("properties", {})
                if not props:
                    ref = form.get("schema", {}).get("$ref")
                    if ref:
                        comp_name = ref.rsplit("/", 1)[-1]
                        props = (
                            schema.get("components", {})
                            .get("schemas", {})
                            .get(comp_name, {})
                            .get("properties", {})
                        )
                for prop in props.values():
                    if prop.get("contentMediaType") == "application/octet-stream":
                        prop.pop("contentMediaType", None)
                        prop["type"] = "string"
                        prop["format"] = "binary"
                    items = prop.get("items", {})
                    if items.get("contentMediaType") == "application/octet-stream":
                        items.pop("contentMediaType", None)
                        items["type"] = "string"
                        items["format"] = "binary"
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi

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
