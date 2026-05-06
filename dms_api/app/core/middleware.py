"""
FastAPI Middleware

Custom middleware for request processing, logging, and error handling.
"""

import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .logging import request_id_ctx, get_logger

logger = get_logger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add request context (ID, timing, logging).

    Features:
    - Generates unique request ID for tracing
    - Measures request duration
    - Logs request/response details
    - Sets context variables for downstream use
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
        request_id_ctx.set(request_id)

        # Store in request state
        request.state.request_id = request_id
        request.state.start_time = time.perf_counter()

        # Log request
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "extra_fields": {
                    "method": request.method,
                    "path": request.url.path,
                    "query": str(request.query_params),
                    "client_ip": self._get_client_ip(request),
                }
            },
        )

        try:
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.perf_counter() - request.state.start_time) * 1000

            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

            # Log response
            logger.info(
                f"Request completed: {request.method} {request.url.path} -> {response.status_code}",
                extra={
                    "extra_fields": {
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": response.status_code,
                        "duration_ms": round(duration_ms, 2),
                    }
                },
            )

            return response

        except Exception as e:
            duration_ms = (time.perf_counter() - request.state.start_time) * 1000
            logger.error(
                f"Request failed: {request.method} {request.url.path} - {e}",
                extra={
                    "extra_fields": {
                        "method": request.method,
                        "path": request.url.path,
                        "error": str(e),
                        "duration_ms": round(duration_ms, 2),
                    }
                },
                exc_info=True,
            )
            raise

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """Extract client IP from request headers."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiting middleware.

    For production, use Redis-based rate limiting.
    """

    def __init__(
        self,
        app: ASGIApp,
        requests_per_window: int = 100,
        window_seconds: int = 60,
    ):
        super().__init__(app)
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = self._get_client_ip(request)
        current_time = time.time()

        # Get or create request list for this client
        if client_ip not in self._requests:
            self._requests[client_ip] = []

        # Filter requests within window
        window_start = current_time - self.window_seconds
        self._requests[client_ip] = [
            t for t in self._requests[client_ip] if t > window_start
        ]

        # Check rate limit
        if len(self._requests[client_ip]) >= self.requests_per_window:
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "error": {
                        "error_code": "RATE_LIMIT_EXCEEDED",
                        "message": "Too many requests",
                        "details": {
                            "retry_after_seconds": self.window_seconds,
                        },
                    },
                },
                headers={"Retry-After": str(self.window_seconds)},
            )

        # Record this request
        self._requests[client_ip].append(current_time)

        return await call_next(request)

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        return response
