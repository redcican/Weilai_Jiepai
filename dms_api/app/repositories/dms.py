"""
DMS Repository

Handles all communication with the upstream DMS API.
Implements retry logic, circuit breaker, and error handling.
"""

import asyncio
import time
import httpx
from functools import lru_cache
from typing import Any
from contextlib import asynccontextmanager

from ..config import Settings, get_settings
from ..core.exceptions import (
    DMSConnectionError,
    DMSTimeoutError,
    DMSUpstreamError,
    DMSCircuitBreakerOpenError,
)
from ..core.logging import get_logger
from ..schemas.common import ResponseCode, DMSUpstreamResponse
from .base import BaseRepository

logger = get_logger(__name__)


class CircuitBreaker:
    """Simple async circuit breaker implementation."""

    def __init__(self, threshold: int = 5, reset_timeout: float = 30.0):
        self.threshold = threshold
        self.reset_timeout = reset_timeout
        self._failures = 0
        self._last_failure_time: float | None = None
        self._state = "closed"  # closed, open, half-open
        self._lock = asyncio.Lock()

    @property
    def is_open(self) -> bool:
        if self._state == "open":
            if self._last_failure_time:
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self.reset_timeout:
                    self._state = "half-open"
                    return False
            return True
        return False

    async def record_success(self) -> None:
        async with self._lock:
            self._failures = 0
            self._state = "closed"

    async def record_failure(self) -> None:
        async with self._lock:
            self._failures += 1
            self._last_failure_time = time.monotonic()
            if self._failures >= self.threshold:
                self._state = "open"
                logger.warning(f"Circuit breaker opened after {self._failures} failures")

    async def check(self) -> None:
        if self.is_open:
            raise DMSCircuitBreakerOpenError(reset_after=self.reset_timeout)


class DMSRepository(BaseRepository):
    """
    Repository for DMS API operations.

    Features:
    - Async HTTP client with connection pooling
    - Automatic retry with exponential backoff
    - Circuit breaker for fault tolerance
    - Request/response logging
    """

    # API Endpoints
    ENDPOINT_ABNORMAL_SAVE = "/api/public/abnormalSave"
    ENDPOINT_BAD_SAVE = "/api/public/badSave"
    ENDPOINT_SIGNAL_LIGHT = "/api/public/signLightChange"
    ENDPOINT_CONTAINER = "/api/public/contrainer"
    ENDPOINT_TICKET_PARSE = "/api/public/ticketParse"

    def __init__(self, settings: Settings):
        self._settings = settings
        self._base_url = str(settings.dms_base_url).rstrip("/")
        self._client: httpx.AsyncClient | None = None
        self._circuit_breaker = CircuitBreaker(
            threshold=settings.circuit_breaker_threshold,
            reset_timeout=settings.circuit_breaker_timeout,
        ) if settings.circuit_breaker_enabled else None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(
                    connect=self._settings.dms_timeout_connect,
                    read=self._settings.dms_timeout_read,
                    write=self._settings.dms_timeout_read,
                    pool=self._settings.dms_timeout_connect,
                ),
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20,
                ),
            )
        return self._client

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: dict[str, tuple] | None = None,
    ) -> DMSUpstreamResponse:
        """
        Make HTTP request to DMS API with retry and circuit breaker.

        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            data: Form data
            files: Files to upload

        Returns:
            Parsed response from DMS API
        """
        # Check circuit breaker
        if self._circuit_breaker:
            await self._circuit_breaker.check()

        client = await self._get_client()
        last_error: Exception | None = None

        for attempt in range(self._settings.dms_max_retries + 1):
            try:
                logger.debug(
                    f"DMS request attempt {attempt + 1}: {method} {endpoint}",
                    extra={"extra_fields": {"params": params}},
                )

                response = await client.request(
                    method=method,
                    url=endpoint,
                    params=params,
                    data=data,
                    files=files,
                )

                json_data = response.json()
                result = DMSUpstreamResponse.model_validate(json_data)

                logger.debug(
                    f"DMS response: code={result.code}, msg={result.msg}",
                )

                # Record success for circuit breaker
                if self._circuit_breaker:
                    await self._circuit_breaker.record_success()

                # Check for retriable error codes
                if result.code.is_retriable and attempt < self._settings.dms_max_retries:
                    delay = self._settings.dms_retry_backoff * (2 ** attempt)
                    logger.warning(f"Retriable error, retrying in {delay}s: {result.msg}")
                    await asyncio.sleep(delay)
                    continue

                return result

            except httpx.ConnectError as e:
                last_error = DMSConnectionError(
                    message=f"Failed to connect to DMS API: {e}",
                    details={"service": "DMS", "endpoint": endpoint},
                )
            except httpx.TimeoutException as e:
                last_error = DMSTimeoutError(
                    message=f"DMS API request timed out: {e}",
                    timeout_seconds=self._settings.dms_timeout_read,
                )
            except httpx.HTTPError as e:
                last_error = DMSConnectionError(
                    message=f"HTTP error: {e}",
                    details={"service": "DMS", "error": str(e)},
                )
            except Exception as e:
                last_error = DMSUpstreamError(
                    message=f"Unexpected error: {e}",
                    details={"error": str(e)},
                )

            # Record failure for circuit breaker
            if self._circuit_breaker:
                await self._circuit_breaker.record_failure()

            # Retry logic
            if attempt < self._settings.dms_max_retries:
                delay = self._settings.dms_retry_backoff * (2 ** attempt)
                logger.warning(
                    f"Request failed, retrying in {delay}s (attempt {attempt + 1}): {last_error}"
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        raise last_error or DMSConnectionError(message="Request failed after all retries")

    # ===== Abnormal Operations =====

    async def save_abnormal(
        self,
        carbin_no: str,
        descr: str,
        file_content: bytes,
        filename: str,
        content_type: str,
    ) -> DMSUpstreamResponse:
        """Save abnormal alert."""
        return await self._request(
            method="POST",
            endpoint=self.ENDPOINT_ABNORMAL_SAVE,
            params={"carbinNo": carbin_no, "descr": descr},
            files={"file": (filename, file_content, content_type)},
        )

    async def save_bad_condition(
        self,
        carbin_no: str,
        abnormal_type: int,
        is_abnormal: int,
        descr: str | None = None,
        file_content: bytes | None = None,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> DMSUpstreamResponse:
        """Save bad condition report."""
        params = {
            "carbinNo": carbin_no,
            "abnormalType": abnormal_type,
            "isAbnormal": is_abnormal,
        }
        if descr:
            params["descr"] = descr

        files = None
        if file_content and filename and content_type:
            files = {"file": (filename, file_content, content_type)}

        return await self._request(
            method="POST",
            endpoint=self.ENDPOINT_BAD_SAVE,
            params=params,
            files=files,
        )

    # ===== Signal Light Operations =====

    async def save_signal_change(
        self,
        file_content: bytes,
        filename: str,
        content_type: str,
    ) -> DMSUpstreamResponse:
        """Save signal light change."""
        return await self._request(
            method="POST",
            endpoint=self.ENDPOINT_SIGNAL_LIGHT,
            files={"file": (filename, file_content, content_type)},
        )

    # ===== Container Operations =====

    async def save_container(
        self,
        train_no: str,
        carbin_no: str,
        container_no: str,
        file_content: bytes,
        filename: str,
        content_type: str,
    ) -> DMSUpstreamResponse:
        """Save container identification."""
        return await self._request(
            method="POST",
            endpoint=self.ENDPOINT_CONTAINER,
            params={
                "trainNo": train_no,
                "carbinNo": carbin_no,
                "contrainerNo": container_no,
            },
            files={"file": (filename, file_content, content_type)},
        )

    # ===== Ticket Operations =====

    async def parse_ticket(
        self,
        file_content: bytes,
        filename: str,
        content_type: str,
    ) -> DMSUpstreamResponse:
        """Parse ticket using OCR."""
        return await self._request(
            method="POST",
            endpoint=self.ENDPOINT_TICKET_PARSE,
            files={"file": (filename, file_content, content_type)},
        )

    # ===== Health Check =====

    async def health_check(self) -> bool:
        """Check if DMS API is reachable."""
        try:
            client = await self._get_client()
            response = await client.get("/", timeout=5.0)
            return response.status_code < 500
        except Exception as e:
            logger.warning(f"DMS health check failed: {e}")
            return False

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


# Dependency injection
_repository: DMSRepository | None = None


async def get_dms_repository(
    settings: Settings = None,
) -> DMSRepository:
    """Get DMS repository instance (singleton)."""
    global _repository
    if _repository is None:
        _repository = DMSRepository(settings or get_settings())
    return _repository


async def close_dms_repository() -> None:
    """Close DMS repository."""
    global _repository
    if _repository:
        await _repository.close()
        _repository = None
