"""
Custom Exceptions for DMS API

Provides a comprehensive exception hierarchy with HTTP status code mapping.
"""

from typing import Any
from fastapi import HTTPException, status


class DMSException(Exception):
    """Base exception for all DMS errors."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code: str = "DMS_ERROR"
    message: str = "An unexpected error occurred"

    def __init__(
        self,
        message: str | None = None,
        details: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ):
        self.message = message or self.__class__.message
        self.details = details or {}
        self.headers = headers
        super().__init__(self.message)

    def to_http_exception(self) -> HTTPException:
        """Convert to FastAPI HTTPException."""
        return HTTPException(
            status_code=self.status_code,
            detail={
                "error_code": self.error_code,
                "message": self.message,
                "details": self.details,
            },
            headers=self.headers,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }


class DMSValidationError(DMSException):
    """Validation error for request data."""

    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    error_code = "VALIDATION_ERROR"
    message = "Validation failed"

    def __init__(
        self,
        message: str | None = None,
        field: str | None = None,
        value: Any = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        super().__init__(message, details=details, **kwargs)


class DMSNotFoundError(DMSException):
    """Resource not found error."""

    status_code = status.HTTP_404_NOT_FOUND
    error_code = "NOT_FOUND"
    message = "Resource not found"

    def __init__(self, resource: str | None = None, resource_id: Any = None, **kwargs):
        details = kwargs.pop("details", {})
        if resource:
            details["resource"] = resource
        if resource_id is not None:
            details["resource_id"] = str(resource_id)
        message = f"{resource} not found" if resource else None
        super().__init__(message, details=details, **kwargs)


class DMSConnectionError(DMSException):
    """Connection error to upstream service."""

    status_code = status.HTTP_502_BAD_GATEWAY
    error_code = "CONNECTION_ERROR"
    message = "Failed to connect to upstream service"

    def __init__(self, service: str | None = None, **kwargs):
        details = kwargs.pop("details", {})
        if service:
            details["service"] = service
        super().__init__(details=details, **kwargs)


class DMSTimeoutError(DMSException):
    """Timeout error for upstream service."""

    status_code = status.HTTP_504_GATEWAY_TIMEOUT
    error_code = "TIMEOUT_ERROR"
    message = "Request to upstream service timed out"

    def __init__(self, timeout_seconds: float | None = None, **kwargs):
        details = kwargs.pop("details", {})
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds
        super().__init__(details=details, **kwargs)


class DMSCircuitBreakerOpenError(DMSException):
    """Circuit breaker is open."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_code = "CIRCUIT_BREAKER_OPEN"
    message = "Service temporarily unavailable"

    def __init__(self, reset_after: float | None = None, **kwargs):
        details = kwargs.pop("details", {})
        if reset_after:
            details["reset_after_seconds"] = reset_after
        headers = {"Retry-After": str(int(reset_after))} if reset_after else None
        super().__init__(details=details, headers=headers, **kwargs)


class DMSAuthenticationError(DMSException):
    """Authentication error."""

    status_code = status.HTTP_401_UNAUTHORIZED
    error_code = "AUTHENTICATION_ERROR"
    message = "Authentication required"

    def __init__(self, **kwargs):
        headers = {"WWW-Authenticate": "Bearer"}
        super().__init__(headers=headers, **kwargs)


class DMSAuthorizationError(DMSException):
    """Authorization error."""

    status_code = status.HTTP_403_FORBIDDEN
    error_code = "AUTHORIZATION_ERROR"
    message = "Access denied"


class DMSRateLimitError(DMSException):
    """Rate limit exceeded."""

    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    error_code = "RATE_LIMIT_EXCEEDED"
    message = "Too many requests"

    def __init__(self, retry_after: int | None = None, **kwargs):
        details = kwargs.pop("details", {})
        if retry_after:
            details["retry_after_seconds"] = retry_after
        headers = {"Retry-After": str(retry_after)} if retry_after else None
        super().__init__(details=details, headers=headers, **kwargs)


class DMSUpstreamError(DMSException):
    """Error from upstream DMS service."""

    status_code = status.HTTP_502_BAD_GATEWAY
    error_code = "UPSTREAM_ERROR"
    message = "Upstream service error"

    def __init__(
        self,
        upstream_code: int | None = None,
        upstream_message: str | None = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if upstream_code is not None:
            details["upstream_code"] = upstream_code
        if upstream_message:
            details["upstream_message"] = upstream_message
        super().__init__(details=details, **kwargs)


class DMSFileError(DMSException):
    """File handling error."""

    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "FILE_ERROR"
    message = "File processing error"

    def __init__(
        self,
        filename: str | None = None,
        reason: str | None = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if filename:
            details["filename"] = filename
        if reason:
            details["reason"] = reason
        super().__init__(details=details, **kwargs)
