"""
Security Utilities

API key authentication and other security features.
"""

from typing import Annotated
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, APIKeyQuery

from ..config import Settings, get_settings
from .exceptions import DMSAuthenticationError

# API Key can be provided via header or query parameter
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)


async def get_api_key(
    api_key_header_value: str | None = Security(api_key_header),
    api_key_query_value: str | None = Security(api_key_query),
    settings: Settings = Depends(get_settings),
) -> str | None:
    """
    Extract and validate API key from request.

    API key can be provided via:
    - Header: X-API-Key
    - Query parameter: api_key

    Returns None if API key auth is disabled.
    """
    if not settings.api_key_enabled:
        return None

    api_key = api_key_header_value or api_key_query_value

    if not api_key:
        raise DMSAuthenticationError(
            message="API key required. Provide via X-API-Key header or api_key query parameter."
        )

    if api_key not in settings.api_keys:
        raise DMSAuthenticationError(message="Invalid API key")

    return api_key


# Type alias for dependency injection
APIKey = Annotated[str | None, Depends(get_api_key)]


class APIKeyValidator:
    """
    Configurable API key validator.

    Can be used with different key sets for different endpoints.
    """

    def __init__(self, valid_keys: list[str] | None = None, required: bool = True):
        self.valid_keys = valid_keys
        self.required = required

    async def __call__(
        self,
        api_key_header_value: str | None = Security(api_key_header),
        api_key_query_value: str | None = Security(api_key_query),
        settings: Settings = Depends(get_settings),
    ) -> str | None:
        # Use provided keys or fall back to settings
        valid_keys = self.valid_keys or settings.api_keys

        if not settings.api_key_enabled and not self.valid_keys:
            return None

        api_key = api_key_header_value or api_key_query_value

        if not api_key:
            if self.required:
                raise DMSAuthenticationError(message="API key required")
            return None

        if api_key not in valid_keys:
            raise DMSAuthenticationError(message="Invalid API key")

        return api_key


def create_api_key_dependency(
    valid_keys: list[str] | None = None,
    required: bool = True,
) -> APIKeyValidator:
    """
    Factory function to create API key dependency.

    Args:
        valid_keys: Optional list of valid keys (uses settings if None)
        required: Whether API key is required

    Returns:
        APIKeyValidator instance
    """
    return APIKeyValidator(valid_keys=valid_keys, required=required)
