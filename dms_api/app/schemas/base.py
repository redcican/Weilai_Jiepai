"""
Base Pydantic v2 Schemas

Provides base classes and common patterns for all schemas.
"""

from datetime import datetime, timezone
from typing import TypeVar, Generic, Any
from pydantic import BaseModel, ConfigDict, Field


class BaseSchema(BaseModel):
    """Base schema with common configuration."""

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
        # Serialize using alias names (camelCase) for REST API convention
        serialize_by_alias=True,
    )


class TimestampMixin(BaseModel):
    """Mixin for timestamp fields."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None


T = TypeVar("T")


class ResponseSchema(BaseSchema, Generic[T]):
    """
    Standard API response wrapper.

    All API responses follow this structure for consistency.
    """

    success: bool = Field(default=True, description="Whether the request succeeded")
    message: str = Field(default="Success", description="Response message")
    data: T | None = Field(default=None, description="Response payload")
    request_id: str | None = Field(default=None, description="Request tracking ID")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def ok(
        cls,
        data: T | None = None,
        message: str = "Success",
        request_id: str | None = None,
    ) -> "ResponseSchema[T]":
        """Create a success response."""
        return cls(
            success=True,
            message=message,
            data=data,
            request_id=request_id,
        )

    @classmethod
    def error(
        cls,
        message: str,
        request_id: str | None = None,
    ) -> "ResponseSchema[None]":
        """Create an error response."""
        return ResponseSchema[None](
            success=False,
            message=message,
            data=None,
            request_id=request_id,
        )


class ErrorDetail(BaseSchema):
    """Error detail for error responses."""

    error_code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional details")
    field: str | None = Field(default=None, description="Field that caused the error")


class ErrorResponse(BaseSchema):
    """Standard error response."""

    success: bool = Field(default=False)
    error: ErrorDetail
    request_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PaginationParams(BaseSchema):
    """Pagination parameters."""

    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size


class PaginatedResponse(BaseSchema, Generic[T]):
    """Paginated response wrapper."""

    success: bool = Field(default=True)
    message: str = Field(default="Success")
    data: list[T] = Field(default_factory=list)
    pagination: dict[str, int] = Field(default_factory=dict)
    request_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def create(
        cls,
        items: list[T],
        total: int,
        page: int,
        page_size: int,
        request_id: str | None = None,
    ) -> "PaginatedResponse[T]":
        """Create a paginated response."""
        total_pages = (total + page_size - 1) // page_size
        return cls(
            data=items,
            pagination={
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1,
            },
            request_id=request_id,
        )
