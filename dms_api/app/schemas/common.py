"""
Common Schemas and Enumerations

Shared types used across multiple endpoints.
"""

from enum import IntEnum, Enum
from pydantic import Field
from .base import BaseSchema


class ResponseCode(IntEnum):
    """DMS API response codes."""

    SUCCESS = 0
    SYSTEM_ERROR = 1
    BUSINESS_ERROR = 2
    PARAMETER_ERROR = 3
    DATABASE_ERROR = 4
    AUTH_EXPIRED = 401

    @property
    def is_success(self) -> bool:
        return self == ResponseCode.SUCCESS

    @property
    def is_retriable(self) -> bool:
        return self in (ResponseCode.SYSTEM_ERROR, ResponseCode.DATABASE_ERROR)

    @property
    def description(self) -> str:
        descriptions = {
            0: "Operation successful",
            1: "System error occurred",
            2: "Business logic error",
            3: "Invalid parameters",
            4: "Database operation failed",
            401: "Authentication expired",
        }
        return descriptions.get(self.value, "Unknown error")


class AbnormalType(IntEnum):
    """Types of carriage abnormalities."""

    UNLOCKED = 1        # 未落锁 - Lock not secured
    FOREIGN_OBJECT = 2  # 有异物 - Foreign object detected

    @property
    def label(self) -> str:
        labels = {1: "未落锁", 2: "有异物"}
        return labels.get(self.value, "未知")

    @property
    def description(self) -> str:
        descriptions = {
            1: "Lock not properly secured",
            2: "Foreign object detected in carriage",
        }
        return descriptions.get(self.value, "Unknown abnormality")


class AbnormalStatus(IntEnum):
    """Abnormality detection status."""

    NORMAL = 0    # 无异常
    ABNORMAL = 1  # 有异常

    @property
    def label(self) -> str:
        return "有异常" if self == AbnormalStatus.ABNORMAL else "无异常"


class HealthStatus(str, Enum):
    """Service health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class FileUpload(BaseSchema):
    """File upload metadata."""

    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME type")
    size: int = Field(..., ge=0, description="File size in bytes")

    model_config = {"extra": "allow"}


class DMSUpstreamResponse(BaseSchema):
    """Raw response from DMS upstream API."""

    code: ResponseCode = Field(..., description="Response code")
    msg: str = Field(default="", description="Response message")
    data: dict | list | None = Field(default=None, description="Response data")

    @property
    def is_success(self) -> bool:
        return self.code == ResponseCode.SUCCESS


class BatchItemResult(BaseSchema):
    """Result for a single item in batch operations."""

    index: int = Field(..., description="Item index in batch")
    success: bool = Field(..., description="Whether this item succeeded")
    error: str | None = Field(default=None, description="Error message if failed")
    data: dict | None = Field(default=None, description="Result data if succeeded")


class BatchResponse(BaseSchema):
    """Response for batch operations."""

    total: int = Field(..., description="Total items processed")
    succeeded: int = Field(..., description="Number of successful items")
    failed: int = Field(..., description="Number of failed items")
    results: list[BatchItemResult] = Field(default_factory=list)

    @property
    def all_succeeded(self) -> bool:
        return self.failed == 0

    @property
    def success_rate(self) -> float:
        return self.succeeded / self.total if self.total > 0 else 0.0
