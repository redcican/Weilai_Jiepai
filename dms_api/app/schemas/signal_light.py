"""
Signal Light Detection Schemas

Schemas for signal light color recognition operations.
"""

from pydantic import Field
from .base import BaseSchema, ResponseSchema


class SignalLightItem(BaseSchema):
    """Single image detection result."""

    filename: str = Field(..., description="Original filename")
    color: str = Field(..., description="Detected color (红色/白色/蓝色/未知)")


class SignalLightBatchResponse(ResponseSchema[list[SignalLightItem]]):
    """Response for batch signal light detection."""

    pass
