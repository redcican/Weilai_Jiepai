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
    confidence: float = Field(0.0, description="Detection confidence (0.0-1.0)")
    scores: dict = Field(default_factory=dict, description="Per-color blob scores")


class SignalLightBatchResponse(ResponseSchema[list[SignalLightItem]]):
    """Response for batch signal light detection."""

    pass
