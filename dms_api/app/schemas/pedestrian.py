"""
Pedestrian Detection Schemas

Schemas for pedestrian anomaly detection operations.
"""

from pydantic import Field
from .base import BaseSchema, ResponseSchema


class PedestrianDetection(BaseSchema):
    """Single bounding box detection."""

    class_name: str = Field(
        ..., alias="className", description="Detected class name (e.g. person)"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence score"
    )
    bbox: list[float] = Field(
        ..., description="Bounding box [x1, y1, x2, y2] in pixels"
    )


class PedestrianItem(BaseSchema):
    """Single image result in batch response."""

    filename: str = Field(..., description="Original filename")
    status: str = Field(
        ..., description="Detection result: 正常 / 异常"
    )
    pedestrian_count: int = Field(
        default=0, alias="pedestrianCount", description="Number of pedestrians detected"
    )
    detections: list[PedestrianDetection] = Field(
        default_factory=list, description="Detection details"
    )


class PedestrianBatchResponse(ResponseSchema[list[PedestrianItem]]):
    """Response for batch pedestrian detection."""

    pass
