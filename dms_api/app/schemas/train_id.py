"""
Train ID Recognition Schemas

Schemas for station-entry vehicle identification operations.
"""

from pydantic import Field
from .base import BaseSchema, ResponseSchema


class TrainIDData(BaseSchema):
    """
    Recognized train identification data.

    Contains vehicle type and number extracted from station-entry camera images.
    """

    vehicle_type: str = Field(
        default="",
        alias="vehicleType",
        description="Vehicle type code (车种), e.g. C64K, C70E",
    )
    vehicle_number: str = Field(
        default="",
        alias="vehicleNumber",
        description="Vehicle number (车号), e.g. 49 31846",
    )
    confidence: float = Field(
        default=0.0,
        description="Average OCR confidence score (0.0 - 1.0)",
    )


class TrainIDResponse(ResponseSchema[TrainIDData]):
    """Response for single image train ID recognition."""

    pass


class TrainIDBatchItem(BaseSchema):
    """Single item in a batch recognition response."""

    filename: str = Field(..., description="Original filename")
    vehicle_type: str = Field(
        default="",
        alias="vehicleType",
        description="Vehicle type code (车种)",
    )
    vehicle_number: str = Field(
        default="",
        alias="vehicleNumber",
        description="Vehicle number (车号)",
    )
    confidence: float = Field(
        default=0.0,
        description="Average OCR confidence score",
    )


class TrainIDBatchResponse(ResponseSchema[list[TrainIDBatchItem]]):
    """Response for batch train ID recognition."""

    pass


# ---------------------------------------------------------------------------
# PaddleOCR single-image recognition schemas
# ---------------------------------------------------------------------------

class PaddleImageData(BaseSchema):
    """Recognized data from single-image PaddleOCR processing."""

    containers: list[str] = Field(
        default_factory=list,
        description="Container IDs (集装箱箱号)",
    )
    train_types: list[str] = Field(
        default_factory=list,
        alias="trainTypes",
        description="Train vehicle types (车种), e.g. C70E, C64K",
    )
    train_numbers: list[str] = Field(
        default_factory=list,
        alias="trainNumbers",
        description="Train vehicle numbers (车号)",
    )


class PaddleImageResponse(ResponseSchema[PaddleImageData]):
    """Response for PaddleOCR single-image recognition."""

    pass


# ---------------------------------------------------------------------------
# Video recognition schemas (deprecated, kept for reference)
# ---------------------------------------------------------------------------
# class VideoTrainIDData(BaseSchema): ...
# class VideoTrainIDResponse(ResponseSchema[VideoTrainIDData]): ...
# class FlatcarVideoData(BaseSchema): ...
# class FlatcarVideoResponse(ResponseSchema[FlatcarVideoData]): ...
